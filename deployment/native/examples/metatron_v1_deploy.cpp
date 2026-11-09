// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// One native 39-input launcher. Only RobotBackend differs across simulation
// and hardware; task goals, FK, history, TensorRT and safety are shared.
#include <mujoco/mujoco.h>
#include <nlohmann/json.hpp>
#include <net/if.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "protomotions/deployment/controller_manager.hpp"
#include "protomotions/deployment/evdev_xbox_input.hpp"
#include "protomotions/deployment/metatron_v1_controller.hpp"
#include "protomotions/deployment/metatron_v1_engine.hpp"
#include "protomotions/deployment/metatron_v1_fk_engine.hpp"
#include "protomotions/deployment/metatron_v1_manifest.hpp"
#include "protomotions/deployment/metatron_v1_task.hpp"
#include "metatron_v1_async_viewer.hpp"
#include "protomotions/deployment/mujoco_backend.hpp"
#include "protomotions/deployment/runtime.hpp"
#include "protomotions/deployment/unitree_dds_config.hpp"
#include "protomotions/deployment/unitree_input.hpp"
#include "protomotions/deployment/unitree_sdk2_backend.hpp"

namespace {
using namespace protomotions::deployment;
using namespace protomotions::deployment::metatron_v1;
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t g_stop = 0;
void signal_stop(int) { g_stop = 1; }

struct Options {
  std::string backend = "mujoco";
  std::string xml, engine, fk_engine, manifest, recovery_plan, xbox_device, net_if;
  std::string glfw_library;
  std::string seat_body;
  std::optional<float> seat_distance;
  bool arm = false;
  bool cuda_graph = false;
  bool viewer = false;
  bool scripted_sim = false;
  bool scripted_seat = false;
  bool unitree_remote = false;
  int steps = 0;
  int dds_participant_index = 0;
};

Options parse(int argc, char** argv) {
  Options args;
  for (int index = 1; index < argc; ++index) {
    const std::string flag(argv[index]);
    if (flag == "--arm") args.arm = true;
    else if (flag == "--cuda-graphs") args.cuda_graph = true;
    else if (flag == "--viewer") args.viewer = true;
    else if (flag == "--scripted-sim") args.scripted_sim = true;
    else if (flag == "--scripted-seat") args.scripted_seat = true;
    else if (flag == "--unitree-remote") args.unitree_remote = true;
    else {
      if (index + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
      const std::string value(argv[++index]);
      if (flag == "--backend") args.backend = value;
      else if (flag == "--xml") args.xml = value;
      else if (flag == "--engine") args.engine = value;
      else if (flag == "--fk-engine") args.fk_engine = value;
      else if (flag == "--manifest") args.manifest = value;
      else if (flag == "--recovery-plan") args.recovery_plan = value;
      else if (flag == "--xbox-device") args.xbox_device = value;
      else if (flag == "--net-if") args.net_if = value;
      else if (flag == "--glfw-lib") args.glfw_library = value;
      else if (flag == "--seat-body") args.seat_body = value;
      else if (flag == "--seat-distance") args.seat_distance = std::stof(value);
      else if (flag == "--steps") args.steps = std::stoi(value);
      else if (flag == "--dds-participant-index")
        args.dds_participant_index = std::stoi(value);
      else throw std::invalid_argument("unknown option " + flag);
    }
  }
  if ((args.backend != "mujoco" && args.backend != "unitree") ||
      (args.backend == "mujoco" && args.xml.empty()) ||
      args.engine.empty() || args.fk_engine.empty() ||
      args.manifest.empty() || args.steps < 0)
    throw std::invalid_argument("required: --backend mujoco|unitree --engine --fk-engine --manifest; MuJoCo also requires --xml");
  if (static_cast<int>(args.scripted_sim) + static_cast<int>(args.scripted_seat) + static_cast<int>(args.unitree_remote) +
          static_cast<int>(!args.xbox_device.empty()) != 1)
    throw std::invalid_argument("select one input: --unitree-remote, --xbox-device PATH, --scripted-sim or --scripted-seat");
  if (args.unitree_remote && args.backend != "unitree")
    throw std::invalid_argument("--unitree-remote requires --backend unitree");
  if (args.backend == "unitree" &&
      (!args.arm || args.net_if.empty() || if_nametoindex(args.net_if.c_str()) == 0))
    throw std::invalid_argument("hardware requires --arm and an existing --net-if");
  if ((args.scripted_sim || args.scripted_seat) && args.backend != "mujoco")
    throw std::invalid_argument("scripted input is simulation-only");
  if ((!args.seat_body.empty() && args.xml.empty()) || (args.scripted_seat && args.seat_body.empty()))
    throw std::invalid_argument("seat interaction requires --xml and --seat-body");
  if (args.seat_distance && (args.seat_body.empty() || !std::isfinite(*args.seat_distance) || *args.seat_distance < 0))
    throw std::invalid_argument("--seat-distance needs a seat and a nonnegative distance in metres");
  if (args.seat_distance && args.backend != "unitree")
    throw std::invalid_argument("--seat-distance calibrates hardware; simulation uses the physical chair pose in MJCF");
  if (args.backend == "unitree" && !args.seat_body.empty() && !args.seat_distance)
    throw std::invalid_argument("hardware seat needs measured --seat-distance (ankle midpoint to front edge, metres)");
  if (args.viewer && (args.xml.empty() || args.glfw_library.empty()))
    throw std::invalid_argument("--viewer needs --xml and --glfw-lib for either backend");
  return args;
}

float yaw(const std::array<double, 4>& q) {
  return std::atan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                    1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
}
float wrap(float value) { return std::remainder(value, 6.283185307179586F); }

std::vector<double> standing_pose() {
  std::vector<double> q(29, 0.0);
  q[0] = q[6] = -0.312;
  q[3] = q[9] = 0.669;
  q[4] = q[10] = -0.363;
  q[15] = q[22] = 0.2;
  q[16] = 0.2;
  q[23] = -0.2;
  q[18] = q[25] = 0.6;
  return q;
}

int sim_substeps_and_validate(const std::string& xml,
                              const nlohmann::json& manifest) {
  char error[1024]{};
  mjModel* model = mj_loadXML(xml.c_str(), nullptr, error, sizeof(error));
  if (!model) throw std::runtime_error(error[0] ? error : "MuJoCo XML load failed");
  try {
    const auto names = manifest.at("robot_contract").at("joint_names");
    if (model->nu != 29 || model->nq != 36)
      throw std::runtime_error("sim model does not have the scene G1's 29 actuators and free root");
    for (int actuator = 0; actuator < 29; ++actuator) {
      const int joint = model->actuator_trnid[2 * actuator];
      const char* name = mj_id2name(model, mjOBJ_JOINT, joint);
      const double kp = model->actuator_gainprm[10 * actuator];
      if (!name || names[actuator] != name || kp <= 0.0 ||
          std::abs(model->actuator_biasprm[10 * actuator + 1] + kp) > 1e-4 ||
          model->actuator_biasprm[10 * actuator + 2] >= 0.0)
        throw std::runtime_error("sim XML needs matching position actuators; run prepare_metatron_mujoco_xml.py");
    }
    if (mj_name2id(model, mjOBJ_GEOM, "floor") < 0)
      throw std::runtime_error("sim XML needs a floor geom");
    const int steps = static_cast<int>(std::round(0.02 / model->opt.timestep));
    if (steps < 1 || std::abs(steps * model->opt.timestep - 0.02) > 1e-5)
      throw std::runtime_error("sim timestep must divide 20 ms control period");
    mj_deleteModel(model);
    return steps;
  } catch (...) {
    mj_deleteModel(model);
    throw;
  }
}

SeatGeometry seat_from_xml(const std::string& xml, const std::string& body_name) {
  char error[1024]{};
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
      mj_loadXML(xml.c_str(), nullptr, error, sizeof(error)), mj_deleteModel);
  if (!model) throw std::runtime_error(error[0] ? error : "seat XML load failed");
  const int body = mj_name2id(model.get(), mjOBJ_BODY, body_name.c_str());
  if (body < 0 || model->body_parentid[body] != 0 || model->body_jntnum[body] != 0 ||
      model->body_mocapid[body] >= 0 || std::abs(model->body_quat[4*body+1]) > 1e-6 ||
      std::abs(model->body_quat[4*body+2]) > 1e-6)
    throw std::invalid_argument("seat body must be fixed in the world, with yaw only");
  SeatGeometry result;
  std::copy_n(model->body_pos+3*body, 3, result.object_position.begin());
  result.yaw = 2*std::atan2(model->body_quat[4*body+3], model->body_quat[4*body]);
  const int floor = mj_name2id(model.get(), mjOBJ_GEOM, "floor");
  if (floor < 0 || model->geom_bodyid[floor] != 0 || model->geom_type[floor] != mjGEOM_PLANE)
    throw std::invalid_argument("seat interaction needs the world floor plane");
  result.ground_z = model->geom_pos[3*floor+2];
  for (int box = 0; box < 2; ++box) {
    const std::string name = body_name + (box == 0 ? "_seat" : "_back");
    const int geom = mj_name2id(model.get(), mjOBJ_GEOM, name.c_str());
    if (geom < 0 || model->geom_bodyid[geom] != body || model->geom_type[geom] != mjGEOM_BOX ||
        std::abs(std::abs(model->geom_quat[4*geom])-1) > 1e-6)
      throw std::invalid_argument("seat needs body-aligned box geoms: " + body_name + "_seat and _back");
    for (int axis = 0; axis < 3; ++axis) {
      result.centers[3*box+axis] = model->geom_pos[3*geom+axis];
      result.extents[3*box+axis] = 2*model->geom_size[3*geom+axis];
    }
  }
  return result;
}

// The chair is centered behind the standing robot, facing the same direction.
// Capture once at startup: the physical chair is stationary while G1 moves in
// its startup odometry frame. Ground comes from actual foot collision surfaces
// rather than assuming the odometry Z origin coincides with the floor.
void calibrate_seat(SeatGeometry& seat, const RobotState& state,
                    const std::string& xml, float distance) {
  char error[1024]{};
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> m(
      mj_loadXML(xml.c_str(), nullptr, error, sizeof(error)), mj_deleteModel);
  if (!m) throw std::runtime_error(error[0] ? error : "seat calibration XML failed");
  std::unique_ptr<mjData, decltype(&mj_deleteData)> d(mj_makeData(m.get()), mj_deleteData);
  if (!d || state.dof_position.size() != 29) throw std::runtime_error("seat calibration needs measured G1 joints");
  std::copy(state.root_position.begin(), state.root_position.end(), d->qpos);
  d->qpos[3] = state.root_orientation_xyzw[3];
  for (int axis = 0; axis < 3; ++axis) d->qpos[4+axis] = state.root_orientation_xyzw[axis];
  for (int joint = 0; joint < m->nu; ++joint)
    d->qpos[m->jnt_qposadr[m->actuator_trnid[2*joint]]] = state.dof_position[joint];
  mj_forward(m.get(), d.get());
  std::array<float, 2> feet{};
  for (const char* name : {"left_ankle_roll_link", "right_ankle_roll_link"}) {
    const int body = mj_name2id(m.get(), mjOBJ_BODY, name);
    if (body < 0) throw std::runtime_error("seat calibration needs ankle roll bodies");
    for (int axis = 0; axis < 2; ++axis) feet[axis] += 0.5*d->xpos[3*body+axis];
  }
  float ground = std::numeric_limits<float>::infinity();
  for (int geom = 0; geom < m->ngeom; ++geom) {
    const char* name = mj_id2name(m.get(), mjOBJ_GEOM, geom);
    if (!name || !(std::string_view(name).starts_with("left_foot") ||
                  std::string_view(name).starts_with("right_foot")) ||
        m->geom_contype[geom] == 0 || m->geom_type[geom] != mjGEOM_CAPSULE) continue;
    const float bottom = d->geom_xpos[3*geom+2] - m->geom_size[3*geom] -
        std::abs(d->geom_xmat[9*geom+8])*m->geom_size[3*geom+1];
    ground = std::min(ground, bottom);
  }
  if (!std::isfinite(ground)) throw std::runtime_error("seat calibration needs capsule foot collision geometry");
  align_seat_behind(seat, feet, yaw(state.base_orientation_xyzw), ground, distance);
  std::cout << "Seat calibrated: object_xyz=" << seat.object_position[0] << ','
            << seat.object_position[1] << ',' << ground << " yaw=" << seat.yaw
            << " ankle_to_front_m=" << distance << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options args = parse(argc, argv);
    const auto json = nlohmann::json::parse(std::ifstream(args.manifest));
    const auto manifest = Manifest::load(args.manifest);
    const auto body_names = json.at("robot_contract").at("body_names").get<std::vector<std::string>>();
    FkEngine fk(args.fk_engine, args.cuda_graph);
    Engine engine(args.engine, args.cuda_graph);
    if (args.cuda_graph && (!fk.cuda_graph_enabled() || !engine.cuda_graph_enabled()))
      throw std::runtime_error("requested CUDA graphs did not capture for both engines");
    RestPose rest = make_rest_pose(fk);
    MetatronTask task(fk, rest, args.recovery_plan, body_names);
    std::optional<SeatGeometry> seat;
    if (!args.seat_body.empty()) {
      sim_substeps_and_validate(args.xml, json);
      seat = seat_from_xml(args.xml, args.seat_body);
      task.configure_seat(*seat);
    }
    engine.warmup(10);
    std::unique_ptr<EvdevXboxInputSource> xbox;
    UnitreeControllerInputSource remote;
    if (!args.xbox_device.empty())
      xbox = std::make_unique<EvdevXboxInputSource>(args.xbox_device);
    std::unique_ptr<AsyncMujocoViewer> viewer;
    if (args.viewer) {
      // The viewer maps measured joints using the MJCF actuator order. Validate
      // that order against the policy even when physics runs on the real G1.
      if (args.backend == "unitree") sim_substeps_and_validate(args.xml, json);
      viewer = std::make_unique<AsyncMujocoViewer>(
          args.xml, args.glfw_library, args.backend == "unitree", args.seat_body);
    }
    std::unique_ptr<RobotBackend> backend;
    UnitreeSdk2Backend* unitree_backend = nullptr;
    if (args.backend == "mujoco") {
      MuJoCoBackendConfig config;
      config.xml_path = args.xml;
      config.policy_anchor_body_name = "torso_link";
      config.substeps_per_command = sim_substeps_and_validate(args.xml, json);
      config.initial_dof_position = standing_pose();
      backend = std::make_unique<MuJoCoBackend>(std::move(config));
    } else {
      configure_unitree_dds(args.net_if, args.dds_participant_index);
      UnitreeSdk2BackendConfig config;
      config.network_interface = args.net_if;
      config.enable_odometry = true;
      config.max_odometry_age = std::chrono::milliseconds(100);
      config.use_lowstate_imu = true;
      config.imu_is_torso = false;
      config.damp_unused_dex3_hands = true;
      config.root_position_without_odometry = {0.0, 0.0, 0.793};
      auto sdk = std::make_unique<UnitreeSdk2Backend>(std::move(config));
      unitree_backend = sdk.get();
      backend = std::move(sdk);
    }
    ControllerManager controllers;
    if (!controllers.add(std::make_unique<MetatronController>(
            manifest.action_map(),
            [&task](Inputs& inputs, const RobotState& state) { task.build(inputs, state); },
            [&engine](const Inputs& inputs) { return engine.run(inputs); })))
      throw std::runtime_error("cannot register Metatron v1 controller");
    SafetyLimits limits;
    limits.max_abs_command = 5.0;
    limits.max_abs_stiffness = 200.0;
    limits.max_abs_damping = 30.0;
    limits.max_cycle_interval = std::chrono::milliseconds(100);
    // Hold at most two missed 50 Hz observations; never infer on a repeated tick.
    limits.repeated_state_grace = std::chrono::milliseconds(60);
    DeploymentRuntime runtime(*backend, controllers, SafetySupervisor(limits));
    RobotState initial;
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    while (Clock::now() < deadline && !backend->read_state(initial))
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (initial.sequence == 0) {
      if (unitree_backend) {
        const auto s = unitree_backend->sensor_status();
        std::cerr << "startup sensors: lowstate=" << s.lowstate_packets
                  << " crc_errors=" << s.lowstate_crc_errors
                  << " lowstate_age_ms=" << s.lowstate_age_ms
                  << " odometry=" << s.odometry_packets
                  << " odometry_age_ms=" << s.odometry_age_ms << '\n';
      }
      throw std::runtime_error("no complete fresh LowState + odometry pair; no safe hold attempted");
    }
    if (args.seat_distance) {
      calibrate_seat(*seat, initial, args.xml, *args.seat_distance);
      task.configure_seat(*seat);
    }
    if (!runtime.start())
      throw std::runtime_error("startup safe hold or state validation failed: " +
                               runtime.last_fault_reason());
    std::signal(SIGINT, signal_stop);
    std::signal(SIGTERM, signal_stop);
    std::cout << "Metatron v1 ready; X=damping, Y=revive idle, A=steering, RB=idle. "
                 "Starts in safe hold. backend=" << args.backend
              << " input=" << (args.unitree_remote ? "G1 remote (RB=R1)" :
                               args.scripted_sim || args.scripted_seat ? "scripted simulation" : "Xbox")
              << " recovery=" << (args.recovery_plan.empty() ? "paper torso-height" : "recorded plan")
              << "\n";
    if (!args.seat_body.empty()) std::cout << "Seat configured from MJCF: B=sit, RB/R1=stand then idle\n";
    std::uint32_t previous = 0;
    bool active = false, steering = false;
    float heading_goal = 0.0F;
    auto next = Clock::now();
    int count = 0;
    std::uint64_t sent = 0, slow = 0, cycle_faults = 0;
    double total_command_ms = 0.0, max_command_ms = 0.0;
    bool seated_seen = false, standing_after_seat = false;
    while (!g_stop && (args.steps == 0 || count < args.steps)) {
      next += std::chrono::milliseconds(20);
      InputState input;
      bool input_ready = false;
      RobotState observed;
      bool has_observed = false;
      if (unitree_backend && (args.unitree_remote || viewer))
        has_observed = unitree_backend->read_state(observed);
      if (args.scripted_seat) {
        input.connected = true;
        input.timestamp = Clock::now();
        input.buttons = count == 10 ? xbox_button::kY :
                        count == 100 ? xbox_button::kB :
                        count == 900 ? xbox_button::kRightBumper : 0;
        input_ready = true;
      } else if (args.scripted_sim) {
        input.connected = true;
        input.timestamp = Clock::now();
        input.buttons = count == 10 ? xbox_button::kY :
                        count == 40 || count == 200 ? xbox_button::kA :
                        count == 60 || count == 130 || count == 140 ? xbox_button::kY :
                        count == 100 || count == 800 ? xbox_button::kRightBumper :
                        count == 120 ? xbox_button::kX : 0;
        input.axes[3] = (count >= 41 && count < 100) ||
                               (count >= 201 && count < 700) ? 0.3F : 0.0F;
        input_ready = true;
      } else if (args.unitree_remote) {
        // Poll transport independently: a faulted runtime stops advancing its
        // last_state, but the operator must still be able to request revival.
        if (has_observed && observed.has_operator_input)
          remote.update(observed.operator_input_data, observed.sequence, observed.timestamp);
        input_ready = remote.poll(input);
      } else input_ready = xbox->poll(input);
      if (!input_ready ||
          Clock::now() - input.timestamp > std::chrono::milliseconds(250)) {
        runtime.emergency_stop();
        active = false;
      } else {
        const auto pressed = input.buttons & ~previous;
        previous = input.buttons;
        // X is a level-triggered safety override: holding X while tapping Y
        // must never revive the policy. Repeated Y while active must not
        // reset policy history, heading, or the current control mode.
        if (input.buttons & xbox_button::kX) {
          if (active || runtime.lifecycle() != LifecycleState::kFault)
            runtime.emergency_stop();
          active = false;
          if (pressed & xbox_button::kX) std::cout << "X: damping hold\n";
        } else if (pressed & xbox_button::kY) {
          if (!active) {
            if (runtime.lifecycle() == LifecycleState::kFault && !runtime.revive())
              throw std::runtime_error("safe revive rejected");
            task.reset();
            if (!runtime.request_controller("metatron_v1"))
              throw std::runtime_error("scene controller selection rejected");
            active = true;
            steering = false;
            std::cout << "Y: idle requested\n";
          } else {
            std::cout << "Y: already active; control unchanged\n";
          }
        } else if (active && (pressed & xbox_button::kRightBumper)) {
          steering = false;
          task.set_idle();
          std::cout << "RB: stand/idle requested\n";
        } else if (active && (pressed & xbox_button::kB) && !args.seat_body.empty()) {
          steering = false;
          task.set_sitting();
          std::cout << "B: sit requested\n";
        } else if (active && (pressed & xbox_button::kA)) {
          steering = true;
          heading_goal = yaw(runtime.last_state().base_orientation_xyzw);
          std::cout << "A: steering\n";
        }
        if (active && steering) {
          const float lx = input.axes[0], ly = input.axes[1];
          const float magnitude = std::min(1.0F, std::hypot(lx, ly));
          if (magnitude > 0.15F) {
            const float amount = (magnitude - 0.15F) / 0.85F;
            const float angle = std::atan2(-lx, ly) * amount;
            const float quantum = 3.14159265358979323846F / 18.0F;
            heading_goal = wrap(yaw(runtime.last_state().base_orientation_xyzw) +
                                std::round(angle / quantum) * quantum);
          }
          const float rx = input.axes[2], ry = input.axes[3];
          const float level = std::min(1.0F, std::hypot(rx, ry));
          const float speed = level <= 0.15F ? 0.0F :
              std::min(1.0F, (level - 0.15F) / 0.85F);
          const float direction = heading_goal + std::atan2(-rx, ry);
          task.set_steering(heading_goal,
                            {speed * std::cos(direction), speed * std::sin(direction)});
        }
      }
      const auto result = runtime.run_cycle(Clock::now());
      seated_seen |= task.seat_phase() == "seated";
      standing_after_seat |= seated_seen && count > 900 && task.seat_phase() == "inactive" &&
                             runtime.last_state().root_position[2] > (seat ? seat->ground_z : 0) + 0.7;
      double command_ms = 0.0;
      if (result == CycleResult::kSent && runtime.last_cycle_timing().valid()) {
        command_ms = std::chrono::duration<double, std::milli>(
            runtime.last_cycle_timing().state_to_command()).count();
        ++sent;
        slow += command_ms > 20.0;
        total_command_ms += command_ms;
        max_command_ms = std::max(max_command_ms, command_ms);
      }
      if (result == CycleResult::kSafetyFault || result == CycleResult::kBackendFault ||
          result == CycleResult::kNoState) {
        ++cycle_faults;
        active = false;
        std::cerr << "safe fault: " << runtime.last_fault_reason() << '\n';
        if (unitree_backend && result == CycleResult::kNoState) {
          const auto s = unitree_backend->sensor_status();
          std::cerr << "sensor status: lowstate=" << s.lowstate_packets
                    << " age_ms=" << s.lowstate_age_ms
                    << " crc_errors=" << s.lowstate_crc_errors
                    << " odometry=" << s.odometry_packets
                    << " age_ms=" << s.odometry_age_ms << '\n';
        }
      }
      if (count % 50 == 0) {
        const auto& state = runtime.last_state();
        std::cout << "cycle=" << count << " result=" << static_cast<int>(result)
                  << " active=" << active << " steering=" << steering
                  << " root_xy=" << state.root_position[0] << ',' << state.root_position[1]
                  << " root_z=" << state.root_position[2]
                  << " recovery=" << static_cast<int>(task.recovery_phase())
                  << " seat=" << task.seat_phase()
                  << " command_ms=" << command_ms
                  << " buttons=" << input.buttons << std::endl;
      }
      if (viewer) {
        // A fault freezes the policy's last accepted state, not the robot.
        // Continue displaying fresh transport measurements while damping.
        viewer->publish(has_observed ? observed : runtime.last_state(),
                        {runtime.lifecycle(), task.mode() == MetatronTask::Mode::kSeat
                             ? std::string(task.seat_phase()) : steering ? "STEERING" : "IDLE", command_ms,
                         runtime.fault_count(), runtime.last_fault_reason(), seat});
        if (viewer->closed()) break;
      }
      ++count;
      std::this_thread::sleep_until(next);
    }
    if (!runtime.stop()) std::cerr << "warning: final safe hold was not confirmed\n";
    std::cout << "summary cycles=" << count << " sent=" << sent
              << " faults=" << cycle_faults << " command_over_20ms=" << slow
              << " command_mean_ms=" << (sent ? total_command_ms / sent : 0.0)
              << " command_max_ms=" << max_command_ms << std::endl;
    if (args.scripted_seat) {
      std::cout << "seat fixture: seated=" << seated_seen << " stood=" << standing_after_seat << '\n';
      if (!seated_seen || !standing_after_seat || cycle_faults) return 2;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Metatron native launch failed: " << error.what() << '\n';
    return 1;
  }
}
