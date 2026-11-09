// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/metatron_v1_seat.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace protomotions::deployment::metatron_v1 {
namespace {
constexpr int kTorso = 16;
float up(const RecoveryPose& p) {
  return 1.0F - 2.0F * (std::pow(p.rotations[64], 2) + std::pow(p.rotations[65], 2));
}
float yaw(const RecoveryPose& p) {
  const float* q = p.rotations.data() + 64;
  return std::atan2(2 * (q[3]*q[2] + q[0]*q[1]), 1 - 2*(q[1]*q[1] + q[2]*q[2]));
}
}

void align_seat_behind(SeatGeometry& seat, std::array<float, 2> feet,
                       float heading, float ground, float distance) {
  if (!std::isfinite(feet[0]) || !std::isfinite(feet[1]) || !std::isfinite(heading) ||
      !std::isfinite(ground) || !std::isfinite(distance) || distance < 0)
    throw std::invalid_argument("seat alignment requires finite coordinates and nonnegative distance");
  seat.yaw = heading;
  const float c = std::cos(heading), s = std::sin(heading);
  const float offset = distance + seat.extents[0]/2 + seat.centers[0];
  seat.object_position = {feet[0]-c*offset+s*seat.centers[1],
                          feet[1]-s*offset-c*seat.centers[1], ground};
  seat.ground_z = ground;
}

SeatInteraction::SeatInteraction(SeatGeometry geometry, float upright_height)
    : geometry_(std::move(geometry)), upright_height_(upright_height) {
  const auto finite = [](const auto& values) {
    return std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); });
  };
  if (!finite(geometry_.object_position) || !finite(geometry_.centers) ||
      !finite(geometry_.extents) || !std::isfinite(geometry_.yaw) ||
      !std::isfinite(geometry_.ground_z) || !std::isfinite(upright_height_) ||
      upright_height_ <= 0 || std::any_of(geometry_.extents.begin(), geometry_.extents.end(),
                                       [](float x) { return x <= 0; }))
    throw std::invalid_argument("seat geometry must have finite poses and positive full extents");
}

void SeatInteraction::reset() noexcept {
  phase_ = Phase::kInactive;
  elapsed_ = stable_ = rise_stable_ = 0;
  stand_initialized_ = boxes_released_ = false;
}
void SeatInteraction::sit() noexcept {
  if (active()) return;
  reset();
  phase_ = Phase::kAlign;
}
void SeatInteraction::stand() noexcept {
  if (!active() || phase_ == Phase::kStand) return;
  phase_ = Phase::kStand;
  elapsed_ = stable_ = rise_stable_ = 0;
  stand_initialized_ = boxes_released_ = false;
}
std::string_view SeatInteraction::phase_name() const noexcept {
  switch (phase_) {
    case Phase::kAlign: return "align";
    case Phase::kSit: return "sit";
    case Phase::kSeated: return "seated";
    case Phase::kStand: return "stand";
    default: return "inactive";
  }
}
std::array<float, 3> SeatInteraction::seat_center() const noexcept {
  const float c = std::cos(geometry_.yaw), s = std::sin(geometry_.yaw);
  return {geometry_.object_position[0] + c*geometry_.centers[0] - s*geometry_.centers[1],
          geometry_.object_position[1] + s*geometry_.centers[0] + c*geometry_.centers[1],
          geometry_.object_position[2] + geometry_.centers[2] + geometry_.extents[2]/2};
}
bool SeatInteraction::supported(const RecoveryPose& p) const noexcept {
  if (!active()) return false;
  const auto center = seat_center();
  const float x = p.positions[48] - center[0], y = p.positions[49] - center[1];
  const float c = std::cos(geometry_.yaw), s = std::sin(geometry_.yaw);
  const float above_seat = p.positions[50] - center[2];
  return p.positions[50] - geometry_.ground_z > 0.35F && up(p) > 0.3F &&
         std::abs(c*x + s*y) <= geometry_.extents[0]/2 + 0.12F &&
         std::abs(-s*x + c*y) <= geometry_.extents[1]/2 + 0.12F &&
         above_seat >= 0.04F && above_seat <= 0.35F;
}
void SeatInteraction::scene(Inputs& inputs) const {
  std::copy(geometry_.centers.begin(), geometry_.centers.end(), inputs.floats(31).begin());
  std::copy(geometry_.extents.begin(), geometry_.extents.end(), inputs.floats(32).begin());
  std::copy(geometry_.object_position.begin(), geometry_.object_position.end(), inputs.floats(33).begin());
  inputs.floats(34) = {0, 0, std::sin(geometry_.yaw/2), std::cos(geometry_.yaw/2)};
  std::fill(inputs.floats(35).begin(), inputs.floats(35).end(), 0);
  inputs.floats(38)[0] = 0;
  std::fill(inputs.booleans(36).begin(), inputs.booleans(36).end(), 1);
  inputs.booleans(37)[0] = !boxes_released_;
}
void SeatInteraction::step(Inputs& inputs, const RecoveryPose& p, float dt) {
  if (!std::isfinite(dt) || dt <= 0) throw std::invalid_argument("seat step needs positive finite dt");
  if (!active()) return;
  elapsed_ += dt;
  if (phase_ == Phase::kAlign) {
    stable_ = std::abs(std::remainder(yaw(p)-geometry_.yaw, 6.283185307F)) < 0.523598776F
                  ? stable_ + dt : 0;
    if (stable_ <= 0.3F) { scene(inputs); return; }
    phase_ = Phase::kSit;
    elapsed_ = stable_ = 0;
  }
  const auto center = seat_center();
  if (phase_ == Phase::kSit && elapsed_ > 2 &&
      std::abs(p.positions[50] - (center[2]+0.12F)) < 0.12F &&
      std::hypot(p.positions[48]-center[0], p.positions[49]-center[1]) < 0.25F)
    phase_ = Phase::kSeated;
  if (phase_ == Phase::kStand) {
    if (!stand_initialized_) {
      std::copy_n(p.positions.begin()+48, 3, stand_start_.begin());
      stand_initialized_ = true;
    }
    const float upright_z = geometry_.ground_z + upright_height_;
    const float release_z = stand_start_[2] + std::max(0.06F, (upright_z-stand_start_[2])*0.5F);
    rise_stable_ = p.positions[50] >= release_z && up(p) > 0.5F ? rise_stable_+dt : 0;
    if (rise_stable_ >= 0.12F) boxes_released_ = true;
    stable_ = p.positions[50]-geometry_.ground_z > upright_height_*0.9F && up(p) > 0.9F
                  ? stable_+dt : 0;
    if (stable_ > 0.4F) { reset(); return; }
  }
  goals(inputs, p);
  scene(inputs);
}
void SeatInteraction::goals(Inputs& inputs, const RecoveryPose& p) const {
  for (std::size_t i = 0; i < kInputCount; ++i)
    if (kInputs[i].is_bool) std::fill(inputs.booleans(i).begin(), inputs.booleans(i).end(), 0);
  const auto center = seat_center();
  const float c = std::cos(geometry_.yaw), s = std::sin(geometry_.yaw);
  for (int slot = 0; slot < 5; ++slot) {
    const float time = slot+1;
    auto& positions = inputs.floats(10);
    std::copy(p.positions.begin(), p.positions.end(), positions.begin()+slot*99);
    std::copy(p.rotations.begin(), p.rotations.end(), inputs.floats(11).begin()+slot*132);
    for (int body = 0; body < 33; ++body)
      positions[(slot*33+body)*3+2] = std::max(geometry_.ground_z+0.06F, positions[(slot*33+body)*3+2]);
    const bool standing = phase_ == Phase::kStand;
    const auto& start = standing ? stand_start_ : center;
    const float forward = standing ? std::min(0.65F, 0.33F*(elapsed_+time)) : -0.02F;
    const float z = standing ? stand_start_[2] +
        (geometry_.ground_z+upright_height_-stand_start_[2])*std::min(1.0F, elapsed_+time) : center[2]+0.12F;
    const std::array<float, 3> torso{start[0]+c*forward, start[1]+s*forward, z};
    std::copy(torso.begin(), torso.end(), positions.begin()+(slot*33+kTorso)*3);
    const std::array<float, 4> rotation{0, 0, std::sin(geometry_.yaw/2), std::cos(geometry_.yaw/2)};
    std::copy(rotation.begin(), rotation.end(), inputs.floats(11).begin()+(slot*33+kTorso)*4);
    for (int side = 0; side < 2; ++side) {
      const int body = side == 0 ? 13 : 7;
      const float lateral = side == 0 ? -0.09F : 0.09F;
      const float foot_forward = standing ? -0.04F : geometry_.extents[0]/2+0.07F;
      const std::array<float, 3> foot{torso[0]+c*foot_forward-s*lateral,
          torso[1]+s*foot_forward+c*lateral, geometry_.ground_z+0.06F};
      std::copy(foot.begin(), foot.end(), positions.begin()+(slot*33+body)*3);
      inputs.booleans(16)[slot*10+2+2*side] = 1;
    }
    inputs.booleans(16)[slot*10] = inputs.booleans(16)[slot*10+1] = 1;
    inputs.booleans(17)[slot] = 1;
    inputs.floats(18)[slot] = time;
    inputs.booleans(3)[slot*2] = inputs.booleans(3)[slot*2+1] = 1;
    inputs.booleans(2)[slot*2] = 1;
  }
}
}  // namespace protomotions::deployment::metatron_v1
