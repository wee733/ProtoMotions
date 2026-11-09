# Metatron v1 on Jetson Thor and G1

Validated on 2026-11-09 with Thor T5000 (aarch64, SM 11.0), L4T 39.2.1,
TensorRT 10.16.2.10, CUDA 13 runtime, and G1's own wireless remote. Thor
performs inference and sends SDK2 commands over Ethernet; no inference service
is installed on G1. This is pretrained inference/deployment, not retraining
the paper's experts or student.

## Sources and model contract

- Upstream code baseline: `b43fc6fcfebb4996eb651dda8ea8ce981c5433a6`.
- Fork: https://github.com/wee733/ProtoMotions, branch `thor/metatron-v1`.
- Project/paper: https://research.nvidia.com/labs/par/projects/metatron/.
- PyTorch package: `nvidia/Metatron-v1` revision
  `ada816723d62c194c82ffd8fab97a3d8a001f94d`, downloaded with
  `python scripts/download_metatron.py` and its integrity checks.
- Native 39-input ONNX: the same official model repository, revision
  `167355ae5630ebaece752545516da5217bf15d04`, `demo/model.onnx`.
  SHA256: `7180e82cae7ec04ff7766c262d430f57573d93678e025d8a71916885072a0cd4`.
  This matches the public demo manifest's `single_file_artifact_sha256`.
- The original ONNX remains unchanged. The upstream graph-safe rewrite removes
  dynamic masked operators for CUDA Graph capture. The generic 37-input export
  is a different ABI and must not replace this launcher's 39-input model.

The paper's external RTX 5090 is replaced by Thor. The deterministic policy,
29-joint action map, accepted-action history and 50 Hz control rate are kept.
The native MuJoCo asset defaults to 500 Hz physics; the paper's G1 training
uses 200 Hz and the released Python MuJoCo configuration uses 1000 Hz. These
simulation settings are distinct from the shared 50 Hz policy rate.

No `--recovery-plan` selects the paper Appendix G(c) torso-height prompt:
enter below 0.65H, exit above 0.90H, first deadline
`clamp((H-h)/0.8, 0.25, 1.25)` seconds, then +0.25 seconds per slot capped at
2 seconds. All other task cues are hidden. A supplied `--recovery-plan`
retains the upstream staged mode. The demo-only recovery clip is not used
on hardware. Height recovery has **not** passed the prone-start test below.

## Local layout and environment

Workspace: `/home/thor/lbz_ws/metatron`; source: `ProtoMotions/`; Python:
`.venv/bin/python`; compiled runtime: `build/native/`; artifacts:
`artifacts/metatron-v1/`; logs: `logs/`.

The venv uses `--system-site-packages` to reuse the functioning system
`torch==2.11.0+cu130` and TensorRT. Project extras `mujoco,dev`, Ninja, and
Ruff 0.8.1 are installed there. MuJoCo is 3.17.0. G1 meshes, the seed motion,
and the official demo zip were fetched through Git LFS. Preserve the model
LICENSE, NOTICE, and model card beside the downloaded artifacts.

This follows the upstream [installation guidance](../../../docs/source/getting_started/installation.rst),
which supports conda/venv for MuJoCo; uv is another supported route, not a
requirement for this C++ runtime. Retain the working environment and the
upstream download, export, build and launch tools. Make targeted adaptations
for observed integration problems instead of introducing a replacement runner.

Build/run commands below use variables only to shorten paths:

```bash
METATRON_ROOT=/home/thor/lbz_ws/metatron
METATRON_ARTIFACTS="$METATRON_ROOT/artifacts/metatron-v1"
cd "$METATRON_ROOT/ProtoMotions"
export NVIDIA_TF32_OVERRIDE=0
```

Keep `NVIDIA_TF32_OVERRIDE=0` during **both engine build and execution**.
These plans' Myelin runtime rejects a different TF32 environment. The C++
runtime accepts TensorRT 10.9+ within 10.x and still checks exact agreement
between headers and the linked library. Rebuild plans for another GPU/runtime.

## Build

```bash
cmake -S deployment/native -B "$METATRON_ROOT/build/native" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$METATRON_ROOT/.venv/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPROTOMOTIONS_BUILD_UNITREE_SDK2=ON \
  -DPROTOMOTIONS_BUILD_MUJOCO=ON -DPROTOMOTIONS_BUILD_METATRON_NATIVE=ON \
  -DUNITREE_SDK2_ROOT=/home/thor/lbz_ws/g1_ws/unitree_ws/src/unitree_sdk2 \
  -DMUJOCO_ROOT="$METATRON_ROOT/.venv/lib/python3.12/site-packages/mujoco" \
  -DTENSORRT_ROOT=/usr -DMETATRON_TENSORRT_INCLUDE=/usr/include/aarch64-linux-gnu \
  -DTENSORRT_LIBRARY_ROOT=/usr/lib/aarch64-linux-gnu \
  -DCUDA_RUNTIME_ROOT=/usr/local/cuda/targets/sbsa-linux \
  -DCUDA_TOOLKIT_INCLUDE_ROOT=/usr/local/cuda
cmake --build "$METATRON_ROOT/build/native" -j6
ctest --test-dir "$METATRON_ROOT/build/native" --output-on-failure
```

The exported FK and compiled plans already exist. To reproduce in a fresh
artifact directory containing the original model and single-file manifest:

```bash
"$METATRON_ROOT/.venv/bin/python" -m deployment.export_fk_to_onnx \
  --checkpoint data/pretrained_models/metatron/G1/inference_last.ckpt \
  --output "$METATRON_ARTIFACTS"
"$METATRON_ROOT/.venv/bin/python" -m deployment.native.tools.rewrite_metatron_graphsafe \
  "$METATRON_ARTIFACTS/model.onnx" "$METATRON_ARTIFACTS/model.graphsafe.onnx"
"$METATRON_ROOT/.venv/bin/python" -m deployment.native.tools.build_metatron_tensorrt \
  --artifact-dir "$METATRON_ARTIFACTS" \
  --onnx "$METATRON_ARTIFACTS/model.graphsafe.onnx" \
  --output "$METATRON_ARTIFACTS/model.graphsafe.opt1.fp32.plan" --optimization-level 1
"$METATRON_ROOT/.venv/bin/python" -m deployment.native.tools.build_metatron_fk_tensorrt \
  --onnx "$METATRON_ARTIFACTS/fk.onnx" --sidecar "$METATRON_ARTIFACTS/fk.yaml" \
  --output "$METATRON_ARTIFACTS/fk.fp32.plan" --manifest "$METATRON_ARTIFACTS/manifest.json"
```

Builders refuse to overwrite plans. Optimization level 1 is essential here:
level 0 measured 17.48 ms median / 21.38 ms P95; level 1 measured 3.44 / 5.58 ms.

## Run and control

Follow the upstream [native launch workflow](metatron_v1_native.md) and run
the existing executable in a terminal in the foreground. The temporary
`metatron-g1.service` used in the initial experiment has been stopped and
removed. There is no systemd deployment, autostart, or added launcher script.
The earlier hardware log is retained at `logs/g1-runtime.log` as test evidence.

With G1 connected, run the following command using the path variables defined
above. Start only one hardware controller process:

```bash
NVIDIA_TF32_OVERRIDE=0 "$METATRON_ROOT/build/native/metatron_v1_deploy_cpp" \
  --backend unitree --net-if enP2p1s0 --arm --unitree-remote \
  --engine "$METATRON_ARTIFACTS/model.graphsafe.opt1.fp32.plan" \
  --fk-engine "$METATRON_ARTIFACTS/fk.fp32.plan" \
  --manifest "$METATRON_ARTIFACTS/manifest.json" --cuda-graphs \
  --viewer --xml "$METATRON_ARTIFACTS/g1_sim.xml" \
  --glfw-lib /lib/aarch64-linux-gnu/libglfw.so.3
```

The optional `--viewer` opens **Metatron - LIVE G1**. It reconstructs the
measured joint pose and odometry using the existing MuJoCo renderer; it does
not simulate the hardware or supply policy state. The overlay reports mode,
measurement age, command timing and faults, and marks stale measurements.
Rendering owns a separate GL/MuJoCo thread; initialization must succeed before
the actuator backend is created. Omit the last two lines for terminal-only use.

Watch status in the same terminal. Closing the viewer stops the controller.
**Ctrl+C** sends SIGINT to the foreground
process; its existing signal handler ends the loop and executes `runtime.stop()`
before exit. X selects damping without exiting. Each new invocation starts
in damping and awaits Y. ROS launch/service/action integration is separate
from process supervision and is not implemented by this command.

Thor: `192.168.123.160/24`; G1: `192.168.123.164`; DDS interface: `enP2p1s0`.
The upstream backend releases the high-level controller and starts in damping.
The remote is decoded from LowState's 40-byte payload, independently of the
runtime's last accepted state, so Y can revive a faulted runtime.

| G1 input | Function |
| --- | --- |
| Y | Activate/revive into idle; repeated Y preserves active mode/history |
| A | Enter steering |
| Left stick | Choose heading; release holds that heading |
| Right stick | Planar travel relative to heading; upstream cap 1 m/s |
| R1 | Return to idle/standing |
| X | Damping, with priority while held |

The same foreground entry point runs the simulation fixture:

```bash
NVIDIA_TF32_OVERRIDE=0 "$METATRON_ROOT/build/native/metatron_v1_deploy_cpp" \
  --backend mujoco --xml "$METATRON_ARTIFACTS/g1_sim.xml" \
  --scripted-sim --steps 1500 \
  --engine "$METATRON_ARTIFACTS/model.graphsafe.opt1.fp32.plan" \
  --fk-engine "$METATRON_ARTIFACTS/fk.fp32.plan" \
  --manifest "$METATRON_ARTIFACTS/manifest.json" --cuda-graphs
```

The fixture tests activation, repeated Y, steering, X stop, revival, sustained
walking, and idle. Ctrl+C also ends a foreground simulation through its normal
stop path.

The foreground workflow was checked in a PTY: 1548 cycles, 1528 policy commands,
zero runtime faults, followed by a real Ctrl+C and exit code 0. The removed
service was confirmed absent, with no remaining controller process. See
`logs/native-foreground.log`. G1 was later reconnected for the live viewer
check: the operator confirmed synchronized posture, standing and walking.
See `logs/g1-live-viewer.log`; this foreground session subsequently exited.

The [official browser demo](https://nvlabs.github.io/ProtoMotions/user_guide/browser_demo.html)
is a separate simulation interface. Upstream distributes its compiled static
bundle, not the TypeScript project or a hardware bridge. The native live
viewer does not make the browser's IK/scene controls command the physical G1.

## Evidence and remaining work

| Check | Observed result |
| --- | --- |
| Official pretrained Python tests | 10 passed, including MuJoCo rollout/reset |
| Native CTest | 17/17 passed |
| Original vs graph-safe ONNX, 48 feeds / 6 goal cases | All four outputs identical |
| Native FP32 policy vs original ONNX | Target max error 5.30e-6 rad; raw action 1.27e-5; gains exact |
| Native FK, 64 feeds | Max position 2.38e-7 m; quaternion 2.38e-7 |
| Policy-only benchmark, 500 calls | P50 3.44 ms, P95 5.58 ms, max 7.62 ms |
| Native 30-second walk/stop simulation | 1480 policy commands, zero faults, zero state-to-command times above 20 ms; mean 3.48 ms, max 9.68 ms; ~2.4 m travel, stable stop |
| Physical G1 | Operator confirmed Y standing and A/joystick walking; runtime logged both transitions and active command transmission. R1 remains available; its use was not observed in this run |
| Live G1 viewer | Operator confirmed measured posture synchronization, standing and walking; viewer-enabled simulation sent 380 commands with zero runtime faults |
| Prone height-only recovery | Failed to stand in 30 s at 500 Hz; 20 s checks at 1000 Hz Euler and 200 Hz implicitfast also stayed low |

The numerical fixture generator is
`python -m deployment.native.tools.make_metatron_policy_fixture --artifact-dir ... --output ...`.
Its 48 feeds test numerical contracts, not skill success rates. Physical timing
samples in `g1-runtime.log` measure state-read to command-ready, excluding sensor
acquisition and downstream transport; the policy benchmark is a different boundary.
The paper's 1.3 ms RTX 5090 result is not a Thor measurement.

During the physical run, another `VLLM::EngineCore` process appeared on the
same Thor GPU (about 40.7 GiB reported GPU memory). The once-per-second control
samples included intervals above 20 ms, with an observed maximum of 31.89 ms;
no runtime fault was logged. Shared GPU contention is a plausible contributor,
but no isolated A/B comparison was performed. This run therefore does not
establish a 20 ms worst-case deadline under concurrent model serving. The
standalone benchmark and the zero-overrun simulation table above must not be
used to claim otherwise. The unrelated serving process was left unchanged.

Object interaction, motion tracking, style selection and PICO control are not
exposed by this launcher. Neither the standing/walking result nor the unit tests
establish reproduction of every paper skill. The failed prone fixture remains
open and must not be presented as successful recovery.
