# Fixed-seat interaction on Thor / G1

This adapter extends the existing native launcher with the official browser
demo's seat goal rule. The pretrained policy, 39-input ABI, actuator mapping,
50 Hz loop and G1 remote are unchanged. It is an adaptation of a published
simulation skill, not a claim that upstream released this complete hardware
entry point. PICO work is deferred at the operator's request.

## Rule and source

Source: the verified official demo bundle, `assets/main-BE8jQTMs.js`, SHA256
`51481f8114aed594d41b200171a62423822f6c5f47d6a7d71c86e2072a576a79`.
The embedded help names `scene_interaction.ts`; its compiled `l_` class
implements `interactionTargets` and the align/sit/seated/stand transitions.
The TypeScript source is not included in the public release.

- Align with the chair's outward heading within 30 degrees for 0.3 seconds.
- Sitting: torso 2 cm behind the seat center and 12 cm above the seat; feet
  7 cm beyond its half-depth relative to the torso, at +/-9 cm stance width
  and 6 cm above the floor. Repeat these goals at horizons 1–5 seconds.
- Standing: raise the torso to the calibrated upright height, advancing at
  0.33 m/s up to 0.65 m from the measured stand-start position. Release scene
  visibility after sufficient rise; return to idle after height and upright
  orientation remain satisfied for 0.4 seconds.
- Keep supported sitting out of automatic fall recovery. Other task cues
  are masked during sitting/standing. No reference motion clip is used.

Only this nearby-chair interaction is adapted. Browser A* navigation and its
room furniture assets are not transferred to hardware. No training is needed.

## Measured chair and coordinate contract

The operator confirmed seat height 0.46 m, width 0.46 m, depth 0.48 m and
back height 0.50 m above the seat. `assets/chair_46cm.xml` contains two simple
boxes. Back thickness is approximated as 0.04 m pending measurement; the seat
collision box fills its bounding volume, so it is not a detailed chair mesh.

The fixed MJCF body is `chair`, with body-aligned box geoms `chair_seat` and
`chair_back`. +X points out of the seat, +Y left and +Z up. MuJoCo `size`
contains half lengths; `scene_boxes_extents` receives full lengths. Object
centers are local, object position/yaw are in the same world frame as G1's
measured state. A single MJCF definition feeds physics, display and policy.

Simulation uses the chair pose in MJCF. Hardware additionally requires
`--seat-distance` in metres: distance from the ankle midpoint to the seat's
front edge, with the chair centered directly behind the standing robot.
At startup the adapter uses actual joint/root pose to locate the ankles and
the lowest sole collision surface, then positions the chair once in the
backend's startup odometry frame. The viewer uses the identical transform.
This corrects odometry Z offset; it does not detect a chair or track its motion.
Move/reorient the chair or restart from a new placement only with updated
measurements. Startup calibration assumes both feet are on the floor.

For the current placement the operator reported approximately **0.28 m**.
This value is a placement measurement, not a reusable property of the chair.

## Prepare and run

Use the environment and paths in [the platform guide](metatron_thor_g1.md).
Prepare a new output file; the official helper refuses to overwrite artifacts:

```bash
PYTHONPATH=deployment/native "$METATRON_ROOT/.venv/bin/python" \
  deployment/native/tools/prepare_metatron_mujoco_xml.py \
  --source protomotions/data/assets/mjcf/g1_holo_compat.xml \
  --output "$METATRON_ARTIFACTS/g1_seat_sim.xml" \
  --seat-model deployment/native/assets/chair_46cm.xml
```

Repeatable MuJoCo test (Y at cycle 10, B at 100, R1 at 900):

```bash
NVIDIA_TF32_OVERRIDE=0 "$METATRON_ROOT/build/native/metatron_v1_deploy_cpp" \
  --backend mujoco --xml "$METATRON_ARTIFACTS/g1_seat_sim.xml" \
  --seat-body chair --scripted-seat --steps 1500 \
  --engine "$METATRON_ARTIFACTS/model.graphsafe.opt1.fp32.plan" \
  --fk-engine "$METATRON_ARTIFACTS/fk.fp32.plan" \
  --manifest "$METATRON_ARTIFACTS/manifest.json" --cuda-graphs \
  --viewer --glfw-lib /lib/aarch64-linux-gnu/libglfw.so.3
```

Current measured hardware placement:

```bash
NVIDIA_TF32_OVERRIDE=0 "$METATRON_ROOT/build/native/metatron_v1_deploy_cpp" \
  --backend unitree --net-if enP2p1s0 --arm --unitree-remote \
  --xml "$METATRON_ARTIFACTS/g1_seat_sim.xml" --seat-body chair --seat-distance 0.28 \
  --engine "$METATRON_ARTIFACTS/model.graphsafe.opt1.fp32.plan" \
  --fk-engine "$METATRON_ARTIFACTS/fk.fp32.plan" \
  --manifest "$METATRON_ARTIFACTS/manifest.json" --cuda-graphs \
  --viewer --glfw-lib /lib/aarch64-linux-gnu/libglfw.so.3
```

Run one controller, directly in the foreground. **Y** activates standing,
**B** requests sitting, **R1** requests standing then idle, **A** requests
steering (standing first if necessary), **X** selects damping. Closing the
viewer or Ctrl+C executes the existing stop path. No systemd unit is used.

## Validation

- 18/18 native CTests passed. Seat checks cover alignment without a timer-only
  bypass, target masks, full/half extents, preserved sensor/action history,
  support detection, stand transitions and nonzero floor/yaw calibration.
- Two 30-second viewer-enabled simulations each completed sitting and rising:
  1490 commands, zero runtime faults and zero state-to-command times over 20 ms.
  The second run measured mean 4.07008 ms and max 9.28661 ms. These fixed-fixture
  results are not a success rate over chair shapes or initial poses.
- Physical controller was run with the reported 0.28 m placement. Logs
  received multiple B/R1 cycles and seated transitions. Several attempts did
  not reach the seated pose criterion; one entered height recovery and then
  returned upright. No runtime fault was recorded in this observation window.
  The operator subsequently confirmed **physical sitting and standing both
  work normally**. This validates this chair/placement, not every recorded
  attempt or a general success rate. See `logs/g1-seat-runtime.log`.

The foreground terminal session subsequently ended (terminal status 130).
No Metatron controller remains running. The piped log has no final summary,
so this session is not evidence of the normal shutdown path.
