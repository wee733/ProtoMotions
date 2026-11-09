// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_recovery.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

using namespace protomotions::deployment::metatron_v1;

int main() {
  HeightRecovery recovery(0.8F);
  RecoveryPose pose;
  for (int body = 0; body < 33; ++body) {
    pose.positions[body * 3 + 2] = 2.0F;  // arbitrary localization origin
    pose.rotations[body * 4 + 3] = 1.0F;
  }
  Inputs inputs;
  pose.positions[16 * 3 + 2] = 2.6F;
  assert(!recovery.step(inputs, pose));

  // A fall preempts every other goal, preserves sensor/action history and
  // requests only local torso height; world Z origin must not affect timing.
  inputs.floats(22)[0] = 0.42F;
  inputs.floats(30)[0] = -0.31F;
  for (std::size_t i = 0; i < kInputCount; ++i)
    if (kInputs[i].is_bool)
      std::fill(inputs.booleans(i).begin(), inputs.booleans(i).end(), 1);
  pose.positions[16 * 3 + 2] = 2.4F;
  assert(recovery.step(inputs, pose));
  assert(inputs.floats(22)[0] == 0.42F);
  assert(inputs.floats(30)[0] == -0.31F);
  assert(std::abs(inputs.floats(9)[0] - 0.5F) < 1e-6F);
  assert(std::abs(inputs.floats(9)[4] - 1.5F) < 1e-6F);
  for (std::size_t slot = 0; slot < 5; ++slot) {
    assert(std::abs(inputs.floats(5)[(slot * 33 + 16) * 3 + 2] - 2.8F) < 1e-6F);
    assert(inputs.booleans(8)[slot] == 1);
    for (std::size_t field = 0; field < 10; ++field)
      assert(inputs.booleans(7)[slot * 10 + field] == (field == 0));
  }
  for (std::size_t i = 0; i < kInputCount; ++i)
    if (kInputs[i].is_bool && i != 7 && i != 8)
      assert(std::all_of(inputs.booleans(i).begin(), inputs.booleans(i).end(),
                         [](auto value) { return value == 0; }));

  // Hysteresis prevents leaving recovery on crossing only the entry threshold.
  pose.positions[16 * 3 + 2] = 2.6F;
  assert(recovery.step(inputs, pose));
  pose.positions[16 * 3 + 2] = 2.75F;
  inputs.reset();
  inputs.booleans(19)[0] = 1;
  assert(!recovery.step(inputs, pose));
  assert(inputs.booleans(19)[0] == 1);  // task prompt is returned unchanged

  pose.positions[16 * 3 + 2] = 2.4F;
  assert(recovery.step(inputs, pose));
  recovery.reset();
  pose.positions[16 * 3 + 2] = 2.6F;
  assert(!recovery.step(inputs, pose));
}
