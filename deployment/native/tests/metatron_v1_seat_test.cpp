// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/metatron_v1_seat.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace protomotions::deployment::metatron_v1;
namespace {
void near(float a, float b) { assert(std::abs(a-b) < 1e-5F); }
RecoveryPose pose(float z, float heading = 0) {
  RecoveryPose p;
  for (int body = 0; body < 33; ++body) p.rotations[body*4+3] = 1;
  p.positions[50] = z;
  p.rotations[66] = std::sin(heading/2);
  p.rotations[67] = std::cos(heading/2);
  return p;
}
}
int main() {
  SeatGeometry geometry{{0, 0, 0}, 0, 0,
                        {0, 0, 0.23F, -0.26F, 0, 0.71F},
                        {0.48F, 0.46F, 0.46F, 0.04F, 0.46F, 0.5F}};
  auto invalid = geometry;
  invalid.extents[0] = -1;
  try { SeatInteraction bad(invalid, 0.8F); assert(false); }
  catch (const std::invalid_argument&) {}
  SeatInteraction seat(geometry, 0.8F);
  Inputs inputs;
  seat.sit();
  // Elapsed time alone must not cause sitting while facing the chair.
  for (int i = 0; i < 300; ++i) seat.step(inputs, pose(0.8F, 3.14159265F), 0.02F);
  assert(seat.phase() == SeatInteraction::Phase::kAlign);
  inputs.floats(22)[0] = 0.42F;
  inputs.floats(30)[0] = -0.3F;
  for (int i = 0; i < 20; ++i) seat.step(inputs, pose(0.8F), 0.02F);
  assert(seat.phase() == SeatInteraction::Phase::kSit);
  near(inputs.floats(22)[0], 0.42F);
  near(inputs.floats(30)[0], -0.3F);
  near(inputs.floats(10)[48], -0.02F);
  near(inputs.floats(10)[50], 0.58F);
  near(inputs.floats(10)[13*3], 0.29F);
  near(inputs.floats(10)[13*3+1], -0.09F);
  near(inputs.floats(10)[7*3+1], 0.09F);
  near(inputs.floats(32)[0], 0.48F);  // full lengths, not MuJoCo half lengths
  assert(inputs.booleans(37)[0]);
  assert(!inputs.booleans(19)[0]);
  assert(!inputs.booleans(8)[0]);
  assert(!inputs.booleans(13)[0]);
  for (int i = 0; i < 110; ++i) seat.step(inputs, pose(0.58F), 0.02F);
  assert(seat.phase() == SeatInteraction::Phase::kSeated);
  assert(seat.supported(pose(0.58F)));
  assert(!seat.supported(pose(0.2F)));
  seat.stand();
  seat.step(inputs, pose(0.58F), 0.02F);
  assert(inputs.floats(10)[48] > 0.3F);
  near(inputs.floats(10)[50], 0.8F);
  for (int i = 0; i < 7; ++i) seat.step(inputs, pose(0.7F), 0.02F);
  assert(!inputs.booleans(37)[0]);
  for (int i = 0; i < 25; ++i) seat.step(inputs, pose(0.8F), 0.02F);
  assert(!seat.active());
  // Translation/yaw invariance: nonzero ground and 90-degree facing turn
  // forward foot offsets along world Y and keep handedness intact.
  geometry.yaw = 1.5707963268F;
  geometry.ground_z = 2;
  geometry.object_position = {3, 4, 2};
  SeatInteraction rotated(geometry, 0.8F);
  rotated.sit();
  for (int i = 0; i < 20; ++i) rotated.step(inputs, pose(2.8F, geometry.yaw), 0.02F);
  near(inputs.floats(10)[48], 3);
  near(inputs.floats(10)[49], 3.98F);
  near(inputs.floats(10)[50], 2.58F);
  near(inputs.floats(10)[13*3], 3.09F);
  near(inputs.floats(10)[13*3+1], 4.29F);
  near(inputs.floats(10)[13*3+2], 2.06F);
  align_seat_behind(geometry, {5, 6}, 1.5707963268F, -0.08F, 0.2F);
  near(geometry.object_position[0], 5);
  near(geometry.object_position[1], 5.56F);
  near(geometry.object_position[2], -0.08F);
  near(geometry.ground_z, -0.08F);
  // Front edge is exactly the measured distance behind the ankles.
  near(geometry.object_position[1] + geometry.extents[0]/2, 5.8F);
  try { align_seat_behind(geometry, {0, 0}, 0, 0, -0.1F); assert(false); }
  catch (const std::invalid_argument&) {}
}
