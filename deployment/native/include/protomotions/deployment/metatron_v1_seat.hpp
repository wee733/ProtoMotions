// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <string_view>

#include "protomotions/deployment/metatron_v1_recovery.hpp"

namespace protomotions::deployment::metatron_v1 {

// One fixed chair, in the same world frame as RobotState. Object +X points
// out of the seat, +Y left, +Z up. Box extents are FULL lengths, in metres.
// Box 0 is the seat; box 1 is the back. Both share the object's yaw.
struct SeatGeometry {
  std::array<float, 3> object_position{};
  float yaw = 0.0F;
  float ground_z = 0.0F;
  std::array<float, 6> centers{};
  std::array<float, 6> extents{};
};

// Position a centered chair behind the measured ankle midpoint. Distance is
// positive from that midpoint back to the seat's front edge, in metres.
void align_seat_behind(SeatGeometry& seat, std::array<float, 2> ankle_midpoint,
                       float heading, float ground_z, float distance);

// Native adapter of the official browser demo's scene_interaction.ts rule
// (compiled release assets/main-BE8jQTMs.js, l_.interactionTargets/step).
// Starts beside the chair; navigation to the chair is not part of this skill.
class SeatInteraction final {
 public:
  enum class Phase { kInactive, kAlign, kSit, kSeated, kStand };
  SeatInteraction(SeatGeometry geometry, float upright_height);
  void reset() noexcept;
  void sit() noexcept;
  void stand() noexcept;
  // Preserve the caller's stationary steering prompt during alignment. Other
  // phases own all task masks, but never change sensor/action history.
  void step(Inputs& inputs, const RecoveryPose& measured, float dt);
  bool supported(const RecoveryPose& measured) const noexcept;
  bool active() const noexcept { return phase_ != Phase::kInactive; }
  float heading() const noexcept { return geometry_.yaw; }
  Phase phase() const noexcept { return phase_; }
  std::string_view phase_name() const noexcept;

 private:
  std::array<float, 3> seat_center() const noexcept;
  void scene(Inputs& inputs) const;
  void goals(Inputs& inputs, const RecoveryPose& measured) const;
  SeatGeometry geometry_;
  float upright_height_;
  Phase phase_ = Phase::kInactive;
  float elapsed_ = 0.0F, stable_ = 0.0F, rise_stable_ = 0.0F;
  bool stand_initialized_ = false, boxes_released_ = false;
  std::array<float, 3> stand_start_{};
};
}  // namespace protomotions::deployment::metatron_v1
