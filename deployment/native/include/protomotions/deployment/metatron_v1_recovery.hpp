// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "protomotions/deployment/metatron_v1_contract.hpp"

namespace protomotions::deployment::metatron_v1 {

struct RecoveryPose {
  std::array<float, 33 * 3> positions{};
  std::array<float, 33 * 4> rotations{};
  std::array<float, 29> joints{};
};

// Paper, appendix G(c): torso-height-only goals with 0.65/0.90 hysteresis.
// This public rule needs no motion recording. Height is relative to the lowest
// FK body origin; timing requests a rise at 0.8 m/s, not a velocity command.
class HeightRecovery final {
 public:
  explicit HeightRecovery(float upright_height);
  void reset() noexcept { active_ = false; }
  bool step(Inputs& inputs, const RecoveryPose& pose);
  bool active() const noexcept { return active_; }

 private:
  float upright_height_;
  bool active_ = false;
};

class StagedRecovery final {
 public:
  enum class Phase { kInactive, kPronePlan, kSeatedPlan, kRamp, kSettle };

  StagedRecovery(const std::string& plan_path,
                 const std::vector<std::string>& policy_body_names);
  void reset() noexcept;
  // Returns true while recovery owns the goal tensors. A completed settle
  // still owns the current cycle; the next cycle returns false.
  bool step(Inputs& inputs, const RecoveryPose& pose, float dt_seconds);
  Phase phase() const noexcept { return phase_; }
  bool just_finished() const noexcept { return just_finished_; }

 private:
  void transition(Phase phase) noexcept;
  bool intentional_low(const Inputs& inputs, float height, float up) const;
  void apply(Inputs& inputs, float height);
  void sample_plan(Inputs& inputs, float source_start_seconds);

  std::vector<std::array<float, 33 * 3>> positions_;
  std::vector<std::array<float, 33 * 4>> rotations_;
  std::vector<std::array<float, 33 * 3>> mirrored_positions_;
  std::vector<std::array<float, 33 * 4>> mirrored_rotations_;
  float frame_dt_seconds_ = 0.0F;
  Phase phase_ = Phase::kInactive;
  float elapsed_ = 0.0F;
  float phase_start_ = 0.0F;
  float ready_since_ = 0.0F;
  bool has_ready_since_ = false;
  bool mirror_prone_ = false;
  bool just_finished_ = false;
};
}  // namespace protomotions::deployment::metatron_v1
