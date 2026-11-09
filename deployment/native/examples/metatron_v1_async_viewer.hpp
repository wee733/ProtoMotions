// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "metatron_v1_mujoco_viewer.hpp"

namespace protomotions::deployment::metatron_v1 {

// Rendering owns its own MuJoCo data and GL context. The control thread only
// publishes a small measured-state snapshot, never waits for a GPU draw.
class AsyncMujocoViewer final {
 public:
  AsyncMujocoViewer(std::string xml, std::string glfw_library, bool hardware = false) {
    std::promise<void> initialized;
    auto ready = initialized.get_future();
    thread_ = std::thread([this, xml = std::move(xml),
                           glfw_library = std::move(glfw_library), hardware,
                           initialized = std::move(initialized)]() mutable {
          bool started = false;
          try {
            MujocoViewer viewer(xml, glfw_library, hardware);
            initialized.set_value();
            started = true;
            while (!stop_.load(std::memory_order_relaxed)) {
              RobotState state;
              ViewerStatus status;
              bool available = false;
              {
                std::lock_guard<std::mutex> lock(mutex_);
                if (has_state_) {
                  state = latest_;
                  status = status_;
                  available = true;
                }
              }
              if (available && !viewer.render(state, status)) {
                closed_.store(true, std::memory_order_relaxed);
                break;
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
          } catch (const std::exception& error) {
            if (!started) initialized.set_exception(std::current_exception());
            std::cerr << "MuJoCo viewer failed: " << error.what() << '\n';
            closed_.store(true, std::memory_order_relaxed);
          }
        });
    // Finish renderer setup before the caller creates an actuator backend.
    try {
      ready.get();
    } catch (...) {
      thread_.join();
      throw;
    }
  }

  ~AsyncMujocoViewer() {
    stop_.store(true, std::memory_order_relaxed);
    if (thread_.joinable()) thread_.join();
  }
  AsyncMujocoViewer(const AsyncMujocoViewer&) = delete;
  AsyncMujocoViewer& operator=(const AsyncMujocoViewer&) = delete;

  void publish(const RobotState& state, const ViewerStatus& status) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_ = state;
    status_ = status;
    has_state_ = true;
  }
  bool closed() const noexcept { return closed_.load(std::memory_order_relaxed); }

 private:
  std::mutex mutex_;
  RobotState latest_;
  ViewerStatus status_;
  bool has_state_ = false;
  std::atomic<bool> stop_{false};
  std::atomic<bool> closed_{false};
  std::thread thread_;
};

}  // namespace protomotions::deployment::metatron_v1
