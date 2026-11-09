# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate repeatable policy parity inputs from the released ONNX and FK.

Exercises reset and incoming-action history, local height, global body goals,
velocity, style, and visible/hidden scenes. These are numerical contract cases,
not a locomotion success benchmark. The binary format is consumed by the
existing graph-safe and native TensorRT parity tools.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import onnxruntime as ort

from deployment.native.examples.metatron_v1_contract import validate_manifest
from deployment.native.tools.metatron_v1_reference import _multimodal_style_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=48)
    args = parser.parse_args()
    if not 6 <= args.samples <= 1000:
        parser.error("--samples must be between 6 and 1000")
    manifest = json.loads((args.artifact_dir / "manifest.json").read_text())
    validate_manifest(manifest)
    model = args.artifact_dir / "model.onnx"
    if hashlib.sha256(model.read_bytes()).hexdigest() != manifest["artifact_sha256"]:
        raise ValueError("reference ONNX differs from the pinned manifest")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    policy = ort.InferenceSession(
        str(model), options, providers=["CPUExecutionProvider"]
    )
    fk = ort.InferenceSession(
        str(args.artifact_dir / "fk.onnx"), options, providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(20261109)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        output.write(b"PM_SCENE_V1\0")
        output.write(struct.pack("<I", args.samples))
        for sample in range(args.samples):
            feed = {
                name: np.zeros(
                    (1, *shape), dtype=np.bool_ if kind == "bool" else np.float32
                )
                for name, shape, kind in zip(
                    manifest["input_names"],
                    manifest["input_shapes"],
                    manifest["input_dtypes"],
                    strict=True,
                )
            }
            q = np.asarray(manifest["action_map"]["offset"], np.float32)
            q = (q + rng.uniform(-0.12, 0.12, 29)).astype(np.float32)[None]
            position = rng.uniform((-2, -2, 0.4), (2, 2, 1)).astype(np.float32)[None]
            rotation = rng.normal(size=(1, 4)).astype(np.float32)
            rotation /= np.linalg.norm(rotation, axis=-1, keepdims=True)
            feed["noisy_dof_pos"][:] = q
            feed["noisy_dof_vel"][:] = rng.uniform(-0.4, 0.4, (1, 29))
            feed["noisy_anchor_pos"][:] = position
            feed["noisy_anchor_rot"][:] = rotation
            feed["noisy_root_local_ang_vel"][:] = rng.uniform(-0.2, 0.2, (1, 3))
            body_pos, body_rot = fk.run(
                None,
                {
                    "noisy_dof_pos": q,
                    "noisy_anchor_pos": position,
                    "noisy_anchor_rot": rotation,
                },
            )
            feed["current_root_pos"][:] = body_pos[:, 0]
            feed["current_root_rot"][:] = body_rot[:, 0]
            for key in ("dof_pos", "dof_vel", "anchor_rot", "root_local_ang_vel"):
                feed[f"noisy_historical_{key}"][:] = feed[f"noisy_{key}"][:, None]
            if sample % 2:
                feed["raw_action_history"][:] = rng.uniform(-0.2, 0.2, (1, 9, 29))
            feed["noisy_historical_raw_actions"][:] = feed["raw_action_history"][:, 1:]

            for prefix in ("", "local_"):
                feed[f"masked_mimic_{prefix}ref_pos"][:] = body_pos[:, None]
                feed[f"masked_mimic_{prefix}ref_rot"][:] = body_rot[:, None]
                feed[f"masked_mimic_{prefix}time_offsets"][:] = [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                ]
            case = sample % 6
            if case == 1:  # recovery-like height prompt
                feed["masked_mimic_local_ref_pos"][0, :, 16, 2] = (
                    body_pos[0, :, 2].min() + 0.7657
                )
                feed["masked_mimic_local_target_poses_masks"][:] = True
                feed["masked_mimic_local_target_bodies_masks"].reshape(5, 5, 2)[
                    :, 0, 0
                ] = True
            elif case in (2, 5):
                feed["masked_mimic_target_poses_masks"][:] = True
                feed["masked_mimic_target_bodies_masks"][:] = True
                feed["masked_mimic_anchor_translation_mask"][:] = True
                feed["masked_mimic_anchor_rotation_mask"][:] = True
            elif case == 3:
                feed["masked_mimic_velocity_command_mask"][:] = True
                feed["masked_mimic_average_velocity"][:] = rng.uniform(-1, 1, (1, 2))
            elif case == 4:
                state = {
                    "body_pos": body_pos[0],
                    "body_rot": body_rot[0],
                    "body_vel": np.zeros_like(body_pos[0]),
                    "body_ang_vel": np.zeros_like(body_pos[0]),
                }
                style, velocity = _multimodal_style_features(
                    [state] * 5, np.array([16, 13, 7, 24, 32])
                )
                feed["masked_mimic_style_pose_samples"][:] = style
                feed["masked_mimic_style_velocity_samples"][:] = velocity
                feed["masked_mimic_style_condition_masks"][:] = True
                feed["masked_mimic_style_frame_masks"][:] = True
            feed["scene_boxes_object_rot"][..., 3] = 1
            feed["scene_boxes_centers"][:] = [[[0, 0, 0.25], [0, 0.3, 0.6]]]
            feed["scene_boxes_extents"][:] = [[[0.5, 0.4, 0.25], [0.5, 0.1, 0.3]]]
            feed["scene_boxes_object_pos"][:] = position + [0.7, 0, -0.5]
            feed["scene_boxes_valid"][:] = True
            feed["scene_boxes_visible"][:] = case == 5
            outputs = policy.run(manifest["output_names"], feed)
            if not all(np.isfinite(value).all() for value in outputs):
                raise ValueError(f"non-finite reference output at sample {sample}")
            for name in manifest["input_names"]:
                output.write(np.ascontiguousarray(feed[name]).tobytes())
            for value in outputs:
                output.write(np.asarray(value, dtype="<f4").tobytes())
    print(f"Policy fixture: {args.output}, {args.samples} samples across 6 goal cases")


if __name__ == "__main__":
    main()
