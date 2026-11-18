"""Skeletal animation evaluation and pose computation for glTF models.

Animation clips from separate animation files are retargeted onto a
character model by matching node names between the animation channels and
the character skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from game.model_previewer.gltf_loader import (
    AnimationChannelData,
    AnimationClipData,
    GltfModel,
    TARGET_PATH_ROTATION,
    TARGET_PATH_SCALE,
    TARGET_PATH_TRANSLATION,
)
from game.model_previewer.transform_math import (
    Matrix4,
    Quaternion,
    Vector3,
    compose_transform_matrix,
    identity_matrix,
    lerp_vectors,
    slerp_quaternion,
)


@dataclass
class AnimationLibrary:
    """A collection of animation clips targeting one rig."""

    name: str
    rig_name: str
    clips: list[AnimationClipData]


def evaluate_global_node_matrices(
    model: GltfModel, clip: AnimationClipData, time_seconds: float
) -> list[Matrix4]:
    """Evaluate a clip into global node matrices for the given model.

    Nodes without an animated channel keep their rest transform.  The
    returned list holds one 4x4 matrix per node, indexed by node index.
    """
    channels_by_node_name: dict[str, dict[str, AnimationChannelData]] = {}
    for channel in clip.channels:
        channels_by_node_name.setdefault(channel.node_name, {})[
            channel.target_path
        ] = channel

    local_matrices: list[Matrix4] = []
    for node in model.nodes:
        node_channels = channels_by_node_name.get(node.name)
        if node_channels is None:
            local_matrices.append(node.local_matrix)
            continue
        translation = _sample_translation(
            node_channels.get(TARGET_PATH_TRANSLATION), time_seconds, node.translation
        )
        rotation = _sample_rotation(
            node_channels.get(TARGET_PATH_ROTATION), time_seconds, node.rotation
        )
        scale = _sample_translation(
            node_channels.get(TARGET_PATH_SCALE), time_seconds, node.scale
        )
        local_matrices.append(compose_transform_matrix(translation, rotation, scale))

    return _accumulate_global_matrices(model, local_matrices)


def compute_rest_pose_global_matrices(model: GltfModel) -> list[Matrix4]:
    """Compute global node matrices for the rest pose of the model."""
    local_matrices = [node.local_matrix for node in model.nodes]
    return _accumulate_global_matrices(model, local_matrices)


def _accumulate_global_matrices(
    model: GltfModel, local_matrices: list[Matrix4]
) -> list[Matrix4]:
    """Traverse the node hierarchy from the roots and multiply parent transforms."""
    global_matrices: list[Matrix4] = [identity_matrix() for _ in model.nodes]
    pending_stack: list[tuple[int, Matrix4]] = [
        (root_index, identity_matrix()) for root_index in model.root_node_indices
    ]
    while pending_stack:
        node_index, parent_matrix = pending_stack.pop()
        combined_matrix = parent_matrix @ local_matrices[node_index]
        global_matrices[node_index] = combined_matrix.astype(np.float32)
        for child_index in model.nodes[node_index].children:
            pending_stack.append((child_index, combined_matrix))
    return global_matrices


def compute_skinning_matrices(
    model: GltfModel,
    global_node_matrices: list[Matrix4],
    offset_matrix: Matrix4,
) -> npt.NDArray[np.float32]:
    """Compute per joint skinning matrices in the order of the skin joints."""
    skin = model.skin
    if skin is None:
        raise ValueError("model has no skin")
    joint_count = len(skin.joint_node_indices)
    skinning_matrices = np.zeros((joint_count, 4, 4), dtype=np.float32)
    for joint_slot in range(joint_count):
        joint_node_index = skin.joint_node_indices[joint_slot]
        skinning_matrices[joint_slot] = offset_matrix @ (
            global_node_matrices[joint_node_index]
            @ skin.inverse_bind_matrices[joint_slot]
        )
    return skinning_matrices


def compute_model_bounds(
    model: GltfModel, global_node_matrices: list[Matrix4]
) -> tuple[Vector3, Vector3]:
    """Compute world space bounds, taking node transforms into account."""
    bounds_minimum = np.full(3, np.inf, dtype=np.float32)
    bounds_maximum = np.full(3, -np.inf, dtype=np.float32)
    for node_index, mesh_index in model.mesh_indices_by_node.items():
        for primitive in model.meshes[mesh_index].primitives:
            if primitive.joint_weights.any():
                corners_minimum = primitive.positions.min(axis=0)
                corners_maximum = primitive.positions.max(axis=0)
            else:
                local_minimum = primitive.positions.min(axis=0)
                local_maximum = primitive.positions.max(axis=0)
                corners_minimum = _transform_point(
                    global_node_matrices[node_index], local_minimum
                )
                corners_maximum = _transform_point(
                    global_node_matrices[node_index], local_maximum
                )
            bounds_minimum = np.minimum(bounds_minimum, corners_minimum)
            bounds_maximum = np.maximum(bounds_maximum, corners_maximum)
    if not np.isfinite(bounds_minimum).all():
        bounds_minimum = np.zeros(3, dtype=np.float32)
        bounds_maximum = np.zeros(3, dtype=np.float32)
    return bounds_minimum, bounds_maximum


def _transform_point(matrix: Matrix4, point: Vector3) -> Vector3:
    """Transform a single point by a 4x4 matrix."""
    transformed = matrix @ np.array(
        [point[0], point[1], point[2], 1.0], dtype=np.float32
    )
    return transformed[:3].astype(np.float32)


def _sample_translation(
    channel: AnimationChannelData | None,
    time_seconds: float,
    rest_value: Vector3 | npt.NDArray[np.float32],
) -> Vector3:
    """Sample a translation or scale channel, falling back to the rest value."""
    if channel is None:
        return np.asarray(rest_value, dtype=np.float32)
    return _sample_values(channel, time_seconds).astype(np.float32)


def _sample_rotation(
    channel: AnimationChannelData | None,
    time_seconds: float,
    rest_value: Quaternion,
) -> Quaternion:
    """Sample a rotation channel with quaternion interpolation."""
    if channel is None:
        return rest_value
    return _sample_values(channel, time_seconds).astype(np.float32)


def _sample_values(
    channel: AnimationChannelData, time_seconds: float
) -> npt.NDArray[np.float32]:
    """Sample one channel at the given time using its interpolation mode."""
    times = channel.times
    values = channel.values
    if time_seconds <= float(times[0]):
        return values[0]
    if time_seconds >= float(times[-1]):
        return values[-1]
    upper_index = int(np.searchsorted(times, time_seconds, side="right"))
    lower_index = upper_index - 1
    frame_span = float(times[upper_index] - times[lower_index])
    if frame_span <= 0.0:
        return values[lower_index]
    interpolation_factor = (time_seconds - float(times[lower_index])) / frame_span
    if channel.target_path == TARGET_PATH_ROTATION:
        return slerp_quaternion(
            values[lower_index], values[upper_index], interpolation_factor
        )
    return lerp_vectors(values[lower_index], values[upper_index], interpolation_factor)
