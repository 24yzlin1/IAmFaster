"""Linear algebra helpers for transforms and quaternions.

Quaternions use the glTF ordering ``(x, y, z, w)``.  Matrices are 4x4
row-major arrays stored with the column-major convention of glTF already
transposed, so plain matrix products behave as expected.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

Matrix4 = npt.NDArray[np.float32]
Vector3 = npt.NDArray[np.float32]
Quaternion = npt.NDArray[np.float32]


def identity_matrix() -> Matrix4:
    """Return a new 4x4 identity matrix."""
    return np.identity(4, dtype=np.float32)


def translation_matrix(translation: Vector3) -> Matrix4:
    """Return a translation matrix for the given vector."""
    matrix = identity_matrix()
    matrix[0, 3] = translation[0]
    matrix[1, 3] = translation[1]
    matrix[2, 3] = translation[2]
    return matrix


def quaternion_to_rotation_matrix(quaternion: Quaternion) -> Matrix4:
    """Convert a quaternion into a 4x4 rotation matrix."""
    position_x, position_y, position_z, position_w = (
        float(value) for value in quaternion
    )
    squared_x = position_x * position_x
    squared_y = position_y * position_y
    squared_z = position_z * position_z
    matrix = identity_matrix()
    matrix[0, 0] = 1.0 - 2.0 * (squared_y + squared_z)
    matrix[0, 1] = 2.0 * (position_x * position_y - position_z * position_w)
    matrix[0, 2] = 2.0 * (position_x * position_z + position_y * position_w)
    matrix[1, 0] = 2.0 * (position_x * position_y + position_z * position_w)
    matrix[1, 1] = 1.0 - 2.0 * (squared_x + squared_z)
    matrix[1, 2] = 2.0 * (position_y * position_z - position_x * position_w)
    matrix[2, 0] = 2.0 * (position_x * position_z - position_y * position_w)
    matrix[2, 1] = 2.0 * (position_y * position_z + position_x * position_w)
    matrix[2, 2] = 1.0 - 2.0 * (squared_x + squared_y)
    return matrix


def compose_transform_matrix(
    translation: Vector3, rotation: Quaternion, scale: Vector3
) -> Matrix4:
    """Compose a local transform matrix from translation, rotation, and scale."""
    matrix = quaternion_to_rotation_matrix(rotation)
    matrix[:3, 0] *= scale[0]
    matrix[:3, 1] *= scale[1]
    matrix[:3, 2] *= scale[2]
    matrix[0, 3] = translation[0]
    matrix[1, 3] = translation[1]
    matrix[2, 3] = translation[2]
    return matrix


def slerp_quaternion(
    first: Quaternion, second: Quaternion, factor: float
) -> Quaternion:
    """Interpolate between two unit quaternions with spherical linear interpolation."""
    second_adjusted = second
    dot_product = float(np.dot(first, second_adjusted))
    if dot_product < 0.0:
        second_adjusted = -second_adjusted
        dot_product = -dot_product
    if dot_product > 0.9995:
        blended = first + np.float32(factor) * (second_adjusted - first)
        return (blended / np.float32(np.linalg.norm(blended))).astype(np.float32)
    clamped_product = min(1.0, max(-1.0, dot_product))
    angle_between = math.acos(clamped_product)
    sine_angle = math.sin(angle_between)
    weight_first = np.float32(math.sin((1.0 - factor) * angle_between) / sine_angle)
    weight_second = np.float32(math.sin(factor * angle_between) / sine_angle)
    return (first * weight_first + second_adjusted * weight_second).astype(np.float32)


def lerp_vectors(
    first: npt.NDArray[np.float32], second: npt.NDArray[np.float32], factor: float
) -> npt.NDArray[np.float32]:
    """Interpolate linearly between two vectors or matrices of equal shape."""
    return (first + np.float32(factor) * (second - first)).astype(np.float32)


def build_perspective_matrix(
    field_of_view_y_degrees: float,
    aspect_ratio: float,
    near_plane_distance: float,
    far_plane_distance: float,
) -> Matrix4:
    """Build a perspective projection matrix."""
    tangent_half_angle = math.tan(math.radians(field_of_view_y_degrees) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = 1.0 / (aspect_ratio * tangent_half_angle)
    matrix[1, 1] = 1.0 / tangent_half_angle
    matrix[2, 2] = (far_plane_distance + near_plane_distance) / (
        near_plane_distance - far_plane_distance
    )
    matrix[2, 3] = (2.0 * far_plane_distance * near_plane_distance) / (
        near_plane_distance - far_plane_distance
    )
    matrix[3, 2] = -1.0
    return matrix


def build_look_at_matrix(
    eye_position: Vector3, target_position: Vector3, world_up: Vector3
) -> Matrix4:
    """Build a right-handed view matrix looking from the eye toward the target."""
    forward_direction = target_position - eye_position
    forward_direction = forward_direction / float(np.linalg.norm(forward_direction))
    right_direction = np.cross(forward_direction, world_up)
    right_length = float(np.linalg.norm(right_direction))
    if right_length < 1e-6:
        right_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right_direction = right_direction / right_length
    up_direction = np.cross(right_direction, forward_direction)
    matrix = identity_matrix()
    matrix[0, :3] = right_direction
    matrix[1, :3] = up_direction
    matrix[2, :3] = -forward_direction
    matrix[0, 3] = -float(np.dot(right_direction, eye_position))
    matrix[1, 3] = -float(np.dot(up_direction, eye_position))
    matrix[2, 3] = float(np.dot(forward_direction, eye_position))
    return matrix
