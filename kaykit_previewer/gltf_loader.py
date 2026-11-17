"""Parser for glTF 2.0 scenes, GLB binary containers, skins, and animations.

Only the subset of the glTF 2.0 specification used by the KayKit asset
packs is supported: static meshes, skinned meshes, embedded and external
textures, and animation clips with STEP or LINEAR sampler interpolation.
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pygame

from kaykit_previewer.transform_math import (
    Matrix4,
    Quaternion,
    Vector3,
    compose_transform_matrix,
)

COMPONENT_TYPE_INT8 = 5120
COMPONENT_TYPE_UNSIGNED_BYTE = 5121
COMPONENT_TYPE_INT16 = 5122
COMPONENT_TYPE_UNSIGNED_SHORT = 5123
COMPONENT_TYPE_UNSIGNED_INT = 5125
COMPONENT_TYPE_FLOAT = 5126

COMPONENT_DTYPES: dict[int, np.dtype[Any]] = {
    COMPONENT_TYPE_INT8: np.dtype(np.int8),
    COMPONENT_TYPE_UNSIGNED_BYTE: np.dtype(np.uint8),
    COMPONENT_TYPE_INT16: np.dtype(np.int16),
    COMPONENT_TYPE_UNSIGNED_SHORT: np.dtype(np.uint16),
    COMPONENT_TYPE_UNSIGNED_INT: np.dtype(np.uint32),
    COMPONENT_TYPE_FLOAT: np.dtype(np.float32),
}

COMPONENT_NORMALIZATION_MAXIMUMS: dict[int, float] = {
    COMPONENT_TYPE_INT8: 127.0,
    COMPONENT_TYPE_UNSIGNED_BYTE: 255.0,
    COMPONENT_TYPE_INT16: 32767.0,
    COMPONENT_TYPE_UNSIGNED_SHORT: 65535.0,
}

GLTF_TYPE_COMPONENT_COUNTS: dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

GLB_MAGIC_NUMBER = 0x46546C67
GLB_VERSION_NUMBER = 2
GLB_JSON_CHUNK_TYPE = 0x4E4F534A
GLB_BINARY_CHUNK_TYPE = 0x004E4942

TARGET_PATH_TRANSLATION = "translation"
TARGET_PATH_ROTATION = "rotation"
TARGET_PATH_SCALE = "scale"

TARGET_PATH_COMPONENT_COUNTS: dict[str, int] = {
    TARGET_PATH_TRANSLATION: 3,
    TARGET_PATH_ROTATION: 4,
    TARGET_PATH_SCALE: 3,
}


@dataclass
class TextureData:
    """Decoded RGBA texture pixels in row-major top-down order."""

    size: tuple[int, int]
    pixel_bytes: bytes


@dataclass
class MaterialData:
    """Material properties relevant for base color rendering."""

    base_color_factor: npt.NDArray[np.float32]
    base_color_texture_index: int | None


@dataclass
class MeshPrimitiveData:
    """One renderable primitive with optional skinning attributes."""

    positions: npt.NDArray[np.float32]
    normals: npt.NDArray[np.float32]
    texture_coordinates: npt.NDArray[np.float32]
    joint_indices: npt.NDArray[np.int32]
    joint_weights: npt.NDArray[np.float32]
    indices: npt.NDArray[np.uint32]
    material_index: int | None


@dataclass
class MeshData:
    """A mesh holding one or more primitives."""

    primitives: list[MeshPrimitiveData]


@dataclass
class NodeData:
    """A scene graph node with a translation-rotation-scale transform."""

    name: str
    translation: Vector3
    rotation: Quaternion
    scale: Vector3
    local_matrix: Matrix4
    mesh_index: int | None
    children: list[int]


@dataclass
class SkinData:
    """A skin binding joints to inverse bind matrices."""

    joint_node_indices: list[int]
    inverse_bind_matrices: npt.NDArray[np.float32]


@dataclass
class AnimationChannelData:
    """A single animation channel targeting a node by name."""

    node_name: str
    target_path: str
    times: npt.NDArray[np.float32]
    values: npt.NDArray[np.float32]


@dataclass
class AnimationClipData:
    """A named animation clip consisting of channels and a duration."""

    name: str
    duration: float
    channels: list[AnimationChannelData]


@dataclass
class GltfModel:
    """Parsed glTF model ready for rendering and animation."""

    rig_name: str
    nodes: list[NodeData]
    root_node_indices: list[int]
    meshes: list[MeshData]
    mesh_indices_by_node: dict[int, int]
    skin: SkinData | None
    animations: list[AnimationClipData]
    materials: list[MaterialData]
    textures: list[TextureData]
    node_indices_by_name: dict[str, int] = field(
        default_factory=lambda: dict[str, int]()
    )
    bounds_minimum: Vector3 = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    bounds_maximum: Vector3 = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )


def load_model(model_path: Path) -> GltfModel:
    """Load a glTF or GLB model from the given path."""
    raw_bytes = model_path.read_bytes()
    if raw_bytes[:4] == b"glTF":
        document, embedded_buffer_data = _parse_glb_container(raw_bytes)
    else:
        document = json.loads(raw_bytes.decode("utf-8"))
        embedded_buffer_data = []
    buffer_data = _load_buffer_data(document, model_path, embedded_buffer_data)

    materials = _load_materials(document)
    textures = _load_textures(document, buffer_data, model_path)
    meshes = [
        _load_mesh(document, buffer_data, mesh_description)
        for mesh_description in document.get("meshes", [])
    ]
    nodes = [
        _load_node(node_description, node_index)
        for node_index, node_description in enumerate(document.get("nodes", []))
    ]
    root_node_indices = _find_root_node_indices(document)
    skin = _load_skin(document, buffer_data)
    animations = _load_animations(document, buffer_data)

    mesh_indices_by_node: dict[int, int] = {}
    for node_index, node_description in enumerate(document.get("nodes", [])):
        if "mesh" in node_description:
            mesh_indices_by_node[node_index] = int(node_description["mesh"])

    node_indices_by_name: dict[str, int] = {}
    for node_index, node in enumerate(nodes):
        node_indices_by_name[node.name] = node_index

    rig_name = nodes[root_node_indices[0]].name if root_node_indices else "scene"

    model = GltfModel(
        rig_name=rig_name,
        nodes=nodes,
        root_node_indices=root_node_indices,
        meshes=meshes,
        mesh_indices_by_node=mesh_indices_by_node,
        skin=skin,
        animations=animations,
        materials=materials,
        textures=textures,
        node_indices_by_name=node_indices_by_name,
    )
    _compute_model_bounds(model)
    return model


def _parse_glb_container(raw_bytes: bytes) -> tuple[dict[str, Any], list[bytes]]:
    """Split a GLB container into its JSON document and binary chunk."""
    if len(raw_bytes) < 12:
        raise ValueError("GLB container is truncated")
    magic_number, version_number, total_length = struct.unpack_from(
        "<III", raw_bytes, 0
    )
    if magic_number != GLB_MAGIC_NUMBER:
        raise ValueError("file is not a GLB container")
    if version_number != GLB_VERSION_NUMBER:
        raise ValueError(f"unsupported GLB container version: {version_number}")
    json_document: dict[str, Any] | None = None
    binary_buffers: list[bytes] = []
    byte_offset = 12
    while byte_offset < total_length:
        chunk_length, chunk_type = struct.unpack_from("<II", raw_bytes, byte_offset)
        byte_offset += 8
        chunk_bytes = raw_bytes[byte_offset : byte_offset + chunk_length]
        byte_offset += chunk_length
        if chunk_type == GLB_JSON_CHUNK_TYPE:
            json_document = json.loads(chunk_bytes.decode("utf-8"))
        elif chunk_type == GLB_BINARY_CHUNK_TYPE:
            binary_buffers.append(chunk_bytes)
    if json_document is None:
        raise ValueError("GLB container has no JSON chunk")
    return json_document, binary_buffers


def _load_buffer_data(
    document: dict[str, Any], model_path: Path, embedded_buffers: list[bytes]
) -> list[bytes]:
    """Resolve every buffer of the document into raw bytes."""
    buffer_data: list[bytes] = []
    for buffer_index, buffer_description in enumerate(document.get("buffers", [])):
        if "uri" not in buffer_description:
            if buffer_index >= len(embedded_buffers):
                raise ValueError(f"buffer {buffer_index} has no binary data")
            buffer_data.append(embedded_buffers[buffer_index])
            continue
        buffer_uri = str(buffer_description["uri"])
        if buffer_uri.startswith("data:"):
            buffer_data.append(_decode_data_uri(buffer_uri))
        else:
            buffer_data.append((model_path.parent / buffer_uri).read_bytes())
    return buffer_data


def _decode_data_uri(data_uri: str) -> bytes:
    """Decode a base64 data URI into raw bytes."""
    encoded_part = data_uri.partition(",")[2]
    return base64.b64decode(encoded_part)


def _read_accessor_bytes(
    document: dict[str, Any], buffer_data: list[bytes], accessor_index: int
) -> npt.NDArray[np.uint8]:
    """Read one accessor as a two dimensional byte array of elements."""
    accessor = document["accessors"][accessor_index]
    if "sparse" in accessor:
        raise ValueError("sparse accessors are not supported")
    component_type = int(accessor["componentType"])
    if component_type not in COMPONENT_DTYPES:
        raise ValueError(f"unsupported accessor component type: {component_type}")
    element_type = accessor["type"]
    if element_type not in GLTF_TYPE_COMPONENT_COUNTS:
        raise ValueError(f"unsupported accessor element type: {element_type}")
    component_dtype = COMPONENT_DTYPES[component_type]
    component_count = GLTF_TYPE_COMPONENT_COUNTS[element_type]
    item_byte_size = component_dtype.itemsize * component_count
    element_count = int(accessor["count"])

    if "bufferView" not in accessor:
        return np.zeros((element_count, item_byte_size), dtype=np.uint8)
    buffer_view = document["bufferViews"][int(accessor["bufferView"])]
    buffer_bytes = buffer_data[int(buffer_view["buffer"])]
    byte_stride = int(buffer_view.get("byteStride") or item_byte_size)
    start_offset = int(buffer_view.get("byteOffset", 0)) + int(
        accessor.get("byteOffset", 0)
    )
    required_length = byte_stride * element_count
    if start_offset + required_length > len(buffer_bytes):
        raise ValueError("accessor data lies outside of its buffer")
    raw_bytes = np.frombuffer(
        buffer_bytes, dtype=np.uint8, count=required_length, offset=start_offset
    ).reshape(element_count, byte_stride)
    return raw_bytes[:, :item_byte_size].copy()


def _read_numeric_accessor(
    document: dict[str, Any], buffer_data: list[bytes], accessor_index: int
) -> npt.NDArray[Any]:
    """Read one accessor as a numeric array preserving its component type."""
    accessor = document["accessors"][accessor_index]
    component_dtype = COMPONENT_DTYPES[int(accessor["componentType"])]
    element_count = int(accessor["count"])
    element_type = accessor["type"]
    component_count = GLTF_TYPE_COMPONENT_COUNTS[element_type]
    element_bytes = _read_accessor_bytes(document, buffer_data, accessor_index)
    values = element_bytes.view(component_dtype)
    return values.reshape(element_count, component_count)


def _read_float_accessor(
    document: dict[str, Any], buffer_data: list[bytes], accessor_index: int
) -> npt.NDArray[np.float32]:
    """Read one accessor and convert it to float32."""
    accessor = document["accessors"][accessor_index]
    component_type = int(accessor["componentType"])
    values = _read_numeric_accessor(document, buffer_data, accessor_index)
    if (
        accessor.get("normalized", False)
        and component_type in COMPONENT_NORMALIZATION_MAXIMUMS
    ):
        values = (
            values.astype(np.float32) / COMPONENT_NORMALIZATION_MAXIMUMS[component_type]
        )
    return values.astype(np.float32)


def _load_materials(document: dict[str, Any]) -> list[MaterialData]:
    """Load material descriptions, falling back to a default material."""
    materials: list[MaterialData] = []
    for material_description in document.get("materials", []):
        metallic_roughness = material_description.get("pbrMetallicRoughness", {})
        base_color_factor = np.asarray(
            metallic_roughness.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        base_color_texture = metallic_roughness.get("baseColorTexture")
        base_color_texture_index = (
            int(base_color_texture["index"]) if base_color_texture is not None else None
        )
        materials.append(
            MaterialData(
                base_color_factor=base_color_factor,
                base_color_texture_index=base_color_texture_index,
            )
        )
    if not materials:
        materials.append(
            MaterialData(
                base_color_factor=np.ones(4, dtype=np.float32),
                base_color_texture_index=None,
            )
        )
    return materials


def _load_textures(
    document: dict[str, Any], buffer_data: list[bytes], model_path: Path
) -> list[TextureData]:
    """Decode every image of the document into RGBA pixel bytes."""
    textures: list[TextureData] = []
    for image_index, image_description in enumerate(document.get("images", [])):
        if "bufferView" in image_description:
            buffer_view = document["bufferViews"][int(image_description["bufferView"])]
            buffer_bytes = buffer_data[int(buffer_view["buffer"])]
            start_offset = int(buffer_view.get("byteOffset", 0))
            byte_length = int(buffer_view["byteLength"])
            image_bytes = buffer_bytes[start_offset : start_offset + byte_length]
            name_hint = image_description.get("name", "image.png")
        elif "uri" in image_description:
            image_uri = str(image_description["uri"])
            if image_uri.startswith("data:"):
                image_bytes = _decode_data_uri(image_uri)
            else:
                image_bytes = (model_path.parent / image_uri).read_bytes()
            name_hint = Path(image_uri).name
        else:
            raise ValueError(f"image {image_index} has neither a buffer view nor a URI")
        surface = pygame.image.load(BytesIO(image_bytes), name_hint)
        textures.append(
            TextureData(
                size=(surface.get_width(), surface.get_height()),
                pixel_bytes=pygame.image.tobytes(surface, "RGBA"),
            )
        )
    return textures


def _load_mesh(
    document: dict[str, Any], buffer_data: list[bytes], mesh_description: dict[str, Any]
) -> MeshData:
    """Load the primitives of one mesh."""
    primitives: list[MeshPrimitiveData] = []
    for primitive_description in mesh_description.get("primitives", []):
        attributes = primitive_description["attributes"]
        positions = _read_float_accessor(
            document, buffer_data, int(attributes["POSITION"])
        )
        if "NORMAL" not in attributes:
            raise ValueError("mesh primitive is missing the NORMAL attribute")
        normals = _read_float_accessor(document, buffer_data, int(attributes["NORMAL"]))
        if "TEXCOORD_0" in attributes:
            texture_coordinates = _read_float_accessor(
                document, buffer_data, int(attributes["TEXCOORD_0"])
            )
        else:
            texture_coordinates = np.zeros((len(positions), 2), dtype=np.float32)
        if "JOINTS_0" in attributes:
            joint_indices = _read_numeric_accessor(
                document, buffer_data, int(attributes["JOINTS_0"])
            ).astype(np.int32)
        else:
            joint_indices = np.zeros((len(positions), 4), dtype=np.int32)
        if "WEIGHTS_0" in attributes:
            joint_weights = _read_float_accessor(
                document, buffer_data, int(attributes["WEIGHTS_0"])
            )
        else:
            joint_weights = np.zeros((len(positions), 4), dtype=np.float32)
        if "indices" in primitive_description:
            indices = (
                _read_numeric_accessor(
                    document, buffer_data, int(primitive_description["indices"])
                )
                .astype(np.uint32)
                .reshape(-1)
            )
        else:
            indices = np.arange(len(positions), dtype=np.uint32)
        material_index: int | None = None
        if "material" in primitive_description:
            material_index = int(primitive_description["material"])
        primitives.append(
            MeshPrimitiveData(
                positions=positions,
                normals=normals,
                texture_coordinates=texture_coordinates,
                joint_indices=joint_indices,
                joint_weights=joint_weights,
                indices=indices,
                material_index=material_index,
            )
        )
    return MeshData(primitives=primitives)


def _load_node(node_description: dict[str, Any], node_index: int) -> NodeData:
    """Load one scene graph node with its local transform."""
    node_name = str(node_description.get("name") or f"node_{node_index}")
    if "matrix" in node_description:
        raise ValueError(
            f"node {node_name} uses a matrix transform, which is not supported"
        )
    translation = np.asarray(
        node_description.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32
    )
    rotation = np.asarray(
        node_description.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32
    )
    scale = np.asarray(node_description.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
    mesh_index: int | None = None
    if "mesh" in node_description:
        mesh_index = int(node_description["mesh"])
    return NodeData(
        name=node_name,
        translation=translation,
        rotation=rotation,
        scale=scale,
        local_matrix=compose_transform_matrix(translation, rotation, scale),
        mesh_index=mesh_index,
        children=[
            int(child_index) for child_index in node_description.get("children", [])
        ],
    )


def _find_root_node_indices(document: dict[str, Any]) -> list[int]:
    """Determine the root node indices from the default scene."""
    scenes = document.get("scenes", [])
    if not scenes:
        raise ValueError("document has no scenes")
    default_scene = scenes[int(document.get("scene", 0))]
    return [int(node_index) for node_index in default_scene["nodes"]]


def _load_skin(document: dict[str, Any], buffer_data: list[bytes]) -> SkinData | None:
    """Load the single skin of the document if present."""
    skins = document.get("skins", [])
    if not skins:
        return None
    if len(skins) > 1:
        raise ValueError("models with more than one skin are not supported")
    skin_description = skins[0]
    inverse_bind_matrices = _read_float_accessor(
        document, buffer_data, int(skin_description["inverseBindMatrices"])
    )
    joint_count = len(skin_description["joints"])
    return SkinData(
        joint_node_indices=[
            int(joint_index) for joint_index in skin_description["joints"]
        ],
        inverse_bind_matrices=inverse_bind_matrices.reshape(joint_count, 4, 4)
        .transpose(0, 2, 1)
        .astype(np.float32),
    )


def _load_animations(
    document: dict[str, Any], buffer_data: list[bytes]
) -> list[AnimationClipData]:
    """Load every animation clip of the document."""
    animation_clips: list[AnimationClipData] = []
    for animation_description in document.get("animations", []):
        clip_channels: list[AnimationChannelData] = []
        clip_duration = 0.0
        for channel_description in animation_description.get("channels", []):
            sampler_description = animation_description["samplers"][
                int(channel_description["sampler"])
            ]
            target = channel_description["target"]
            target_path = str(target["path"])
            if target_path not in (
                TARGET_PATH_TRANSLATION,
                TARGET_PATH_ROTATION,
                TARGET_PATH_SCALE,
            ):
                raise ValueError(f"unsupported animation target path: {target_path}")
            interpolation = sampler_description.get("interpolation", "LINEAR")
            if interpolation not in ("LINEAR", "STEP"):
                raise ValueError(
                    f"unsupported animation interpolation: {interpolation}"
                )
            times = _read_float_accessor(
                document, buffer_data, int(sampler_description["input"])
            ).reshape(-1)
            values = _read_float_accessor(
                document, buffer_data, int(sampler_description["output"])
            ).reshape(-1, TARGET_PATH_COMPONENT_COUNTS[target_path])
            node_index = int(target["node"])
            node_name = str(
                document["nodes"][node_index].get("name") or f"node_{node_index}"
            )
            if len(times) > 0:
                clip_duration = max(clip_duration, float(times[-1]))
            clip_channels.append(
                AnimationChannelData(
                    node_name=node_name,
                    target_path=target_path,
                    times=times,
                    values=values,
                )
            )
        animation_name = str(
            animation_description.get("name") or f"animation_{len(animation_clips)}"
        )
        animation_clips.append(
            AnimationClipData(
                name=animation_name,
                duration=clip_duration,
                channels=clip_channels,
            )
        )
    return animation_clips


def _compute_model_bounds(model: GltfModel) -> None:
    """Compute axis aligned bounds over all primitive positions."""
    bounds_minimum = np.full(3, np.inf, dtype=np.float32)
    bounds_maximum = np.full(3, -np.inf, dtype=np.float32)
    for mesh in model.meshes:
        for primitive in mesh.primitives:
            primitive_minimum = primitive.positions.min(axis=0)
            primitive_maximum = primitive.positions.max(axis=0)
            bounds_minimum = np.minimum(bounds_minimum, primitive_minimum)
            bounds_maximum = np.maximum(bounds_maximum, primitive_maximum)
    if not np.isfinite(bounds_minimum).all():
        bounds_minimum = np.zeros(3, dtype=np.float32)
        bounds_maximum = np.zeros(3, dtype=np.float32)
    model.bounds_minimum = bounds_minimum
    model.bounds_maximum = bounds_maximum
