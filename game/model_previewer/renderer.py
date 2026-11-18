"""OpenGL rendering of glTF models with skeletal skinning.

Uses a core profile context created by moderngl on top of the pygame
window.  Skinning happens in the vertex shader driven by per joint
matrices uploaded each frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import moderngl
import numpy as np
import numpy.typing as npt
import pygame

from game.model_previewer.animation import compute_rest_pose_global_matrices
from game.model_previewer.gltf_loader import GltfModel, MeshPrimitiveData, TextureData
from game.model_previewer.transform_math import Matrix4, identity_matrix

MAXIMUM_JOINT_COUNT = 64
GRID_EXTENT = 10.0
GRID_STEP = 0.5

MESH_VERTEX_SHADER = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
in vec2 in_texture_coordinates;
in ivec4 in_joint_indices;
in vec4 in_joint_weights;

uniform mat4 u_view_projection;
uniform mat4 u_model_matrix;
uniform mat4 u_joint_matrices[64];
uniform bool u_is_skinned;

out vec3 v_world_position;
out vec3 v_world_normal;
out vec2 v_texture_coordinates;

void main() {
    mat4 transform_matrix = u_model_matrix;
    if (u_is_skinned) {
        transform_matrix =
            u_joint_matrices[in_joint_indices.x] * in_joint_weights.x +
            u_joint_matrices[in_joint_indices.y] * in_joint_weights.y +
            u_joint_matrices[in_joint_indices.z] * in_joint_weights.z +
            u_joint_matrices[in_joint_indices.w] * in_joint_weights.w;
    }
    vec4 world_position = transform_matrix * vec4(in_position, 1.0);
    v_world_position = world_position.xyz;
    v_world_normal = normalize(mat3(transform_matrix) * in_normal);
    v_texture_coordinates = in_texture_coordinates;
    gl_Position = u_view_projection * world_position;
}
"""

MESH_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_world_position;
in vec3 v_world_normal;
in vec2 v_texture_coordinates;

uniform sampler2D u_base_color_texture;
uniform bool u_use_texture;
uniform vec4 u_base_color_factor;

out vec4 fragment_color;

void main() {
    vec4 base_color = u_base_color_factor;
    if (u_use_texture) {
        base_color *= texture(u_base_color_texture, v_texture_coordinates);
    }
    vec3 light_direction = normalize(vec3(0.4, 0.9, 0.55));
    float diffuse_light = max(dot(normalize(v_world_normal), light_direction), 0.0);
    float fill_light = max(dot(normalize(v_world_normal), vec3(-0.5, 0.2, -0.6)), 0.0);
    vec3 color = base_color.rgb * (0.30 + 0.70 * diffuse_light + 0.15 * fill_light);
    fragment_color = vec4(color, base_color.a);
}
"""

GRID_VERTEX_SHADER = """
#version 330 core
in vec3 in_position;

uniform mat4 u_view_projection;

void main() {
    gl_Position = u_view_projection * vec4(in_position, 1.0);
}
"""

GRID_FRAGMENT_SHADER = """
#version 330 core
uniform vec3 u_line_color;

out vec4 fragment_color;

void main() {
    fragment_color = vec4(u_line_color, 1.0);
}
"""

TEXT_VERTEX_SHADER = """
#version 330 core
in vec2 in_position;
in vec2 in_texture_coordinates;

uniform mat4 u_projection;

out vec2 v_texture_coordinates;

void main() {
    v_texture_coordinates = in_texture_coordinates;
    gl_Position = u_projection * vec4(in_position, 0.0, 1.0);
}
"""

TEXT_FRAGMENT_SHADER = """
#version 330 core
in vec2 v_texture_coordinates;

uniform sampler2D u_text_texture;

out vec4 fragment_color;

void main() {
    fragment_color = texture(u_text_texture, v_texture_coordinates);
}
"""


def get_uniform(program: moderngl.Program, uniform_name: str) -> moderngl.Uniform:
    """Return the uniform member of a program, validating its kind."""
    member = program[uniform_name]
    if not isinstance(member, moderngl.Uniform):
        raise RuntimeError(f"shader member {uniform_name} is not a uniform")
    return member


def _write_matrix_uniform(
    uniform: moderngl.Uniform, matrix: npt.NDArray[np.float32]
) -> None:
    """Write a row-major matrix into a mat4 uniform in column-major byte order."""
    uniform.write(matrix.transpose().astype("f4").tobytes())


@dataclass
class PrimitiveDrawItem:
    """GPU resources for one renderable mesh primitive."""

    vertex_array: moderngl.VertexArray
    is_skinned: bool
    model_matrix: Matrix4
    base_color_factor: npt.NDArray[np.float32]
    texture: moderngl.Texture | None


@dataclass
class ModelRenderResources:
    """GPU resources for one uploaded model."""

    draw_items: list[PrimitiveDrawItem]
    joint_count: int


class Renderer:
    """Renders models, a ground grid, and a text overlay."""

    def __init__(self, context: moderngl.Context, window_size: tuple[int, int]) -> None:
        self._context = context
        self._window_size = window_size
        self._mesh_program = context.program(
            vertex_shader=MESH_VERTEX_SHADER, fragment_shader=MESH_FRAGMENT_SHADER
        )
        self._grid_program = context.program(
            vertex_shader=GRID_VERTEX_SHADER, fragment_shader=GRID_FRAGMENT_SHADER
        )
        self._text_program = context.program(
            vertex_shader=TEXT_VERTEX_SHADER, fragment_shader=TEXT_FRAGMENT_SHADER
        )
        self._grid_vertex_array = self._build_grid_vertex_array()
        self._wireframe_enabled = False
        context.enable(context.DEPTH_TEST)
        context.enable(context.CULL_FACE)
        self._joint_uniform_input = np.tile(
            identity_matrix(), (MAXIMUM_JOINT_COUNT, 1, 1)
        ).astype(np.float32)

    @property
    def wireframe_enabled(self) -> bool:
        """Whether wireframe rendering is active."""
        return self._wireframe_enabled

    def set_wireframe_enabled(self, enabled: bool) -> None:
        """Toggle wireframe rendering."""
        self._wireframe_enabled = enabled
        self._context.wireframe = enabled

    def upload_model(self, model: GltfModel, offset_matrix: Matrix4) -> ModelRenderResources:
        """Upload a model to the GPU and return its render resources."""
        rest_pose_global_matrices = compute_rest_pose_global_matrices(model)
        draw_items: list[PrimitiveDrawItem] = []
        for node_index, mesh_index in model.mesh_indices_by_node.items():
            node_model_matrix = (offset_matrix @ rest_pose_global_matrices[node_index]).astype(
                np.float32
            )
            for primitive in model.meshes[mesh_index].primitives:
                draw_items.append(
                    self._upload_primitive(model, primitive, node_model_matrix, offset_matrix)
                )
        joint_count = 0
        if model.skin is not None:
            joint_count = len(model.skin.joint_node_indices)
        return ModelRenderResources(draw_items=draw_items, joint_count=joint_count)

    def _upload_primitive(
        self,
        model: GltfModel,
        primitive: MeshPrimitiveData,
        node_model_matrix: Matrix4,
        offset_matrix: Matrix4,
    ) -> PrimitiveDrawItem:
        """Upload one primitive and create its vertex array."""
        position_buffer = self._context.buffer(primitive.positions.astype("f4").tobytes())
        normal_buffer = self._context.buffer(primitive.normals.astype("f4").tobytes())
        texture_coordinate_buffer = self._context.buffer(
            primitive.texture_coordinates.astype("f4").tobytes()
        )
        joint_index_buffer = self._context.buffer(primitive.joint_indices.astype("i4").tobytes())
        joint_weight_buffer = self._context.buffer(primitive.joint_weights.astype("f4").tobytes())
        index_buffer = self._context.buffer(primitive.indices.astype("u4").tobytes())
        vertex_array = self.create_vertex_array(
            self._mesh_program,
            [
                (position_buffer, "3f", "in_position"),
                (normal_buffer, "3f", "in_normal"),
                (texture_coordinate_buffer, "2f", "in_texture_coordinates"),
                (joint_index_buffer, "4i", "in_joint_indices"),
                (joint_weight_buffer, "4f", "in_joint_weights"),
            ],
            index_buffer,
        )
        is_skinned = bool(primitive.joint_weights.any())
        material_index = primitive.material_index
        base_color_factor = (
            model.materials[material_index].base_color_factor
            if material_index is not None
            else np.ones(4, dtype=np.float32)
        )
        texture: moderngl.Texture | None = None
        texture_index = (
            model.materials[material_index].base_color_texture_index
            if material_index is not None
            else None
        )
        if texture_index is not None and texture_index < len(model.textures):
            texture = self._upload_texture(model.textures[texture_index])
        return PrimitiveDrawItem(
            vertex_array=vertex_array,
            is_skinned=is_skinned,
            model_matrix=node_model_matrix if not is_skinned else offset_matrix,
            base_color_factor=base_color_factor,
            texture=texture,
        )

    def _upload_texture(self, texture_data: TextureData) -> moderngl.Texture:
        """Upload decoded texture pixels into an OpenGL texture."""
        texture = self._context.texture(texture_data.size, 4, texture_data.pixel_bytes)
        texture.build_mipmaps()
        texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        return texture

    def create_vertex_array(
        self,
        program: moderngl.Program,
        content: Sequence[object],
        index_buffer: moderngl.Buffer | None = None,
    ) -> moderngl.VertexArray:
        """Create a vertex array, wrapping the partially typed moderngl API."""
        vertex_array: moderngl.VertexArray = self._context.vertex_array(  # pyright: ignore[reportUnknownMemberType]
            program, content, index_buffer
        )
        return vertex_array

    def render_grid(self, view_projection: Matrix4) -> None:
        """Draw the ground grid."""
        _write_matrix_uniform(
            get_uniform(self._grid_program, "u_view_projection"), view_projection
        )
        get_uniform(self._grid_program, "u_line_color").value = (0.32, 0.34, 0.38)
        self._grid_vertex_array.render(moderngl.LINES)

    def render_model(
        self,
        resources: ModelRenderResources,
        view_projection: Matrix4,
        skinning_matrices: npt.NDArray[np.float32] | None = None,
    ) -> None:
        """Draw every primitive of the model."""
        _write_matrix_uniform(
            get_uniform(self._mesh_program, "u_view_projection"), view_projection
        )
        if resources.joint_count:
            if skinning_matrices is None:
                raise ValueError("skinning matrices are required for a skinned model")
            self._joint_uniform_input[: resources.joint_count] = skinning_matrices
            get_uniform(self._mesh_program, "u_joint_matrices").write(
                self._joint_uniform_input.transpose(0, 2, 1).astype("f4").tobytes()
            )
        model_matrix_uniform = get_uniform(self._mesh_program, "u_model_matrix")
        skinned_uniform = get_uniform(self._mesh_program, "u_is_skinned")
        base_color_uniform = get_uniform(self._mesh_program, "u_base_color_factor")
        textured_uniform = get_uniform(self._mesh_program, "u_use_texture")
        for draw_item in resources.draw_items:
            _write_matrix_uniform(model_matrix_uniform, draw_item.model_matrix)
            skinned_uniform.value = draw_item.is_skinned
            base_color_uniform.value = tuple(
                float(channel) for channel in draw_item.base_color_factor
            )
            if draw_item.texture is not None:
                draw_item.texture.use(0)
                textured_uniform.value = True
            else:
                textured_uniform.value = False
            draw_item.vertex_array.render(moderngl.TRIANGLES)

    def draw_text_overlay(self, lines: list[str]) -> None:
        """Render overlay text lines at the top left of the window."""
        if not lines:
            return
        line_height = 20
        padding = 8
        overlay_width = self._window_size[0]
        overlay_height = len(lines) * line_height + 2 * padding
        overlay_surface = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
        overlay_surface.fill((16, 18, 24, 170))
        font = pygame.font.SysFont("consolas", 14)
        for line_index, line_text in enumerate(lines):
            text_surface = font.render(line_text, True, (225, 228, 235, 255), None)
            overlay_surface.blit(text_surface, (10, padding + line_index * line_height))
        texture = self._context.texture(
            (overlay_width, overlay_height),
            4,
            pygame.image.tobytes(overlay_surface, "RGBA"),
        )
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        quad_vertices = np.asarray(
            [
                0.0, 0.0, 0.0, 0.0,
                float(overlay_width), 0.0, 1.0, 0.0,
                0.0, float(overlay_height), 0.0, 1.0,
                float(overlay_width), float(overlay_height), 1.0, 1.0,
            ],
            dtype=np.float32,
        )
        quad_buffer = self._context.buffer(quad_vertices.tobytes())
        quad_vertex_array = self.create_vertex_array(
            self._text_program,
            [(quad_buffer, "2f 2f", "in_position", "in_texture_coordinates")],
        )
        projection_matrix = np.asarray(
            [
                [2.0 / overlay_width, 0.0, 0.0, -1.0],
                [0.0, -2.0 / self._window_size[1], 0.0, 1.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        _write_matrix_uniform(
            get_uniform(self._text_program, "u_projection"), projection_matrix
        )
        texture.use(0)
        self._context.disable(self._context.DEPTH_TEST)
        self._context.disable(self._context.CULL_FACE)
        self._context.enable(self._context.BLEND)
        quad_vertex_array.render(moderngl.TRIANGLE_STRIP)
        self._context.disable(self._context.BLEND)
        self._context.enable(self._context.DEPTH_TEST)
        self._context.enable(self._context.CULL_FACE)
        quad_vertex_array.release()
        quad_buffer.release()
        texture.release()

    def _build_grid_vertex_array(self) -> moderngl.VertexArray:
        """Create the ground grid line geometry."""
        line_positions: list[float] = []
        line_count = int(GRID_EXTENT / GRID_STEP)
        for line_index in range(-line_count, line_count + 1):
            coordinate = line_index * GRID_STEP
            line_positions.extend([coordinate, 0.0, -GRID_EXTENT, coordinate, 0.0, GRID_EXTENT])
            line_positions.extend([-GRID_EXTENT, 0.0, coordinate, GRID_EXTENT, 0.0, coordinate])
        vertex_buffer = self._context.buffer(
            np.asarray(line_positions, dtype=np.float32).tobytes()
        )
        return self.create_vertex_array(
            self._grid_program, [(vertex_buffer, "3f", "in_position")]
        )
