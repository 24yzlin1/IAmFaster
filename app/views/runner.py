"""Render the track and the animated KayKit knight in a shared depth buffer."""

from pathlib import Path
from typing import Final

import moderngl
import numpy
import pygame

from app.models.runner import ACTION_DURATION, LANE_WIDTH, OBSTACLE_STYLES, Hazard, Motion, Obstacle, RunnerModel
from kaykit_previewer.animation import compute_skinning_matrices, evaluate_global_node_matrices
from kaykit_previewer.gltf_loader import AnimationClipData, GltfModel, MeshData, MeshPrimitiveData, load_model
from kaykit_previewer.renderer import ModelRenderResources, Renderer, get_uniform
from kaykit_previewer.transform_math import Matrix4, Vector3, build_look_at_matrix, build_perspective_matrix, identity_matrix

ASSETS: Final[Path] = Path(__file__).resolve().parents[2] / "assets"
WINDOW_SIZE: Final[tuple[int, int]] = (960, 640)
CLIP_NAMES: Final[dict[Motion, str]] = {
    Motion.RUN: "Running_A", Motion.JUMP: "Jump_Full_Short", Motion.SLIDE: "Lie_Idle"
}
type Triple = tuple[float, float, float]

# Dimensions are in track units. The arch must clear the sliding knight across its shoulders.
OBSTACLE_APPEARANCE: Final[dict[Hazard, tuple[str, Triple]]] = {
    Hazard.HURDLE: ("yellow", (1.85, 0.55, 0.60)),
    Hazard.OVERHEAD_BAR: ("blue", (2.36, 2.30, 0.60)),
    Hazard.WALL: ("red", (1.85, 2.90, 0.60)),
}

BOX_VERTEX: Final[str] = """
#version 330
uniform mat4 projection;
uniform vec3 center;
uniform vec3 size;
out float light;
const int corners[36] = int[36](0,1,3,0,3,2,4,6,7,4,7,5,0,4,5,0,5,1,
                                2,3,7,2,7,6,0,2,6,0,6,4,1,5,7,1,7,3);
void main() {
    int corner = corners[gl_VertexID];
    vec3 point = vec3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1) - 0.5;
    gl_Position = projection * vec4(center + point * size, 1);
    light = (gl_VertexID / 6 == 3) ? 1.0 : ((gl_VertexID / 6 < 2) ? 0.82 : 0.65);
}
"""
BOX_FRAGMENT: Final[str] = """
#version 330
uniform vec3 color;
in float light;
out vec4 pixel;
void main() { pixel = vec4(color * light, 1); }
"""


class RunnerView:
    def __init__(self, context: moderngl.Context) -> None:
        self.context: moderngl.Context = context
        self.renderer: Renderer = Renderer(context, WINDOW_SIZE)
        self.character: GltfModel = load_model(
            ASSETS / "KayKit_Adventurers_2.0_FREE/Characters/gltf/Knight.glb"
        )
        self.resources: ModelRenderResources = self.renderer.upload_model(self.character, identity_matrix())
        self.obstacle_resources: dict[tuple[Hazard, int], ModelRenderResources] = {}
        obstacle_directory: Path = ASSETS / "KayKit_Platformer_Pack_1.0_FREE/Assets/gltf"
        kind: Hazard
        color: str
        dimensions: Triple
        for kind, (color, dimensions) in OBSTACLE_APPEARANCE.items():
            style_index: int
            style: str
            for style_index, style in enumerate(OBSTACLE_STYLES[kind]):
                model_path: Path = obstacle_directory / color / f"{style}_{color}.gltf"
                self.obstacle_resources[kind, style_index] = self._upload_obstacle(
                    model_path, dimensions, widen_opening=style in {"arch", "arch_tall"}
                )
        directory: Path = ASSETS / "KayKit_Character_Animations_1.1/Animations/gltf/Rig_Medium"
        self.clips: dict[str, AnimationClipData] = {}
        path: Path
        for path in sorted(directory.glob("*.glb")):
            self.clips.update({clip.name: clip for clip in load_model(path).animations})
        self.projection: Matrix4 = build_perspective_matrix(56, WINDOW_SIZE[0] / WINDOW_SIZE[1], 0.1, 120) @ build_look_at_matrix(
            numpy.array([0, 4.2, 7], dtype=numpy.float32),
            numpy.array([0, 1, -5], dtype=numpy.float32),
            numpy.array([0, 1, 0], dtype=numpy.float32),
        )
        self.program: moderngl.Program = context.program(vertex_shader=BOX_VERTEX, fragment_shader=BOX_FRAGMENT)
        self.box: moderngl.VertexArray = self.renderer.create_vertex_array(self.program, [])
        get_uniform(self.program, "projection").write(self.projection.T.copy().tobytes())

    def _upload_obstacle(self, path: Path, dimensions: Triple, *, widen_opening: bool) -> ModelRenderResources:
        scenery: GltfModel = load_model(path)
        minimum: Vector3 = scenery.bounds_minimum
        maximum: Vector3 = scenery.bounds_maximum
        scale: Vector3 = numpy.divide(dimensions, maximum - minimum, dtype=numpy.float32)
        base_center: Vector3 = (minimum + maximum) * numpy.float32(0.5)
        base_center[1] = minimum[1]
        if widen_opening:
            # Widen the passage while keeping both outer edges within the lane.
            half_width: float = float(maximum[0] - minimum[0]) / 2
            mesh: MeshData
            primitive: MeshPrimitiveData
            for mesh in scenery.meshes:
                for primitive in mesh.primitives:
                    horizontal: Vector3 = primitive.positions[:, 0] - base_center[0]
                    fraction_squared: Vector3 = numpy.square(horizontal / half_width).astype(numpy.float32)
                    primitive.positions[:, 0] += 0.24 * horizontal * (1 - fraction_squared)
                    primitive.normals[:, 0] /= 1 + 0.24 * (1 - 3 * fraction_squared)
        transform: Matrix4 = identity_matrix()
        transform[:3, :3] = numpy.diag(scale)
        transform[:3, 3] = -base_center * scale
        return self.renderer.upload_model(scenery, transform)

    def _box(self, center: Triple, size: Triple, color: Triple) -> None:
        get_uniform(self.program, "center").value = center
        get_uniform(self.program, "size").value = size
        get_uniform(self.program, "color").value = color
        self.box.render(moderngl.TRIANGLES, vertices=36)

    def render(self, model: RunnerModel) -> None:
        self.context.clear(0.54, 0.75, 0.84, 1.0)
        self.context.disable(moderngl.CULL_FACE)
        self._box((0, -0.15, -43), (7.2, 0.3, 100), (0.24, 0.32, 0.36))
        boundary: float
        for boundary in (-3.6, -1.2, 1.2, 3.6):
            marker: int
            for marker in range(-1, 17):
                self._box((boundary, 0.01, model.distance % 5 - marker * 5), (0.06, 0.02, 2), (0.85, 0.84, 0.65))
        self.context.enable(moderngl.CULL_FACE)
        obstacle: Obstacle
        for obstacle in model.obstacles:
            location: Matrix4 = identity_matrix()
            location[0, 3] = obstacle.lane * LANE_WIDTH
            location[2, 3] = model.distance - obstacle.distance
            resources: ModelRenderResources = self.obstacle_resources[obstacle.kind, obstacle.style_index]
            self.renderer.render_model(resources, self.projection @ location)
        clip: AnimationClipData = self.clips[CLIP_NAMES[model.motion]]
        phase: float = model.motion_time % clip.duration
        if model.motion is not Motion.RUN:
            phase = clip.duration * model.motion_time / ACTION_DURATION
        pose: list[Matrix4] = evaluate_global_node_matrices(self.character, clip, phase)
        placement: Matrix4 = identity_matrix()
        placement[0, 0] = placement[2, 2] = -1.0
        placement[0, 3] = model.lateral_position
        placement[1, 3] = model.height
        if model.motion is Motion.SLIDE:
            # Center the feet-first pose on the lane and rest its armor on the track.
            placement[0, 3] += 0.29
            placement[1, 3] += 0.14
            placement[2, 3] -= 0.62
        self.renderer.render_model(self.resources, self.projection, compute_skinning_matrices(self.character, pose, placement))
        lines: list[str] = [f"Distance  {int(model.distance)} m"]
        if model.crashed:
            lines.append("Run over. Press R or make a fist to restart.")
        else:
            lines.append("Left / Right: change lane    Up: jump    Down: slide")
        self.renderer.draw_text_overlay(lines)
        pygame.display.flip()
