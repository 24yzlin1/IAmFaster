"""Render the track and the animated KayKit knight in a shared depth buffer."""

from pathlib import Path
from typing import Final

import moderngl
import numpy
import pygame

from app.models.runner import ACTION_DURATION, LANE_WIDTH, Hazard, Motion, Obstacle, RunnerModel
from kaykit_previewer.animation import compute_skinning_matrices, evaluate_global_node_matrices
from kaykit_previewer.gltf_loader import AnimationClipData, GltfModel, load_model
from kaykit_previewer.renderer import ModelRenderResources, Renderer, get_uniform
from kaykit_previewer.transform_math import Matrix4, build_look_at_matrix, build_perspective_matrix, identity_matrix

ASSETS: Final[Path] = Path(__file__).resolve().parents[2] / "assets"
WINDOW_SIZE: Final[tuple[int, int]] = (960, 640)
CLIP_NAMES: Final[dict[Motion, str]] = {
    Motion.RUN: "Running_A", Motion.JUMP: "Jump_Full_Short", Motion.SLIDE: "Lie_Idle"
}
# Leave clearance above the grounded knight's helmet throughout the slide.
BAR_CLEARANCE: Final[float] = 1.42
BAR_THICKNESS: Final[float] = 0.45

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
type Triple = tuple[float, float, float]


class RunnerView:
    def __init__(self, context: moderngl.Context) -> None:
        self.context: moderngl.Context = context
        self.renderer: Renderer = Renderer(context, WINDOW_SIZE)
        self.character: GltfModel = load_model(
            ASSETS / "KayKit_Adventurers_2.0_FREE/Characters/gltf/Knight.glb"
        )
        self.resources: ModelRenderResources = self.renderer.upload_model(self.character, identity_matrix())
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
        obstacle: Obstacle
        for obstacle in model.obstacles:
            lateral: float = obstacle.lane * LANE_WIDTH
            depth: float = model.distance - obstacle.distance
            if obstacle.kind is Hazard.WALL:
                self._box((lateral, 1.45, depth), (1.85, 2.9, 0.6), (0.58, 0.28, 0.24))
            elif obstacle.kind is Hazard.HURDLE:
                self._box((lateral, 0.275, depth), (1.85, 0.55, 0.6), (1.0, 0.62, 0.17))
            else:
                self._box((lateral, BAR_CLEARANCE + BAR_THICKNESS / 2, depth),
                          (2.1, BAR_THICKNESS, 0.6), (0.25, 0.75, 0.86))
                side: int
                for side in (-1, 1):
                    self._box((lateral + side, BAR_CLEARANCE / 2, depth),
                              (0.08, BAR_CLEARANCE, 0.3), (0.34, 0.48, 0.5))
        self.context.enable(moderngl.CULL_FACE)
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
