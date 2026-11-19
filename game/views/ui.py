"""2D overlay user interface drawn directly on the OpenGL window.

The window runs a core-profile OpenGL context, so 2D elements are drawn
as textured quads under an orthographic pixel projection.  Text is
rendered with pygame fonts into surfaces uploaded as textures each call,
which is fast enough for menu-scale interfaces.
"""

from typing import Final

import moderngl
import numpy
import pygame

from game.model_previewer.renderer import get_uniform

UI_VERTEX_SHADER: Final[str] = """
#version 330
in vec2 in_position;
in vec2 in_texture_coordinates;
uniform mat4 u_projection;
out vec2 v_texture_coordinates;
void main() {
    v_texture_coordinates = in_texture_coordinates;
    gl_Position = u_projection * vec4(in_position, 0.0, 1.0);
}
"""
UI_FRAGMENT_SHADER: Final[str] = """
#version 330
uniform sampler2D u_texture;
uniform vec4 u_tint;
in vec2 v_texture_coordinates;
out vec4 pixel;
void main() { pixel = texture(u_texture, v_texture_coordinates) * u_tint; }
"""

type Rect = tuple[int, int, int, int]  # x, y, width, height in pixels, origin top-left
type Color = tuple[int, int, int]
type Alignment = str  # "left" or "center" within the rect

DEFAULT_TEXT_COLOR: Final[Color] = (230, 228, 235)
HIGHLIGHT_TEXT_COLOR: Final[Color] = (255, 200, 80)


class UiLayer:
    """Draws translucent panels and text on top of the 3D scene."""

    def __init__(
        self, context: moderngl.Context, window_size: tuple[int, int],
        font_names: str = "jetbrainsmono",
    ) -> None:
        self._context: moderngl.Context = context
        self._window_size: tuple[int, int] = window_size
        self._font_names: str = font_names
        self._program: moderngl.Program = context.program(
            vertex_shader=UI_VERTEX_SHADER, fragment_shader=UI_FRAGMENT_SHADER
        )
        # A 1x1 white texture lets draw_panel tint arbitrary solid rectangles.
        self._white_texture: moderngl.Texture = context.texture((1, 1), 4)
        self._white_texture.write(b"\xff\xff\xff\xff")
        self._fonts: dict[int, pygame.font.Font] = {}
        width: int
        height: int
        width, height = window_size
        projection: tuple[tuple[float, float, float, float], ...] = (
            (2.0 / width, 0.0, 0.0, -1.0),
            (0.0, -2.0 / height, 0.0, 1.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        self._projection_uniform: moderngl.Uniform = get_uniform(self._program, "u_projection")
        self._projection_uniform.write(
            numpy.asarray(projection, dtype=numpy.float32).transpose().tobytes()
        )

    def draw_panel(self, rect: Rect, color: Color, alpha: float = 1.0) -> None:
        """Fill a rectangle with a solid color at the given opacity."""
        red, green, blue = color
        self._draw_quad(rect, self._white_texture, (red / 255, green / 255, blue / 255, alpha))

    def draw_text(
        self,
        text: str,
        rect: Rect,
        *,
        font_size: int = 28,
        color: Color = DEFAULT_TEXT_COLOR,
        align: Alignment = "center",
    ) -> None:
        """Render one text line inside the rectangle."""
        font: pygame.font.Font = self._font(font_size)
        text_surface: pygame.Surface = font.render(text, True, color, None)
        left, top, width, height = rect
        surface_width: int = text_surface.get_width()
        x: int
        if align == "left":
            x = left
        elif align == "center":
            x = left + (width - surface_width) // 2
        else:
            raise ValueError(f"unknown text alignment: {align!r}")
        y: int = top + (height - text_surface.get_height()) // 2
        texture: moderngl.Texture = self._upload_surface(text_surface)
        self._draw_quad(
            (x, y, surface_width, text_surface.get_height()), texture, (1.0, 1.0, 1.0, 1.0)
        )
        texture.release()

    def _font(self, size: int) -> pygame.font.Font:
        """Return a cached system font of the requested size."""
        font: pygame.font.Font | None = self._fonts.get(size)
        if font is None:
            font = pygame.font.SysFont(self._font_names, size)
            self._fonts[size] = font
        return font

    def _upload_surface(self, surface: pygame.Surface) -> moderngl.Texture:
        """Upload a pygame surface as an RGBA texture."""
        texture: moderngl.Texture = self._context.texture(
            surface.get_size(), 4, pygame.image.tobytes(surface, "RGBA")
        )
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return texture

    def _draw_quad(self, rect: Rect, texture: moderngl.Texture, tint: tuple[float, float, float, float]) -> None:
        """Draw one textured quad covering the pixel rectangle."""
        left, top, width, height = rect
        right: float = float(left + width)
        bottom: float = float(top + height)
        vertices: bytes = numpy.asarray(
            [
                left, top, 0.0, 0.0,
                right, top, 1.0, 0.0,
                left, bottom, 0.0, 1.0,
                right, bottom, 1.0, 1.0,
            ],
            dtype=numpy.float32,
        ).tobytes()
        buffer: moderngl.Buffer = self._context.buffer(vertices)
        vertex_array: moderngl.VertexArray = self._context.vertex_array(  # pyright: ignore[reportUnknownMemberType]
            self._program, [(buffer, "2f 2f", "in_position", "in_texture_coordinates")]
        )
        tint_uniform: moderngl.Uniform = get_uniform(self._program, "u_tint")
        tint_uniform.value = tint
        texture.use(0)
        self._context.disable(self._context.DEPTH_TEST)
        self._context.disable(self._context.CULL_FACE)
        self._context.enable(self._context.BLEND)
        vertex_array.render(moderngl.TRIANGLE_STRIP)
        self._context.disable(self._context.BLEND)
        self._context.enable(self._context.DEPTH_TEST)
        self._context.enable(self._context.CULL_FACE)
        vertex_array.release()
        buffer.release()
