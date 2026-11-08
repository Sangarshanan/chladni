"""Interactive Shader Format (ISF) support for chladni."""

import json
from pathlib import Path
import re
import time
import numpy as np

import moderngl

from .gfx import draw_geometry, set_blend
from .signals import evaluate
from .visuals import Geometry

from .shader_config import (
    SHADER_PRESETS as _PRESETS,
    _HEADER_REGEX,
    _VERTEX_SHADER,
    _FRAGMENT_PREAMBLE,
    _SHADER_SUFFIXES,
)


def _source_path(source: str) -> Path | None:
    """Return the shader file `source` refers to, or None for presets and raw GLSL."""
    if source.strip().lower() in _PRESETS or "\n" in source or len(source) > 1024:
        return None
    path = Path(source).expanduser()
    try:
        if path.is_file():
            return path.resolve()
    except OSError:
        return None
    if path.suffix.lower() in _SHADER_SUFFIXES:
        raise FileNotFoundError(f"ISF shader file not found: {source}")
    return None


def parse_isf_source(source_or_name_or_path: str) -> tuple[dict, str]:
    """Parse an ISF file path, preset name, or raw shader string into (metadata, glsl_body)."""
    key = source_or_name_or_path.strip().lower()
    path = _source_path(source_or_name_or_path)
    if key in _PRESETS:
        raw_code = _PRESETS[key]
    elif path is not None:
        raw_code = path.read_text()
    else:
        raw_code = source_or_name_or_path

    # Extract JSON header
    match = _HEADER_REGEX.search(raw_code)
    if match:
        header_json = "{" + match.group(1).strip() + "}"
        try:
            metadata = json.loads(header_json)
        except Exception as e:
            print(f"[chladni] Warning: Failed to parse ISF JSON header: {e}")
            metadata = {}
        body = raw_code[match.end() :].strip()
    else:
        metadata = {}
        body = raw_code.strip()

    return metadata, body


_MAIN_RE = re.compile(r"\bvoid\s+main\s*\(\s*(?:void)?\s*\)")
_ALPHA_RE = re.compile(r"\balpha\b")


def _build_full_fragment_shader(metadata: dict, body: str) -> str:
    """Generate modern GLSL 330 fragment shader with ISF uniforms and declared inputs.

    Shaders that don't read `alpha` themselves get their output scaled by it, so
    `node.alpha` acts as a layer opacity for any ISF file.
    """
    lines = [_FRAGMENT_PREAMBLE]

    # Declare uniform for each input
    inputs = metadata.get("INPUTS", [])
    for inp in inputs:
        name = inp.get("NAME")
        if not name:
            continue
        itype = inp.get("TYPE", "float").lower()
        if itype == "float":
            lines.append(f"uniform float {name};")
            if inp.get("INTEGRATE"):
                lines.append(f"uniform float {name}_phase;")
        elif itype in ("long", "int", "integer"):
            lines.append(f"uniform int {name};")
        elif itype == "bool":
            lines.append(f"uniform bool {name};")
        elif itype == "color":
            lines.append(f"uniform vec4 {name};")
        elif itype == "point2d":
            lines.append(f"uniform vec2 {name};")
        elif itype in ("image", "texture"):
            lines.append(f"uniform sampler2D {name};")

    if _MAIN_RE.search(body) and not _ALPHA_RE.search(body):
        body = _MAIN_RE.sub("void isf_user_main()", body, count=1)
        body += "\nvoid main() { isf_user_main(); gl_FragColor *= alpha; }\n"

    lines.append("\n// User ISF Shader Body\n")
    lines.append(body)
    return "\n".join(lines)


class ISFNode(Geometry):
    """Base class for ISF shader nodes.

    blend_mode ("replace", "alpha", "add" or "screen") sets how the node composites
    over layers below it, and `alpha` scales its opacity.
    """

    def __init__(
        self,
        source_or_path="plasma",
        x=-1.0,
        y=-1.0,
        width=2.0,
        height=2.0,
        alpha=1.0,
        blend_mode="replace",
        clamp_inputs=True,
        **kwargs,
    ):
        super().__init__()
        metadata, body = parse_isf_source(source_or_path)

        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "_body", body)
        object.__setattr__(self, "_inputs_meta", {})
        object.__setattr__(self, "_inputs_values", {})

        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.alpha = alpha
        self.blend_mode = blend_mode
        self.clamp_inputs = clamp_inputs

        source_path = _source_path(source_or_path)
        self.watch_paths = (str(source_path),) if source_path else ()

        # Parse declared inputs
        inputs = metadata.get("INPUTS", [])
        self._phase_wrap = {}
        self._phases = {}
        for inp in inputs:
            name = inp.get("NAME")
            if not name:
                continue
            itype = inp.get("TYPE", "float").lower()
            default = inp.get("DEFAULT")
            if default is None:
                if itype == "float":
                    default = 0.0
                elif itype in ("long", "int", "integer"):
                    default = 0
                elif itype == "bool":
                    default = False
                elif itype == "color":
                    default = (1.0, 1.0, 1.0, 1.0)
                elif itype == "point2d":
                    default = (0.5, 0.5)
            self._inputs_meta[name] = inp
            self._inputs_values[name] = default
            if itype == "float" and inp.get("INTEGRATE"):
                self._phase_wrap[name] = float(inp.get("WRAP") or 0.0)
                self._phases[name] = 0.0

        # Apply any initial parameter kwargs passed to constructor
        for k, v in kwargs.items():
            if k in self._inputs_meta:
                self._inputs_values[k] = v
            else:
                setattr(self, k, v)

        self._time = 0.0
        self._dt = 0.016
        self._frame_index = 0

        self._prog = None
        self._vbo = None
        self._vao = None
        self._bounds = None
        self._error = None

    def __setattr__(self, name, value):
        if hasattr(self, "_inputs_meta") and name in self._inputs_meta:
            self._inputs_values[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if "_inputs_values" in self.__dict__ and name in self._inputs_values:
            return self._inputs_values[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def _float_input(self, name):
        """Evaluate a float input, limited to its declared range (widened to include DEFAULT)."""
        value = evaluate(self._inputs_values.get(name))
        if value is None:
            return None
        value = float(value)
        meta = self._inputs_meta.get(name, {})
        lo, hi = meta.get("MIN"), meta.get("MAX")
        if self.clamp_inputs and lo is not None and hi is not None:
            default = meta.get("DEFAULT")
            if isinstance(default, (int, float)):
                lo, hi = min(lo, default), max(hi, default)
            value = min(max(value, float(lo)), float(hi))
        return value

    def update(self, dt):
        self._time += dt
        self._dt = dt
        self._frame_index += 1
        for name, wrap in self._phase_wrap.items():
            rate = self._float_input(name) or 0.0
            phase = self._phases[name] + rate * dt
            self._phases[name] = phase % wrap if wrap > 0.0 else phase

    def _ensure_pipeline(self, ctx):
        if self._error is not None:
            raise self._error
        if self._prog is None:
            try:
                full_frag = _build_full_fragment_shader(self._metadata, self._body)
                self._prog = ctx.program(
                    vertex_shader=_VERTEX_SHADER, fragment_shader=full_frag
                )
            except Exception as exc:
                self._error = exc
                raise
            self._vbo = ctx.buffer(reserve=24 * 4, dynamic=True)
            self._vao = ctx.vertex_array(
                self._prog, [(self._vbo, "2f 2f", "in_pos", "in_uv")]
            )

        bounds = (
            float(evaluate(self.x)),
            float(evaluate(self.y)),
            float(evaluate(self.width)),
            float(evaluate(self.height)),
        )
        if self._bounds != bounds:
            self._bounds = bounds
            x0, y0, bw, bh = bounds
            x1, y1 = x0 + bw, y0 + bh
            # (0, 0) bottom-left UV to (1, 1) top-right UV
            quad = np.array(
                [
                    x0,
                    y0,
                    0.0,
                    0.0,
                    x1,
                    y0,
                    1.0,
                    0.0,
                    x1,
                    y1,
                    1.0,
                    1.0,
                    x0,
                    y0,
                    0.0,
                    0.0,
                    x1,
                    y1,
                    1.0,
                    1.0,
                    x0,
                    y1,
                    0.0,
                    1.0,
                ],
                dtype="f4",
            )
            self._vbo.write(quad.tobytes())

    def _apply_uniforms(self, w, h):
        prog = self._prog
        if "RENDERSIZE" in prog:
            prog["RENDERSIZE"].value = (float(w), float(h))
        if "TIME" in prog:
            prog["TIME"].value = float(self._time)
        if "TIMEDELTA" in prog:
            prog["TIMEDELTA"].value = float(self._dt)
        if "FRAMEINDEX" in prog:
            prog["FRAMEINDEX"].value = int(self._frame_index)
        if "alpha" in prog:
            prog["alpha"].value = float(evaluate(self.alpha))
        if "DATE" in prog:
            now = time.time()
            lt = time.localtime(now)
            seconds = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec + now % 1.0
            # Shaders commonly take log(DATE.w); keep it finite right after midnight.
            prog["DATE"].value = (
                float(lt.tm_year),
                float(lt.tm_mon),
                float(lt.tm_mday),
                max(seconds, 1.0),
            )
        for name, phase in self._phases.items():
            if f"{name}_phase" in prog:
                prog[f"{name}_phase"].value = float(phase)

        for name, val in self._inputs_values.items():
            if name not in prog:
                continue
            meta = self._inputs_meta.get(name, {})
            itype = meta.get("TYPE", "float").lower()
            if itype == "float":
                res = self._float_input(name)
                if res is not None:
                    prog[name].value = res
                continue
            res = evaluate(val)
            if res is None:
                continue
            if itype in ("long", "int", "integer"):
                prog[name].value = int(res)
            elif itype == "bool":
                prog[name].value = bool(res)
            elif itype == "color":
                if isinstance(res, (tuple, list)):
                    if len(res) == 3:
                        prog[name].value = (
                            float(res[0]),
                            float(res[1]),
                            float(res[2]),
                            1.0,
                        )
                    elif len(res) >= 4:
                        prog[name].value = (
                            float(res[0]),
                            float(res[1]),
                            float(res[2]),
                            float(res[3]),
                        )
            elif itype == "point2d":
                if isinstance(res, (tuple, list)) and len(res) >= 2:
                    prog[name].value = (float(res[0]), float(res[1]))

    def close(self):
        """Release ModernGL resources."""
        for res in (self._vao, self._vbo, self._prog):
            if res is not None:
                try:
                    res.release()
                except Exception:
                    pass
        self._vao = self._vbo = self._prog = None

    def vertices(self):
        return np.zeros((0, 2), dtype="f4"), np.zeros((0, 3), dtype="f4"), "points"


# ISFGenerator


class ISFGenerator(ISFNode):
    """An ISF visual generator (source) that produces procedural imagery."""

    def __init__(self, source="plasma", **kwargs):
        super().__init__(source_or_path=source, **kwargs)

    @classmethod
    def plasma(cls, **kwargs):
        return cls("plasma", **kwargs)

    @classmethod
    def waves(cls, **kwargs):
        return cls("waves", **kwargs)

    @classmethod
    def tunnel(cls, **kwargs):
        return cls("tunnel", **kwargs)

    def render(self, ctx):
        self._ensure_pipeline(ctx)
        w, h = ctx.fbo.size
        self._apply_uniforms(w, h)
        set_blend(ctx, self.blend_mode)
        self._vao.render(moderngl.TRIANGLES)
        set_blend(ctx, None)


# ISFFilter
class ISFFilter(ISFNode):
    """An ISF post-processing filter that transforms an upstream visual or camera feed."""

    def __init__(self, source="glitch", upstream=None, **kwargs):
        super().__init__(source_or_path=source, **kwargs)
        self.upstream = upstream
        self._fbo = None
        self._fbo_tex = None
        self._dummy_tex = None

        # Identify image uniform name (typically 'inputImage')
        self._image_uniform_name = "inputImage"
        for name, meta in self._inputs_meta.items():
            if meta.get("TYPE", "").lower() in ("image", "texture"):
                self._image_uniform_name = name
                break

    @classmethod
    def glitch(cls, **kwargs):
        return cls("glitch", **kwargs)

    @classmethod
    def kaleidoscope(cls, **kwargs):
        return cls("kaleidoscope", **kwargs)

    @classmethod
    def rgb_shift(cls, **kwargs):
        return cls("rgb_shift", **kwargs)

    @classmethod
    def pixelate(cls, **kwargs):
        return cls("pixelate", **kwargs)

    def _get_upstream_nodes(self):
        if self.upstream is not None:
            return (
                [self.upstream]
                if not isinstance(self.upstream, (list, tuple))
                else list(self.upstream)
            )
        if self.inputs:
            return list(self.inputs)
        return []

    def _ensure_fbo(self, ctx, w, h):
        if self._fbo is None or self._fbo_tex.size != (w, h):
            if self._fbo_tex is not None:
                self._fbo_tex.release()
            if self._fbo is not None:
                self._fbo.release()
            self._fbo_tex = ctx.texture((w, h), 4, dtype="f2")
            self._fbo_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._fbo_tex.repeat_x = self._fbo_tex.repeat_y = False
            self._fbo = ctx.framebuffer(color_attachments=[self._fbo_tex])

    def render(self, ctx):
        upstreams = self._get_upstream_nodes()
        prev_fbo = ctx.fbo
        w, h = prev_fbo.size

        self._ensure_pipeline(ctx)
        if upstreams:
            self._ensure_fbo(ctx, w, h)
            self._fbo.use()
            set_blend(ctx, None)
            ctx.clear(0.0, 0.0, 0.0, 0.0)
            for up in upstreams:
                draw_geometry(ctx, up)
            prev_fbo.use()
            self._fbo_tex.use(0)
        else:
            if self._dummy_tex is None:
                self._dummy_tex = ctx.texture(
                    (2, 2), 4, np.zeros((2, 2, 4), dtype=np.uint8).tobytes()
                )
            self._dummy_tex.use(0)

        self._apply_uniforms(w, h)
        if self._image_uniform_name in self._prog:
            self._prog[self._image_uniform_name].value = 0

        set_blend(ctx, self.blend_mode)
        self._vao.render(moderngl.TRIANGLES)
        set_blend(ctx, None)

    def close(self):
        super().close()
        for res in (self._fbo, self._fbo_tex, self._dummy_tex):
            if res is not None:
                try:
                    res.release()
                except Exception:
                    pass
        self._fbo = self._fbo_tex = self._dummy_tex = None
