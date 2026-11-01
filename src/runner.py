"""
ModernGL window that executes `run.py` to build a `Pipeline`, renders it
every frame, and hot-reloads the script whenever its file changes on disk.
"""

import os
import sys
import time
import traceback

import glfw
import moderngl
import numpy as np

from .core import Pipeline
from .visuals import Output

_VERTEX_SHADER = """
#version 330
in vec2 in_pos;
in vec3 in_color;
uniform float point_size;
out vec3 v_color;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    gl_PointSize = point_size;
    v_color = in_color;
}
"""

_FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"""


class Runner:
    """Watches `script_path`, executes it to build a Pipeline, and renders it each frame."""

    def __init__(self, script_path, width=900, height=900, title="chladni"):
        self.script_path = os.path.abspath(script_path)
        self.width = width
        self.height = height
        self.title = title
        self._mtime = None
        self.pipeline = None

    def _load(self):
        """Re-execute the script in a fresh namespace and swap in its Pipeline."""
        source = open(self.script_path).read()
        namespace = {"__file__": self.script_path, "__name__": "__main__"}
        code = compile(source, self.script_path, "exec")
        exec(code, namespace)
        pipeline = namespace.get("pipeline")
        if not isinstance(pipeline, Pipeline):
            raise RuntimeError(
                "run.py must assign a Pipeline to a module-level `pipeline` variable, "
                "e.g. `with Pipeline() as pipeline: ...`"
            )
        self.pipeline = pipeline

    def _reload_if_changed(self):
        mtime = os.path.getmtime(self.script_path)
        if mtime == self._mtime:
            return
        self._mtime = mtime
        try:
            self._load()
            print(f"[chladni] reloaded {os.path.basename(self.script_path)}")
        except Exception:
            print(f"[chladni] reload failed, keeping previous pipeline:\n{traceback.format_exc()}")

    def run(self):
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed")
        glfw.make_context_current(window)

        ctx = moderngl.create_context()
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        program = ctx.program(vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER)
        modes = {
            "fan": moderngl.TRIANGLE_FAN,
            "triangles": moderngl.TRIANGLES,
            "lines": moderngl.LINE_LOOP,
            "points": moderngl.POINTS,
        }

        self._reload_if_changed()
        last_time = time.perf_counter()

        while not glfw.window_should_close(window):
            self._reload_if_changed()

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            self.pipeline.tick(dt)

            outputs = [node for node in self.pipeline.nodes if isinstance(node, Output)]
            ctx.clear(*(outputs[0].background if outputs else (0.0, 0.0, 0.0)))

            for output in outputs:
                for geometry in output.layers:
                    if hasattr(geometry, "render"):
                        geometry.render(ctx)
                        continue
                    positions, colors, mode = geometry.vertices()
                    if len(positions) == 0:
                        continue
                    vertex_data = np.hstack([positions, colors]).astype("f4").tobytes()
                    vbo = ctx.buffer(vertex_data)
                    vao = ctx.vertex_array(program, [(vbo, "2f 3f", "in_pos", "in_color")])
                    program["point_size"].value = float(getattr(geometry, "size", 1.0))
                    vao.render(modes[mode])
                    vao.release()
                    vbo.release()

            glfw.swap_buffers(window)
            glfw.poll_events()

        glfw.terminate()


def main():
    script = sys.argv[1] if len(sys.argv) > 1 else "run.py"
    Runner(script).run()


if __name__ == "__main__":
    main()
