"""
ModernGL window that executes the script, renders the visuals
and hot-reloads the script whenever its file changes on disk.
"""

import argparse
import os
import time
import traceback

import glfw
import moderngl
import numpy as np

from .core import Pipeline
from .gfx import draw_geometry, set_blend, state
from .post import PostChain
from .visuals import Output

_MAX_DT = 0.1  # clamp stalls (window drag, breakpoint) so animation never jumps


class Runner:
    """Watches `script_path`, executes it to build a Pipeline, and renders it each frame."""

    def __init__(self, script_path, width=900, height=900, title="chladni"):
        self.script_path = os.path.abspath(script_path)
        self.width = width
        self.height = height
        self.title = title
        self._mtimes = {}
        self._last_check = 0.0
        self._reported = set()
        self.pipeline = None

    def _watched_paths(self):
        paths = [self.script_path]
        if self.pipeline is not None:
            for node in self.pipeline.nodes:
                paths.extend(getattr(node, "watch_paths", ()))
        return paths

    def _snapshot(self):
        mtimes = {}
        for path in self._watched_paths():
            try:
                mtimes[path] = os.path.getmtime(path)
            except OSError:
                pass
        return mtimes

    def _load(self):
        """Re-execute the script in a fresh namespace and swap in its Pipeline."""
        with open(self.script_path) as f:
            source = f.read()
        namespace = {"__file__": self.script_path, "__name__": "__main__"}
        code = compile(source, self.script_path, "exec")
        exec(code, namespace)
        pipeline = namespace.get("pipeline")
        if not isinstance(pipeline, Pipeline):
            raise RuntimeError(
                "run.py must assign a Pipeline to a module-level `pipeline` variable, "
                "e.g. `with Pipeline() as pipeline: ...`"
            )
        self._close_pipeline(self.pipeline)
        self.pipeline = pipeline
        self._reported.clear()

    @staticmethod
    def _close_pipeline(pipeline):
        """Release GPU resources held by nodes of a pipeline that is being replaced."""
        for node in getattr(pipeline, "nodes", ()):
            close = getattr(node, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    traceback.print_exc()

    def _reload_if_changed(self, force=False):
        now = time.perf_counter()
        if not force and now - self._last_check < 0.25:
            return
        self._last_check = now
        mtimes = self._snapshot()
        if not force and mtimes == self._mtimes:
            return
        self._mtimes = mtimes
        try:
            self._load()
            self._mtimes = self._snapshot()
            print(f"[chladni] reloaded {os.path.basename(self.script_path)}")
        except Exception:
            print(
                f"[chladni] reload failed, keeping previous pipeline:\n{traceback.format_exc()}"
            )

    def _draw_layers(self, ctx, outputs):
        for output in outputs:
            for geometry in output.layers:
                set_blend(ctx, None)
                try:
                    draw_geometry(ctx, geometry)
                except Exception as exc:
                    key = (id(geometry), str(exc))
                    if key not in self._reported:
                        self._reported.add(key)

    def run(self, max_frames=None, screenshot=None):
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, True)

        window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed")
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        ctx = moderngl.create_context()
        post = PostChain(ctx)
        gfx = state(ctx)

        self._reload_if_changed(force=True)
        if self.pipeline is None:
            glfw.terminate()
            raise RuntimeError("could not load the script; see the error above")
        last_time = time.perf_counter()
        frame = 0

        try:
            while not glfw.window_should_close(window):
                self._reload_if_changed()

                now = time.perf_counter()
                dt = min(now - last_time, _MAX_DT)
                last_time = now

                fb_w, fb_h = glfw.get_framebuffer_size(window)
                if fb_w == 0 or fb_h == 0:  # minimised
                    glfw.poll_events()
                    continue
                win_h = max(glfw.get_window_size(window)[1], 1)

                self.pipeline.tick(dt)

                outputs = [n for n in self.pipeline.nodes if isinstance(n, Output)]
                settings = outputs[0].settings() if outputs else Output().settings()
                background = outputs[0].background if outputs else (0.0, 0.0, 0.0)

                scene = post.begin((fb_w, fb_h), settings["render_scale"])
                gfx.pixel_scale = scene.size[1] / win_h
                ctx.viewport = (0, 0, scene.size[0], scene.size[1])
                ctx.clear(*background, 1.0)
                self._draw_layers(ctx, outputs)

                post.finish(settings, dt, ctx.screen, (fb_w, fb_h))

                frame += 1
                if max_frames is not None and frame >= max_frames:
                    if screenshot:
                        self._save_screenshot(ctx, screenshot, fb_w, fb_h)
                    break

                glfw.swap_buffers(window)
                glfw.poll_events()
        finally:
            self._close_pipeline(self.pipeline)
            post.release()
            glfw.terminate()

    @staticmethod
    def _save_screenshot(ctx, path, width, height):
        import cv2

        data = np.frombuffer(
            ctx.screen.read(viewport=(0, 0, width, height), components=3), np.uint8
        )
        image = data.reshape(height, width, 3)[::-1]
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"[chladni] saved {path} ({width}x{height})")


def main():
    parser = argparse.ArgumentParser(
        description="Run a chladni script in a live window."
    )
    parser.add_argument("script", nargs="?", default="run.py")
    parser.add_argument(
        "--frames", type=int, default=None, help="exit after this many frames"
    )
    parser.add_argument(
        "--screenshot", default=None, help="save the last frame here (needs --frames)"
    )
    args = parser.parse_args()
    Runner(args.script).run(max_frames=args.frames, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
