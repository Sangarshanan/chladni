r"""
Interactive Livecoding Text Editor Overlay for Chladni.

Renders an on-screen code editor directly over running visuals (similar to
Hydra, LiveCodeLab, and TidalCycles). Features:
  - Non-intrusive semi-transparent feathered panel on the left of the screen
  - Crisp Retina/High-DPI monospace typography (Menlo / SF Mono / Consolas)
  - Python syntax token highlighting tailored for live visual coding
  - Bi-directional sync: edit on screen (Cmd+Enter / Cmd+S) or edit in VS Code
  - Keyboard navigation, auto-indentation, undo, clipboard copy/paste
  - Evaluation flash pulse and non-fatal inline error reporting
  - Hotkey toggle (Cmd+E, Ctrl+E, or Ctrl+\) to show/hide the code overlay
"""

import os
import re
import time
import numpy as np
import moderngl
import glfw
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Monospace Font Discovery
# ---------------------------------------------------------------------------

_FONT_PATHS = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Monaco.dfont",
    "/Library/Fonts/Courier New.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "C:\\Windows\\Fonts\\consola.ttf",
]


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Syntax Tokenizer & Visual Palette
# ---------------------------------------------------------------------------

_TOKEN_REGEX = re.compile(
    r"(?P<COMMENT>#[^\n]*)|"
    r"(?P<STRING>'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"|'[^'\n]*'|\"[^\"\n]*\")|"
    r"(?P<NUMBER>\b\d+\.?\d*(?:[eE][+-]?\d+)?\b)|"
    r"(?P<KEYWORD>\b(from|import|with|as|def|class|return|if|elif|else|for|while|try|except|finally|in|is|not|and|or|pass|break|continue|lambda|global|nonlocal|raise|yield|async|await)\b)|"
    r"(?P<BUILTIN>\b(True|False|None|self|Pipeline|AudioBands|Impulse|Noise|GaussianNoise|ISFGenerator|ISFFilter|CameraOverlay|GestureInput|Circle|Bars|Particles|Output|int|float|str|list|dict|tuple|set|range|len|print)\b)|"
    r"(?P<OPERATOR>[+\-*/%=&|^~<>!@:]+|>>)|"
    r"(?P<IDENTIFIER>\b[a-zA-Z_]\w*\b)|"
    r"(?P<PUNCTUATION>[(){}\[\],.])|"
    r"(?P<WHITESPACE>[ \t]+)"
)

_THEME = {
    "KEYWORD": (255, 110, 180, 255),    # Hot Pink
    "BUILTIN": (85, 230, 255, 255),     # Electric Cyan
    "STRING": (140, 240, 160, 255),     # Mint / Emerald
    "NUMBER": (255, 205, 90, 255),      # Amber Gold
    "OPERATOR": (180, 195, 255, 255),   # Lavender Blue
    "COMMENT": (120, 138, 165, 215),    # Muted Slate
    "IDENTIFIER": (238, 244, 255, 255), # Crisp White
    "PUNCTUATION": (175, 190, 215, 235),
    "DEFAULT": (238, 244, 255, 255),
}

# ---------------------------------------------------------------------------
# Shaders for Texture Overlay Blitting
# ---------------------------------------------------------------------------

_OVERLAY_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_OVERLAY_FRAG = """
#version 330
uniform sampler2D tex;
uniform float opacity;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec4 c = texture(tex, v_uv);
    f_color = vec4(c.rgb * opacity, c.a * opacity);
}
"""


# ---------------------------------------------------------------------------
# LiveEditor Class
# ---------------------------------------------------------------------------

class LiveEditor:
    """Livecoding text editor overlay rendered on top of the visual output."""

    def __init__(self, script_path: str, runner, window=None):
        self.script_path = os.path.abspath(script_path)
        self.runner = runner
        self.window = window

        self.visible = True
        self.opacity = 1.0
        self.lines: list[str] = [""]
        self.cursor_row = 0
        self.cursor_col = 0
        self.scroll_row = 0

        self.is_dirty = False
        self.last_disk_mtime = 0.0
        self.last_error: str | None = None
        self.error_line: int | None = None

        self.flash_alpha = 0.0
        self.blink_time = 0.0
        self.undo_stack: list[tuple[list[str], int, int]] = []

        # GPU resources
        self._prog = None
        self._vbo = None
        self._vao = None
        self._tex = None
        self._last_bounds = None
        self._needs_redraw = True

        # Cached layout & font metrics
        self._cached_font = None
        self._cached_font_size = 0
        self._char_w = 10.0
        self._line_h = 24
        self._cached_size = (0, 0)
        self._backdrop_img: Image.Image | None = None

        self._load_from_disk()

        if self.window is not None:
            self.attach_window(self.window)

    def attach_window(self, window):
        """Register GLFW input callbacks for typing, shortcuts, and scrolling."""
        self.window = window
        glfw.set_char_callback(window, self._on_char)
        glfw.set_key_callback(window, self._on_key)
        glfw.set_mouse_button_callback(window, self._on_mouse_button)
        glfw.set_scroll_callback(window, self._on_scroll)

    def _load_from_disk(self):
        """Read initial or external source code from file."""
        if not os.path.isfile(self.script_path):
            self.lines = ["# Welcome to Chladni Livecoding", ""]
            return
        try:
            with open(self.script_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.lines = content.splitlines() or [""]
            self.cursor_row = min(self.cursor_row, len(self.lines) - 1)
            self.cursor_col = min(self.cursor_col, len(self.lines[self.cursor_row]))
            self.last_disk_mtime = os.path.getmtime(self.script_path)
            self.is_dirty = False
            self.last_error = None
            self.error_line = None
            self._needs_redraw = True
        except Exception as e:
            print(f"[editor] load error: {e}")

    def on_external_reload(self):
        """Called by Runner when script was edited externally (e.g. in VS Code)."""
        if not os.path.isfile(self.script_path):
            return
        try:
            mtime = os.path.getmtime(self.script_path)
            if mtime == self.last_disk_mtime:
                return
            self.last_disk_mtime = mtime
            with open(self.script_path, "r", encoding="utf-8") as f:
                disk_text = f.read()
            # If disk differs from editor buffer, update editor
            if disk_text != "\n".join(self.lines):
                self.lines = disk_text.splitlines() or [""]
                self.cursor_row = min(self.cursor_row, len(self.lines) - 1)
                self.cursor_col = min(self.cursor_col, len(self.lines[self.cursor_row]))
                self.is_dirty = False
                self.last_error = None
                self.error_line = None
                self._needs_redraw = True
        except Exception as e:
            print(f"[editor] external sync error: {e}")

    def evaluate(self):
        """Save buffer to disk and hot-swap pipeline immediately."""
        code = "\n".join(self.lines)
        if not code.endswith("\n"):
            code += "\n"
        try:
            # 1. Write buffer to disk
            with open(self.script_path, "w", encoding="utf-8") as f:
                f.write(code)

            # 2. Update runner's mtime snapshot to prevent double-load
            self.last_disk_mtime = os.path.getmtime(self.script_path)
            if hasattr(self.runner, "_mtimes"):
                self.runner._mtimes[self.script_path] = self.last_disk_mtime

            # 3. Reload running pipeline
            self.runner._load()
            self.last_error = None
            self.error_line = None
            self.is_dirty = False
            self.flash_alpha = 1.0  # Trigger evaluation glow pulse
            print(f"[editor] evaluated & saved {os.path.basename(self.script_path)}")
        except Exception as exc:
            err_line = getattr(exc, "lineno", None)
            self.error_line = err_line
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[editor] eval error: {self.last_error}")
        self._needs_redraw = True

    def _push_undo(self):
        """Save buffer state for Ctrl+Z undo."""
        if len(self.undo_stack) > 60:
            self.undo_stack.pop(0)
        self.undo_stack.append(([line for line in self.lines], self.cursor_row, self.cursor_col))

    def _undo(self):
        if not self.undo_stack:
            return
        lines, row, col = self.undo_stack.pop()
        self.lines = lines
        self.cursor_row = min(row, len(self.lines) - 1)
        self.cursor_col = min(col, len(self.lines[self.cursor_row]))
        self.is_dirty = True
        self._needs_redraw = True

    # -----------------------------------------------------------------------
    # Editing Operations
    # -----------------------------------------------------------------------

    def insert_char(self, c: str):
        self._push_undo()
        line = self.lines[self.cursor_row]
        self.lines[self.cursor_row] = line[:self.cursor_col] + c + line[self.cursor_col:]
        self.cursor_col += len(c)
        self.is_dirty = True
        self.blink_time = 0.0
        self._ensure_cursor_visible()
        self._needs_redraw = True

    def insert_newline(self):
        self._push_undo()
        line = self.lines[self.cursor_row]
        prefix = line[:self.cursor_col]
        suffix = line[self.cursor_col:]

        # Smart indentation: inherit leading whitespace and add 4 spaces if ends with ':'
        indent_len = len(prefix) - len(prefix.lstrip(" "))
        indent = " " * indent_len
        if prefix.rstrip().endswith(":"):
            indent += "    "

        self.lines[self.cursor_row] = prefix
        self.lines.insert(self.cursor_row + 1, indent + suffix)
        self.cursor_row += 1
        self.cursor_col = len(indent)
        self.is_dirty = True
        self.blink_time = 0.0
        self._ensure_cursor_visible()
        self._needs_redraw = True

    def backspace(self):
        self._push_undo()
        if self.cursor_col > 0:
            line = self.lines[self.cursor_row]
            # Smart unindent: if cursor is preceded by up to 4 spaces, delete the block
            prefix = line[:self.cursor_col]
            if prefix.endswith("    ") and prefix.strip() == "":
                self.lines[self.cursor_row] = line[:self.cursor_col - 4] + line[self.cursor_col:]
                self.cursor_col -= 4
            else:
                self.lines[self.cursor_row] = line[:self.cursor_col - 1] + line[self.cursor_col:]
                self.cursor_col -= 1
        elif self.cursor_row > 0:
            prev_line = self.lines[self.cursor_row - 1]
            curr_line = self.lines.pop(self.cursor_row)
            self.cursor_row -= 1
            self.cursor_col = len(prev_line)
            self.lines[self.cursor_row] = prev_line + curr_line
        self.is_dirty = True
        self.blink_time = 0.0
        self._ensure_cursor_visible()
        self._needs_redraw = True

    def delete(self):
        self._push_undo()
        line = self.lines[self.cursor_row]
        if self.cursor_col < len(line):
            self.lines[self.cursor_row] = line[:self.cursor_col] + line[self.cursor_col + 1:]
        elif self.cursor_row < len(self.lines) - 1:
            next_line = self.lines.pop(self.cursor_row + 1)
            self.lines[self.cursor_row] = line + next_line
        self.is_dirty = True
        self.blink_time = 0.0
        self._needs_redraw = True

    def _ensure_cursor_visible(self, margin=3):
        # Auto-scroll so cursor row stays comfortably visible
        if self.cursor_row < self.scroll_row + margin:
            self.scroll_row = max(0, self.cursor_row - margin)
        # Approximate visible lines based on line_h
        visible_lines = max(5, self._cached_size[1] // max(1, self._line_h) - 4)
        if self.cursor_row >= self.scroll_row + visible_lines - margin:
            self.scroll_row = self.cursor_row - visible_lines + margin + 1

    # -----------------------------------------------------------------------
    # GLFW Event Handlers
    # -----------------------------------------------------------------------

    def _on_char(self, window, codepoint):
        if not self.visible:
            return
        c = chr(codepoint)
        if c.isprintable():
            self.insert_char(c)

    def _on_key(self, window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return

        cmd = bool(mods & (glfw.MOD_CONTROL | glfw.MOD_SUPER))
        shift = bool(mods & glfw.MOD_SHIFT)

        # Toggle editor visibility: Cmd+E, Ctrl+E, Ctrl+\, or F1
        if (cmd and key in (glfw.KEY_E, glfw.KEY_BACKSLASH)) or key == glfw.KEY_F1:
            self.visible = not self.visible
            self._needs_redraw = True
            return

        if not self.visible:
            return

        # Evaluation & Save: Cmd+Enter, Ctrl+Enter, Cmd+S, Ctrl+S
        if cmd and key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER, glfw.KEY_S):
            self.evaluate()
            return

        # Undo: Cmd+Z, Ctrl+Z
        if cmd and key == glfw.KEY_Z:
            self._undo()
            return

        # Clipboard Copy: Cmd+C
        if cmd and key == glfw.KEY_C:
            line = self.lines[self.cursor_row]
            glfw.set_clipboard_string(window, line)
            return

        # Clipboard Paste: Cmd+V
        if cmd and key == glfw.KEY_V:
            clip = glfw.get_clipboard_string(window)
            if clip:
                self._push_undo()
                for c in clip:
                    if c == "\n":
                        self.insert_newline()
                    elif c.isprintable() or c == "\t":
                        self.insert_char("    " if c == "\t" else c)
            return

        # Navigation & Editing
        if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            self.insert_newline()
        elif key == glfw.KEY_BACKSPACE:
            self.backspace()
        elif key == glfw.KEY_DELETE:
            self.delete()
        elif key == glfw.KEY_TAB:
            if shift:
                # Unindent
                line = self.lines[self.cursor_row]
                if line.startswith("    "):
                    self.lines[self.cursor_row] = line[4:]
                    self.cursor_col = max(0, self.cursor_col - 4)
                    self.is_dirty = True
                    self._needs_redraw = True
            else:
                self.insert_char("    ")
        elif key == glfw.KEY_UP:
            if self.cursor_row > 0:
                self.cursor_row -= 1
                self.cursor_col = min(self.cursor_col, len(self.lines[self.cursor_row]))
                self.blink_time = 0.0
                self._ensure_cursor_visible()
                self._needs_redraw = True
        elif key == glfw.KEY_DOWN:
            if self.cursor_row < len(self.lines) - 1:
                self.cursor_row += 1
                self.cursor_col = min(self.cursor_col, len(self.lines[self.cursor_row]))
                self.blink_time = 0.0
                self._ensure_cursor_visible()
                self._needs_redraw = True
        elif key == glfw.KEY_LEFT:
            if self.cursor_col > 0:
                self.cursor_col -= 1
            elif self.cursor_row > 0:
                self.cursor_row -= 1
                self.cursor_col = len(self.lines[self.cursor_row])
            self.blink_time = 0.0
            self._ensure_cursor_visible()
            self._needs_redraw = True
        elif key == glfw.KEY_RIGHT:
            if self.cursor_col < len(self.lines[self.cursor_row]):
                self.cursor_col += 1
            elif self.cursor_row < len(self.lines) - 1:
                self.cursor_row += 1
                self.cursor_col = 0
            self.blink_time = 0.0
            self._ensure_cursor_visible()
            self._needs_redraw = True
        elif key == glfw.KEY_HOME:
            line = self.lines[self.cursor_row]
            first_non_blank = len(line) - len(line.lstrip(" "))
            self.cursor_col = 0 if self.cursor_col == first_non_blank else first_non_blank
            self.blink_time = 0.0
            self._needs_redraw = True
        elif key == glfw.KEY_END:
            self.cursor_col = len(self.lines[self.cursor_row])
            self.blink_time = 0.0
            self._needs_redraw = True
        elif key == glfw.KEY_PAGE_UP:
            self.cursor_row = max(0, self.cursor_row - 15)
            self.cursor_col = min(self.cursor_col, len(self.lines[self.cursor_row]))
            self._ensure_cursor_visible()
            self._needs_redraw = True
        elif key == glfw.KEY_PAGE_DOWN:
            self.cursor_row = min(len(self.lines) - 1, self.cursor_row + 15)
            self.cursor_col = min(self.cursor_col, len(self.lines[self.cursor_row]))
            self._ensure_cursor_visible()
            self._needs_redraw = True

    def _on_mouse_button(self, window, button, action, mods):
        if not self.visible or action != glfw.PRESS:
            return
        x, y = glfw.get_cursor_pos(window)
        win_w, _ = glfw.get_window_size(window)
        panel_w = max(420, min(int(win_w * 0.48), 750))

        if x > panel_w:
            return  # Click was in visual area outside the editor

        top_margin = int(38 * (self._cached_size[0] / panel_w if panel_w else 1.0))
        gutter_w = int(self._char_w * 5.8)
        row = max(0, int((y * (self._cached_size[1] / max(1, glfw.get_window_size(window)[1])) - top_margin) / max(1, self._line_h)) + self.scroll_row)
        col = max(0, round((x * (self._cached_size[0] / panel_w) - gutter_w) / max(1.0, self._char_w)))

        self.cursor_row = min(row, len(self.lines) - 1)
        self.cursor_col = min(col, len(self.lines[self.cursor_row]))
        self.blink_time = 0.0
        self._needs_redraw = True

    def _on_scroll(self, window, xoffset, yoffset):
        if not self.visible:
            return
        self.scroll_row = max(0, min(len(self.lines) - 3, int(self.scroll_row - yoffset * 3)))
        self._needs_redraw = True

    # -----------------------------------------------------------------------
    # Rendering & ModernGL Overlay Blitting
    # -----------------------------------------------------------------------

    def _build_backdrop(self, width: int, height: int):
        """Build semi-transparent dark panel with horizontal feather on right edge."""
        arr = np.zeros((height, width, 4), dtype=np.uint8)
        base_color = np.array([10, 12, 18], dtype=np.uint8)
        base_alpha = 195  # Translucent dark charcoal
        feather_w = min(48, max(16, width // 10))
        solid_w = width - feather_w

        arr[:, :solid_w, :3] = base_color
        arr[:, :solid_w, 3] = base_alpha

        alpha_curve = np.linspace(base_alpha, 0, feather_w, endpoint=True, dtype=np.uint8)
        arr[:, solid_w:, :3] = base_color
        arr[:, solid_w:, 3] = alpha_curve

        self._backdrop_img = Image.fromarray(arr, mode="RGBA")

    def _draw_editor_image(self, tex_w: int, tex_h: int, pixel_ratio: float) -> Image.Image:
        """Render editor state, syntax tokens, and UI elements to an RGBA PIL image."""
        font_size = int(14.5 * pixel_ratio)
        if self._cached_font is None or self._cached_font_size != font_size:
            self._cached_font = _get_font(font_size)
            self._cached_font_size = font_size
            self._char_w = max(1.0, self._cached_font.getlength("M"))
            self._line_h = max(12, int(font_size * 1.48))

        font = self._cached_font
        char_w = self._char_w
        line_h = self._line_h
        gutter_w = int(char_w * 5.6)
        top_bar_h = int(36 * pixel_ratio)
        bottom_bar_h = int(28 * pixel_ratio)

        # 1. Base translucent feathered panel
        img = self._backdrop_img.copy()
        draw = ImageDraw.Draw(img)

        # 2. Evaluation flash animation
        if self.flash_alpha > 0.001:
            flash_a = int(75 * self.flash_alpha)
            draw.rectangle([0, 0, tex_w, tex_h], fill=(90, 230, 255, flash_a))

        # 3. Top Header Bar
        filename = os.path.basename(self.script_path)
        status_text = "● LIVE" if not self.is_dirty else "● EDITING"
        status_color = (80, 240, 160, 255) if not self.is_dirty else (255, 190, 80, 255)

        draw.rectangle([0, 0, tex_w - 30, top_bar_h], fill=(16, 20, 30, 160))
        draw.text((int(14 * pixel_ratio), int(8 * pixel_ratio)), status_text, fill=status_color, font=font)
        draw.text((int(90 * pixel_ratio), int(8 * pixel_ratio)), filename, fill=(210, 225, 245, 230), font=font)

        hint_text = "Cmd+Enter: Eval | Cmd+E: Hide"
        hint_w = font.getlength(hint_text)
        draw.text((tex_w - int(hint_w + 35 * pixel_ratio), int(8 * pixel_ratio)), hint_text, fill=(120, 135, 160, 190), font=font)

        # Gutter separator line
        draw.line([(gutter_w - 6, top_bar_h), (gutter_w - 6, tex_h - bottom_bar_h)], fill=(45, 55, 75, 140), width=1)

        # 4. Visible Lines
        max_visible = max(1, (tex_h - top_bar_h - bottom_bar_h) // line_h)
        start_row = self.scroll_row
        end_row = min(len(self.lines), start_row + max_visible + 1)

        for i in range(start_row, end_row):
            y = top_bar_h + (i - start_row) * line_h

            # Active cursor line highlight
            if i == self.cursor_row:
                draw.rectangle([0, y, tex_w - 30, y + line_h], fill=(45, 80, 130, 65))
                # Accent indicator in margin
                draw.rectangle([0, y, int(3 * pixel_ratio), y + line_h], fill=(90, 230, 255, 220))

            # Error line highlight
            if self.error_line is not None and (i + 1) == self.error_line:
                draw.rectangle([0, y, tex_w - 30, y + line_h], fill=(220, 60, 80, 75))

            # Line Number
            line_num_str = f"{i + 1:>4} "
            num_color = (240, 245, 255, 240) if i == self.cursor_row else (90, 105, 130, 180)
            draw.text((int(8 * pixel_ratio), y), line_num_str, fill=num_color, font=font)

            # Code text with syntax token coloring
            line_str = self.lines[i]
            x = gutter_w
            for match in _TOKEN_REGEX.finditer(line_str):
                kind = match.lastgroup
                val = match.group()
                color = _THEME.get(kind, _THEME["DEFAULT"])
                draw.text((x, y), val, fill=color, font=font)
                x += char_w * len(val)

            # Draw Cursor
            if i == self.cursor_row and (int(self.blink_time * 2.2) % 2 == 0):
                cur_x = gutter_w + char_w * self.cursor_col
                draw.rectangle([cur_x, y + 2, cur_x + max(2, int(2.5 * pixel_ratio)), y + line_h - 2], fill=(90, 235, 255, 255))

        # 5. Bottom Status / Error Bar
        draw.rectangle([0, tex_h - bottom_bar_h, tex_w - 30, tex_h], fill=(14, 18, 26, 210))
        if self.last_error:
            # Error banner
            err_short = self.last_error.replace("\n", " ")[:75]
            draw.text((int(14 * pixel_ratio), tex_h - bottom_bar_h + int(5 * pixel_ratio)), f"✖ {err_short}", fill=(255, 95, 110, 255), font=font)
        else:
            coord_str = f"Ln {self.cursor_row + 1}, Col {self.cursor_col + 1}"
            draw.text((int(14 * pixel_ratio), tex_h - bottom_bar_h + int(5 * pixel_ratio)), coord_str, fill=(120, 138, 165, 200), font=font)
            save_str = "● Unsaved edits" if self.is_dirty else "Saved & Hot"
            save_color = (255, 180, 80, 220) if self.is_dirty else (80, 220, 160, 220)
            draw.text((int(150 * pixel_ratio), tex_h - bottom_bar_h + int(5 * pixel_ratio)), save_str, fill=save_color, font=font)

        return img

    def render(self, ctx: moderngl.Context, fb_size: tuple[int, int], win_w: int, win_h: int, dt: float):
        """Render editor overlay onto the current screen framebuffer."""
        if not self.visible or win_w <= 0 or win_h <= 0:
            return

        # Advance animations
        self.blink_time += dt
        if self.flash_alpha > 0.0:
            self.flash_alpha = max(0.0, self.flash_alpha - dt * 3.8)
            self._needs_redraw = True

        # Calculate responsive panel layout
        pixel_ratio = fb_size[0] / max(1, win_w)
        panel_w_win = max(420, min(int(win_w * 0.48), 750))
        tex_w = int(panel_w_win * pixel_ratio)
        tex_h = fb_size[1]

        # Allocate / Re-allocate resources on resize
        if self._cached_size != (tex_w, tex_h):
            self._cached_size = (tex_w, tex_h)
            self._build_backdrop(tex_w, tex_h)
            if self._tex is not None:
                self._tex.release()
            self._tex = ctx.texture((tex_w, tex_h), 4)
            self._tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._needs_redraw = True

        # Compile GPU shader program once
        if self._prog is None:
            self._prog = ctx.program(vertex_shader=_OVERLAY_VERT, fragment_shader=_OVERLAY_FRAG)
            self._vbo = ctx.buffer(reserve=24 * 4, dynamic=True)
            self._vao = ctx.vertex_array(self._prog, [(self._vbo, "2f 2f", "in_pos", "in_uv")])

        # Redraw PIL image & upload texture when needed or blinking
        # Blink cycle update:
        blink_active = (int(self.blink_time * 2.2) % 2 == 0)
        if getattr(self, "_last_blink_state", None) != blink_active:
            self._last_blink_state = blink_active
            self._needs_redraw = True

        if self._needs_redraw and self._backdrop_img is not None:
            img = self._draw_editor_image(tex_w, tex_h, pixel_ratio)
            self._tex.write(img.tobytes())
            self._needs_redraw = False

        # Compute quad geometry in NDC (left of screen)
        bounds = (panel_w_win, win_w)
        if self._last_bounds != bounds:
            self._last_bounds = bounds
            x0 = -1.0
            x1 = -1.0 + 2.0 * (panel_w_win / win_w)
            y0 = -1.0
            y1 = 1.0
            # (UV: y0=-1.0 maps to v=1.0 bottom, y1=1.0 maps to v=0.0 top)
            quad = np.array([
                x0, y0, 0.0, 1.0,
                x1, y0, 1.0, 1.0,
                x1, y1, 1.0, 0.0,
                x0, y0, 0.0, 1.0,
                x1, y1, 1.0, 0.0,
                x0, y1, 0.0, 0.0,
            ], dtype="f4")
            self._vbo.write(quad.tobytes())

        # Render texture quad with premultiplied alpha
        self._tex.use(0)
        self._prog["tex"].value = 0
        self._prog["opacity"].value = float(self.opacity)

        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self._vao.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.BLEND)

    def close(self):
        """Release ModernGL resources."""
        for res in (self._vao, self._vbo, self._prog, self._tex):
            if res is not None:
                try:
                    res.release()
                except Exception:
                    pass
        self._vao = self._vbo = self._prog = self._tex = None
