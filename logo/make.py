import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def generate_minimal_chladni_logo(
    filename_svg="logo/chladni_logo.svg",
    filename_png="logo/chladni_logo.png",
    r_outer=1.0,
    r_max=0.82,
    r_min=0.27,
    pinch=1.8,
    center_dot_radius=0.042,
    line_width=8,
    color="black",
    bg_color="white"
):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # 1. Outer Plate Boundary
    outer_circle = Circle(
        (0, 0), r_outer, fill=False, edgecolor=color, linewidth=line_width * 1.15
    )
    ax.add_patch(outer_circle)

    # 2. Central Vibration Node (Dot)
    center_node = Circle(
        (0, 0), center_dot_radius, fill=True, facecolor=color, edgecolor="none"
    )
    ax.add_patch(center_node)

    # 3. 4-Fold Standing Wave Loop (Polar Harmonic Curve)
    theta = np.linspace(0, 2 * np.pi, 1000)
    r_wave = r_min + (r_max - r_min) * ((1 + np.cos(4 * theta)) / 2) ** pinch

    x = r_wave * np.cos(theta)
    y = r_wave * np.sin(theta)

    ax.plot(x, y, color=color, linewidth=line_width, solid_joinstyle="round")

    # Canvas formatting and export
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(filename_svg, format="svg", bbox_inches="tight", pad_inches=0.1, facecolor=bg_color)
    plt.savefig(filename_png, format="png", bbox_inches="tight", pad_inches=0.1, facecolor=bg_color)
    plt.close()
    print(f"Saved '{filename_svg}' and '{filename_png}'.")

if __name__ == "__main__":
    generate_minimal_chladni_logo()
