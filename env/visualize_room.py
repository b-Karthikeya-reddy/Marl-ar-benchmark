from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from ikea_furniture_env import IKEAFurnitureEnv

ROOM_TYPES = ["Living Room", "Bedroom", "Kitchen", "Bathroom", "Dining Room"]
AGENT_COLORS = {
    "layout_agent": "#3498db",
    "style_agent": "#e74c3c",
}


def run_episode(room_type="Living Room", room_size=15, num_furniture=5, target_utilization=0.5):
    env = IKEAFurnitureEnv(
        room_size=room_size,
        num_furniture=num_furniture,
        room_type=room_type,
        target_utilization=target_utilization,
    )
    env.reset()

    done = False
    while not done:
        try:
            agent = env.agent_selection
            _, _, term, trunc, _ = env.last()
            action = None if term or trunc else env.action_space(agent).sample()
            env.step(action)
            done = all(
                env.terminations.get(agent_name, False) or env.truncations.get(agent_name, False)
                for agent_name in env.possible_agents
            )
        except Exception:
            done = True

    return env


def _draw_room(ax, env, title, include_labels=True):
    ax.set_xlim(0, env.room_size)
    ax.set_ylim(0, env.room_size)
    ax.set_facecolor("#f5f5f5")
    ax.grid(True, alpha=0.3)

    for item in env.placed_furniture:
        color = AGENT_COLORS[item["agent"]]
        rect = patches.Rectangle(
            (item["y"], env.room_size - item["x"] - item["height"]),
            item["width"],
            item["height"],
            linewidth=2,
            edgecolor="black",
            facecolor=color,
            alpha=0.7,
        )
        ax.add_patch(rect)

        if include_labels:
            ax.text(
                item["y"] + item["width"] / 2,
                env.room_size - item["x"] - item["height"] / 2,
                f"{item['category']}\n{item['real_size']}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
                wrap=True,
            )

    utilization = env._current_utilization() * 100
    ax.set_title(f"{title}\nUtilization: {utilization:.1f}%", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Room Width (grid units)", fontsize=11)
    ax.set_ylabel("Room Length (grid units)", fontsize=11)


def plot_room_layout(env, output_path=None, title=None, show=False, include_labels=True):
    title = title or f"MARL Furniture Placement - {env.room_type}"
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    _draw_room(ax, env, title, include_labels=include_labels)

    legend_elements = [
        patches.Patch(facecolor=AGENT_COLORS["layout_agent"], label="Layout Agent"),
        patches.Patch(facecolor=AGENT_COLORS["style_agent"], label="Style Agent"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)

    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_room_layout_grid(results, output_path, show=False):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    flat_axes = axes.flatten()

    for ax, result in zip(flat_axes, results):
        _draw_room(ax, result["env"], result["room_type"], include_labels=False)

    for ax in flat_axes[len(results):]:
        ax.axis("off")

    legend_elements = [
        patches.Patch(facecolor=AGENT_COLORS["layout_agent"], label="Layout Agent"),
        patches.Patch(facecolor=AGENT_COLORS["style_agent"], label="Style Agent"),
    ]
    fig.legend(handles=legend_elements, loc="upper right")
    fig.suptitle("MARL Furniture Placement by Room Type", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.97, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_utilization_summary(results, output_path, show=False):
    room_types = [result["room_type"] for result in results]
    utilizations = [result["utilization"] for result in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(room_types, utilizations, color="#5dade2", edgecolor="black")
    ax.axhline(50, color="#e74c3c", linestyle="--", linewidth=2, label="50% Target")
    ax.set_ylim(0, max(60, max(utilizations) + 5))
    ax.set_ylabel("Space Utilization (%)")
    ax.set_title("Space Utilization by Room Type", fontsize=16, fontweight="bold")
    ax.legend()

    for bar, value in zip(bars, utilizations):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    output_path = Path(__file__).resolve().parents[1] / "progress" / "room_visualization.png"
    env = run_episode(room_type="Living Room", room_size=15, target_utilization=0.5)
    plot_room_layout(env, output_path=output_path, show=True)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()