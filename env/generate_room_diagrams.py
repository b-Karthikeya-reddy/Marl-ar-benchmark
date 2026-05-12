from pathlib import Path

from visualize_room import (
    ROOM_TYPES,
    plot_room_layout,
    plot_room_layout_grid,
    plot_utilization_summary,
    run_episode,
)


def slugify(room_type):
    return room_type.lower().replace(" ", "_")


def main():
    output_dir = Path(__file__).resolve().parents[1] / "progress" / "diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for room_type in ROOM_TYPES:
        env = run_episode(room_type=room_type, room_size=15, target_utilization=0.5)
        utilization = env._current_utilization() * 100
        room_output = output_dir / f"{slugify(room_type)}_layout.png"
        plot_room_layout(
            env,
            output_path=room_output,
            title=f"MARL Furniture Placement - {room_type}",
            show=False,
        )
        results.append({
            "room_type": room_type,
            "env": env,
            "utilization": utilization,
        })
        print(f"Saved {room_type} layout to {room_output}")

    plot_room_layout_grid(results, output_dir / "all_rooms_layouts.png", show=False)
    plot_utilization_summary(results, output_dir / "utilization_summary.png", show=False)
    print(f"Saved combined diagrams to {output_dir}")


if __name__ == "__main__":
    main()
