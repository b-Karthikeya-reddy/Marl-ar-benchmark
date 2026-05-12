from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from gymnasium import spaces
from pettingzoo import AECEnv

ROOM_CATEGORIES = {
    "Living Room": [
        "Three-Seated Sofas", "Two-Seated Sofas", "ArmChairs", "FootStools",
        "Cabinets and Display Cabinets", "Coffee and Side Tables",
        "GRONLID System", "VIMLE System", "EKET System", "HAVSTA Systems",
        "PLATSA Combinations", "VALENTUNA System", "LIDHULT System",
        "SODERHAMN System", "KUNGSHAMN System", "Outdoor Sofas",
    ],
    "Bedroom": [
        "Beds", "Mattresses", "Wardrobes", "Wordrobes",
        "Chest of Drawers", "Bedside Tables", "Dressing Tables",
        "Clothes Organizer", "Open Storage Wordrobes",
        "Freestanding Wardrobes", "PAX Wardrobes",
        "PLATSA Modular Storage System", "Drawer Units Storage Cabinets",
    ],
    "Kitchen": [
        "Cookware", "Frying Pans and Woks", "Kitchen Appliances",
        "Tableware", "DinnerWare", "Serveware", "Cooking Utensils",
        "Kitchen Islands and Trolleys", "Kitchen Taps and Sinks",
        "DishWasher", "Kithcen and Workshops",
    ],
    "Bathroom": [
        "BathroomStorage", "BathroomAccessories", "Bathroom Wall Cabinets",
        "BathroomSinkAccessories", "ToiletAccessories",
        "Towels&Bathmats", "VanityUnits", "Towel40times70",
    ],
    "Home Office": [
        "Office Desks", "Desks Chairs", "Desks Table and Legs",
        "Wall Shelves", "Storage Boxes Baskets", "Paper & Media Organizers",
        "Drawer Units Storage Cabinets", "Home Office Lighting",
    ],
    "Dining Room": [
        "DinnerWare", "Tableware", "Serveware",
        "Kitchen Islands and Trolleys", "Cooking Utensils",
    ],
}

EXCLUDED_CATEGORIES = [
    "Cushion and Cushion Covers", "BedSpreads and Throws",
    "Sofa Covers", "Sofa Accessories and Legs",
    "Decoration", "PaperShop", "Babies Tableware",
    "BabyAccessories", "ChildSafety",
]

DEFAULT_IKEA_CSV = Path(__file__).resolve().parents[1] / "datasets" / "ikea_furniture.csv"


def load_ikea_data(csv_path=None, room_type="Living Room"):
    csv_path = Path(csv_path) if csv_path is not None else DEFAULT_IKEA_CSV
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["length", "width"])
    categories = ROOM_CATEGORIES.get(room_type, [])

    # The dataset labels are inconsistent, so match by both room and category.
    df = df[df["room"].isin(categories) | df["category"].isin(categories)]
    df = df[~df["category"].isin(EXCLUDED_CATEGORIES)]
    df = df[df["length"] >= 40]
    df["grid_length"] = (df["length"] / 30).apply(lambda x: max(1, int(x)))
    df["grid_width"] = (df["width"] / 30).apply(lambda x: max(1, int(x)))
    df["grid_area"] = df["grid_length"] * df["grid_width"]
    df = df.sort_values(["grid_area", "length", "width"], ascending=False).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No IKEA furniture rows matched room type '{room_type}'.")
    return df


class IKEAFurnitureEnv(AECEnv):
    metadata = {"name": "ikea_furniture_env_v0"}

    def __init__(
        self,
        room_size=10,
        num_furniture=5,
        room_type="Living Room",
        target_utilization=0.5,
        max_turns=None,
    ):
        super().__init__()
        self.room_size = room_size
        self.num_furniture = num_furniture
        self.room_type = room_type
        self.target_utilization = target_utilization
        self.max_turns = max_turns or (room_size * room_size * 2)

        self.furniture_df = load_ikea_data(room_type=self.room_type)
        print(f"Loaded {len(self.furniture_df)} real IKEA furniture items for {self.room_type}!")

        self.possible_agents = ["layout_agent", "style_agent"]
        self.agents = self.possible_agents[:]
        self.observation_spaces = {
            agent: spaces.Box(
                low=0,
                high=1,
                shape=(room_size, room_size),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(room_size * room_size)
            for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def get_random_furniture(self):
        item = self.furniture_df.sample(1).iloc[0]
        return {
            "category": item["category"],
            "room": item["room"],
            "length": int(item["grid_length"]),
            "width": int(item["grid_width"]),
            "area": int(item["grid_area"]),
            "real_length": item["length"],
            "real_width": item["width"],
        }

    def _current_utilization(self):
        return float(np.sum(self.room) / (self.room_size ** 2))

    def _placement_is_valid(self, x, y, width, height):
        if x + width > self.room_size or y + height > self.room_size:
            return False
        return not np.any(self.room[x:x + width, y:y + height] != 0)

    def _iter_orientations(self, length, width):
        orientations = [(int(length), int(width))]
        if int(length) != int(width):
            orientations.append((int(width), int(length)))
        return orientations

    def _placement_score(self, x, y, width, height, preferred_xy):
        px, py = preferred_xy
        x0 = max(0, x - 1)
        y0 = max(0, y - 1)
        x1 = min(self.room_size, x + width + 1)
        y1 = min(self.room_size, y + height + 1)
        neighborhood = self.room[x0:x1, y0:y1]
        adjacent_occupied = float(np.sum(neighborhood))
        wall_contacts = int(x == 0) + int(y == 0)
        wall_contacts += int(x + width == self.room_size) + int(y + height == self.room_size)
        distance_penalty = abs(x - px) + abs(y - py)
        return (adjacent_occupied * 3.0) + (wall_contacts * 2.0) - (distance_penalty * 0.25)

    def _find_best_position(self, width, height, preferred_xy):
        preferred_x, preferred_y = preferred_xy
        if self._placement_is_valid(preferred_x, preferred_y, width, height):
            return preferred_x, preferred_y

        best_position = None
        best_score = float("-inf")
        for x in range(self.room_size - width + 1):
            for y in range(self.room_size - height + 1):
                if not self._placement_is_valid(x, y, width, height):
                    continue
                score = self._placement_score(x, y, width, height, preferred_xy)
                if score > best_score:
                    best_score = score
                    best_position = (x, y)
        return best_position

    def _choose_candidate_rows(self):
        free_cells = int((self.room_size ** 2) - np.sum(self.room))
        utilization = self._current_utilization()
        candidates = self.furniture_df[self.furniture_df["grid_area"] <= free_cells]
        if candidates.empty:
            return candidates

        if utilization < 0.2:
            target_area = max(4, int(free_cells * 0.18))
        elif utilization < 0.35:
            target_area = max(3, int(free_cells * 0.12))
        else:
            target_area = max(1, int(free_cells * 0.08))

        ranked = candidates.assign(
            target_gap=(candidates["grid_area"] - target_area).abs(),
            area_bias=-candidates["grid_area"],
        )
        return ranked.sort_values(["target_gap", "area_bias"], ascending=[True, True])

    def _plan_placement(self, preferred_xy):
        candidates = self._choose_candidate_rows()
        if candidates.empty:
            return None

        for item in candidates.itertuples(index=False):
            base_furniture = {
                "category": item.category,
                "room": item.room,
                "length": int(item.grid_length),
                "width": int(item.grid_width),
                "area": int(item.grid_area),
                "real_length": item.length,
                "real_width": item.width,
            }
            for width, height in self._iter_orientations(item.grid_length, item.grid_width):
                if width > self.room_size or height > self.room_size:
                    continue
                position = self._find_best_position(width, height, preferred_xy)
                if position is None:
                    continue
                furniture = dict(base_furniture)
                furniture["length"] = int(width)
                furniture["width"] = int(height)
                furniture["x"], furniture["y"] = map(int, position)
                return furniture
        return None

    def _finish_episode(self, termination=True):
        target = self.terminations if termination else self.truncations
        for agent in self.possible_agents:
            target[agent] = True

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.room = np.zeros((self.room_size, self.room_size))
        self.room_labels = [["" for _ in range(self.room_size)] for _ in range(self.room_size)]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = defaultdict(int, {"layout_agent": 0, "style_agent": 0})
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[0]
        self.placements = {"layout_agent": 0, "style_agent": 0}
        self.placed_furniture = []
        self.total_steps = 0
        self.failed_turns = 0
        return self.observe(self.agent_selection), {}

    def observe(self, agent):
        return self.room.copy()

    def step(self, action):
        current_agent = self.agent_selection

        if self.terminations.get(current_agent, False) or self.truncations.get(current_agent, False):
            self._was_dead_step(action)
            return

        x = action // self.room_size
        y = action % self.room_size
        self.rewards = {"layout_agent": 0, "style_agent": 0}

        furniture = self._plan_placement((x, y))
        if furniture is None:
            self.rewards[current_agent] = -5
            self.failed_turns += 1
        else:
            fw = furniture["length"]
            fh = furniture["width"]
            start_x = furniture["x"]
            start_y = furniture["y"]
            self.room[start_x:start_x + fw, start_y:start_y + fh] = 1

            placed_area = fw * fh
            self.rewards[current_agent] = max(1.0, placed_area * 0.2)
            self.placements[current_agent] += 1
            self.failed_turns = 0
            self.placed_furniture.append({
                "agent": current_agent,
                "room": furniture["room"],
                "category": furniture["category"],
                "x": start_x,
                "y": start_y,
                "width": fw,
                "height": fh,
                "area": placed_area,
                "real_size": f"{furniture['real_length']}x{furniture['real_width']}cm",
            })

        self.total_steps += 1
        utilization = self._current_utilization()
        if utilization >= self.target_utilization:
            self.rewards[current_agent] += 5
            self._finish_episode(termination=True)
        elif utilization >= 0.3:
            self.rewards[current_agent] += 2
        elif self.failed_turns >= len(self.possible_agents) * 2:
            self._finish_episode(termination=True)
        elif self.total_steps >= self.max_turns:
            self._finish_episode(termination=False)

        self._cumulative_rewards[current_agent] += self.rewards[current_agent]

        if not all(
            self.terminations.get(agent, False) or self.truncations.get(agent, False)
            for agent in self.possible_agents
        ):
            self.agent_selection = (
                self.agents[1] if current_agent == self.agents[0] else self.agents[0]
            )

    def render(self):
        print(f"\nRoom Layout ({self.room_size}x{self.room_size}):")
        print(self.room)
        utilization = self._current_utilization() * 100
        print(f"Space Utilization: {utilization:.1f}%")
        print(f"Target Utilization: {self.target_utilization * 100:.1f}%")
        print(f"\nFurniture Placed:")
        for item in self.placed_furniture:
            print(
                f"  [{item['agent']}] {item['category']} at ({item['x']},{item['y']}) "
                f"size {item['width']}x{item['height']} grid units "
                f"({item['real_size']})"
            )