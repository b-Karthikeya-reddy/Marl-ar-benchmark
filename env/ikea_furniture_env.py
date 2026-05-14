import numpy as np
import pandas as pd
from pettingzoo import AECEnv
from gymnasium import spaces
from pathlib import Path
from collections import defaultdict

ROOM_CATEGORIES = {
    "Living Room": [
        "Three-Seated Sofas", "Two-Seated Sofas", "ArmChairs", "FootStools",
        "Cabinets and Display Cabinets", "Coffee and Side Tables", "GRONLID System",
        "VIMLE System", "EKET System", "HAVSTA Systems", "PLATSA Combinations",
        "VALENTUNA System", "LIDHULT System", "SODERHAMN System", "KUNGSHAMN System",
        "Outdoor Sofas", "Sofas", "Armchairs & chaise longues", "TV & media furniture",
        "Sideboards, buffets & console tables", "Coffee tables", 
    ],

    "Bedroom": [
        "Beds", "Mattresses", "Wardrobes", "Wordrobes", "Chest of Drawers",
        "Bedside Tables", "Dressing Tables", "Clothes Organizer", "Open Storage Wordrobes",
        "Freestanding Wardrobes", "PAX Wardrobes", "PLATSA Modular Storage System",
        "Drawer Units Storage Cabinets", "Bed frames", "Beds & mattresses", "Bedside tables", "Chest of drawers & drawer units",
    ],

    "Kitchen": [
        "Cookware", "Frying Pans and Woks", "Kitchen Appliances", "Tableware",
        "DinnerWare", "Serveware", "Cooking Utensils", "Kitchen Islands and Trolleys",
        "Kitchen Taps and Sinks", "DishWasher", "Kithcen and Workshops", "Modular Kitchen", "Bar furniture", "Kitchen islands & trolleys",
    ],

    "Bathroom": [
        "BathroomStorage", "BathroomAccessories", "Bathroom Wall Cabinets", "BathroomSinkAccessories",
        "ToiletAccessories", "Towels&Bathmats", "VanityUnits", "Towel40times70", "Bathroom storage", "Bathroom vanities",
    ],

    "Dining Room": [
        "DinnerWare", "Tableware", "Serveware", "Kitchen Islands and Trolleys", "Cooking Utensils",
        "Dining tables", "Chairs", "Bar furniture",
    ],
}

EXCLUDED_CATEGORIES = [
    "Cushion and Cushion Covers", "BedSpreads and Throws", "Sofa Covers", "Sofa Accessories and Legs",
    "Decoration", "PaperShop", "Babies Tableware", "BabyAccessories", "ChildSafety", "Lighting",
    "Rugs, mats & flooring", "Curtains & blinds", "Bed textiles", "Bathroom textiles",
]

OLD_DATASET = (Path(__file__).resolve().parents[1] / "datasets" / "ikea_furniture.csv")

NEW_DATASET = (Path(__file__).resolve().parents[1] / "datasets" / "ikea_new_w_prices.csv")

def load_furniture(room_type):
    old_df = pd.read_csv(OLD_DATASET)
    new_df = pd.read_csv(NEW_DATASET)
    old_df = old_df.rename(columns={
        "length": "depth_cm",
        "width": "width_cm"
    })
    old_df["name"] = old_df["category"]
    old_df["price"] = np.nan
    old_df = old_df[[
        "room",
        "category",
        "name",
        "depth_cm",
        "width_cm",
        "price"
    ]]

    new_df = new_df.rename(columns={
        "depth": "depth_cm",
        "width": "width_cm"
    })
    new_df["room"] = new_df["category"]
    new_df = new_df[[
        "room",
        "category",
        "name",
        "depth_cm",
        "width_cm",
        "price"
    ]]

    df = pd.concat([old_df, new_df], ignore_index=True)
    df = df.dropna(subset=["depth_cm", "width_cm"])
    categories = ROOM_CATEGORIES.get(room_type, [])

    mask = (df["room"].isin(categories) | df["category"].isin(categories))
    df = df[mask]
    df = df[~df["category"].isin(EXCLUDED_CATEGORIES)]

    df["grid_w"] = (
        df["width_cm"] / 30
    ).astype(int).clip(lower=1)

    df["grid_h"] = (
        df["depth_cm"] / 30
    ).astype(int).clip(lower=1)

    df["area"] = df["grid_w"] * df["grid_h"]
    return df.sort_values("area", ascending=False)

class IKEAFurnitureEnv(AECEnv):
    metadata = {"name": "ikea_env"}
    
    def __init__(
        self,
        room_type="Living Room",
        room_size=15,
        budget_limit=2000,
        target_utilization=0.5,
    ):
        super().__init__()

        self.room_type = room_type
        self.room_size = room_size
        self.budget_limit = budget_limit
        self.target_utilization = target_utilization

        self.furniture = load_furniture(room_type)
        print(
            f"\nLoaded {len(self.furniture)} "
            f"IKEA furniture items "
            f"for {room_type}"
        )

        self.possible_agents = ["layout_agent", "style_agent", "budget_agent"]
        self.agents = self.possible_agents[:]

        obs_space = spaces.Box(
            low=0,
            high=1,
            shape=(room_size, room_size),
            dtype=np.float32,
        )

        act_space = spaces.Discrete(room_size * room_size)
        self.observation_spaces = {
            a: obs_space for a in self.possible_agents
        }

        self.action_spaces = {
            a: act_space for a in self.possible_agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.room = np.zeros((self.room_size, self.room_size))
        self.rewards = {a: 0 for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}
        self._cumulative_rewards = defaultdict(float)
        self.agent_selection = self.possible_agents[0]
        self.total_spend = 0
        self.steps = 0
        self.placed_furniture = []
        return self.observe(self.agent_selection), {}

    def observe(self, agent):
        return self.room.copy()

    def utilization(self):
        return np.sum(self.room) / (self.room_size ** 2)

    def valid_position(self, x, y, w, h):
        if x + w > self.room_size:
            return False

        if y + h > self.room_size:
            return False

        return not np.any(
            self.room[x:x+w, y:y+h]
        )

    def place_furniture(self, x, y, agent):
        items = self.furniture.copy()

        if agent == "budget_agent":
            items = items.sort_values("price")

        for item in items.itertuples():
            w = int(item.grid_w)
            h = int(item.grid_h)

            if self.valid_position(x, y, w, h):
                self.room[x:x+w, y:y+h] = 1

                price = (
                    item.price
                    if not np.isnan(item.price)
                    else 0
                )

                self.total_spend += price

                self.placed_furniture.append({
                    "agent": agent,
                    "name": item.name,
                    "category": item.category,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "depth": item.depth_cm,
                    "width": item.width_cm,
                    "price": price,
                })

                return w * h
        return 0

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent]:
            self._was_dead_step(action)
            return

        self.rewards = {a: 0 for a in self.possible_agents}
        x = action // self.room_size
        y = action % self.room_size
        area = self.place_furniture(x, y, agent)

        if area == 0:
            reward = -1
        else:
            reward = area * 0.5
            if self.total_spend > self.budget_limit:
                reward -= 5
            if self.utilization() >= self.target_utilization:
                reward += 10

        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward
        self.steps += 1

        done = (
            self.utilization() >= self.target_utilization
            or self.steps >= 100
        )
        if done:
            for a in self.possible_agents:
                self.terminations[a] = True

        index = self.possible_agents.index(agent)
        self.agent_selection = self.possible_agents[
            (index + 1) % len(self.possible_agents)
        ]

    def render(self):
        print(f"Room Layout ({self.room_size}x{self.room_size}):")
        print(self.room)

        print(
            f"\nSpace Utilization: "
            f"{self.utilization() * 100:.1f}%"
        )

        print(
            f"Target Utilization: "
            f"{self.target_utilization * 100:.1f}%"
        )

        print(
            f"Total Spend: "
            f"${self.total_spend:.2f}"
        )

        print("Furniture Placed:")

        for item in self.placed_furniture:
            print(
                f"  [{item['agent']}] "
                f"{item['category']} "
                f"at ({item['x']},{item['y']}) "
                f"size {item['w']}x{item['h']} "
                f"({item['depth']}x{item['width']}cm) "
                f"| ${item['price']:.2f}"
            )