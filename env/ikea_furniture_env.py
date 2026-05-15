import numpy as np
import pandas as pd
from pettingzoo import ParallelEnv
from gymnasium import spaces
from pathlib import Path

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
    
    df = df[df["room"].isin(categories) | df["category"].isin(categories)]
    df["grid_w"] = (df["width_cm"] / 30).astype(int).clip(lower=1)
    df["grid_h"] = (df["depth_cm"] / 30).astype(int).clip(lower=1)
    df["area"] = df["grid_w"] * df["grid_h"]
    return df.sort_values("area", ascending=False)


class IKEAFurnitureEnv(ParallelEnv):
    metadata = {"name": "ikea_env"}
    render_mode = "human"
    
    def __init__(self, room_type="Living Room", room_size=15, budget_limit=2000, target_utilization=0.5):
        super().__init__()

        self.room_type = room_type
        self.room_size = room_size
        self.budget_limit = budget_limit
        self.target_utilization = target_utilization

        self.furniture = load_furniture(room_type)
        self.possible_agents = ["layout_agent", "style_agent", "budget_agent"]
        self.observation_spaces = {
            a: spaces.Box(0, 1, (room_size, room_size), np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.Discrete(room_size * room_size)
            for a in self.possible_agents
        }
        self.reset()

    def reset(self, seed=None, options=None):
        self.room = np.zeros((self.room_size, self.room_size), dtype=np.float32)
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.total_spend = 0
        self.steps = 0
        return {a: self.observe() for a in self.agents}, self.infos

    def observe(self):
        return self.room.copy()

    def util(self):
        return float(np.mean(self.room))

    def valid(self, x, y, w, h):
        if x + w > self.room_size or y + h > self.room_size:
            return False
        return not np.any(self.room[x:x+w, y:y+h])

    def place(self, x, y):
        for item in self.furniture.itertuples():
            w, h = int(item.grid_w), int(item.grid_h)
            if self.valid(x, y, w, h):
                self.room[x:x+w, y:y+h] = 1
                price = 0 if np.isnan(item.price) else item.price
                self.total_spend += price
                return w * h
        return 0

    def step(self, actions):
        rewards = {}
        for agent, action in actions.items():
            x = action // self.room_size
            y = action % self.room_size
            area = self.place(x, y)
            reward = area * 0.5
            if area == 0:
                reward -= 1
            if self.total_spend > self.budget_limit:
                reward -= 5
            if self.util() >= self.target_utilization:
                reward += 10
            rewards[agent] = reward
            
        self.steps += 1
        done = self.util() >= self.target_utilization or self.steps >= 100

        if done:
            self.agents = []
        obs = {a: self.observe() for a in self.possible_agents if a in self.agents}
        terminations = {a: done for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}
        return obs, rewards, terminations, truncations, infos

    def render(self):
        print(self.room)
        print("Util:", self.util())
        print("Spend:", self.total_spend)