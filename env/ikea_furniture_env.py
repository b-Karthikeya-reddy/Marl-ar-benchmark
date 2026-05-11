import numpy as np
import pandas as pd
from pettingzoo import AECEnv
from gymnasium import spaces
from collections import defaultdict

# Load IKEA furniture data
ROOM_CATEGORIES = {
    "Living Room": [
        "Three-Seated Sofas", "Two-Seated Sofas", "ArmChairs", "FootStools",
        "Cabinets and Display Cabinets", "Coffee and Side Tables",
        "GRONLID System", "VIMLE System", "EKET System", "HAVSTA Systems",
        "PLATSA Combinations", "VALENTUNA System", "LIDHULT System",
        "SODERHAMN System", "KUNGSHAMN System", "Outdoor Sofas"
    ],
    "Bedroom": [
        "Beds", "Mattresses", "Wardrobes", "Wordrobes",
        "Chest of Drawers", "Bedside Tables", "Dressing Tables",
        "Clothes Organizer", "Open Storage Wordrobes",
        "Freestanding Wardrobes", "PAX Wardrobes",
        "PLATSA Modular Storage System", "Drawer Units Storage Cabinets"
    ],
    "Kitchen": [
        "Cookware", "Frying Pans and Woks", "Kitchen Appliances",
        "Tableware", "DinnerWare", "Serveware", "Cooking Utensils",
        "Kitchen Islands and Trolleys", "Kitchen Taps and Sinks",
        "DishWasher", "Kithcen and Workshops"
    ],
    "Bathroom": [
        "BathroomStorage", "BathroomAccessories", "Bathroom Wall Cabinets",
        "BathroomSinkAccessories", "ToiletAccessories",
        "Towels&Bathmats", "VanityUnits", "Towel40times70"
    ],
    "Home Office": [
        "Office Desks", "Desks Chairs", "Desks Table and Legs",
        "Wall Shelves", "Storage Boxes Baskets", "Paper & Media Organizers",
        "Drawer Units Storage Cabinets", "Home Office Lighting"
    ],
    "Dining Room": [
        "DinnerWare", "Tableware", "Serveware",
        "Kitchen Islands and Trolleys", "Cooking Utensils"
    ]
}

EXCLUDED_CATEGORIES = [
    "Cushion and Cushion Covers", "BedSpreads and Throws",
    "Sofa Covers", "Sofa Accessories and Legs",
    "Decoration", "PaperShop", "Babies Tableware",
    "BabyAccessories", "ChildSafety"
]

def load_ikea_data(csv_path="../datasets/ikea_furniture.csv", room_type="Living Room"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['length', 'width'])
    
    # Filter by mapped categories
    categories = ROOM_CATEGORIES.get(room_type, [])
    df = df[df['room'].isin(categories)]
    
    # Remove excluded categories
    df = df[~df['category'].isin(EXCLUDED_CATEGORIES)]
    
    # Filter out tiny items
    df = df[df['length'] >= 40]
    
    df['grid_length'] = (df['length'] / 30).apply(lambda x: max(1, int(x)))
    df['grid_width'] = (df['width'] / 30).apply(lambda x: max(1, int(x)))
    return df

class IKEAFurnitureEnv(AECEnv):
    metadata = {"name": "ikea_furniture_env_v0"}

    # ADD room_type to the parameters here
    def __init__(self, room_size=10, num_furniture=5, room_type="Living Room"):
        super().__init__()
        self.room_size = room_size
        self.num_furniture = num_furniture
        self.room_type = room_type
        
        # Pass the room_type to the data loader
        self.furniture_df = load_ikea_data(room_type=self.room_type)
        print(f"Loaded {len(self.furniture_df)} real IKEA furniture items for {self.room_type}!")
        
        self.possible_agents = ["layout_agent", "style_agent"]
        self.agents = self.possible_agents[:]

        self.observation_spaces = {
            agent: spaces.Box(
                low=0, high=1,
                shape=(room_size, room_size),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(room_size * room_size)
            for agent in self.possible_agents
        }

    def get_random_furniture(self):
        """Pick a random furniture item from IKEA dataset"""
        item = self.furniture_df.sample(1).iloc[0]
        return {
            'category': item['category'],
            'room': item['room'],
            'length': item['grid_length'],
            'width': item['grid_width'],
            'real_length': item['length'],
            'real_width': item['width']
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.room = np.zeros((self.room_size, self.room_size))
        self.room_labels = [["" for _ in range(self.room_size)] 
                           for _ in range(self.room_size)]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = defaultdict(int, {"layout_agent": 0, "style_agent": 0})
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[0]
        self.placements = {"layout_agent": 0, "style_agent": 0}
        self.placed_furniture = []
        return self.observe(self.agent_selection), {}

    def observe(self, agent):
        return self.room.copy()

    def step(self, action):
        current_agent = self.agent_selection

        if self.terminations.get(current_agent, False) or \
           self.truncations.get(current_agent, False):
            self._was_dead_step(action)
            return

        x = action // self.room_size
        y = action % self.room_size

        self.rewards = {"layout_agent": 0, "style_agent": 0}

        # Get a real furniture item from IKEA
        furniture = self.get_random_furniture()
        fw = furniture['length']
        fh = furniture['width']

        # Check if furniture fits
        if x + fw > self.room_size or y + fh > self.room_size:
            self.rewards[current_agent] = -5
        elif np.any(self.room[x:x+fw, y:y+fh] != 0):
            self.rewards[current_agent] = -10
        else:
            # Place furniture with real dimensions
            self.room[x:x+fw, y:y+fh] = 1
            self.rewards[current_agent] = 1
            self.placements[current_agent] += 1
            self.placed_furniture.append({
                'agent': current_agent,
                'category': furniture['category'],
                'x': x, 'y': y,
                'width': fw, 'height': fh,
                'real_size': f"{furniture['real_length']}x{furniture['real_width']}cm"
            })

        utilization = np.sum(self.room) / (self.room_size ** 2)
        if 0.3 < utilization < 0.7:
            self.rewards[current_agent] += 2

        self._cumulative_rewards[current_agent] += self.rewards[current_agent]

        if self.placements[current_agent] >= self.num_furniture:
            self.terminations[current_agent] = True

        self.agent_selection = (
            self.agents[1]
            if current_agent == self.agents[0]
            else self.agents[0]
        )

    def render(self):
        print(f"\nRoom Layout ({self.room_size}x{self.room_size}):")
        print(self.room)
        utilization = np.sum(self.room) / (self.room_size ** 2) * 100
        print(f"Space Utilization: {utilization:.1f}%")
        print(f"\nFurniture Placed:")
        for item in self.placed_furniture:
            print(f"  [{item['agent']}] {item['category']} at ({item['x']},{item['y']}) "
                  f"size {item['width']}x{item['height']} grid units "
                  f"({item['real_size']})")