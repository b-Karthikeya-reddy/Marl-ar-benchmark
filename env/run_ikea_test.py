import matplotlib.pyplot as plt
from ikea_furniture_env import IKEAFurnitureEnv
import numpy as np
from collections import Counter

ROOMS = ["Living Room", "Bedroom", "Kitchen", "Bathroom", "Dining Room"]

#okay so this is added bc i have to map both datasets together for the cosine similarity score. so i was able to essentially merge
#both of them together by making a map and just filtering things. for example, two seated sofas falls under sofas and armchairs, so
#do three seated sofas. 
CATEGORY_MAP = {
    "Two-Seated Sofas": "Sofas & armchairs", "Three-Seated Sofas": "Sofas & armchairs",
    "Armchairs": "Sofas & armchairs", "Coffee Tables": "Tables & desks", "Side Tables": "Tables & desks",
    "TV Benches": "TV & media furniture", "Bookcases": "Bookcases & shelving units", "Shelving Units": "Bookcases & shelving units",
    "Beds": "Beds", "Wardrobes": "Wardrobes", "Dining Tables": "Tables & desks", "Dining Chairs": "Chairs",
    "Bathroom Storage": "Cabinets & cupboards", "BathroomBaseCabinets": "Cabinets & cupboards", "Bathroom High Cabinet": "Cabinets & cupboards",
    "BathroomSinkAccessories": "Cabinets & cupboards", "Kitchen Cabinets": "Cabinets & cupboards", "Kitchen Islands and Trolleys": "Trolleys",
    "Worktops and Worktop Accessories": "Tables & desks", "Kitchen Splashbacks": "Cabinets & cupboards"
}

ALL_CATEGORIES = [
    "Tables & desks", "Bookcases & shelving units", "Chairs", "Sofas & armchairs", "Cabinets & cupboards", "Wardrobes",
    "Outdoor furniture", "Beds", "TV & media furniture", "Chests of drawers & drawer units", "Children's furniture", "Nursery furniture",
    "Bar furniture", "Trolleys", "Café furniture", "Sideboards, buffets & console tables", "Room dividers"
]

#this is probably subject to change, this is just preference. will have to find a way to make this more ideal by doing more research
IDEAL_ROOMS_AND_PIECES = {
    "Living Room": {
        "Sofas & armchairs": 2,
        "Tables & desks": 1,
        "TV & media furniture": 3,
        "Bookcases & shelving units": 2,
    },

    "Bedroom": {
        "Beds": 1,
        "Wardrobes": 2,
        "Chests of drawers & drawer units": 3,
    },

    "Kitchen": {
        "Cabinets & cupboards": 5,
        "Tables & desks": 2,
        "Chairs": 4,
        "Trolleys": 1,
    },

    "Bathroom": {
        "Cabinets & cupboards": 2,
        "Trolleys": 1,
    },

    "Dining Room": {
        "Tables & desks": 1,
        "Chairs": 4,
        "Sideboards, buffets & console tables": 1,
    }
}


#this is the main cosine similarity scorer function. 
#a high cosine similarity score means that the correct furniture was placed accordingly (could have some better improvements)
#a low cosinse similarity score means that the wrong furniture was placed and lots of room for improvement (dataset related)
class CosineSimilarityScore:
    def __init__(self):
        self.categories = ALL_CATEGORIES

    def normalize_category(self, category):
        return CATEGORY_MAP.get(category, category)

    def build_vector(self, items_dict):
        vec = np.zeros(len(self.categories))
        for i, cat in enumerate(self.categories):
            vec[i] = items_dict.get(cat, 0)
        return vec

    def cosine_similarity(self, v1, v2):
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return 0.0
        return np.dot(v1, v2) / denom

    def score_room(self, room_type, placed_items):
        placed_counts = Counter()
        for item in placed_items:
            if isinstance(item, dict):
                raw_cat = item.get("category", "")
            else:
                continue
            normalized = self.normalize_category(raw_cat)
            placed_counts[normalized] += 1
        placed_vector = self.build_vector(placed_counts)
        ideal_vector = self.build_vector(IDEAL_ROOMS_AND_PIECES[room_type])
        return self.cosine_similarity(placed_vector, ideal_vector)
results = []
scorer = CosineSimilarityScore()
print("{}Ikea Furniture Placement Test{}".format('\033[1m', '\033[0m'))

for room in ROOMS:
    env = IKEAFurnitureEnv(
        room_type=room,
        room_size=15,
        budget_limit=1000
    )
    env.reset()
    done = False
    rewards = {"layout_agent": 0, "style_agent": 0, "budget_agent": 0}

    while not done:
        agent = env.agent_selection
        _, _, term, trunc, _ = env.last()
        action = None if term or trunc else env.action_space(agent).sample()
        env.step(action)
        for k in rewards:
            rewards[k] += env.rewards.get(k, 0)
        done = all(
            env.terminations[a] or env.truncations[a]
            for a in env.possible_agents
        )
    similarity = scorer.score_room(room, env.placed_furniture)
    
    print(f"{room}")
    env.render()
    results.append({
        "room": room,
        "layout": rewards["layout_agent"],
        "style": rewards["style_agent"],
        "budget": rewards["budget_agent"],
        "utilization": env.utilization() * 100,
        "spend": env.total_spend,
        "cosine_similarity": similarity
    })
    print('\n')
    print('{}SUMMARY{}'.format('\033[1m', '\033[0m'))
    print(f"Layout Reward: {rewards['layout_agent']:.2f}")
    print(f"Style Reward: {rewards['style_agent']:.2f}")
    print(f"Budget Reward: {rewards['budget_agent']:.2f}")
    print(f"Spend: ${env.total_spend:.2f}")
    print(f"Utilization: {env.utilization() * 100:.1f}%")
    print("----------------------------------------------")
    print(f"Cosine Similarity: {similarity:.3f}")
    print("----------------------------------------------")

x = range(len(results))
labels = [r["room"] for r in results]
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("IKEA Furniture Placements MARL Results")

ax[0, 0].bar(x, [r["layout"] for r in results])
ax[0, 0].set_title("Layout Agent Rewards")
ax[0, 1].bar(x, [r["style"] for r in results])
ax[0, 1].set_title("Style Agent Rewards")
ax[1, 0].bar(x, [r["budget"] for r in results])
ax[1, 0].set_title("Budget Agent Rewards")
ax[1, 1].bar(x, [r["utilization"] for r in results])
ax[1, 1].axhline(50, linestyle="--")
ax[1, 1].set_title("Utilization %")

for a in ax.flat:
    a.set_xticks(x)
    a.set_xticklabels(labels, rotation=15)

plt.tight_layout()
plt.savefig("reward_visualization.png")
plt.show()

print('\n')
print("{}Budget Summary{}".format('\033[1m', '\033[0m'))
for r in results:
    print(f"{r['room']}: ${r['spend']:.2f}")
    
print('\n')
print("{}Cosine Similarity Score Summary{}".format('\033[1m', '\033[0m'))
for r in results:
    print(f"{r['room']}: {r['cosine_similarity']:.3f}")