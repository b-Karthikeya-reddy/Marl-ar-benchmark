import matplotlib.pyplot as plt
from ikea_furniture_env import IKEAFurnitureEnv

ROOMS = ["Living Room", "Bedroom", "Kitchen", "Bathroom", "Dining Room"]
results = []
print("{}Ikea Furniture Placement Test{}".format('\033[1m', '\033[0m'))

for room in ROOMS:
    env = IKEAFurnitureEnv(room_type=room, room_size=15, budget_limit=1000)
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

    print(f"{room}")
    env.render()
    results.append({
        "room": room,
        "layout": rewards["layout_agent"],
        "style": rewards["style_agent"],
        "budget": rewards["budget_agent"],
        "utilization": env.utilization() * 100,
        "spend": env.total_spend
    })
    print('\n')
    print('{}SUMMARY{}'.format('\033[1m', '\033[0m'))
    print(f"Layout Reward: {rewards['layout_agent']:.2f}")
    print(f"Style Reward: {rewards['style_agent']:.2f}")
    print(f"Budget Reward: {rewards['budget_agent']:.2f}")
    print(f"Spend: ${env.total_spend:.2f}")
    print(f"Utilization: {env.utilization() * 100:.1f}%")
    print("----------------------------------------------")


#plotting the rewards and agents as professor said too
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

#final print focusing on the budget agent
print('\n')
print("{}Budget Summary{}".format('\033[1m', '\033[0m'))
for r in results:
    print(f"{r['room']}: ${r['spend']:.2f}")