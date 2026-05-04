from furniture_env import FurnitureEnv
import numpy as np

env = FurnitureEnv(room_size=10, num_furniture=5)
results = []

for episode in range(5):
    env.reset()
    episode_rewards = {"layout_agent": 0, "style_agent": 0}
    constraint_violations = 0

    for _ in range(env.num_furniture * 2):
        agent = env.agent_selection

        if env.terminations[agent]:
            env.step(None)
            continue

        action = env.action_space(agent).sample()
        env.step(action)

        if env.rewards[agent] == -10:
            constraint_violations += 1

        episode_rewards[agent] += env.rewards[agent]

    results.append(episode_rewards)
    print(f"Episode {episode + 1}: Rewards={episode_rewards} | Violations={constraint_violations}")
    env.render()

print("\nTest Results")
avg_layout = np.mean([r["layout_agent"] for r in results])
avg_style = np.mean([r["style_agent"] for r in results])
print(f"Average Layout Agent Reward: {avg_layout:.2f}")
print(f"Average Style Agent Reward: {avg_style:.2f}")