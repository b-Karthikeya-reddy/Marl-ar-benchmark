from furniture_env import FurnitureEnv
import numpy as np

env = FurnitureEnv(room_size=10)
results = []

for episode in range(5):
    env.reset()
    episode_rewards = {"layout_agent": 0, "style_agent": 0}

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            env.step(None)
        else:
            action = env.action_space(agent).sample()
            env.step(action)
            # Only capture reward when agent actually acted
            episode_rewards[agent] += env.rewards[agent]

    results.append(episode_rewards)
    print(f"Episode {episode + 1}: {episode_rewards}")
    env.render()

print("\nTest Results")
avg_layout = np.mean([r["layout_agent"] for r in results])
avg_style = np.mean([r["style_agent"] for r in results])
print(f"Average Layout Agent Reward: {avg_layout:.2f}")
print(f"Average Style Agent Reward: {avg_style:.2f}")