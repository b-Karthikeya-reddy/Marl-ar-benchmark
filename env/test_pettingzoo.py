from pettingzoo.butterfly import cooperative_pong_v6

# Initialize the environment
env = cooperative_pong_v6.env()
env.reset()

print("PettingZoo environment successfully set up!")
print("Agents:", env.agents)

# Run a few steps
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    
    env.step(action)

env.close()
print("Environment ran successfully!")