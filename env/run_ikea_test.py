from ikea_furniture_env import IKEAFurnitureEnv

env = IKEAFurnitureEnv(room_size=10, num_furniture=5)

print("=== IKEA MARL Furniture Placement Test ===\n")

for episode in range(3):
    env.reset()
    done = False
    
    while not done:
        try:
            agent = env.agent_selection
            obs, reward, term, trunc, info = env.last()
            
            if term or trunc:
                action = None
            else:
                action = env.action_space(agent).sample()
            
            env.step(action)
            
            # Check if all agents are done
            if all(env.terminations.get(a, False) or 
                   env.truncations.get(a, False) 
                   for a in env.possible_agents):
                done = True
                
        except Exception:
            done = True
    
    print(f"\nEpisode {episode + 1}:")
    env.render()
    print("-" * 50)