from ikea_furniture_env import IKEAFurnitureEnv

room_types = ["Living Room", "Bedroom", "Kitchen", "Bathroom", "Dining Room"]

print("=== IKEA MARL Furniture Placement Test ===\n")

for episode, room_type in enumerate(room_types):
    env = IKEAFurnitureEnv(room_size=15, num_furniture=5, room_type=room_type)
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

            if all(env.terminations.get(a, False) or
                   env.truncations.get(a, False)
                   for a in env.possible_agents):
                done = True

        except Exception:
            done = True

    print(f"\nEpisode {episode + 1} — {room_type}:")
    env.render()
    print("-" * 50)