import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ikea_furniture_env import IKEAFurnitureEnv

# Run one episode
env = IKEAFurnitureEnv(room_size=10, num_furniture=5)
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

# Visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Draw room grid
ax.set_xlim(0, env.room_size)
ax.set_ylim(0, env.room_size)
ax.set_facecolor('#f5f5f5')
ax.grid(True, alpha=0.3)

# Colors for each agent
colors = {
    'layout_agent': '#3498db',
    'style_agent': '#e74c3c'
}

# Draw each piece of furniture
for item in env.placed_furniture:
    color = colors[item['agent']]
    rect = patches.Rectangle(
        (item['y'], env.room_size - item['x'] - item['height']),
        item['width'],
        item['height'],
        linewidth=2,
        edgecolor='black',
        facecolor=color,
        alpha=0.7
    )
    ax.add_patch(rect)
    
    # Add label
    ax.text(
        item['y'] + item['width']/2,
        env.room_size - item['x'] - item['height']/2,
        f"{item['category']}\n{item['real_size']}",
        ha='center', va='center',
        fontsize=7, fontweight='bold',
        color='white',
        wrap=True
    )

# Legend
legend_elements = [
    patches.Patch(facecolor='#3498db', label='Layout Agent'),
    patches.Patch(facecolor='#e74c3c', label='Style Agent')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

# Labels
ax.set_title('MARL Furniture Placement - AR Room Visualization', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Room Width (grid units)', fontsize=12)
ax.set_ylabel('Room Length (grid units)', fontsize=12)

# Add utilization text
utilization = sum(env.room.flatten()) / (env.room_size ** 2) * 100
ax.text(0.02, 0.02, f'Space Utilization: {utilization:.1f}%', 
        transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('../progress/room_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization saved to progress/room_visualization.png")