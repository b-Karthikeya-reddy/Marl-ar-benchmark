import numpy as np

class FurnitureEnv:
    """
    Simple 2-agent furniture placement environment.
    - Room is represented as a grid
    - Agent 1 (Layout Agent): Places furniture in the room
    - Agent 2 (Style Agent): Rates the placement
    """
    
    def __init__(self, room_width=10, room_height=10):
        self.room_width = room_width
        self.room_height = room_height
        self.grid = np.zeros((room_width, room_height))
        self.agents = ["layout_agent", "style_agent"]
        self.furniture_placed = []
        self.reset()
    
    def reset(self):
        """Reset the environment"""
        self.grid = np.zeros((self.room_width, self.room_height))
        self.furniture_placed = []
        print("Room reset! Empty grid ready.")
        return self.grid
    
    def place_furniture(self, x, y, width, height):
        """Layout agent places furniture at position x,y"""
        # Check if furniture fits
        if x + width > self.room_width or y + height > self.room_height:
            print(f"Layout Agent: Furniture doesn't fit at ({x},{y})!")
            return -1  # Negative reward
        
        # Check if space is already occupied
        if np.any(self.grid[x:x+width, y:y+height] != 0):
            print(f"Layout Agent: Space already occupied at ({x},{y})!")
            return -1  # Negative reward
        
        # Place furniture
        self.grid[x:x+width, y:y+height] = 1
        self.furniture_placed.append({
            'x': x, 'y': y, 
            'width': width, 'height': height
        })
        print(f"Layout Agent: Placed furniture at ({x},{y}) size {width}x{height}")
        return 1  # Positive reward
    
    def style_score(self):
        """Style agent rates the current layout"""
        if not self.furniture_placed:
            return 0
        
        # Simple scoring: reward for space utilization
        occupied = np.sum(self.grid)
        total = self.room_width * self.room_height
        utilization = occupied / total
        
        score = utilization * 100
        print(f"Style Agent: Layout score = {score:.1f}%")
        return score
    
    def render(self):
        """Display the room grid"""
        print("\nCurrent Room Layout:")
        print("0 = empty, 1 = furniture")
        print(self.grid)

# Test the environment
env = FurnitureEnv(room_width=10, room_height=10)

print("=== Testing 2-Agent Furniture Placement ===\n")

# Layout agent places furniture
env.place_furniture(0, 0, 3, 2)  # Sofa
env.place_furniture(4, 4, 2, 2)  # Coffee table
env.place_furniture(7, 0, 2, 3)  # Bookshelf

# Style agent rates the layout
env.style_score()

# Render the room
env.render()