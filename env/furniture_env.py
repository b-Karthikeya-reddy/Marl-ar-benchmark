import numpy as np
from pettingzoo import AECEnv
from gymnasium import spaces

class FurnitureEnv(AECEnv):
    metadata = {"name": "furniture_env_v0"}

    def __init__(self, room_size=10, num_furniture=5):
        super().__init__()
        self.room_size = room_size
        self.num_furniture = num_furniture
        
        #started w the two agents, having a layout one and style one
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

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.room = np.zeros((self.room_size, self.room_size))
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {"layout_agent": 0, "style_agent": 0}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[0]

        self.placements = {"layout_agent": 0, "style_agent": 0}
        return self.observe(self.agent_selection), {}

    def observe(self, agent):
        return self.room.copy()

    def step(self, action):
        current_agent = self.agent_selection

        if self.terminations.get(current_agent, False) or self.truncations.get(current_agent, False):
            self._was_dead_step(action)
            return

        x = action // self.room_size
        y = action % self.room_size

        self.rewards = {"layout_agent": 0, "style_agent": 0}

        if self.room[x][y] == 1:
            self.rewards[current_agent] = -10
        else:
            self.room[x][y] = 1
            self.rewards[current_agent] = 1
            self.placements[current_agent] += 1

        utilization = np.sum(self.room) / (self.room_size ** 2)
        if 0.3 < utilization < 0.7:
            self.rewards[current_agent] += 2

        self._cumulative_rewards[current_agent] += self.rewards[current_agent]

        if self.placements[current_agent] >= self.num_furniture:
            self.terminations[current_agent] = True

        self._cumulative_rewards = {"layout_agent": self._cumulative_rewards.get("layout_agent", 0), "style_agent": self._cumulative_rewards.get("style_agent", 0)}

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
        print(f"Furniture Placed: {int(np.sum(self.room))}/{self.num_furniture * 2} pieces")