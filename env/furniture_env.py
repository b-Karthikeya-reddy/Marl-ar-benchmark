import numpy as np
from pettingzoo import AECEnv
from gymnasium import spaces

class FurnitureEnv(AECEnv):
    metadata = {"name": "furniture_env_v0"}

    def __init__(self, room_size=10):
        super().__init__()
        self.room_size = room_size
        
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
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[0]
        return self.observe(self.agent_selection), {}

    def observe(self, agent):
        return self.room.copy()

    def step(self, action):
        current_agent = self.agent_selection

        # If agent is done, skip
        if self.terminations[current_agent] or self.truncations[current_agent]:
            self._was_dead_step(action)
            return

        # Convert action to (x, y) position
        x = action // self.room_size
        y = action % self.room_size

        # Reset rewards each step
        self.rewards = {"layout_agent": 0, "style_agent": 0}

        # Hard constraint — can't place where furniture already exists
        if self.room[x][y] == 1:
            self.rewards[current_agent] = -10
        else:
            self.room[x][y] = 1
            self.rewards[current_agent] = 1

        # Space utilization bonus
        utilization = np.sum(self.room) / (self.room_size ** 2)
        if 0.3 < utilization < 0.7:
            self.rewards[current_agent] += 2

        # Mark only current agent as done
        self.terminations[current_agent] = True

        # Move to next agent
        self.agent_selection = (
            self.agents[1]
            if current_agent == self.agents[0]
            else self.agents[0]
        )

    def render(self):
        print(f"\nRoom Layout ({self.room_size}x{self.room_size}):")
        print(self.room)