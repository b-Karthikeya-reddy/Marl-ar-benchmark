from stable_baselines3 import PPO
from supersuit import concat_vec_envs_v1
from supersuit import pettingzoo_env_to_vec_env_v1
from ikea_furniture_env import IKEAFurnitureEnv

env = IKEAFurnitureEnv(room_type="Living Room", room_size=15, budget_limit=1000)
env = pettingzoo_env_to_vec_env_v1(env)
env = concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    gamma=0.99
)

model.learn(total_timesteps=100000)
model.save("mappo_ikea")
print("done w training")