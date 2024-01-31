import numpy as np
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.envs import PortfolioEnv

# Example data (replace with your own)
prices = np.random.randn(100, 5)  # Historical price data for 5 assets over 100 time steps

# Custom Gym environment for portfolio optimization
env = PortfolioEnv(prices=prices)

# Define and train the RL agent (A2C in this example)
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Use the trained agent to make portfolio decisions
observation = env.reset()
for _ in range(len(prices)):
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)

# The 'action' variable contains the portfolio weights chosen by the RL agent at each time step
