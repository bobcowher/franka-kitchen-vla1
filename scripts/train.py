import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import Agent

# agent = Agent(data_path="dataset/microwave", name="bc_microwave")
agent = Agent(data_path="dataset", name="bc_network")

epochs = 10001
# epochs = 100001

agent.train(epochs=epochs, batch_size=64)
