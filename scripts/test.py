import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import Agent
from tasks import task_from_argv

task = task_from_argv(sys.argv)

agent = Agent(eval=True)
agent.model.load_checkpoint()

agent.test(task, render_mode="human", delay=0.05)
