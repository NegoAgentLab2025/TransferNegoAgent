from agent import Agent
from utils import get_utility
import numpy as np

class TimeAgent(Agent):

    def __init__(self, max_round, name, beta = 1):
        super().__init__(max_round=max_round, name=name)
        self.beta = beta

    def concess(self):
        self.s = max(self.u_min, self.u_min + (self.u_max - self.u_min) * (1 - (self.t / self.max_round) ** (1 / self.beta)))

    def reset(self):
        super().reset()
    
    def receive(self, last_action=None):
        super().receive(last_action)

    def act(self):
        return self.gen_offer()