import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent import Agent
from utils import get_utility

LOG_STD_MIN = -20
LOG_STD_MAX = 2

# 多控制层网络结构
class MultiControllerNet(nn.Module):
    # expert_policies：a list, like ["agentlg", "atlas3", ... , "yxagent"], used for labes
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, output_dim, expert_policies):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden1_dim)
        self.fc2 = torch.nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = torch.nn.Linear(hidden2_dim, hidden3_dim)

        for identity in expert_policies:
            # setattr(obj, 'x', y) means that obj.x = y
            setattr(self, 'fc4-{}'.format(identity), nn.Linear(hidden3_dim, hidden4_dim))
            setattr(self, 'fc5-{}'.format(identity), nn.Linear(hidden4_dim, output_dim))

    def forward(self, expert, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(getattr(self, 'fc4-{}'.format(expert))(x))

        output = getattr(self, 'fc5-{}'.format(expert))(x)
        return output


# SAC actor输出的是随机策略，采样自一个正态分布，网络输出正态分布的均值和方差
class ActorMultiControllerNet(nn.Module):
    # expert_policies：a list, like ["agentlg", "atlas3", ... , "yxagent"], used for labes
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, output_dim, expert_policies):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = 1

        self.fc1 = torch.nn.Linear(input_dim, hidden1_dim)
        self.fc2 = torch.nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = torch.nn.Linear(hidden2_dim, hidden3_dim)

        for identity in expert_policies:
            # setattr(obj, 'x', y) means that obj.x = y
            setattr(self, 'fc4-{}'.format(identity), nn.Linear(hidden3_dim, hidden4_dim))
            setattr(self, 'fc5-{}-mu'.format(identity), nn.Linear(hidden4_dim, output_dim))
            setattr(self, 'fc5-{}-std'.format(identity), nn.Linear(hidden4_dim, output_dim))

    def forward(self, expert, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(getattr(self, 'fc4-{}'.format(expert))(x))

        mu = getattr(self, 'fc5-{}-mu'.format(expert))(x)
        std = getattr(self, 'fc5-{}-std'.format(expert))(x)
        return mu, std

    def action(self, expert, x, is_deterministic=True):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(getattr(self, 'fc4-{}'.format(expert))(x))

        mu = getattr(self, 'fc5-{}-mu'.format(expert))(x)
        std = getattr(self, 'fc5-{}-std'.format(expert))(x)        

        log_std = torch.tanh(std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)
        pi = mu + torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(self.device) * std

        pi = torch.tanh(pi) * self.max_action
        mu = torch.tanh(mu) * self.max_action

        if is_deterministic:
            return mu
        else:
            return pi


class DistillationAgent(Agent):
    def __init__(self, max_round, name, expert_policies, device) -> None:
        super().__init__(max_round=max_round, name=name)

        self.device = device
        self.actor = ActorMultiControllerNet(input_dim=7, hidden1_dim=64, hidden2_dim=128, hidden3_dim=64, hidden4_dim=32, output_dim=1, expert_policies=expert_policies).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.Q1_network = MultiControllerNet(input_dim=8, hidden1_dim=64, hidden2_dim=128, hidden3_dim=64, hidden4_dim=32, output_dim=1, expert_policies=expert_policies).to(self.device)
        self.Q1_optimizer = torch.optim.Adam(self.Q1_network.parameters(), lr=1e-4)
        self.Q2_network = MultiControllerNet(input_dim=8, hidden1_dim=64, hidden2_dim=128, hidden3_dim=64, hidden4_dim=32, output_dim=1, expert_policies=expert_policies).to(self.device)
        self.Q2_optimizer = torch.optim.Adam(self.Q2_network.parameters(), lr=1e-4)
        self.V_network = MultiControllerNet(input_dim=7, hidden1_dim=64, hidden2_dim=128, hidden3_dim=64, hidden4_dim=32, output_dim=1, expert_policies=expert_policies).to(self.device)
        self.V_optimizer = torch.optim.Adam(self.V_network.parameters(), lr=1e-4)

        # self.criteria = torch.nn.KLDivLoss(size_average=False)
        self.criteria = torch.nn.MSELoss(reduction='sum')

        self.expert_label = None

    def load(self, filepath):
        self.actor.load_state_dict(torch.load('{}/student_actor.pt'.format(filepath)))
        self.Q1_network.load_state_dict(torch.load('{}/student_Q1.pt'.format(filepath)))
        self.Q2_network.load_state_dict(torch.load('{}/student_Q2.pt'.format(filepath)))
        self.V_network.load_state_dict(torch.load('{}/student_V.pt'.format(filepath)))

    def reset(self):
        super().reset()
        self.update_obs()

    def update_obs(self):
        if len(self.utility_received) >= 3:
            utility_received_3 = self.utility_received[-3]
            utility_received_2 = self.utility_received[-2]
            utility_received_1 = self.utility_received[-1]
        elif len(self.utility_received) >= 2:
            utility_received_3 = 0
            utility_received_2 = self.utility_received[-2]
            utility_received_1 = self.utility_received[-1]
        elif len(self.utility_received) >= 1:

            utility_received_3 = 0
            utility_received_2 = 0
            utility_received_1 = self.utility_received[-1]
        else:
            utility_received_3 = 0
            utility_received_2 = 0
            utility_received_1 = 0
        
        if len(self.utility_proposed) >= 3:
            utility_proposed_3 = self.utility_proposed[-3]
            utility_proposed_2 = self.utility_proposed[-2]
            utility_proposed_1 = self.utility_proposed[-1]
        elif len(self.utility_proposed) >= 2:
            utility_proposed_3 = 1
            utility_proposed_2 = self.utility_proposed[-2]
            utility_proposed_1 = self.utility_proposed[-1]
        elif len(self.utility_proposed) >= 1:
            utility_proposed_3 = 1
            utility_proposed_2 = 1
            utility_proposed_1 = self.utility_proposed[-1]
        else:
            utility_proposed_3 = 1
            utility_proposed_2 = 1
            utility_proposed_1 = 1

        self.obs = [self.t / self.max_round] + [utility_received_3] + [utility_proposed_3] + [utility_received_2] + [utility_proposed_2] + [utility_received_1] + [utility_proposed_1]


    def receive(self, last_action):
        if last_action is not None:
            oppo_offer = last_action
            utility = get_utility(oppo_offer, self.prefer, self.condition, self.domain_type, self.issue_value)
            self.offer_received.append(oppo_offer)
            self.utility_received.append(utility)
            if self.domain_type == "DISCRETE":
                value_recv = []
                for i in range(len(oppo_offer)):
                    value_recv.append(self.issue_value[i][oppo_offer[i]])
                self.value_received.append(value_recv)
                self.update_oppo_issue_value_estimation(oppo_offer, self.relative_t)
            self.oppo_prefer_estimater()
            self.update_obs()
            rl_utility = 0.5 * (self.act() + 1) * (1 - self.u_min) + self.u_min
            self.accept = (rl_utility <= utility and self.t <= self.max_round)


    def act(self):
        state = torch.FloatTensor(np.array(self.obs, dtype=float).reshape(1, -1)).to(self.device)
        action = self.actor.action(self.expert_label, state).cpu().data.numpy().flatten()
        return action