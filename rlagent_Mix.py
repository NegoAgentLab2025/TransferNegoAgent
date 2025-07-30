from utils import get_utility
from algorithm.sac import SAC, SAC_Mix, Actor, Critic, VNetwork
import torch.nn as nn
import torch.nn.functional as F
import torch
#from algorithm.td3 import TD3
#from algorithm.sac2 import SAC2
import numpy as np
from agent import Agent

class RLAgent(Agent):
    def __init__(self, max_round, name, device, issue_num=3, algo='sac', seed=3):
        super().__init__(max_round=max_round, name=name)
        self.max_round = max_round
        self.issue_num = issue_num
        self.state_dim = 7
        self.action_dim = 1
        self.algo = algo
        self.device = device
        self.mix_factor = 1
        self.N = 4
        self.seed = seed

        self.teacher_actor = [Actor(self.state_dim, self.action_dim, max_action=float(1), device=device)] * (self.N + 1)
        self.teacher_critic = [Critic(self.state_dim, self.action_dim)] * (self.N + 1)
        self.teacher_VNetwork = [VNetwork(self.state_dim)] * (self.N + 1)

        # self.teacher_W = [0] * (self.N + 1)
        # self.teacher_U = [0] * (self.N + 1)
        self.teacher_P = [0] * (self.N + 1)
        self.teacher_index = [0]

        for i in range(1, self.N + 1):
            self.teacher_actor[i].load_state_dict(torch.load('./sac_models/teacher/%s/seed%s/SAC_actor.pth' %(i, self.seed)))
            self.teacher_actor[i].eval()
            self.teacher_critic[i].load_state_dict(torch.load('./sac_models/teacher/%s/seed%s/SAC_critic.pth' %(i, self.seed)))
            self.teacher_critic[i].eval()
            self.teacher_VNetwork[i].load_state_dict(torch.load('./sac_models/teacher/%s/seed%s/SAC_V_network.pth' %(i, self.seed)))
            self.teacher_VNetwork[i].eval()

        if self.algo == 'sac':
            self.use_automatic_entropy_tuning = True
            self.target_entropy = None
            self.offer_policy = SAC_Mix(state_dim=self.state_dim, action_dim=self.action_dim, max_action=float(1),
                                            use_automatic_entropy_tuning=self.use_automatic_entropy_tuning,
                                            target_entropy=self.target_entropy, device=device, 
                                            teacher_actor=self.teacher_actor, teacher_critic=self.teacher_critic, teacher_VNetwork=self.teacher_VNetwork)
        # elif self.algo == 'td3':
        #     self.offer_policy = TD3(state_dim=self.state_dim, action_dim=self.action_dim, max_action=float(1), device=device)
        # elif self.algo == 'sac2':
        #     self.use_automatic_entropy_tuning = True
        #     self.target_entropy = None
        #     self.offer_policy = SAC2(state_dim=self.state_dim, action_dim=self.action_dim, max_action=float(1), use_automatic_entropy_tuning=self.use_automatic_entropy_tuning, target_entropy=self.target_entropy, device=device)
    
    def load_teacher(self, label):
        if self.algo == 'sac':
            self.offer_policy.load(filename='SAC', directory="./sac_models/teacher/{}".format(label))

    def load(self, label, seed=3):
        if self.algo == 'sac':
            self.offer_policy.load(filename='SAC', directory="./sac_models/teacher/{}/seed{}".format(label, str(seed)))
        # elif self.algo == 'td3':
        #     self.offer_policy.load(filename='TD3', directory='./td3_models/{}/seed{}'.format(label, str(seed)))
        # elif self.algo == 'sac2':
        #     self.offer_policy.load(filename='SAC2', directory="./sac2_models/{}/seed{}".format(label, str(seed)))

    # def load_teachers(self):
    #     for i in range(1, self.N + 1):
    #         self.teacher_actor[i].load_state_dict(torch.load('./sac_models/teacher/%s/SAC_actor.pth' %i))
    #         self.teacher_actor[i].eval()
    #         self.teacher_critic[i].load_state_dict(torch.load('./sac_models/teacher/%s/SAC_critic.pth' %i))
    #         self.teacher_critic[i].eval()
    #         self.teacher_VNetwork[i].load_state_dict(torch.load('./sac_models/teacher/%s/SAC_V_network.pth' %i))
    #         self.teacher_VNetwork[i].eval()
    #     self.offer_policy.load_teachers(self.teacher_actor, self.teacher_critic, self.teacher_VNetwork)

    def train(self, offer_replay_buffer, iterations, batch_size, discount, tau, policy_noise, noise_clip, policy_freq, mix_factor, teacher_P=[]):
        if self.algo == 'sac':#or self.algo == 'sac2'
            self.offer_policy.train(offer_replay_buffer, iterations, batch_size, discount, tau, mix_factor, teacher_P = self.teacher_P)
        # elif self.algo == 'td3':
        #     self.offer_policy.train(offer_replay_buffer, iterations, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

    def reset(self):
        super().reset()
        self.update_obs()

    def save(self, directory):
        if self.algo == 'sac':
            self.offer_policy.save(filename="SAC", directory=directory)
        # elif self.algo == 'td3':
        #     self.offer_policy.save(filename='TD3', directory=directory)
        # elif self.algo == 'sac2':
        #     self.offer_policy.save(filename="SAC2", directory=directory)

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
            rl_utility = 0.5 * (int(self.offer_policy.select_action(np.array(self.obs), self.mix_factor, self.teacher_P)) + 1) * (1 - self.u_min) + self.u_min
            self.accept = (rl_utility <= utility and self.t <= self.max_round)
    

    def compute_p(self, mix_factor, teacher_evaluation, teacher_W):
        #print('更新p！')
        sum = 0
        self.mix_factor = mix_factor
        min_W = min(teacher_evaluation[1:])
        max_W = max(teacher_evaluation[1:])
        tmp = [0] * 7
        
        if min_W != max_W:
            for i in range(1, self.N + 1):
                tmp[i] = (teacher_evaluation[i] - min_W) / (max_W - min_W)

        for i in range(1, self.N + 1):
            sum += np.exp(1 * tmp[i])
        for i in range(1, self.N + 1):
            tmp[i] = np.exp(1 * tmp[i]) / sum
        
        teacher_P = [0.5*a + 0.5*b for a,b in zip(tmp, teacher_W)]
        
        max1 = teacher_P.index(max(teacher_P))

        for i, p in enumerate(teacher_P):
            if i not in [max1]:
                teacher_P[i] = 0
        
        self.teacher_P = teacher_P
        #tmp与teacher_W的组合
        



    def act(self, mix_factor=0):
        action = self.offer_policy.select_action(np.array(self.obs), mix_factor, self.teacher_P)
        return action

    # def update_W(self, episode_reward):
    #     self.teacher_W[self.teacher_index[0]] = (self.teacher_W[self.teacher_index[0]] * self.teacher_U[self.teacher_index[0]] + episode_reward) \
    #                                             / (self.teacher_U[self.teacher_index[0]] + 1)
    #     self.teacher_U[self.teacher_index[0]] += 1
        # print('tau:', self.tau)

    # 为了收集teacher models的transitions，新增加的计算mu, std, Q1, Q2, V方法
    def get_mu_std(self, state):
        return self.offer_policy.actor.mu_std(state)

    def Q_values(self, state, action):
        # print("state.shape:", state.shape, "action.shape:", action.shape)
        return self.offer_policy.critic(state, action)

    def V_value(self, state):
        return self.offer_policy.V_network(state)

    # 从蒸馏网络加载共享线性层的参数
    def loadSharedLayersParameter(self, student_model_path, opponent_label=None):
        if self.algo == 'sac2':
            self.offer_policy.loadSharedLayersParameter(student_model_path, opponent_label)
        else:
            raise NotImplementedError
        
    # 冻结共享层的参数
    def frozenSharedLayersParameter(self):
        if self.algo == 'sac2':
            self.offer_policy.frozenSharedLayersParameter()
        else:
            raise NotImplementedError

