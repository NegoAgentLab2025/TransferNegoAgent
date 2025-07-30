import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (torch.exp(log_std) + 1e-8)).pow(2) + 2 * log_std + np.log(2 * np.pi))
    likelihood = pre_sum.sum(dim=1).reshape([-1,1])
    return likelihood


def clip_by_pass_grad(x, l=-1., u=1., device='cpu'):
    clip_up = (x > u).type(torch.FloatTensor).to(device)
    clip_low = (x < l).type(torch.FloatTensor).to(device)
    return x + ((u - x) * clip_up + (l - x) * clip_low).detach()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, device):
        super(Actor, self).__init__()

        self.max_action = max_action
        self.device = device

        self.l1 = nn.Linear(state_dim, 128).to(device)
        self.l2 = nn.Linear(128, 128).to(device)
        self.l3 = nn.Linear(128, 64).to(device)
        self.l4 = nn.Linear(64, action_dim).to(device)
        self.l5 = nn.Linear(64, action_dim).to(device)

        
    # Addition - Specific for Policy Distillation
    def mu_std(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        mu = self.l4(x)
        std = self.l5(x)

        return mu, std

    def pre_squash(self, x):
        x = F.relu(self.l1(x)).to(self.device)
        x = F.relu(self.l2(x)).to(self.device)
        x = F.relu(self.l3(x)).to(self.device)
        mu = self.l4(x).to(self.device)
        log_std = torch.tanh(self.l5(x)).to(self.device)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std).to(self.device)
        temp = torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(self.device)
        pi = mu + temp * std
        pi = pi
        return mu, pi, log_std

    def action(self, x, is_deterministic):
        mu, pi, _ = self.pre_squash(x)
        mu = torch.tanh(mu) * self.max_action
        pi = torch.tanh(pi) * self.max_action
        action = mu if is_deterministic else pi
        return action

    def forward(self, x):
        mu, pi, log_std = self.pre_squash(x)
        logp_pi = gaussian_likelihood(pi, mu, log_std)

        squashed_mu = torch.tanh(mu)
        squashed_pi = torch.tanh(pi)
        # FIXME 1218 - add this value to avoid numerical issues
        logp_pi -= torch.log(clip_by_pass_grad(torch.ones_like(squashed_pi) - squashed_pi.pow(2), l=0,
                                               u=1, device=self.device) + 1e-6).sum(dim=1, keepdim=True)

        scaled_mu = squashed_mu * self.max_action
        scaled_pi = squashed_pi * self.max_action

        return scaled_mu, scaled_pi, logp_pi


class Actor_Mix(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, device, teacher_actor):
        super(Actor_Mix, self).__init__()

        self.max_action = max_action
        self.device = device
        self.teacher_actor = teacher_actor
        self.N = 4

        self.l1 = nn.Linear(state_dim, 128).to(device)
        self.l2 = nn.Linear(128, 128).to(device)
        self.l3 = nn.Linear(128, 64).to(device)
        self.l4 = nn.Linear(64, action_dim).to(device)
        self.l5 = nn.Linear(64, action_dim).to(device)

    # Addition - Specific for Policy Distillation
    def mu_std(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        mu = self.l4(x)
        std = self.l5(x)

        return mu, std

    def pre_squash(self, x, mix_factor, teacher_P):

        # x = F.relu(self.l1(x)).to(self.device)
        h_t1 = [0]
        sum1 = 0
        for i in range(1, self.N + 1):
            h_t1.append(self.teacher_actor[i].l1(x))
        for i in range(1, self.N + 1):
            sum1 = sum1 + h_t1[i] * teacher_P[i]
            
        #h_t1 = self.teacher_actor.l1(x)
        h1 = self.l1(x) * (1 - mix_factor) + sum1 * mix_factor

        h1 = F.relu(h1).to(self.device)
        for i in range(1, self.N + 1):
            h_t1[i] = F.relu(h_t1[i]).to(self.device)

        # x = F.relu(self.l2(x)).to(self.device)
        h_t2 = [0]
        sum2 = 0
        for i in range(1, self.N + 1):
            h_t2.append(self.teacher_actor[i].l2(h_t1[i]))
        for i in range(1, self.N + 1):
            sum2 = sum2 + h_t2[i] * teacher_P[i]
        
        #h_t2 = self.teacher_actor.l2(h_t1)
        h2 = self.l2(h1) * (1 - mix_factor) + sum2 * mix_factor

        h2 = F.relu(h2).to(self.device)
        for i in range(1, self.N + 1):
            h_t2[i] = F.relu(h_t2[i]).to(self.device)

        # x = F.relu(self.l3(x)).to(self.device)
        h_t3 = [0]
        sum3 = 0
        for i in range(1, self.N + 1):
            h_t3.append(self.teacher_actor[i].l3(h_t2[i]))
        for i in range(1, self.N + 1):
            sum3 = sum3 + h_t3[i] * teacher_P[i]
        
        #h_t3 = self.teacher_actor.l3(h_t2)
        h3 = self.l3(h2) * (1 - mix_factor) + sum3 * mix_factor

        x = F.relu(h3).to(self.device)

        mu = self.l4(x).to(self.device)
        log_std = torch.tanh(self.l5(x)).to(self.device)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std).to(self.device)
        temp = torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(self.device)
        pi = mu + temp * std
        pi = pi
        return mu, pi, log_std

    def action(self, x, mix_factor, teacher_P, is_deterministic):
        mu, pi, _ = self.pre_squash(x, mix_factor, teacher_P)
        mu = torch.tanh(mu) * self.max_action
        pi = torch.tanh(pi) * self.max_action
        action = mu if is_deterministic else pi
        return action

    def forward(self, x, mix_factor, teacher_P):
        mu, pi, log_std = self.pre_squash(x, mix_factor, teacher_P)
        logp_pi = gaussian_likelihood(pi, mu, log_std)

        squashed_mu = torch.tanh(mu)
        squashed_pi = torch.tanh(pi)
        # FIXME 1218 - add this value to avoid numerical issues
        logp_pi -= torch.log(clip_by_pass_grad(torch.ones_like(squashed_pi) - squashed_pi.pow(2), l=0,
                                               u=1, device=self.device) + 1e-6).sum(dim=1, keepdim=True)

        scaled_mu = squashed_mu * self.max_action
        scaled_pi = squashed_pi * self.max_action

        return scaled_mu, scaled_pi, logp_pi

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 1)

        # Q2 architecture
        self.l5 = nn.Linear(state_dim + action_dim, 128)
        self.l6 = nn.Linear(128, 128)
        self.l7 = nn.Linear(128, 64)
        self.l8 = nn.Linear(64, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = F.relu(self.l3(x1))
        x1 = self.l4(x1)

        x2 = F.relu(self.l5(xu))
        x2 = F.relu(self.l6(x2))
        x2 = F.relu(self.l7(x2))
        x2 = self.l8(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = F.relu(self.l3(x1))
        x1 = self.l4(x1)
        return x1

class Critic_Mix(nn.Module):
    def __init__(self, state_dim, action_dim, teacher_critic):
        super(Critic_Mix, self).__init__()

        self.teacher_critic = teacher_critic
        self.N = 4
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 1)

        # Q2 architecture
        self.l5 = nn.Linear(state_dim + action_dim, 128)
        self.l6 = nn.Linear(128, 128)
        self.l7 = nn.Linear(128, 64)
        self.l8 = nn.Linear(64, 1)

    def forward(self, x, u, mix_factor, teacher_P): 
        xu = torch.cat([x, u], 1)

        # x1 = F.relu(self.l1(xu))
        h_t1 = [0]
        sum1 = 0
        for i in range(1, self.N + 1):
            h_t1.append(self.teacher_critic[i].l1(xu))
        for i in range(1, self.N + 1):
            sum1 = sum1 + h_t1[i] * teacher_P[i]
        
        h1 = self.l1(xu) * (1 - mix_factor) + sum1 * mix_factor
        h1 = F.relu(h1)
        for i in range(1, self.N + 1):
            h_t1[i] = F.relu(h_t1[i])

        # x1 = F.relu(self.l2(x1))
        h_t2 = [0]
        sum2 = 0
        for i in range(1, self.N + 1):
            h_t2.append(self.teacher_critic[i].l2(h_t1[i]))
        for i in range(1, self.N + 1):
            sum2 = sum2 + h_t2[i] * teacher_P[i]
        
        h2 = self.l2(h1) * (1 - mix_factor) + sum2 * mix_factor
        h2 = F.relu(h2)
        for i in range(1, self.N + 1):
            h_t2[i] = F.relu(h_t2[i])

        # x1 = F.relu(self.l3(x1))
        h_t3 = [0]
        sum3 = 0
        for i in range(1, self.N + 1):
            h_t3.append(self.teacher_critic[i].l3(h_t2[i]))
        for i in range(1, self.N + 1):
            sum3 = sum3 + h_t3[i] * teacher_P[i]
        
        h3 = self.l3(h2) * (1 - mix_factor) + sum3 * mix_factor
        h3 = F.relu(h3)
        for i in range(1, self.N + 1):
            h_t3[i] = F.relu(h_t3[i])

        x1 = self.l4(h3)

        # x2 = F.relu(self.l5(xu))
        h_t5 = [0]
        sum5 = 0
        for i in range(1, self.N + 1):
            h_t5.append(self.teacher_critic[i].l5(xu))
        for i in range(1, self.N + 1):
            sum5 = sum5 + h_t5[i] * teacher_P[i]
        
        h5 = self.l5(xu) * (1 - mix_factor) + sum5 * mix_factor
        h5 = F.relu(h5)
        for i in range(1, self.N + 1):
            h_t5[i] = F.relu(h_t5[i])

        # x2 = F.relu(self.l6(x2))
        h_t6 = [0]
        sum6 = 0
        for i in range(1, self.N + 1):
            h_t6.append(self.teacher_critic[i].l6(h_t5[i]))
        for i in range(1, self.N + 1):
            sum6 = sum6 + h_t6[i] * teacher_P[i]
        
        h6 = self.l6(h5) * (1 - mix_factor) + sum6 * mix_factor
        h6 = F.relu(h6)
        for i in range(1, self.N + 1):
            h_t6[i] = F.relu(h_t6[i])

        # x2 = F.relu(self.l7(x2))
        h_t7 = [0]
        sum7 = 0
        for i in range(1, self.N + 1):
            h_t7.append(self.teacher_critic[i].l7(h_t6[i]))
        for i in range(1, self.N + 1):
            sum7 = sum7 + h_t7[i] * teacher_P[i]
        
        h7 = self.l7(h6) * (1 - mix_factor) + sum7 * mix_factor
        h7 = F.relu(h7)
        for i in range(1, self.N + 1):
            h_t7[i] = F.relu(h_t7[i])

        x2 = self.l8(h7)
        return x1, x2

    def Q1(self, x, u, mix_factor, teacher_P):
        xu = torch.cat([x, u], 1)

        # # x1 = F.relu(self.l1(xu))
        # h_t1 = self.teacher_critic.l1(xu)
        # h1 = self.l1(xu) * (1 - mix_factor) + h_t1 * mix_factor
        # h1 = F.relu(h1)
        # h_t1 = F.relu(h_t1)

        # # x1 = F.relu(self.l2(x1))
        # h_t2 = self.teacher_critic.l2(h_t1)
        # h2 = self.l2(h1) * (1 - mix_factor) + h_t2 * mix_factor
        # h2 = F.relu(h2)
        # h_t2 = F.relu(h_t2)

        # # x1 = F.relu(self.l3(x1))
        # h_t3 = self.teacher_critic.l3(h_t2)
        # h3 = self.l3(h2) * (1 - mix_factor) + h_t3 * mix_factor
        # h3 = F.relu(h3)
        # h_t3 = F.relu(h_t3)

        h_t1 = [0]
        sum1 = 0
        for i in range(1, self.N + 1):
            h_t1.append(self.teacher_critic[i].l1(xu))
        for i in range(1, self.N + 1):
            sum1 = sum1 + h_t1[i] * teacher_P[i]
        
        h1 = self.l1(xu) * (1 - mix_factor) + sum1 * mix_factor
        h1 = F.relu(h1)
        for i in range(1, self.N + 1):
            h_t1[i] = F.relu(h_t1[i])

        # x1 = F.relu(self.l2(x1))
        h_t2 = [0]
        sum2 = 0
        for i in range(1, self.N + 1):
            h_t2.append(self.teacher_critic[i].l2(h_t1[i]))
        for i in range(1, self.N + 1):
            sum2 = sum2 + h_t2[i] * teacher_P[i]
        
        h2 = self.l2(h1) * (1 - mix_factor) + sum2 * mix_factor
        h2 = F.relu(h2)
        for i in range(1, self.N + 1):
            h_t2[i] = F.relu(h_t2[i])

        # x1 = F.relu(self.l3(x1))
        h_t3 = [0]
        sum3 = 0
        for i in range(1, self.N + 1):
            h_t3.append(self.teacher_critic[i].l3(h_t2[i]))
        for i in range(1, self.N + 1):
            sum3 = sum3 + h_t3[i] * teacher_P[i]
        
        h3 = self.l3(h2) * (1 - mix_factor) + sum3 * mix_factor
        h3 = F.relu(h3)
        for i in range(1, self.N + 1):
            h_t3[i] = F.relu(h_t3[i])

        x1 = self.l4(h3)
        return x1

class VNetwork(nn.Module):
    def __init__(self, state_dim):
        super(VNetwork, self).__init__()

        # V architecture
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)

        return x

class VNetwork_Mix(nn.Module):
    def __init__(self, state_dim, teacher_VNetwork):
        super(VNetwork_Mix, self).__init__()
        self.teacher_VNetwork = teacher_VNetwork
        self.N = 4
        # V architecture
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 1)

    def forward(self, x, mix_factor, teacher_P):

        # # x = F.relu(self.l1(x))
        # h_t1 = self.teacher_VNetwork.l1(x)
        # h1 = self.l1(x) * (1 - mix_factor) + h_t1 * mix_factor
        # h1 = F.relu(h1)
        # h_t1 = F.relu(h_t1)

        # # x = F.relu(self.l2(x))
        # h_t2 = self.teacher_VNetwork.l2(h_t1)
        # h2 = self.l2(h1) * (1 - mix_factor) + h_t2 * mix_factor
        # h2 = F.relu(h2)
        # h_t2 = F.relu(h_t2)

        # # x = F.relu(self.l3(x))
        # h_t3 = self.teacher_VNetwork.l3(h_t2)
        # h3 = self.l3(h2) * (1 - mix_factor) + h_t3 * mix_factor
        # h3 = F.relu(h3)
        # h_t3 = F.relu(h_t3)

        h_t1 = [0]
        sum1 = 0
        for i in range(1, self.N + 1):
            h_t1.append(self.teacher_VNetwork[i].l1(x))
        for i in range(1, self.N + 1):
            sum1 = sum1 + h_t1[i] * teacher_P[i]
        
        h1 = self.l1(x) * (1 - mix_factor) + sum1 * mix_factor
        h1 = F.relu(h1)
        for i in range(1, self.N + 1):
            h_t1[i] = F.relu(h_t1[i])

        # x1 = F.relu(self.l2(x1))
        h_t2 = [0]
        sum2 = 0
        for i in range(1, self.N + 1):
            h_t2.append(self.teacher_VNetwork[i].l2(h_t1[i]))
        for i in range(1, self.N + 1):
            sum2 = sum2 + h_t2[i] * teacher_P[i]
        
        h2 = self.l2(h1) * (1 - mix_factor) + sum2 * mix_factor
        h2 = F.relu(h2)
        for i in range(1, self.N + 1):
            h_t2[i] = F.relu(h_t2[i])

        # x1 = F.relu(self.l3(x1))
        h_t3 = [0]
        sum3 = 0
        for i in range(1, self.N + 1):
            h_t3.append(self.teacher_VNetwork[i].l3(h_t2[i]))
        for i in range(1, self.N + 1):
            sum3 = sum3 + h_t3[i] * teacher_P[i]
        
        h3 = self.l3(h2) * (1 - mix_factor) + sum3 * mix_factor
        h3 = F.relu(h3)
        for i in range(1, self.N + 1):
            h_t3[i] = F.relu(h_t3[i])

        x = self.l4(h3)
        return x


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, device, use_automatic_entropy_tuning=True, target_entropy=None):
        self.device = device

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod([action_dim]).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)

        self.actor = Actor(state_dim, action_dim, max_action, device).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.V_network = VNetwork(state_dim).to(self.device)
        self.V_network_target = VNetwork(state_dim).to(self.device)
        self.V_network_target.load_state_dict(self.V_network.state_dict())
        self.vf_optimizer = torch.optim.Adam([{'params': self.critic.parameters()},
                                              {'params': self.V_network.parameters()}], lr=1e-3)

        self.max_action = max_action

    def select_action(self, state, is_deterministic=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor.action(state, is_deterministic).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001):

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Select action according to policy and add clipped noise
            _, pi, logp_pi = self.actor(state)

            # Compute alpha loss
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = 0.2

            # Compute the target Q value
            V_ = self.V_network_target(next_state)
            target_Q = reward + (done * discount * V_).detach()

            # Compute the target V value
            Q1_pi, Q2_pi = self.critic(state, pi)
            target_V = torch.min(Q1_pi, Q2_pi)
            target_V = (target_V - alpha * logp_pi).detach()

            # Get current Q/V estimates
            current_Q1, current_Q2 = self.critic(state, action)
            current_V = self.V_network(state)            

            # Compute critic loss
            vf_loss = 0.5 * F.mse_loss(current_Q1, target_Q) + 0.5 * F.mse_loss(current_Q2, target_Q) \
                          + 0.5 * F.mse_loss(current_V, target_V)

            # Optimize the critic
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

            # Compute actor loss
            # _, pi_a, logp_pi_a = self.actor(state)
            Q1_pi = self.critic.Q1(state, pi)
            actor_loss = (alpha * logp_pi - Q1_pi).mean()

            # Optimize alpha
            if self.use_automatic_entropy_tuning:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.V_network.parameters(), self.V_network_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        # 修改
        torch.save(self.V_network.state_dict(), '%s/%s_V_network.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))
        torch.save(self.alpha_optimizer.state_dict(), '%s/%s_alpha_optimizer.pth' % (directory, filename))
        torch.save(self.vf_optimizer.state_dict(), '%s/%s_vf_optimizer.pth' % (directory, filename))
        torch.save(self.log_alpha, '%s/%s_log_alpha.pth' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        # 修改
        self.V_network.load_state_dict(torch.load('%s/%s_V_network.pth' % (directory, filename)))
        # self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.log_alpha = torch.load('%s/%s_log_alpha.pth' % (directory, filename))
        # print('self.log_alpha:', self.log_alpha)
        print(directory, ' self.alpha:', self.log_alpha.exp())
        # self.alpha_optimizer.load_state_dict(torch.load('%s/%s_alpha_optimizer.pth' % (directory, filename)))
        # self.vf_optimizer.load_state_dict(torch.load('%s/%s_vf_optimizer.pth' % (directory, filename)))
        self.V_network_target.load_state_dict(self.V_network.state_dict())

class SAC_distill(object):
    def __init__(self, state_dim, action_dim, max_action, device, use_automatic_entropy_tuning=True,
                 target_entropy=None):
        self.device = device
        self.N = 4
        self.teacher_actor = [None] * (self.N + 1)
        self.teacher_critic = [None] * (self.N + 1)
        self.teacher_VNetwork = [None] * (self.N + 1)
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod([action_dim]).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)

        self.actor = Actor(state_dim, action_dim, max_action, device).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.V_network = VNetwork(state_dim).to(self.device)
        self.V_network_target = VNetwork(state_dim).to(self.device)
        self.V_network_target.load_state_dict(self.V_network.state_dict())
        self.vf_optimizer = torch.optim.Adam([{'params': self.critic.parameters()},
                                              {'params': self.V_network.parameters()}], lr=1e-3)

        self.max_action = max_action
        # self.criteria = torch.nn.KLDivLoss(size_average=False)
        self.criteria = torch.nn.MSELoss(reduction='sum').to(self.device)

    def load_teachers(self, teacher_actor, teacher_critic, teacher_VNetwork):
        for i in range(self.N + 1):
            self.teacher_actor[i] = teacher_actor[i].to(self.device)
            self.teacher_critic[i] = teacher_critic[i].to(self.device)
            self.teacher_VNetwork[i] = teacher_VNetwork[i].to(self.device)

    def select_action(self, state, is_deterministic=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor.action(state, is_deterministic).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001, distill_factor=1, teacher_P=[]):

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            teacher_Q1 = [None] * (self.N + 1)
            teacher_Q2 = [None] * (self.N + 1)
            teacher_V = [None] * (self.N + 1)
            teacher_mu = [None] * (self.N + 1)
            teacher_std = [None] * (self.N + 1)

            # Select action according to policy and add clipped noise
            _, pi, logp_pi = self.actor(state)

            # Compute alpha loss
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = 0.2

            # Compute the target Q value
            V_ = self.V_network_target(next_state)
            target_Q = reward + (done * discount * V_).detach()

            # Compute the target V value
            Q1_pi, Q2_pi = self.critic(state, pi)
            target_V = torch.min(Q1_pi, Q2_pi)
            target_V = (target_V - alpha * logp_pi).detach()

            # Get current Q/V estimates
            current_Q1, current_Q2 = self.critic(state, action)
            current_V = self.V_network(state)

            # Compute critic loss
            vf_loss = 0.5 * F.mse_loss(current_Q1, target_Q) + 0.5 * F.mse_loss(current_Q2, target_Q) \
                      + 0.5 * F.mse_loss(current_V, target_V)
            for i in range(self.N + 1):
                teacher_Q1[i], teacher_Q2[i] = self.teacher_critic[i](state, action)
                teacher_V[i] = self.teacher_VNetwork[i](state)
            
            tmp1 = 0
            for i in range(self.N + 1):
                tmp1 = tmp1 + (F.mse_loss(current_Q1, teacher_Q1[i]) + F.mse_loss(current_Q2, teacher_Q2[i]) \
                                                + F.mse_loss(current_V, teacher_V[i])) * teacher_P[i]
            vf_distill_loss = distill_factor * tmp1
            vf_loss += vf_distill_loss

            # Optimize the critic
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

            # Compute actor loss
            # _, pi_a, logp_pi_a = self.actor(state)
            Q1_pi = self.critic.Q1(state, pi)
            actor_loss = (alpha * logp_pi - Q1_pi).mean()
            mu, std = self.actor.mu_std(state)
            
            for i in range(self.N + 1):
                teacher_mu[i], teacher_std[i] = self.teacher_actor[i].mu_std(state)

            tmp2 = 0
            for i in range(self.N + 1):
                tmp2 = tmp2 + (F.mse_loss(mu, teacher_mu[i]) + F.mse_loss(std, teacher_std[i])) * teacher_P[i]

            actor_distill_loss = distill_factor * tmp2
            actor_loss += actor_distill_loss

            # Optimize alpha
            if self.use_automatic_entropy_tuning:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.V_network.parameters(), self.V_network_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        # 修改
        torch.save(self.V_network.state_dict(), '%s/%s_V_network.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))
        torch.save(self.alpha_optimizer.state_dict(), '%s/%s_alpha_optimizer.pth' % (directory, filename))
        torch.save(self.vf_optimizer.state_dict(), '%s/%s_vf_optimizer.pth' % (directory, filename))
        torch.save(self.log_alpha, '%s/%s_log_alpha.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        # 修改
        self.V_network.load_state_dict(torch.load('%s/%s_V_network.pth' % (directory, filename)))
        # self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.log_alpha = torch.load('%s/%s_log_alpha.pth' % (directory, filename))
        # print('self.log_alpha:', self.log_alpha)
        print(directory, ' self.alpha:', self.log_alpha.exp())
        # self.alpha_optimizer.load_state_dict(torch.load('%s/%s_alpha_optimizer.pth' % (directory, filename)))
        # self.vf_optimizer.load_state_dict(torch.load('%s/%s_vf_optimizer.pth' % (directory, filename)))
        self.V_network_target.load_state_dict(self.V_network.state_dict())

class SAC_Mix(object):
    def __init__(self, state_dim, action_dim, max_action, device, teacher_actor, teacher_critic, teacher_VNetwork,
                 use_automatic_entropy_tuning=True, target_entropy=None):
        self.device = device
        self.N = 4

        self.teacher_actor = [None] * (self.N + 1)
        self.teacher_critic = [None] * (self.N + 1)
        self.teacher_VNetwork = [None] * (self.N + 1)

        for i in range(self.N + 1):
            self.teacher_actor[i] = teacher_actor[i].to(self.device)
            self.teacher_critic[i] = teacher_critic[i].to(self.device)
            self.teacher_VNetwork[i] = teacher_VNetwork[i].to(self.device)
        
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod([action_dim]).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)

        self.max_action = max_action
        
        self.actor = Actor_Mix(state_dim, action_dim, max_action, device, self.teacher_actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic = Critic_Mix(state_dim, action_dim, self.teacher_critic).to(self.device)
        self.V_network = VNetwork_Mix(state_dim, self.teacher_VNetwork).to(self.device)
        self.V_network_target = VNetwork_Mix(state_dim, self.teacher_VNetwork).to(self.device)
        self.V_network_target.load_state_dict(self.V_network.state_dict())
        self.vf_optimizer = torch.optim.Adam([{'params': self.critic.parameters()},
                                              {'params': self.V_network.parameters()}], lr=1e-3)

        # self.criteria = torch.nn.KLDivLoss(size_average=False)
        self.criteria = torch.nn.MSELoss(reduction='sum').to(self.device)

    # def load_teachers(self, teacher_actor, teacher_critic, teacher_VNetwork):
    #     for i in range(self.N + 1):
    #         self.teacher_actor[i] = teacher_actor[i].to(self.device)
    #         self.teacher_critic[i] = teacher_critic[i].to(self.device)
    #         self.teacher_VNetwork[i] = teacher_VNetwork[i].to(self.device)

    def select_action(self, state, mix_factor=0, teacher_P=[], is_deterministic=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor.action(state, mix_factor, teacher_P, is_deterministic).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001, mix_factor=1, teacher_P=[]):

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Select action according to policy and add clipped noise
            _, pi, logp_pi = self.actor(state, mix_factor, teacher_P)

            # Compute alpha loss
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = 0.2

            # Compute the target Q value
            V_ = self.V_network_target(next_state, mix_factor, teacher_P)
            target_Q = reward + (done * discount * V_).detach()

            # Compute the target V value
            Q1_pi, Q2_pi = self.critic(state, pi, mix_factor, teacher_P)
            target_V = torch.min(Q1_pi, Q2_pi)
            target_V = (target_V - alpha * logp_pi).detach()

            # Get current Q/V estimates
            current_Q1, current_Q2 = self.critic(state, action, mix_factor, teacher_P)
            current_V = self.V_network(state, mix_factor, teacher_P)

            # Compute critic loss
            vf_loss = 0.5 * F.mse_loss(current_Q1, target_Q) + 0.5 * F.mse_loss(current_Q2, target_Q) \
                          + 0.5 * F.mse_loss(current_V, target_V)

            # Optimize the critic
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

            # Compute actor loss
            # _, pi_a, logp_pi_a = self.actor(state)
            Q1_pi = self.critic.Q1(state, pi, mix_factor, teacher_P)
            actor_loss = (alpha * logp_pi - Q1_pi).mean()

            # Optimize alpha
            if self.use_automatic_entropy_tuning:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.V_network.parameters(), self.V_network_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        # 修改
        torch.save(self.V_network.state_dict(), '%s/%s_V_network.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))
        torch.save(self.alpha_optimizer.state_dict(), '%s/%s_alpha_optimizer.pth' % (directory, filename))
        torch.save(self.vf_optimizer.state_dict(), '%s/%s_vf_optimizer.pth' % (directory, filename))
        torch.save(self.log_alpha, '%s/%s_log_alpha.pth' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        # 修改
        self.V_network.load_state_dict(torch.load('%s/%s_V_network.pth' % (directory, filename)))
        # self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.log_alpha = torch.load('%s/%s_log_alpha.pth' % (directory, filename))
        # print('self.log_alpha:', self.log_alpha)
        print(directory, ' self.alpha:', self.log_alpha.exp())
        # self.alpha_optimizer.load_state_dict(torch.load('%s/%s_alpha_optimizer.pth' % (directory, filename)))
        # self.vf_optimizer.load_state_dict(torch.load('%s/%s_vf_optimizer.pth' % (directory, filename)))
        self.V_network_target.load_state_dict(self.V_network.state_dict())

