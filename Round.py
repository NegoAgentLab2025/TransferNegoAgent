# -*- coding:utf-8 -*- 
import numpy as np
from scipy.stats import norm


class Round:
    def __init__(self, policy, op_policy):
        # policy 表示自己的策略个数，op_policy表示对手策略个数
        self.policy = policy
        self.op_policy = op_policy
        # 样本个数
        self.count = np.zeros((self.policy, self.op_policy), dtype=float)
        # 每个策略的高斯分布参数（mu, sigma）
        self.round_model = np.zeros((self.policy, self.op_policy, 2), dtype=float)
        # performance model的(mu, sigma)计算所用的数据
        self.round_data = np.zeros((self.policy, self.op_policy, 1), dtype=float)
        self.round_data = self.round_data.tolist()

    def updateParameter(self, policy_index, op_policy_index, value):
        if self.count[policy_index][op_policy_index] == 0:
            self.round_data[policy_index][op_policy_index][0] = value
            # self.round_model[policy_index, op_policy_index, 0] = value
        else:
            self.round_data[policy_index][op_policy_index].append(value)
            # 均值
            # self.round_model[policy_index, op_policy_index, 0] = np.mean(self.round_data[policy_index][op_policy_index])
            # 标准差
            # self.round_model[policy_index, op_policy_index, 1] = np.var(self.round_data[policy_index][op_policy_index])

        # 样本个数
        self.count[policy_index, op_policy_index] += 1.0
        # print("用于计算均值标准差的数据：", self.round_data[policy_index][op_policy_index])


    def finalCalcParameter(self, policy_index, op_policy_index):
        # 均值
        self.round_model[policy_index, op_policy_index, 0] = np.mean(self.round_data[policy_index][op_policy_index])
        # 标准差
        self.round_model[policy_index, op_policy_index, 1] = np.var(self.round_data[policy_index][op_policy_index])


    def predictReward(self, policy_index, op_policy_index):
        return self.round_model[policy_index, op_policy_index, 0]

    def predictCdf(self, value, policy_index, op_policy_index):
        mu = self.round_model[policy_index, op_policy_index, 0]
        std = self.round_model[policy_index, op_policy_index, 1]
        if std < 0.01:
            std = 0.01
        return norm.cdf(value, mu, std)

    def predictPdf(self, value, policy_index, op_policy_index, render=False):
        # 1. 用蒙特卡洛来计算概率还是行不通的
        # 2. norm.pdf(x, loc, scale)返回的是概率密度函数的值，并不能如此计算概率
        # a = norm.pdf(value, mu, std)
        # 3. 这里考虑到标准差很小，于是直接按照到mu的距离来计算概率
        # 由于正态分布是连续分布，不存在计算某一点处的概率（只有计算某区间内的概率）
        # 这里统一std为1，计算[value-0.5, value+0.5]区间内的概率
        # std = 1
        # a = norm.cdf(value+0.5, mu, std) - norm.cdf(value-0.5, mu, std)
        # 4. 但是这种方法，使概率区分度很小
        # 故需要选择一个递减函数，这里选择了y = 16^(-x)，x = |value-mu|
        mu = self.round_model[policy_index, op_policy_index, 0]
        std = self.round_model[policy_index, op_policy_index, 1]
        value_mu_abs = abs(mu - value) / 200
        a = pow(16, -value_mu_abs)      

        if render:
            print("     value: ", value, " mu: ", mu, " std: ", std, " 概率: ", a)
        return a

