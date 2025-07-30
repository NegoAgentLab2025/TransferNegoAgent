# -*- coding:utf-8 -*- 
import numpy
from Performance import Performance
from Round import Round
import random

"""
BPR算法 - Added agreement rounds

"""


class BPR:
    """
    初始化BPR
    policy: 策略数
    belief0: BPR的belief矩阵
    """
    def __init__(self, policy_num):
        self.policy = policy_num
        self.op_policy = policy_num
        # 对手策略的belief
        # random initialization
        self.belief0 = numpy.random.rand(self.op_policy)
        self.belief0 = self.belief0[:] / sum(self.belief0)
        # print("初始 belief : ", self.belief0)

        # 自己的performance_model
        self.performance_model = Performance(self.policy, self.op_policy)
        # round model
        self.round_model = Round(self.policy, self.op_policy)

    def reset(self):
        # random initialization
        self.belief0 = numpy.random.rand(self.op_policy)
        self.belief0 = self.belief0[:] / sum(self.belief0)

    def reset_uniform(self):
        # uniform distribution initialization
        self.belief0 = numpy.full(self.op_policy, 1/self.op_policy)


    """
    BPR下选择策略
    """
    def mindBelief(self, render=False):
        # 初始化策略max_value
        #print 'belief0'
        max_value = numpy.zeros(self.policy, dtype=float)
        min_value = numpy.zeros(self.policy, dtype=float)
        for i in range(self.policy):
            for j in range(self.op_policy):
                # 双方策略获得的reward（改）
                value = self.performance_model.predictReward(i, j)
                max_value[i] += self.belief0[j] * value
        if render:
            print("max_value : ", max_value)
        # 取最大的value 'max(max_value)' 进行F*Belief计算
        for i in range(self.policy):
            for j in range(self.op_policy):
                value = self.performance_model.predictCdf(max(max_value), i, j)
                min_value[i] += self.belief0[j] * value
        if render:
            print("min_value : ", min_value)
            print("idx : ", numpy.argmin(min_value))
        return numpy.argmin(min_value)

    """
    learning - BPR下选择策略
        learning实验中，需要考虑equipped policies
    """
    def mindBelief_forLearning(self, render=False):
        # 初始化策略max_value
        #print 'belief0'
        max_value = numpy.zeros(self.policy, dtype=float)
        min_value = numpy.zeros(self.policy, dtype=float)
        for i in range(self.policy):
            for j in range(self.op_policy):
                # 双方策略获得的reward（改）
                value = self.performance_model.predictReward(i, j)
                max_value[i] += self.belief0[j] * value
        if render:
            print("max_value : ", max_value)
        # 取最大的value 'max(max_value)' 进行F*Belief计算
        for i in range(self.policy):
            for j in range(self.op_policy):
                value = self.performance_model.predictCdf(max(max_value), i, j)
                min_value[i] += self.belief0[j] * value
        if render:
            print("min_value : ", min_value)
            print("idx : ", numpy.argmin(min_value))
        return min_value
        # return numpy.argmin(min_value)

    def mindBeliefBayes(self):
        max_value = numpy.zeros(self.policy, dtype=float)
        for i in range(self.policy):
            for j in range(self.op_policy):
                # 双方策略获得的reward（改）
                max_value[i] += self.belief0[j] * self.performance_model.predictReward(i, j)
        
        return numpy.argmax(max_value)


    def mindBeliefBayes_forLearning(self):
        max_value = numpy.zeros(self.policy, dtype=float)
        for i in range(self.policy):
            for j in range(self.op_policy):
                # 双方策略获得的reward（改）
                max_value[i] += self.belief0[j] * self.performance_model.predictReward(i, j)
        
        # return numpy.argmax(max_value)
        return max_value


    # 更新belief0
    def updatePolicy(self, policy, reward, round):
        # p * belief 根据对手的reward更新belief0
        probability = numpy.zeros(self.op_policy, dtype=float)
        probability_total = 0.0

        for j in range(self.op_policy):
            _probability_reward = self.performance_model.predictPdf(reward, policy, j)
            _probability_round = self.round_model.predictPdf(round, policy, j)
            soft_max = _probability_reward * _probability_round * self.belief0[j]
            soft_max = max(soft_max, 0.0000000001)
            probability[j] = soft_max
            probability_total += soft_max

        for j in range(self.op_policy):
            self.belief0[j] = probability[j] / probability_total
            self.belief0[j] = max(self.belief0[j], 0.1)
            self.belief0[j] = min(self.belief0[j], 0.40)
        self.belief0 = self.belief0[:] / sum(self.belief0)
        # print("belief model更新为: ", self.belief0)


    # 更新perforamce_model
    def updatePerformanceModel(self, policy, op_policy, reward):
        self.performance_model.updateParameter(policy, op_policy, reward)


    def finalCalcPerformaceModel(self, policy, op_policy):
        self.performance_model.finalCalcParameter(policy, op_policy)


    def savePerformanceModel(self, domain):  # filename is like './performanceModels/Acquisition.npy'
        filename = './performanceModels/' + domain + '.npy'
        numpy.save(filename, self.performance_model.performance_model)  # 三维数组


    def loadPerformanceModel(self, domain):
        filename = './performanceModels/' + domain + '.npy'
        self.performance_model.performance_model = numpy.load(filename)
        return numpy.load(filename)

    # 更新round_model
    def updateRoundModel(self, policy, op_policy, reward):
        self.round_model.updateParameter(policy, op_policy, reward)


    def finalCalcRoundModel(self, policy, op_policy):
        self.round_model.finalCalcParameter(policy, op_policy)


    def saveRoundModel(self, domain):  # filename is like './roundModels/Acquisition.npy'
        filename = './roundModels/' + domain + '.npy'
        numpy.save(filename, self.round_model.round_model)  # 三维数组


    def loadRoundModel(self, domain):
        filename = './roundModels/' + domain + '.npy'
        self.round_model.round_model = numpy.load(filename)
        return numpy.load(filename)