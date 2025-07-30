from sys import implementation
from xml import dom
import numpy as np
from scipy.stats import wasserstein_distance
import os
import argparse
import random
import torch
import pandas as pd
from utils import get_utility
from utils import ReplayBuffer
from rlagent_Mix import RLAgent
from negotiation import Negotiation
from AgreeableAgent2018 import AgreeableAgent2018
from agent36 import Agent36
from atlas3 import Atlas3
from ponpokoagent import PonPokoAgent
from parscat import ParsCat
from agentlg import AgentLG
from omacagent import OMAC
from CUHKAgent import CUHKAgent
from HardHeaded import HardHeadedAgent
from YXAgent import YXAgent
from ParsAgent import ParsAgent
from caduceus import Caduceus
from theFawkes import TheFawkes
from caduceusDC16 import CaduceusDC16
from randomagent import RandomAgent
from timedependent import TimeAgent

def evaluate_policy(rl_agent, opponent, domain, eval_episodes=1000, render=False):
    rl_agent = rl_agent
    opponent = opponent
    avg_reward = 0
    avg_round = 0
    avg_oppo = 0
    succ_avg_reward = 0
    succ_counts = 0
    succ_avg_oppo = 0
    domain_file = domain

    succ_domains = []

    allDomains = ["Acquisition", "Animal", "Coffee", "DefensiveCharms", "Camera", "DogChoosing", "FiftyFifty2013", \
                "HouseKeeping", "Icecream", "Kitchen", "Laptop", "NiceOrDie", "Outfit", "planes", "SmartPhone", \
                "Ultimatum", "Wholesaler", "Lunch"]

    # if args.oppo_type == "parscat":
    #     allDomains = ["Acquisition", "Animal", "Coffee", "DefensiveCharms", "Camera", "DogChoosing", "FiftyFifty2013", \
    #             "HouseKeeping", "Icecream", "Kitchen", "Laptop", "Outfit", "planes", "SmartPhone", \
    #             "Wholesaler", "RentalHouse-B", "Barter-C", "Amsterdam-B"]
    

    for j in range(eval_episodes):
        if render:
            print("----------- a new episode when evaluating ---------")

        if args.domain_type == "REAL":
            negotiation = Negotiation(max_round=args.max_round, issue_num=3, domain_type=args.domain_type, domain_file=None)
        elif args.domain_type == "DISCRETE":
            if domain_file == 'randomDomain':
                negotiation = Negotiation(max_round=args.max_round, domain_type=args.domain_type, domain_file=random.choice(allDomains))
            else:
                negotiation = Negotiation(max_round=args.max_round, domain_type=args.domain_type, domain_file=domain_file)

        negotiation.add(opponent)
        negotiation.add(rl_agent)
        negotiation.reset()

        if render:
            if negotiation.domain_type == 'DISCRETE':
                print("DISCRETE domain: ", negotiation.domain_file)

        last_offer = None
        accept = False
        current_player = 0
        episode_reward = 0
        episode_round = 0
        for i in range(1, negotiation.max_round + 1):
            if render:
                print("Round:", i)
            current_player = 1 - i % 2

            negotiation.agents_pool[current_player].set_t(i)
            negotiation.agents_pool[1 - current_player].set_t(i)
            if i == 1:
                negotiation.agents_pool[current_player].receive(last_offer)
            
            if negotiation.agents_pool[current_player].__class__.__name__ == "RLAgent":
                if render:
                    print("  RL agent's obs: ", rl_agent.obs)
                action = negotiation.agents_pool[current_player].act()

                last_utility = 0.5 * (action + 1) * (rl_agent.u_max - rl_agent.u_min) + rl_agent.u_min
                rl_agent.s = last_utility
                last_offer = rl_agent.gen_offer()
            else:
                last_offer = negotiation.agents_pool[current_player].act()
            
            if (last_offer is None) and (negotiation.agents_pool[current_player].__class__.__name__ in Negotiation.ANAC_classname_list):
                if negotiation.agents_pool[current_player].accept == True:                     
                    accept = True
                    episode_round = i
                    if render:
                        print(negotiation.agents_pool[current_player].name, "accept the offer.\n")                    
                    episode_reward = negotiation.agents_pool[1-current_player].utility_proposed[-1]
                    avg_oppo +=  negotiation.agents_pool[current_player].utility_received[-1]
                    succ_avg_reward += episode_reward
                    succ_avg_oppo += negotiation.agents_pool[current_player].utility_received[-1]
                    succ_counts += 1
                    if negotiation.domain_file not in succ_domains:
                        succ_domains.append(negotiation.domain_file)
                    break                
                elif negotiation.agents_pool[current_player].terminate == True:  
                    if render: 
                        print(negotiation.agents_pool[current_player].name, "end the negotiation.")
                    episode_reward = 0
                break
            elif last_offer is None:
                print("Error exist: agent's offer is None.")
                exit(-1)

            negotiation.agents_pool[1 - current_player].receive(last_offer)
            if render:
                print("  " + negotiation.agents_pool[current_player].name, "'s action", last_offer)
                print("  utility to %s: %f, utility to %s: %f\n" % (negotiation.agents_pool[current_player].name,
                      get_utility(last_offer, negotiation.agents_pool[current_player].prefer, negotiation.agents_pool[current_player].condition, negotiation.agents_pool[current_player].domain_type, negotiation.agents_pool[current_player].issue_value),
                      negotiation.agents_pool[1 - current_player].name, get_utility(last_offer, negotiation.agents_pool[1 - current_player].prefer,
                      negotiation.agents_pool[1 - current_player].condition, negotiation.agents_pool[1 - current_player].domain_type, negotiation.agents_pool[1 - current_player].issue_value)))
            if i == negotiation.max_round:
                episode_reward = 0
            if negotiation.agents_pool[1 - current_player].accept:
                accept = True
                episode_round = i + 1
                if render:
                    print("Round:", i+1)
                    print("  "+negotiation.agents_pool[1 - current_player].name, "accept the offer.\n")
                if last_offer is None:
                    episode_reward = 0
                else:
                    episode_reward = get_utility(offer=last_offer, prefer=rl_agent.prefer, condition=rl_agent.condition, domain_type=rl_agent.domain_type, issue_value=rl_agent.issue_value)
                    avg_oppo +=  get_utility(offer=last_offer, prefer=opponent.prefer, condition=opponent.condition, domain_type=opponent.domain_type, issue_value=opponent.issue_value)
                    succ_avg_reward += episode_reward
                    succ_avg_oppo += get_utility(offer=last_offer, prefer=opponent.prefer, condition=opponent.condition, domain_type=opponent.domain_type, issue_value=opponent.issue_value)
                    succ_counts += 1
                    if negotiation.domain_file not in succ_domains:
                        succ_domains.append(negotiation.domain_file)
                break

            if render:
                print()

        if render:
            if accept:
                print("Negotiation success")
            else:                
                print("Negotiation failed")
            print("rl received reward: %f\n" % episode_reward)
        
        if accept == False:
            episode_round = args.max_round

        avg_reward += episode_reward
        avg_round += episode_round
        
    avg_reward /= eval_episodes
    avg_round /= eval_episodes
    avg_oppo /= eval_episodes

    if succ_counts != 0:
        succ_avg_reward /= succ_counts
        succ_avg_oppo /= succ_counts

    if render:
        print("---------------------------------------")
        print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
        print("Average finished rounds: %f" % (avg_round))
        print("Opponent get average utility %f" % (avg_oppo))
        print("协商成功的次数：", succ_counts)
        print("成功协商的平均效用：", succ_avg_reward)
        print("成功协商的对手平均效用：", succ_avg_oppo)
        print("成功的domains：", succ_domains)
        print("---------------------------------------")
    return avg_reward

# 在20个discrete domain评估
def evaluate_tournament(negotiation, opposition="low", eval_episodes=480, render=False, rl_agent_group=None, agent_group_1=None, args=None):
    rows = []
    sessions = []
    mix_factor = 0.5
    
    for (v,u) in zip(agent_group_1,rl_agent_group): 
        rl_agent = u
        opponent = v

        avg_reward = 0
        avg_round = 0
        avg_oppo = 0
        succ_avg_reward = 0            
        succ_avg_round = 0
        succ_avg_oppo = 0
        succ_counts = 0

        succ_domains = []
        row = []

        # [0, 0.25] 10, [0.25, 0.5] 6, [0.5, 1] 4 左闭右开
        allDomains = ["Acquisition", "Animal", "Coffee", "DefensiveCharms", "Camera", "DogChoosing", "FiftyFifty2013", \
                    "HouseKeeping", "Icecream", "Kitchen", "Laptop", "NiceOrDie", "Outfit", "planes", "SmartPhone", \
                    "Ultimatum", "Wholesaler", "Lunch"]
        allTeachers = [CUHK_agent, theFawkes, atlas3_agent, caduceus_agent, ponpoko_agent, agreeable_agent2018]

        teacher_evaluation = [0] * 7

        for j in range(eval_episodes):
            if render:
                print("----------- a new episode when evaluating ---------")

            teacher_W = choose_teacher(rl_agent=rl_agent, opponent=opponent, teachers=allTeachers, domains=[allDomains[j%len(allDomains)]])
            
            if args.domain_type == "REAL":
                negotiation = Negotiation(max_round=args.max_round, issue_num=3, domain_type=args.domain_type, domain_file=args.domain_file)
            elif args.domain_type == "DISCRETE":
                negotiation = Negotiation(max_round=args.max_round, domain_type=args.domain_type, domain_file=allDomains[j%len(allDomains)])

            negotiation.clear()

            negotiation.add(rl_agent)   # 两者的顺序在negotiation.reset()中会被打乱。
            negotiation.add(opponent)   # 这，仅适用于找应对各个对手，分别哪个rl model最强。

            if render:
                if negotiation.domain_type == 'DISCRETE':
                    print("DISCRETE domain : ", allDomains[j%len(allDomains)])
            negotiation.reset(opposition=opposition)

            last_offer = None
            accept = False
            current_player = 0
            episode_reward = 0
            episode_round = 0
            session = []

            for i in range(1, negotiation.max_round + 1):
                if render:
                    print("Round:", i)
                current_player = 1 - i % 2

                negotiation.agents_pool[current_player].set_t(i)
                negotiation.agents_pool[1 - current_player].set_t(i)
                if i == 1:
                    negotiation.agents_pool[current_player].receive(last_offer)
                
                if negotiation.agents_pool[current_player].__class__.__name__ == "RLAgent":
                    if render:
                        print("  RL agent's obs: ", rl_agent.obs)
                    # action = negotiation.agents_pool[current_player].act()
                    action = rl_agent.act(mix_factor)

                    last_utility = 0.5 * (action + 1) * (rl_agent.u_max - rl_agent.u_min) + rl_agent.u_min
                    rl_agent.s = last_utility
                    last_offer = rl_agent.gen_offer()
                else:
                    last_offer = negotiation.agents_pool[current_player].act()
                
                if (last_offer is None) and (negotiation.agents_pool[current_player].__class__.__name__ == "CUHKAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "HardHeadedAgent" \
                    or negotiation.agents_pool[current_player].__class__.__name__ == "YXAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "OMAC" \
                    or negotiation.agents_pool[current_player].__class__.__name__ == "AgentLG" or negotiation.agents_pool[current_player].__class__.__name__ == "ParsAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "Caduceus" or negotiation.agents_pool[current_player].__class__.__name__ == "Atlas3" or negotiation.agents_pool[current_player].__class__.__name__ == "PonPokoAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "AgreeableAgent2018" or negotiation.agents_pool[current_player].__class__.__name__ == "TheFawkes" or negotiation.agents_pool[current_player].__class__.__name__ == "Agent36"):
                    if negotiation.agents_pool[current_player].accept == True:                     
                        accept = True
                        episode_round = i
                        if render:
                            print(negotiation.agents_pool[current_player].name, "accept the offer.\n")                    
                        episode_reward = negotiation.agents_pool[1-current_player].utility_proposed[-1]
                        avg_oppo += negotiation.agents_pool[current_player].utility_received[-1]
                        succ_avg_reward += episode_reward
                        succ_avg_oppo += negotiation.agents_pool[current_player].utility_received[-1]
                        succ_counts += 1
                        succ_avg_round += episode_round
                        if allDomains[j%len(allDomains)] not in succ_domains:
                            succ_domains.append(allDomains[j%len(allDomains)])
                        break                
                    elif negotiation.agents_pool[current_player].terminate == True:  
                        if render: 
                            print(negotiation.agents_pool[current_player].name, "end the negotiation.")
                        episode_reward = 0
                    break
                elif last_offer is None:
                    print("Error exist: agent's offer is None.")
                    exit(-1)
                    
                negotiation.agents_pool[1 - current_player].receive(last_offer)
                if render:
                    print("  " + negotiation.agents_pool[current_player].name, "'s action", last_offer)
                    print("  utility to %s: %f, utility to %s: %f\n" % (negotiation.agents_pool[current_player].name,
                        get_utility(last_offer, negotiation.agents_pool[current_player].prefer, negotiation.agents_pool[current_player].condition, negotiation.agents_pool[current_player].domain_type, negotiation.agents_pool[current_player].issue_value),
                        negotiation.agents_pool[1 - current_player].name, get_utility(last_offer, negotiation.agents_pool[1 - current_player].prefer,
                        negotiation.agents_pool[1 - current_player].condition, negotiation.agents_pool[1 - current_player].domain_type, negotiation.agents_pool[1 - current_player].issue_value)))
                if i == negotiation.max_round:
                    episode_reward = 0
                if negotiation.agents_pool[1 - current_player].accept:
                    accept = True
                    episode_round = i + 1
                    if render:
                        print("Round:", i+1)
                        print("  "+negotiation.agents_pool[1 - current_player].name, "accept the offer.\n")
                    if last_offer is None:
                        episode_reward = 0
                    else:
                        episode_reward = get_utility(offer=last_offer, prefer=rl_agent.prefer, condition=rl_agent.condition, domain_type=rl_agent.domain_type, issue_value=rl_agent.issue_value)
                        avg_oppo +=  get_utility(offer=last_offer, prefer=opponent.prefer, condition=opponent.condition, domain_type=opponent.domain_type, issue_value=opponent.issue_value)
                        succ_avg_reward += episode_reward
                        succ_avg_oppo += get_utility(offer=last_offer, prefer=opponent.prefer, condition=opponent.condition, domain_type=opponent.domain_type, issue_value=opponent.issue_value)
                        succ_counts += 1
                        succ_avg_round += episode_round
                        if allDomains[j%len(allDomains)] not in succ_domains:
                            succ_domains.append(allDomains[j%len(allDomains)])
                    break

                if render:
                    print()

            if render:
                if accept:
                    print("Negotiation success")
                else:                
                    print("Negotiation failed")
                print("rl received reward: %f\n" % episode_reward)
            
            if accept == False:
                episode_round = args.max_round

            avg_reward += episode_reward
            avg_round += episode_round

            evaluation_reward_teacher_1 = evaluate_policy(teacher_1, opponent, domain=allDomains[j%len(allDomains)], eval_episodes=5, render=False)
            evaluation_reward_teacher_2 = evaluate_policy(teacher_2, opponent, domain=allDomains[j%len(allDomains)], eval_episodes=5, render=False)
            evaluation_reward_teacher_3 = evaluate_policy(teacher_3, opponent, domain=allDomains[j%len(allDomains)], eval_episodes=5, render=False)
            evaluation_reward_teacher_4 = evaluate_policy(teacher_4, opponent, domain=allDomains[j%len(allDomains)], eval_episodes=5, render=False)
            evaluation_reward_teacher_5 = evaluate_policy(teacher_5, opponent, domain=allDomains[j%len(allDomains)], eval_episodes=5, render=False)
            evaluation_reward_teacher_6 = evaluate_policy(teacher_6, opponent, domain=allDomains[j%len(allDomains)], eval_episodes=5, render=False)

            teacher_evaluation[1] += evaluation_reward_teacher_1
            teacher_evaluation[2] += evaluation_reward_teacher_2
            teacher_evaluation[3] += evaluation_reward_teacher_3
            teacher_evaluation[4] += evaluation_reward_teacher_4
            teacher_evaluation[5] += evaluation_reward_teacher_5
            teacher_evaluation[6] += evaluation_reward_teacher_6

            rl_agent.compute_p(mix_factor, teacher_evaluation, teacher_W)
            
            # session.append(last_offer)#最终的offer
            # session.append(avg_oppo)#对手得到的累计奖励
            # session.append(episode_reward)#我们得到的奖励
            # session.append(episode_round)#谈判进行的轮次

            # sessions.append(session)

        # df2 = pd.DataFrame(sessions, columns=['Offer', 'reward_oppo', 'reward_rl', 'round'])
        # df2.to_excel("Nego_session_fifty.xlsx")

        avg_reward /= eval_episodes
        avg_round /= eval_episodes
        avg_oppo /= eval_episodes

        if succ_counts != 0:
            succ_avg_reward /= succ_counts                
            succ_avg_round /= succ_counts
            succ_avg_oppo /= succ_counts

        # 记录eval_episodes个sessions的平均数据
        row.append(opponent.__class__.__name__)
        row.append(rl_agent.name)
        row.append(succ_avg_oppo)
        row.append(succ_avg_reward)
        row.append(avg_oppo)
        row.append(avg_reward)
        row.append(succ_avg_round)
        row.append(avg_round)
        
        row.append(succ_counts)
        row.append(eval_episodes)

        rows.append(row)

    df1 = pd.DataFrame(rows, columns=['Agent1', 'Agent2', 'utility1_succ', 'utility2_succ', 'utility1_total', 'utility2_total', 'avg_round_succ', 'avg_round_total', 'num_succ', 'num_total'])
    df1.to_excel("RL_Tournament.xlsx")

def evaluate_teacher(rl_agent, opponent, domain, eval_episodes=10, render=False):
    rl_agent = rl_agent
    opponent = opponent
    avg_reward = 0
    avg_round = 0
    avg_oppo = 0
    succ_avg_reward = 0
    succ_counts = 0
    succ_avg_oppo = 0
    domain_file = domain

    succ_domains = []
    
    receive_offer = {}
    number_offer = 0

    for j in range(eval_episodes):
        if render:
            print("----------- a new episode when evaluating ---------")

        if args.domain_type == "REAL":
            negotiation = Negotiation(max_round=args.max_round, issue_num=3, domain_type=args.domain_type, domain_file=None)
        elif args.domain_type == "DISCRETE":
            negotiation = Negotiation(max_round=args.max_round, domain_type=args.domain_type, domain_file=domain_file)

        negotiation.add(opponent)
        negotiation.add(rl_agent)
        negotiation.reset()

        if render:
            if negotiation.domain_type == 'DISCRETE':
                print("DISCRETE domain: ", negotiation.domain_file)

        last_offer = None
        accept = False
        current_player = 0
        episode_reward = 0
        episode_round = 0
        for i in range(1, negotiation.max_round + 1):
            if render:
                print("Round:", i)
            current_player = 1 - i % 2

            negotiation.agents_pool[current_player].set_t(i)
            negotiation.agents_pool[1 - current_player].set_t(i)
            if i == 1:
                negotiation.agents_pool[current_player].receive(last_offer)
            
            if negotiation.agents_pool[current_player].__class__.__name__ == "RLAgent":
                if render:
                    print("  RL agent's obs: ", rl_agent.obs)
                action = negotiation.agents_pool[current_player].act()

                last_utility = 0.5 * (action + 1) * (rl_agent.u_max - rl_agent.u_min) + rl_agent.u_min
                rl_agent.s = last_utility
                last_offer = rl_agent.gen_offer()
            else:
                last_offer = negotiation.agents_pool[current_player].act()
                number_offer += 1
                tmp_utility = get_utility(last_offer, negotiation.agents_pool[current_player].prefer, negotiation.agents_pool[current_player].condition, negotiation.agents_pool[current_player].domain_type, negotiation.agents_pool[current_player].issue_value)
                tmp_utility = round(tmp_utility, 2)
                if tmp_utility not in receive_offer:
                    receive_offer[tmp_utility] = 1
                else:
                    receive_offer[tmp_utility] += 1
            
            if (last_offer is None) and (negotiation.agents_pool[current_player].__class__.__name__ in Negotiation.ANAC_classname_list):
                if negotiation.agents_pool[current_player].accept == True:                     
                    accept = True
                    episode_round = i
                    if render:
                        print(negotiation.agents_pool[current_player].name, "accept the offer.\n")                    
                    episode_reward = negotiation.agents_pool[1-current_player].utility_proposed[-1]
                    avg_oppo +=  negotiation.agents_pool[current_player].utility_received[-1]
                    succ_avg_reward += episode_reward
                    succ_avg_oppo += negotiation.agents_pool[current_player].utility_received[-1]
                    succ_counts += 1
                    if negotiation.domain_file not in succ_domains:
                        succ_domains.append(negotiation.domain_file)
                    break                
                elif negotiation.agents_pool[current_player].terminate == True:  
                    if render: 
                        print(negotiation.agents_pool[current_player].name, "end the negotiation.")
                    episode_reward = 0
                break
            elif last_offer is None:
                print("Error exist: agent's offer is None.")
                exit(-1)

            negotiation.agents_pool[1 - current_player].receive(last_offer)
            if render:
                print("  " + negotiation.agents_pool[current_player].name, "'s action", last_offer)
                print("  utility to %s: %f, utility to %s: %f\n" % (negotiation.agents_pool[current_player].name,
                      get_utility(last_offer, negotiation.agents_pool[current_player].prefer, negotiation.agents_pool[current_player].condition, negotiation.agents_pool[current_player].domain_type, negotiation.agents_pool[current_player].issue_value),
                      negotiation.agents_pool[1 - current_player].name, get_utility(last_offer, negotiation.agents_pool[1 - current_player].prefer,
                      negotiation.agents_pool[1 - current_player].condition, negotiation.agents_pool[1 - current_player].domain_type, negotiation.agents_pool[1 - current_player].issue_value)))
            if i == negotiation.max_round:
                episode_reward = 0
            if negotiation.agents_pool[1 - current_player].accept:
                accept = True
                episode_round = i + 1
                if render:
                    print("Round:", i+1)
                    print("  "+negotiation.agents_pool[1 - current_player].name, "accept the offer.\n")
                if last_offer is None:
                    episode_reward = 0
                else:
                    episode_reward = get_utility(offer=last_offer, prefer=rl_agent.prefer, condition=rl_agent.condition, domain_type=rl_agent.domain_type, issue_value=rl_agent.issue_value)
                    avg_oppo +=  get_utility(offer=last_offer, prefer=opponent.prefer, condition=opponent.condition, domain_type=opponent.domain_type, issue_value=opponent.issue_value)
                    succ_avg_reward += episode_reward
                    succ_avg_oppo += get_utility(offer=last_offer, prefer=opponent.prefer, condition=opponent.condition, domain_type=opponent.domain_type, issue_value=opponent.issue_value)
                    succ_counts += 1
                    if negotiation.domain_file not in succ_domains:
                        succ_domains.append(negotiation.domain_file)
                break

            if render:
                print()

        if render:
            if accept:
                print("Negotiation success")
            else:                
                print("Negotiation failed")
            print("rl received reward: %f\n" % episode_reward)
        
        if accept == False:
            episode_round = args.max_round

        avg_reward += episode_reward
        avg_round += episode_round

    for key, value in receive_offer.items():
        receive_offer[key] = value / number_offer

    return receive_offer

def wasserstein(dictA, dictB):
    dictA_key = []
    dictA_value = []

    dictB_key = []
    dictB_value = []

    for key, value in dictA.items():
        dictA_key.append(key)
        dictA_value.append(value)

    for key, value in dictB.items():
        dictB_key.append(key)
        dictB_value.append(value)

    distance = wasserstein_distance(dictA_key, dictB_key, dictA_value, dictB_value)
    return distance

def choose_teacher(rl_agent, opponent, teachers, domains):

    teacher_W = [0] * 7
    teacher_distance = [0] * 7

    for domain in domains:
        student_domain = evaluate_teacher(rl_agent = rl_agent, opponent=opponent, domain=domain, eval_episodes=3, render=False)
        for i, teacher in enumerate(teachers):
            teacher_domain = evaluate_teacher(rl_agent = rl_agent, opponent=teacher, domain=domain, eval_episodes=3, render=False)
            teacher_distance[i] += wasserstein(student_domain, teacher_domain)
    #取倒数
    for distance in teacher_distance:
        if distance == 0:
            distance = 100000
        distance = 1/distance

    min_W = min(teacher_distance[1:])
    max_W = max(teacher_distance[1:])
    tmp = [0] * 7
        
    if min_W != max_W:
        for i in range(1, 7):
            tmp[i] = (teacher_distance[i] - min_W) / (max_W - min_W)
            
    sum = 0
    for i in range(1, 7):
        sum += np.exp(1 * tmp[i])
    for i in range(1, 7):
        teacher_W[i] = np.exp(1 * tmp[i]) / sum

    return teacher_W

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=9, type=int)                         # Sets PyTorch and Numpy seeds
    parser.add_argument("--max_round", default=30, type=int)                 # How many steps in an negotiation
    parser.add_argument("--save_models", action="store_true", default=True)    # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)               # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=128, type=int)                 # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                # Discount factor
    parser.add_argument("--tau", default=0.002, type=float)                    # Target network update rate
    parser.add_argument("--use_automatic_entropy_tuning", default=True, type=bool)
    parser.add_argument("--target_entropy", default=None, type=float)
    parser.add_argument("--gpu_no", default='0', type=str)                    # GPU number, -1 means CPU
    parser.add_argument("--eval_opposition" ,default="low", type=str)
    parser.add_argument("--domain_type", default="DISCRETE", type=str)             # "REAL" or "DISCRETE"
    parser.add_argument("--domain_file", default="Acquisition", type=str)     # Only the DISCRETE domain needs to specify this arg 
    args = parser.parse_args()

    if args.save_models and not os.path.exists("./sac_models"):
        os.makedirs("./sac_models")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    setup_seed(args.seed)

    teacher_1 = RLAgent(args.max_round, 'teacher_1', device, 3)
    teacher_1.load_teacher('1')

    teacher_2 = RLAgent(args.max_round, 'teacher_2', device, 3)
    teacher_2.load_teacher('2')

    teacher_3 = RLAgent(args.max_round, 'teacher_3', device, 3)
    teacher_3.load_teacher('3')

    teacher_4 = RLAgent(args.max_round, 'teacher_4', device, 3)
    teacher_4.load_teacher('4')

    teacher_5 = RLAgent(args.max_round, 'teacher_5', device, 3)
    teacher_5.load_teacher('5')

    teacher_6 = RLAgent(args.max_round, 'teacher_6', device, 3)
    teacher_6.load_teacher('6')

    if args.domain_type == "REAL":
        negotiation = Negotiation(max_round=args.max_round, issue_num=3, domain_type=args.domain_type, domain_file=args.domain_file)
    elif args.domain_type == "DISCRETE":
        negotiation = Negotiation(max_round=args.max_round, domain_type=args.domain_type, domain_file=args.domain_file)
    
    ponpoko_agent = PonPokoAgent(max_round=args.max_round, name="ponpoko agent")
    parscat_agent = ParsCat(max_round=args.max_round, name="parscat agent")
    atlas3_agent = Atlas3(max_round=args.max_round, name="atlas3 agent")
    agentlg_agent = AgentLG(max_round=args.max_round, name="AgentLG agent")
    omac_agent = OMAC(max_round=args.max_round, name="omac agent")
    CUHK_agent = CUHKAgent(max_round=args.max_round, name="CUHK agent")    
    HardHeaded_agent = HardHeadedAgent(max_round=args.max_round, name="HardHeaded agent")
    YX_Agent = YXAgent(max_round=args.max_round, name="YXAgent")
    ParsAgent_agent = ParsAgent(max_round=args.max_round, name="ParsAgent")
    caduceus_agent = Caduceus(max_round=args.max_round, name="Caduceus agent")
    agreeable_agent2018 = AgreeableAgent2018(max_round=args.max_round)
    agent36 = Agent36(max_round=args.max_round, name="Agent36")
    theFawkes = TheFawkes(max_round=args.max_round)
    caduceusDC16 = CaduceusDC16(max_round=args.max_round)
    Random_Agent = RandomAgent(max_round=args.max_round, name="random agent")
    Time_Agent = TimeAgent(max_round=args.max_round, name="timedependent agent")


    rl_agent_group = []

    offer_model_list = ["ponpoko", "yxagent", 'atlas3', 'parsagent', 'caduceus', 'agreeable', 'agent36']
    #offer_model_list = ['agent36']
    for i in range(len(offer_model_list)):
        rl_agent = RLAgent(max_round=args.max_round, name="rl agent", device=device)
        rl_agent.load(label=offer_model_list[i])
        rl_agent.name = "RL(against " + offer_model_list[i] + ")"
        rl_agent_group.append(rl_agent)

    
    # offer_model_list = ["cuhkagent", 'atlas3','agentlg']  # 现在只有这两个0.85模型yxagent猜测不会好，临时用这个

    agent_group_1 = [ponpoko_agent, YX_Agent, atlas3_agent, ParsAgent_agent, caduceus_agent, agreeable_agent2018, agent36]
    #agent_group_1 = [agent36]

    print("start testing ...")
    evaluate_tournament(negotiation, opposition=args.eval_opposition, eval_episodes=50, render=False, rl_agent_group=rl_agent_group, agent_group_1=agent_group_1, args=args)
