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
#from rlagent_Mix import RLAgent
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
from MiCROAgent import MiCROAgent


# 在20个discrete domain评估
def evaluate_policy(negotiation, opposition="low", eval_episodes=500, render=False, agent_group_1=None, agent_group_2=None, args=None):
    rows = []
    for v in agent_group_1:
        for u in agent_group_2:        

            agent = u
            opponent = v


            # [0, 0.25] 10, [0.25, 0.5] 6, [0.5, 1] 4 左闭右开
            allDomains = ["domain01", "domain09", "domain11", "domain13", "domain15", "domain16", "domain17", "domain19", "domain37", "domain39", "domain44", "domain46"]
        
            for domain in allDomains:
                if domain == "domain37":
                    if u in [caduceusDC16] or v in [caduceusDC16]:
                        break
                row = []
                avg_reward = 0
                avg_round = 0
                avg_oppo = 0
                succ_avg_reward = 0            
                succ_avg_round = 0
                succ_avg_oppo = 0
                succ_counts = 0

                succ_domains = []
                print(u, " ", v, " ", domain)
                for j in range(eval_episodes):
                    if render:
                        print("----------- a new episode when evaluating ---------")

                    if args.domain_type == "REAL":
                        negotiation = Negotiation(max_round=args.max_round, issue_num=3, domain_type=args.domain_type, domain_file=args.domain_file)
                    elif args.domain_type == "DISCRETE":
                        # negotiation = Negotiation(max_round=args.max_round, domain_type=args.domain_type, domain_file=allDomains[j%len(allDomains)])
                        negotiation = Negotiation(max_round=args.max_round, domain_type=args.domain_type, domain_file=domain)

                    negotiation.clear()

                    negotiation.add(agent)   # 两者的顺序在negotiation.reset()中会被打乱。
                    negotiation.add(opponent)   # 这，仅适用于找应对各个对手，分别哪个rl model最强。

                    if render:
                        if negotiation.domain_type == 'DISCRETE':
                            print("DISCRETE domain : ", domain)
                    negotiation.reset(opposition=opposition)

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
                        
                        # if negotiation.agents_pool[current_player].__class__.__name__ == "RLAgent":
                        #     if render:
                        #         print("  RL agent's obs: ", rl_agent.obs)
                        #     action = negotiation.agents_pool[current_player].act()

                        #     last_utility = 0.5 * (action + 1) * (rl_agent.u_max - rl_agent.u_min) + rl_agent.u_min
                        #     rl_agent.s = last_utility
                        #     last_offer = rl_agent.gen_offer()
                        # else:
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
                                succ_avg_round += episode_round
                                if domain not in succ_domains:
                                    succ_domains.append(domain)
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
                                episode_reward = get_utility(offer=last_offer, prefer=agent.prefer, condition=agent.condition, domain_type=agent.domain_type, issue_value=agent.issue_value)
                                avg_oppo +=  get_utility(offer=last_offer, prefer=opponent.prefer, condition=opponent.condition, domain_type=opponent.domain_type, issue_value=opponent.issue_value)
                                succ_avg_reward += episode_reward
                                succ_avg_oppo += get_utility(offer=last_offer, prefer=opponent.prefer, condition=opponent.condition, domain_type=opponent.domain_type, issue_value=opponent.issue_value)
                                succ_counts += 1
                                succ_avg_round += episode_round
                                if domain not in succ_domains:
                                    succ_domains.append(domain)
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
                    succ_avg_round /= succ_counts
                    succ_avg_oppo /= succ_counts

                # 记录eval_episodes个sessions的平均数据
                row.append(opponent.__class__.__name__)
                row.append(agent.__class__.__name__)
                row.append(domain)
                row.append(succ_avg_oppo)
                row.append(succ_avg_reward)
                row.append(avg_oppo)
                row.append(avg_reward)
                row.append(succ_avg_round)
                row.append(avg_round)
                row.append(succ_counts)
                row.append(eval_episodes)

                rows.append(row)

    # rows= np.array(rows)
    df1 = pd.DataFrame(rows, columns=['Agent1', 'Agent2', 'Domain', 'utility1_succ', 'utility2_succ', 'utility1_total', 'utility2_total', 'avg_round_succ', 'avg_round_total', 'num_succ', 'num_total'])
    df1.to_excel("Opponent_tournament.xlsx")


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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    micro_agent = MiCROAgent(max_round=args.max_round, name="micro agent")



    rl_agent_group = []

    offer_model_list = ["ponpoko", "yxagent", 'atlas3', 'parsagent', 'caduceus', 'agreeable', 'caduceusDC16', 'agent36']

    agent_group_1 = [agreeable_agent2018, agent36, ponpoko_agent, caduceusDC16, caduceus_agent, YX_Agent, atlas3_agent, ParsAgent_agent]
    agent_group_2 = [agreeable_agent2018, agent36, ponpoko_agent, caduceusDC16, caduceus_agent, YX_Agent, atlas3_agent, ParsAgent_agent]


    print("start testing ...")
    evaluate_policy(negotiation, opposition=args.eval_opposition, eval_episodes=10, render=False, agent_group_1=agent_group_1, agent_group_2=agent_group_2, args=args)
