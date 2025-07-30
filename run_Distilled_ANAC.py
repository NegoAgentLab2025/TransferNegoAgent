import torch
import argparse
import numpy as np
from utils import get_utility
from negotiation import Negotiation
from distillation import DistillationAgent

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


def evaluate_policy(domain, distillation_label, opponent, eval_episodes=1000, render=False):
    print('domain: {}\n'.format(domain))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert_policies = ["agentlg", "atlas3", "caduceus", "cuhkagent", "hardheaded", "omac", "parsagent", "parscat", "ponpoko", "yxagent"]
    student = DistillationAgent(args.max_round, 'distillation model', expert_policies, device)
    student.load('./student_model')
    student.expert_label = distillation_label

    average_episode_reward = 0.0

    for j in range(eval_episodes):
        if render:
            print("----------- a new episode when evaluating, No.{}---------".format(j+1))

        if args.domain_type == "REAL":
            negotiation = Negotiation(max_round=args.max_round, issue_num=3, domain_type=args.domain_type, domain_file=None)
        elif args.domain_type == "DISCRETE":
            negotiation = Negotiation(max_round=args.max_round, domain_type=args.domain_type, domain_file=domain)

        negotiation.clear()
        negotiation.add(student)
        negotiation.add(opponent)
        negotiation.reset()                

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
            
            if negotiation.agents_pool[current_player].__class__.__name__ == "DistillationAgent":
                if render:
                    print("  Distilled-{}'s obs: {}".format(student.expert_label, negotiation.agents_pool[current_player].obs))
                action = negotiation.agents_pool[current_player].act()
                last_utility = 0.5 * (action + 1) * (negotiation.agents_pool[current_player].u_max - negotiation.agents_pool[current_player].u_min) + negotiation.agents_pool[current_player].u_min
                negotiation.agents_pool[current_player].s = last_utility
                last_offer = negotiation.agents_pool[current_player].gen_offer()
                if render:
                    print("  Distilled-{}'s target utility: {}".format(student.expert_label, last_utility))
            else:
                last_offer = negotiation.agents_pool[current_player].act()

            if (last_offer is None) and (negotiation.agents_pool[current_player].__class__.__name__ == "CUHKAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "HardHeadedAgent" \
                or negotiation.agents_pool[current_player].__class__.__name__ == "YXAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "OMAC" \
                or negotiation.agents_pool[current_player].__class__.__name__ == "AgentLG" or negotiation.agents_pool[current_player].__class__.__name__ == "ParsAgent" \
                    or negotiation.agents_pool[current_player].__class__.__name__ == "Caduceus" or negotiation.agents_pool[current_player].__class__.__name__ == "Atlas3" or negotiation.agents_pool[current_player].__class__.__name__ == "PonPokoAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "ParsCat"):
                if negotiation.agents_pool[current_player].accept == True:
                    accept = True
                    episode_round = i - 1
                    episode_reward = negotiation.agents_pool[1-current_player].utility_proposed[-1]
                    if render:
                        print(negotiation.agents_pool[current_player].name, "accept the offer.\n")

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
            # if i == negotiation.max_round:                            
            #     episode_reward = 0

            if negotiation.agents_pool[1 - current_player].accept:
                accept = True
                episode_round = i
                if(negotiation.agents_pool[1 - current_player].__class__.__name__ == 'DistillationAgent'):
                    episode_reward = negotiation.agents_pool[1 - current_player].utility_received[-1]
                else:
                    episode_reward = negotiation.agents_pool[current_player].utility_proposed[-1]
                if render:
                    print("Round:", episode_round)
                    print("  "+negotiation.agents_pool[1 - current_player].name, "accept the offer.\n")
                
                break
            if render:
                print()

        if render:
            print('episode reward:', episode_reward, 'episode_round:', episode_round)
            print()
            if accept:
                print("Negotiation success\n")
            else:
                print("Negotiation failed\n")

        average_episode_reward += episode_reward
    
    average_episode_reward /= eval_episodes    

    print('average episode reward: {}\n'.format(average_episode_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=3, type=int)
    parser.add_argument("--max_round", default=30, type=int)       
    parser.add_argument("--domain_type", default="DISCRETE", type=str)     
    parser.add_argument("--distillation_label", default='caduceus', type=str)
    parser.add_argument("--anac_agent", default='caduceus', type=str)
    parser.add_argument("--domain", default='DogChoosing', type=str)
    args = parser.parse_args()

    # 设置torch和numpy的seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    if args.anac_agent == "ponpoko":
        opponent =  ponpoko_agent
    elif args.anac_agent == "atlas3":
        opponent = atlas3_agent
    elif args.anac_agent == "parscat":
        opponent = parscat_agent
    elif args.anac_agent == "agentlg":
        opponent = agentlg_agent
    elif args.anac_agent == "omac":
        opponent = omac_agent
    elif args.anac_agent == "cuhkagent":
        opponent = CUHK_agent
    elif args.anac_agent == "hardheaded":
        opponent = HardHeaded_agent
    elif args.anac_agent == "yxagent":
        opponent = YX_Agent
    elif args.anac_agent == "parsagent":
        opponent = ParsAgent_agent
    elif args.anac_agent == "caduceus":
        opponent = caduceus_agent

    evaluate_policy(eval_episodes=10, render=True, domain=args.domain, distillation_label=args.distillation_label, opponent=opponent)
