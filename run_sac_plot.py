# -*- coding: UTF-8 -*-
#from torch.utils.tensorboard import SummaryWriter
from rl_plotter.logger import Logger
import numpy as np
import os
import argparse
import random
import torch
from utils import get_utility
from utils import ReplayBuffer
from rlagent import RLAgent
from negotiation import Negotiation
from atlas3 import Atlas3
from ponpokoagent import PonPokoAgent
from parscat import ParsCat
from agentlg import AgentLG
from omacagent import OMAC
from CUHKAgent import CUHKAgent
from HardHeaded import HardHeadedAgent
from YXAgent import YXAgent
from ParsAgent import ParsAgent

# All Domain training, single domain evaluation

# 在20个discrete domain评估
def evaluate_policy(rl_agent, opponent, eval_episodes=1000, render=False):
    rl_agent = rl_agent
    opponent = opponent
    avg_reward = 0
    avg_round = 0
    avg_oppo = 0
    succ_avg_reward = 0
    succ_counts = 0
    succ_avg_oppo = 0

    succ_domains = []

    # allDomains = ["Acquisition", "Animal", "Coffee", "DefensiveCharms", "Camera", "DogChoosing", "FiftyFifty2013", \
    #             "HouseKeeping", "Icecream", "Kitchen", "Laptop", "NiceOrDie", "Outfit", "planes", "SmartPhone", \
    #             "Ultimatum", "Wholesaler", "RentalHouse-B", "Barter-C", "Amsterdam-B"]

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
            negotiation = Negotiation(max_round=args.max_round, domain_type=args.domain_type, domain_file=args.domain)

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
            
            if (last_offer is None) and (negotiation.agents_pool[current_player].__class__.__name__ == "CUHKAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "HardHeadedAgent" \
                        or negotiation.agents_pool[current_player].__class__.__name__ == "YXAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "OMAC" \
                        or negotiation.agents_pool[current_player].__class__.__name__ == "AgentLG" or negotiation.agents_pool[current_player].__class__.__name__ == "ParsAgent" \
                        or negotiation.agents_pool[current_player].__class__.__name__ == "Caduceus" or negotiation.agents_pool[current_player].__class__.__name__ == "Atlas3" or negotiation.agents_pool[current_player].__class__.__name__ == "PonPokoAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "ParsCat"):
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="sac", type=str)
    parser.add_argument("--seed", default=2, type=int)                         # Sets PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=5e4, type=int)            # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=100, type=float)               # How often (episodes) we evaluate
    parser.add_argument("--max_round", default=30, type=int)                   # How many steps in an negotiation
    parser.add_argument("--max_timesteps", default=200000, type=float)        # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true", default=True)    # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)               # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=128, type=int)                 # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                # Discount factor
    parser.add_argument("--tau", default=0.002, type=float)                    # Target network update rate
    parser.add_argument("--use_automatic_entropy_tuning", default=True, type=bool)
    parser.add_argument("--target_entropy", default=None, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)		# Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)		# Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
    parser.add_argument("--gpu_no", default='0', type=str)                     # GPU number, -1 means CPU
    parser.add_argument("--offer_model", default="", type=str)                 # The model used for giving offer
    parser.add_argument("--mode", default="train", type=str)                   # Using train mode or test mode
    parser.add_argument("--oppo_type", default="atlas3", type=str)          # the opponent concess strategy in the negotiation: time, behavior, mix
    parser.add_argument("--domain_type", default="DISCRETE", type=str)         # "REAL" or "DISCRETE"
    parser.add_argument("--domain", default="Amsterdam-B", type=str)
    args = parser.parse_args()

    if args.save_models:
        if args.algo == 'td3' and not os.path.exists("./td3_models"):
            os.makedirs("./td3_models")
        elif args.algo == 'sac' and not os.path.exists("./sac_models"):
            os.makedirs("./sac_models")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    setup_seed(args.seed)

    # RL agent to be trained
    rl_agent = RLAgent(args.max_round, 'rl agent', device, 3, args.algo)
    if args.offer_model != "":
        rl_agent.load(args.offer_model, args.seed)

    # opponent
    ponpoko_agent = PonPokoAgent(max_round=args.max_round, name="ponpoko agent")
    parscat_agent = ParsCat(max_round=args.max_round, name="parscat agent")
    atlas3_agent = Atlas3(max_round=args.max_round, name="atlas3 agent")
    agentlg_agent = AgentLG(max_round=args.max_round, name="AgentLG agent")
    omac_agent = OMAC(max_round=args.max_round, name="omac agent")
    CUHK_agent = CUHKAgent(max_round=args.max_round, name="CUHK agent")    
    HardHeaded_agent = HardHeadedAgent(max_round=args.max_round, name="HardHeaded agent")
    YX_Agent = YXAgent(max_round=args.max_round, name="YXAgent")
    ParsAgent_agent = ParsAgent(max_round=args.max_round, name="ParsAgent")

    if args.oppo_type == "ponpoko":
        opponent =  ponpoko_agent
    elif args.oppo_type == "atlas3":
        opponent = atlas3_agent
    elif args.oppo_type == "parscat":
        opponent = parscat_agent
    elif args.oppo_type == "agentlg":
        opponent = agentlg_agent
    elif args.oppo_type == "omac":
        opponent = omac_agent
    elif args.oppo_type == "cuhkagent":
        opponent = CUHK_agent
    elif args.oppo_type == "hardheaded":
        opponent = HardHeaded_agent
    elif args.oppo_type == "yxagent":
        opponent = YX_Agent
    elif args.oppo_type == "parsagent":
        opponent = ParsAgent_agent


    if args.mode == "train":
        if args.save_models:
            if args.algo == 'sac' and not os.path.exists("./sac_models/{}/{}/seed{}".format(args.oppo_type, args.domain, str(args.seed))):
                os.makedirs("./sac_models/{}/{}/seed{}".format(args.oppo_type, args.domain, str(args.seed)))
            elif args.algo == 'td3' and not os.path.exists("./td3_models/{}/seed{}".format(args.oppo_type, str(args.seed))):
                os.makedirs("./td3_models/{}/seed{}".format(args.oppo_type, str(args.seed)))

        #logdir = './runs/runs_{}_{}_{}_{}'.format(args.algo, args.domain, args.oppo_type, str(args.seed))
        #writer = SummaryWriter(log_dir=logdir)
        logger = Logger(exp_name="your_exp_name", env_name="your_env_name", seed=args.seed)

        print("start training ...")
        offer_replay_buffer = ReplayBuffer()
        #evaluations = [evaluate_policy(rl_agent=rl_agent, opponent=opponent, eval_episodes=5, render=False)]
        #writer.add_scalar('reward', evaluate_policy(rl_agent=rl_agent, opponent=opponent, eval_episodes=5, render=False), 0)
        #evaluation_tmp = evaluate_policy(rl_agent=rl_agent, opponent=opponent, eval_episodes=5, render=False)
        #evaluations = [evaluation_tmp]
        #writer.add_scalar('reward', evaluation_tmp, 0)
        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        episode_num_last_eval = 0
        reward_records = []

        #allDomains = ["Acquisition", "Animal", "Coffee", "DefensiveCharms", "Camera", "DogChoosing", "FiftyFifty2013", "HouseKeeping", "Icecream", "Kitchen", "Laptop", "NiceOrDie", "Outfit", "planes", "SmartPhone", "Ultimatum", "Wholesaler", "RentalHouse-B", "Barter-C", "Amsterdam-B"]
        if args.oppo_type == 'parscat':
            allDomains = ["Acquisition", "Animal", "Coffee", "DefensiveCharms", "Camera", "DogChoosing", "FiftyFifty2013", "HouseKeeping", "Icecream", "Kitchen", "Laptop", "Outfit", "planes", "SmartPhone", "Wholesaler", "RentalHouse-B", "Barter-C", "Amsterdam-B"]

        while total_timesteps < args.max_timesteps:
            # A new episode begins.
            if args.domain_type == "REAL":
                negotiation = Negotiation(max_round=args.max_round, issue_num=3, domain_type=args.domain_type, domain_file=None)
            elif args.domain_type == "DISCRETE":
                #negotiation = Negotiation(max_round=args.max_round, domain_type=args.domain_type, domain_file=random.choice(allDomains), train_mode=True)
                negotiation = Negotiation(max_round=args.max_round, domain_type=args.domain_type, domain_file=args.domain, train_mode=True)

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            episode_num_last_eval += 1

            negotiation.add(rl_agent)
            negotiation.add(opponent)            
            negotiation.reset()

            last_offer = None
            current_player = 0
            obs = None
            action = None
            new_obs = None
            done = False

            for i in range(1, negotiation.max_round + 1):
                if i == 1:
                    negotiation.agents_pool[current_player].receive(last_offer)
                episode_timesteps += 1
                current_player = 1 - i % 2
                negotiation.agents_pool[current_player].set_t(i)
                negotiation.agents_pool[1 - current_player].set_t(i)

                if negotiation.agents_pool[current_player].__class__.__name__ == "RLAgent":
                    obs = negotiation.agents_pool[current_player].obs
                    if total_timesteps < args.start_timesteps:
                        action = np.random.uniform(-1, 1, [1])
                    else:
                        action = rl_agent.act()
                    # last_utility 为 网络输出的action转为有效范围[u_min, u.max]内的utility
                    last_utility = 0.5 * (action + 1) * (rl_agent.u_max - rl_agent.u_min) + rl_agent.u_min
                    rl_agent.s = last_utility
                    last_offer = rl_agent.gen_offer()
                else:
                    last_offer = negotiation.agents_pool[current_player].act()                

                reward = 0
                negotiation.agents_pool[1 - current_player].receive(last_offer)

                if negotiation.agents_pool[current_player].__class__.__name__ != "RLAgent":
                    new_obs = negotiation.agents_pool[1 - current_player].obs

                if episode_timesteps >= negotiation.max_round:
                    done = True
                    reward = -1                 
                    new_obs = obs

                if negotiation.agents_pool[1 - current_player].accept:
                    done = True
                    reward = get_utility(last_offer, rl_agent.prefer, rl_agent.condition, rl_agent.domain_type, rl_agent.issue_value)
                    new_obs = obs
                
                if (last_offer is None) and (negotiation.agents_pool[current_player].__class__.__name__ == "CUHKAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "HardHeadedAgent" \
                        or negotiation.agents_pool[current_player].__class__.__name__ == "YXAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "OMAC" \
                        or negotiation.agents_pool[current_player].__class__.__name__ == "AgentLG" or negotiation.agents_pool[current_player].__class__.__name__ == "ParsAgent" \
                        or negotiation.agents_pool[current_player].__class__.__name__ == "Caduceus" or negotiation.agents_pool[current_player].__class__.__name__ == "Atlas3" or negotiation.agents_pool[current_player].__class__.__name__ == "PonPokoAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "ParsCat"):
                    if negotiation.agents_pool[current_player].accept == True:
                        done = True
                        reward = negotiation.agents_pool[1-current_player].utility_proposed[-1]
                    elif negotiation.agents_pool[current_player].terminate == True:     
                        done = True
                        reward = -1
                elif last_offer is None:
                    print("Training code error existing: agent's offer is None.")
                    exit(-1)

                done_bool = float(done)

                if obs is not None and new_obs is not None and (obs != new_obs or done):
                    offer_replay_buffer.add((obs, new_obs, action, reward, done_bool))

                episode_reward += reward
                
                total_timesteps += 1
                timesteps_since_eval += 1
                
                if done:
                    break
            
            #reward_records.append(episode_reward)
            if total_timesteps != 0 and total_timesteps % 200 == 0:
                print('Total T:', total_timesteps, 'Episode Num:', episode_num, 'Episode T:', episode_timesteps, 'Reward:', episode_reward)

            if total_timesteps >= args.start_timesteps:
                rl_agent.train(offer_replay_buffer=offer_replay_buffer, iterations=episode_timesteps, batch_size=args.batch_size, discount=args.discount, tau=args.tau, policy_noise=args.policy_noise, noise_clip=args.noise_clip, policy_freq=args.policy_freq)
            if episode_num_last_eval >= args.eval_freq:
                episode_num_last_eval -= args.eval_freq
                #evaluations.append(evaluate_policy(rl_agent=rl_agent, opponent=opponent, eval_episodes=5, render=False))
                #writer.add_scalar('reward', evaluate_policy(rl_agent=rl_agent, opponent=opponent, eval_episodes=5, render=False), total_timesteps)
                #evaluation_tmp = evaluate_policy(rl_agent=rl_agent, opponent=opponent, eval_episodes=5, render=False)
                #evaluations.append(evaluation_tmp)
                #riter.add_scalar('reward', evaluation_tmp, episode_num)
                logger.update(score=[evaluate_policy(rl_agent=rl_agent, opponent=opponent, eval_episodes=5, render=False)], total_steps=total_timesteps)
                if args.save_models:
                    if args.algo == 'sac':
                        rl_agent.save(directory="./sac_models/{}/{}/seed{}".format(args.oppo_type, args.domain, str(args.seed)))
                    elif args.algo == 'td3':
                        rl_agent.save(directory="./td3_models/{}/seed{}".format(args.oppo_type, str(args.seed)))

        # Final evaluation
        #evaluations.append(evaluate_policy(rl_agent=rl_agent, opponent=opponent, eval_episodes=5, render=False))
        #writer.add_scalar('reward', evaluate_policy(rl_agent=rl_agent, opponent=opponent, eval_episodes=5, render=False), total_timesteps)
        #evaluation_tmp = evaluate_policy(rl_agent=rl_agent, opponent=opponent, eval_episodes=5, render=False)
        #evaluations.append(evaluation_tmp)
        #writer.add_scalar('reward', evaluation_tmp, episode_num)
        logger.update(score=[evaluate_policy(rl_agent=rl_agent, opponent=opponent, eval_episodes=5, render=False)], total_steps=total_timesteps)
        if args.save_models:
            if args.algo == 'sac':
                rl_agent.save(directory="./sac_models/{}/{}/seed{}".format(args.oppo_type, args.domain, str(args.seed)))
            elif args.algo == 'td3':
                rl_agent.save(directory="./td3_models/{}/seed{}".format(args.oppo_type, str(args.seed)))

        # reward_records = np.array(reward_records)
        # evaluations = np.array(evaluations)
        # reward_records = np.around(reward_records, decimals=2)
        # evaluations = np.around(evaluations, decimals=2)

        # #保存训练过程数据
        # with open(str(args.domain) + "_seed" + str(args.seed) + "_train.txt", "w") as f:      
        #     for i in reward_records:
        #         f.write(str(i))
        #         f.write("\n")
        # with open(str(args.domain) + "_seed" + str(args.seed) + "_evaluation.txt", "w") as f:      
        #     for i in evaluations:
        #         f.write(str(i))
        #         f.write("\n")       

    if args.mode == "test":
        print("start testing ...")
        evaluate_policy(rl_agent=rl_agent, opponent=opponent, eval_episodes=5, render=True)
