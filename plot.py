import matplotlib
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import scipy
from scipy.signal import savgol_filter
import seaborn as sns
from matplotlib.ticker import FuncFormatter, ScalarFormatter, MultipleLocator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def paths(file_path):
    path_collection = []
    for dirpath, dirnames, filenames in os.walk(file_path):
        for file_name in filenames:
            #if file_name == 'out.json':
            #    fullpath = os.path.join(dirpath, file_name)
            #    path_collection.append(fullpath)
            fullpath = os.path.join(dirpath, file_name)
            path_collection.append(fullpath)
    return path_collection


def calcuate(tmp, epis=20000):
    #tmp = np.array(tmp['discount_reward_mean'])
    #tmp = np.array(tmp)
    tmp = np.array(tmp)
    epis = min(epis, len(tmp))
    tmp = tmp[0:epis]
    result = []
    for i in range(epis):
        result.append([np.mean(tmp[i])])
    tmp = result
    tmp_out = np.mean(tmp, axis=1)
    return tmp_out


def get_mean_std(path, epis=20000):
    reward_mean = []
    for file in paths(path):
        with open(file, 'r') as f:
            #print(file)
            #source = json.load(f)
            #source = list(f)
            source = [eval(x.strip('\n')) for x in f]
            source = scipy.signal.savgol_filter(source,39,3)
            #print(source)   
            reward_mean.append(calcuate(source, epis=epis))
    array = np.array(reward_mean)
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    return mean, std

def plot_line(mean, std, color, label, marker='*'):
    plt.plot(np.arange(0, len(mean)*1.45, 1.45)[::delta], mean[::delta], label=label, markevery=10000, linestyle='--',
             markeredgewidth=1, linewidth=3, marker=marker, markersize=20.0, color=color)
    plt.fill_between(np.arange(0, len(mean)*1.45, 1.45)[::delta], (mean - std)[::delta], (mean + std)[::delta], interpolate=True, linewidth=0.0, alpha=0.3, color=color)

def plot_line_transfer(mean, std, color, label, marker='*'):
    plt.plot(np.arange(len(mean))[::delta], mean[::delta], label=label, markevery=10000, linestyle='--',
             markeredgewidth=1, linewidth=3, marker=marker, markersize=20.0, color=color)
    plt.fill_between(np.arange(len(mean))[::delta], (mean - std)[::delta], (mean + std)[::delta], interpolate=True, linewidth=0.0, alpha=0.3, color=color)

def plot_block(opponent_agent, evaluate_domain, transfer, x_range):
    path_basline = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\{}\\baseline\\{}".format(opponent_agent, evaluate_domain)
    path_basline_1 = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\{}\\baseline_1\\{}".format(opponent_agent, evaluate_domain)
    path_transfer = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\{}\\{}\\{}".format(opponent_agent, transfer, evaluate_domain)

    # path_28 = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\{}\\mix_28\\{}".format(opponent_agent, evaluate_domain)
    # path_55 = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\{}\\mix_55\\{}".format(opponent_agent, evaluate_domain)
    # path_82 = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\{}\\mix_82\\{}".format(opponent_agent, evaluate_domain)

    mean_baseline, std_baseline = get_mean_std(path_basline, epis=20000)
    mean_baseline_1, std_baseline_1 = get_mean_std(path_basline_1, epis=20000)
    mean_transfer, std_transfer = get_mean_std(path_transfer, epis=20000)

    # mean_28, std_28 = get_mean_std(path_28, epis=20000)
    # mean_55, std_55 = get_mean_std(path_55, epis=20000)
    # mean_82, std_82 = get_mean_std(path_82, epis=20000)

    plot_line_transfer(mean_baseline, std_baseline, '#2878b5', "learn from scratch", marker='d')
    plot_line_transfer(mean_baseline_1, std_baseline_1, '#845EC2', "learn from teachers", marker='d')
    plot_line_transfer(mean_transfer, std_transfer, '#4D8076', "transfer", marker='d')
    # plot_line_transfer(mean_28, std_28, '#845EC2', "mix_28", marker='d')
    # plot_line_transfer(mean_55, std_55, '#B39CD0', "mix_55", marker='d')
    # plot_line_transfer(mean_82, std_82, '#4D8076', "mix_82", marker='d')

    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(xfmt)

    xminorLocator = MultipleLocator(2000)
    ax.xaxis.set_major_locator(xminorLocator)

    fig = plt.gcf()
    fig.set_size_inches(17, 11)

    x = range(0, x_range)

    plt.ylabel('Average reward', fontsize=20)
    plt.xlabel('Training episodes', fontsize=20)
    plt.xticks(x[::20], [100 * i for i in x[::20]], fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 1.0)
    plt.xlim(0, 408)
    # plt.ylim(-280, -70, 50)

    # plt.scatter(226, 0.64, s=200, color='green')
    # plt.scatter(207, 0.64, s=200, color='#EE7942')
    plt.legend(loc='lower right', fontsize=20)
    # plt.margins(0, 0)
    # plt.show()

    #plt.savefig('{}_{}_{}.pdf'.format(transfer, opponent_agent, evaluate_domain), bbox_inches='tight', pad_inches=0)
    plt.savefig('{}_{}_mix.pdf'.format(opponent_agent, evaluate_domain), bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    sns.set_style('darkgrid')
    # sns.set_context(font_scale=3)
    matplotlib.rcParams.update({'font.size': 40, 'font.family': 'serif'})

    delta = 1
    
    #mix_10
    #agreeable 216    agent36 261    thefawkes 366

    #plot_block("agreeable", "Acquisition", "mix_100", 213)
    #plot_block("agreeable", "Animal", "mix_100", 213)
    #plot_block("agreeable", "DefensiveCharms", "mix_100", 213)
    #plot_block("agreeable", "DogChoosing", "mix_100", 213)
    #plot_block("agreeable", "HouseKeeping", "mix_100", 213)
    #plot_block("agreeable", "Kitchen", "mix_100", 213)
    #plot_block("agreeable", "Outfit", "mix_100", 213)
    #plot_block("agreeable", "SmartPhone", "mix_100", 213)
    #plot_block("agreeable", "Wholesaler", "mix_100", 213)

    #mix-280    baseline-511    baseline_1-452
    #plot_block("agent36", "Acquisition", "mix", 280)
    #plot_block("agent36", "Amsterdam", "mix", 280)
    #plot_block("agent36", "Animal", "mix", 280)
    #plot_block("agent36", "Barter", "mix", 280)
    #plot_block("agent36", "Camera", "mix", 280)
    #plot_block("agent36", "Coffee", "mix", 280)
    #plot_block("agent36", "DefensiveCharms", "mix", 280)
    #plot_block("agent36", "DogChoosing", "mix", 280)
    #plot_block("agent36", "FiftyFifty2013", "mix", 280)
    #plot_block("agent36", "HouseKeeping", "mix", 280)
    #plot_block("agent36", "Icecream", "mix", 280)
    #plot_block("agent36", "Kitchen", "mix", 280)
    #plot_block("agent36", "Laptop", "mix", 280)
    #plot_block("agent36", "NiceOrDie", "mix", 280)
    #plot_block("agent36", "Outfit", "mix", 280)
    #plot_block("agent36", "planes", "mix", 280)
    #plot_block("agent36", "RentalHouse", "mix", 280)
    #plot_block("agent36", "SmartPhone", "mix", 280)
    #plot_block("agent36", "Ultimatum", "mix", 280)
    #plot_block("agent36", "Wholesaler", "mix", 280)

    #plot_block("thefawkes", "Acquisition", "mix_100", 358)
    #plot_block("thefawkes", "Animal", "mix_100", 358)
    #plot_block("thefawkes", "DefensiveCharms", "mix_100", 358)
    #plot_block("thefawkes", "DogChoosing", "mix_100", 358)
    #plot_block("thefawkes", "HouseKeeping", "mix_100", 358)
    #plot_block("thefawkes", "Kitchen", "mix_100", 358)
    #plot_block("thefawkes", "Outfit", "mix_100", 358)
    #plot_block("thefawkes", "SmartPhone", "mix_100", 358)
    #plot_block("thefawkes", "Wholesaler", "mix_100", 358)

    #plot_block("caduceusDC16", "Acquisition", "mix_1", 176)
    #plot_block("caduceusDC16", "Animal", "mix_1", 176)
    #plot_block("caduceusDC16", "DefensiveCharms", "mix_1", 176)
    #plot_block("caduceusDC16", "DogChoosing", "mix_1", 176)
    #plot_block("caduceusDC16", "HouseKeeping", "mix_1", 176)
    #plot_block("caduceusDC16", "Kitchen", "mix_1", 176)
    #plot_block("caduceusDC16", "Outfit", "mix_1", 176)
    #plot_block("caduceusDC16", "SmartPhone", "mix_1", 176)
    #plot_block("caduceusDC16", "Wholesaler", "mix_1", 176)

    #213
    #plot_block("caduceus", "Acquisition", "mix", 171)
    #plot_block("caduceus", "Animal", "mix", 171)
    #plot_block("caduceus", "DefensiveCharms", "mix", 171)
    #plot_block("caduceus", "DogChoosing", "mix", 171)
    #plot_block("caduceus", "HouseKeeping", "mix", 171)
    #plot_block("caduceus", "Kitchen", "mix", 171)
    #plot_block("caduceus", "Outfit", "mix", 171)
    #plot_block("caduceus", "SmartPhone", "mix", 171)
    #plot_block("caduceus", "Wholesaler", "mix", 171)

    #mix-254   baseline-408    baseline_1-283
    #plot_block("parsagent", "Acquisition", "mix", 254)
    #plot_block("parsagent", "Amsterdam", "mix", 254)
    #plot_block("parsagent", "Animal", "mix", 254)
    #plot_block("parsagent", "Barter", "mix", 254)
    #plot_block("parsagent", "Camera", "mix", 254)
    #plot_block("parsagent", "Coffee", "mix", 254)
    #plot_block("parsagent", "DefensiveCharms", "mix", 254)
    #plot_block("parsagent", "DogChoosing", "mix", 254)
    #plot_block("parsagent", "FiftyFifty2013", "mix", 254)
    #plot_block("parsagent", "HouseKeeping", "mix", 254)
    #plot_block("parsagent", "Icecream", "mix", 254)
    #plot_block("parsagent", "Kitchen", "mix", 254)
    #plot_block("parsagent", "Laptop", "mix", 254)
    #plot_block("parsagent", "NiceOrDie", "mix", 254)
    #plot_block("parsagent", "Outfit", "mix", 254)
    #plot_block("parsagent", "planes", "mix", 254)
    #plot_block("parsagent", "RentalHouse", "mix", 254)
    #plot_block("parsagent", "SmartPhone", "mix", 254)
    #plot_block("parsagent", "Ultimatum", "mix", 254)
    #plot_block("parsagent", "Wholesaler", "mix", 254)

    #263
    #plot_block("ponpoko", "Acquisition", "mix", 279)
    #plot_block("ponpoko", "Animal", "mix", 279)
    #plot_block("ponpoko", "DefensiveCharms", "mix", 279)
    #plot_block("ponpoko", "DogChoosing", "mix", 279)
    #plot_block("ponpoko", "HouseKeeping", "mix", 279)
    #plot_block("ponpoko", "Kitchen", "mix", 279)
    #plot_block("ponpoko", "Outfit", "mix", 279)
    #plot_block("ponpoko", "SmartPhone", "mix", 279)
    #plot_block("ponpoko", "Wholesaler", "mix", 279)

    #mix-354    baseline-433    baseline_1-284
    #plot_block("yxagent", "Acquisition", "mix", 354)
    #plot_block("yxagent", "Amsterdam", "mix", 354)
    #plot_block("yxagent", "Animal", "mix", 354)
    #plot_block("yxagent", "Barter", "mix", 354)
    #plot_block("yxagent", "Camera", "mix", 354)
    #plot_block("yxagent", "Coffee", "mix", 354)
    #plot_block("yxagent", "DefensiveCharms", "mix", 354)
    #plot_block("yxagent", "DogChoosing", "mix", 354)
    #plot_block("yxagent", "FiftyFifty2013", "mix", 354)
    #plot_block("yxagent", "HouseKeeping", "mix", 354)
    #plot_block("yxagent", "Icecream", "mix", 354)
    #plot_block("yxagent", "Kitchen", "mix", 354)
    #plot_block("yxagent", "Laptop", "mix", 354)
    #plot_block("yxagent", "NiceOrDie", "mix", 354)
    #plot_block("yxagent", "Outfit", "mix", 354)
    #plot_block("yxagent", "planes", "mix", 354)
    #plot_block("yxagent", "RentalHouse", "mix", 354)
    #plot_block("yxagent", "SmartPhone", "mix", 354)
    #plot_block("yxagent", "Ultimatum", "mix", 354)
    #plot_block("yxagent", "Wholesaler", "mix", 354)

    #mix-367   #baseline-460    baseline_1-387
    #plot_block("omac", "Acquisition", "mix", 367)
    #plot_block("omac", "Amsterdam", "mix", 367)
    #plot_block("omac", "Animal", "mix", 367)
    #plot_block("omac", "Barter", "mix", 367)
    #plot_block("omac", "Camera", "mix", 367)
    #plot_block("omac", "Coffee", "mix", 367)
    #plot_block("omac", "DefensiveCharms", "mix", 367)
    #plot_block("omac", "DogChoosing", "mix", 367)
    #plot_block("omac", "FiftyFifty2013", "mix", 367)
    #plot_block("omac", "HouseKeeping", "mix", 367)
    #plot_block("omac", "Icecream", "mix", 367)
    #plot_block("omac", "Kitchen", "mix", 367)
    #plot_block("omac", "Laptop", "mix", 367)
    #plot_block("omac", "NiceOrDie", "mix", 367)
    #plot_block("omac", "Outfit", "mix", 367)
    #plot_block("omac", "planes", "mix", 367)
    #plot_block("omac", "RentalHouse", "mix", 367)
    #plot_block("omac", "SmartPhone", "mix", 367)
    #plot_block("omac", "Ultimatum", "mix", 367)
    #plot_block("omac", "Wholesaler", "mix", 367)

    #mix-328    baseline-413    baseline_1-354
    #plot_block("agentlg", "Acquisition", "mix", 328)
    #plot_block("agentlg", "Amsterdam", "mix", 328)
    #plot_block("agentlg", "Animal", "mix", 328)
    #plot_block("agentlg", "Barter", "mix", 328)
    #plot_block("agentlg", "Camera", "mix", 328)
    #plot_block("agentlg", "Coffee", "mix", 328)
    #plot_block("agentlg", "DefensiveCharms", "mix", 328)
    #plot_block("agentlg", "DogChoosing", "mix", 328)
    #plot_block("agentlg", "FiftyFifty2013", "mix", 328)
    #plot_block("agentlg", "HouseKeeping", "mix", 328)
    #plot_block("agentlg", "Icecream", "mix", 328)
    #plot_block("agentlg", "Kitchen", "mix", 328)
    #plot_block("agentlg", "Laptop", "mix", 328)
    #plot_block("agentlg", "NiceOrDie", "mix", 328)
    #plot_block("agentlg", "Outfit", "mix", 328)
    #plot_block("agentlg", "planes", "mix", 328)
    #plot_block("agentlg", "RentalHouse", "mix", 328)
    #plot_block("agentlg", "SmartPhone", "mix", 328)
    #plot_block("agentlg", "Ultimatum", "mix", 328)
    #plot_block("agentlg", "Wholesaler", "mix", 328)

    #mix-216   #baseline-398    baseline_1-410
    #plot_block("hardheaded", "Acquisition", "mix", 216)
    #plot_block("hardheaded", "Amsterdam", "mix", 216)
    #plot_block("hardheaded", "Animal", "mix", 216)
    #plot_block("hardheaded", "Barter", "mix", 216)
    #plot_block("hardheaded", "Camera", "mix", 216)
    #plot_block("hardheaded", "Coffee", "mix", 216)
    #plot_block("hardheaded", "DefensiveCharms", "mix", 216)
    #plot_block("hardheaded", "DogChoosing", "mix", 216)
    #plot_block("hardheaded", "FiftyFifty2013", "mix", 216)
    #plot_block("hardheaded", "HouseKeeping", "mix", 216)
    #plot_block("hardheaded", "Icecream", "mix", 216)
    #plot_block("hardheaded", "Kitchen", "mix", 216)
    #plot_block("hardheaded", "Laptop", "mix", 216)
    #plot_block("hardheaded", "NiceOrDie", "mix", 216)
    #plot_block("hardheaded", "Outfit", "mix", 216)
    #plot_block("hardheaded", "planes", "mix", 216)
    #plot_block("hardheaded", "RentalHouse", "mix", 216)
    #plot_block("hardheaded", "SmartPhone", "mix", 216)
    #plot_block("hardheaded", "Ultimatum", "mix", 216)
    plot_block("hardheaded", "Wholesaler", "mix", 216)

    #transfer
    # transfer_agentlg_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agentlg\\transfer\\DefensiveCharms\\rewards" #229
    # transfer_agentlg_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agentlg\\transfer\\FiftyFifty2013\\rewards" #229
    # transfer_agentlg_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agentlg\\transfer\\HouseKeeping\\rewards" #229
    # transfer_agentlg_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agentlg\\transfer\\Kitchen\\rewards" #229
    # transfer_agentlg_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agentlg\\transfer\\Lunch\\rewards" #229
    # transfer_agentlg_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agentlg\\transfer\\Outfit\\rewards" #229

    # transfer_hardheaded_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\hardheaded\\transfer\\DefensiveCharms\\rewards" #356
    # transfer_hardheaded_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\hardheaded\\transfer\\FiftyFifty2013\\rewards" #356
    # transfer_hardheaded_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\hardheaded\\transfer\\HouseKeeping\\rewards" #356
    # transfer_hardheaded_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\hardheaded\\transfer\\Kitchen\\rewards" #356
    # transfer_hardheaded_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\hardheaded\\transfer\\Lunch\\rewards" #356
    # transfer_hardheaded_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\hardheaded\\transfer\\Outfit\\rewards" #356

    # transfer_caduceus_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceus\\transfer\\DefensiveCharms\\rewards" #264
    # transfer_caduceus_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceus\\transfer\\FiftyFifty2013\\rewards" #264
    # transfer_caduceus_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceus\\transfer\\HouseKeeping\\rewards" #264
    # transfer_caduceus_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceus\\transfer\\Kitchen\\rewards" #264
    # transfer_caduceus_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceus\\transfer\\Lunch\\rewards" #264
    # transfer_caduceus_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceus\\transfer\\Outfit\\rewards" #264

    # transfer_parsagent_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parsagent\\transfer\\DefensiveCharms\\rewards" #228
    # transfer_parsagent_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parsagent\\transfer\\FiftyFifty2013\\rewards" #228
    # transfer_parsagent_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parsagent\\transfer\\HouseKeeping\\rewards" #228
    # transfer_parsagent_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parsagent\\transfer\\Kitchen\\rewards" #228
    # transfer_parsagent_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parsagent\\transfer\\Lunch\\rewards" #228
    # transfer_parsagent_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parsagent\\transfer\\Outfit\\rewards" #228

    # transfer_ponpoko_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\ponpoko\\transfer\\DefensiveCharms\\rewards" #196
    # transfer_ponpoko_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\ponpoko\\transfer\\FiftyFifty2013\\rewards" #196
    # transfer_ponpoko_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\ponpoko\\transfer\\HouseKeeping\\rewards" #196
    # transfer_ponpoko_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\ponpoko\\transfer\\Kitchen\\rewards" #196
    # transfer_ponpoko_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\ponpoko\\transfer\\Lunch\\rewards" #196
    # transfer_ponpoko_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\ponpoko\\transfer\\Outfit\\rewards" #196

    # transfer_yxagent_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\yxagent\\transfer\\DefensiveCharms\\rewards" #294
    # transfer_yxagent_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\yxagent\\transfer\\FiftyFifty2013\\rewards" #294
    # transfer_yxagent_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\yxagent\\transfer\\HouseKeeping\\rewards" #294
    # transfer_yxagent_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\yxagent\\transfer\\Kitchen\\rewards" #294
    # transfer_yxagent_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\yxagent\\transfer\\Lunch\\rewards" #294
    # transfer_yxagent_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\yxagent\\transfer\\Outfit\\rewards" #294

    # transfer_agreeable_acquisition = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agreeable\\transfer\\Acquisition\\rewards" #440
    # transfer_agreeable_animal = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agreeable\\transfer\\Animal\\rewards" #440
    # transfer_agreeable_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agreeable\\transfer\\DefensiveCharms\\rewards" #440
    # transfer_agreeable_dog = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agreeable\\transfer\\DogChoosing\\rewards" #440
    # transfer_agreeable_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agreeable\\transfer\\HouseKeeping\\rewards" #440
    # transfer_agreeable_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agreeable\\transfer\\Kitchen\\rewards" #440
    # transfer_agreeable_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agreeable\\transfer\\Outfit\\rewards" #440
    # transfer_agreeable_smart = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agreeable\\transfer\\SmartPhone\\rewards" #440
    # transfer_agreeable_whole = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agreeable\\transfer\\Wholesaler\\rewards" #440

    # transfer_caduceusDC16_acquisition = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\caduceusDC16\\transfer\\Acquisition\\rewards" #402
    # transfer_caduceusDC16_animal = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\caduceusDC16\\transfer\\Animal\\rewards" #402
    # transfer_caduceusDC16_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\caduceusDC16\\transfer\\DefensiveCharms\\rewards" #402
    # transfer_caduceusDC16_dog = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\caduceusDC16\\transfer\\DogChoosing\\rewards" #402
    # transfer_caduceusDC16_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\caduceusDC16\\transfer\\HouseKeeping\\rewards" #402
    # transfer_caduceusDC16_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\caduceusDC16\\transfer\\Kitchen\\rewards" #402
    # transfer_caduceusDC16_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\caduceusDC16\\transfer\\Outfit\\rewards" #402
    # transfer_caduceusDC16_smart = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\caduceusDC16\\transfer\\SmartPhone\\rewards" #402
    # transfer_caduceusDC16_whole = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\caduceusDC16\\transfer\\Wholesaler\\rewards" #402

    # transfer_thefawkes_acquisition = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\transfer\\Acquisition\\rewards" #211
    # transfer_thefawkes_animal = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\transfer\\Animal\\rewards" #211
    # transfer_thefawkes_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\transfer\\DefensiveCharms\\rewards" #211
    # transfer_thefawkes_dog = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\transfer\\DogChoosing\\rewards" #211
    # transfer_thefawkes_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\transfer\\HouseKeeping\\rewards" #211
    # transfer_thefawkes_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\transfer\\Kitchen\\rewards" #211
    # transfer_thefawkes_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\transfer\\Outfit\\rewards" #211
    # transfer_thefawkes_smart = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\transfer\\SmartPhone\\rewards" #211
    # transfer_thefawkes_whole = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\transfer\\Wholesaler\\rewards" #211

    # transfer_agent36_acquisition = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agent36\\transfer\\Acquisition\\rewards" #540
    # transfer_agent36_animal = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agent36\\transfer\\Animal\\rewards" #540
    # transfer_agent36_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agent36\\transfer\\DefensiveCharms\\rewards" #540
    # transfer_agent36_dog = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agent36\\transfer\\DogChoosing\\rewards" #540
    # transfer_agent36_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agent36\\transfer\\HouseKeeping\\rewards" #540
    # transfer_agent36_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agent36\\transfer\\Kitchen\\rewards" #540
    # transfer_agent36_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agent36\\transfer\\Outfit\\rewards" #540
    # transfer_agent36_smart = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agent36\\transfer\\SmartPhone\\rewards" #540
    # transfer_agent36_whole = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\change_teacher\\agent36\\transfer\\Wholesaler\\rewards" #540


    # #path_transfer = "D:\\multi_issue_negotiation_simple\\txt\\ponpoko\\planes\\transfer\\rewards"
    # path_agentlg_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agentlg\\DefensiveCharms\\rewards" #609
    # path_agentlg_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agentlg\\FiftyFifty2013\\rewards" #609
    # path_agentlg_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agentlg\\HouseKeeping\\rewards" #609
    # path_agentlg_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agentlg\\Kitchen\\rewards" #609
    # path_agentlg_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agentlg\\Lunch\\rewards" #609
    # path_agentlg_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agentlg\\Outfit\\rewards" #609

    # path_atlas3_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\atlas3\\DefensiveCharms\\rewards" #325
    # path_atlas3_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\atlas3\\FiftyFifty2013\\rewards" #325
    # path_atlas3_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\atlas3\\HouseKeeping\\rewards" #325
    # path_atlas3_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\atlas3\\Kitchen\\rewards" #325
    # path_atlas3_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\atlas3\\Lunch\\rewards" #325
    # path_atlas3_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\atlas3\\Outfit\\rewards" #325

    # path_cuhkagent_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\cuhkagent\\DefensiveCharms\\rewards" #355
    # path_cuhkagent_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\cuhkagent\\FiftyFifty2013\\rewards" #355
    # path_cuhkagent_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\cuhkagent\\HouseKeeping\\rewards" #355
    # path_cuhkagent_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\cuhkagent\\Kitchen\\rewards" #355
    # path_cuhkagent_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\cuhkagent\\Lunch\\rewards" #355
    # path_cuhkagent_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\cuhkagent\\Outfit\\rewards" #355

    # path_hardheaded_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\hardheaded\\DefensiveCharms\\rewards" #466
    # path_hardheaded_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\hardheaded\\FiftyFifty2013\\rewards" #466
    # path_hardheaded_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\hardheaded\\HouseKeeping\\rewards" #466
    # path_hardheaded_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\hardheaded\\Kitchen\\rewards" #466
    # path_hardheaded_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\hardheaded\\Lunch\\rewards" #466
    # path_hardheaded_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\hardheaded\\Outfit\\rewards" #466

    # path_omac_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\omac\\DefensiveCharms\\rewards" #623
    # path_omac_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\omac\\FiftyFifty2013\\rewards" #623
    # path_omac_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\omac\\HouseKeeping\\rewards" #623
    # path_omac_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\omac\\Kitchen\\rewards" #623
    # path_omac_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\omac\\Lunch\\rewards" #623
    # path_omac_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\omac\\Outfit\\rewards" #623

    # path_parscat_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parscat\\DefensiveCharms\\rewards" #256
    # path_parscat_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parscat\\FiftyFifty2013\\rewards" #256
    # path_parscat_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parscat\\HouseKeeping\\rewards" #256
    # path_parscat_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parscat\\Kitchen\\rewards" #256
    # path_parscat_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parscat\\Lunch\\rewards" #256
    # path_parscat_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parscat\\Outfit\\rewards" #256

    # path_caduceus_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceus\\DefensiveCharms\\rewards" #323
    # path_caduceus_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceus\\FiftyFifty2013\\rewards" #323
    # path_caduceus_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceus\\HouseKeeping\\rewards" #323
    # path_caduceus_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceus\\Kitchen\\rewards" #323
    # path_caduceus_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceus\\Lunch\\rewards" #323
    # path_caduceus_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceus\\Outfit\\rewards" #323

    # path_parsagent_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parsagent\\DefensiveCharms\\rewards" #418
    # path_parsagent_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parsagent\\FiftyFifty2013\\rewards" #418
    # path_parsagent_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parsagent\\HouseKeeping\\rewards" #418
    # path_parsagent_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parsagent\\Kitchen\\rewards" #418
    # path_parsagent_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parsagent\\Lunch\\rewards" #418
    # path_parsagent_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\parsagent\\Outfit\\rewards" #418

    # path_ponpoko_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\ponpoko\\DefensiveCharms\\rewards" #321
    # path_ponpoko_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\ponpoko\\FiftyFifty2013\\rewards" #321
    # path_ponpoko_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\ponpoko\\HouseKeeping\\rewards" #321
    # path_ponpoko_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\ponpoko\\Kitchen\\rewards" #321
    # path_ponpoko_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\ponpoko\\Lunch\\rewards" #321
    # path_ponpoko_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\ponpoko\\Outfit\\rewards" #321

    # path_yxagent_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\yxagent\\DefensiveCharms\\rewards" #431
    # path_yxagent_fifty = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\yxagent\\FiftyFifty2013\\rewards" #431
    # path_yxagent_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\yxagent\\HouseKeeping\\rewards" #431
    # path_yxagent_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\yxagent\\Kitchen\\rewards" #431
    # path_yxagent_lunch = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\yxagent\\Lunch\\rewards" #431
    # path_yxagent_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\yxagent\\Outfit\\rewards" #431

    # path_agreeable_acquisition = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agreeable\\Acquisition\\rewards" #165
    # path_agreeable_animal = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agreeable\\Animal\\rewards" #165
    # path_agreeable_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agreeable\\DefensiveCharms\\rewards" #165
    # path_agreeable_dog = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agreeable\\DogChoosing\\rewards" #165
    # path_agreeable_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agreeable\\HouseKeeping\\rewards" #165
    # path_agreeable_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agreeable\\Kitchen\\rewards" #165
    # path_agreeable_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agreeable\\Outfit\\rewards" #165
    # path_agreeable_smart = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agreeable\\SmartPhone\\rewards" #165
    # path_agreeable_whole = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agreeable\\Wholesaler\\rewards" #165

    # path_caduceusDC16_acquisition = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceusDC16\\Acquisition\\rewards" #170
    # path_caduceusDC16_animal = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceusDC16\\Animal\\rewards" #170
    # path_caduceusDC16_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceusDC16\\DefensiveCharms\\rewards" #170
    # path_caduceusDC16_dog = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceusDC16\\DogChoosing\\rewards" #170
    # path_caduceusDC16_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceusDC16\\HouseKeeping\\rewards" #170
    # path_caduceusDC16_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceusDC16\\Kitchen\\rewards" #170
    # path_caduceusDC16_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceusDC16\\Outfit\\rewards" #170
    # path_caduceusDC16_smart = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceusDC16\\SmartPhone\\rewards" #170
    # path_caduceusDC16_whole = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\caduceusDC16\\Wholesaler\\rewards" #170

    # path_thefawkes_acquisition = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\Acquisition\\rewards" #206
    # path_thefawkes_animal = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\Animal\\rewards" #206
    # path_thefawkes_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\DefensiveCharms\\rewards" #206
    # path_thefawkes_dog = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\DogChoosing\\rewards" #206
    # path_thefawkes_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\HouseKeeping\\rewards" #206
    # path_thefawkes_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\Kitchen\\rewards" #206
    # path_thefawkes_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\Outfit\\rewards" #206
    # path_thefawkes_smart = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\SmartPhone\\rewards" #206
    # path_thefawkes_whole = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\thefawkes\\Wholesaler\\rewards" #206

    # path_agent36_acquisition = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agent36\\Acquisition\\rewards" #238
    # path_agent36_animal = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agent36\\Animal\\rewards" #238
    # path_agent36_defensive = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agent36\\DefensiveCharms\\rewards" #238
    # path_agent36_dog = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agent36\\DogChoosing\\rewards" #238
    # path_agent36_house = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agent36\\HouseKeeping\\rewards" #238
    # path_agent36_kitchen = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agent36\\Kitchen\\rewards" #238
    # path_agent36_outfit = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agent36\\Outfit\\rewards" #238
    # path_agent36_smart = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agent36\\SmartPhone\\rewards" #238
    # path_agent36_whole = "D:\\multi_issue_negotiation_simple\\txt\\opponents\\agent36\\Wholesaler\\rewards" #238

    # #transfer
    # mean_transfer_agentlg_defensive, std_transfer_agentlg_defensive = get_mean_std(transfer_agentlg_defensive, epis=20000)
    # mean_transfer_agentlg_fifty, std_transfer_agentlg_fifty = get_mean_std(transfer_agentlg_fifty, epis=20000)
    # mean_transfer_agentlg_house, std_transfer_agentlg_house = get_mean_std(transfer_agentlg_house, epis=20000)
    # mean_transfer_agentlg_kitchen, std_transfer_agentlg_kitchen = get_mean_std(transfer_agentlg_kitchen, epis=20000)
    # mean_transfer_agentlg_lunch, std_transfer_agentlg_lunch = get_mean_std(transfer_agentlg_lunch, epis=20000)
    # mean_transfer_agentlg_outfit, std_transfer_agentlg_outfit = get_mean_std(transfer_agentlg_outfit, epis=20000)

    # mean_transfer_hardheaded_defensive, std_transfer_hardheaded_defensive = get_mean_std(transfer_hardheaded_defensive, epis=20000)
    # mean_transfer_hardheaded_fifty, std_transfer_hardheaded_fifty = get_mean_std(transfer_hardheaded_fifty, epis=20000)
    # mean_transfer_hardheaded_house, std_transfer_hardheaded_house = get_mean_std(transfer_hardheaded_house, epis=20000)
    # mean_transfer_hardheaded_kitchen, std_transfer_hardheaded_kitchen = get_mean_std(transfer_hardheaded_kitchen, epis=20000)
    # mean_transfer_hardheaded_lunch, std_transfer_hardheaded_lunch = get_mean_std(transfer_hardheaded_lunch, epis=20000)
    # mean_transfer_hardheaded_outfit, std_transfer_hardheaded_outfit = get_mean_std(transfer_hardheaded_outfit, epis=20000)

    # mean_transfer_caduceus_defensive, std_transfer_caduceus_defensive = get_mean_std(transfer_caduceus_defensive, epis=20000)
    # mean_transfer_caduceus_fifty, std_transfer_caduceus_fifty = get_mean_std(transfer_caduceus_fifty, epis=20000)
    # mean_transfer_caduceus_house, std_transfer_caduceus_house = get_mean_std(transfer_caduceus_house, epis=20000)
    # mean_transfer_caduceus_kitchen, std_transfer_caduceus_kitchen = get_mean_std(transfer_caduceus_kitchen, epis=20000)
    # mean_transfer_caduceus_lunch, std_transfer_caduceus_lunch = get_mean_std(transfer_caduceus_lunch, epis=20000)
    # mean_transfer_caduceus_outfit, std_transfer_caduceus_outfit = get_mean_std(transfer_caduceus_outfit, epis=20000)

    # mean_transfer_parsagent_defensive, std_transfer_parsagent_defensive = get_mean_std(transfer_parsagent_defensive, epis=20000)
    # mean_transfer_parsagent_fifty, std_transfer_parsagent_fifty = get_mean_std(transfer_parsagent_fifty, epis=20000)
    # mean_transfer_parsagent_house, std_transfer_parsagent_house = get_mean_std(transfer_parsagent_house, epis=20000)
    # mean_transfer_parsagent_kitchen, std_transfer_parsagent_kitchen = get_mean_std(transfer_parsagent_kitchen, epis=20000)
    # mean_transfer_parsagent_lunch, std_transfer_parsagent_lunch = get_mean_std(transfer_parsagent_lunch, epis=20000)
    # mean_transfer_parsagent_outfit, std_transfer_parsagent_outfit = get_mean_std(transfer_parsagent_outfit, epis=20000)

    # mean_transfer_ponpoko_defensive, std_transfer_ponpoko_defensive = get_mean_std(transfer_ponpoko_defensive, epis=20000)
    # mean_transfer_ponpoko_fifty, std_transfer_ponpoko_fifty = get_mean_std(transfer_ponpoko_fifty, epis=20000)
    # mean_transfer_ponpoko_house, std_transfer_ponpoko_house = get_mean_std(transfer_ponpoko_house, epis=20000)
    # mean_transfer_ponpoko_kitchen, std_transfer_ponpoko_kitchen = get_mean_std(transfer_ponpoko_kitchen, epis=20000)
    # mean_transfer_ponpoko_lunch, std_transfer_ponpoko_lunch = get_mean_std(transfer_ponpoko_lunch, epis=20000)
    # mean_transfer_ponpoko_outfit, std_transfer_ponpoko_outfit = get_mean_std(transfer_ponpoko_outfit, epis=20000)

    # mean_transfer_yxagent_defensive, std_transfer_yxagent_defensive = get_mean_std(transfer_yxagent_defensive, epis=20000)
    # mean_transfer_yxagent_fifty, std_transfer_yxagent_fifty = get_mean_std(transfer_yxagent_fifty, epis=20000)
    # mean_transfer_yxagent_house, std_transfer_yxagent_house = get_mean_std(transfer_yxagent_house, epis=20000)
    # mean_transfer_yxagent_kitchen, std_transfer_yxagent_kitchen = get_mean_std(transfer_yxagent_kitchen, epis=20000)
    # mean_transfer_yxagent_lunch, std_transfer_yxagent_lunch = get_mean_std(transfer_yxagent_lunch, epis=20000)
    # mean_transfer_yxagent_outfit, std_transfer_yxagent_outfit = get_mean_std(transfer_yxagent_outfit, epis=20000)

    # mean_transfer_agreeable_acquisition, std_transfer_agreeable_acquisition = get_mean_std(transfer_agreeable_acquisition, epis=20000)
    # mean_transfer_agreeable_animal, std_transfer_agreeable_animal = get_mean_std(transfer_agreeable_animal, epis=20000)
    # mean_transfer_agreeable_defensive, std_transfer_agreeable_defensive = get_mean_std(transfer_agreeable_defensive, epis=20000)
    # mean_transfer_agreeable_dog, std_transfer_agreeable_dog = get_mean_std(transfer_agreeable_dog, epis=20000)
    # mean_transfer_agreeable_house, std_transfer_agreeable_house = get_mean_std(transfer_agreeable_house, epis=20000)
    # mean_transfer_agreeable_kitchen, std_transfer_agreeable_kitchen = get_mean_std(transfer_agreeable_kitchen, epis=20000)
    # mean_transfer_agreeable_outfit, std_transfer_agreeable_outfit = get_mean_std(transfer_agreeable_outfit, epis=20000)
    # mean_transfer_agreeable_smart, std_transfer_agreeable_smart = get_mean_std(transfer_agreeable_smart, epis=20000)
    # mean_transfer_agreeable_whole, std_transfer_agreeable_whole = get_mean_std(transfer_agreeable_whole, epis=20000)

    # mean_transfer_caduceusDC16_acquisition, std_transfer_caduceusDC16_acquisition = get_mean_std(transfer_caduceusDC16_acquisition, epis=20000)
    # mean_transfer_caduceusDC16_animal, std_transfer_caduceusDC16_animal = get_mean_std(transfer_caduceusDC16_animal, epis=20000)
    # mean_transfer_caduceusDC16_defensive, std_transfer_caduceusDC16_defensive = get_mean_std(transfer_caduceusDC16_defensive, epis=20000)
    # mean_transfer_caduceusDC16_dog, std_transfer_caduceusDC16_dog = get_mean_std(transfer_caduceusDC16_dog, epis=20000)
    # mean_transfer_caduceusDC16_house, std_transfer_caduceusDC16_house = get_mean_std(transfer_caduceusDC16_house, epis=20000)
    # mean_transfer_caduceusDC16_kitchen, std_transfer_caduceusDC16_kitchen = get_mean_std(transfer_caduceusDC16_kitchen, epis=20000)
    # mean_transfer_caduceusDC16_outfit, std_transfer_caduceusDC16_outfit = get_mean_std(transfer_caduceusDC16_outfit, epis=20000)
    # mean_transfer_caduceusDC16_smart, std_transfer_caduceusDC16_smart = get_mean_std(transfer_caduceusDC16_smart, epis=20000)
    # mean_transfer_caduceusDC16_whole, std_transfer_caduceusDC16_whole = get_mean_std(transfer_caduceusDC16_whole, epis=20000)

    # mean_transfer_thefawkes_acquisition, std_transfer_thefawkes_acquisition = get_mean_std(transfer_thefawkes_acquisition, epis=20000)
    # mean_transfer_thefawkes_animal, std_transfer_thefawkes_animal = get_mean_std(transfer_thefawkes_animal, epis=20000)
    # mean_transfer_thefawkes_defensive, std_transfer_thefawkes_defensive = get_mean_std(transfer_thefawkes_defensive, epis=20000)
    # mean_transfer_thefawkes_dog, std_transfer_thefawkes_dog = get_mean_std(transfer_thefawkes_dog, epis=20000)
    # mean_transfer_thefawkes_house, std_transfer_thefawkes_house = get_mean_std(transfer_thefawkes_house, epis=20000)
    # mean_transfer_thefawkes_kitchen, std_transfer_thefawkes_kitchen = get_mean_std(transfer_thefawkes_kitchen, epis=20000)
    # mean_transfer_thefawkes_outfit, std_transfer_thefawkes_outfit = get_mean_std(transfer_thefawkes_outfit, epis=20000)
    # mean_transfer_thefawkes_smart, std_transfer_thefawkes_smart = get_mean_std(transfer_thefawkes_smart, epis=20000)
    # mean_transfer_thefawkes_whole, std_transfer_thefawkes_whole = get_mean_std(transfer_thefawkes_whole, epis=20000)

    # mean_transfer_agent36_acquisition, std_transfer_agent36_acquisition = get_mean_std(transfer_agent36_acquisition, epis=20000)
    # mean_transfer_agent36_animal, std_transfer_agent36_animal = get_mean_std(transfer_agent36_animal, epis=20000)
    # mean_transfer_agent36_defensive, std_transfer_agent36_defensive = get_mean_std(transfer_agent36_defensive, epis=20000)
    # mean_transfer_agent36_dog, std_transfer_agent36_dog = get_mean_std(transfer_agent36_dog, epis=20000)
    # mean_transfer_agent36_house, std_transfer_agent36_house = get_mean_std(transfer_agent36_house, epis=20000)
    # mean_transfer_agent36_kitchen, std_transfer_agent36_kitchen = get_mean_std(transfer_agent36_kitchen, epis=20000)
    # mean_transfer_agent36_outfit, std_transfer_agent36_outfit = get_mean_std(transfer_agent36_outfit, epis=20000)
    # mean_transfer_agent36_smart, std_transfer_agent36_smart = get_mean_std(transfer_agent36_smart, epis=20000)
    # mean_transfer_agent36_whole, std_transfer_agent36_whole = get_mean_std(transfer_agent36_whole, epis=20000)

    # #mean_transfer, std_transfer = get_mean_std(path_transfer, epis=20000)
    # mean_agentlg_defensive, std_agentlg_defensive = get_mean_std(path_agentlg_defensive, epis=20000)
    # mean_agentlg_fifty, std_agentlg_fifty = get_mean_std(path_agentlg_fifty, epis=20000)
    # mean_agentlg_house, std_agentlg_house = get_mean_std(path_agentlg_house, epis=20000)
    # mean_agentlg_kitchen, std_agentlg_kitchen = get_mean_std(path_agentlg_kitchen, epis=20000)
    # mean_agentlg_lunch, std_agentlg_lunch = get_mean_std(path_agentlg_lunch, epis=20000)
    # mean_agentlg_outfit, std_agentlg_outfit = get_mean_std(path_agentlg_outfit, epis=20000)

    # mean_atlas3_defensive, std_atlas3_defensive = get_mean_std(path_atlas3_defensive, epis=20000)
    # mean_atlas3_fifty, std_atlas3_fifty = get_mean_std(path_atlas3_fifty, epis=20000)
    # mean_atlas3_house, std_atlas3_house = get_mean_std(path_atlas3_house, epis=20000)
    # mean_atlas3_kitchen, std_atlas3_kitchen = get_mean_std(path_atlas3_kitchen, epis=20000)
    # mean_atlas3_lunch, std_atlas3_lunch = get_mean_std(path_atlas3_lunch, epis=20000)
    # mean_atlas3_outfit, std_atlas3_outfit = get_mean_std(path_atlas3_outfit, epis=20000)

    # mean_cuhkagent_defensive, std_cuhkagent_defensive = get_mean_std(path_cuhkagent_defensive, epis=20000)
    # mean_cuhkagent_fifty, std_cuhkagent_fifty = get_mean_std(path_cuhkagent_fifty, epis=20000)
    # mean_cuhkagent_house, std_cuhkagent_house = get_mean_std(path_cuhkagent_house, epis=20000)
    # mean_cuhkagent_kitchen, std_cuhkagent_kitchen = get_mean_std(path_cuhkagent_kitchen, epis=20000)
    # mean_cuhkagent_lunch, std_cuhkagent_lunch = get_mean_std(path_cuhkagent_lunch, epis=20000)
    # mean_cuhkagent_outfit, std_cuhkagent_outfit = get_mean_std(path_cuhkagent_outfit, epis=20000)

    # mean_hardheaded_defensive, std_hardheaded_defensive = get_mean_std(path_hardheaded_defensive, epis=20000)
    # mean_hardheaded_fifty, std_hardheaded_fifty = get_mean_std(path_hardheaded_fifty, epis=20000)
    # mean_hardheaded_house, std_hardheaded_house = get_mean_std(path_hardheaded_house, epis=20000)
    # mean_hardheaded_kitchen, std_hardheaded_kitchen = get_mean_std(path_hardheaded_kitchen, epis=20000)
    # mean_hardheaded_lunch, std_hardheaded_lunch = get_mean_std(path_hardheaded_lunch, epis=20000)
    # mean_hardheaded_outfit, std_hardheaded_outfit = get_mean_std(path_hardheaded_outfit, epis=20000)

    # mean_omac_defensive, std_omac_defensive = get_mean_std(path_omac_defensive, epis=20000)
    # mean_omac_fifty, std_omac_fifty = get_mean_std(path_omac_fifty, epis=20000)
    # mean_omac_house, std_omac_house = get_mean_std(path_omac_house, epis=20000)
    # mean_omac_kitchen, std_omac_kitchen = get_mean_std(path_omac_kitchen, epis=20000)
    # mean_omac_lunch, std_omac_lunch = get_mean_std(path_omac_lunch, epis=20000)
    # mean_omac_outfit, std_omac_outfit = get_mean_std(path_omac_outfit, epis=20000)

    # mean_parscat_defensive, std_parscat_defensive = get_mean_std(path_parscat_defensive, epis=20000)
    # mean_parscat_fifty, std_parscat_fifty = get_mean_std(path_parscat_fifty, epis=20000)
    # mean_parscat_house, std_parscat_house = get_mean_std(path_parscat_house, epis=20000)
    # mean_parscat_kitchen, std_parscat_kitchen = get_mean_std(path_parscat_kitchen, epis=20000)
    # mean_parscat_lunch, std_parscat_lunch = get_mean_std(path_parscat_lunch, epis=20000)
    # mean_parscat_outfit, std_parscat_outfit = get_mean_std(path_parscat_outfit, epis=20000)

    # mean_caduceus_defensive, std_caduceus_defensive = get_mean_std(path_caduceus_defensive, epis=20000)
    # mean_caduceus_fifty, std_caduceus_fifty = get_mean_std(path_caduceus_fifty, epis=20000)
    # mean_caduceus_house, std_caduceus_house = get_mean_std(path_caduceus_house, epis=20000)
    # mean_caduceus_kitchen, std_caduceus_kitchen = get_mean_std(path_caduceus_kitchen, epis=20000)
    # mean_caduceus_lunch, std_caduceus_lunch = get_mean_std(path_caduceus_lunch, epis=20000)
    # mean_caduceus_outfit, std_caduceus_outfit = get_mean_std(path_caduceus_outfit, epis=20000)

    # mean_parsagent_defensive, std_parsagent_defensive = get_mean_std(path_parsagent_defensive, epis=20000)
    # mean_parsagent_fifty, std_parsagent_fifty = get_mean_std(path_parsagent_fifty, epis=20000)
    # mean_parsagent_house, std_parsagent_house = get_mean_std(path_parsagent_house, epis=20000)
    # mean_parsagent_kitchen, std_parsagent_kitchen = get_mean_std(path_parsagent_kitchen, epis=20000)
    # mean_parsagent_lunch, std_parsagent_lunch = get_mean_std(path_parsagent_lunch, epis=20000)
    # mean_parsagent_outfit, std_parsagent_outfit = get_mean_std(path_parsagent_outfit, epis=20000)

    # mean_ponpoko_defensive, std_ponpoko_defensive = get_mean_std(path_ponpoko_defensive, epis=20000)
    # mean_ponpoko_fifty, std_ponpoko_fifty = get_mean_std(path_ponpoko_fifty, epis=20000)
    # mean_ponpoko_house, std_ponpoko_house = get_mean_std(path_ponpoko_house, epis=20000)
    # mean_ponpoko_kitchen, std_ponpoko_kitchen = get_mean_std(path_ponpoko_kitchen, epis=20000)
    # mean_ponpoko_lunch, std_ponpoko_lunch = get_mean_std(path_ponpoko_lunch, epis=20000)
    # mean_ponpoko_outfit, std_ponpoko_outfit = get_mean_std(path_ponpoko_outfit, epis=20000)

    # mean_yxagent_defensive, std_yxagent_defensive = get_mean_std(path_yxagent_defensive, epis=20000)
    # mean_yxagent_fifty, std_yxagent_fifty = get_mean_std(path_yxagent_fifty, epis=20000)
    # mean_yxagent_house, std_yxagent_house = get_mean_std(path_yxagent_house, epis=20000)
    # mean_yxagent_kitchen, std_yxagent_kitchen = get_mean_std(path_yxagent_kitchen, epis=20000)
    # mean_yxagent_lunch, std_yxagent_lunch = get_mean_std(path_yxagent_lunch, epis=20000)
    # mean_yxagent_outfit, std_yxagent_outfit = get_mean_std(path_yxagent_outfit, epis=20000)

    # mean_agreeable_acquisition, std_agreeable_acquisition = get_mean_std(path_agreeable_acquisition, epis=20000)
    # mean_agreeable_animal, std_agreeable_animal = get_mean_std(path_agreeable_animal, epis=20000)
    # mean_agreeable_defensive, std_agreeable_defensive = get_mean_std(path_agreeable_defensive, epis=20000)
    # mean_agreeable_dog, std_agreeable_dog = get_mean_std(path_agreeable_dog, epis=20000)
    # mean_agreeable_house, std_agreeable_house = get_mean_std(path_agreeable_house, epis=20000)
    # mean_agreeable_kitchen, std_agreeable_kitchen = get_mean_std(path_agreeable_kitchen, epis=20000)
    # mean_agreeable_outfit, std_agreeable_outfit = get_mean_std(path_agreeable_outfit, epis=20000)
    # mean_agreeable_smart, std_agreeable_smart = get_mean_std(path_agreeable_smart, epis=20000)
    # mean_agreeable_whole, std_agreeable_whole = get_mean_std(path_agreeable_whole, epis=20000)

    # mean_caduceusDC16_acquisition, std_caduceusDC16_acquisition = get_mean_std(path_caduceusDC16_acquisition, epis=20000)
    # mean_caduceusDC16_animal, std_caduceusDC16_animal = get_mean_std(path_caduceusDC16_animal, epis=20000)
    # mean_caduceusDC16_defensive, std_caduceusDC16_defensive = get_mean_std(path_caduceusDC16_defensive, epis=20000)
    # mean_caduceusDC16_dog, std_caduceusDC16_dog = get_mean_std(path_caduceusDC16_dog, epis=20000)
    # mean_caduceusDC16_house, std_caduceusDC16_house = get_mean_std(path_caduceusDC16_house, epis=20000)
    # mean_caduceusDC16_kitchen, std_caduceusDC16_kitchen = get_mean_std(path_caduceusDC16_kitchen, epis=20000)
    # mean_caduceusDC16_outfit, std_caduceusDC16_outfit = get_mean_std(path_caduceusDC16_outfit, epis=20000)
    # mean_caduceusDC16_smart, std_caduceusDC16_smart = get_mean_std(path_caduceusDC16_smart, epis=20000)
    # mean_caduceusDC16_whole, std_caduceusDC16_whole = get_mean_std(path_caduceusDC16_whole, epis=20000)

    # mean_thefawkes_acquisition, std_thefawkes_acquisition = get_mean_std(path_thefawkes_acquisition, epis=20000)
    # mean_thefawkes_animal, std_thefawkes_animal = get_mean_std(path_thefawkes_animal, epis=20000)
    # mean_thefawkes_defensive, std_thefawkes_defensive = get_mean_std(path_thefawkes_defensive, epis=20000)
    # mean_thefawkes_dog, std_thefawkes_dog = get_mean_std(path_thefawkes_dog, epis=20000)
    # mean_thefawkes_house, std_thefawkes_house = get_mean_std(path_thefawkes_house, epis=20000)
    # mean_thefawkes_kitchen, std_thefawkes_kitchen = get_mean_std(path_thefawkes_kitchen, epis=20000)
    # mean_thefawkes_outfit, std_thefawkes_outfit = get_mean_std(path_thefawkes_outfit, epis=20000)
    # mean_thefawkes_smart, std_thefawkes_smart = get_mean_std(path_thefawkes_smart, epis=20000)
    # mean_thefawkes_whole, std_thefawkes_whole = get_mean_std(path_thefawkes_whole, epis=20000)

    # mean_agent36_acquisition, std_agent36_acquisition = get_mean_std(path_agent36_acquisition, epis=20000)
    # mean_agent36_animal, std_agent36_animal = get_mean_std(path_agent36_animal, epis=20000)
    # mean_agent36_defensive, std_agent36_defensive = get_mean_std(path_agent36_defensive, epis=20000)
    # mean_agent36_dog, std_agent36_dog = get_mean_std(path_agent36_dog, epis=20000)
    # mean_agent36_house, std_agent36_house = get_mean_std(path_agent36_house, epis=20000)
    # mean_agent36_kitchen, std_agent36_kitchen = get_mean_std(path_agent36_kitchen, epis=20000)
    # mean_agent36_outfit, std_agent36_outfit = get_mean_std(path_agent36_outfit, epis=20000)
    # mean_agent36_smart, std_agent36_smart = get_mean_std(path_agent36_smart, epis=20000)
    # mean_agent36_whole, std_agent36_whole = get_mean_std(path_agent36_whole, epis=20000)

    #transfer_agentlg
    # # DefensiveCharms
    # plot_line_transfer(mean_agentlg_defensive, std_agentlg_defensive, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agentlg_defensive, std_transfer_agentlg_defensive, '#9ac9db', "transfer", marker='d')
    # # FiftyFifty2013
    # plot_line_transfer(mean_agentlg_fifty, std_agentlg_fifty, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agentlg_fifty, std_transfer_agentlg_fifty, '#9ac9db', "transfer", marker='d')
    # # HouseKeeping
    # plot_line_transfer(mean_agentlg_house, std_agentlg_house, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agentlg_house, std_transfer_agentlg_house, '#9ac9db', "transfer", marker='d')
    # # Kitchen
    # plot_line_transfer(mean_agentlg_kitchen, std_agentlg_kitchen, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agentlg_kitchen, std_transfer_agentlg_kitchen, '#9ac9db', "transfer", marker='d')
    # # Lunch
    # plot_line_transfer(mean_agentlg_lunch, std_agentlg_lunch, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agentlg_lunch, std_transfer_agentlg_lunch, '#9ac9db', "transfer", marker='d')
    # # Outfit
    # plot_line_transfer(mean_agentlg_outfit, std_agentlg_outfit, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agentlg_outfit, std_transfer_agentlg_outfit, '#9ac9db', "transfer", marker='d')

    #transfer_hardheaded
    # # DefensiveCharms
    # plot_line_transfer(mean_hardheaded_defensive, std_hardheaded_defensive, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_hardheaded_defensive, std_transfer_hardheaded_defensive, '#9ac9db', "transfer", marker='d')
    # # FiftyFifty2013
    # plot_line_transfer(mean_hardheaded_fifty, std_hardheaded_fifty, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_hardheaded_fifty, std_transfer_hardheaded_fifty, '#9ac9db', "transfer", marker='d')
    # # HouseKeeping
    # plot_line_transfer(mean_hardheaded_house, std_hardheaded_house, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_hardheaded_house, std_transfer_hardheaded_house, '#9ac9db', "transfer", marker='d')
    # # Kitchen
    # plot_line_transfer(mean_hardheaded_kitchen, std_hardheaded_kitchen, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_hardheaded_kitchen, std_transfer_hardheaded_kitchen, '#9ac9db', "transfer", marker='d')
    # # Lunch
    # plot_line_transfer(mean_hardheaded_lunch, std_hardheaded_lunch, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_hardheaded_lunch, std_transfer_hardheaded_lunch, '#9ac9db', "transfer", marker='d')
    # # Outfit
    # plot_line_transfer(mean_hardheaded_outfit, std_hardheaded_outfit, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_hardheaded_outfit, std_transfer_hardheaded_outfit, '#9ac9db', "transfer", marker='d')

    #transfer_caduceus
    # # DefensiveCharms
    # plot_line_transfer(mean_caduceus_defensive, std_caduceus_defensive, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceus_defensive, std_transfer_caduceus_defensive, '#9ac9db', "transfer", marker='d')
    # # FiftyFifty2013
    # plot_line_transfer(mean_caduceus_fifty, std_caduceus_fifty, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceus_fifty, std_transfer_caduceus_fifty, '#9ac9db', "transfer", marker='d')
    # # HouseKeeping
    # plot_line_transfer(mean_caduceus_house, std_caduceus_house, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceus_house, std_transfer_caduceus_house, '#9ac9db', "transfer", marker='d')
    # # Kitchen
    # plot_line_transfer(mean_caduceus_kitchen, std_caduceus_kitchen, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceus_kitchen, std_transfer_caduceus_kitchen, '#9ac9db', "transfer", marker='d')
    # # Lunch
    # plot_line_transfer(mean_caduceus_lunch, std_caduceus_lunch, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceus_lunch, std_transfer_caduceus_lunch, '#9ac9db', "transfer", marker='d')
    # # Outfit
    # plot_line_transfer(mean_caduceus_outfit, std_caduceus_outfit, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceus_outfit, std_transfer_caduceus_outfit, '#9ac9db', "transfer", marker='d')

    #transfer_parsagent
    # # DefensiveCharms
    # plot_line_transfer(mean_parsagent_defensive, std_parsagent_defensive, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_parsagent_defensive, std_transfer_parsagent_defensive, '#9ac9db', "transfer", marker='d')
    # # FiftyFifty2013
    # plot_line_transfer(mean_parsagent_fifty, std_parsagent_fifty, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_parsagent_fifty, std_transfer_parsagent_fifty, '#9ac9db', "transfer", marker='d')
    # # HouseKeeping
    # plot_line_transfer(mean_parsagent_house, std_parsagent_house, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_parsagent_house, std_transfer_parsagent_house, '#9ac9db', "transfer", marker='d')
    # # Kitchen
    # plot_line_transfer(mean_parsagent_kitchen, std_parsagent_kitchen, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_parsagent_kitchen, std_transfer_parsagent_kitchen, '#9ac9db', "transfer", marker='d')
    # # Lunch
    # plot_line_transfer(mean_parsagent_lunch, std_parsagent_lunch, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_parsagent_lunch, std_transfer_parsagent_lunch, '#9ac9db', "transfer", marker='d')
    # # Outfit
    # plot_line_transfer(mean_parsagent_outfit, std_parsagent_outfit, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_parsagent_outfit, std_transfer_parsagent_outfit, '#9ac9db', "transfer", marker='d')

    #transfer_ponpoko
    # # DefensiveCharms
    # plot_line_transfer(mean_ponpoko_defensive, std_ponpoko_defensive, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_ponpoko_defensive, std_transfer_ponpoko_defensive, '#9ac9db', "transfer", marker='d')
    # # FiftyFifty2013
    # plot_line_transfer(mean_ponpoko_fifty, std_ponpoko_fifty, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_ponpoko_fifty, std_transfer_ponpoko_fifty, '#9ac9db', "transfer", marker='d')
    # # HouseKeeping
    # plot_line_transfer(mean_ponpoko_house, std_ponpoko_house, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_ponpoko_house, std_transfer_ponpoko_house, '#9ac9db', "transfer", marker='d')
    # # Kitchen
    # plot_line_transfer(mean_ponpoko_kitchen, std_ponpoko_kitchen, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_ponpoko_kitchen, std_transfer_ponpoko_kitchen, '#9ac9db', "transfer", marker='d')
    # # Lunch
    # plot_line_transfer(mean_ponpoko_lunch, std_ponpoko_lunch, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_ponpoko_lunch, std_transfer_ponpoko_lunch, '#9ac9db', "transfer", marker='d')
    # # Outfit
    # plot_line_transfer(mean_ponpoko_outfit, std_ponpoko_outfit, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_ponpoko_outfit, std_transfer_ponpoko_outfit, '#9ac9db', "transfer", marker='d')

    #transfer_yxagent
    # # DefensiveCharms
    # plot_line_transfer(mean_yxagent_defensive, std_yxagent_defensive, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_yxagent_defensive, std_transfer_yxagent_defensive, '#9ac9db', "transfer", marker='d')
    # # FiftyFifty2013
    # plot_line_transfer(mean_yxagent_fifty, std_yxagent_fifty, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_yxagent_fifty, std_transfer_yxagent_fifty, '#9ac9db', "transfer", marker='d')
    # # HouseKeeping
    # plot_line_transfer(mean_yxagent_house, std_yxagent_house, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_yxagent_house, std_transfer_yxagent_house, '#9ac9db', "transfer", marker='d')
    # # Kitchen
    # plot_line_transfer(mean_yxagent_kitchen, std_yxagent_kitchen, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_yxagent_kitchen, std_transfer_yxagent_kitchen, '#9ac9db', "transfer", marker='d')
    # # Lunch
    # plot_line_transfer(mean_yxagent_lunch, std_yxagent_lunch, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_yxagent_lunch, std_transfer_yxagent_lunch, '#9ac9db', "transfer", marker='d')
    # # Outfit
    # plot_line_transfer(mean_yxagent_outfit, std_yxagent_outfit, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_yxagent_outfit, std_transfer_yxagent_outfit, '#9ac9db', "transfer", marker='d')

    #transfer_agreeable
    # Acquisition
    # plot_line_transfer(mean_agreeable_acquisition, std_agreeable_acquisition, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agreeable_acquisition, std_transfer_agreeable_acquisition, '#9ac9db', "transfer", marker='d')
    # Animal
    # plot_line_transfer(mean_agreeable_animal, std_agreeable_animal, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agreeable_animal, std_transfer_agreeable_animal, '#9ac9db', "transfer", marker='d')
    # DefensiveCharms
    # plot_line_transfer(mean_agreeable_defensive, std_agreeable_defensive, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agreeable_defensive, std_transfer_agreeable_defensive, '#9ac9db', "transfer", marker='d')
    # DogChoosing
    # plot_line_transfer(mean_agreeable_dog, std_agreeable_dog, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agreeable_dog, std_transfer_agreeable_dog, '#9ac9db', "transfer", marker='d')
    # HouseKeepig
    # plot_line_transfer(mean_agreeable_house, std_agreeable_house, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agreeable_house, std_transfer_agreeable_house, '#9ac9db', "transfer", marker='d')
    # Kitchen
    # plot_line_transfer(mean_agreeable_kitchen, std_agreeable_kitchen, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agreeable_kitchen, std_transfer_agreeable_kitchen, '#9ac9db', "transfer", marker='d')
    # Outfit
    # plot_line_transfer(mean_agreeable_outfit, std_agreeable_outfit, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agreeable_outfit, std_transfer_agreeable_outfit, '#9ac9db', "transfer", marker='d')
    # Smartphone
    # plot_line_transfer(mean_agreeable_smart, std_agreeable_smart, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agreeable_smart, std_transfer_agreeable_smart, '#9ac9db', "transfer", marker='d')
    # Wholesaler
    # plot_line_transfer(mean_agreeable_whole, std_agreeable_whole, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agreeable_whole, std_transfer_agreeable_whole, '#9ac9db', "transfer", marker='d')

    #transfer_thefawkes
    # Acquisition
    # plot_line_transfer(mean_thefawkes_acquisition, std_thefawkes_acquisition, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_thefawkes_acquisition, std_transfer_thefawkes_acquisition, '#9ac9db', "transfer", marker='d')
    # Animal
    # plot_line_transfer(mean_thefawkes_animal, std_thefawkes_animal, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_thefawkes_animal, std_transfer_thefawkes_animal, '#9ac9db', "transfer", marker='d')
    # DefensiveCharms
    # plot_line_transfer(mean_thefawkes_defensive, std_thefawkes_defensive, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_thefawkes_defensive, std_transfer_thefawkes_defensive, '#9ac9db', "transfer", marker='d')
    # DogChoosing
    # plot_line_transfer(mean_thefawkes_dog, std_thefawkes_dog, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_thefawkes_dog, std_transfer_thefawkes_dog, '#9ac9db', "transfer", marker='d')
    # HouseKeepig
    # plot_line_transfer(mean_thefawkes_house, std_thefawkes_house, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_thefawkes_house, std_transfer_thefawkes_house, '#9ac9db', "transfer", marker='d')
    # Kitchen
    # plot_line_transfer(mean_thefawkes_kitchen, std_thefawkes_kitchen, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_thefawkes_kitchen, std_transfer_thefawkes_kitchen, '#9ac9db', "transfer", marker='d')
    # Outfit
    # plot_line_transfer(mean_thefawkes_outfit, std_thefawkes_outfit, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_thefawkes_outfit, std_transfer_thefawkes_outfit, '#9ac9db', "transfer", marker='d')
    # Smartphone
    # plot_line_transfer(mean_thefawkes_smart, std_thefawkes_smart, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_thefawkes_smart, std_transfer_thefawkes_smart, '#9ac9db', "transfer", marker='d')
    # Wholesaler
    # plot_line_transfer(mean_thefawkes_whole, std_thefawkes_whole, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_thefawkes_whole, std_transfer_thefawkes_whole, '#9ac9db', "transfer", marker='d')

    #transfer_caduceusDC16
    # Acquisition
    # plot_line_transfer(mean_caduceusDC16_acquisition, std_caduceusDC16_acquisition, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceusDC16_acquisition, std_transfer_caduceusDC16_acquisition, '#9ac9db', "transfer", marker='d')
    # Animal
    # plot_line_transfer(mean_caduceusDC16_animal, std_caduceusDC16_animal, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceusDC16_animal, std_transfer_caduceusDC16_animal, '#9ac9db', "transfer", marker='d')
    # DefensiveCharms
    # plot_line_transfer(mean_caduceusDC16_defensive, std_caduceusDC16_defensive, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceusDC16_defensive, std_transfer_caduceusDC16_defensive, '#9ac9db', "transfer", marker='d')
    # DogChoosing
    # plot_line_transfer(mean_caduceusDC16_dog, std_caduceusDC16_dog, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceusDC16_dog, std_transfer_caduceusDC16_dog, '#9ac9db', "transfer", marker='d')
    # HouseKeepig
    # plot_line_transfer(mean_caduceusDC16_house, std_caduceusDC16_house, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceusDC16_house, std_transfer_caduceusDC16_house, '#9ac9db', "transfer", marker='d')
    # Kitchen
    # plot_line_transfer(mean_caduceusDC16_kitchen, std_caduceusDC16_kitchen, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceusDC16_kitchen, std_transfer_caduceusDC16_kitchen, '#9ac9db', "transfer", marker='d')
    # Outfit
    # plot_line_transfer(mean_caduceusDC16_outfit, std_caduceusDC16_outfit, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceusDC16_outfit, std_transfer_caduceusDC16_outfit, '#9ac9db', "transfer", marker='d')
    # Smartphone
    # plot_line_transfer(mean_caduceusDC16_smart, std_caduceusDC16_smart, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceusDC16_smart, std_transfer_caduceusDC16_smart, '#9ac9db', "transfer", marker='d')
    # Wholesaler
    # plot_line_transfer(mean_caduceusDC16_whole, std_caduceusDC16_whole, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_caduceusDC16_whole, std_transfer_caduceusDC16_whole, '#9ac9db', "transfer", marker='d')

    #transfer_agent36
    # Acquisition
    # plot_line_transfer(mean_agent36_acquisition, std_agent36_acquisition, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agent36_acquisition, std_transfer_agent36_acquisition, '#9ac9db', "transfer", marker='d')
    # Animal
    # plot_line_transfer(mean_agent36_animal, std_agent36_animal, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agent36_animal, std_transfer_agent36_animal, '#9ac9db', "transfer", marker='d')
    # DefensiveCharms
    # plot_line_transfer(mean_agent36_defensive, std_agent36_defensive, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agent36_defensive, std_transfer_agent36_defensive, '#9ac9db', "transfer", marker='d')
    # DogChoosing
    # plot_line_transfer(mean_agent36_dog, std_agent36_dog, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agent36_dog, std_transfer_agent36_dog, '#9ac9db', "transfer", marker='d')
    # HouseKeepig
    # plot_line_transfer(mean_agent36_house, std_agent36_house, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agent36_house, std_transfer_agent36_house, '#9ac9db', "transfer", marker='d')
    # Kitchen
    # plot_line_transfer(mean_agent36_kitchen, std_agent36_kitchen, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agent36_kitchen, std_transfer_agent36_kitchen, '#9ac9db', "transfer", marker='d')
    # Outfit
    # plot_line_transfer(mean_agent36_outfit, std_agent36_outfit, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agent36_outfit, std_transfer_agent36_outfit, '#9ac9db', "transfer", marker='d')
    # Smartphone
    # plot_line_transfer(mean_agent36_smart, std_agent36_smart, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agent36_smart, std_transfer_agent36_smart, '#9ac9db', "transfer", marker='d')
    # Wholesaler
    # plot_line_transfer(mean_agent36_whole, std_agent36_whole, '#2878b5', "baseline", marker='d')
    # plot_line_transfer(mean_transfer_agent36_whole, std_transfer_agent36_whole, '#9ac9db', "transfer", marker='d')

    #plot_line_transfer(mean_transfer, std_transfer, 'green', "Transfer", marker='d')

    # Acquisition
    # plot_line_transfer(mean_agreeable_acquisition, std_agreeable_acquisition, '#845EC2', "agreeable", marker='d')
    # plot_line_transfer(mean_caduceusDC16_acquisition, std_caduceusDC16_acquisition, '#B39CD0', "caduceusDC16", marker='d')
    # plot_line_transfer(mean_thefawkes_acquisition, std_thefawkes_acquisition, '#4D8076', "thefawkes", marker='d')
    # plot_line_transfer(mean_agent36_acquisition, std_agent36_acquisition, '#00C9A7', "agent36", marker='d')

    # Animal
    # plot_line_transfer(mean_agreeable_animal, std_agreeable_animal, '#845EC2', "agreeable", marker='d')
    # plot_line_transfer(mean_caduceusDC16_animal, std_caduceusDC16_animal, '#B39CD0', "caduceusDC16", marker='d')
    # plot_line_transfer(mean_thefawkes_animal, std_thefawkes_animal, '#4D8076', "thefawkes", marker='d')
    # plot_line_transfer(mean_agent36_animal, std_agent36_animal, '#00C9A7', "agent36", marker='d')

    # DefensiveCharms
    # plot_line_transfer(mean_agentlg_defensive, std_agentlg_defensive, '#2878b5', "agentlg", marker='d')
    # plot_line_transfer(mean_atlas3_defensive, std_atlas3_defensive, '#9ac9db', "atlas3", marker='d')
    # plot_line_transfer(mean_cuhkagent_defensive, std_cuhkagent_defensive, '#f8ac8c', "cuhkagent", marker='d')
    # plot_line_transfer(mean_hardheaded_defensive, std_hardheaded_defensive, '#c82423', "hardheaded", marker='d')
    # plot_line_transfer(mean_omac_defensive, std_omac_defensive, '#BEB8DC', "omac", marker='d')
    # plot_line_transfer(mean_parscat_defensive, std_parscat_defensive, '#ff8884', "parscat", marker='d')
    # plot_line_transfer(mean_caduceus_defensive, std_caduceus_defensive, '#845EC2', "caduceus", marker='d')
    # plot_line_transfer(mean_parsagent_defensive, std_parsagent_defensive, '#B39CD0', "parsagent", marker='d')
    # plot_line_transfer(mean_ponpoko_defensive, std_ponpoko_defensive, '#4D8076', "ponpoko", marker='d')
    # plot_line_transfer(mean_yxagent_defensive, std_yxagent_defensive, '#00C9A7', "yxagent", marker='d')

    # plot_line_transfer(mean_agreeable_defensive, std_agreeable_defensive, '#845EC2', "agreeable", marker='d')
    # plot_line_transfer(mean_caduceusDC16_defensive, std_caduceusDC16_defensive, '#B39CD0', "caduceusDC16", marker='d')
    # plot_line_transfer(mean_thefawkes_defensive, std_thefawkes_defensive, '#4D8076', "thefawkes", marker='d')
    # plot_line_transfer(mean_agent36_defensive, std_agent36_defensive, '#00C9A7', "agent36", marker='d')

    # DogChoosing
    # plot_line_transfer(mean_agreeable_dog, std_agreeable_dog, '#845EC2', "agreeable", marker='d')
    # plot_line_transfer(mean_caduceusDC16_dog, std_caduceusDC16_dog, '#B39CD0', "caduceusDC16", marker='d')
    # plot_line_transfer(mean_thefawkes_dog, std_thefawkes_dog, '#4D8076', "thefawkes", marker='d')
    # plot_line_transfer(mean_agent36_dog, std_agent36_dog, '#00C9A7', "agent36", marker='d')
    

    # FiftyFifty2013
    # plot_line_transfer(mean_agentlg_fifty, std_agentlg_fifty, '#2878b5', "agentlg", marker='d')
    # plot_line_transfer(mean_atlas3_fifty, std_atlas3_fifty, '#9ac9db', "atlas3", marker='d')
    # plot_line_transfer(mean_cuhkagent_fifty, std_cuhkagent_fifty, '#f8ac8c', "cuhkagent", marker='d')
    # plot_line_transfer(mean_hardheaded_fifty, std_hardheaded_fifty, '#c82423', "hardheaded", marker='d')
    # plot_line_transfer(mean_omac_fifty, std_omac_fifty, '#BEB8DC', "omac", marker='d')
    # plot_line_transfer(mean_parscat_fifty, std_parscat_fifty, '#ff8884', "parscat", marker='d')
    # plot_line_transfer(mean_caduceus_fifty, std_caduceus_fifty, '#845EC2', "caduceus", marker='d')
    # plot_line_transfer(mean_parsagent_fifty, std_parsagent_fifty, '#B39CD0', "parsagent", marker='d')
    # plot_line_transfer(mean_ponpoko_fifty, std_ponpoko_fifty, '#4D8076', "ponpoko", marker='d')
    # plot_line_transfer(mean_yxagent_fifty, std_yxagent_fifty, '#00C9A7', "yxagent", marker='d')
    
    # HouseKeeping
    # plot_line_transfer(mean_agentlg_house, std_agentlg_house, '#2878b5', "agentlg", marker='d')
    # plot_line_transfer(mean_atlas3_house, std_atlas3_house, '#9ac9db', "atlas3", marker='d')
    # plot_line_transfer(mean_cuhkagent_house, std_cuhkagent_house, '#f8ac8c', "cuhkagent", marker='d')
    # plot_line_transfer(mean_hardheaded_house, std_hardheaded_house, '#c82423', "hardheaded", marker='d')
    # plot_line_transfer(mean_omac_house, std_omac_house, '#BEB8DC', "omac", marker='d')
    # plot_line_transfer(mean_parscat_house, std_parscat_house, '#ff8884', "parscat", marker='d')
    # plot_line_transfer(mean_caduceus_house, std_caduceus_house, '#845EC2', "caduceus", marker='d')
    # plot_line_transfer(mean_parsagent_house, std_parsagent_house, '#B39CD0', "parsagent", marker='d')
    # plot_line_transfer(mean_ponpoko_house, std_ponpoko_house, '#4D8076', "ponpoko", marker='d')
    # plot_line_transfer(mean_yxagent_house, std_yxagent_house, '#00C9A7', "yxagent", marker='d')

    # plot_line_transfer(mean_agreeable_house, std_agreeable_house, '#845EC2', "agreeable", marker='d')
    # plot_line_transfer(mean_caduceusDC16_house, std_caduceusDC16_house, '#B39CD0', "caduceusDC16", marker='d')
    # plot_line_transfer(mean_thefawkes_house, std_thefawkes_house, '#4D8076', "thefawkes", marker='d')
    # plot_line_transfer(mean_agent36_house, std_agent36_house, '#00C9A7', "agent36", marker='d')

    # Kitchen
    # plot_line_transfer(mean_agentlg_kitchen, std_agentlg_kitchen, '#2878b5', "agentlg", marker='d')
    # plot_line_transfer(mean_atlas3_kitchen, std_atlas3_kitchen, '#9ac9db', "atlas3", marker='d')
    # plot_line_transfer(mean_cuhkagent_kitchen, std_cuhkagent_kitchen, '#f8ac8c', "cuhkagent", marker='d')
    # plot_line_transfer(mean_hardheaded_kitchen, std_hardheaded_kitchen, '#c82423', "hardheaded", marker='d')
    # plot_line_transfer(mean_omac_kitchen, std_omac_kitchen, '#BEB8DC', "omac", marker='d')
    # plot_line_transfer(mean_parscat_kitchen, std_parscat_kitchen, '#ff8884', "parscat", marker='d')
    # plot_line_transfer(mean_caduceus_kitchen, std_caduceus_kitchen, '#845EC2', "caduceus", marker='d')
    # plot_line_transfer(mean_parsagent_kitchen, std_parsagent_kitchen, '#B39CD0', "parsagent", marker='d')
    # plot_line_transfer(mean_ponpoko_kitchen, std_ponpoko_kitchen, '#4D8076', "ponpoko", marker='d')
    # plot_line_transfer(mean_yxagent_kitchen, std_yxagent_kitchen, '#00C9A7', "yxagent", marker='d')

    # plot_line_transfer(mean_agreeable_kitchen, std_agreeable_kitchen, '#845EC2', "agreeable", marker='d')
    # plot_line_transfer(mean_caduceusDC16_kitchen, std_caduceusDC16_kitchen, '#B39CD0', "caduceusDC16", marker='d')
    # plot_line_transfer(mean_thefawkes_kitchen, std_thefawkes_kitchen, '#4D8076', "thefawkes", marker='d')
    # plot_line_transfer(mean_agent36_kitchen, std_agent36_kitchen, '#00C9A7', "agent36", marker='d')

    # Lunch
    # plot_line_transfer(mean_agentlg_lunch, std_agentlg_lunch, '#2878b5', "agentlg", marker='d')
    # plot_line_transfer(mean_atlas3_lunch, std_atlas3_lunch, '#9ac9db', "atlas3", marker='d')
    # plot_line_transfer(mean_cuhkagent_lunch, std_cuhkagent_lunch, '#f8ac8c', "cuhkagent", marker='d')
    # plot_line_transfer(mean_hardheaded_lunch, std_hardheaded_lunch, '#c82423', "hardheaded", marker='d')
    # plot_line_transfer(mean_omac_lunch, std_omac_lunch, '#BEB8DC', "omac", marker='d')
    # plot_line_transfer(mean_parscat_lunch, std_parscat_lunch, '#ff8884', "parscat", marker='d')
    # plot_line_transfer(mean_caduceus_lunch, std_caduceus_lunch, '#845EC2', "caduceus", marker='d')
    # plot_line_transfer(mean_parsagent_lunch, std_parsagent_lunch, '#B39CD0', "parsagent", marker='d')
    # plot_line_transfer(mean_ponpoko_lunch, std_ponpoko_lunch, '#4D8076', "ponpoko", marker='d')
    # plot_line_transfer(mean_yxagent_lunch, std_yxagent_lunch, '#00C9A7', "yxagent", marker='d')

    # Outfit
    # plot_line_transfer(mean_agentlg_outfit, std_agentlg_outfit, '#2878b5', "agentlg", marker='d')
    # plot_line_transfer(mean_atlas3_outfit, std_atlas3_outfit, '#9ac9db', "atlas3", marker='d')
    # plot_line_transfer(mean_cuhkagent_outfit, std_cuhkagent_outfit, '#f8ac8c', "cuhkagent", marker='d')
    # plot_line_transfer(mean_hardheaded_outfit, std_hardheaded_outfit, '#c82423', "hardheaded", marker='d')
    # plot_line_transfer(mean_omac_outfit, std_omac_outfit, '#BEB8DC', "omac", marker='d')
    # plot_line_transfer(mean_parscat_outfit, std_parscat_outfit, '#ff8884', "parscat", marker='d')
    # plot_line_transfer(mean_caduceus_outfit, std_caduceus_outfit, '#845EC2', "caduceus", marker='d')
    # plot_line_transfer(mean_parsagent_outfit, std_parsagent_outfit, '#B39CD0', "parsagent", marker='d')
    # plot_line_transfer(mean_ponpoko_outfit, std_ponpoko_outfit, '#4D8076', "ponpoko", marker='d')
    # plot_line_transfer(mean_yxagent_outfit, std_yxagent_outfit, '#00C9A7', "yxagent", marker='d')

    # plot_line_transfer(mean_agreeable_outfit, std_agreeable_outfit, '#845EC2', "agreeable", marker='d')
    # plot_line_transfer(mean_caduceusDC16_outfit, std_caduceusDC16_outfit, '#B39CD0', "caduceusDC16", marker='d')
    # plot_line_transfer(mean_thefawkes_outfit, std_thefawkes_outfit, '#4D8076', "thefawkes", marker='d')
    # plot_line_transfer(mean_agent36_outfit, std_agent36_outfit, '#00C9A7', "agent36", marker='d')

    # SmartPhone
    # plot_line_transfer(mean_agreeable_smart, std_agreeable_smart, '#845EC2', "agreeable", marker='d')
    # plot_line_transfer(mean_caduceusDC16_smart, std_caduceusDC16_smart, '#B39CD0', "caduceusDC16", marker='d')
    # plot_line_transfer(mean_thefawkes_smart, std_thefawkes_smart, '#4D8076', "thefawkes", marker='d')
    # plot_line_transfer(mean_agent36_smart, std_agent36_smart, '#00C9A7', "agent36", marker='d')

    # Wholesaler
    # plot_line_transfer(mean_agreeable_whole, std_agreeable_whole, '#845EC2', "agreeable", marker='d')
    # plot_line_transfer(mean_caduceusDC16_whole, std_caduceusDC16_whole, '#B39CD0', "caduceusDC16", marker='d')
    # plot_line_transfer(mean_thefawkes_whole, std_thefawkes_whole, '#4D8076', "thefawkes", marker='d')
    # plot_line_transfer(mean_agent36_whole, std_agent36_whole, '#00C9A7', "agent36", marker='d')