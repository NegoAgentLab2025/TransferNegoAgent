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
            #source = scipy.signal.savgol_filter(source,7,3)
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

def plot_block(x_range):
    path_tom0 = "D:\\multi_issue_negotiation_simple\\txt\\tom\\tom\\tom0"
    path_tom1 = "D:\\multi_issue_negotiation_simple\\txt\\tom\\tom\\tom1"

    mean_tom0, std_tom0 = get_mean_std(path_tom0, epis=20000)
    mean_tom1, std_tom1 = get_mean_std(path_tom1, epis=20000)

    plot_line_transfer(mean_tom0, std_tom0, '#2878b5', "Bayes-ToM0", marker='d')
    plot_line_transfer(mean_tom1, std_tom1, '#4D8076', "Bayes-ToM1", marker='d')

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
    plt.xticks(x[::25], [8 * i for i in x[::25]], fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 1.0)
    plt.xlim(0, 100)
    # plt.ylim(-280, -70, 50)

    # plt.scatter(226, 0.64, s=200, color='green')
    # plt.scatter(207, 0.64, s=200, color='#EE7942')
    plt.legend(loc='lower right', fontsize=20)
    # plt.margins(0, 0)
    # plt.show()

    #plt.savefig('{}_{}_{}.pdf'.format(transfer, opponent_agent, evaluate_domain), bbox_inches='tight', pad_inches=0)
    plt.savefig('ns.pdf', bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    sns.set_style('darkgrid')
    # sns.set_context(font_scale=3)
    matplotlib.rcParams.update({'font.size': 40, 'font.family': 'serif'})

    delta = 1

    plot_block(100)
