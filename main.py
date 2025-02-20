import logging
import coloredlogs
import sys
#import gym
import numpy as np
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

from collections import namedtuple
from util import *
from feature import features, feature_extraction
from Env.Env224 import PlaceingEnv as Env224
from Env.Env16  import PlaceingEnv as Env16
from algorithm.PPO import PPO as Agent_PPO
from algorithm.SORS import SORS as Agent_SORS
from algorithm.BERTangent import BERT as Agent_BERTangent
from algorithm.AttentionReward import SORS as Agent_AttnR
from algorithm.DREAMPlace_test import DREAM_test as Agent_DREAM_test
from algorithm.L_grid import L_grid as Agent_Large_grid
from place_db.DREAMPlace_setup import *
sys.path.append("../install/dreamplace/")
sys.path.append("../install/")
import dreamplace.configure as configure
import Params
import PlaceDB
import NonLinearPlace

from placeDB import placeDB as placeDB
from manual_reward_tunning import reward_tunning, region_map

np.set_printoptions(threshold=np.inf)
logger_reward = setup_logger('logger_reward',  'loggers/logger_reward.log')
logger_reward.disabled = False


args = dotdict({
    #============Unmutable args================================#
    'data_path': './data/',
    'plt_path': './plot/', ## Store every plot information, include WL distribution, center info distribution for every node, etc...
    'design_folder': '../benchmark',
    'reward_folder': './reward/',
    'wl_folder': './wl/',
    #==============Mutable args================================#
    'design': 'adaptec1',
    'clip_param': 0.2,
    'max_grad_norm': 0.5,
    'epoch': 10, #default: 10
    'update_period': 15, # how many episodes per update (agent)
    #       ===reward setting===       #
    'epoch_r_sub': 3, # for one epoch_sub: train using [numbers] of data
    'epoch_r': 1, # for one epoch: randomly pick [numbers] of data to training
    'r_update_period': 200, # how many episodes per update (reward) default(200)
    'learning_rate_r': 1e-4,
    'load_reward': False,
    'reward_model_file': './network/reward_model.pth',
    #       ===reward setting===       #
    'manual_placed_num': 120,  ## None: All placed 
    'grid': 224,
    'batch_size': 64,
    'gamma': 0.99, # RL discount factor default 0.99
    'lr': 2.5e-4, #learning rate  default: 2.5e-3
    'algorithm': 'AttnR', # PPO, SORS, SAC, DREAM_test, Large-grid, BERTagent, AttnR ...

})
if args.algorithm == 'Large_grid' or args.algorithm == 'BERTagent':
    args.grid = 16


device0 = torch.device('cuda:0') 
device1 = torch.device('cuda:1')
torch.cuda.empty_cache()

Transition = namedtuple('Transition',['state','action','a_log_prob','next_state','reward'])

def plot_reward(folder, filename, reward_iter):
    plt.plot(reward_iter)
    name = folder + filename
    name = name + '.png'
    plt.savefig(name, format='png')
    plt.cla()


def main():
    # Generate placing data (macros)
    placedb_raw = placeDB(args)

    # Load in processed feature data
    #feature_tool = features(args.data_path + args.design + '_processed.pkl')
    #TCG_trim, center_trim, region_trim, neighbor_trim = feature_tool.trim_data()
    #r_map = region_map(region_trim, placedb_raw, args.grid)
    r_map = None

    # Construct RL env
    if args.algorithm == 'Large_grid':
        E = Env16(args.grid, args, placedb_raw, r_map)
    else:
        E = Env224(args.grid, args, placedb_raw, r_map)
    # Initialize RL agent and alg
    if args.algorithm == 'PPO':
        A = Agent_PPO(args, placedb_raw, E)
    elif args.algorithm == 'SORS':
        A = Agent_SORS(args, placedb_raw, E)
    elif args.algorithm == 'DREAM_test':
        A = Agent_DREAM_test(args, placedb_raw, E)
    elif args.algorithm == 'Large_grid':
        A = Agent_Large_grid(args, placedb_raw, E)
    elif args.algorithm == 'BERTagent':
        A = Agent_BERTangent(args, placedb_raw, E)
    elif args.algorithm == 'AttnR':
        A = Agent_AttnR(args, placedb_raw, E)
    
    # Generate DREAMPlace setup
    #params, placedb_dreamplace = DREAMPlace_setup(A, placedb_raw, args)
    params, placedb_dreamplace = DREAMPlace_setup2(args)
    # !!Note!! #
    # If you use setup2, you need to place every cell back
    # to center in every iteration. Or it will use the previous
    # result.   #

    A.learn(params, placedb_raw, placedb_dreamplace, NonLinearPlace)
    #A.learn_sparse(params, placedb_raw)
    #A.train_reward_test()
    #A.reward_data_generate(placedb_raw)
    

if __name__ == "__main__":
    main()
