from collections import namedtuple
import os 
import numpy as np
import sys
import time
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tempfile
import shutil
import numpy as np
import pickle
import atexit

from typing import List
from torch.distributions import Categorical
from util import *
import torchvision
from tqdm import tqdm 

device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')
logger_reward = setup_logger('logger_reward',  'loggers/logger_reward.log')
logger_reward.disabled = False
torch.set_printoptions(threshold=float('inf'))

class CNNCoarse(nn.Module):
    def __init__(self, res_net):
        super(CNNCoarse, self).__init__()
        self.cnn = res_net
        self.cnn.fc = torch.nn.Linear(512, 16*7*7)
        self.deconv = nn.Sequential( #global mask decorder
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding = 1), #14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding = 1), #28
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding = 1), #56
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1, output_padding = 1), #112
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding = 1), #224
        )
    def forward(self, x):
        x = self.cnn(x).reshape(-1, 16, 7, 7)
        return self.deconv(x)
    
class Actor(nn.Module):
    def __init__(self, cnncoarse, grid):
        super(Actor, self).__init__()
        self.grid = grid
        self.cnncoarse = cnncoarse
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        cnn_coarse = x[:, 1: 1+self.grid*self.grid].reshape(-1,1,self.grid,self.grid)
        cnn_coarse_3c = cnn_coarse.repeat(1, 3, 1, 1) # repeat the original image from 1 channel to 3 channels
        cnn_coarse_output = self.cnncoarse(cnn_coarse_3c).reshape(-1, self.grid*self.grid)
        mask = x[:, 1+self.grid*self.grid: 1+self.grid*self.grid*2].float().reshape(-1, self.grid*self.grid)
        out = torch.where(mask < 1.0, -1.0e10, cnn_coarse_output.double())
        out = self.softmax(out)
        return out

class DREAM_test():
    def __init__(self, args, placedb_raw, E):
        super(DREAM_test, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained = True).to(device1)
        self.cnn_coarse = CNNCoarse(self.resnet).to(device1)
        self.actor_net = Actor(cnncoarse=self.cnn_coarse, grid=E.grid).float().to(device1)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.lr)
        
        self.E = E
        self.args = args
        self.manual_placed_num = args.manual_placed_num
        if args.manual_placed_num == None:
            self.placed_macros = placedb_raw.node_cntself.buffer_capacity = self.update_period * placedb_raw.node_cnt
        else:
            if args.manual_placed_num > placedb_raw.node_cnt:
                self.manual_placed_num = placedb_raw.node_cnt
                args.manual_placed_num = self.manual_placed_num
            self.placed_macros = self.manual_placed_num
    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device1).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor_net(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()
    def learn(self, params, placedb_dreamplace, NonLinearPlace):
        state = self.E.reset()
        for i in range(self.placed_macros):
            # If I didn't set random seed, the result will be all same without knowing why
            torch.seed()
            np.random.seed(None)
            # ============================================================================
            state_tmp = state.copy()
            action, action_log_prob = self.select_action(state)
            n_state, _, done = self.E.step(action)
            state = n_state
        while True:
            ## write in placedb_dreamplace
            node_pos_temp = self.E.node_pos
            for n in self.E.node_pos:
                if n == "V":
                    continue
                x, y, x_m, y_m = self.E.node_pos[n]
                x = x * self.E.ratio
                y = y * self.E.ratio
                id = placedb_dreamplace.node_name2id_map[n]
                placedb_dreamplace.node_x[id] = x + random.randint(-10, 10)
                placedb_dreamplace.node_y[id] = y + random.randint(-10, 10)
                node_pos_temp[n] = (placedb_dreamplace.node_x[id]/self.E.ratio, placedb_dreamplace.node_y[id]/self.E.ratio, x_m, y_m)
            np.random.seed(params.random_seed)
            placer = NonLinearPlace.NonLinearPlace(params, placedb_dreamplace, None)
            metrics = placer(params, placedb_dreamplace)
            wl = float(metrics[-1].hpwl.data)
            logger_reward.info(f'This episode wl: {wl}')
            plot_macro('macro_no_DREAMPlace.png', node_pos_temp, self.args.grid)
