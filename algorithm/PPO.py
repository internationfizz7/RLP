import argparse
from collections import namedtuple
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import sys
import time
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

#sys.path.append("../")
import Env
import torchvision
from tqdm import tqdm
import random   
from util import *
from manual_reward_tunning import reward_tunning, region_map

# set device to cpu or cuda
device = torch.device('cuda:0')

#logger
logger_reward = setup_logger('logger_reward',  'loggers/logger_reward.log')
logger_reward.disabled = False
torch.set_printoptions(threshold=float('inf'))   # print all tensor
Transition = namedtuple('Transition',['state','action','a_log_prob','next_state','reward'])
Transition_reward = namedtuple('Transition_reward',['i','x','y'])
def plot_reward(folder, filename, reward_iter):
    plt.plot(reward_iter)
    name = folder + filename
    name = name + '.png'
    plt.savefig(name, format='png')
    plt.cla()

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

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.state_value = nn.Linear(64, 1)
        self.pos_emb = nn.Embedding(1400, 64)
    def forward(self, x):
        x1 = F.relu(self.fc1(self.pos_emb(x[:, 0].long()))) #x[:, 0] is t (in paper), pos_emd output postion embedding , than fed into FC(two level relu)
        x2 = F.relu(self.fc2(x1)) 
        value = self.state_value(x2)
        return value


class PPO():
    def __init__(self, args, placedb_raw, E):
        super(PPO, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained = True).to(device)
        self.cnn_coarse = CNNCoarse(self.resnet).to(device)
        self.actor_net = Actor(cnncoarse=self.cnn_coarse, grid=E.grid).float().to(device)
        self.critic_net = Critic().float().to(device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), args.lr)
        self.args = args
        self.buffer = []
        self.counter = 0
        self.E = E
        self.clip_param = args.clip_param
        self.max_grad_norm = args.max_grad_norm
        self.epoch = args.epoch
        self.manual_placed_num = args.manual_placed_num
        if args.manual_placed_num == None:
            self.placed_macros = placedb_raw.node_cnt
            self.buffer_capacity = self.args.update_period * placedb_raw.node_cnt
        else:
            if args.manual_placed_num > placedb_raw.node_cnt:
                self.manual_placed_num = placedb_raw.node_cnt
                args.manual_placed_num = self.manual_placed_num
            self.buffer_capacity = self.args.update_period * (self.manual_placed_num)
            self.placed_macros = self.manual_placed_num
        self.batch_size = args.batch_size
        self.gamma = args.gamma
    
    def load_nn(self, path):
        ### TODO: load pretrained network
        return 0
    
    def save_nn(self, path):
        ### TODO: safe network
        return 0
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor_net(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()
    
    def store_transition(self, transition_mem):
        # transition_mem is a list
        self.buffer.extend(transition_mem)
        self.counter+=len(transition_mem)
        return self.counter % self.buffer_capacity == 0 # return true if the buffer is full, else false
    
    def update(self):
        # Extract buffer data
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        del self.buffer[:]
        # Discounted reward
        target_list = []
        target = 0
        for i in range(reward.shape[0]-1, -1, -1):
            if state[i, 0] >= self.placed_macros -1:
                target = reward[i, 0].item()
            #r = reward[i, 0].item()
            #target = r + self.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_v_all = torch.tensor(np.array([t for t in target_list]), dtype=torch.float).view(-1, 1).to(device)

        for _ in range(self.epoch):
            for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True),disable = False):
                action_probs = self.actor_net(state[index].to(device))
                dist = Categorical(action_probs)
                action_log_prob = dist.log_prob(action[index].squeeze()) #squeeze: two-d -> one-d
                ratio = torch.exp(action_log_prob - old_action_log_prob[index].squeeze()) # one-d - one-d
                target_v = target_v_all[index]
                #critic_net_output = self.critic_net(state[index].to(device))
                advantage = target_v.detach()
                #advantage = (target_v - critic_net_output).detach()

                # Actor optimization
                L1 = ratio * advantage.squeeze() 
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage.squeeze() 
                action_loss = -torch.min(L1, L2).mean() # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic optimizer
                #value_loss = F.smooth_l1_loss(self.critic_net(state[index].to(device)), target_v)
                #self.critic_net_optimizer.zero_grad()
                #value_loss.backward()
                #nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                #self.critic_net_optimizer.step()

    def learn(self, params, placedb_dreamplace, NonLinearPlace): ##神棍版本 RL from manual dense reward
        # Start RL process
        train_iter = 0
        episode_iter = 0
        result_dir_temp = params.result_dir
        reward_iter = []
        wl_iter = []
        best_hpwl = 99999999999
        while True:
            episode_reward = 0
            episode_iter += 1
            state = self.E.reset()
            trans_temp = []
            for _ in range(self.placed_macros):
                # If I didn't set random seed, the result will be all same without knowing why
                torch.seed()
                np.random.seed(None)
                # ============================================================================
                state_tmp = state.copy()
                action, action_log_prob = self.select_action(state)
                n_state, reward, done = self.E.step(action)
                episode_reward += reward
                trans = Transition(state_tmp, action, action_log_prob, n_state, reward)
                trans_temp.append(trans)
                state = n_state
            plot_macro('macro_no_DREAMPlace.png', self.E.node_pos, self.args.grid)
            np.random.seed(params.random_seed)
            ## Reset cells pos
            placedb_dreamplace.node_x[0:placedb_dreamplace.num_movable_nodes] = np.random.normal(
                loc=(placedb_dreamplace.xl * 1.0 + placedb_dreamplace.xh * 1.0) / 2,
                scale=(placedb_dreamplace.xh - placedb_dreamplace.xl) * 0.001,
                size=placedb_dreamplace.num_movable_nodes)
            placedb_dreamplace.node_y[0:placedb_dreamplace.num_movable_nodes] = np.random.normal(
                loc=(placedb_dreamplace.yl * 1.0 + placedb_dreamplace.yh * 1.0) / 2,
                scale=(placedb_dreamplace.yh - placedb_dreamplace.yl) * 0.001,
                size=placedb_dreamplace.num_movable_nodes)
            ## Write in placedb_dreamplace
            for n in self.E.node_pos:
                if n == "V":
                    continue
                x, y, _, _ = self.E.node_pos[n]
                x = x * self.E.scale_x
                y = y * self.E.scale_y
                id = placedb_dreamplace.node_name2id_map[n]
                placedb_dreamplace.node_x[id] = x
                placedb_dreamplace.node_y[id] = y
            ## DREAMPlace
            best_temp = 9999999999999999
            iter = 0
            while True:
                np.random.seed(params.random_seed)
                params.result_dir = result_dir_temp + str(episode_iter) + '_' + str(iter)
                placer = NonLinearPlace.NonLinearPlace(params, placedb_dreamplace, None)
                metrics = placer(params, placedb_dreamplace)
                wl = float(metrics[-1].hpwl.data)
                iter += 1
                if wl >= best_temp:
                    break
                else:
                    best_temp = wl
            
            ## Record renewed (If better than history)
            if wl < best_hpwl:
                best_hpwl = wl
                logger_reward.info(f'Best record: {best_hpwl}')
            ## Reward tunning
            hpwl_reward = reward_tunning(self.args.design, wl)
            ## Give the episode reward back to transition
            """"
            for tran in trans_temp:
                tran = tran._replace(reward = reward + hpwl_reward)
            trans_temp[-1] = trans_temp[-1]._replace(reward=episode_reward)
            """
            reward_iter.append(episode_reward)
            wl_iter.append(wl)
            ## Training (if the buffer is full)
            if self.store_transition(trans_temp):
                train_iter += 1
                print(f' Train at #{train_iter}')
                self.update()
            ## plot
            plot_reward(self.args.reward_folder,self.args.design,reward_iter)
            plot_reward(self.args.wl_folder,self.args.design,wl_iter)
            
    
    def learn_sparse(self, params, placedb_raw): ## RL from sparse reward
        # Start RL process
        train_iter = 0
        episode_iter = 0
        result_dir_temp = params.result_dir
        reward_iter = []
        wl_iter = []
        best_hpwl = 99999999999
        overall_wl = 0
        while True:
            episode_reward = 0
            episode_iter += 1
            state = self.E.reset()
            trans_temp = []
            for _ in range(self.placed_macros):
                # If I didn't set random seed, the result will be all same without knowing why
                torch.seed()
                np.random.seed(None)
                # ============================================================================
                state_tmp = state.copy()
                action, action_log_prob = self.select_action(state)
                n_state, reward, done = self.E.step(action)
                #episode_reward += reward
                trans = Transition(state_tmp, action, action_log_prob, n_state, 0)
                trans_temp.append(trans)
                state = n_state
            
            wl = cal_hpwl(placedb_raw, self.E.node_pos, self.E.ratio)
            
            ## Record renewed (If better than history)
            if wl < best_hpwl:
                best_hpwl = wl
                logger_reward.info(f'Best record: {best_hpwl}')
            ## Reward tunning
            #hpwl_reward = reward_tunning(self.args.design, wl)
            episode_reward = -wl
            ## Give the episode reward back to transition
            """"
            for tran in trans_temp:
                tran = tran._replace(reward = reward + hpwl_reward)
            trans_temp[-1] = trans_temp[-1]._replace(reward=episode_reward)
            """
            trans_temp[-1] = trans_temp[-1]._replace(reward=episode_reward)
            reward_iter.append(episode_reward)
            overall_wl += wl
            ## Training (if the buffer is full)
            if self.store_transition(trans_temp):
                wl_iter.append(overall_wl)
                overall_wl = 0
                train_iter += 1
                print(f' Train at #{train_iter}')
                self.update()
            ## plot
            plot_reward(self.args.reward_folder,self.args.design,reward_iter)
            plot_reward(self.args.wl_folder,self.args.design,wl_iter)
            plot_macro('macro_no_DREAMPlace.png', self.E.node_pos, self.args.grid)

    ## function to generate reward training data (no containing training)    
    def reward_data_generate(self, placedb_raw):
        data_path = "./reward_training_data/adaptec1.pkl"
        print('Start generating placing data...')
        episode_data_list = []
        try:
            while True:
                state = self.E.reset()
                trans_temp = []
                for i in range(self.placed_macros):
                    torch.seed()
                    np.random.seed(None)
                    state_tmp = state.copy()
                    action, action_log_prob = self.select_action(state)
                    n_state, reward, done = self.E.step(action)
                    trans = Transition_reward(i, action//self.args.grid, action%self.args.grid) # placed_num, placed_x, placed_y
                    trans_temp.append(trans)
                    state = n_state
                wl = cal_hpwl(placedb_raw, self.E.node_pos, self.E.ratio)
                episode_data_list.append((trans_temp, wl))
        except KeyboardInterrupt:
            print('Store data...')
            if os.path.exists(data_path):
                with open(data_path, "rb") as file:
                    data = pickle.load(file)
            else:
                data = []
            for episode_data in episode_data_list:
                data.append(episode_data)
            with open(data_path, "wb") as file:
                pickle.dump(data, file)
            
            

     


