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
from itertools import product

from typing import List
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from transformers import BertConfig, BertModel

#sys.path.append("../")
import Env
from util import *
import torchvision
from tqdm import tqdm 


# set device to cuda
device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')

# logger
logger_reward = setup_logger('logger_reward',  'loggers/logger_reward.log')
logger_reward.disabled = False
logger_temp = setup_logger('logger_temp', 'loggers/logger_temp.log')
logger_temp.disabled = False
logger_temp2 = setup_logger('logger_temp2', 'loggers/logger_temp2.log')
logger_temp2.disabled = False
torch.set_printoptions(threshold=float('inf'))

Transition = namedtuple('Transition',['state','action_list','action','a_log_prob','next_state','reward'])

def plot_reward(folder, filename, reward_iter):
    plt.plot(reward_iter)
    name = folder + filename
    name = name + '.png'
    plt.savefig(name, format='png')
    plt.cla()

class BERTpredictor(nn.Module):
    def __init__(self, config, action_size):
        super().__init__()
        self.bert = BertModel(config)
        self.fc = nn.Linear(config.hidden_size, 8*1*1)
        self.deconv = nn.Sequential( # recover to a image-like probability distributions from bert
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding = 1), #2
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding = 1), #4
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1, output_padding = 1), #8
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding = 1), #16
            #nn.ReLU(),
            #nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding = 1), #224
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
        pooled_output = outputs.pooler_output
        out = self.fc(pooled_output).reshape(-1, 8, 1, 1)
        predictions = self.deconv(out)
        
        return predictions

class Actor(nn.Module):
    def __init__(self, bert, grid, max_length, action_num):
        super(Actor, self).__init__()
        self.grid = grid
        self.bert = bert
        self.max_length = max_length
        self.action_num = action_num
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, x, seq): ## x: input, seq: a sequence of each actions until t
        B_actions, B_attention_masks = self.encode(seq)
        B_actions = torch.tensor(B_actions).long().to(device1)
        B_attention_masks = torch.tensor(B_attention_masks).long().to(device1)
        bert_out = self.bert(B_actions, B_attention_masks).reshape(-1, self.grid * self.grid)
        #mask = x[:, 1+self.grid*self.grid: 1+self.grid*self.grid*2].float().reshape(-1, self.grid*self.grid)
        #out = torch.where(mask < 1.0, -1.0e10, bert_out.double())
        out = bert_out
        out = self.softmax(out)
        return out
    
    def encode(self, sequences):
        batch_actions = []
        batch_attention_masks = []
        for sequence in sequences:
        
            actions = [action for action in sequence]
            if len(actions) < self.max_length:
                actions = actions + [self.action_num] * (self.max_length-len(actions))
            else:
                actions = actions[:self.max_length]
            attention_mask = [1 if a != self.action_num else 0 for a in actions]
            
            batch_actions.append(actions)
            batch_attention_masks.append(attention_mask)
        return batch_actions, batch_attention_masks
        
        
class BERT():
    def __init__(self, args, placedb_raw, E):
        super(BERT, self).__init__()
        self.E = E
        self.args = args
        ## BERT config
        config = BertConfig(
            vocab_size = args.grid*args.grid+1,
            hidden_size = 128,
            num_hidden_layers = 4,
            num_attention_heads = 4,
            intermediate_size = 728,
            max_position_embeddings = args.manual_placed_num,
            type_vocab_size = 1,
            hidden_dropout_prob = 0.1,
            attention_probs_dropout_prob = 0.1,
        )
        ## BERT
        self.bert = BERTpredictor(config,args.grid*args.grid).to(device1)
        self.actor_net = Actor(bert = self.bert, grid=E.grid, max_length = args.manual_placed_num, action_num= args.grid*args.grid).float().to(device1)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.lr)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.current_episode = []
        self.buffer = []
        self.counter = 0
        self.clip_param = args.clip_param
        self.max_grad_norm = args.max_grad_norm
        self.epoch = args.epoch
        self.update_period = args.update_period
        self.manual_placed_num = args.manual_placed_num
        if args.manual_placed_num == None:
            self.placed_macros = placedb_raw.node_cnt
            self.buffer_capacity = self.update_period * placedb_raw.node_cnt
        else:
            if args.manual_placed_num > placedb_raw.node_cnt:
                self.manual_placed_num = placedb_raw.node_cnt
                args.manual_placed_num = self.manual_placed_num
            self.buffer_capacity = self.update_period * (self.manual_placed_num)
            self.placed_macros = self.manual_placed_num
            

    def select_action(self, state, action_list):
        state = torch.from_numpy(state).float().to(device1).unsqueeze(0)
        
        if len(action_list) == 0:
            num_actions = self.args.grid*self.args.grid  
            action_probs = torch.ones(num_actions).float().to(device1) / num_actions  
        else:
            action_list = [action_list]
            with torch.no_grad():
                action_probs = self.actor_net(state, action_list)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()
    
    def store_transition(self, transition_mem): # if full, flush
        # transition_mem is a list
        self.buffer.extend(transition_mem)
        self.counter+=len(transition_mem)
        return self.counter % self.buffer_capacity == 0 # return true if the buffer is full, else false
    
    def update(self):
        # Extract buffer data
        action_list = [t.action_list for t in self.buffer]
        ## since action_list is ragged,,
        max_length = self.args.manual_placed_num
        padded_list = [sublist + [self.args.grid*self.args.grid] * (max_length - len(sublist)) for sublist in action_list]
        action_list = torch.tensor(padded_list)
        
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device1)
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device1)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device1)
        del self.buffer[:]
        # Discounted reward
        target_list = []
        target = 0
        for i in range(reward.shape[0]-1, -1, -1):
            if state[i, 0] >= self.placed_macros -1:
                target = 0
            r = reward[i, 0].item()
            target = r + self.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_v_all = torch.tensor(np.array([t for t in target_list]), dtype=torch.float).view(-1, 1).to(device1)

        for _ in range(self.epoch):
            for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True),disable = False):
                action_probs = self.actor_net(state[index].to(device1), action_list[index])
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
    
    def learn(self, params, placedb_raw, placedb_dreamplace, NonLinearPlace):
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
            action_list = []
            for _ in range(self.placed_macros):
                # If I didn't set random seed, the result will be all same without knowing why
                torch.seed()
                np.random.seed(None)
                # ============================================================================
                state_tmp = state.copy()
                action_list_temp = action_list.copy()
                action, action_log_prob = self.select_action(state, action_list)
                action_list.append(action)
                n_state, reward, done = self.E.step(action)
                episode_reward += reward
                trans = Transition(state_tmp, action_list_temp, action, action_log_prob, n_state, 0)
                trans_temp.append(trans)
                state = n_state
            wl = cal_hpwl(placedb_raw, self.E.node_pos, self.E.ratio)
            logger_temp.info(f'wl: {wl}')
            ## Record renewed (If better than history)
            if wl < best_hpwl:
                best_hpwl = wl
                logger_reward.info(f'Best record: {best_hpwl}')
            trans_temp[-1] = trans_temp[-1]._replace(reward=-wl)
            wl_iter.append(wl)
            ## Training (if the buffer is full)
            if self.store_transition(trans_temp):
                train_iter += 1
                print(f' Train at #{train_iter}')
                self.update()
            ## plot
            plot_reward(self.args.wl_folder,self.args.design,wl_iter)
            plot_macro('macro_no_DREAMPlace.png', self.E.node_pos, self.args.grid)
            