from collections import namedtuple
import os 
import numpy as np
import sys
import time
import random
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import tempfile
import shutil
import numpy as np
import pickle
import atexit
from itertools import product
import torchvision
from tqdm import tqdm 

from typing import List
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


#sys.path.append("../")
import Env
from util import *
from models.GPT2 import GPT as GPT

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

Transition = namedtuple('Transition',['state','action','a_log_prob','next_state','reward'])
Transition_reward = namedtuple('Transition_reward',['i','x','y'])
def plot_reward(folder, filename, reward_iter):
    plt.plot(reward_iter)

    name = folder + filename
    name = name + '.png'
    plt.savefig(name, format='png')
    plt.cla()

class PairDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.pairs = list(product(range(len(dataset)), repeat=2))  # 兩兩組合，允許重複
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx_a, idx_b = self.pairs[idx]
        traj_a = self.dataset[idx_a]
        traj_b = self.dataset[idx_b]
        return traj_a, traj_b
    
class PreferenceDataset(Dataset):
    def __init__(
        self,
        max_in_memory_num,  # Maximum number of trajectories to keep in memory
        max_per_file_num,  # Maximum number of trajectories per file
        save_dir=None,  # Directory to save trajectory files
        ac_dim=None,
        t_dim=None
    ):
        self.max_in_memory_num = max_in_memory_num
        self.max_per_file_num = max_per_file_num

        self.save_dir = tempfile.mkdtemp() if save_dir is None else save_dir
        if save_dir is None:
            atexit.register(lambda: shutil.rmtree(self.save_dir))

        self.trajs = []  # Trajectories in memory
        self.traj_files = [os.path.join(self.save_dir, f) for f in os.listdir(self.save_dir) if f.startswith('trajs')]

        self.ac_dim = ac_dim
        self.t_dim = t_dim

    def flush(self):
        """Save trajectories in memory to files."""
        os.makedirs(self.save_dir, exist_ok=True)

        for i in range(0, len(self.trajs), self.max_per_file_num):
            chunk = self.trajs[i:i + self.max_per_file_num]
            filename = os.path.join(self.save_dir, f'trajs-{len(self.traj_files)}.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(chunk, f)
            self.traj_files.append(filename)
        self.trajs = []

    def add_traj(self, new_traj):
        """Add a new trajectory to memory and flush if necessary."""
        self.trajs.append(new_traj)
        #if len(self.trajs) >= self.max_in_memory_num:
        #    self.flush()

    def clear_traj(self):
        self.trajs = []

    def _load_trajectory(self, file_idx):
        """Load trajectories from a specific file."""
        with open(self.traj_files[file_idx], 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        """Return total number of trajectories available."""
        return len(self.trajs) + len(self.traj_files) * self.max_per_file_num

    def __getitem__(self, index):
        """Retrieve a trajectory by index."""
        if index < len(self.trajs):
            return self.trajs[index]
        else:
            file_idx = (index - len(self.trajs)) // self.max_per_file_num
            within_file_idx = (index - len(self.trajs)) % self.max_per_file_num
            file_trajectories = self._load_trajectory(file_idx)
            return file_trajectories[within_file_idx]

    def collate_fn(self, batch):
        """Prepare a batch of trajectory pairs and their preferences."""
        batch_size = len(batch)
        #print(f'batch_size: {batch_size}')
        states_a, states_b, labels = [], [], []

        
        traj_a, traj_b = batch[0], batch[1]

        for traj_a, traj_b in batch:
            state_a = traj_a['states']
            state_b = traj_b['states']

        label = int(torch.sum(traj_a['score']) < torch.sum(traj_b['score']))

        states_a.append(state_a)
        states_b.append(state_b)
        labels.append(label)

        return torch.stack(states_a), torch.stack(states_b), torch.tensor(labels, dtype=torch.int64)

class LargeBatchProcessor:
    ## Input the "comparing dataset" (in shape [traj1, traj2, label]...)
    ## Output the batch packing of "comparing dataset"
    def __init__(self, dataloader, batch_size):
        self.dataloader = dataloader
        self.batch_size = batch_size
    def __iter__(self):
        """Combine smaller batches into larger batches."""
        current_large_batch = []

        for batch in self.dataloader:
            current_large_batch.append(batch)

            # If we have accumulated enough small batches, yield the large batch
            if len(current_large_batch) == self.batch_size:
                yield self._combine_batches(current_large_batch)
                current_large_batch = []
            # Yield any remaining small batches as a final large batch
        if current_large_batch:
            yield self._combine_batches(current_large_batch)

    def _combine_batches(self, batches):
        """Combine multiple smaller batches into a single large batch."""
        states_a = torch.cat([batch[0] for batch in batches], dim=0)
        states_b = torch.cat([batch[1] for batch in batches], dim=0)
        labels = torch.cat([batch[2] for batch in batches], dim=0)
        return states_a, states_b, labels


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, out_d):
        super(PositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(max_seq_len, out_d)
        positional_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j//2) / out_d) for j in range(out_d)]
            for pos in range(max_seq_len)
        ])
        positional_encoding[:, 0::2] = np.sin(positional_encoding[:, 0::2])
        positional_encoding[:, 1::2] = np.cos(positional_encoding[:, 1::2])
        self.embedding.weight = nn.Parameter(torch.tensor(positional_encoding, dtype=torch.float32), requires_grad=False)
    def forward(self, x): # input (batch, seq, 1)
        x = x.squeeze(-1).long()
        return self.embedding(x) # (batch, seq, out_d (2*d_model))

class xy_embedding(nn.Module):
    def __init__(self, grid, d_model):
        super(xy_embedding, self).__init__()
        self.grid = grid
        self.embed_x = nn.Embedding(grid, d_model) #nn.Embedding(input_dictionary, output_size)
        self.embed_y = nn.Embedding(grid, d_model)
    def forward(self, pos): ## pos is in shape (batch, seq, 2)
        """
        x = pos // self.grid
        y = pos %  self.grid
        pos = torch.cat((x,y), dim = 2)
        """
        x_embed = self.embed_x(pos[:, :, 0].long())
        y_embed = self.embed_y(pos[:, :, 1].long())
        xy_embed = torch.cat((x_embed,y_embed), dim = 2)
        return xy_embed

# FeedForward for selfattention
class FeedForward(nn.Module):
    def __init__(self, in_d, hidden_d):
        super().__init__()
        self.fc1 = nn.Linear(in_d, hidden_d)
        self.fc2 = nn.Linear(hidden_d, in_d)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
# Residual connection for selfattention
class ResidualConnection(nn.Module):
    def __init__(self, in_d):
        super().__init__()
        self.norm = nn.LayerNorm(in_d)
    def forward(self, x, sublayer):
        norm_x = self.norm(x)
        return x + sublayer(norm_x)

# Attention Block for selfattention
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_d, num_heads, hidden_d):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_d, num_heads=num_heads)
        self.ffn = FeedForward(in_d, hidden_d)
        self.residual_attn = ResidualConnection(in_d)
        self.residual_ffn = ResidualConnection(in_d)
    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.residual_attn(x, lambda x: self.attn(x, x, x, need_weights=False)[0])
        x = x.transpose(0, 1)
        x = self.residual_ffn(x, self.ffn)
        return x

class SelfAttentionStack(nn.Module):
    def __init__(self, in_d, seq_length, num_heads=1, num_layers=5):
        super(SelfAttentionStack, self).__init__()
        self.hidden_d = 2*in_d
        self.layers = nn.ModuleList([
            SelfAttentionBlock(in_d, num_heads, self.hidden_d) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(in_d, 1)
        self.norm = nn.LayerNorm(seq_length)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x).squeeze(-1)  # Shape: (batch, seq_len)
        x = self.norm(x)
        return x

class RewardFunction(nn.Module):
    def __init__(self, seq_len, grid, d_model):
        super(RewardFunction, self).__init__()
        self.PE = PositionalEncoding(seq_len, 2*d_model)
        self.xy_embedding = xy_embedding(grid, d_model)
        self.Attn = SelfAttentionStack(2*d_model, seq_len)
    def forward(self, input): #input shape:  (batch, seq_len, 3)
        t = input[:,:,0].long() # (batch, seq_len, 1)
        xy = input[:,:,1:3].long() # (batch, seq_len, 2)
        pe = self.PE(t) 
        xy = self.xy_embedding(xy) 
        embed = pe + xy # (batch, seq_len, 2*d_model)
        out = self.Attn(embed) # (batch, seq_len)
        #out2 = out.squeeze(-1) # (batch, seq_len)
        out_sum = out.sum(dim=1, keepdim=True) # (batch, 1)
        return out, out_sum

    
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
    
class SORS():
    def __init__(self, args, placedb_raw, E):
        super(SORS, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained = True).to(device1)
        if args.grid > 100:
            d_model = 32
        else:
            d_model = 8
        self.dataset = PreferenceDataset(max_in_memory_num=10, max_per_file_num=5, ac_dim= 1, t_dim = 1)
        self.cnn_coarse = CNNCoarse(self.resnet).to(device1)
        self.actor_net = Actor(cnncoarse=self.cnn_coarse, grid=E.grid).float().to(device1)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.lr)
        self.args = args
        self.buffer = []
        self.buffer_r = []
        self.counter = 0
        self.counter_r = 0
        self.E = E
        self.clip_param = args.clip_param
        self.max_grad_norm = args.max_grad_norm
        self.epoch = args.epoch
        self.epoch_r = args.epoch_r
        self.epoch_r_sub = args.epoch_r_sub
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
        self.AttnR = RewardFunction(self.manual_placed_num, self.args.grid, d_model).to(device1)
        if args.load_reward == True:
            self.AttnR = torch.load(args.reward_model_file)
        self.reward_optimizer = optim.AdamW(self.AttnR.parameters(), lr=self.args.learning_rate_r, betas = (0.9, 0.98), weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.reward_optimizer, T_max=50)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
    
    def store_reward_model(self):
        torch.save(self.AttnR, self.args.reward_model_file)
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device1).unsqueeze(0)
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
    
    def update_r(self):
        batch_size = len(self.buffer_r) // 1 # how many data in a batch (to be trained in pair)
        batch_num = self.args.epoch_r
        indexed_data = list(enumerate(self.buffer_r))
        for i in tqdm(range(batch_num), desc=f'Epoch: '):
            sampling = random.sample(indexed_data, batch_size)
            sampling_data = [item[1] for item in sampling]
            #sampling_indices = [item[0] for item in sampling]
            for d in sampling_data:
                trajectory_data = {}
                trans, wl = d
                trans = torch.tensor(trans)
                trajectory_data['states'] = trans
                score = torch.tensor(-wl)
                trajectory_data['score'] = score
                self.dataset.add_traj(trajectory_data)
            self.update_r_sub()
            self.dataset.clear_traj()
        self.buffer_r = []

    def update_r_sub(self):
        pair_dataset = PairDataset(self.dataset)
        dataloader = DataLoader(pair_dataset, batch_size=2, collate_fn=self.dataset.collate_fn, shuffle = True)
        for i in tqdm(range(self.epoch_r_sub),desc=f'   Epoch_sub: '):
            batchs = LargeBatchProcessor(dataloader, batch_size=1000)
            for batch in batchs: # one batch update function
                """
                y=0 if x1>x2 (x1 is better trajectory), 1 otherwise.
                """
                x1, x2, y = batch
                x1, x2, y = x1.float().to(device1), x2.float().to(device1), y.long().to(device1).reshape(-1,)
                self.reward_optimizer.zero_grad()
                _, v1 = self.AttnR(x1)
                _, v2 = self.AttnR(x2)
                v1 = v1.reshape(-1,)
                v2 = v2.reshape(-1,)

                logits = torch.stack([v1, v2], dim=1) # [B, 2]
                loss = F.cross_entropy(logits, y)

                loss.backward()
                self.reward_optimizer.step()   

    def update(self):
        # Extract buffer data
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device1)
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device1)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device1)
        del self.buffer[:]
        
        for _ in range(self.epoch):
            for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True),disable = False):
                action_probs = self.actor_net(state[index].to(device1))
                dist = Categorical(action_probs)
                action_log_prob = dist.log_prob(action[index].squeeze()) #squeeze: two-d -> one-d
                ratio = torch.exp(action_log_prob - old_action_log_prob[index].squeeze()) # one-d - one-d
                target_v = reward[index]
                advantage = target_v.detach()

                # Actor optimization
                L1 = ratio * advantage.squeeze() 
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage.squeeze() 
                action_loss = -torch.min(L1, L2).mean() # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
    
    def learn_random_order(self, params, placedb_raw, placedb_dreamplace, NonLinearPlace): # Real SORS, the macros are placed in random order
        # start RL process
        train_iter = 0
        train_iter_r = 0
        episode_iter = 0
        result_dir_temp = params.result_dir
        reward_iter = []
        wl_iter = []
        best_hpwl = 99999999999
        while True:
            episode_reward = 0
            params.result_dir = result_dir_temp + str(episode_iter)
            episode_iter += 1
            state = self.E.reset()
            trans_temp = []
            trans_r_temp = []
            order = list(range(self.placed_macros))
            random.shuffle(order)
            for i in order:
                # If I didn't set random seed, the result will be all same without knowing why
                torch.seed()
                np.random.seed(None)
                # ============================================================================
                state_temp = state.copy()
                action, action_log_prob = self.select_action(state)
                n_state, _, done = self.E.step(action)
                ## store trajectory data (for agent)
                trans = Transition(state_temp, action, action_log_prob, n_state, 0)
                trans_temp.append(trans)
                ## store trajectory data (for reward)
                trans_r = Transition_reward(i, action//self.args.grid, action%self.args.grid) # placed_num, placed_x, placed_y
                trans_r_temp.append(trans_r)
                state = n_state
            wl = cal_hpwl(placedb_raw, self.E.node_pos, self.E.ratio)
    
    def learn(self, params, placedb_raw, placedb_dreamplace, NonLinearPlace): # Real SORS, update RL and reward simultaneously
        # start RL process
        train_iter = 0
        train_iter_r = 0
        episode_iter = 0
        result_dir_temp = params.result_dir
        reward_iter = []
        wl_iter = []
        best_hpwl = 99999999999
        while True:
            episode_reward = 0
            params.result_dir = result_dir_temp + str(episode_iter)
            episode_iter += 1
            state = self.E.reset()
            trans_temp = []
            trans_r_temp = []
            for i in range(self.placed_macros):
                # If I didn't set random seed, the result will be all same without knowing why
                torch.seed()
                np.random.seed(None)
                # ============================================================================
                state_temp = state.copy()
                action, action_log_prob = self.select_action(state)
                n_state, _, done = self.E.step(action)
                ## store trajectory data (for agent)
                trans = Transition(state_temp, action, action_log_prob, n_state, 0)
                trans_temp.append(trans)
                ## store trajectory data (for reward)
                trans_r = Transition_reward(i, action//self.args.grid, action%self.args.grid) # placed_num, placed_x, placed_y
                trans_r_temp.append(trans_r)
                state = n_state
            wl = cal_hpwl(placedb_raw, self.E.node_pos, self.E.ratio)
            """
            ## DREAMPlace
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
            wl = best_temp
            """
            ## store in buffer_r=====================
            self.buffer_r.append((trans_r_temp, wl))
            self.counter_r += 1
            ##=======================================
            logger_temp.info(f'wl: {wl}')
            ## Record renewed (If better than history)
            if wl < best_hpwl:
                best_hpwl = wl
            ## Reward shaping (by reward function) and store trajectory
            trans_r_temp = torch.tensor(trans_r_temp)
            trans_r_temp = trans_r_temp.unsqueeze(0).to(device1)
            reward, reward_sum = self.AttnR(trans_r_temp)
            reward_list = reward.cpu().detach().numpy().flatten().tolist()
            reward_list = [(x+1.5)*10 for x in reward_list]
            for i in range(len(reward_list)):
                episode_reward += reward_list[i]
                trans_temp[i] = trans_temp[i]._replace(reward=reward_list[i])
            reward_iter.append(episode_reward)
            wl_iter.append(wl)
            ## Training the reward
            if self.counter_r % self.args.r_update_period == 0:
                train_iter_r += 1
                print(f'        Train reward at #{train_iter_r}')
                self.update_r()
            ## Training the agent
            if self.store_transition(trans_temp):
                train_iter += 1
                print(f' Train agent at #{train_iter}')
                self.update()
            ## plot 
            plot_reward(self.args.reward_folder,self.args.design,reward_iter)
            plot_reward(self.args.wl_folder,self.args.design,wl_iter)
            plot_macro('macro_no_DREAMPlace.png', self.E.node_pos, self.args.grid)

    def learn_test(self, params, placedb_raw, placedb_dreamplace, NonLinearPlace): # use one time trained reward to perform RL (not real SORS)
        # start RL process
        train_iter = 0
        episode_iter = 0
        result_dir_temp = params.result_dir
        reward_iter = []
        wl_iter = []
        best_hpwl = 99999999999999
        while True:
            episode_reward = 0
            params.result_dir = result_dir_temp + str(episode_iter)
            episode_iter += 1
            state = self.E.reset()
            trans_temp = []
            trans_r_temp = []
            for i in range(self.placed_macros):
                # If I didn't set random seed, the result will be all same without knowing why
                torch.seed()
                np.random.seed(None)
                # =============================================================================
                state_temp = state.copy()
                action, action_log_prob = self.select_action(state)
                n_state, _, done = self.E.step(action)
                trans = Transition(state_temp, action, action_log_prob, n_state, 0)
                trans_temp.append(trans)
                ## store trajectory data
                trans_r = Transition_reward(i, action//self.args.grid, action%self.args.grid) # placed_num, placed_x, placed_y
                trans_r_temp.append(trans_r)
            wl = cal_hpwl(placedb_raw, self.E.node_pos, self.E.ratio)
            logger_temp.info(f'wl: {wl}')
            ## Record renewed (If better than history)
            if wl < best_hpwl:
                best_hpwl = wl
                
            ## Reward shaping (by reward function) and store trajectory
            trans_r_temp = torch.tensor(trans_r_temp)
            trans_r_temp = trans_r_temp.unsqueeze(0).to(device1)
            reward, reward_sum = self.AttnR(trans_r_temp)
            reward_list = reward.cpu().detach().numpy().flatten().tolist()
            #reward_list = [(x+1.5)*10 for x in reward_list]
            logger_reward.info(f'Reward: {reward_list}')
            for i in range(len(reward_list)):
                episode_reward += reward_list[i]
                trans_temp[i] = trans_temp[i]._replace(reward=reward_list[i])
            reward_iter.append(episode_reward)
            wl_iter.append(wl)
            ## Training
            if self.store_transition(trans_temp):
                train_iter += 1
                print(f' Train at #{train_iter}')
                self.update()
            ## plot
            plot_reward(self.args.reward_folder,self.args.design,reward_iter)
            plot_reward(self.args.wl_folder,self.args.design,wl_iter)




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
            
             

    def train_reward_test(self): # train reward only ONCE
        data_path = "./reward_training_data/adaptec1.pkl"
        if os.path.exists(data_path):
            with open(data_path, "rb") as file:
                data = pickle.load(file)
        else:
            data = []
        print(len(data))
        if len(data) < 5000:
            print('Not enough')
            sys.exit()
        data = data[:5000]
        indexed_data = list(enumerate(data))
        num = [0] * 5000
        batch_num = 20
        for i in tqdm(range(batch_num),desc=f'Epoch: '):
            sampling = random.sample(indexed_data, 500)
            sampling_data = [item[1] for item in sampling]
            sampling_indices = [item[0] for item in sampling]
            for d in sampling_data: # 這裡別把它改成data，一定會當機ㅎㅅㅎ
                trajectory_data = {}
                trans, wl = d
                trans = torch.tensor(trans)
                trajectory_data['states'] = trans
                score = torch.tensor(-wl)
                trajectory_data['score']  = score
                self.dataset.add_traj(trajectory_data)
            for idx in sampling_indices:
                num[idx] += 1
            self.update_r_sub()
            self.dataset.clear_traj()
        sorted_indices = sorted(range(len(num)), key=lambda x: num[x], reverse=True)
        sorted_data = [data[idx] for idx in sorted_indices[:3000]]
        
        """
        data = data[:20000]
        batch_size = 500
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        for batch in batches:
            for d in batch: # 這裡別把它改成data，一定會當機ㅎㅅㅎ
                trajectory_data = {}
                trans, wl = d
                trans = torch.tensor(trans)
                trajectory_data['states'] = trans
                score = torch.tensor(-wl)
                trajectory_data['score']  = score
                self.dataset.add_traj(trajectory_data)
            self.update_r()
            self.dataset.clear_traj()
        """
        raw_score = []
        after_score = []
        for d in sorted_data:
            trans, wl = d
            raw_score.append(-wl)
            trans = torch.tensor(trans)
            trans = trans.unsqueeze(0).to(device1)
            _, score_sum = self.AttnR(trans)
            after_score.append(score_sum.item())
        for d in data:
            trans, wl = d
            trans = torch.tensor(trans)
            trans = trans.unsqueeze(0).to(device1)
            a, a_sum = self.AttnR(trans)
            print(f'a: {a}')
            print(f'a_sum: {a_sum}')
            break
        
        plot_reward(self.args.reward_folder,'raw',raw_score)  
        plot_reward(self.args.reward_folder,'after',after_score) 
        # store the network
        self.store_reward_model()


    

            


