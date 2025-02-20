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

Transition = namedtuple('Transition',['state','action','a_log_prob','next_state','reward'])

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

class TimeDistributed(nn.Module): #使每一個timestep的序列輸入都有一個對應的Linear層
    def __init__(self, layer):
        super(TimeDistributed, self).__init__()
        self.layer = layer

    def forward(self, x):
        outputs = []
        batch, time_steps = x.shape[:2]
        x_reshaped = x.contiguous().view(batch* time_steps, *x.size()[2:])
        out = self.layer(x_reshaped)

        # 3. 將結果重塑回 (batch_size, time_steps, ...)
        out = out.view(batch, time_steps, *out.size()[1:])
        """
        for t in range(*x.shape[:2]):
            xt = x_reshaped[:,t,:].view(batch, 1,*x.shape[2:])
            output = self.layer(xt)
            outputs.append(output.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        """
        return out

class Phi(nn.Module):
    def __init__(self, grid, resnet, out_dim, last_activation='relu', time_distributed=False):
        """
        A CNN (res_net) with optional TimeDistributed support.
        """
        super(Phi, self).__init__()
        self.grid = grid
        self.layers = nn.ModuleList()
        self.cnn = resnet
        self.cnn.fc = torch.nn.Linear(512,out_dim)
        if time_distributed:
            self.cnn = TimeDistributed(self.cnn)
        final_sequence = [self.cnn]
        if last_activation:
            last_activation = getattr(nn, last_activation.capitalize(), nn.ReLU)()
            final_sequence.append(last_activation)
        self.layers.append(nn.Sequential(*final_sequence))

        

    def forward(self, inputs):
        # If inputs is a list, concatenate along the last dimension
        b, t = inputs.shape[:2]
        x = inputs[:,:, 1: 1+1*self.grid*self.grid].view(b,t,1,self.grid,self.grid)
        x = x.repeat(1,1,3,1,1)
        for layer in self.layers:
            x = layer(x)
        return x

    def decay_vars(self):
        """
        Return only weights (without biases) for all layers in the network.
        """
        return [layer[0].weight for layer in self.layers if isinstance(layer[0], nn.Linear)]

class W(nn.Module):
    def __init__(self, in_dim, out_dim, time_distributed=False):
        """
        A multilayer perceptron (MLP) with optional TimeDistributed support.
        """
        super(W, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        if time_distributed:
            self.layer = TimeDistributed(self.layer)

    def forward(self, inputs):
        # If inputs is a list, concatenate along the last dimension
        if isinstance(inputs, (list, tuple)):
            inputs = torch.cat(inputs, dim=-1)
        
        x = inputs
        x = self.layer(x)
        return x

class RewardV2(nn.Module):
    """
    Seperate phi and reward_weight
    """
    def __init__(self, grid, phi_dim = 4, ac_dim = 1, 
                 use_state_and_action = False):
        super(RewardV2, self).__init__()
        self.grid = grid
        self.ob_dim = 1+ 2*grid*grid
        self.use_state_and_action = use_state_and_action
        if self.use_state_and_action:
            self.in_dim = self.ob_dim + ac_dim
        else:
            self.in_dim = self.ob_dim

        
        self.phi_dim = phi_dim 

        self.w_net = W(self.phi_dim, 1)
        self.phi_net = Phi(grid = self.grid,resnet = torchvision.models.resnet18(pretrained = True), out_dim = phi_dim,last_activation = 'tanh', time_distributed = True)
        
    def R(self, x):
        """
        inp:
            x: Ragged Tensor [B,T(None;ragged),feature_dim]
        out:
            R: tf.Tensor shape of [B]
        """
        batch_size = x.size(0)
        phi = self.phi_net(x)
        return torch.sum(self.w_net(phi), dim=1).squeeze(-1)
    def prepare_update(self, learning_rate, weight_decay=1e-5):
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        def update_fn(batch):
            """
            y=0 if x1> x2 (x1 is better trajectory) 1 otherwise.
            """
            x1, x2, y = batch
            x1, x2, y = x1.float().to(device1), x2.float().to(device1), y.long().to(device1).reshape(-1,)
            optimizer.zero_grad()
            v1 = self.R(x1).reshape(-1,)
            v2 = self.R(x2).reshape(-1,)

            logits = torch.stack([v1, v2], dim=1) # [B, 2]
            loss = F.cross_entropy(logits, y)

            loss.backward()
            optimizer.step()

            return loss.item()
        return update_fn
    def forward(self, s, a):
        """
        inp: 
            s: Tensor [B, state_dim]
            a: Tensor [B, action_dim]
        med:
            s: Tensor [B, 1, state_dim]
            a: Tensor [B, 1, action_dim]
        out:
            R: Tensor [B]
        """
        batch_s, batch_a = s.shape[:1], a.shape[:1]
        s = s.reshape(*batch_s, 1, -1)
        a = a.reshape(*batch_a, 1, -1)
        return torch.matmul(self.phi(s, a), self.w()).squeeze(-1)
    def phi(self, s, a):
        """
        inp:
            s: Tensor [B, state_dim]
            a: Tensor [B, action_dim]
        out:
            phi(x): Tensor [B, phi_dim]
        """
        if self.use_state_and_action:
            phi = self.phi_net(torch.cat([s,a], dim=-1))
        else:
            phi = self.phi_net(s)
        return phi
    def w(self, normalize=True):
        """
        out:
            weight vector: Tensor shape of [phi_dim, 1]
        """
        if normalize:
            return F.normalize(self.w_net.layer.weight, dim=0).T
        else:
            return self.w_net.weight.T

class RewardV2Ensemble(nn.Module):
    """
    Ensemble of Reward V2
    """
    def __init__(self, grid, num_ensembles = 4, reward_args = None):
        super(RewardV2Ensemble, self).__init__()
        reward_args = reward_args or {
            'grid': grid,
            'phi_dim': 4,
            'ac_dim': 1,
            'use_state_and_action': False
        }
        self.ensembles = nn.ModuleList([RewardV2(**reward_args) for _ in range(num_ensembles)])
        self.phi_dim = self.ensembles[0].phi_dim * num_ensembles
    def R(self, x):
        """
        inp: 
            x: Tensor [B, T, feature_dim]
        out:
            R: Tensor [B]
        """
        return sum([reward.R(x) for reward in self.ensembles]) / len(self.ensembles)
    def prepare_update(self, learning_rate, weight_decay=1e-5):
        update_fns = [reward.prepare_update(learning_rate, weight_decay) for reward in self.ensembles]
        def update_fn(e_data):
            """
            y = 0 if x1 > x2 (x1 is better trajectory) 1 otherwise.
            """
            
            losses = []
            for update_fn, batch in zip(update_fns, e_data):#####WARNING 有一點問題 e_data餵進去會怪怪
                loss = update_fn(batch)
                losses.append(loss)
            return sum(losses) / len(losses)
        return update_fn
    def forward(self, s, a):
        """
        inp: 
            s: Tensor [B, state_dim]
            a: Tensor [B, action_dim]
        out:
            R: Tensor [B]
        """
        return sum([reward(s, a) for reward in self.ensembles]) / len(self.ensembles)
    def phi(self, s, a):
        """
        inp:
            s: Tensor [B, state_dim]
            a: Tensor [B, action_dim]
        out:
            phi(x): Tensor shape of [B, phi_dim]
        """
        return torch.cat([reward.phi(s, a) for reward in self.ensembles], dim=-1)

    def w(self):
        """
        out:
            weight vector: Tensor shape of [phi_dim, 1]
        """
        return torch.cat([reward.w() for reward in self.ensembles], dim=0)

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
        use_state_and_action=False,  # Use state-only or state-action pairs
        ob_dim=None,
        ac_dim=None,
    ):
        self.max_in_memory_num = max_in_memory_num
        self.max_per_file_num = max_per_file_num

        self.save_dir = tempfile.mkdtemp() if save_dir is None else save_dir
        if save_dir is None:
            atexit.register(lambda: shutil.rmtree(self.save_dir))

        self.trajs = []  # Trajectories in memory
        self.traj_files = [os.path.join(self.save_dir, f) for f in os.listdir(self.save_dir) if f.startswith('trajs')]
        self.use_state_and_action = use_state_and_action

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim

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
          if self.use_state_and_action:
              state_a = torch.cat((traj_a['states'][:-1], traj_a['actions']), dim=-1)
              state_b = torch.cat((traj_b['states'][:-1], traj_b['actions']), dim=-1)
          else:
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

class SORS():
    def __init__(self, args, placedb_raw, E):
        super(SORS, self).__init__()
        # TODO: create NN network
        # TODO: create optimizer
        self.resnet = torchvision.models.resnet18(pretrained = True).to(device1)
        self.cnn_coarse = CNNCoarse(self.resnet).to(device1)
        self.actor_net = Actor(cnncoarse=self.cnn_coarse, grid=E.grid).float().to(device1)
        self.initial_actor_state = self.actor_net.state_dict()
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.lr)
        self.R = RewardV2(grid = args.grid).to(device1)
        self.r_update = self.R.prepare_update(args.learning_rate_r)
        self.E = E
        self.args = args
        self.batch_size = args.batch_size
        self.current_episode = []
        self.replay_buffer = []
        self.dataset = PreferenceDataset(max_in_memory_num=10, max_per_file_num=5, ob_dim= 1+2*224*224, ac_dim = 1)
        self.counter = 0
        self.clip_param = args.clip_param
        self.max_grad_norm = args.max_grad_norm
        self.epoch = args.epoch
        self.epoch_r = args.epoch_r
        self.r_update_period = args.r_update_period
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
            
    def reset_model(self):
        self.actor_net.load_state_dict(self.initial_actor_state)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device1).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor_net(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()
    
    def store_transition(self, transition_mem): # if full, flush
        ## flush (pick first episode to remove and add new episode to the end)
        if self.counter > self.buffer_capacity :
            self.replay_buffer = self.replay_buffer[len(transition_mem):]
        self.replay_buffer.extend(transition_mem)
        self.counter += len(transition_mem)
    
    def store_transition_no_flush(self, temp_mem, transition_mem): # no flush version
        temp_mem.extend(transition_mem)

    def update(self):
        # Extract buffer data
        states = torch.tensor(np.array([t.state for t in self.replay_buffer]), dtype=torch.float)
        actions = torch.tensor(np.array([t.action for t in self.replay_buffer]), dtype=torch.float).view(-1, 1).to(device1)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.replay_buffer]), dtype=torch.float).view(-1, 1).to(device1)

        #reward = self.R(states,actions) # a list with shaped reward
        #target_v_all = reward.view(-1, 1).to(device1)

        for _ in range(self.epoch):
            for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True),disable = False):
                action_probs = self.actor_net(states[index].to(device1))
                reward = self.R(states[index].to(device1),actions[index]) # a list with shaped reward
                dist = Categorical(action_probs)
                action_log_prob = dist.log_prob(actions[index].squeeze()) #squeeze: two-d -> one-d
                ratio = torch.exp(action_log_prob - old_action_log_prob[index].squeeze())
                #target_v = target_v_all[index]
                target_v = reward.view(-1, 1).to(device1)
                advantage = target_v.detach()

                # Actor optimization
                L1 = ratio * advantage.squeeze()
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage
                action_loss = -torch.min(L1, L2).mean() # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

    def update_r(self):
        pair_dataset = PairDataset(self.dataset)
        dataloader = DataLoader(pair_dataset, batch_size=2, collate_fn=self.dataset.collate_fn, shuffle = True)
        for i in tqdm(range(self.epoch_r),desc=f'Epoch: '):
            batchs = LargeBatchProcessor(dataloader, batch_size=3)
            for batch in batchs:
                self.r_update(batch)
    
    def print_dense_reward(self, traj):
        states = torch.tensor(np.array([t.state for t in traj]), dtype=torch.float)
        actions = torch.tensor(np.array([t.action for t in traj]), dtype = torch.float)
        for i in range(len(states)):
            reward = self.R(states[i].unsqueeze(0).to(device1),actions[i]).detach()
            logger_temp.info(f'state{i}: {reward}')
        logger_temp.info(f' ')
        
    
    def learn(self, params, placedb_raw, placedb_dreamplace, NonLinearPlace):
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
            trajectory_data = {}
            for i in range(self.placed_macros):
                # If I didn't set random seed, the result will be all same without knowing why
                torch.seed()
                np.random.seed(None)
                # ============================================================================
                state_tmp = state.copy()
                action, action_log_prob = self.select_action(state)
                n_state, _, done = self.E.step(action)
                trans = Transition(state_tmp, action, action_log_prob, n_state, None)
                trans_temp.append(trans)
                ## store trajectory data
                if i == 0:
                    trajectory_data['states'] = torch.tensor(state_tmp).reshape(1, -1)
                    trajectory_data['actions'] = torch.tensor(action).reshape(1, -1)
                else:
                    trajectory_data['states'] = torch.cat((trajectory_data['states'],torch.tensor(state_tmp).reshape(1, -1)), dim=0)
                    trajectory_data['actions'] = torch.cat((trajectory_data['actions'],torch.tensor(action).reshape(1, -1)), dim=0)
                state = n_state
            """
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
            ## write in placedb_dreamplace
            for n in self.E.node_pos:
                if n == "V":
                    continue
                x, y, _, _ = self.E.node_pos[n]
                x = x * self.E.scale_x
                y = y * self.E.scale_y
                id = placedb_dreamplace.node_name2id_map[n]
                placedb_dreamplace.node_x[id] = x
                placedb_dreamplace.node_y[id] = y
            """
            ## DREAMPlace
            """
            best_temp = 9999999999999
            while True:
                np.random.seed(params.random_seed)
                placer = NonLinearPlace.NonLinearPlace(params, placedb_dreamplace, None)
                metrics = placer(params, placedb_dreamplace)
                wl = float(metrics[-1].hpwl.data)
                if wl >= best_temp:
                    wl = best_temp
                    break
                else:
                    best_temp = wl
            """
            
            """
            np.random.seed(params.random_seed)
            placer = NonLinearPlace.NonLinearPlace(params, placedb_dreamplace, None)
            metrics = placer(params, placedb_dreamplace)
            wl = float(metrics[-1].hpwl.data)
            trajectory_data['score'] = torch.tensor(-wl) # score equals to minus hpwl
            """
            wl = cal_hpwl(placedb_raw, self.E.node_pos, self.E.ratio)
            trajectory_data['score'] = torch.tensor(-wl) # score equals to minus hpwl
            logger_temp.info(f'wl: {wl}')
            ## Record renewed (If better than history)
            if wl < best_hpwl:
                best_hpwl = wl
                logger_reward.info(f'Best record: {best_hpwl}')
            ## Reward shaping
            self.store_transition(trans_temp)
            ## print trajectory dense reward
            self.print_dense_reward(trans_temp)
            self.dataset.add_traj(trajectory_data)
            reward_iter.append(episode_reward)
            wl_iter.append(wl)
            
            ## Training the agent
            if (episode_iter) % self.update_period == 0:
                ## TODO: update agent
                train_iter += 1
                self.update()
            
            ## Training the reward
            if (episode_iter) % self.r_update_period == 0:
                ## TODO: update reward
                train_iter_r += 1
                self.update_r()
                self.reset_model()
            ## Plot
            plot_reward(self.args.reward_folder,self.args.design,reward_iter)
            plot_reward(self.args.wl_folder,self.args.design,wl_iter)
            plot_macro('macro_no_DREAMPlace.png', self.E.node_pos, self.args.grid)
            
    def reward_test(self, params, placedb_raw):
        # start RL process
        train_iter = 0
        train_iter_r = 0
        episode_iter = 0
        result_dir_temp = params.result_dir
        reward_iter = []
        temp_mem = []
        wl_iter = []
        best_hpwl = 99999999999
        while True:
            episode_reward = 0
            params.result_dir = result_dir_temp + str(episode_iter)
            episode_iter += 1
            state = self.E.reset()
            trans_temp = []
            trajectory_data = {}
            for i in range(self.placed_macros):
                # If I didn't set random seed, the result will be all same without knowing why
                torch.seed()
                np.random.seed(None)
                # ============================================================================
                state_tmp = state.copy()
                action, action_log_prob = self.select_action(state)
                n_state, _, done = self.E.step(action)
                trans = Transition(state_tmp, action, action_log_prob, n_state, None)
                trans_temp.append(trans)
                ## store trajectory data
                if i == 0:
                    trajectory_data['states'] = torch.tensor(state_tmp).reshape(1, -1)
                    trajectory_data['actions'] = torch.tensor(action).reshape(1, -1)
                else:
                    trajectory_data['states'] = torch.cat((trajectory_data['states'],torch.tensor(state_tmp).reshape(1, -1)), dim=0)
                    trajectory_data['actions'] = torch.cat((trajectory_data['actions'],torch.tensor(action).reshape(1, -1)), dim=0)
                state = n_state
           
            
            wl = cal_hpwl(placedb_raw, self.E.node_pos, self.E.ratio)
            trajectory_data['score'] = torch.tensor(-wl) # score equals to minus hpwl
            logger_temp.info(f'wl: {wl}')
            ## Record renewed (If better than history)
            if wl < best_hpwl:
                best_hpwl = wl
                logger_reward.info(f'Best record: {best_hpwl}')
            ## Reward shaping
            self.store_transition_no_flush(temp_mem, trans_temp)
            ## print trajectory dense reward
            self.print_dense_reward(trans_temp)
            self.dataset.add_traj(trajectory_data)
            #reward_iter.append(episode_reward)
            wl_iter.append(wl)
            
            
            ## Training the reward
            if (episode_iter) % self.r_update_period == 0:
                ## TODO: update reward
                train_iter_r += 1
                self.update_r()
                break
        traj = []
        for i in range(len(temp_mem)):
            traj.append(temp_mem[i])
            if (i+1) % self.args.manual_placed_num == 0:
                states = torch.tensor(np.array([t.state for t in traj]), dtype=torch.float)
                actions = torch.tensor(np.array([t.action for t in traj]), dtype = torch.float)
                r = 0
                for i in range(len(states)):
                    reward = self.R(states[i].unsqueeze(0).to(device1),actions[i]).detach()
                    logger_temp2.info(f'state{i}: {reward}')
                    r += reward
                reward_iter.append(r)
                logger_temp2.info(f' ')
                
                traj = []
        plot_reward(self.args.reward_folder,self.args.design,reward_iter)
        plot_reward(self.args.wl_folder,self.args.design,wl_iter)
                

