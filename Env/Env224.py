import sys
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import os
import time
import logging
from util import *

class PlaceingEnv():
    def __init__(self, grid, args, placedb_raw, region_map):
        """
        grid: the grid num
        placedb_raw: (not DREAMPlace) placedb_raw
        """
        self.grid = grid
        self.args = args
        self.placedb_raw = placedb_raw
        self.num_macro = (self.placedb_raw.node_info)
        self.scale_x = (self.placedb_raw.max_width) / self.grid
        self.scale_y = (self.placedb_raw.max_height) / self.grid
        self.ratio = self.placedb_raw.max_height / self.grid
        self.manual_placed_num = args.manual_placed_num
        self.region_map = region_map
    
    def reset(self):
        self.num_macro_placed = 0 ### num_macro_placed代表已擺置多少元件
        self.node_pos = {} ### node_pos 是一個dictionary，key為元件名稱，value為 (x, y, size_x, size_y)
        ## Placing canvas (is all zero in initialization)
        canvas = np.zeros((self.grid, self.grid)) ### canvas 1 代表有擺東西
        ## Placing mask
        next_x = math.ceil(max(1, self.placedb_raw.node_info[self.placedb_raw.node_id_to_name[self.num_macro_placed]]['x']/self.scale_x))
        next_y = math.ceil(max(1, self.placedb_raw.node_info[self.placedb_raw.node_id_to_name[self.num_macro_placed]]['y']/self.scale_y))
        mask = self.get_mask(next_x, next_y) ### mask 0 代表invalid
        ## State
        self.state = np.concatenate((np.array([self.num_macro_placed]), canvas.flatten(),
                                     mask.flatten()), axis = 0)
        return self.state
    
    def step(self, action):
        ### TODO: Check if this action is available using mask in self.state
        canvas = self.state[1: 1+self.grid*self.grid].reshape(self.grid, self.grid)
        mask   = self.state[1+self.grid*self.grid: 1+self.grid*self.grid*2].reshape(self.grid, self.grid)
        reward = 0
        x = round(action // self.grid)
        y = round(action %  self.grid)
        if mask[x][y] == 0: ### mask 0 代表invalid
            reward += -200000
        node_name = self.placedb_raw.node_id_to_name[self.num_macro_placed]
        if self.region_map != None:
            reward = self.region_map[node_name][x][y]
        else:
            reward = 0
        ### TODO: change canvas
        size_x = math.ceil(max(1, self.placedb_raw.node_info[node_name]['x']/self.scale_x))
        size_y = math.ceil(max(1, self.placedb_raw.node_info[node_name]['y']/self.scale_y))
        canvas[x : x+size_x, y : y+size_y] = 1.0 ### canvas 1 代表有擺東西
        ### TODO: Place the macro
        self.node_pos[node_name] = (x, y, size_x, size_y) ### node_pos 是一個dictionary，key為元件名稱，value為 (x, y, size_x, size_y)
        
        self.num_macro_placed += 1

        ### TODO: DONE
        if self.num_macro_placed == self.num_macro or \
            (self.manual_placed_num is not None and self.num_macro_placed == self.manual_placed_num): 
            done = True
        else:
            done = False

        ### TODO: get mask
        if not done:
            next_x = math.ceil(max(1, self.placedb_raw.node_info[self.placedb_raw.node_id_to_name[self.num_macro_placed]]['x']/self.scale_x))
            next_y = math.ceil(max(1, self.placedb_raw.node_info[self.placedb_raw.node_id_to_name[self.num_macro_placed]]['y']/self.scale_y))
            mask = self.get_mask(next_x, next_y)
        else:
            mask = np.zeros((self.grid, self.grid))
        self.state = np.concatenate((np.array([self.num_macro_placed]), canvas.flatten(),
                                     mask.flatten()), axis = 0)
        return self.state, reward, done

    def get_mask(self, next_x, next_y):
        mask = np.ones((self.grid, self.grid))
        for node_name in self.node_pos:
            startx = max(0, self.node_pos[node_name][0] - next_x + 1)
            starty = max(0, self.node_pos[node_name][1] - next_y + 1)
            endx = min(self.node_pos[node_name][0] + self.node_pos[node_name][2] - 1, self.grid - 1)
            endy = min(self.node_pos[node_name][1] + self.node_pos[node_name][3] - 1, self.grid - 1)
            mask[startx: endx + 1, starty : endy + 1] = 0
        mask[self.grid - next_x + 1:,:] = 0
        mask[:, self.grid - next_y + 1:] = 0
        return mask
        



    
    
    