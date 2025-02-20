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
        next_x = self.placedb_raw.node_info[self.placedb_raw.node_id_to_name[self.num_macro_placed]]['x']/self.scale_x
        next_y = self.placedb_raw.node_info[self.placedb_raw.node_id_to_name[self.num_macro_placed]]['y']/self.scale_y
        mask = self.get_mask(canvas, next_x, next_y) ### mask 0 代表invalid
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
        #reward = self.region_map[node_name][x][y]
        ### TODO: change canvas
        size_x = self.placedb_raw.node_info[node_name]['x']/self.scale_x
        size_y = self.placedb_raw.node_info[node_name]['y']/self.scale_y
        canvas = self.get_canvas(canvas, x, y, size_x, size_y)
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
            next_x = self.placedb_raw.node_info[self.placedb_raw.node_id_to_name[self.num_macro_placed]]['x']/self.scale_x
            next_y = self.placedb_raw.node_info[self.placedb_raw.node_id_to_name[self.num_macro_placed]]['y']/self.scale_y
            mask = self.get_mask(canvas, next_x, next_y)
        else:
            mask = np.zeros((self.grid, self.grid))
        """
        print(f'canvas: {canvas}')
        print(f'mask: {mask}')
        """
        self.state = np.concatenate((np.array([self.num_macro_placed]), canvas.flatten(),
                                     mask.flatten()), axis = 0)
        return self.state, reward, done

    def get_mask(self, canvas, next_x, next_y):
        mask = np.ones((self.grid, self.grid))
        ceil_x = math.ceil(max(1, next_x))
        ceil_y = math.ceil(max(1, next_y))
        mask[self.grid - ceil_x + 1:,:] = 0
        mask[:, self.grid - ceil_y + 1:] = 0
        """
        mask = np.zeros((self.grid, self.grid))
        ceil_x = math.ceil(max(1, next_x))
        ceil_y = math.ceil(max(1, next_y))
        #  conv kernel (ceil_x, ceil_y)
        kernel = np.ones((ceil_x, ceil_y))
        kernel[ceil_x-1,:] *= (next_x % 1)
        kernel[:,ceil_y-1] *= (next_y % 1)
        for x_index in range(0, self.grid - ceil_x + 1):
            for y_index in range(0, self.grid - ceil_y + 1):
                #  Conv
                occupied_grid_count = ceil_x * ceil_y - (ceil_x - 1) * (ceil_y - 1)
                value = 0
                for x in range(ceil_x):
                    for y in range(ceil_y):
                        value += 1-kernel[x,y]-canvas[x_index+x,y_index+y]
                value /= occupied_grid_count
                mask[x_index,y_index] = max(0,value)
        """

        return mask
    
    def get_canvas(self, canvas, place_x, place_y, size_x, size_y):
        ceil_x = math.ceil(max(1, size_x))
        ceil_y = math.ceil(max(1, size_y))
        for x in range(ceil_x - 1):
            for y in range(ceil_y - 1):
                canvas[place_x+x,place_y+y] = 1
        edge_value_x = size_x%1
        edge_value_y = size_y%1
        corner_value = edge_value_x * edge_value_y
        corner_value_orig = canvas[place_x+ceil_x-1, place_y+ceil_y-1]
        canvas[place_x+ceil_x-1,:] = np.maximum(1, canvas[place_x+ceil_x-1,:]+edge_value_x)
        canvas[:,place_y+ceil_y-1] = np.maximum(1, canvas[:,place_y+ceil_y-1]+edge_value_y)
        canvas[place_x+ceil_x-1,place_y+ceil_y-1] = max(1, corner_value_orig+corner_value)
        return canvas


        