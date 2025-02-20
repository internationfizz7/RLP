import numpy as np
import sys
import math
from collections import namedtuple
from collections import Counter
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
### feature.py ###
### include two classes
### 1. construct a feature information from 
###    an existing placement result. 
### 2. common feature extraction using several 
###    placement result features.  

#feature = namedtuple('TCG', 'centerInfo', 'regionInfo', 'neighborInfo', 'Gv', 'Gh', 'Gd')

def feature_extraction(node_info, max_width, max_height, neighborInfo, dia = 500): ### dia is the diameter for neighbors detection (default 100)
    node_num = len(node_info)
    node_info = node_info

    ### TCG ###
    TCG = None

    ### center info ###
    centerInfo = {}
    center_x, center_y = max_width / 2, max_height / 2

    for n in node_info:
        x_l, y_l, size_x, size_y = node_info[n]['raw_x'], node_info[n]['raw_y'], node_info[n]['x'], node_info[n]['y']
        x_c = x_l + 0.5 * size_x
        y_c = y_l + 0.5 * size_y
        dis = math.sqrt(math.pow(center_x - x_c,2) + math.pow(center_y - y_c,2))
        centerInfo[n] = dis

    ### region info ###
    regionInfo = {}
    max_w, max_h = max_width, max_height

    for n in node_info:
        x_l, y_l, size_x, size_y = node_info[n]['raw_x'], node_info[n]['raw_y'], node_info[n]['x'], node_info[n]['y']
        x_c = x_l + 0.5 * size_x
        y_c = y_l + 0.5 * size_y
        x_to_right = max_w - x_c
        y_to_top   = max_h - y_c

        region_node = [x_to_right, x_c, y_to_top, y_c]
        regionInfo[n] = region_node

    ### neighbor info ###
    #Gv = np.zeros((node_num, node_num)) 
    #Gh = np.zeros((node_num, node_num))
    #Gd = np.zeros((node_num, node_num))

    for n in node_info:
        x_l, y_l, size_x, size_y = node_info[n]['raw_x'], node_info[n]['raw_y'], node_info[n]['x'], node_info[n]['y']
        x_c = x_l + 0.5 * size_x
        y_c = y_l + 0.5 * size_y
        x_r = x_l + size_x
        y_r = y_l + size_y
        for n_ in node_info:
            if n==n_:
                continue
            x_l_, y_l_, size_x_, size_y_ = node_info[n_]['raw_x'], node_info[n_]['raw_y'], node_info[n_]['x'], node_info[n_]['y']
            x_c_ = x_l_ + 0.5 * size_x_
            y_c_ = y_l_ + 0.5 * size_y_
            x_r_ = x_l_ + size_x_
            y_r_ = y_l_ + size_y_
            
            # Ignore the overlap condition
            dis = math.sqrt(math.pow(abs(x_c - x_c_),2) + math.pow(abs(y_c - y_c_),2))
            if dis <= dia:
                neighborInfo[node_info[n]['id']][node_info[n_]['id']] += 1
                ##TODO TCG Gv
                ##TODO TCG Gh
                ##TODO TCG Gd
    
    return [TCG, centerInfo, regionInfo]

# When you have a set of new data:
## 1. Turn the new data into a pack and store it in {benchmark}.pkl

## 2. Process new data

## 3. Add new processed_data into {benchmark}_processed.pkl

# When you want to trim_data (of a set of processed data)
### (In everywhere) ###
## 1. feature_tool = features(data_path) data_path is in form _processed
## 2. TCG_trim, center_trim, region_trim, neighbor_trim = feature_tool.trim_data() 
    

class features():   ## To do execution on PROCESSED data
    def __init__(self, path):
        self.feature_data = self.load_data(path)

    def load_data(self, path):
        feature_data = []
        if path == None:
            return feature_data
        else:
            with open(path, "rb") as file:
                feature_data = pickle.load(file)
            return feature_data

    ### Process a set of new data and store them into original processed data set
    def append_processed_data(self, placedb_list, processed_data):
        node_num = len(placedb_list[0][0])
        if processed_data:
            neighborInfo = processed_data[-1]
            processed_data = processed_data[:-1]
        else:
            neighborInfo = np.zeros((node_num, node_num))
        for placedb in placedb_list:
            node_info = placedb[0]
            max_width = placedb[1]
            max_height = placedb[2]
            feature = feature_extraction(node_info, max_width, max_height, neighborInfo)
            processed_data.append([placedb, feature])
        processed_data.append(neighborInfo)
        self.feature_data = processed_data
        return processed_data
    
    def trim_data(self, frequency = 0.1):
        ### this function uses statistics to get common information
        #TCGs = np.array(f.TCG for f in self.feature_data[:-1])
        keys = list(self.feature_data[0][1][1].keys())
        centerInfos = np.array([list(f[1][1].values()) for f in self.feature_data[:-1]])
        regionInfos = {k: np.array([d[1][2][k] for d in self.feature_data[:-1]]) for k in keys}
        neighborInfos = np.array(self.feature_data[-1])
        #Ghs = np.array(f.Gh for f in self.feature_data)
        #Gvs = np.array(f.Gv for f in self.feature_data)
        #Gds = np.array(f.Gd for f in self.feature_data)
        
        ### TCGs 
        TCG_trim = None

        ### centerInfos (dict)
        mean_center = np.mean(centerInfos, axis = 0)
        std_center  = np.std(centerInfos, axis = 0)
        CV_center   = std_center / mean_center
        center_trim = {keys[i]: mean_center[i] if CV_center[i] < 0.3 else 0 for i in range(len(mean_center))}
        ### regionInfos

        region_trim = {}
        for k in keys:
            mean_region = np.mean(regionInfos[k], axis=0)
            std_region = np.std(regionInfos[k], axis=0)
            CV_region = std_region / mean_region
            region_trim[k] = [mean_region[i] if CV_region[i] < 0.5 else 0 for i in range(len(mean_region))]
        
        ### neighborInfos
        threshold = frequency * len(centerInfos) ### len(centerInfos) is the num of data
        neighbor_trim = np.where(neighborInfos > threshold, neighborInfos, 0)

        return TCG_trim, center_trim, region_trim, neighbor_trim
        
        

