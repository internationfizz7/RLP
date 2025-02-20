import sys
import numpy as np
import pickle
import logging
from util import *

sys.path.append("../install/dreamplace/")
sys.path.append("../install/")
import dreamplace.configure as configure
import Params
import PlaceDB
import NonLinearPlace

from placeDB import placeDB as placeDB
from feature import features, feature_extraction




## function of generating DREAMPlace placing data
def prototyping_generate():
    args = dotdict({
        #======== placedb raw ===========#
        'design_folder': "../benchmark/",
        'design': "adaptec1",
        #======== placedb dreamplace=====#
        'benchmark_path': "../benchmark/adaptec1_all_free/adaptec1.json",
        #======== data path =============#
        'data_path': "./data/adaptec1.pkl",
        'data_path_processed': "./data/adaptec1_processed.pkl"

    })


    input_path = args.benchmark_path
    output_path = './data/' + args.design

    print('Start generating placing data...')
    placedb_raw = placeDB(args)
    
    params = Params.Params()
    params.load(input_path)
    placedb_dreamplace = PlaceDB.PlaceDB()
    params.stop_overflow = 0.04
    params.plot_flag = 0
    placedb_dreamplace(params)
    result_dir = params.result_dir
    
    placedb_list = []
    try:
        while True:
            np.random.seed(None)
            placer = NonLinearPlace.NonLinearPlace(params, placedb_dreamplace, None)
            metrics = placer(params, placedb_dreamplace)
            wl = float(metrics[-1].hpwl.data)

            placedb = [] # [node_info, max_width, max_height, wirelength]

            ###TODO 從dreamplace擺置結果抓macro位置出來，放到placedb_raw內 (raw_x, raw_y)
            for n in placedb_raw.node_info:
                id = placedb_dreamplace.node_name2id_map[n]
                placedb_raw.node_info[n]['raw_x'] = placedb_dreamplace.node_x[id]
                placedb_raw.node_info[n]['raw_y'] = placedb_dreamplace.node_y[id]

            placedb.append(placedb_raw.node_info)
            placedb.append(placedb_raw.max_width)
            placedb.append(placedb_raw.max_height)
            placedb.append(wl)

            placedb_list.append(placedb)
    except KeyboardInterrupt:
        print('Store data...')
        ### Store to data set (not processed)
        data_path = args.data_path
        with open(data_path, "rb") as file:
            data = pickle.load(file)
        for placedb in placedb_list:
            data.append(placedb)
        with open(data_path, "wb") as file:
            pickle.dump(placedb, file)
        ### Store to processed data set
        data_path_processed = args.data_path_processed
        feature_tool = features(data_path_processed)
        processed_data = feature_tool.feature_data
        processed_data = feature_tool.append_processed_data(placedb_list, processed_data)
        with open(data_path_processed, "wb") as file:
            pickle.dump(data_path_processed, file)





    






