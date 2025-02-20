import logging
import coloredlogs
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')
sys.path.append("../install/dreamplace/")
sys.path.append("../install/")

from collections import namedtuple
import dreamplace.configure as configure
import Params
import PlaceDB
import NonLinearPlace

def DREAMPlace_setup(A, placedb_raw, args):
    ### generate another pl file (informing what macros needed to be placed by DREAMPlace and what is placed by RL) and then delete
    ### if the macro needed to be placed by DREAMPlace, in the pl file it SHOULDN'T be marked as /FIXED
    if args.design[0:3] == 'ada':
        folder_path = args.design_folder+'/'+args.design+'/'
        old_name = args.design + '.pl'
        old_path = os.path.join(folder_path, old_name)
        if os.path.isfile(old_path):
            new_name = f"temp_{old_name}"
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
        #=====================================================================================================
        filename = folder_path + 'temp_' + args.design + '.pl'
        content_pl = []
        with open(filename, "r") as file:
            for line in file:
                content_pl.append(line)
        ## put the first #manual_placed_macros into queue
        queue = []
        for id in range(A.placed_macros):
            node_name = placedb_raw.node_id_to_name[id]
            queue.append(node_name)

        filename = folder_path + args.design + '.pl'
        with open(filename, "w") as file:
            for line in content_pl:
                if (line.startswith("\n")):
                    file.write(line)
                    continue
                line_temp = line
                line = line.strip().split()
                if line[-1] != '/FIXED' or line[0] in queue:
                    file.write(line_temp)
                else:
                    re_line = " ".join(line[:-1])
                    file.write(re_line + '\n')

        path = args.design_folder+'/'+args.design+'/'+args.design+'.json'
        params = Params.Params()
        params.load(path)
        np.random.seed(params.random_seed)
        placedb_dreamplace = PlaceDB.PlaceDB()
        params.stop_overflow = 0.04
        params.plot_flag = 0
        placedb_dreamplace(params)

        ### delete the file and rerename
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print('Remove File Error!!!')
        
        old_name = 'temp_' + args.design + '.pl'
        old_path = os.path.join(folder_path, old_name)
        if os.path.isfile(old_path):
            new_name = args.design + '.pl'
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
    #elif args.design[0:3] == 'ibm':
        
    return params, placedb_dreamplace

def DREAMPlace_setup2(args):
    ### This setup unable the "all center" flag, and thus macros (and cells) are placed according to initial pos (if there's no manual
    ### initial sol, than use the pl pos as initialization)
    if args.design[0:3] == 'ada':
        folder_path = args.design_folder+'/'+args.design+'/'
        old_name = args.design + '.pl'
        old_path = os.path.join(folder_path, old_name)
        if os.path.isfile(old_path):
            new_name = f"temp_{old_name}"
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
        #=====================================================================================================
        filename = folder_path + 'temp_' + args.design + '.pl'
        content_pl = []
        with open(filename, "r") as file:
            for line in file:
                content_pl.append(line)

        filename = folder_path + args.design + '.pl'
        with open(filename, "w") as file:
            for line in content_pl:
                if (line.startswith("\n")):
                    file.write(line)
                    continue
                line_temp = line
                line = line.strip().split()
                if line[-1] != '/FIXED':
                    file.write(line_temp)
                else:
                    re_line = " ".join(line[:-1])
                    file.write(re_line + '\n')

        path = args.design_folder+'/'+args.design+'/'+args.design+'.json'
        params = Params.Params()
        params.load(path)
        np.random.seed(params.random_seed)
        placedb_dreamplace = PlaceDB.PlaceDB()
        params.stop_overflow = 0.04
        params.plot_flag = 0
        params.random_center_init_flag = 0
        placedb_dreamplace(params)

        ### delete the file and rerename
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print('Remove File Error!!!')
        
        old_name = 'temp_' + args.design + '.pl'
        old_path = os.path.join(folder_path, old_name)
        if os.path.isfile(old_path):
            new_name = args.design + '.pl'
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
        
        return params, placedb_dreamplace
    elif args.design[0:3] == 'ibm': ## Actually did nothing
        path = args.design_folder+'/'+args.design+'/'+args.design+'_new'+'.json'
        params = Params.Params()
        params.load(path)
        np.random.seed(params.random_seed)
        placedb_dreamplace = PlaceDB.PlaceDB()
        params.stop_overflow = 0.04
        params.plot_flag = 0
        params.random_center_init_flag = 0
        placedb_dreamplace(params)
        
        return params, placedb_dreamplace



