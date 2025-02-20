import numpy as np
#### This is 工人智慧  ########
# TODO: 根據長期觀察一筆測資的 -hpwl naive reward，得出一個reward tunning 方法

def reward_tunning(design, wl):
    if design == 'adaptec1':
        # ranging from 0.9 ~ 1.4 (1e8)
        reward = ((-wl) + 1.2e8) / 2.5e8 #(5e7)
    
    return reward


#### This is region map ######
# TODO: 對每一個元件就會有一張map，這個map反應出這個元件擺在哪個grid會有多少的
#       displacement，其位移基準點是從 region_trim_info 來的
#       若displacement橫跨整個die，給0分 (最少就是0分) 最多(完全沒有位移)給1分

def region_map(region_trim, placedb_raw, grid):
    region_map = {}
    max_width = placedb_raw.max_width
    max_height = placedb_raw.max_height
    scale_width = max_width / grid
    scale_height = max_height / grid
    for node_name, values in region_trim.items():
        x_to_right, x_c, y_to_top, y_c = values  # 
        map = np.zeros((grid, grid))
        # x (width) coor
        for i in range(0, grid):
            if  x_c != 0:
                center = (i+0.5) * scale_width
                dis = abs(x_c - center)
                ratio = dis / max_width
                map[i,:] += ratio * ratio
        # y (height) coor
        for i in range(0, grid):
            if y_c != 0:
                center = (i+0.5) * scale_height
                dis = abs(y_c - center)
                ratio = dis / max_height
                map[:,i] += ratio * ratio
        # sqrt and minus
        map = np.sqrt(map)
        map = np.zeros((grid, grid)) - map

        region_map[node_name] = map
    return region_map

        
    