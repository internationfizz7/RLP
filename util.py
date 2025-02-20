## containing HPWL calculator, ploter ...
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def plt_re_iter(folder, filename, reward_list):
    plt.plot(reward_list)
    name = folder + filename
    name = name + '.png'
    plt.savefig(name, format = 'png')
    plt.cla()


def cal_hpwl(placedb, node_pos, ratio):
    hpwl = 0.0
    for net_name in placedb.net_info:
        max_x = 0.0
        min_x = placedb.max_height * 1.1
        max_y = 0.0
        min_y = placedb.max_height * 1.1
        for node_name in placedb.net_info[net_name]["nodes"]:
            if node_name not in node_pos:
                continue
            h = placedb.node_info[node_name]['x']
            w = placedb.node_info[node_name]['y']
            pin_x = node_pos[node_name][0] * ratio + h / 2.0 + placedb.net_info[net_name]["nodes"][node_name]["x_offset"]  #最左下+一半長寬+offset從中間算
            pin_y = node_pos[node_name][1] * ratio + w / 2.0 + placedb.net_info[net_name]["nodes"][node_name]["y_offset"]
            max_x = max(pin_x, max_x)
            min_x = min(pin_x, min_x)
            max_y = max(pin_y, max_y)
            min_y = min(pin_y, min_y)
        for port_name in placedb.net_info[net_name]["ports"]:
            h = placedb.port_info[port_name]['x']
            w = placedb.port_info[port_name]['y']
            pin_x = h
            pin_y = w
            max_x = max(pin_x, max_x)
            min_x = min(pin_x, min_x)
            max_y = max(pin_y, max_y)
            min_y = min(pin_y, min_y)
        if min_x <= placedb.max_height:
            hpwl_tmp = (max_x - min_x) + (max_y - min_y)
        else:
            hpwl_tmp = 0
        if "weight" in placedb.net_info[net_name]:
            hpwl_tmp *= placedb.net_info[net_name]["weight"]
        hpwl += hpwl_tmp
        
    return hpwl


def setup_logger(name, log_file, level=logging.INFO):

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file,mode='w')        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def plot_macro(file_path, node_pos, grid):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    for node_name in node_pos:
        x, y, size_x, size_y = node_pos[node_name]
        ax1.add_patch(
            patches.Rectangle(
                (x/grid, y/grid),   # (x,y)
                size_x/grid,          # width
                size_y/grid, linewidth=1, edgecolor='k',
            )
        )
        ax1.text(
            x / grid + size_x / (2 * grid),  # 中心 x 座標
            y / grid + size_y / (2 * grid),  # 中心 y 座標加上適當偏移
            node_name[-4:],                      # macro 名稱
            fontsize=6,                     # 字體大小
            ha='center',                    # 水平對齊
            va='bottom',                    # 垂直對齊
            color='blue'                    # 字體顏色
        )
    fig1.savefig(file_path, dpi=90, bbox_inches='tight')
    plt.close()

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file,mode='w')        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


