import logging
import coloredlogs
import sys
#import gym
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
import random
#import matplotlib.pyplot as plt
from collections import Counter
from feature import features, feature_extraction

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

import numpy as np
import matplotlib.pyplot as plt

# 生成 x 軸數據
x = np.linspace(0, 10, 50)

# 生成第一條趨勢線
y1 =np.sin(x) + np.random.normal(0, 0.1, len(x))  # 加上小的隨機擾動

# 生成第二條趨勢相近的線
y2 = 13+(y1/2.5)+np.random.normal(0, 0.08, len(x))  # 在 y1 的基礎上增加更小的擾動

# 畫圖
plt.figure(figsize=(8, 5))
plt.plot(x, y1, label="Line 1", color="blue")
plt.plot(x, y2, label="Line 2", color="red", linestyle="dashed")

# 添加標籤和圖例
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("GAT_Transformer")
plt.legend()
plt.grid(True)

# 顯示圖表
plt.savefig('./plot_.png', format='png')
plt.cla()

sys.exit()
def is_convertible(s):
    if 'e' in s.lower():
        return False
    else:
        return True
   

filename = '../benchmark/ibm01/ibm01.nets'
a = []
with open(filename, "r") as file:
    for line in file:
        a.append(line)
filename = '../benchmark/ibm01/ibm01_new.nets'
with open(filename, "w") as file:
    for line in a :
        if (line.startswith("\n")):
            file.write(line)
            continue
        line_temp = line
        line = line.strip().split()
        if line[0][0] == 'p':
            re_line = line_temp[:-1]
            re_line += ' : 0 0\n'
            file.write(re_line)
        elif line[0][0] == 'a':
            if is_convertible(line[3]) == False:
                line[3] = '0'
            elif is_convertible(line[4]) == False:
                line[4] = '0'
            re_line = " ".join(line[:])
            re_line = " "+ re_line + '\n'
            file.write(re_line)
        else:
            file.write(line_temp)

