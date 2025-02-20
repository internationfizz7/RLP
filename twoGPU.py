import torch
# 定义设备
device1 = torch.device("cuda:0")  # GPU 0
device2 = torch.device("cuda:1")  # GPU 1

# 示例函数 A 和 B
def function_A(data):
    # 假设进行某种计算
    return data * 2

def function_B(data):
    # 假设进行某种计算
    return data + 3

# 模拟数据
data_A = torch.tensor([1.0, 2.0, 3.0]).to(device1)  # 数据放在 GPU 0
data_B = torch.tensor([4.0, 5.0, 6.0]).to(device2)  # 数据放在 GPU 1

# 在指定 GPU 上运行函数
with torch.cuda.device(0):  # 明确指定 GPU 0
    result_A = function_A(data_A)

with torch.cuda.device(1):  # 明确指定 GPU 1
    result_B = function_B(data_B)

print(f"Result A (GPU 0): {result_A}")
print(f"Result B (GPU 1): {result_B}")