import ast
import torch
import json

def read_loss(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data


# 打开两个文件并逐行读取
def compute_loss_differences(file1, file2):
    # 使用生成器逐行读取损失
    losses1 = read_loss(file1)
    losses2 = read_loss(file2)

    differences = []
    for loss1, loss2 in zip(losses1, losses2):
        diff = abs(loss1['train_loss'] - loss2['train_loss'])
        differences.append(diff)
        print(f"Difference: {diff}")

    return differences

def compute_tensor_differences(file1, file2):
    tensor_dict1 = torch.load(file1, map_location='cpu')
    tensor_dict2 = torch.load(file2, map_location='cpu')

    differences = {}
    for key in tensor_dict1:
        if key in tensor_dict2:
            # 计算差值
            diff = tensor_dict1[key] - tensor_dict2[key]
            differences[key] = diff
        else:
            print(f"Key {key} not found in both dictionaries")

    for key, value in differences.items():
        print(f"Difference in {key}: {value.abs().mean()}")


compute_loss_differences('w_FA.log', 'wo_FA.log')

# compute_tensor_differences('sp_1.pt', 'sp_2.pt')