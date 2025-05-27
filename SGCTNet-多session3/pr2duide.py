import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #随机梯度下降（SGD）、Adam、Adagrad、RMSprop
import torch_geometric
from torch_scatter import scatter_add
import numpy as np
import torch.optim as optim
import os, random
import RGNN_with_trans
import pandas as pd
from scipy.spatial import distance_matrix
import scipy.io


data_dir = "D:/dataset/SEED-IV/eeg_feature_smooth/"
root_dir = "D:/dataset/SEED-IV/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
y_true = None
y_score = None
def dependent_loaders(batch_size, K):#提取第K个受试者的数据//实验没有考虑时间序列
    seesion1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
    seesion2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
    seesion3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    labels = [seesion1_label, seesion2_label, seesion3_label]

    trials_seesion1 = [[3,5,6,8,20,22],[0,7,9,11,12,13],[1,4,10,14,16,17],[2,15,18,19,21,23]]#0,1,2,3标签位置
    trials_seesion2 = [[3,4,6,13,18,20],[1,14,15,17,21,23],[0,5,7,10,12,16],[2,8,9,11,19,22]]
    trials_seesion3 = [[11,15,18,19,21,23],[0,3,7,8,10,22],[1,2,9,12,16,20],[4,5,6,13,14,17]]
    trials = [trials_seesion1, trials_seesion2, trials_seesion3]
    eeg_dict = {"train":[], "test":[]}
    eeg = []
    print("--- Subject dependent loader ---")
    for session in range(3,4):
        #print("we are in session:{:d}".format(session))
        session_labels = labels[session-1]
        trial = trials[session-1]
        session_folder = "D:/dataset/SEED-IV/eeg_feature_smooth/"+str(session)
        file = ''
        for File in os.listdir(session_folder):#提取第K个受试者的数据
            p=int(File.split("_")[0])
            if p == K:
                file = File
                break
        
        file_path = "{}/{}".format(session_folder, file)

        num = []
        if session==3:
            for i in range(4):#随机取某个同类实验
                num.append(trial[i][4])
                num.append(trial[i][5])
        else :pass
        print(num)
        
        for videoclip in range(24):
            X_psd = scipy.io.loadmat(file_path)['psd_LDS{}'.format(videoclip + 1)]
            X_de = scipy.io.loadmat(file_path)['de_LDS{}'.format(videoclip + 1)]

            y = session_labels[videoclip]
            y = torch.tensor([y]).long()
            edge_index = torch.tensor(np.array([np.arange(62*62)//62, np.arange(62*62)%62]))
            for t in range(X_psd.shape[1]):#每一列？
                x_psd = torch.tensor(X_psd[:, t, :]).float()
                x_psd = (x_psd-x_psd.mean())/x_psd.std()#这里就进行了标准化
                x_de = torch.tensor(X_de[:, t, :]).float()
                x_de = (x_de-x_de.mean())/x_de.std()
                '''
                if videoclip >= 16:#24个实验，前16个为训练集，后8个为测试集
                    eeg_dict["test"].append((torch_geometric.data.Data(x=x_psd, y=y, edge_index=edge_index),torch_geometric.data.Data(x=x_de, y=y, edge_index=edge_index)))
                else :
                    eeg_dict["train"].append((torch_geometric.data.Data(x=x_psd, y=y, edge_index=edge_index),torch_geometric.data.Data(x=x_de, y=y, edge_index=edge_index)))
                '''
                if session == 3:
                #保证测试集和验证集的分类的分布相同
                   if videoclip == num[0] or videoclip == num[1] or videoclip == num[2] or videoclip == num[3] or videoclip == num[4] or videoclip == num[5] or videoclip == num[6] or videoclip == num[7]:#24个实验，四类情绪分别随机取2个实验，一个session测试集为8个
                       eeg_dict["test"].append((torch_geometric.data.Data(x=x_psd, y=y, edge_index=edge_index),torch_geometric.data.Data(x=x_de, y=y, edge_index=edge_index)))
                   else:
                       eeg_dict["train"].append((torch_geometric.data.Data(x=x_psd, y=y, edge_index=edge_index),torch_geometric.data.Data(x=x_de, y=y, edge_index=edge_index)))
                else:  eeg_dict["train"].append((torch_geometric.data.Data(x=x_psd, y=y, edge_index=edge_index),torch_geometric.data.Data(x=x_de, y=y, edge_index=edge_index)))
    loaders_dict = {"train":[], "test":[]}
    
    # loaders_dict["train"] = torch_geometric.loader.DataLoader(eeg_dict["train"], batch_size=batch_size, shuffle=True, drop_last=False)#drop_last：如果为True，最后一个不完整的批次将被丢弃
    loaders_dict["test"] = torch_geometric.loader.DataLoader(eeg_dict["test"], batch_size=batch_size, shuffle=False, drop_last=False)            
    
    return  loaders_dict["test"]

def get_adjacency_matrix():#在location中找到对应的Channel位置
    channel_order =pd.read_excel(root_dir+"Channel Order.xlsx", header=None)
    channel_location = pd.read_csv(root_dir+"channel_locations.txt", sep="\t",header=None)
    channel_location.columns = ["Channel", "X", "Y", "Z"]
    channel_location["Channel"] = channel_location["Channel"].apply(lambda x: x.strip().upper())
    filtered_df = pd.DataFrame(columns=["Channel", "X", "Y", "Z"])
    for channel in channel_location["Channel"]:
        for used in channel_order[0]:
            if channel == used:
                filtered_df = pd.concat([channel_location.loc[channel_location['Channel']==channel],filtered_df], ignore_index=True)
                # Concatenate each row from the "channel_location" dataframe whose channel name is present in the Excel sheet

    filtered_df = filtered_df.reindex(index=filtered_df.index[::-1]).reset_index(drop=True)#将DataFrame对象filtered_df的索引进行倒序排列，并通过设置drop=True参数来删除原来的索引列。最后将结果赋值给filtered_df变量。
    filtered_matrix = np.asarray(filtered_df.values[:, 1:4], dtype=float)#用于将输入的数据转换为 NumPy 数组

    distances_matrix = distance_matrix(filtered_matrix, filtered_matrix)## Compute a matrix of distances for each combination of channels in the filtered_matrix dataframe
    
    delta = 10
    adjacency_matrix = np.minimum(np.ones([62,62]), delta/(distances_matrix**2)) # Computes the adjacency matrix cells as min(1, delta/d_ij), N.B. zero division error can arise here for the diagonal cells, "1" value will be chosen automatically instead

    return torch.tensor(adjacency_matrix).float()

def get_model(subj_num=1,session_num=1):
    edge_weight = get_adjacency_matrix()
    model = RGNN_with_trans.FuseModel(62, True, edge_weight.to(device), 5, 16, 4).to(device)
    
    model_weight_path = f"./Pro_log/SGCTNet_{subj_num}.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    return model

def acc_subj(num_subj=1,session_num=3,batch_size = 16):
    global y_score
    global y_true

    net = get_model(subj_num=num_subj, session_num=session_num)

    tes_loader = dependent_loaders(batch_size, num_subj)
    print(len(tes_loader))
    tes_acc = 0
    with torch.no_grad():
        for batch_test in tes_loader:
            outputs, _ = net(batch_test)
            y_list = batch_test[1].y.to('cpu')

            if y_score is None:
                y_score = F.softmax(outputs, dim=1)
                y_true = y_list
            else:
                y_score = torch.cat((y_score, F.softmax(outputs, dim=1)), dim=0)
                y_true = torch.cat((y_true, y_list), dim=0)

            predict_y = torch.max(outputs, dim=1)[1]
            tes_acc += (predict_y == batch_test[1].y).sum().item() #/ batch_test[1].y.size(0)

        tes_acc = tes_acc / len(tes_loader.dataset)
        print(tes_acc)
    return tes_acc


# --------------------------------------------得到结果
best_Acc = []
for i in range(1, 16):
    print(f'------------------start subect:{i}--------------------- ')
    best_model_acc = acc_subj(num_subj=i,session_num=3,batch_size = 16)

    best_Acc.append(best_model_acc)

k = 0
result_string = ''
for i in best_Acc:
    k=k+1
    str1 = f'subject{k} acc:{i} \n'
    result_string = ''.join([result_string, str1])

# 所有subjects的平均准确率、标准差
mean_acc = np.mean(best_Acc)
sta = np.std(best_Acc, ddof=1)
mean_str = f'mean_acc:{mean_acc} sta:{sta}'

result_string = ''.join([result_string, mean_str]) 

print(result_string)

print(y_score.shape, y_true.shape)


y_score = y_score.cpu().numpy()
y_true = y_true.cpu().numpy()

# from sklearn.preprocessing import MinMaxScaler
# y_score = MinMaxScaler().fit_transform(y_score) # 归一化【0,1】
print(np.bincount(y_true))

for i in range(100):
    print(y_score[i], y_true[i])

#----------------------------------------画PR曲线

print(len(y_true))

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_recall_curve, auc, average_precision_score
# from sklearn.preprocessing import label_binarize
# from scipy.interpolate import interp1d

# # 假设 y_true 和 y_score 已经定义
# # y_true: 原始标签，形状 (N,)；取值范围是 [0, n_classes - 1]
# # y_score: 预测的概率分数，形状为 (N, n_classes)

# plt.rcParams['font.family'] = 'Times New Roman'

# n_classes = 4
# class_labels = ['NE', 'SA', 'FE', 'HA']
# colors = ['#ffb347', '#aec6cf', '#77dd77', '#cdb4db']

# # 将真实标签二值化
# y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

# # 1. 绘制每个类别的 PR 曲线并存储 (precision, recall)
# plt.figure(figsize=(7, 6))
# precision_list, recall_list = [], []
# aucs = []

# for i, color in enumerate(colors):
#     precision_i, recall_i, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
#     auc_i = average_precision_score(y_true_bin[:, i], y_score[:, i])
#     aucs.append(auc_i)
    
#     # 将各类别曲线设为“虚线”（--）
#     plt.plot(
#         recall_i,
#         precision_i,
#         color=color,
#         linestyle='--',   # <--- 改成虚线
#         lw=2,
#         label=f'{class_labels[i]} (area = {auc_i:.4f})'
#     )
    
#     precision_list.append(precision_i)
#     recall_list.append(recall_i)



# # 2. 计算并绘制 Macro-average PR 曲线（实线）
# all_recall = np.linspace(0, 1, 100)

# precision1, recall2, _ = precision_recall_curve(y_true_bin.ravel(),y_score.ravel())
# average_precision = average_precision_score(y_true_bin, y_score, average="macro")

# plt.plot(
#     recall2,
#     precision1,
#     color='#8B0000',
#     linestyle='-',  # <--- 改成实线
#     lw=2,
#     label=f'Macro-average (area = {average_precision:.4f})'
# )

# # # 3. 计算并绘制 Micro-average PR 曲线（实线）
# # precision3, recall4, _ = precision_recall_curve(y_true_bin.ravel(),y_score.ravel())
# # average_precision2 = average_precision_score(y_true_bin, y_score, average="micro")

# # plt.plot(
# #     recall4,
# #     precision3,
# #     color='#4b0082',
# #     linestyle='-',  # <--- 改成实线
# #     lw=2,
# #     label=f'Micro-average (area = {average_precision2:.4f})'
# # )


# print('Average precision score, macro-averaged over all classes: {0:0.4f}'.format(average_precision))

# # 4. 图形样式设置并显示
# plt.xlabel('Recall', fontsize=14)
# plt.ylabel('Precision', fontsize=14)
# plt.title('PR Curve - SGCTNet', fontsize=16)
# plt.legend(loc='best', fontsize=12)
# plt.grid(alpha=0.5)
# plt.tight_layout()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.preprocessing import label_binarize
from scipy.interpolate import interp1d

# 假设 y_true 和 y_score 已经定义
# y_true: 原始标签，形状 (N,)；取值范围是 [0, n_classes - 1]
# y_score: 预测的概率分数，形状为 (N, n_classes)

plt.rcParams['font.family'] = 'Times New Roman'

n_classes = 4
class_labels = ['NE', 'SA', 'FE', 'HA']
colors = ['#ffb347', '#aec6cf', '#77dd77', '#cdb4db']

# 将真实标签二值化
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

# 1. 绘制每个类别的 PR 曲线并存储 (precision, recall)
plt.figure(figsize=(4.5, 4.5))
precision_list, recall_list = [], []
aucs = []

for i, color in enumerate(colors):
    precision_i, recall_i, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
    auc_i = average_precision_score(y_true_bin[:, i], y_score[:, i])
    aucs.append(auc_i)
    
    # 将各类别曲线设为“虚线”（--）
    plt.plot(
        recall_i,
        precision_i,
        color=color,
        linestyle='--',   # <--- 改成虚线
        lw=2,
        label=f'{class_labels[i]} (area = {auc_i:.4f})'
    )
    
    precision_list.append(precision_i)
    recall_list.append(recall_i)



# 2. 计算并绘制 Macro-average PR 曲线（实线）
all_recall = np.linspace(0, 1, 100)

precision1, recall2, _ = precision_recall_curve(y_true_bin.ravel(),y_score.ravel())
average_precision = average_precision_score(y_true_bin, y_score, average="macro")

plt.plot(
    recall2,
    precision1,
    color='#8B0000',
    linestyle='-',  # <--- 改成实线
    lw=2,
    label=f'Macro-average (area = {average_precision:.4f})'
)

# # 3. 计算并绘制 Micro-average PR 曲线（实线）
# precision3, recall4, _ = precision_recall_curve(y_true_bin.ravel(),y_score.ravel())
# average_precision2 = average_precision_score(y_true_bin, y_score, average="micro")

# plt.plot(
#     recall4,
#     precision3,
#     color='#4b0082',
#     linestyle='-',  # <--- 改成实线
#     lw=2,
#     label=f'Micro-average (area = {average_precision2:.4f})'
# )


print('Average precision score, macro-averaged over all classes: {0:0.4f}'.format(average_precision))

# 4. 图形样式设置并显示
plt.xlabel('Recall', fontsize=10)
plt.ylabel('Precision', fontsize=10)
# plt.title('PR Curve - RGNN', fontsize=16)
plt.legend(loc='best', fontsize=14)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()
