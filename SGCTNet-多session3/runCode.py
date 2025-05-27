
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #随机梯度下降（SGD）、Adam、Adagrad、RMSprop
import torch_geometric
from torch_scatter import scatter_add
import os
import scipy.io
import random

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

dev = ("cuda" if torch.cuda.is_available() else "cpu")#可用cuda
print(dev)

root_dir = "D:/dataset/SEED-IV/"

# 整个训练脚本产生的结果都将存入这个脚本中，模型，log等等
def cre_prolog():
    # 获取当前脚本的绝对路径  
    current_script_path = os.path.abspath(__file__)  
    # 获取当前脚本的所在目录  
    parent_dir = os.path.dirname(current_script_path)  
    # 定义要创建的Pro_log目录的路径
    pro_log_dir_path = os.path.join(parent_dir, 'Pro_log')  
      
    # 检查Pro_log目录是否存在，如果不存在则创建  
    if not os.path.exists(pro_log_dir_path):  
        os.makedirs(pro_log_dir_path)  
        print(f"Directory '{pro_log_dir_path}' created.")  

    return pro_log_dir_path
exp_dir = cre_prolog()
log_file = f"{exp_dir}/log.txt"

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

#-----------Subject dependent----------
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
    for session in range(1,4):
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
    
    loaders_dict["train"] = torch_geometric.loader.DataLoader(eeg_dict["train"], batch_size=batch_size, shuffle=True, drop_last=False)#drop_last：如果为True，最后一个不完整的批次将被丢弃
    loaders_dict["test"] = torch_geometric.loader.DataLoader(eeg_dict["test"], batch_size=batch_size, shuffle=True, drop_last=False)            
    
    return  loaders_dict

import stop_with_loss

import My_NewSCI_module
import TransGCN_Unet
import MBT
import RGNN_with_trans
from torch.optim.lr_scheduler import StepLR
from RGNN_with_trans import SGCN_Transformer

def subject_dependent_training(data_loaders, K,edge_weight, rand_adj=False, L1_alpha=0.005, noise_level=0., num_hiddens=16, num_epochs=500,Lr=0.001):
    print("TransGCN_Unet_position_changed_session1-使用session1预训练好的结果_多session数据训练")
    eps3 = noise_level/3
    eps4 = noise_level/4
    emoDL_map = torch.tensor([
        [1-3*eps4, eps4, eps4, eps4],
        [eps3, 1-2*eps3, eps3, 0],
        [eps4, eps4, 1-3*eps4,eps4],
        [eps3, 0, eps3, 1-2*eps3]
        ]).to(dev)
    best_model_acc = 0
    print("PAZIENTE no."+ str(K) +" :")

    model = RGNN_with_trans.FuseModel(62, True, edge_weight.cuda(), 5, num_hiddens, 4) 
    model = model.to(dev)
    optimizer = optim.Adam(model.parameters(), lr=Lr)
    old_loss, old_epoch = 999999, 0
    best_epoch = 0
    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        sum_loss_train = 0
        sum_loss_test = 0
        sum_accuracy_train = 0
        sum_accuracy_test = 0
        
        #训练网络
        model.train()
        i =0
        for batch_train in data_loaders["train"]:
            i=i+1
    
            pred_fuse, edge_weight_fuse = model(batch_train)#edge_weight_psd, edge_weight_de, 
            y_list = batch_train[1].y.to('cpu')
            loss = None
            optimizer.zero_grad()
            target = emoDL_map[y_list].to(dev)
            loss = F.kl_div(F.log_softmax(pred_fuse, -1), target, reduction="sum") +  L1_alpha * torch.norm(edge_weight_fuse)  + stop_with_loss.parameter_Regular(model,lambada=0.001)#+ L1_alpha/3 * torch.norm(edge_weight_psd) + L1_alpha/3 * torch.norm(edge_weight_de)

            pred_target = pred_fuse.max(1)

            sum_loss_train += loss.item()
            loss.backward()
            optimizer.step()

            _,pred_target = pred_fuse.max(1)
            batch_accuracy = (pred_target == batch_train[1].y).sum().item()
            sum_accuracy_train += batch_accuracy

        #测试网络
        model.eval()
        for batch_test in data_loaders["test"]:
            pred_fuse,  edge_weight_fuse = model(batch_test)#edge_weight_psd, edge_weight_de,
            y_list = batch_test[1].y.to('cpu')
            loss = None
            target = emoDL_map[y_list].to(dev)
            loss = F.kl_div(F.log_softmax(pred_fuse, -1), target, reduction="sum") +  L1_alpha * torch.norm(edge_weight_fuse)  #+ stop_with_loss.parameter_Regular(model,lambada=0.001)#+ L1_alpha/3 * torch.norm(edge_weight_psd) + L1_alpha/3 * torch.norm(edge_weight_de)
            
            sum_loss_test += loss.item()
            _,pred_target = pred_fuse.max(1)
            batch_accuracy = (pred_target == batch_test[1].y).sum().item()
            sum_accuracy_test += batch_accuracy
        # 在每个epoch开始时，调用学习率调度器
        #scheduler.step()   

        x_trian = len(data_loaders["train"].dataset)
        y_test = len(data_loaders["test"].dataset)
        epoch_loss_train = sum_loss_train / x_trian
        epoch_accuracy_train = sum_accuracy_train / x_trian
        epoch_loss_test = sum_loss_test / y_test
        epoch_accuracy_test = sum_accuracy_test / y_test
        
        if epoch_accuracy_test > best_model_acc:
            best_model_acc = epoch_accuracy_test
            best_epoch = epoch+1
            torch.save(model.state_dict(),f'./Pro_log/SGCTNet_{K}.pth')
        
        epoch = epoch + 1
        print(f"Epoch {epoch}:",
                    f"TrL={epoch_loss_train:.4f},",
                    f"TrA={epoch_accuracy_train:.4f},",
                    f"TeL={epoch_loss_test:.4f},",
                    f"TeA={epoch_accuracy_test:.4f},",
                    f"TeA={best_model_acc:.4f},"
                    )
        with open(log_file, 'a') as file:  
            line_to_write = f'Epoch {epoch} tra_loss: {epoch_loss_train:.4f}, tra_acc: {epoch_accuracy_train:.4f}, tes_acc: {epoch_loss_test:.4f}, tes_loss: {epoch_accuracy_test:.4f}, best_acc: {best_model_acc:.4f}, best_epoch: {best_epoch}\n'  
            file.write(line_to_write) 
        
        if epoch_accuracy_test>0.99999:
            break
    print("best_model_acc:{},best_epoch:{} ".format(best_model_acc,best_epoch))

    return best_model_acc,best_epoch


#保存数据文件
batch_size = 16
number = 1
# 设置随机种子
x = 12
np.random.seed(x)  # 设置 NumPy 模块的随机种子
random.seed(x)     # 设置 Python 随机模块的随机种子
os.environ['PYTHONHASHSEED'] = str(x)  # 设置 Python 哈希种子

# 设置 PyTorch 的随机种子
torch.manual_seed(x)   # 设置 PyTorch 的随机种子
torch.cuda.manual_seed(x)  # 设置 CUDA 的随机种子
torch.cuda.manual_seed_all(x)  # 设置 CUDA 设备的随机种子
torch.backends.cudnn.benchmark = False  # 关闭 cudnn 的 benchmark 模式
torch.backends.cudnn.deterministic = True  # 设置 cudnn 的随机种子

# 在导入 random 模块之后再次设置 Python 随机种子
import random
random.seed(x)

# # 设置 PyTorch 使用确定性算法
# torch.use_deterministic_algorithms(True)

# 设置 CUDA 随机种子并保存和恢复 CUDA 随机状态
torch.cuda.manual_seed(x)  # 设置 CUDA 随机种子

adj = get_adjacency_matrix()
best_Acc = []
for i in range(1,16):
    print(f'------------------start subect:{i}--------------------- ')
    with open(log_file, 'a') as file:  
        file.write(f'\n---------------------------------------------------------------------start subect:{i}---------------------------------------------------------------------\n ')

    dataloader_train = dependent_loaders(batch_size, i) 
    best_model_acc, best_epoch= subject_dependent_training(dataloader_train,i,adj,num_epochs=300,Lr=0.002)
    best_Acc.append(best_model_acc)

k = 0
result_string = ''
for i in best_Acc:
    k=k+1
    str1 = f'subject{k} acc:{i} \n'
    result_string = ''.join([result_string, str1])

# 所有subjects的平均准确率、标准差
mean_acc = np.mean(best_Acc)
sta = np.std(best_Acc, ddof=0)
mean_str = f'mean_acc:{mean_acc} sta:{sta}'

result_string = ''.join([result_string, mean_str]) 

with open(log_file, 'a') as file:  
        file.write(result_string)
print(result_string)