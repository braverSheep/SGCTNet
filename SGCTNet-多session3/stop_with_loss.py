import torch

def stop_lossUP(old_loss,old_epoch, new_loss, new_epoch, Times):
    if (old_loss < new_loss) :
        if (new_epoch - old_epoch >= Times):
            return 0 #应该停止整个
        else :
            return 1 #说明损失值没有在减小，但epoch数没到
    else :
        return 2 #说明损失值在减少

def parameter_Regular(model,lambada=0.0005):
# 添加正则化项
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.norm(param, p=2)
    return reg_loss * lambada