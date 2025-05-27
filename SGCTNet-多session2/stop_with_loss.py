import torch

def stop_lossUP(old_loss,old_epoch, new_loss, new_epoch, Times):
    if (old_loss < new_loss) :
        if (new_epoch - old_epoch >= Times):
            return 0 #Ӧ��ֹͣ����
        else :
            return 1 #˵����ʧֵû���ڼ�С����epoch��û��
    else :
        return 2 #˵����ʧֵ�ڼ���

def parameter_Regular(model,lambada=0.0005):
# ���������
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.norm(param, p=2)
    return reg_loss * lambada