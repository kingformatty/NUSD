# source: https://github.com/dong-8080/ETDNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bp


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)
        return x


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale -1

        self.convs = []
        self.bns = [] 
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))

        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out

class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = torch.tanh(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out

def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels)
    )


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1) # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1) # equals V and k in the paper

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

class ECAPA_TDNN(nn.Module):
    def __init__(self, in_channels=80, channels=512, att_dim=128, embd_dim=192, class_num=1):
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        self.dropout = nn.Dropout(p=0.5)
        
        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, 1536, kernel_size=1)
        self.pooling = AttentiveStatsPool(1536, att_dim)
        self.bn1 = nn.BatchNorm1d(3072)

        self.linear2 = nn.Linear(3072, embd_dim)
        self.bn2 = nn.BatchNorm1d(embd_dim)
        self.linear3 = nn.Linear(embd_dim, class_num)
        self.bn3 = nn.BatchNorm1d(class_num)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = x.transpose(1, 2)
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        out = self.bn2(self.linear2(out))
        out = self.bn3(self.linear3(out))
        # out = self.softmax(out)
        out = self.sigmoid(out)
        return out



class ECAPA_TDNN_small(nn.Module):
    def __init__(self, in_channels=80, channels=128, att_dim=64, embd_dim=128, class_num=1):
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        self.dropout = nn.Dropout(p=0.5)
        
        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, 384, kernel_size=1)
        self.pooling = AttentiveStatsPool(384, att_dim)
        self.bn1 = nn.BatchNorm1d(768)

        self.linear2 = nn.Linear(768, embd_dim)
        self.bn2 = nn.BatchNorm1d(embd_dim)
        self.linear3 = nn.Linear(embd_dim, class_num)
        self.bn3 = nn.BatchNorm1d(class_num)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = x.transpose(1, 2)
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        out_embd = self.bn2(self.linear2(out))
        out_pred = self.bn3(self.linear3(out_embd))
        # out = self.softmax(out)
        out_pred = self.sigmoid(out_pred)

        return out_pred,out_embd


class ECAPA_TDNN_Med(nn.Module):
    def __init__(self, in_channels=40, channels=256, att_dim=64, embd_dim=128, class_num=1):
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        self.dropout = nn.Dropout(p=0.5)
        
        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, 768, kernel_size=1)
        self.pooling = AttentiveStatsPool(768, att_dim)
        self.bn1 = nn.BatchNorm1d(1536)

        self.linear2 = nn.Linear(1536, embd_dim)
        self.bn2 = nn.BatchNorm1d(embd_dim)
        self.linear3 = nn.Linear(embd_dim, class_num)
        self.bn3 = nn.BatchNorm1d(class_num)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = x.transpose(1, 2)
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        out = self.bn2(self.linear2(out))
        out = self.bn3(self.linear3(out))
        # out = self.softmax(out)
        out = self.sigmoid(out)
        return out

if __name__ == '__main__':
    # just for verifing the model
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # print("使用的device", device)

    # x = torch.zeros(2, 80, 200).to(device)
    ecapa_small_40mel = ECAPA_TDNN_Med(in_channels=40)
    ecapa_small_80mel = ECAPA_TDNN_Med(in_channels=80)
    # ecapa_large_40mel = ECAPA_TDNN(in_channels=40)
    # ecapa_large_80mel = ECAPA_TDNN(in_channels=80)
    # out = model(x)
    # print(out.shape)
    pytorch_total_params_ecapa_small_40mel = sum(p.numel() for p in ecapa_small_40mel.parameters() if p.requires_grad)
    print(pytorch_total_params_ecapa_small_40mel)
    pytorch_total_params_ecapa_small_80mel = sum(p.numel() for p in ecapa_small_80mel.parameters() if p.requires_grad)
    print(pytorch_total_params_ecapa_small_80mel)
    # pytorch_total_params_ecapa_large_40mel = sum(p.numel() for p in ecapa_large_40mel.parameters() if p.requires_grad)
    # print(pytorch_total_params_ecapa_large_40mel)
    # pytorch_total_params_ecapa_large_80mel = sum(p.numel() for p in ecapa_large_80mel.parameters() if p.requires_grad)
    # print(pytorch_total_params_ecapa_large_80mel)





