import torch
from torch import nn
from torch.nn import functional as F

class SPABlock(nn.Module):
    def __init__(self, in_channels, k=8, adaptive = False, reduction=16, learning=False, mode='pow'):
        """
        Salient Positions Selection (SPS) algorithm
        :param in_channels: 待处理数据的通道数目
        :param k=5, 默认的选择通道数目
        :param kadaptive = False: k是否需要根据通道数进行自适应选择
        :param learning=False: k是否需要学习
        :param mode='power':挑选k个位置的计算方法
        :return out, [batchsize, self.k, channels]
        """
        super(SPABlock, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.k = k
        self.adptive = adaptive
        self.reduction = reduction
        self.learing = learning
        if self.learing is True:
            self.k = nn.Parameter(torch.tensor(self.k))

        self.mode = mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x, return_info=False):
        input_shape = x.shape
        if len(input_shape)==4:
            x = x.view(x.size(0), self.in_channels, -1)
            x = x.permute(0, 2, 1)
        batch_size,N = x.size(0),x.size(1)

        #（B, H*W，C）
        if self.mode == 'pow':
            x_pow = torch.pow(x,2)# （batchsize，H*W，channel）
            x_powsum = torch.sum(x_pow,dim=2)# （batchsize，H*W）

        if self.adptive is True:
            self.k = N//self.reduction
            if self.k == 0:
                self.k = 1

        outvalue, outindices = x_powsum.topk(k=self.k, dim=-1, largest=True, sorted=True)

        outindices = outindices.unsqueeze(2).expand(batch_size, self.k, x.size(2))
        out = x.gather(dim=1, index=outindices).to(self.device)

        if return_info is True:
            return out, outindices, outvalue
        else:
            return out

class SPANet(nn.Module):
    def __init__(self, in_channels, inter_channels=None,sub_sample=False,adaptive=False,k=8):
        """
        :param in_channels:
        :param inter_channels:   是降维的手段之一，从通道数目上降维,如果是None，就会降维；不降维，可以直接指定
        :param sub_sample:       是降维的手段之一，从空间像素数目上降维
        :param adaptive:     用于spablock自适应的设置k
        :param k:            用于设定spablock的显著性位置的数目,默认为8
        :return:
        """
        super(SPANet, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_2d = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )

        self.theta = conv_2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if self.sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)

        self.spablock = SPABlock(self.inter_channels,k=k,adaptive=adaptive).to(self.device)

    def forward(self, x, return_spa_map=False):
        """
        :param x: (b, c, h, w)
        :param return_spa_map: if True return z, nl_map, else only return z.
        :return:z (b, c, h, w)
        """
        batch_size, c, h, w = x.size(0),x.size(1),x.size(2),x.size(3)
        g_x = self.g(x)
        g_xh,g_xw = g_x.size(2),g_x.size(3)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        theta_x = theta_x.to(self.device)
        if return_spa_map is True:
            theta_xk, theta_xk_indices,theta_xk_pow = self.spablock.forward(theta_x,return_spa_map)
        else:
            theta_xk = self.spablock.forward(theta_x, return_spa_map)
        theta_xk = theta_xk.to(device=self.device)

        phi_x = theta_xk.permute(0,2,1)  #(b,  c， k)

        # (b,  c， k) * (b,  k， c) --> (b,  c， c)
        f = torch.matmul(phi_x, theta_xk)

        # (b,  c， c)
        attmapsfusion = F.softmax(f, dim=-1)

        # (b,  c， c)  * (b,  c， n)
        g_x = g_x.view(batch_size,self.inter_channels,-1)
        aggregation = torch.matmul(attmapsfusion,g_x)

        aggregation = aggregation.view(batch_size,self.inter_channels,g_xh,g_xw)

        # (b, c, h, w)
        W_y = self.W(aggregation)
        if self.sub_sample is True:
            W_y = self.up(W_y)

        # (b, c, h, w)
        z = W_y + x

        if return_spa_map:
            return z, [attmapsfusion, theta_xk, phi_x, theta_xk_indices,theta_xk_pow]
        return z

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sub_sample_ = True
    img = torch.randn(2, 3, 20, 20).to(device)
    net = SPANet(in_channels=3, sub_sample=sub_sample_, inter_channels=3, adaptive=True, k=9).to(device)
    print(net)
    out = net(img)
    tmp = out-img
    print(out.size())

    spa = SPABlock(in_channels=3, k=8, adaptive=True).to(device)
    out = spa(img)
    print(out.shape)
