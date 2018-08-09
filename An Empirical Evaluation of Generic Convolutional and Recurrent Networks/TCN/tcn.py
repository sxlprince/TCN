import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):   #2 4 8 16
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        #print ("因果卷积前的shape：",x.size())
        #print("因果卷积后的shape：", x[:, :, :-self.chomp_size].size())
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):        #16 * 600 * 80
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):

        #n_inputs = 600 600 600 600
        #n_outputs = 600 600 600 600
        #kernel_size = 3
        #stride = 1
        #dilation = 1 2 4 8
        #padding = 2 4 8 16
        #dropout = 0.45

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,    #16 * 600 * 82
                                           stride=stride, padding=padding, dilation=dilation))
        #print (self.conv1)
        #加 padding 是为了保证因果后句子长度不变， 输出是16 * 600 * 82
        self.chomp1 = Chomp1d(padding)    #16 * 600 * 80
        #print (self.chomp1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,    #16 * 600 * 82
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)    #16 * 600 * 80
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        #print ("经过两次空洞因果卷积输出数据shape：", out.size())
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

#16 * 600 * 80
class TemporalConvNet(nn.Module):    #TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):

        #num_inputs = 600
        #num_channels = [600,600,600,600]
        #kernel_size = 3
        #dropout = 0.45

        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)   #4
        for i in range(num_levels):
            dilation_size = 2 ** i               #1
            in_channels = num_inputs if i == 0 else num_channels[i-1]  #600  600 600 600
            out_channels = num_channels[i]                              #600  600 600 600
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # print ("输入数据shape1：", x.size())
        # print ("输出数据shape：", self.network(x).size())
        # print ('sxlllllllllllllllllllllllllllllllllllllllllllllll')
        return self.network(x)
