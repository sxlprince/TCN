import torch
from torch import nn
import sys
sys.path.append("../../")
from TCN.tcn import TemporalConvNet


class TCN(nn.Module):                  #16 * 80

    # TCN(args.emsize, n_words, num_chans, dropout=dropout, emb_dropout=emb_dropout, kernel_size=k_size, tied_weights=tied)

    def __init__(self, input_size, output_size, num_channels,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, tied_weights=False):

        #input_size = 600
        #output_size = 10000     vocab_size 不重复单词个数
        #num_channels = [600,600,600,600]
        #kernel_size = 3
        #dropout = 0.45
        #emb_dropout = 0.25
        #tied_weights= True

        super(TCN, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)    #vocab_10000 * 600  随机初始化
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], output_size)
        if tied_weights:
            if num_channels[-1] != input_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.drop(self.encoder(input))         #16 * 80 * 600
        #print ("输入数据shape：",emb.transpose(1,2).size())
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)      #16 * 80 * 10000
        return y.contiguous()

