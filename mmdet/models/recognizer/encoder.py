import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, nh):
        super(Encoder, self).__init__()

        self.rnn = nn.GRU(512, nh, num_layers=2, bidirectional=False)

        self.fc1 = nn.Linear(512*2, 512)
        self.fc2 = nn.Linear(512*2, 512)
        self.fc3 = nn.Linear(6, 1)

    def forward(self, feature):
        inputs = F.max_pool2d(feature, kernel_size=(6, 1), stride=(6, 1), ceil_mode=True)

        ###############
        #增加一个分枝，把原始分枝的feature的通道直接压缩为6，改为 使用线性学习的方法来将feature通道压缩为6，然后将2分枝特征加起来
        # b, c, h, w = feature.size()
        # inputs2 = feature.permute(0,1,3,2)
        # inputs2 = inputs2.contiguous().view(-1, h)
        # inputs2 = self.fc3(inputs2)
        # inputs2 = inputs2.view(b, c, 1, w)
        #
        # inputs = inputs + inputs2
        ###############

        inputs = inputs.squeeze(2)
        inputs = inputs.permute(2, 0, 1)

        _, hidden = self.rnn(inputs)

        hidden = hidden.permute(1, 2, 0)

        return hidden