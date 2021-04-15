import torch
import torch.nn as nn

def mask_softmax(input, dim, mask):
    input_max = torch.max(input, dim=dim, keepdim=True)[0]
    input_exp = torch.exp(input - input_max)
    input_exp = input_exp * mask.float()

    input_softmax = input_exp / torch.sum(input_exp, dim=dim, keepdim=True) + 0.0000001
    return input_softmax

class Attentiondecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, noy=False, show_attention=False):
        super(Attentiondecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.noy = noy
        self.show_attention = show_attention

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout1 = nn.Dropout(self.dropout_p)

        self.rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers=2)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)

        self.conv1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(512, 1, kernel_size=1, stride=1, bias=False)


    def forward(self, input, hidden=None, feature=None, mask=None):
        if not self.noy:
            input = self.embedding(input)
        input = self.dropout1(input)
        input = input.unsqueeze(0)

        _, hidden = self.rnn(input, hidden)

        hidden_tmp = hidden[1:].permute(1, 2, 0).unsqueeze(3)
        hidden_tmp = self.conv1(hidden_tmp)
        hidden_tmp = hidden_tmp.expand_as(feature)

        encode_conv = self.conv2(feature)
        encode_conv = self.conv3(torch.tanh(encode_conv + hidden_tmp)).view(encode_conv.shape[0], 1, -1)
        mask = mask.view(mask.shape[0], mask.shape[1], -1)
        w = mask_softmax(encode_conv, dim=2, mask=mask)

        feature = feature.view(feature.shape[0], feature.shape[1], -1)
        c = torch.sum(feature * w, 2)

        output = torch.cat([hidden[1], c], 1)
        output = self.out(output)

        if self.show_attention:
            return output, hidden, w
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, nh=256, nclass=37, dropout_p=0.1, noy=False, show_attention=False):
        super(Decoder, self).__init__()
        self.hidden_size = nh
        self.decoder = Attentiondecoder(nh, nclass, dropout_p, noy=noy, show_attention=show_attention)

    def forward(self, input, hidden, feature, mask):
        return self.decoder(input, hidden, feature, mask)
