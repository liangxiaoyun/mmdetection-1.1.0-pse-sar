import torch
import collections

class strLabelConverterForAttention(object):
    """Convert between str and label.
    NOTE:
        Insert `EOS` to the alphabet for attention.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, text_max_len):
        self.alphabet = alphabet
        self.text_max_len = text_max_len

        self.dict = {}
        self.dict['EOS'] = 0       # 开始
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i + 1 # 从1开始编码
        self.v2k_dict = dict()
        for ink, k in enumerate(list(self.dict.values())):
            self.v2k_dict[k] = list(self.dict.keys())[ink]

    def encode(self, text, ignore_index=False, more_length=False):
        """对target_label做编码和对齐
        对target txt每个字符串的开始加上SOS，最后加上EOS，并用最长的字符串做对齐
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor targets:max_length × batch_size
        """

        text = [self.dict[item] for item in text]  # 编码

        nb = 1
        if more_length:
            ratio = 2
        else:
            ratio = 1
        targets = torch.zeros(nb, (self.text_max_len + 2) * ratio)  # use ‘blank’ for pading
        if ignore_index:
            targets[:, :] = -1
        for i in range(nb):
            targets[i][0] = 0  # 开始
            for j, t in enumerate(text):
                targets[i][j+1] = t
            targets[i][len(text) + 1] = 0
        text = targets.transpose(0, 1).contiguous()  # .contiguous()把tensor变成在内存中连续分布的形式
        # 如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy
        text = text.long()
        return torch.LongTensor(text)

    def decode(self, t):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """

        # texts = list(self.dict.keys())[list(self.dict.values()).index(t)]
        return self.v2k_dict[t.item()]