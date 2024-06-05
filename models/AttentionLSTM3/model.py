import torch
from torch import nn
from torch import Tensor
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Encoder(nn.Module):

    def __init__(self, input_channels, output_channels,sentence_length,kernel_sizes=[5,5,5],output_size=128):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_sizes = kernel_sizes

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=output_channels[0], kernel_size=kernel_sizes[0])
        # sequence_length = sentence_length - kernel_sizes[0] -1
        self.mod_seq_len = sentence_length - kernel_sizes[0] + 1
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(output_channels[0])

        self.pos_encoder = PositionalEncoding(output_channels[0])
        self.attention1 = nn.MultiheadAttention(embed_dim=output_channels[0], num_heads=16, batch_first=True)
        self.layernorm1 = nn.LayerNorm([self.mod_seq_len, output_channels[0]])


        self.conv2 = nn.Conv1d(in_channels=output_channels[0], out_channels=output_channels[1], kernel_size=kernel_sizes[1])
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(output_channels[1])
        self.mod_seq_len2 = self.mod_seq_len - kernel_sizes[1] + 1

        self.lstm1=nn.LSTM(input_size=output_channels[1],hidden_size=output_channels[1],num_layers=1,batch_first=True)

        self.conv3 = nn.Conv1d(in_channels=output_channels[1], out_channels=output_channels[2], kernel_size=kernel_sizes[2])
        self.relu3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm1d(output_channels[2])

        self.mod_seq_len3 = self.mod_seq_len2 - kernel_sizes[2] + 1
        self.attention2 = nn.MultiheadAttention(embed_dim=output_channels[2], num_heads=16, batch_first=True)
        self.layernorm2 = nn.LayerNorm([self.mod_seq_len3, output_channels[2]])

        #print(output_channels[2]*self.mod_seq_len3)
        self.dense = nn.Linear(output_channels[2]*self.mod_seq_len3, output_size)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, seq = x.size()

        x=self.batchnorm1(self.relu1(self.conv1(x)))
        #print(x.shape)
        x=x.permute(2,0,1)
        #print(x.shape)
        x=self.pos_encoder(x)
        x=x.permute(1,0,2)

        #print(x.shape)
        res=x
        out,_=self.attention1(x,x,x)
        out+=res
        out=self.layernorm1(out)

        out=out.permute(0,2,1)
        out=self.batchnorm2(self.relu2(self.conv2(out)))

        out=out.permute(0,2,1)
        out,_=self.lstm1(out)

        out=out.permute(0,2,1)
        out=self.batchnorm3(self.relu3(self.conv3(out)))

        out=out.permute(0,2,1)
        res2=out
        out2,_=self.attention2(out,out,out)
        out2+=res2
        out2=self.layernorm2(out2)

        out2=out2.view(batch,-1)
        out2=self.dense(out2)

        return out2

class AttentionLSTM3(nn.Module):

    """
    This will contain the previously build encoder model at different scales.
    the different scales will be achieved using 1d convolutional layers with different kernel sizes.
    output from each of these layers will be concatenated and passed to a dense layer.
    """

    def __init__(self,input_channel,sequence_length,hidden_size):

        super().__init__()
        self.input_channel = input_channel
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size


        self.conv1 = nn.Conv1d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(32)

        self.encoder1 = Encoder(32,[128,256,128],sequence_length,[7,7,7],128)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.relu2=nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.sequence_length2 = sequence_length - 5 + 1
        #print(self.sequence_length2)

        self.encoder2 = Encoder(64,[128,256,128],self.sequence_length2,[7,5,3],128)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.relu3=nn.ReLU()
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.sequence_length3 = self.sequence_length2 - 5 + 1

        self.encoder3 = Encoder(128,[128,256,128],self.sequence_length3,[5,5,3],128)

        self.conv4=nn.Conv1d(128,256,kernel_size=5)
        self.relu4=nn.ReLU()
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.sequence_length4 = self.sequence_length3 - 5 + 1

        self.encoder4 = Encoder(256,[256,256,128],self.sequence_length4,[5,5,3],128)

        self.dense = nn.Linear(128*4,256)
        self.reluF=nn.ReLU()


    def forward(self, x: Tensor) -> Tensor:

        #print("$"*100,"\n",x.shape)
        x=self.batchnorm1(self.relu1(self.conv1(x)))
        #print(x.shape)
        out1=self.encoder1(x)

        x=self.batchnorm2(self.relu2(self.conv2(x)))
        #print(x.shape)
        out2=self.encoder2(x)

        x=self.batchnorm3(self.relu3(self.conv3(x)))
        #print(x.shape)
        out3=self.encoder3(x)

        x=self.batchnorm4(self.relu4(self.conv4(x)))
        #print(x.shape)
        out4=self.encoder4(x)

        y=torch.cat([out1,out2,out3,out4],dim=1)
        y=self.dense(y)
        y=self.reluF(y)

        return y




model=AttentionLSTM3(5,30,128)
x=torch.randn(128,5,30)
print(model(x).shape)
