class lstm(nn.Module):
    def __init__(self,input_dim,num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=128, num_layers=num_layers, batch_first=True,bidirectional=True)

    def forward(self,x):
        out, _ = self.lstm(x)
        return out

class conv(nn.Module):
    def __init__(self,input_channels,output_channels, kernel_size):
        super().__init__()
        self.convs= nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_channels)
        )
    def forward(self,x):
        out = self.convs(x)
        return out      
        
class lstm_res(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = lstm(7,2)
        self.conv1 = conv(1,128,5)
        self.conv2 = conv(128,64,5)
        self.conv3 = conv(1,64,1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Dense = nn.Sequential(nn.Linear(64, 32),nn.ReLU(),nn.Linear(32, 18))

    def forward(self,x):
        lstm_out1 = self.lstm1(x)
        conv_out1 = self.conv1(lstm_out1.unsqueeze(1))    ##dimension add for cnn layers
        conv_out2 = self.conv2(conv_out1)
        conv_out3 = self.conv3(lstm_out1.unsqueeze(1))

        upsampled = F.interpolate(conv_out3, size=conv_out2.shape[2:], mode= 'nearest')
        out = (conv_out2 + upsampled)
        out = self.avg_pool(out).view(out.shape[0],-1)

        out = self.Dense(out)
        return out
