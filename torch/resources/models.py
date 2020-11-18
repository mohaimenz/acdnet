import torch;
import torch.nn as nn;
import numpy as np;
import random;

#Reproducibility
seed = 42;
random.seed(seed);
np.random.seed(seed);
torch.manual_seed(seed);
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed);
torch.backends.cudnn.deterministic = True;
torch.backends.cudnn.benchmark = False;
###########################################

class ACDNet(nn.Module):
    def __init__(self, input_length, n_class, sr, ch_conf=None):
        super(ACDNet, self).__init__();
        self.input_length = input_length;
        self.ch_config = ch_conf;

        stride1 = 2;
        stride2 = 2;
        channels = 8;
        k_size = (3, 3);
        n_frames = (sr/1000)*10; #No of frames per 10ms

        sfeb_pool_size = int(n_frames/(stride1*stride2));
        tfeb_pool_size = (2,2);
        if self.ch_config is None:
            self.ch_config = [channels, channels*8, channels*4, channels*8, channels*8, channels*16, channels*16, channels*32, channels*32, channels*64, channels*64, n_class];
        avg_pool_kernel_size = (1,4) if self.ch_config[1] < 64 else (2,4);
        fcn_no_of_inputs = self.ch_config[-1];
        conv1, bn1 = self.make_layers(1, self.ch_config[0], (1, 9), (1, stride1));
        conv2, bn2 = self.make_layers(self.ch_config[0], self.ch_config[1], (1, 5), (1, stride2));
        conv3, bn3 = self.make_layers(1, self.ch_config[2], k_size, padding=1);
        conv4, bn4 = self.make_layers(self.ch_config[2], self.ch_config[3], k_size, padding=1);
        conv5, bn5 = self.make_layers(self.ch_config[3], self.ch_config[4], k_size, padding=1);
        conv6, bn6 = self.make_layers(self.ch_config[4], self.ch_config[5], k_size, padding=1);
        conv7, bn7 = self.make_layers(self.ch_config[5], self.ch_config[6], k_size, padding=1);
        conv8, bn8 = self.make_layers(self.ch_config[6], self.ch_config[7], k_size, padding=1);
        conv9, bn9 = self.make_layers(self.ch_config[7], self.ch_config[8], k_size, padding=1);
        conv10, bn10 = self.make_layers(self.ch_config[8], self.ch_config[9], k_size, padding=1);
        conv11, bn11 = self.make_layers(self.ch_config[9], self.ch_config[10], k_size, padding=1);
        conv12, bn12 = self.make_layers(self.ch_config[10], self.ch_config[11], (1, 1));
        fcn = nn.Linear(fcn_no_of_inputs, n_class);
        nn.init.kaiming_normal_(fcn.weight, nonlinearity='sigmoid') # kaiming with sigoid is equivalent to lecun_normal in keras

        self.sfeb = nn.Sequential(
            #Start: Filter bank
            conv1, bn1, nn.ReLU(),\
            conv2, bn2, nn.ReLU(),\
            nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        );

        self.tfeb = nn.Sequential(
            conv3, bn3, nn.ReLU(), nn.MaxPool2d(kernel_size=tfeb_pool_size),\
            conv4, bn4, nn.ReLU(),\
            conv5, bn5, nn.ReLU(), nn.MaxPool2d(kernel_size=tfeb_pool_size),\
            conv6, bn6, nn.ReLU(),\
            conv7, bn7, nn.ReLU(), nn.MaxPool2d(kernel_size=tfeb_pool_size),\
            conv8, bn8, nn.ReLU(),\
            conv9, bn9, nn.ReLU(), nn.MaxPool2d(kernel_size=tfeb_pool_size),\
            conv10, bn10, nn.ReLU(),\
            conv11, bn11, nn.ReLU(), nn.MaxPool2d(kernel_size=tfeb_pool_size),\
            nn.Dropout(0.2),\
            conv12, bn12, nn.ReLU(), nn.AvgPool2d(kernel_size = avg_pool_kernel_size),\
            nn.Flatten(),\
            fcn
        )

        self.output = nn.Sequential(
            nn.Softmax(dim=1)
        );

    def forward(self, x):
        x = self.sfeb(x);
        #swapaxes
        x = x.permute((0, 2, 1, 3));
        x = self.tfeb(x);
        y = self.output[0](x);
        return y;

    def make_layers(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=0, bias=False):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias);
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu'); # kaiming with relu is equivalent to he_normal in keras
        bn = nn.BatchNorm2d(out_channels);
        return conv, bn;

def GetACDNetModel(input_len=66650, nclass=50, sr=44100, channel_config=None):
    net = ACDNet(input_len, nclass, sr, ch_conf=channel_config);
    return net;

#quantization:
from torch.quantization import QuantStub, DeQuantStub
class ACDNetQuant(nn.Module):
    def __init__(self, input_length, n_class, sr, ch_conf=None):
        super(ACDNetQuant, self).__init__();
        self.input_length = input_length;
        self.ch_config = ch_conf;

        stride1 = 2;
        stride2 = 2;
        channels = 8;
        k_size = (3, 3);
        n_frames = (sr/1000)*10; #No of frames per 10ms

        sfeb_pool_size = int(n_frames/(stride1*stride2));
        pool_size = (2,2);
        if self.ch_config is None:
            self.ch_config = [channels, channels*8, channels*4, channels*8, channels*8, channels*16, channels*16, channels*32, channels*32, channels*64, channels*64, n_class];
        avg_pool_kernel_size = (1,4) if self.ch_config[1] < 64 else (2,4);
        fcn_no_of_inputs = self.ch_config[-1];
        conv1, bn1 = self.make_layers(1, self.ch_config[0], (1, 9), (1, stride1));
        conv2, bn2 = self.make_layers(self.ch_config[0], self.ch_config[1], (1, 5), (1, stride2));
        conv3, bn3 = self.make_layers(1, self.ch_config[2], k_size, padding=1);
        conv4, bn4 = self.make_layers(self.ch_config[2], self.ch_config[3], k_size, padding=1);
        conv5, bn5 = self.make_layers(self.ch_config[3], self.ch_config[4], k_size, padding=1);
        conv6, bn6 = self.make_layers(self.ch_config[4], self.ch_config[5], k_size, padding=1);
        conv7, bn7 = self.make_layers(self.ch_config[5], self.ch_config[6], k_size, padding=1);
        conv8, bn8 = self.make_layers(self.ch_config[6], self.ch_config[7], k_size, padding=1);
        conv9, bn9 = self.make_layers(self.ch_config[7], self.ch_config[8], k_size, padding=1);
        conv10, bn10 = self.make_layers(self.ch_config[8], self.ch_config[9], k_size, padding=1);
        conv11, bn11 = self.make_layers(self.ch_config[9], self.ch_config[10], k_size, padding=1);
        conv12, bn12 = self.make_layers(self.ch_config[10], self.ch_config[11], (1, 1));
        fcn = nn.Linear(fcn_no_of_inputs, n_class);
        nn.init.kaiming_normal_(fcn.weight, nonlinearity='sigmoid') # kaiming with sigoid is equivalent to lecun_normal in keras

        self.sfeb = nn.Sequential(
            #Start: Filter bank
            conv1, bn1, nn.ReLU(),\
            conv2, bn2, nn.ReLU(),\
            nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        );

        self.tfeb = nn.Sequential(
            conv3, bn3, nn.ReLU(), nn.MaxPool2d(kernel_size=(2,2)),\
            conv4, bn4, nn.ReLU(),\
            conv5, bn5, nn.ReLU(), nn.MaxPool2d(kernel_size=(2,2)),\
            conv6, bn6, nn.ReLU(),\
            conv7, bn7, nn.ReLU(), nn.MaxPool2d(kernel_size=(2,2)),\
            conv8, bn8, nn.ReLU(),\
            conv9, bn9, nn.ReLU(), nn.MaxPool2d(kernel_size=(2,2)),\
            conv10, bn10, nn.ReLU(),\
            conv11, bn11, nn.ReLU(), nn.MaxPool2d(kernel_size=(2,2)),\
            nn.Dropout(0.2),\
            conv12, bn12, nn.ReLU(), nn.AvgPool2d(kernel_size = avg_pool_kernel_size),\
            nn.Flatten(),\
            fcn
        )

        self.output = nn.Sequential(
            nn.Softmax(dim=1)
        );
        self.quant = QuantStub();
        self.dequant = DeQuantStub();
    def forward(self, x):
        #Quantize input
        x = self.quant(x);
        x = self.sfeb(x);
        #swapaxes
        x = x.permute((0, 2, 1, 3));
        x = self.tfeb(x);
        #DeQuantize features before feeding to softmax
        x = self.dequant(x);
        y = self.output[0](x);
        return y;

    def make_layers(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=0, bias=False):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias);
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu'); # kaiming with relu is equivalent to he_normal in keras
        bn = nn.BatchNorm2d(out_channels);
        return conv, bn;

def GetACDNetQuantModel(input_len=66650, nclass=50, sr=44100, channel_config=None):
    net = ACDNetQuant(input_len, nclass, sr, ch_conf=channel_config);
    return net;
