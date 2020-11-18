import tensorflow.keras.backend as K;
from tensorflow import keras;
from tensorflow.keras.models import Model;
import tensorflow.keras.layers as L
import numpy as np;

#acdnet in functional way for safe serialization for embedded devices
class ACDNet:
    def __init__(self, input_length=66650, n_class=50, sr=44100, ch_conf=None):

        self.input_length = input_length;
        self.ch_config = ch_conf;

        stride1 = 2;
        stride2 = 2;
        channels = 8;
        k_size = (3, 3);
        n_frames = (sr/1000)*10; #No of frames per 10ms

        self.sfeb_pool_size = int(n_frames/(stride1*stride2));
        self.tfeb_pool_size = (2,2);
        if self.ch_config is None:
            self.ch_config = [channels, channels*8, channels*4, channels*8, channels*8, channels*16, channels*16, channels*32, channels*32, channels*64, channels*64, n_class];
        self.avg_pool_kernel_size = (1,4) if self.ch_config[1] < 64 else (2,4);

        self.conv1 = ConvBlock(self.ch_config[0], (1, 9), (1, stride1));
        self.conv2 = ConvBlock(self.ch_config[1], (1, 5), (1, stride2));
        self.conv3 = ConvBlock(self.ch_config[2], k_size, padding='same');
        self.conv4 = ConvBlock(self.ch_config[3], k_size, padding='same');
        self.conv5 = ConvBlock(self.ch_config[4], k_size, padding='same');
        self.conv6 = ConvBlock(self.ch_config[5], k_size, padding='same');
        self.conv7 = ConvBlock(self.ch_config[6], k_size, padding='same');
        self.conv8 = ConvBlock(self.ch_config[7], k_size, padding='same');
        self.conv9 = ConvBlock(self.ch_config[8], k_size, padding='same');
        self.conv10 = ConvBlock(self.ch_config[9], k_size, padding='same');
        self.conv11 = ConvBlock(self.ch_config[10], k_size, padding='same');
        self.conv12 = ConvBlock(self.ch_config[11], (1, 1));

        self.fcn = L.Dense(n_class, kernel_initializer=keras.initializers.he_normal());

    def createModel(self):
        #batch, rows, columns, channels
        input = L.Input(shape=(1, self.input_length, 1));

        #Start: SFEB
        sfeb = self.conv1(input);
        sfeb = self.conv2(sfeb);
        sfeb = L.MaxPooling2D(pool_size=(1, self.sfeb_pool_size))(sfeb);
        #End: SFEB

        #swapaxes
        tfeb = L.Permute((3, 2, 1))(sfeb);

        # exit();
        #Start: HLFE
        tfeb = self.conv3(tfeb);
        tfeb = L.MaxPooling2D(pool_size=self.tfeb_pool_size)(tfeb);

        tfeb = self.conv4(tfeb);
        tfeb = self.conv5(tfeb);
        tfeb = L.MaxPooling2D(pool_size=self.tfeb_pool_size)(tfeb);

        tfeb = self.conv6(tfeb);
        tfeb = self.conv7(tfeb);
        tfeb = L.MaxPooling2D(pool_size=self.tfeb_pool_size)(tfeb);

        tfeb = self.conv8(tfeb);
        tfeb = self.conv9(tfeb);
        tfeb = L.MaxPooling2D(pool_size=self.tfeb_pool_size)(tfeb);

        tfeb = self.conv10(tfeb);
        tfeb = self.conv11(tfeb);
        tfeb = L.MaxPooling2D(pool_size=self.tfeb_pool_size)(tfeb);

        tfeb =  L.Dropout(rate=0.2)(tfeb);

        tfeb = self.conv12(tfeb);
        tfeb = L.AveragePooling2D(pool_size=self.avg_pool_kernel_size)(tfeb);

        tfeb = L.Flatten()(tfeb);
        tfeb = self.fcn(tfeb);
        #End: tfeb

        output = L.Softmax()(tfeb);

        model = Model(inputs=input, outputs=output);
        # model.summary();
        return model;

class ConvBlock:
    def __init__(self, filters, kernel_size, stride=(1,1), padding='valid', use_bias=False):
        self.conv = L.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding, kernel_initializer=keras.initializers.he_normal(), use_bias=use_bias);

    def __call__(self, x):
        layer = self.conv(x);
        layer = L.BatchNormalization()(layer);
        layer = L.ReLU()(layer);
        return layer;

def GetAcdnetModel(input_length=66650, n_class=50, sr=44100, ch_config=None):
    acdnet = ACDNet(input_length, n_class, sr, ch_config);
    return acdnet.createModel();

# net = GetAcdnetModel();
# net.summary();
