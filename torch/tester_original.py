import sys;
import os;
import glob;
import math;
import numpy as np;
import random;
import time;
import torch;
import torch.optim as optim;

sys.path.append(os.getcwd());
sys.path.append(os.path.join(os.getcwd(), 'common'));
import common.utils as U;
import common.opts as opts;
import resources.models as models;
import resources.calculator as calc;

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

class Trainer:
    def __init__(self, opt=None, split=0):
        self.opt = opt;
        self.split = split;
        self.testX = None;
        self.testY = None;
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");

    def load_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'test_data_44khz/fold{}_test4000.npz'.format(self.split)), allow_pickle=True);
        self.testX = torch.tensor(np.moveaxis(data['x'], 3, 1)).to(self.device);
        self.testY = torch.tensor(data['y']).to(self.device);
        print(self.testX.shape);
        print(self.testY.shape);

    def __validate(self, net, lossFunc, testX, testY):
        net.eval();
        with torch.no_grad():
            y_pred = None;
            batch_size = (self.opt.batchSize//self.opt.nCrops)*self.opt.nCrops;
            for idx in range(math.ceil(len(testX)/batch_size)):
                x = testX[idx*batch_size : (idx+1)*batch_size];
                #print(x.shape);
                # exit();
                scores = net(x);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

            acc, loss = self.__compute_accuracy(y_pred, testY, lossFunc);
        net.train();
        return acc, loss;

    #Calculating average prediction (10 crops) and final accuracy
    def __compute_accuracy(self, y_pred, y_target, lossFunc):
        with torch.no_grad():
            #Reshape to shape theme like each sample comtains 10 samples, calculate mean and find theindices that has highest average value for each sample
            y_pred = (y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(dim=1);
            y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(dim=1);

            loss = lossFunc(y_pred.float().log(), y_target.float()).item();

            y_pred = y_pred.argmax(dim=1);
            y_target = y_target.argmax(dim=1);

            acc = (((y_pred==y_target)*1).float().mean()*100).item();
            # valLossFunc = torch.nn.KLDivLoss();
            loss = lossFunc(y_pred.float().log(), y_target.float()).item();
            # loss = 0.0;
        return acc, loss;

    def TestModel(self, run=1):
        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        dir = os.getcwd();
        net_path = self.opt.modelPath;
        print(net_path)
        file_paths = glob.glob(net_path);
        for f in file_paths:
            state = torch.load(f, map_location=self.device);
            config = state['config'];
            weight = state['weight'];
            net = models.GetACDNetModel(self.opt.inputLength, 50, self.opt.sr, config).to(self.device);
            net.load_state_dict(weight);
            print('Model found at: {}'.format(f));
            calc.summary(net, (1,1,self.opt.inputLength));
            # exit();
            self.load_data();
            net.eval();
            #Test standard way with 10 crops
            val_acc, val_loss = self.__validate(net, lossFunc, self.testX, self.testY);
            print('Testing - Val: Loss {:.3f}  Acc(top1) {:.2f}%'.format(val_loss, val_acc));


if __name__ == '__main__':
    opt = opts.parse();
    # -- Run for all splits
    opt.testCropWise = False;
    opt.inputLength = 66650;
    opt.sr = 44100;

    modelPath = os.getcwd()+'/torch/pruned_models/tay_80.pt';
    for split in range(2,3):
        opt.modelPath = modelPath.format(split);
        # print(opt.modelPath)

        trainer = Trainer(opt, split);
        trainer.TestModel();
