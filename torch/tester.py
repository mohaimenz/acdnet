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
import common.opts as opts;
import resources.models as models;
import resources.calculator as calc;

class Trainer:
    def __init__(self, opt=None):
        self.opt = opt;
        self.testX = None;
        self.testY = None;

    def load_test_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'test_data_{}khz/fold{}_test4000.npz'.format(self.opt.sr//1000, self.opt.split)), allow_pickle=True);
        self.testX = torch.tensor(np.moveaxis(data['x'], 3, 1)).to(self.opt.device);
        self.testY = torch.tensor(data['y']).to(self.opt.device);

    def __validate(self, net, lossFunc):
        if self.testX is None:
            self.load_test_data();

        net.eval();
        with torch.no_grad():
            y_pred = None;
            batch_size = (self.opt.batchSize//self.opt.nCrops)*self.opt.nCrops;
            for idx in range(math.ceil(len(self.testX)/batch_size)):
                x = self.testX[idx*batch_size : (idx+1)*batch_size];
                scores = net(x);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

            acc, loss = self.__compute_accuracy(y_pred, self.testY, lossFunc);
        net.train();
        return acc, loss;

    #Calculating average prediction (10 crops) and final accuracy
    def __compute_accuracy(self, y_pred, y_target, lossFunc):
        with torch.no_grad():
            #Reshape to shape theme like each sample comtains 10 samples, calculate mean and find theindices that has highest average value for each sample
            if self.opt.nCrops == 1:
                y_pred = y_pred.argmax(dim=1);
                y_target = y_target.argmax(dim=1);
            else:
                y_pred = (y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(dim=1).argmax(dim=1);
                y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(dim=1).argmax(dim=1);
            acc = (((y_pred==y_target)*1).float().mean()*100).item();
            # valLossFunc = torch.nn.KLDivLoss();
            loss = lossFunc(y_pred.float().log(), y_target.float()).item();
            # loss = 0.0;
        return acc, loss;

    def TestModel(self, run=1):
        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        dir = os.getcwd();
        net_path = self.opt.model_path;
        print(net_path)
        file_paths = glob.glob(net_path);
        for f in file_paths:
            state = torch.load(f, map_location=self.opt.device);
            config = state['config'];
            weight = state['weight'];
            net = models.GetACDNetModel(self.opt.inputLength, 50, self.opt.sr, config).to(self.opt.device);
            net.load_state_dict(weight);
            print('Model found at: {}'.format(f));
            # calc.summary(net, (1,1,self.opt.inputLength));
            self.load_test_data();
            net.eval();
            #Test standard way with 10 crops
            val_acc, val_loss = self.__validate(net, lossFunc);
            print('Testing - Val: Loss {:.3f}  Acc(top1) {:.2f}%'.format(val_loss, val_acc));


if __name__ == '__main__':
    opt = opts.parse();
    valid_path = False;
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    while not valid_path:
        model_path = input("Enter model path\n:");
        file_paths = glob.glob(os.path.join(os.getcwd(), model_path));
        print(file_paths);
        if len(file_paths)>0 and os.path.isfile(file_paths[0]):
            state = torch.load(file_paths[0], map_location=opt.device);
            opt.model_path = file_paths[0];
            print('Model has been found at: {}'.format(opt.model_path));
            valid_path = True;

    valid_fold = False;
    while not valid_fold:
        fold = input("Select the fold on which the model was Validated:\n 1. Fold-1\n 2. Fold-2\n 3. Fold-3\n 4. Fold-4\n 5. Fold-5\n :")
        if fold in ['1','2','3','4','5']:
            opt.split = int(fold);
            valid_fold = True;
    trainer = Trainer(opt);
    trainer.TestModel();
