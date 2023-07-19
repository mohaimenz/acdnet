import sys;
import os;
import torch;
import numpy as np;
import torch.optim as optim;
from operator import itemgetter;
from heapq import nsmallest;
import time;
import glob;
import math;
import random;

sys.path.append(os.getcwd());
sys.path.append(os.path.join(os.getcwd(), 'common'));
sys.path.append(os.path.join(os.getcwd(), 'torch/resources'));
import common.utils as U;
import common.opts as opt;
import resources.models as models;
import resources.calculator as calc;
import resources.train_generator as train_generator;
from resources.pruning_tools import filter_pruning, filter_pruner;

#Reproducibility
seed = 42;
random.seed(seed);
np.random.seed(seed);
torch.manual_seed(seed);
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed);
torch.backends.cudnn.deterministic = True;
torch.backends.cudnn.benchmark = False;

class PruningTrainer:
    def __init__(self, opt):
        self.opt = opt;
        self.opt.channels_to_prune_per_iteration = 1;
        self.opt.finetune_epoch_per_iteration = 2;
        self.opt.lr=0.001;
        self.opt.schedule = [0.5, 0.8];
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        self.opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        self.pruner = None;
        self.iterations = 0;
        self.cur_acc = 0.0;
        self.cur_iter = 1;
        self.cur_lr = self.opt.lr;
        self.net = None;
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean');
        self.trainGen = train_generator.setup(self.opt, self.opt.split);
        self.testX = None;
        self.testY = None;
        self.load_test_data();

    def PruneAndTrain(self):
        dir = os.getcwd();
        self.net = models.GetACDNetModel().to(self.opt.device);
        state = torch.load(self.opt.model_path, map_location=self.opt.device);
        self.net.load_state_dict(state['weight']);
        self.pruner = filter_pruning.Magnitude(self.net, self.opt) if self.opt.prune_type == 1 else filter_pruning.Taylor(self.net, self.opt);
        self.validate();
        calc.summary(self.net, (1, 1, self.opt.inputLength), brief=False); # shape of one sample for inferenceing
        # exit();
        #Make sure all the layers are trainable
        for param in self.net.parameters():
            param.requires_grad = True
        self.iterations = self.estimate_pruning_iterations();
        # exit();
        for i in range(1, self.iterations):
            self.cur_iter = i;
            iter_start = time.time();
            print("\nIteration {} of {} starts..".format(i, self.iterations-1), flush=True);
            print("Ranking channels.. ", flush=True);
            prune_targets = self.get_candidates_to_prune(self.opt.channels_to_prune_per_iteration);
            # prune_targets = [(40,3)];
            print("Pruning channels: {}".format(prune_targets), flush=True);
            self.net = filter_pruner.prune_layers(self.net, prune_targets, self.opt.prune_all, self.opt.device);
            calc.summary(self.net, (1, 1, self.opt.inputLength), brief=True); # shape of one sample for inferenceing
            self.validate();
            print("Fine tuning {} epochs to recover from prunning iteration.".format(self.opt.finetune_epoch_per_iteration), flush=True);

            if self.cur_iter in list(map(int, np.array(self.iterations)*self.opt.schedule)):
                self.cur_lr *= 0.1;
            optimizer = optim.SGD(self.net.parameters(), lr=self.cur_lr, momentum=0.9);
            self.train(optimizer, epoches = self.opt.finetune_epoch_per_iteration);
            print("Iteration {}/{} finished in {}".format(self.cur_iter, self.iterations+1, U.to_hms(time.time()-iter_start)), flush=True);
            print("Total channels prunned so far: {}".format(i*self.opt.channels_to_prune_per_iteration), flush=True);
            self.__save_model(self.net);

        calc.summary(self.net, (1, 1, self.opt.inputLength)); # shape of one sample for inferenceing
        self.__save_model(self.net);

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.pruner.reset();
        if self.opt.prune_type == 1:
            self.pruner.compute_filter_magnitude();
        else:
            self.train_epoch(rank_filters = True);
            self.pruner.normalize_ranks_per_layer();

        return self.pruner.get_prunning_plan(num_filters_to_prune);

    def estimate_pruning_iterations(self):
        # get total number of variables from all conv2d featuremaps
        prunable_count = sum(self.get_channel_list(self.opt.prune_all));
        total_count= sum(self.get_channel_list());
        #iterations_reqired = int((prunable_count * self.opt.prune_ratio) / self.opt.channels_to_prune_per_iteration);
        #prune_ratio works with the total number of channels, not only with the prunable channels. i.e. 80% or total will be pruned from total or from only features
        iterations_reqired = int((total_count * self.opt.prune_ratio) / self.opt.channels_to_prune_per_iteration);
        print('Total Channels: {}, Prunable: {}, Non-Prunable: {}'.format(total_count, prunable_count, total_count - prunable_count), flush=True);
        print('No. of Channels to prune per iteration: {}'.format(self.opt.channels_to_prune_per_iteration), flush=True);
        print('Total Channels to prune ({}%): {}'.format(int(self.opt.prune_ratio*100), int(total_count * self.opt.prune_ratio)-1), flush=True);
        print('Total iterations required: {}'.format(iterations_reqired-1), flush=True);
        return iterations_reqired;

    def get_channel_list(self, prune_all=True):
        ch_conf = [];
        if prune_all:
            for name, module in enumerate(self.net.sfeb):
                if issubclass(type(module), torch.nn.Conv2d):
                    ch_conf.append(module.out_channels);

        for name, module in enumerate(self.net.tfeb):
            if issubclass(type(module), torch.nn.Conv2d):
                ch_conf.append(module.out_channels);

        return ch_conf;

    def load_test_data(self):
        if(self.testX is None):
            data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'test_data_20khz/fold{}_test4000.npz'.format(self.opt.split)), allow_pickle=True);
            self.testX = torch.tensor(np.moveaxis(data['x'], 3, 1)).to(self.opt.device);
            self.testY = torch.tensor(data['y']).to(self.opt.device);

    #Calculating average prediction (10 crops) and final accuracy
    def compute_accuracy(self, y_pred, y_target):
        with torch.no_grad():
            #Reshape to shape theme like each sample comtains 10 samples, calculate mean and find the indices that has highest average value for each sample
            y_pred = (y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(dim=1).argmax(dim=1);
            y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(dim=1).argmax(dim=1);
            acc = (((y_pred==y_target)*1).float().mean()*100).item();
            # valLossFunc = torch.nn.KLDivLoss();
            loss = self.criterion(y_pred.float().log(), y_target.float()).item();
            # loss = 0.0;
        return acc, loss;

    def train(self, optimizer = None, epoches=10):
        for i in range(epoches):
            # print("Epoch: ", i);
            self.train_epoch(optimizer);
            self.validate();
        print("Finished fine tuning.", flush=True);

    def train_batch(self, optimizer, batch, label, rank_filters):
        self.net.zero_grad()
        if rank_filters:
            output = self.pruner.forward(batch);
            self.criterion(output.log(), label).backward();
        else:
            self.criterion(self.net(batch), label).backward();
            optimizer.step();

    def train_epoch(self, optimizer = None, rank_filters = False):
        if rank_filters is False and optimizer is None:
            print('Please provide optimizer to train_epoch', flush=True);
            exit();
        n_batches = math.ceil(len(self.trainGen.data)/self.opt.batchSize);
        for b_idx in range(n_batches):
            x,y = self.trainGen.__getitem__(b_idx)
            x = torch.tensor(np.moveaxis(x, 3, 1)).to(self.opt.device);
            y = torch.tensor(y).to(self.opt.device);
            self.train_batch(optimizer, x, y, rank_filters);

    def validate(self):
        self.net.eval();
        with torch.no_grad():
            y_pred = None;
            batch_size = (self.opt.batchSize//self.opt.nCrops)*self.opt.nCrops;
            for idx in range(math.ceil(len(self.testX)/batch_size)):
                x = self.testX[idx*batch_size : (idx+1)*batch_size];
                scores = self.net(x);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

            acc, loss = self.compute_accuracy(y_pred, self.testY);
        print('Current Testing Performance - Val: Loss {:.3f}  Acc(top1) {:.3f}%'.format(loss, acc), flush=True);
        self.cur_acc = acc;
        self.net.train();
        return acc, loss;

    def __save_model(self, net):
        net.ch_config = self.get_channel_list();
        dir = os.getcwd();
        fname = "{}/torch/pruned_models/{}.pt";
        old_model = fname.format(dir, self.opt.model_name.lower());
        if os.path.isfile(old_model):
            os.remove(old_model);
        torch.save({'weight':net.state_dict(), 'config':net.ch_config}, fname.format(dir, self.opt.model_name.lower()));
