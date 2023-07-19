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
sys.path.append(os.path.join(os.getcwd(), 'torch/resources'));
import common.utils as U;
import common.opts as opt;
import resources.models as models;
import resources.calculator as calc;
import resources.train_generator as train_generator;
import resources.pruning_tools.weight_pruning as weight_pruner;

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

class PruningTrainer:
    def __init__(self, opt):
        self.opt = opt;
        #Conditional compression settings
        self.opt.LR = 0.01;
        self.opt.schedule = [0.15, 0.30, 0.45, 0.60, 0.75];
        self.opt.warmup = 0;
        self.opt.prune_ratio = 0.95;
        self.opt.prune_algo = 'l0norm';
        self.opt.prune_interval = 1;
        self.opt.nEpochs = 500;

        self.testX = None;
        self.testY = None;
        self.bestAcc = 0.0;
        self.bestAccEpoch = 0;
        self.trainGen = train_generator.setup(self.opt, self.opt.split);
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        self.start_time = time.time();

    def PruneAndTrain(self):
        self.load_test_data();
        print(self.device);
        loss_func = torch.nn.KLDivLoss(reduction='batchmean');

        #Load saved model dict
        dir = os.getcwd();
        net = models.GetACDNetModel().to(self.device);
        file_paths = glob.glob(self.opt.model_path);
        if len(file_paths)>0 and os.path.isfile(file_paths[0]):
            net.load_state_dict(torch.load(file_paths[0], map_location=self.device)['weight']);
            print('Model Loaded from: {}'.format(file_paths[0]));
        else:
            print('Model is not found at: {}'.format(net_path));
            exit();

        calc.summary(net, (1,1,self.opt.inputLength))
        net.eval();
        val_acc, val_loss = self.__validate(net, loss_func);
        print('Testing - Val: Loss {:.3f}  Acc(top1) {:.3f}%'.format(val_loss, val_acc));
        net.train();

        optimizer = optim.SGD(net.parameters(), lr=self.opt.LR, weight_decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True)

        weight_name = ["weight"]# if not self.opt.factorize else ["weightA", "weightB", "weightC"]
        layers_n = weight_pruner.layers_n(net, param_name=["weight"])[1];
        all_num = sum(layers_n.values());
        print("\t TOTAL PRUNABLE PARAMS: {}".format(all_num));
        print("\t PRUNE RATIO :{}".format(self.opt.prune_ratio));
        sparse_factor = int(all_num * (1-self.opt.prune_ratio));
        print("\t SPARSE FACTOR: {}".format(sparse_factor));
        model_size = (sparse_factor * 4)/1024**2;
        print("\t MODEL SIZE: {:.2f} MB".format(model_size));
        prune_algo = getattr(weight_pruner, self.opt.prune_algo);
        prune_func = lambda m: prune_algo(m, sparse_factor, param_name=weight_name);

        for epoch_idx in range(self.opt.nEpochs):
            epoch_start_time = time.time();
            optimizer.param_groups[0]['lr'] = self.__get_lr(epoch_idx+1);
            cur_lr = optimizer.param_groups[0]['lr'];
            running_loss = 0.0;
            running_acc = 0.0;
            n_batches = math.ceil(len(self.trainGen.data)/self.opt.batchSize);
            net.train();
            for batch_idx in range(n_batches):
                # with torch.no_grad():
                x,y = self.trainGen.__getitem__(batch_idx)
                x = torch.tensor(np.moveaxis(x, 3, 1)).to(self.device);
                y = torch.tensor(y).to(self.device);

                # zero the parameter gradients
                optimizer.zero_grad();

                # forward + backward + optimize
                outputs = net(x);
                running_acc += (((outputs.data.argmax(dim=1) == y.argmax(dim=1))*1).float().mean()).item();
                loss = loss_func(outputs.log(), y);

                loss.backward();
                optimizer.step();

                running_loss += loss.item();

                with torch.no_grad():
                    prune_func(net);

            prune_func(net)

            tr_acc = (running_acc / n_batches)*100;
            tr_loss = running_loss / n_batches;

            #Epoch wise validation Validation
            epoch_train_time = time.time() - epoch_start_time;
            net.eval();
            val_acc, val_loss = self.__validate(net, loss_func);
            #Save best model
            self.__save_model(val_acc, epoch_idx, net);

            self.__on_epoch_end(epoch_start_time, epoch_train_time, epoch_idx, cur_lr, tr_loss, tr_acc, val_loss, val_acc);

            running_loss = 0;
            running_acc = 0;
            net.train();

        total_time_taken = time.time() - self.start_time;
        print("Execution finished in: {}".format(U.to_hms(total_time_taken)));

    def load_test_data(self):
        if(self.testX is None):
            data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'test_data_20khz/fold{}_test4000.npz'.format(self.opt.split)), allow_pickle=True);
            self.testX = torch.tensor(np.moveaxis(data['x'], 3, 1)).to(self.device);
            self.testY = torch.tensor(data['y']).to(self.device).to(self.device);

    def __get_lr(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule]);
        decay = sum(epoch > divide_epoch);
        if epoch <= self.opt.warmup:
            decay = 1;
        return self.opt.LR * np.power(0.1, decay);

    def __validate(self, net, lossFunc):
        with torch.no_grad():
            y_pred = None;
            batch_size = (self.opt.batchSize//self.opt.nCrops)*self.opt.nCrops
            for idx in range(math.ceil(len(self.testX)/batch_size)):
                x = self.testX[idx*batch_size : (idx+1)*batch_size];
                scores = net(x);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

            acc, loss = self.__compute_accuracy(y_pred, self.testY, lossFunc);
        return acc, loss;

    #Calculating average prediction (10 crops) and final accuracy
    def __compute_accuracy(self, y_pred, y_target, lossFunc):
        with torch.no_grad():
            #Reshape to shape theme like each sample comtains 10 samples, calculate mean and find theindices that has highest average value for each sample
            y_pred = (y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(dim=1).argmax(dim=1);
            y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(dim=1).argmax(dim=1);
            acc = (((y_pred==y_target)*1).float().mean()*100).item();
            # valLossFunc = torch.nn.KLDivLoss();
            loss = lossFunc(y_pred.float().log(), y_target.float()).item();
            # loss = 0.0;
        return acc, loss;

    def __on_epoch_end(self, epoch_start_time, train_time, epochIdx, lr, tr_loss, tr_acc, val_loss, val_acc):
        epoch_time = time.time() - epoch_start_time;
        val_time = epoch_time - train_time;
        total_time = time.time() - self.start_time;
        line = '{} Epoch: {}/{} | Time: {} (Train {}  Val {}) | Train: LR {}  Loss {:.2f}  Acc {:.2f}% | Val: Loss {:.2f}  Acc(top1) {:.2f}% | HA {:.2f}@{}\n'.format(
            U.to_hms(total_time), epochIdx+1, self.opt.nEpochs, U.to_hms(epoch_time), U.to_hms(train_time), U.to_hms(val_time),
            lr, tr_loss, tr_acc, val_loss, val_acc, self.bestAcc, self.bestAccEpoch);
        # print(line)
        sys.stdout.write(line);
        sys.stdout.flush();

    def __save_model(self, acc, epochIdx, net):
        if acc > self.bestAcc:
            dir = os.getcwd();
            fname = "{}/torch/pruned_models/{}.pt";
            old_model = fname.format(dir, self.opt.model_name.lower());
            if os.path.isfile(old_model):
                os.remove(old_model);
            self.bestAcc = acc;
            self.bestAccEpoch = epochIdx +1;
            torch.save({'weight':net.state_dict(), 'config':net.ch_config}, fname.format(dir, self.opt.model_name.lower()));
