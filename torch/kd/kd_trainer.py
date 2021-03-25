import sys;
import os;
import glob;
import math;
import numpy as np;
import glob;
import random;
import time;
import torch;
import torch.nn.functional as F
import torch.optim as optim;

# print(os.path.join(os.getcwd(), 'torch/resources'));
sys.path.append(os.getcwd());
sys.path.append(os.path.join(os.getcwd(), 'common'));
sys.path.append(os.path.join(os.getcwd(), 'torch'));

import common.utils as U;
import common.opts as opts;
import resources.models as models;
import resources.calculator as calc;
import resources.train_generator as train_generator;

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

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Trainer:
    def __init__(self, opt=None):
        self.opt = opt;
        self.testX = None;
        self.testY = None;
        self.bestAcc = 0.0;
        self.bestAccEpoch = 0;
        self.trainGen = train_generator.setup(opt, True);

    def Train(self):
        train_start_time = time.time();
        teacher = None;
        student = None;
        if self.opt.teacher_path != '' and self.opt.student_path != '':
            teacher_path = self.opt.teacher_path;
            tpaths = glob.glob(teacher_path);
            if len(tpaths)>0 and os.path.isfile(tpaths[0]):
                state = torch.load(tpaths[0], map_location=self.opt.device);
                teacher = models.GetACDNetModelV2(self.opt.inputLength, 50, self.opt.sr, channel_config=state['config']).to(self.opt.device);
                teacher.load_state_dict(state['weight']);
                print('Teacher Model Loaded');
            else:
                print('Teacher Model has not been found');
                exit();

            student_path = self.opt.student_path;
            spaths = glob.glob(student_path);
            if len(spaths)>0 and os.path.isfile(spaths[0]):
                state = torch.load(spaths[0], map_location=self.opt.device);
                student = models.GetACDNetModelV2(self.opt.inputLength, 50, self.opt.sr, channel_config=state['config']).to(self.opt.device);
                print('Student Model Loaded');
            else:
                print('Student Model has not been found');
                exit();
        else:
            print('Teacher and Student paths are missing');
            exit();

        no_softmax = torch.nn.Sequential(
            Identity()
        );
        teacher.output = no_softmax;
        student.output = no_softmax;

        calc.summary(teacher, (1,1,opt.inputLength));
        calc.summary(student, (1,1,opt.inputLength));

        training_text = "KD Training";
        print("{} has been started. You will see update after finishing every training epoch and validation".format(training_text));

        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        optimizer = optim.SGD(student.parameters(), lr=self.opt.LR, weight_decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True);

        # teacher.eval();
        # val_acc, val_loss = self.__validate(teacher, lossFunc);
        # print('Accuracy: {:.2f}'.format(val_acc));
        # exit();

        # self.opt.nEpochs = 1957 if self.opt.split == 4 else 2000;
        teacher.eval();
        for epochIdx in range(self.opt.nEpochs):
            epoch_start_time = time.time();
            optimizer.param_groups[0]['lr'] = self.__get_lr(epochIdx+1);
            cur_lr = optimizer.param_groups[0]['lr'];
            running_loss = 0.0;
            running_acc = 0.0;
            self.trainGen.load_data(epochIdx+1);
            n_batches = math.ceil(len(self.trainGen.data)/self.opt.batchSize);
            for batchIdx in range(n_batches):
                # with torch.no_grad():
                x,y = self.trainGen.__get_batch__(batchIdx)
                x = torch.tensor(np.moveaxis(x, 3, 1)).to(self.opt.device);
                y = torch.tensor(y).to(self.opt.device);
                optimizer.zero_grad();

                teacher_output = teacher(x);
                student_output = student(x);
                running_acc += (((student_output.data.argmax(dim=1) == y.argmax(dim=1))*1).float().mean()).item();
                # loss = lossFunc(outputs.log(), y);
                loss = self.kd_loss(student_output, y, teacher_output);
                loss.backward();
                optimizer.step();

                running_loss += loss.item();

            tr_acc = (running_acc / n_batches)*100;
            tr_loss = running_loss / n_batches;

            #Epoch wise validation Validation
            epoch_train_time = time.time() - epoch_start_time;

            student.eval();
            student.output = torch.nn.Sequential(torch.nn.Softmax(dim=1));
            val_acc, val_loss = self.__validate(student, lossFunc);
            #Save best model
            self.__save_model(val_acc, epochIdx, student);
            self.__on_epoch_end(epoch_start_time, epoch_train_time, epochIdx, cur_lr, tr_loss, tr_acc, val_loss, val_acc);
            student.output = no_softmax;
            running_loss = 0;
            running_acc = 0;
            student.train();

        total_time_taken = time.time() - train_start_time;
        print("Execution finished in: {}".format(U.to_hms(total_time_taken)));


    def kd_loss(self, student_output, targets, teacher_output):
        #Source1: (Section 2 and its last paragraph is very important) - https://arxiv.org/pdf/1503.02531.pdf
        #Source2: https://medium.com/analytics-vidhya/knowledge-distillation-dark-knowledge-of-neural-network-9c1dfb418e6a
        #Source3: https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
        # When the student model is very small compared to the teacher model, lower temperatures(T) works better.
        # Because as we raise the temperature ,the resulting outputs will be richer in information, a small model cannot be able to capture all the informations.
        T = self.opt.temperature;
        # Hinton et al. observed better results when keeping α value much smaller than β (here 1-alphs);
        alpha = 0.1;
        kd_loss = 0.0;
        #KD loss should be: loss(student_output/T, teacher_outputs/T)* (1-alpha)*T*T  + loss(student_output, targets) * alpha
        # print(type(self.opt.kdLossType));
        if self.opt.kdLossType == 1:
            # print('KDLoss 1');
            # exit();
            kd_loss = F.cross_entropy(student_output/T, torch.max(F.softmax(teacher_output/T, dim=1), 1)[1]) * (1. - alpha) * T * T + \
                  F.cross_entropy(student_output, torch.max(targets,1)[1]) * alpha;
        elif self.opt.kdLossType == 2:
            # print('KDLoss 2');
            # exit();
            kd_loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_output/T, dim=1), F.softmax(teacher_output/T, dim=1)) * (1. - alpha) * T * T + \
                  torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_output, dim=1), targets) * alpha;
        else:
            # print('KDLoss 3');
            # exit();
            kd_loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_output/T, dim=1), F.softmax(teacher_output/T, dim=1)) * (1. - alpha) * T * T + \
                  F.cross_entropy(student_output, torch.max(targets,1)[1]) * alpha;

        return kd_loss

    def load_test_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'test_data_{}khz/fold{}_test4000.npz'.format(self.opt.sr//1000, self.opt.split)), allow_pickle=True);
        self.testX = torch.tensor(np.moveaxis(data['x'], 3, 1)).to(self.opt.device);
        self.testY = torch.tensor(data['y']).to(self.opt.device);

    def __get_lr(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule]);
        decay = sum(epoch > divide_epoch);
        if epoch <= self.opt.warmup:
            decay = 1;
        return self.opt.LR * np.power(0.1, decay);

    def __get_batch(self, index):
        x = self.trainX[index*self.opt.batchSize : (index+1)*self.opt.batchSize];
        y = self.trainY[index*self.opt.batchSize : (index+1)*self.opt.batchSize];
        return x.to(self.opt.device), y.to(self.opt.device);

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

    def __on_epoch_end(self, start_time, train_time, epochIdx, lr, tr_loss, tr_acc, val_loss, val_acc):
        epoch_time = time.time() - start_time;
        val_time = epoch_time - train_time;
        line = 'KDL{}-T{}-F{} Epoch: {}/{} | Time: {} (Train {}  Val {}) | Train: LR {}  Loss {:.2f}  Acc {:.2f}% | Val: Loss {:.2f}  Acc(top1) {:.2f}% | HA {:.2f}@{}\n'.format(
            self.opt.kdLossType, self.opt.temperature, self.opt.split, epochIdx+1, self.opt.nEpochs, U.to_hms(epoch_time), U.to_hms(train_time), U.to_hms(val_time),
            lr, tr_loss, tr_acc, val_loss, val_acc, self.bestAcc, self.bestAccEpoch);
        # print(line)
        sys.stdout.write(line);
        sys.stdout.flush();

    def __save_model(self, acc, epochIdx, net):
        if acc > self.bestAcc:
            dir = os.getcwd();
            fname = "{}/torch/trained_models/kd/{}_a{:.2f}_e{}.pt";
            old_model = fname.format(dir, self.opt.model_name.lower(), self.bestAcc, self.bestAccEpoch);
            if os.path.isfile(old_model):
                os.remove(old_model);
            self.bestAcc = acc;
            self.bestAccEpoch = epochIdx +1;
            torch.save({'weight':net.state_dict(), 'config':net.ch_config}, fname.format(dir, self.opt.model_name.lower(), self.bestAcc, self.bestAccEpoch));

def Train(opt):
    print('Starting {} model Training'.format(opt.model_name.upper(), opt.split));
    opts.display_info(opt);
    trainer = Trainer(opt);
    trainer.Train();

if __name__ == '__main__':
    opt = opts.parse();
    opt.kdLossType = int(opt.kdLossType);
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    opt.sr = 20000;
    opt.inputLength = 30225;
    #Update the student model (Compressed re-initialized) model paths according your file names
    tpaths = {1:'20khz_sp1_base_model_85.75.pt', 2:'20khz_sp2_base_model_85.75.pt', 3:'20khz_sp3_base_model_88.75.pt', 4:'20khz_sp4_base_model_91.00.pt', 5:'20khz_sp5_base_model_85.00.pt'};
    opt.student_path = 'torch/pruned_models/v2_hybrid_tay_full_prune_1659.pt';
    t_start,f_start = eval(opt.start_at);
    t_stop, f_stop = eval(opt.stop_after);
    # print(t_start, ' ', f_start);
    # exit();
    for t in range(t_start, t_stop+1):
        opt.temperature = t;
        f = f_start if t == t_start else 1;
        for s in range(f, f_stop+1):
            opt.split = s;
            opt.teacher_path = 'torch/resources/pretrained_models/20khz/{}'.format(tpaths[s]);
            opt.model_name = 'KDL{}_T{}_F{}_Trained'.format(opt.kdLossType, t, s);
            Train(opt);
