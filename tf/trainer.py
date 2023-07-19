import sys;
import os;
import glob;
import math;
import numpy as np;
import time;
from tensorflow import keras;

sys.path.append(os.getcwd());
sys.path.append(os.path.join(os.getcwd(), 'common'));
import common.utils as U;
import common.opts as opts;
import resources.models as models;
import resources.train_generator as train_generator;

#Every run is producing very close accuracy.
#So, avoiding the reproducibility.
#For reproducibility in tensorflow you need to set operation level seed in addition to graph-level seed.
#Look at: https://stackoverflow.com/questions/38469632/tensorflow-non-repeatable-results

class Trainer:
    def __init__(self, opt=None):
        self.opt = opt;
        self.trainGen = train_generator.setup(self.opt, self.opt.split);

    def Train(self):
        model = models.GetAcdnetModel(self.opt.inputLength, 50, self.opt.sr, ch_config = self.opt.model_config);
        model.summary();

        loss = 'kullback_leibler_divergence';
        optimizer = keras.optimizers.SGD(learning_rate=self.opt.LR, weight_decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True)

        model.compile(loss=loss, optimizer=optimizer , metrics=['accuracy']);

        # learning schedule callback
        lrate = keras.callbacks.LearningRateScheduler(self.GetLR);
        best_model = keras.callbacks.ModelCheckpoint('tf/trained_models/{}.h5'.format(self.opt.model_name), monitor='val_acc', save_best_only=True, verbose=0);
        custom_evaluator = CustomCallback(self.opt);
        callbacks_list = [lrate, custom_evaluator, best_model];

        model.fit(self.trainGen, epochs=self.opt.nEpochs, steps_per_epoch=len(self.trainGen.data)//self.trainGen.batch_size, callbacks=callbacks_list, verbose=0);

    def GetLR(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule]);
        decay = sum(epoch > divide_epoch);
        if epoch <= self.opt.warmup:
            decay = 1;
        return self.opt.LR * np.power(0.1, decay);

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, opt):
        self.opt = opt;
        self.testX = None;
        self.testY = None;
        self.curEpoch = 0;
        self.curLr = opt.LR;
        self.cur_epoch_start_time = time.time();
        self.bestAcc = 0.0;
        self.bestAccEpoch = 0;

    def on_epoch_begin(self, epoch, logs=None):
        self.curEpoch = epoch+1;
        self.curLr = Trainer(self.opt).GetLR(epoch+1);
        self.cur_epoch_start_time = time.time();

    def on_epoch_end(self, epoch, logs=None):
        train_time = time.time() - self.cur_epoch_start_time;
        self.load_test_data();
        val_acc, val_loss = self.validate(self.model);
        logs['val_acc'] = val_acc;
        logs['val_loss'] = val_loss;
        if val_acc > self.bestAcc:
            self.bestAcc = val_acc;
            self.bestAccEpoch = epoch + 1;
        epoch_time = time.time() - self.cur_epoch_start_time;
        val_time = epoch_time - train_time;
        # print(logs);
        line = 'SP-{}, Epoch: {}/{} | Time: {} (Train {}  Val {}) | Train: LR {}  Loss {:.2f}  Acc {:.2f}% | Val: Loss {:.2f}  Acc(top1) {:.2f}% | HA {:.2f}@{}\n'.format(
            self.opt.split, epoch+1, self.opt.nEpochs, U.to_hms(epoch_time), U.to_hms(train_time), U.to_hms(val_time),
            self.curLr, logs['loss'], logs['accuracy'] if 'accuracy' in logs else logs['acc'], val_loss, val_acc, self.bestAcc, self.bestAccEpoch);
        # print(line)
        sys.stdout.write(line);
        sys.stdout.flush();

    def load_test_data(self):
        if self.testX is None:
            data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'test_data_20khz/fold{}_test4000.npz'.format(self.opt.split)), allow_pickle=True);
            self.testX = data['x'];
            self.testY = data['y'];

    def validate(self, model):
        y_pred = None;
        y_target = None;
        batch_size = (self.opt.batchSize//self.opt.nCrops)*self.opt.nCrops;
        for batchIndex in range(math.ceil(len(self.testX) / batch_size)):
            x = self.testX[batchIndex*batch_size : (batchIndex+1)*batch_size];
            y = self.testY[batchIndex*batch_size : (batchIndex+1)*batch_size];
            scores = model.predict(x, batch_size=len(y), verbose=0);
            y_pred = scores if y_pred is None else np.concatenate((y_pred, scores));
            y_target = y if y_target is None else np.concatenate((y_target, y));
            #break;

        acc, loss = self.compute_accuracy(y_pred, y_target);
        return acc, loss;

    #Calculating average prediction (10 crops) and final accuracy
    def compute_accuracy(self, y_pred, y_target):
        #Reshape y_pred to shape it like each sample comtains 10 samples.
        if self.opt.nCrops > 1:
            y_pred = (y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(axis=1);
            y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(axis=1);

        loss = keras.losses.KLD(y_target, y_pred).numpy().mean();

        #Get the indices that has highest average value for each sample
        y_pred = y_pred.argmax(axis=1);
        y_target = y_target.argmax(axis=1);
        accuracy = (y_pred==y_target).mean()*100;

        return accuracy, loss;


if __name__ == '__main__':
    opt = opts.parse();
    opt.sr = 20000;
    opt.inputLength = 30225;
    import torch;
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    valid_path = False;
    while not valid_path:
        model_path = input("Enter pytorch model path to be re-built and trained in Tensorflow\n:");
        file_paths = glob.glob(os.path.join(os.getcwd(), model_path));
        if len(file_paths)>0 and os.path.isfile(file_paths[0]):
            state = torch.load(file_paths[0], map_location=opt.device);
            opt.model_config = state['config'];
            opt.model_path = file_paths[0];
            print('Model has been found at: {}'.format(opt.model_path));
            valid_path = True;

    valid_model_name = False;
    while not valid_model_name:
        model_name = input('Enter a name that will be used to save the trained model: ');
        if model_name != '':
            opt.model_name = model_name;
            valid_model_name = True;

    valid_fold = False;
    while not valid_fold:
        fold = input("Which fold do you want your model to be Validated:\n 1. Fold-1\n 2. Fold-2\n 3. Fold-3\n 4. Fold-4\n 5. Fold-5\n :")
        if fold in ['1','2','3','4','5']:
            opt.split = int(fold);
            valid_fold = True;

    trainer = Trainer(opt);
    trainer.Train();
