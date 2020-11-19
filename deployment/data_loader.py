import sys;
import os;
import glob;
import math;
import numpy as np;

cwd_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(cwd_path)
common_path = os.path.join(parent_path,'common')

sys.path.append(cwd_path);
sys.path.append(parent_path);
sys.path.append(common_path);

import common.opts as opts;
import tf.resources.train_generator as train_generator;


class Trainer:
    def __init__(self, opt=None):
        self.opt = opt;
        self.trainX = None;
        self.trainY = None;
        self.testX = None;
        self.testY = None;

    def load_training_data(self):
        if self.trainX is None:
            print('Loading training/calibration data');
            #Loading all the 1600 samples to train one epoch at once.
            self.opt.sr = 20000;
            self.opt.inputLength = 30225;
            self.opt.batchSize = 1600;
            trainGen = train_generator.setup(self.opt, self.opt.split);
            self.trainX, self.trainY = trainGen.__getitem__(0);
            print('Done');

            #Revert back the batch size settings to its original state
            self.opt.batchSize = 64;

    def load_test_data(self):
        if self.testX is None:
            print('Loading test data');
            data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'test_data_20khz/fold{}_test4000.npz'.format(self.opt.split)), allow_pickle=True);
            self.testX = data['x'];
            self.testY = data['y'];
            print('Done');


if __name__ == '__main__':
    opt = opts.parse();
    valid_fold = False;
    while not valid_fold:
        fold = input("Which fold do you want your model to be calibrated and validated during quantization (Just enter the fold number):\n 1. Fold-1\n 2. Fold-2\n 3. Fold-3\n 4. Fold-4\n 5. Fold-5\n :")
        if fold in ['1','2','3','4','5']:
            opt.split = int(fold);
            valid_fold = True;

    trainer = Trainer(opt);
    trainer.load_training_data();
    trainer.load_test_data();
