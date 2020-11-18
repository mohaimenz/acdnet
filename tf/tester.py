import os;
import sys;
import glob;
import numpy as np;
import math;
from tensorflow import keras;
import tensorflow as tf;

sys.path.append(os.getcwd());
sys.path.append(os.path.join(os.getcwd(), 'common'));
import common.opts as opts;

class Tester:
    def __init__(self, opt=None):
        self.opt = opt;
        self.testX = None;
        self.testY = None;

    # Loading Test data
    def load_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'test_data_20khz/fold{}_test4000.npz'.format(self.opt.split)), allow_pickle=True);
        self.testX = data['x'];
        self.testY = data['y'];
        print(self.testX.shape);
        print(self.testY.shape);

    #Test the model with test data
    def validate(self, model):
        y_pred = None;
        y_target = self.testY;
        batch_size = (self.opt.batchSize//self.opt.nCrops)*self.opt.nCrops;
        for idx in range(math.ceil(len(self.testX)/batch_size)):
            x = self.testX[idx*batch_size : (idx+1)*batch_size];
            scores = model.predict(x, batch_size=len(x), verbose=0);
            y_pred = scores if y_pred is None else np.concatenate((y_pred, scores));

        acc, loss = self.compute_accuracy(y_pred, y_target);
        return acc, loss;

    #Calculating average prediction (10 crops) and final accuracy
    def compute_accuracy(self, y_pred, y_target):
        #Reshape y_pred to shape it like each sample comtains 10 samples.
        y_pred = y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1]);
        y_target = y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])

        #Calculate the average of class predictions for 10 crops of a sample
        y_pred = np.mean(y_pred, axis=1);
        y_target = np.mean(y_target,axis=1);

        loss = keras.losses.KLD(y_target, y_pred).numpy().mean();

        #Get the indices that has highest average value for each sample
        y_pred = y_pred.argmax(axis=1);
        y_target = y_target.argmax(axis=1);
        accuracy = (y_pred==y_target).mean()*100;

        return accuracy, loss;

    #Load and test the saved model
    def TestModel(self):
        model = keras.models.load_model(self.opt.model_path);
        print('Model Loaded.');
        model.summary();
        self.load_data();
        print('Test dataset loaded.');
        val_acc, val_loss = self.validate(model);
        print('Testing - Val: Loss {:.2f}  Acc(top1) {:.2f}%'.format(val_loss, val_acc));

if __name__ == '__main__':
    opt = opts.parse();
    opt.sr = 20000;
    opt.inputLength = 30225;

    valid_path = False;
    while not valid_path:
        model_path = input("Enter model path\n:");
        file_paths = glob.glob(os.path.join(os.getcwd(), model_path));
        if len(file_paths)>0 and os.path.isfile(file_paths[0]):
            opt.model_path = file_paths[0];
            print('Model has been found at: {}'.format(opt.model_path));
            valid_path = True;

    valid_fold = False;
    while not valid_fold:
        fold = input("Select the fold on which the model was Trained:\n 1. Fold-1\n 2. Fold-2\n 3. Fold-3\n 4. Fold-4\n 5. Fold-5\n :")
        if fold in ['1','2','3','4','5']:
            opt.split = int(fold);
            valid_fold = True;

    tester = Tester(opt);
    tester.TestModel();
