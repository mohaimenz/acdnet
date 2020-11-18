import os;
import sys;
import time;
import numpy as np;
import random;
import opts;
import utils as U;

class ValGenerator():
    #Generates data for Keras
    def __init__(self, samples, labels, options):
        random.seed(42);
        #Initialization
        self.data = [(samples[i], labels[i]) for i in range (0, len(samples))];
        self.opt = options;
        self.batch_size = options.batchSize // options.nCrops;
        self.preprocess_funcs = self.preprocess_setup();

    def get_data(self):
        #Generate one batch of data
        x, y = self.generate();
        x = np.expand_dims(x, axis=1)
        x = np.expand_dims(x, axis=3)
        print(x.shape);
        print(y.shape);
        return x, y

    def generate(self):
        #Generates data containing batch_size samples
        sounds = [];
        labels = [];
        indexes = None;
        for i in range(self.batch_size):
            sound, target = self.data[i];
            sound = self.preprocess(sound).astype(np.float32)
            label = np.zeros((self.opt.nCrops, self.opt.nClasses));
            label[:,target] = 1;

            sounds.append(sound);
            labels.append(label);

        sounds = np.asarray(sounds);
        labels = np.asarray(labels);

        sounds = sounds.reshape(sounds.shape[0]*sounds.shape[1], sounds.shape[2]);
        labels = labels.reshape(labels.shape[0]*labels.shape[1], labels.shape[2]);

        return sounds, labels;

    def preprocess_setup(self):
        funcs = []
        funcs += [U.padding(self.opt.inputLength // 2),
                  U.normalize(32768.0),
                  U.multi_crop(self.opt.inputLength, self.opt.nCrops)]

        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound;

if __name__=='__main__':
    opt = opts.parse();
    opts.display_info(opt);
    opt.batchSize=4000;

    for sr in [44100, 20000]:
        opt.sr = sr;
        opt.inputLength = 66650 if sr == 44100 else 30225;
        mainDir = os.getcwd();
        test_data_dir = os.path.join(mainDir, 'datasets/esc50/test_data_{}khz'.format(sr//1000));
        print(test_data_dir)
        if not os.path.exists(test_data_dir):
            os.mkdir(test_data_dir);

        val_sounds = [];
        val_labels = [];
        dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.sr // 1000)), allow_pickle=True);
        for s in opt.splits:
            start_time = time.perf_counter();
            sounds = dataset['fold{}'.format(s)].item()['sounds'];
            labels = dataset['fold{}'.format(s)].item()['labels'];
            val_sounds.extend(sounds);
            val_labels.extend(labels);

            valGen = ValGenerator(sounds, labels, opt);
            valX, valY = valGen.get_data();

            print('{}/fold{}_test4000'.format(test_data_dir, s));
            np.savez_compressed('{}/fold{}_test4000'.format(test_data_dir, s), x=valX, y=valY);
            print('split-{} test with shape x{} and y{} took {:.2f} secs'.format(s, valX.shape, valY.shape, time.perf_counter()-start_time));
            sys.stdout.flush();
