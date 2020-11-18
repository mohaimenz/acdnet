import os;
import sys;
import numpy as np;
import random;

sys.path.append(os.getcwd());
sys.path.append(os.path.join(os.getcwd(), 'common'));
import common.utils as U;

class Generator():
    #Generates data for Keras
    def __init__(self, samples, labels, options):
        random.seed(42);
        #Initialization
        self.data = [(samples[i], labels[i]) for i in range (0, len(samples))];
        self.opt = options;
        self.batch_size = options.batchSize;
        self.preprocess_funcs = self.preprocess_setup();

    def __len__(self):
        #Denotes the number of batches per epoch
        return int(np.floor(len(self.data) / self.batch_size));
        #return len(self.samples);

    def __getitem__(self, batchIndex):
        #Generate one batch of data
        batchX, batchY = self.generate_batch(batchIndex);
        batchX = np.expand_dims(batchX, axis=1);
        batchX = np.expand_dims(batchX, axis=3);
        return batchX, batchY

    def generate_batch(self, batchIndex):
        #Generates data containing batch_size samples
        sounds = [];
        labels = [];
        indexes = None;
        for i in range(self.batch_size):
            # Training phase of BC learning
            # Select two training examples
            while True:
                sound1, label1 = self.data[random.randint(0, len(self.data) - 1)]
                sound2, label2 = self.data[random.randint(0, len(self.data) - 1)]
                if label1 != label2:
                    break
            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)

            # Mix two examples
            r = np.array(random.random())
            sound = U.mix(sound1, sound2, r, self.opt.sr).astype(np.float32)
            eye = np.eye(self.opt.nClasses)
            label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)

            #For stronger augmentation
            sound = U.random_gain(6)(sound).astype(np.float32)

            sounds.append(sound);
            labels.append(label);

        sounds = np.asarray(sounds);
        labels = np.asarray(labels);

        return sounds, labels;

    def preprocess_setup(self):
        funcs = []
        if self.opt.strongAugment:
            funcs += [U.random_scale(1.25)]

        funcs += [U.padding(self.opt.inputLength // 2),
                  U.random_crop(self.opt.inputLength),
                  U.normalize(32768.0)]
        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound;

def setup(opt, split):
    dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.sr // 1000)), allow_pickle=True);
    train_sounds = []
    train_labels = []
    for i in range(1, opt.nFolds + 1):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        labels = dataset['fold{}'.format(i)].item()['labels']
        if i != split:
            train_sounds.extend(sounds)
            train_labels.extend(labels)

    trainGen = Generator(train_sounds, train_labels, opt);

    return trainGen
