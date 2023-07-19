import sys;
import os;
import glob;
import math;
import random;
import torch;
import numpy as np

sys.path.append(os.getcwd());
sys.path.append(os.path.join(os.getcwd(), 'common'));
import common.opts as opts;
import resources.models as models;
import resources.calculator as calc;
import resources.train_generator as train_generator;

class Trainer:
    def __init__(self, opt=None, split=0):
        self.opt = opt;
        self.testX = None;
        self.testY = None;
        self.trainX = None;
        self.trainY = None;
        #self.opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        self.opt.device = torch.device("cpu")
        self.trainGen = train_generator.setup(self.opt, self.opt.split);

    def load_train_data(self):
        print('Preparing calibration dataset..');
        x,y = self.trainGen.__getitem__(0);
        self.trainX = torch.tensor(np.moveaxis(x, 3, 1)).to(self.opt.device);
        self.trainY = torch.tensor(y).to(self.opt.device);
        print('Calibration dataset is ready');
        self.opt.batchSize = 64;

    def load_test_data(self):
        if(self.testX is None):
            data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'test_data_20khz/fold{}_test4000.npz'.format(self.opt.split)), allow_pickle=True);
            self.testX = torch.tensor(np.moveaxis(data['x'], 3, 1)).to(self.opt.device);
            self.testY = torch.tensor(data['y']).to(self.opt.device);

    def __validate(self, net, testX, testY):
        net.eval();
        with torch.no_grad():
            y_pred = None;
            batch_size = (self.opt.batchSize//self.opt.nCrops)*self.opt.nCrops;
            for idx in range(math.ceil(len(self.testX)/batch_size)):
                x = self.testX[idx*batch_size : (idx+1)*batch_size];
                #print(x.shape);
                # exit();
                scores = net(x);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

            acc = self.__compute_accuracy(y_pred, self.testY);
        return acc;

    #Calculating average prediction (10 crops) and final accuracy
    def __compute_accuracy(self, y_pred, y_target):
        print(y_pred.shape);
        with torch.no_grad():
            y_pred = (y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(dim=1);
            y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(dim=1);

            y_pred = y_pred.argmax(dim=1);
            y_target = y_target.argmax(dim=1);

            acc = (((y_pred==y_target)*1).float().mean()*100).item();
        return acc;

    def __load_model(self, quant=False):
        state = torch.load(self.opt.model_path, map_location=self.opt.device);
        if quant:
            net = models.GetACDNetQuantModel(input_len=self.opt.inputLength, nclass=50, sr=self.opt.sr, channel_config=state['config']).to(self.opt.device);
        else:
            net = models.GetACDNetModel(input_len=self.opt.inputLength, nclass=50, sr=self.opt.sr, channel_config=state['config']).to(self.opt.device);
        net.load_state_dict(state['weight']);
        return net;

    def __calibrate(self, net):
        self.load_train_data();
        net.eval();
        with torch.no_grad():
            for i in range(1,2):
                x_pred = None;
                for idx in range(math.ceil(len(self.trainX)/self.opt.batchSize)):
                    x = self.trainX[idx*self.opt.batchSize : (idx+1)*self.opt.batchSize];
                    #print(x.shape);
                    # exit();
                    scores = net(x);
                    x_pred = scores.data if x_pred is None else torch.cat((x_pred, scores.data));

                x_pred = x_pred.argmax(dim=1);
                x_target = self.trainY.argmax(dim=1);

                acc = (((x_pred==x_target)*1).float().mean()*100).item();
                print('calibrate accuracy is: {:.2f}'.format(acc));
        return acc;

    def QuantizeModel(self):
        net = self.__load_model(True);
        config = net.ch_config;
        net.eval();

        #Fuse modules to
        torch.quantization.fuse_modules(net.sfeb, ['0','1','2'], inplace=True);
        torch.quantization.fuse_modules(net.sfeb, ['3','4','5'], inplace=True);

        torch.quantization.fuse_modules(net.tfeb, ['0','1','2'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['4','5','6'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['7','8','9'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['11','12','13'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['14','15','16'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['18','19','20'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['21','22','23'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['25','26','27'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['28','29','30'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['33','34','35'], inplace=True);

        # Specify quantization configuration
        net.qconfig = torch.quantization.get_default_qconfig('qnnpack');
        torch.backends.quantized.engine = 'qnnpack';
        print(net.qconfig);

        torch.quantization.prepare(net, inplace=True);

        # Calibrate with the training data
        self.__calibrate(net);

        # Convert to quantized model
        torch.quantization.convert(net, inplace=True);
        print('Post Training Quantization: Convert done');

        print("Size of model after quantization");
        torch.save(net.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')

        self.load_test_data();
        val_acc = self.__validate(net, self.testX, self.testY);
        print('Testing: Acc(top1) {:.2f}%'.format(val_acc));

        torch.jit.save(torch.jit.script(net), '{}/torch/quantized_models/{}.pt'.format(os.getcwd(), self.opt.model_name));

    def TestModel(self, quant=False):
        if quant:
            net = torch.jit.load(os.getcwd() + '/torch/quantized_models/' + self.opt.model_name + '.pt')
        else:
            net = self.__load_model();
            calc.summary(net, (1,1,self.opt.inputLength));
        self.load_test_data();
        net.eval();
        val_acc = self.__validate(net, self.testX, self.testY);
        print('Testing: Acc(top1) {:.2f}%'.format(val_acc));

    def GetModelSize(self):
        orig_net_path = self.opt.model_path;
        print('Full precision model size (KB):', os.path.getsize(orig_net_path)/(1024));
        quant_net_path = os.getcwd()+'/20khz_l0_tay_full_80_86.5_1403_quant.onnx';
        print('Quantized model size (KB):', os.path.getsize(quant_net_path)/(1024))


if __name__ == '__main__':
    opt = opts.parse();
    opt.batchSize = 1600;
    valid_path = False;
    while not valid_path:
        model_path = input("Enter the model PATH for 8-bit post training quantization\n:");
        file_paths = glob.glob(os.path.join(os.getcwd(), model_path));
        if len(file_paths)>0 and os.path.isfile(file_paths[0]):
            state = torch.load(file_paths[0], map_location='cpu');
            opt.model_path = file_paths[0];
            print('Model has been found at: {}'.format(opt.model_path));
            valid_path = True;

    valid_model_name = False;
    while not valid_model_name:
        model_name = input('Enter a name that will be used to save the quantized model model: ');
        if model_name != '':
            opt.model_name = model_name;
            valid_model_name = True;

    valid_fold = False;
    while not valid_fold:
        fold = input("Enter the fold number on which the model was validated:\n 1. Fold-1\n 2. Fold-2\n 3. Fold-3\n 4. Fold-4\n 5. Fold-5\n :")
        if fold in ['1','2','3','4','5']:
            opt.split = int(fold);
            valid_fold = True;

    trainer = Trainer(opt);

    print('Testing performance of the provided model.....');
    trainer.TestModel();

    print('Quantization process is started.....');
    trainer.QuantizeModel();
    print('Quantization done');

    print('Testing quantized model.');
    trainer.TestModel(True);
    print('Finished');
