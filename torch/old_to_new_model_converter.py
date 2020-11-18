# # Check the model is working
import os;
import sys;
import glob;
import torch;
sys.path.append(os.path.join(os.getcwd(), 'torch'));
sys.path.append(os.path.join(os.getcwd(), 'resources'));
import resources.models as models;
import resources.calculator as calculator;
if __name__ == '__main__':
    # net = models.GetACDNetModel();
    dir = os.getcwd();

    #load pruned config
    name = 'acdnet_trained_model_fold3';
    net_path = '{}/torch/resources/pretrained_models/sp3_base_model_89.25.pt'.format(dir, name);
    file_paths = glob.glob(net_path);
    state = torch.load(file_paths[0], map_location='cpu');
    # config = state['config'];
    config = None;
    # state = state['model'];


    # #Load trained weight
    # config = [4, 32, 12, 12, 23, 18, 38, 43, 62, 58, 77, 37];
    # name = 'acdnet_weight_pruned_model';
    # net_path = '{}/torch/resources/pretrained_models/{}.pth'.format(dir, name);
    # file_paths = glob.glob(net_path);
    # state = torch.load(file_paths[0], map_location='cpu');

    net = models.GetACDNetModel(66650, 50, 44100, config).to('cpu');


    new_state = {};

    new_state['sfeb.0.weight'] = state["filterbank.0.weight"];
    new_state['sfeb.1.weight'] = state["filterbank.1.weight"];
    new_state['sfeb.1.bias'] = state["filterbank.1.bias"];
    new_state['sfeb.1.running_mean'] = state["filterbank.1.running_mean"];
    new_state['sfeb.1.running_var'] = state["filterbank.1.running_var"];
    new_state['sfeb.1.num_batches_tracked'] = state["filterbank.1.num_batches_tracked"];
    new_state['sfeb.3.weight'] = state["filterbank.3.weight"];
    new_state['sfeb.4.weight'] = state["filterbank.4.weight"];
    new_state['sfeb.4.bias'] = state["filterbank.4.bias"];
    new_state['sfeb.4.running_mean'] = state["filterbank.4.running_mean"];
    new_state['sfeb.4.running_var'] = state["filterbank.4.running_var"];
    new_state['sfeb.4.num_batches_tracked'] = state["filterbank.4.num_batches_tracked"];
    new_state["tfeb.0.weight"] = state["features.0.weight"];
    new_state["tfeb.1.weight"] = state["features.1.weight"];
    new_state["tfeb.1.bias"] = state["features.1.bias"];
    new_state["tfeb.1.running_mean"] = state["features.1.running_mean"];
    new_state["tfeb.1.running_var"] = state["features.1.running_var"];
    new_state["tfeb.1.num_batches_tracked"] = state["features.1.num_batches_tracked"];
    new_state["tfeb.4.weight"] = state["features.4.weight"];
    new_state["tfeb.5.weight"] = state["features.5.weight"];
    new_state["tfeb.5.bias"] = state["features.5.bias"];
    new_state["tfeb.5.running_mean"] = state["features.5.running_mean"];
    new_state["tfeb.5.running_var"] = state["features.5.running_var"];
    new_state["tfeb.5.num_batches_tracked"] = state["features.5.num_batches_tracked"];
    new_state["tfeb.7.weight"] = state["features.7.weight"];
    new_state["tfeb.8.weight"] = state["features.8.weight"];
    new_state["tfeb.8.bias"] = state["features.8.bias"];
    new_state["tfeb.8.running_mean"] = state["features.8.running_mean"];
    new_state["tfeb.8.running_var"] = state["features.8.running_var"];
    new_state["tfeb.8.num_batches_tracked"] = state["features.8.num_batches_tracked"];
    new_state["tfeb.11.weight"] = state["features.11.weight"];
    new_state["tfeb.12.weight"] = state["features.12.weight"];
    new_state["tfeb.12.bias"] = state["features.12.bias"];
    new_state["tfeb.12.running_mean"] = state["features.12.running_mean"];
    new_state["tfeb.12.running_var"] = state["features.12.running_var"];
    new_state["tfeb.12.num_batches_tracked"] = state["features.12.num_batches_tracked"];
    new_state["tfeb.14.weight"] = state["features.14.weight"];
    new_state["tfeb.15.weight"] = state["features.15.weight"];
    new_state["tfeb.15.bias"] = state["features.15.bias"];
    new_state["tfeb.15.running_mean"] = state["features.15.running_mean"];
    new_state["tfeb.15.running_var"] = state["features.15.running_var"];
    new_state["tfeb.15.num_batches_tracked"] = state["features.15.num_batches_tracked"];
    new_state["tfeb.18.weight"] = state["features.18.weight"];
    new_state["tfeb.19.weight"] = state["features.19.weight"];
    new_state["tfeb.19.bias"] = state["features.19.bias"];
    new_state["tfeb.19.running_mean"] = state["features.19.running_mean"];
    new_state["tfeb.19.running_var"] = state["features.19.running_var"];
    new_state["tfeb.19.num_batches_tracked"] = state["features.19.num_batches_tracked"];
    new_state["tfeb.21.weight"] = state["features.21.weight"];
    new_state["tfeb.22.weight"] = state["features.22.weight"];
    new_state["tfeb.22.bias"] = state["features.22.bias"];
    new_state["tfeb.22.running_mean"] = state["features.22.running_mean"];
    new_state["tfeb.22.running_var"] = state["features.22.running_var"];
    new_state["tfeb.22.num_batches_tracked"] = state["features.22.num_batches_tracked"];
    new_state["tfeb.25.weight"] = state["features.25.weight"];
    new_state["tfeb.26.weight"] = state["features.26.weight"];
    new_state["tfeb.26.bias"] = state["features.26.bias"];
    new_state["tfeb.26.running_mean"] = state["features.26.running_mean"];
    new_state["tfeb.26.running_var"] = state["features.26.running_var"];
    new_state["tfeb.26.num_batches_tracked"] = state["features.26.num_batches_tracked"];
    new_state["tfeb.28.weight"] = state["features.28.weight"];
    new_state["tfeb.29.weight"] = state["features.29.weight"];
    new_state["tfeb.29.bias"] = state["features.29.bias"];
    new_state["tfeb.29.running_mean"] = state["features.29.running_mean"];
    new_state["tfeb.29.running_var"] = state["features.29.running_var"];
    new_state["tfeb.29.num_batches_tracked"] = state["features.29.num_batches_tracked"];
    new_state["tfeb.33.weight"] = state["features.33.weight"];
    new_state["tfeb.34.weight"] = state["features.34.weight"];
    new_state["tfeb.34.bias"] = state["features.34.bias"];
    new_state["tfeb.34.running_mean"] = state["features.34.running_mean"];
    new_state["tfeb.34.running_var"] = state["features.34.running_var"];
    new_state["tfeb.34.num_batches_tracked"] = state["features.34.num_batches_tracked"];
    new_state["tfeb.38.weight"] = state["classifier.0.weight"];
    new_state["tfeb.38.bias"] = state["classifier.0.bias"];


    net.load_state_dict(new_state);
    print('Model found at: {}'.format(file_paths[0]));
    model_path = '{}/torch/resources/pretrained_models/new_{}.pt'.format(dir, name);
    torch.save({'weight':net.state_dict(), 'config':config}, model_path);

    calculator.summary(net, (1, 1, 66650), brief=False);
