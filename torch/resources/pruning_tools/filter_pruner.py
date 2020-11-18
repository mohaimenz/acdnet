import torch;
import numpy as np;

def prune_layers(net, prune_targets, prune_all=True, device='cpu'):
    net = net.cpu();
    sfeb_lyrs = list(enumerate(net.sfeb));
    tfeb_lyrs = list(enumerate(net.tfeb));
    all_features = sfeb_lyrs + tfeb_lyrs if prune_all else tfeb_lyrs;
    conv_indexes = get_conv_layer_indexes(net, prune_all);

    for layer_index, channel_index in prune_targets:
        # print(layer_index, ' ', channel_index)
        conv_idx, conv = all_features[layer_index];

        # Do not want to reduce the output channels of 2nd conv less than 32 to keep the network working
        if layer_index == 3 and conv.out_channels <= 32:
            print('!!!NOTICE!!! ({},{}) not pruned. Layer 3 is already in lowest threshold 32'.format(layer_index, channel_index));
        else:
            #remove output channel from conv layer
            new_conv = get_new_conv(conv, channel_index, False, device);
            all_features[layer_index] = (conv_idx, new_conv);
            bn_idx, bn =  all_features[layer_index+1];
            new_bn = get_new_bn(bn, channel_index);
            all_features[layer_index + 1] = (bn_idx, new_bn);

            #Adjust the AvgPool layers kernel size and stride when layer 3 has less than 64 out channels
            if list(list(all_features[len(all_features)-3])[1].kernel_size)[0]>1 and layer_index == 3 and new_conv.out_channels < 64:
                idx, avg_pool = all_features[len(all_features)-3];
                kh, kw = avg_pool.kernel_size;
                all_features[len(all_features)-3] = (idx, torch.nn.AvgPool2d(kernel_size=(1, kw)));


            if layer_index == max(conv_indexes):
                #last conv layer
                #Adjust first linear layer's no. of input
                lin_idx, linear = all_features[-1];
                new_linear = get_new_linear(linear, channel_index);
                all_features[-1] = (0, new_linear);
            else:
                #Adjust next conv layer's no. of in_channels
                next_conv_lyr_id = conv_indexes[conv_indexes.index(layer_index)+1];
                next_conv_id, next_conv = all_features[next_conv_lyr_id];
                #we do not touch the first conv layers in features as the channels are always fixed by the permute
                # print('Next conv layer id: ', next_conv_lyr_id);
                if (prune_all and next_conv_lyr_id !=  len(sfeb_lyrs)) or prune_all is False:
                    # print('Adjusting Next conv layer');
                    next_new_conv = get_new_conv(next_conv, channel_index, True, device);
                    all_features[next_conv_lyr_id] = (next_conv_id, next_new_conv);

    tfeb_starts_at = 0;
    if prune_all:
        tfeb_starts_at = len(sfeb_lyrs);
        del net.sfeb;
        net.sfeb = torch.nn.Sequential(*[module for idx, module in all_features[0:tfeb_starts_at]]);
    del net.tfeb;
    net.tfeb = torch.nn.Sequential(*[module for idx, module in all_features[tfeb_starts_at:]]);

    net = net.to(device);
    return net;

def get_conv_layer_indexes(net, prune_all=False):
    indexes = [];
    layer_index = 0;
    if prune_all:
        for index, module in enumerate(net.sfeb):
            if issubclass(type(module), torch.nn.Conv2d):
                indexes.append(index);
            layer_index += 1;

    for index, module in enumerate(net.tfeb):
        if issubclass(type(module), torch.nn.Conv2d):
            indexes.append(layer_index);
        layer_index += 1;
    return indexes;

def get_new_conv(conv, channel_index, next_conv, device):
    in_chnls = conv.in_channels if next_conv is False else conv.in_channels - 1;
    out_chnls = conv.out_channels - 1 if next_conv is False else conv.out_channels;
    new_conv = torch.nn.Conv2d(in_channels = in_chnls, out_channels = out_chnls,\
            kernel_size = conv.kernel_size, stride = conv.stride, padding = conv.padding,\
            dilation = conv.dilation, groups = conv.groups, bias = (conv.bias is not None))

    dim = 0 if next_conv is False else 1;
    new_conv.weight.data = get_new_weight(conv.weight.data.cpu(), dim, channel_index);
    # new_conv.weight.data = new_conv.weight.data.to(device);

    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data if next_conv is True else get_new_weight(conv.bias.data.cpu(), dim, channel_index);
        # new_conv.bias.data = new_conv.bias.data.to(device);

    return new_conv;

def get_new_bn(bn, channel_index):
    new_bn = torch.nn.BatchNorm2d(num_features=int(bn.num_features - 1),
            eps=bn.eps, momentum=bn.momentum, affine=bn.affine, track_running_stats=bn.track_running_stats);

    new_bn.weight.data = get_new_weight(bn.weight.data, 0, channel_index);
    new_bn.bias.data = get_new_weight(bn.bias.data, 0, channel_index);

    if bn.track_running_stats:
        new_bn.running_mean.data = get_new_weight(bn.running_mean.data, 0, channel_index);
        new_bn.running_var.data = get_new_weight(bn.running_var.data, 0, channel_index);

    return new_bn

def get_new_linear(linear, channel_index):
    new_linear = torch.nn.Linear(in_features=int(linear.in_features - 1),
                out_features=linear.out_features, bias=linear.bias is not None);
    new_linear.weight.data = get_new_weight(linear.weight.data, 1, channel_index);
    new_linear.bias.data = linear.bias.data;

    return new_linear;

def get_new_weight(weight, dim, channel_index):
    indexes = list(set(range(weight.size(dim))) - set({channel_index}));
    new_data = torch.index_select(weight, dim, torch.tensor(indexes));
    return new_data;

# # Testing
# import os;
# import models;
# import glob;
# import calculator as calc;
# import sys;
# sys.path.append(os.getcwd());
# sys.path.append(os.path.join(os.getcwd(), 'common'));
# import common.opts as opts;
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
# dir = os.getcwd();
# net = models.GetACDNetModel().to(device);
# net_path = '{}/torch/resources/pretrained_models/acdnet_trained_model_fold2_88.00.pt'.format(dir);
# file_paths = glob.glob(net_path);
# if len(file_paths)>0 and os.path.isfile(file_paths[0]):
#     state = torch.load(file_paths[0], map_location=device);
#     net.load_state_dict(state['weight']);
#
# # calc.summary(net, (1,1, 66650));
# # print(list(enumerate(net.tfeb)));
# print(get_conv_layer_indexes(net, True));
# # print(net.output);
# net = prune_layers(net, [(18, 10), (25, 15), (40,20), (40,25)], True, device);
# # print(net.tfeb);
# # print(net.output);
# calc.summary(net, (1,1,66650));
