import torch;
import torch.optim as optim;
from operator import itemgetter;
from heapq import nsmallest;

class Magnitude:
    def __init__(self, net, opt):
        self.opt = opt;
        self.net = net;
        self.reset();

    def reset(self):
        self.filter_ranks = [];

    def compute_filter_magnitude(self):
        layer_index = 0;
        if self.opt.prune_all:
            for layer, (name, module) in enumerate(self.net.sfeb._modules.items()):
                if isinstance(module, torch.nn.modules.conv.Conv2d):
                    # Do not want to reduce the output channels of 2nd conv less than 32 to keep the network working
                    if layer_index != 3 or (layer_index == 3 and module.out_channels > 32):
                        f_data = module.weight.detach().clone();
                        self.add_to_filter_rank(layer_index, f_data);
                layer_index += 1;

        for layer, (name, module) in enumerate(self.net.tfeb._modules.items()):
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                f_data = module.weight.detach().clone();
                self.add_to_filter_rank(layer_index, f_data);
            layer_index += 1;
        # print(self.filter_ranks);
        # exit();

    def add_to_filter_rank(self, layer_index, f_data):
        mag_data = self.get_magnitude(f_data);
        for i, m in enumerate(mag_data):
            self.filter_ranks.append((layer_index, i, m.data));

    def get_magnitude(self, f_data):
        f_data = torch.abs(f_data);
        f_sum = torch.abs(f_data.sum(dim=(1, 2, 3)).data);
        f_sum = self.get_layerwise_normalized_data(f_sum);
        return f_sum;

    def get_layerwise_normalized_data(self, data):
        data = data / (torch.sqrt(torch.sum(data * data)));
        return data;

    def lowest_ranking_filters(self, num):
        return nsmallest(num, self.filter_ranks, itemgetter(2));

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune);
        targets = [];
        for lidx, fidx, value in filters_to_prune:
            targets.append((lidx, fidx));
        return targets;

class Taylor:
    def __init__(self, net, opt):
        self.opt = opt;
        self.net = net;
        self.reset();

    def reset(self):
        self.filter_ranks = {};

    def forward(self, x):
        self.activations = [];
        self.gradients = [];
        self.grad_index = 0;
        self.activation_to_layer = {};

        activation_index = 0;
        layer_index = 0;
        if self.opt.prune_all:
            for layer, (name, module) in enumerate(self.net.sfeb._modules.items()):
                # print(layer_index);
                x = module(x);
                if isinstance(module, torch.nn.modules.conv.Conv2d):
                    # Do not want to reduce the output channels of 2nd conv less than 32 to keep the network working
                    if layer_index != 3 or (layer_index == 3 and module.out_channels > 32):
                        x.register_hook(self.compute_rank);
                        self.activations.append(x);
                        self.activation_to_layer[activation_index] = layer_index
                        activation_index += 1;
                layer_index += 1;
        else:
            x = self.net.sfeb(x);

        x = x.permute((0, 2, 1, 3));
        for layer, (name, module) in enumerate(self.net.tfeb._modules.items()):
            x = module(x);
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank);
                self.activations.append(x);
                self.activation_to_layer[activation_index] = layer_index;
                activation_index += 1;
            layer_index += 1;
        return self.net.output(x);

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1;
        activation = self.activations[activation_index];
        taylor = activation * grad;

        # Get the average value for every filter,accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data;

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_();
            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.opt.device);

        self.filter_ranks[activation_index] += taylor;
        self.grad_index += 1;

    def lowest_ranking_filters(self, num):
        data = [];
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]));

        return nsmallest(num, data, itemgetter(2));

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i]);
            v = v / torch.sqrt(torch.sum(v * v));
            self.filter_ranks[i] = v.cpu();

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune);

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {};
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = [];
            filters_to_prune_per_layer[l].append(f);

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l]);
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i;

        filters_to_prune = [];
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i));

        return filters_to_prune;
