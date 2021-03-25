import torch;
from collections import OrderedDict;
import numpy as np;
import math;
'''
A simplified 3-D Tensor (channels, height, width) for CNN
'''
def summary(net, inputs, brief=False):
    calc = Calculator(net, inputs);
    calc.calculate();
    if brief:
        return calc.quick_summary();
    else:
        calc.detailed_summary();

class Input(object):
    def __init__(self, channels, height, width):
        self.c = channels;
        self.h = height;
        self.w = width;

class Calculator(object):
    def __init__(self, net, inputs):
        self.params = 0;
        self.flops = 0;
        self.summary = OrderedDict();
        self.net = net;
        self.inputs = inputs;

    def calculate(self):
        input = Input(*self.inputs)
        modules = [];
        for name, module in enumerate(self.net.sfeb):
            modules.append(module);
        modules.append('permute');
        for name, module in enumerate(self.net.tfeb):
            modules.append(module);
        # modules.append(torch.nn.Flatten());
        try:
            for name, module in enumerate(self.net.output):
                modules.append(module);
        except:
            print('net.output is not iterable');

        for module in modules:
            if module == 'permute':
                input = self.Permute(input);
            elif issubclass(type(module), torch.nn.Conv2d):
                # print(module.padding);
                input = self.Conv2d(module, input);
                # print(module.weight.requires_grad);
                # break;
            elif issubclass(type(module), torch.nn.BatchNorm2d):
                input = self.BatchNorm2d(module, input);
                # break;
            elif issubclass(type(module), (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
                name = 'MaxPool2d' if issubclass(type(module), torch.nn.MaxPool2d) else 'AvgPool2d';
                input = self.Pool2d(module, input, name);
                # exit();
            elif issubclass(type(module), torch.nn.Flatten):
                # print('Flatten')
                input=self.Flatten(input);
            elif issubclass(type(module), torch.nn.Linear):
                input=self.Linear(module, input);
            elif issubclass(type(module), (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Softmax)):
                name = 'ReLu' if issubclass(type(module), torch.nn.ReLU) \
                        else ('Sigmoid' if issubclass(type(module), torch.nn.Sigmoid) else 'Softmax');
                input = self.Activation(module, input, name);

    def Conv2d(self, module, input):
        kh, kw = module.kernel_size;
        sh, sw = module.stride;
        ph, pw = module.padding;
        in_ch = module.in_channels;
        out_ch = module.out_channels;
        groups = module.groups;
        out_h = (input.h - kh + 2 * ph) // sh + 1;
        out_w = (input.w - kw + 2 * pw) // sw + 1;

        params = out_ch * in_ch // groups * kh * kw;
        flops = out_ch * out_h * out_w * in_ch // groups * kh * kw;

        if module.bias is not None:
            params += out_ch;
            flops += out_ch * out_w * out_h;
        out_shape = (out_ch, out_h, out_w);
        self.add_to_summary(*('Conv2d', (input.c, input.h, input.w), out_shape, params, flops));
        return Input(*out_shape);

    def BatchNorm2d(self, module, input):
        #params
        params = 0;
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params = torch.prod(torch.LongTensor(list(module.weight.size())));
        if module.bias is not None:
            params += torch.prod(torch.LongTensor(list(module.bias.size())));

        # Batch normalization can be combined with the preceding convolution, so there are no FLOPs.
        flops = 0;

        self.add_to_summary(*('BatchNorm2d', (input.c, input.h, input.w), (input.c, input.h, input.w), params, 0));
        # print(input.c, '-', input.h, '-', input.w);
        return input;

    def Pool2d(self, module, input, name='Pool2d'):
        kh, kw = module.kernel_size;
        sh, sw = module.stride;
        ph, pw = module.padding, module.padding;
        out_ch = input.c;
        scale = math.ceil if module.ceil_mode else math.floor;
        out_h = scale((input.h - kh + 2 * ph) / sh) + 1;
        out_w = scale((input.w - kw + 2 * pw) / sw) + 1;
        flops = out_ch * out_h * out_w * kh * kw;
        out_shape = (out_ch, out_h, out_w);
        self.add_to_summary(*(name, (input.c, input.h, input.w), out_shape, 0, flops));
        # print(out_shape);
        return Input(*out_shape);

    def Permute(self, input):
        # print(input.c, '-', input.h, '-', input.w);
        out_shape = (input.h, input.c, input.w);
        self.add_to_summary(*("Permute", (input.c, input.h, input.w), out_shape, 0, 0));
        # print(out_shape);
        return Input(*out_shape);

    def Flatten(self, input):
        out_w = input.c * input.h * input.w;
        self.add_to_summary(*('Flatten', (input.c, input.h, input.w), (1, out_w), 0, 0));
        # print((1,1,out_w));
        return Input(1,1,out_w);

    def Linear(self, module, input):
        # assert input.w == module.in_features, 'input width({}) and module.in_features({}) must be same'.format(input.w, module.in_features);
        # assert input.h == 1 and input.w > 1, 'height and width in Linear layer is same';

        params = module.in_features * module.out_features;
        flops = module.in_features * module.out_features;
        if module.bias is not None:
            params += module.out_features;
            flops += module.out_features;
        self.add_to_summary(*('Linear', (1, input.w), (1, module.out_features), params, flops));
        # print((1,1,module.out_features))
        return Input(1,1,module.out_features);

    def Activation(self, module, input, name):
        flops = input.c * input.h * input.w;
        in_shape = (1, input.w) if input.c == 1 and input.h == 1 else (input.c, input.h, input.w);
        self.add_to_summary(*(name, in_shape, in_shape, 0, flops));
        return input;

    def add_to_summary(self, module_name, in_shape, out_shape, params, flops):
        dict_key = '{}-{}'.format(module_name, len(self.summary) + 1);
        self.summary[dict_key] = OrderedDict();
        self.summary[dict_key]['module_name'] = module_name;
        self.summary[dict_key]['input_shape'] = in_shape;
        self.summary[dict_key]['output_shape'] = out_shape;
        self.summary[dict_key]['params'] = params;
        self.summary[dict_key]['flops'] = flops;
        self.params += params;
        self.flops += flops;

    def quick_summary(self):
        #bytes = params * 32 bits / 8bits = params * 4, MBits = bits/2024, MBytes = bytes/2014
        input_size = abs(np.prod(self.inputs)*4./(1024*1024));
        params_size = abs(self.params*4./(1024*1024));
        total_size = input_size+params_size;
        str = "Input: {:.3f} MB, Params: {:,} ({:.3f} MB), Total: {:.2f} MB, FLOPs: {:,}".format(
            input_size, self.params, params_size, total_size, self.flops
        );
        print(str);
        return self.params, params_size, self.flops;

    def detailed_summary(self):
        summary_str = "+----------------------------------------------------------------------------+" + "\n";
        summary_str += "+                           Pytorch Model Summary                            +" + "\n";
        summary_str += "------------------------------------------------------------------------------" + "\n";
        summary_str += "{:>15} {:>17} {:>17} {:>10} {:>12}".format(
            "Layer (type)", "Input Shape", "Output Shape", "Param #", "FLOPS #");
        summary_str += "\n";
        summary_str += "==============================================================================" + "\n";
        for layer in self.summary:
            summary_str += "{:>15} {:>17} {:>17} {:>10} {:>12}".format(
                layer, str(self.summary[layer]['input_shape']),
                str(self.summary[layer]['output_shape']), '{0:,}'.format(self.summary[layer]['params']),
                '{0:,}'.format(self.summary[layer]['flops']));
            summary_str += "\n";
        summary_str += "==============================================================================" + "\n";
        summary_str += "Total Params: {0:,}\n".format(self.params);
        summary_str += "Total FLOPs : {0:,}\n".format(self.flops);
        summary_str += "------------------------------------------------------------------------------" + "\n";
        #bytes = params * 32 bits / 8bits = params * 4, MBits = bits/2024, MBytes = bytes/2014
        input_size = abs(np.prod(self.inputs)*4./(1024*1024));
        params_size = abs(self.params*4./(1024*1024));
        summary_str += "Input size (MB) : {:.2f}\n".format(input_size);
        summary_str += "Params size (MB): {:.2f}\n".format(params_size);
        summary_str += "Total size (MB) : {:.2f}\n".format(input_size+params_size);
        summary_str += "------------------------------------------------------------------------------" + "\n";
        print(summary_str);
