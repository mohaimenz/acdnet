import sys;
import os;
import glob;
import torch;

sys.path.append(os.getcwd());
sys.path.append(os.path.join(os.getcwd(), 'common'));
sys.path.append(os.path.join(os.getcwd(), 'torch'));
sys.path.append(os.path.join(os.getcwd(), 'resources'));
import common.opts as opts;
import resources.models as models;


if __name__ == '__main__':
    opt = opts.parse();

    valid_pruning_type = False;
    while not valid_pruning_type:
        prune_type = input('Enter an option: \n1. Magnitude Pruning\n2. Taylor Pruning\n3. Weight Pruning\n4. Hybrid Pruning\n:');
        if prune_type in ['1', '2', '3', '4']:
            opt.prune_type = int(prune_type);
            valid_pruning_type = True;

    valid_pruning_type = False;
    if opt.prune_type == 4:
        while not valid_pruning_type:
            prune_type = input('Enter hybrid pruning option: \n1. Weight -> Magnitude Pruning\n2. Weight -> Taylor Pruning\n:');
            if prune_type in ['1', '2']:
                opt.prune_type = int(prune_type);
                valid_pruning_type = True;

    valid_path = False;
    while not valid_path:
        model_path = input("Enter the model path (For hybrid pruning, please provide the weight pruned model path)\n:");
        if model_path != '':
            mainDir = os.getcwd();#os.path.dirname(os.path.abspath('.'));
            file_paths = glob.glob(os.path.join(mainDir, model_path));
            if len(file_paths)>0 and os.path.isfile(file_paths[0]):
                state = torch.load(file_paths[0], map_location='cpu');
                opt.model_path = file_paths[0];
                print('Model has been found at: {}'.format(opt.model_path));
                valid_path = True;

    if opt.prune_type != 3:
        valid_prune_area = False;
        while not valid_prune_area:
            prune_area = input('Which part of the network to prune? \n1. Only TFEB\n2. FUll Network\n:');
            if prune_area in ['1', '2']:
                opt.prune_all = True if prune_area == '2' else False;
                valid_prune_area = True;

        valid_prune_ratio = False;
        while not valid_prune_ratio:
            prune_ratio = input('Enter prune ratio (0.1 to 0.99)\n:');
            try:
                val = float(prune_ratio);
                if val >=0.1 and val<1.0:
                    opt.prune_ratio = val;
                    valid_prune_ratio = True;
            except:
                valid_prune_ratio = False;

    valid_model_name = False;
    while not valid_model_name:
        model_name = input('Enter a name that will be used to save the trained model: ');
        if model_name != '':
            opt.model_name = model_name;
            valid_model_name = True;

    valid_fold = False;
    split = None;
    fold = 0;
    while not valid_fold:
        fold = input("Please enter the fold for fine-tuning:\n 1. Fold-1\n 2. Fold-2\n 3. Fold-3\n 4. Fold-4\n 5. Fold-5\n :")
        if fold in ['1','2','3','4','5']:
            opt.split = int(fold);
            valid_fold = True;

    if opt.prune_type == 3:
        import pruning.weight_pruning as wp;
        # opts.display_info(opt);
        print('+-- Split {} --+'.format(opt.split));
        wpt = wp.PruningTrainer(opt);
        wpt.PruneAndTrain();
    else:
        import pruning.filter_pruning as fp;
        fpt = fp.PruningTrainer(opt)
        fpt.PruneAndTrain();
