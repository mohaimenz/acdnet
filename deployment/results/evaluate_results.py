#!/usr/bin/env python3

# Used to quickly print the main details of the latests model convert results

import os
import numpy as np
import pprint
from tabulate import tabulate

def print_results(npy_path):
    '''Print a table of results from the requested npy results file'''
    
    print(f'Loading {npy_path}')
    data = np.load(npy_path, allow_pickle=True)    
    
    print_fields = ['model', 'dtype', 'set_name', 'tf_accuracy', 'tflite_accuracy', 'tflite_arena', 'input_size', 'output_size']

    filter_data = [{f: r[f] for f in print_fields if f in r.keys() } for r in data]  
    print(tabulate(filter_data, headers = 'keys'))


if __name__ == '__main__':  

    if len(sys.argv) > 1:
        assert len(sys.argv) == 2, \
            f'Expected 1 arguments: {sys.argv[0]} path/to/result.npy'

    activity_path = sys.argv[1]

    assert os.path.exists(npy_path), \
        f'Numpy npy result file not found with path: {npy_path}'

    print_results(npy_path)
