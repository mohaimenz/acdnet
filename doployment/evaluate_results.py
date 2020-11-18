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

  result_files = [r for r in os.listdir('results') if r.startswith('result') and r.endswith('npy')]
  result_files = sorted(result_files)

  if len(result_files) > 0:
      print_results(os.path.join('results',result_files[0]))
