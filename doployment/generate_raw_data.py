#!/usr/bin/env python3

import os
from common import download, get_data_path, get_cast, data_support, load_data, save_raw, save_scores, run_tests

def get_cast_input_data(data_path, dtype = None):
    '''Retrieves the input data set, casting to target dtype'''
    x, y = load_data(data_path)
    if dtype is None:
      print('dtype not provided')
      return x,y   
    return get_cast(dtype)(x, axis = -2), y

def generate_raw(set_name, dtype):
  assert set_name in data_support.keys(), 'Unknown set'

  # Download the files if they do not already exist locally
  download(set_name)

  # Fetch the data set path
  data_src_path = get_data_path(set_name, 'test')
  
  x, y = get_cast_input_data(data_src_path, dtype)

  # Save the raw files into the predetermined location
  save_raw(x, set_name, dtype)

  # Save the expected scores into the predetermined location
  save_scores(y, set_name, dtype)

if __name__ == '__main__':
  # Setup environment
  run_tests()

  # generate the requested data set, quantized in the requested data type  
  generate_raw('aug-data-20khz', 'int8')