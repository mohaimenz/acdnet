#!/usr/bin/env python3

import os
from lib import *
from  data_loader import Trainer

def get_cast_input_data(trainer, dtype):
  '''Retrieves the input data set, casting to target dtype'''
  x,y = trainer.testX, trainer.testY
  
  if dtype is None:
    print('dtype not provided')
    return x,y

  return get_cast(dtype)(x, axis = -2), y

def generate_raw(trainer, dtype, dest_path): 
  x, y = get_cast_input_data(trainer, dtype)

  endian = 'little'

  # Save the raw files into the predetermined location
  save_raw(x, dtype, endian, dest_path)

  # Save the expected scores into the predetermined location
  save_scores(y, dtype, endian, dest_path)

if __name__ == '__main__':
  opt = parse()
  
  # Model is not required, but we are reusing the argument parser
  assert os.path.exists(opt.model), \
    f'File not found: {opt.model}\n'

  # Setup environment
  run_tests()

  # Setup required variables
  opt = opts.parse();
  
  dtype = 'int8'
  
  cwd_path = os.path.dirname(os.path.abspath(__file__))
  dest_path = os.path.join(cwd_path, 'data')  

  # Get the data sets
  trainer = Trainer(opt);  
  trainer.load_test_data();

  # generate the requested data set, quantized in the requested data type  
  generate_raw(trainer, dtype, dest_path)