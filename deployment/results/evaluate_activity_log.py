#!/usr/bin/env python3

# Used to quickly evaluate accuracy of scores in the latests activity.log
# Requires:
#   - path to activity.log file
#   - path to npz data with ['y'] expected results

import os
import sys
import numpy as np
import datetime
from tabulate import tabulate

crop_count = 10

model_name_index = 0
duration_index = 58
target_index = 59
feature_number_index = 52

feature_count = 4000
class_count = 50
class_index = 1
v1_field_count = 59

field_split_char = ','

def evaluate_accuracy(y_pred, y_target, ncrops):
  '''Performs the calculation of accuracy, when group of ncrop sequential result scores'''

  y_pred = y_pred.reshape(y_pred.shape[0] // ncrops, ncrops, y_pred.shape[1])
  y_target = y_target.reshape(y_target.shape[0] // ncrops, ncrops, y_target.shape[1])

  #Calculate the average of class predictions for 10 crops of a sample
  y_pred = np.mean(y_pred, axis=1)
  y_target = np.mean(y_target,axis=1)

  #Get the indices that has highest average value for each sample
  y_pred = y_pred.argmax(axis=1)
  y_target = y_target.argmax(axis=1)

  accuracy = (y_pred==y_target).mean()
  return accuracy

def evaluate_log(log_path, npz_path, ncrops):    
  '''Loads an activity log file and numpy npz file then evaluates the result'''  

  print(f'Evaluating  {log_path}')
  print(f'Source data {npz_path}\n')

  get_fields = lambda line: [f.strip() for f in line.split(field_split_char)]
  get_scores = lambda l : [int(f) for f in get_fields(l)[class_index:class_index + class_count]]  
  get_duration = lambda l : int(get_fields(l)[duration_index])
  get_model_name = lambda l : str(get_fields(l)[model_name_index].split(':')[1].strip())

  # Load activity log data
  with open(log_path, 'r') as fp:

    lines = (line.rstrip() for line in fp) # All lines including the blank ones
    lines = [line for line in lines if line] # Non-blank lines

    # Retrieve only the latest set of records i.e. the last '4000'
    y_pred = np.array([get_scores(l) for l in lines[-feature_count:]]) 
  
  last_line = get_fields(lines[-1])

  if len(last_line) == v1_field_count:
      target = 'spresense'
      duration_unit = 'ms'

  else:
      target = last_line[target_index]
      duration_unit = 'ns'

  # Load npz target data
  with np.load(npz_path, allow_pickle = True) as data:
    y_target = data['y']

  assert len(y_pred) == len(y_target), f'Error: Number of records do not match {len(y_pred)} != {len(y_target)}'

  # Calculate Accuracy  
  accuracy = evaluate_accuracy(y_pred, y_target, ncrops)
  
  # Calculate Inference Duration
  duration = np.array([get_duration(l) for l in lines])
 
  model_name = get_model_name(lines[-1])

  result = {
      'model' : model_name,
      'log' : log_path,
      'src_data' : npz_path,
      'target' : target,
      'accuracy' : accuracy,
      'inference' : {
          'unit' : duration_unit,
          'min' : duration.min(),
          'max' : duration.max(),
          'mean' : duration.mean(),
      }
  }

  print(f'Model    {result["model"]}')
  print(f'Target   {result["target"]}')
  print(f'Accuracy {result["accuracy"]}')
  print(f'Inference duration ({result["inference"]["unit"]})')
  print(f' Min     {result["inference"]["min"]}')
  print(f' Max     {result["inference"]["max"]}')
  print(f' Mean    {result["inference"]["mean"]}')
  
  assert int(last_line[feature_number_index]) + 1 == len(y_target), \
    f"Error: Expected {len(y_target) + 1} records. Got {last_line[feature_number_index]}"
  
  report_datetime = datetime.datetime.now()
  results_path = f'results/activity_{report_datetime:%y%m%d_%H%M}.npy'
  np.save(results_path, result, allow_pickle=True)

  print(f'\nResults written to: {results_path}')

if __name__ == '__main__':  

  if len(sys.argv) > 1:
    assert len(sys.argv) == 3, \
      f'Expected 2 arguments: {sys.argv[0]} path/to/activity.log path/to/npz_source'

    activity_path = sys.argv[1]
    npz_path = sys.argv[2]
  
  assert os.path.exists(activity_path), \
    f'Activity log not found with path: {activity_path}'

  assert os.path.exists(npz_path), \
    f'Numpy npz source file not found with path: {npz_path}'
    
  # Evaluate the supplied activity log
  evaluate_log(activity_path, npz_path, crop_count)  
