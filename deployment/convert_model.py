#!/usr/bin/env python3

import os
import sys
import datetime
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras

from lib import *
from results import *
from  data_loader import Trainer

def main(model_path, dtype, results_path, dest_path, trainer):
  '''This operation converts the models into TFLite and c++ format'''
  
  assert dtype in quant_support.keys(), \
    f'Unknown quantization target dtype {dtype}'

  cwd_path = os.path.dirname(os.path.abspath(__file__))
  crop_size = 10

  results = []
  report_datetime = datetime.datetime.now()  
  results_path = os.path.join(results_path, f'results_{report_datetime:%y%m%d_%H%M}.npy')

  converter = KerasConverter(model_path, dtype, dest_path, trainer)
  converter.load_model()
  input_size, output_size = converter.get_model_size()

  try:    
    # Conversion
    converter.generate_tflite()

    # Evaluation
    print(f'Calculating TF Accuracy : {model_path}')
    tf_accuracy, tf_pred, tf_data = converter.get_tf_accuracy(crop_size)
      
    print(f'Calculating TFLite Accuracy : {model_path} {dtype}')
    tflite_accuracy, tflite_pred, tflite_data = converter.get_tflite_accuracy(crop_size)
    tflite_arena_size = converter.get_arena_size()

    print(f'Final accuracy : {model_path}')
    print(f'TF Accuracy : {tf_accuracy}')            
    print(f'TFLite Accuracy ({dtype}) : {tflite_accuracy}')
    print(f'Arena size (approx): {tflite_arena_size}')

    fold = trainer.opt.split
    dataset = trainer.opt.dataset

    # Summarize
    result = {
      'model' : model_path,
      'dtype' : dtype,
      'fold' :  fold,
      'dataset' : dataset,
      'tf_accuracy' :tf_accuracy,
      'tf_pred' : tf_pred,
      'tf_data' : tf_data,
      'tflite_accuracy' : tflite_accuracy,
      'tflite_pred' : tflite_pred,
      'tflite_data' : tflite_data,
      'tflite_arena' : tflite_arena_size,
      'tf_summary' : converter.get_tf_summary(),
      'tflite_summary' : converter.get_tflite_summary(),
      'input_size' : input_size,
      'output_size' : output_size
    }

    results.append(result)

    np.save(results_path,results,allow_pickle=True)

    # Cleanup
    model_file = os.path.basename(model_path)

    src_tflite_path = os.path.join(dest_path,'g_model.tflite')
    src_cc_path = os.path.join(dest_path,'g_model.cc')

    # Convert TFLite to CC file, for inclusion in c++ applications
    os.system(f'xxd -i {src_tflite_path} > {src_cc_path}')

    # Copy tflite file into models folder
    dest_tflite_name = \
      f'{model_file}_{dtype}_{dataset}_fold{fold}.tflite'
    
    # Embed model metadata in cc file
    modify_cc_file(src_cc_path, dest_tflite_name, input_size, tflite_arena_size)

    dest_tflite_path = \
        os.path.join(cwd_path, 'src','models',dest_tflite_name)
    
    shutil.copy(src_tflite_path, dest_tflite_path)

    # Copy CC file into models folder
    dest_cc_name = \
      f'{model_file}_{dtype}_{dataset}_fold{fold}.cc'
    
    dest_cc_path = \
      os.path.join(cwd_path, 'src','models', dest_cc_name)

    shutil.copy(src_cc_path, dest_cc_path)

    # Install into TFLITE Micro
    tflite_micro_model_path = \
      os.path.join(cwd_path,'src','tflite_micro', 'model.cc')

    shutil.copy(src_cc_path,  tflite_micro_model_path)

    # Install into TFLITE_x86_64 
    tflite_micro_model_path = \
      os.path.join(cwd_path,'src','tflite_x86_64','app', 'model.cpp')

    shutil.copy(src_cc_path,  tflite_micro_model_path)

    # Remove temp files
    os.unlink(src_tflite_path)
    os.unlink(src_cc_path)

  except Exception as ex:
      print(f'Error: {str(ex)}')
  
  print_results(results_path)

def modify_cc_file(src_path, model_name, feature_size, arena_size): 
  with open(src_path, 'r') as original:
    data = original.read()
  
  gen_model_name = data.splitlines()[0].split(' ')[2][:-2]

  data = data.replace(gen_model_name, 'g_model_tflite')
  data = data.replace('unsigned char', 'const unsigned char')
  data = data.replace('unsigned int', 'const unsigned int')
  data = data.replace('g_model_tflite[] = {','g_model_tflite[] DATA_ALIGN_ATTRIBUTE = {')
  data = f'#include "model.h"\n\n{data}'
  data = f'{data}\nconst char g_model_name[] = "{str(model_name)}";'
  data = f'{data}\nconst unsigned int g_arena_size = {int(arena_size)};'
  data = f'{data}\nconst unsigned int g_feature_size = {int(feature_size)};\n\n'

  with open(src_path, 'w') as modified:
    modified.write(data)


class KerasConverter():
  def __init__(self, model_file, dtype, dest_path, trainer:Trainer):
    '''Initialise the Keras model converter'''

    self.model_file = model_file
    self.dtype = dtype
    self.quant_support = quant_support[dtype]
    self.tflite_path = f'{dest_path}/g_model.tflite'
    self.cc_path = f'{dest_path}/model.cc'
    self.h_path = f'{dest_path}/model.h'
    self.trainer = trainer    

  def load_model(self):
    print(f'Loading model: {self.model_file}')
    self.model = keras.models.load_model(self.model_file)
  
  def get_model_size(self):
    '''Returns the input size and output size'''
    return self.model.inputs[0].shape[-2], self.model.outputs[0].shape[-1]

  def get_tf_summary(self):
    '''Displays a model summary'''
    stringlist = []
    self.model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    return short_model_summary

  def get_tflite_summary(self):
    '''Displays a model summary'''
    with tf.io.gfile.GFile(self.tflite_path, 'rb') as f:
      model_content = f.read()
        
    interpreter = tf.lite.Interpreter(model_content = model_content)
    interpreter.allocate_tensors()

    return (interpreter.get_tensor_details(), \
      interpreter._get_ops_details())

  def get_arena_size(self):
    '''Returns the approx tensor_arena size'''
    tensor_details, op_details = self.get_tflite_summary()

    get_tensor = lambda index : [t for t in tensor_details if t['index'] == index][0]
    get_node_tensors = lambda n : [get_tensor(t) for t in np.concatenate((n['inputs'],n['outputs']), axis= None)]    
    get_tensor_size = lambda t : np.prod(t['shape']) * np.dtype(t['dtype']).itemsize
    get_node_tensor_sizes = lambda o : np.sum([get_tensor_size(t) for t in get_node_tensors(o)])   
    get_max_node_size = lambda : np.max([get_node_tensor_sizes(o) for o in op_details])

    return get_max_node_size() 

  def get_input_data(self):
    '''Retrieves the input data set'''
    x,y = self.trainer.testX, self.trainer.testY
    return x, y

  def get_cast_input_data(self, dtype = None):
    '''Retrieves the input data set, casting to target dtype'''
    x, y = self.get_input_data()

    if dtype is None:
      print('dtype not provided')
      return x,y   
    return get_cast(dtype)(x, axis = -2), y
  
  def get_rep_data(self, dtype = None):
    '''Retrieves the reprepresentative data set'''
    x, y = self.trainer.trainX, self.trainer.trainY

    if dtype is None:
      return x,y
    return get_cast(dtype)(x, axis=-2), y

  def evaluate_accuracy(self, y_pred, y_target, crops = 1):
    '''A common accuracy operation which supports multi-crop'''
    y_pred = y_pred.reshape(y_pred.shape[0]//crops, crops, y_pred.shape[1])
    y_target = y_target.reshape(y_target.shape[0]//crops, crops, y_target.shape[1])

    #Calculate the average of class predictions for 10 crops of a sample
    y_pred = np.mean(y_pred, axis=1)
    y_target = np.mean(y_target,axis=1)

    #Get the indices that has highest average value for each sample
    y_pred = y_pred.argmax(axis=1)
    y_target = y_target.argmax(axis=1)

    accuracy = (y_pred==y_target).mean()
    return accuracy

  def predict_tf(self, x_data):
    '''Calculate the output of a single inference of the TF model'''
    x = tf.expand_dims(x_data, 0).numpy()
    return self.model.predict([x])

  def get_tf_accuracy(self, crops = 1):
    '''Calculate accuracy of the TF model'''
    x_data, y_data = self.get_input_data()

    y_pred = self.model.predict([x_data])

    accuracy = self.evaluate_accuracy(y_pred, y_data, crops)
    return accuracy, y_pred, y_data

  def predict_tflite(self, x_data):
    '''Calculate output of single inference of the TFLite model'''
    with tf.io.gfile.GFile(self.tflite_path, 'rb') as f:
      model_content = f.read()
        
    interpreter = tf.lite.Interpreter(model_content = model_content)
    interpreter.allocate_tensors()
    
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    input_dtype = interpreter.get_input_details()[0]['dtype']
    x = get_cast(input_dtype)(x_data, axis=-2)

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()

    return interpreter.get_tensor(output_index)[0]

  def get_tflite_accuracy(self, crops = 1):
    '''Calculate the accuracy of the TFLite model'''
    with tf.io.gfile.GFile(self.tflite_path, 'rb') as f:
      model_content = f.read()
        
    interpreter = tf.lite.Interpreter(model_content = model_content)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    input_dtype = interpreter.get_input_details()[0]['dtype']
    print(f'Input dtype {str(input_dtype)}')

    x_data, y_data = self.get_cast_input_data(input_dtype)

    def predict(x_input):
      
      x_input = tf.expand_dims(x_input, 0).numpy()
      interpreter.set_tensor(input_index, x_input)
      # Run inference.
      interpreter.invoke()
      return interpreter.get_tensor(output_index)[0]
    
    y_pred = np.array([predict(x) for x in x_data])
    print(y_pred.shape)
    print(y_data.shape)
    accuracy = self.evaluate_accuracy(y_pred, y_data, crops)
    return accuracy, y_pred, y_data  

  def generate_tflite(self):
    '''Generates a TFLite file from a Keras model'''

    # Construction of a TFLite converter
    tf_converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
    
    tf_converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if 'supported_ops' in self.quant_support:
      print(f'Targetting Supported Ops {self.quant_support["supported_ops"]}')
      tf_converter.target_spec.supported_ops = self.quant_support['supported_ops']

    if 'supported_types' in self.quant_support:
      print(f'Targetting Supported Types{self.quant_support["supported_types"]}')
      tf_converter.target_spec.supported_types = self.quant_support['supported_types']

    if 'input_type' in self.quant_support:
      print(f'Targetting input type : {self.quant_support["input_type"]}')
      tf_converter.inference_input_type = self.quant_support['input_type']

    if 'output_type' in self.quant_support:
      print(f'Targetting output type : {self.quant_support["output_type"]}')
      tf_converter.inference_output_type = self.quant_support['output_type']

    # Supplying a representative dataset is required for full integer 
    # quantization, and also avoids dynamic range quantization

    rep_data, _ = self.get_rep_data(None)
    print(f'Representative dataset dtype : {rep_data.dtype}')

    def representative_dataset_no_padding():
      for i in range(len(rep_data)):
        if rep_data[i:i+1,:,0,:] != 0 and rep_data[i:i+1,:,-1,:] != 0:
          yield([rep_data[i:i+1,:,:,:]])

    def representative_dataset():
      for i in range(len(rep_data)):
        yield([rep_data[i:i+1,:,:,:]])

    tf_converter.representative_dataset = representative_dataset

    tflite_model = tf_converter.convert()
    bytes_written = open(self.tflite_path, 'wb').write(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input_type = interpreter.get_input_details()[0]['dtype']
    output_type = interpreter.get_output_details()[0]['dtype']

    print('TFLite input dtype : ', input_type)
    print('TFLite output dtype : ', output_type)
    
    return bytes_written

if __name__ == '__main__':

    opt = parse()
    
    assert os.path.exists(opt.model), \
      f'File not found: {opt.model}\n'

    # Setup environment
    run_tests()

    # Setup required variables
    opt = opts.parse();
    
    dtype = 'int8'
    
    cwd_path = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(cwd_path, 'results')
    tmp_path = os.path.join(cwd_path, 'tmp')      

    # Get the data sets
    trainer = Trainer(opt);
    trainer.load_training_data();
    trainer.load_test_data();

    # Start conversion
    main(sys.argv[1], dtype, result_path, tmp_path, trainer)
