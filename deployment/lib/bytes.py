import struct
import math
import numpy as np
import os

def to_bytes(x_data, dtype, endian):
  '''Gets the content of scalar array x_data, as a byte array'''
  assert endian in ['little','big']
  if dtype == 'int8':
    format = f'{len(x_data)}b'   # signed char
  elif dtype == 'uint8':
    format = f'{len(x_data)}B'   # unsigned char
  elif dtype == 'int16':
    format = f'<{len(x_data)}h' if endian == 'little' else f'>{len(x_data)}h'  # short
  elif dtype == 'float16':
    format = f'<{len(x_data)}e' if endian == 'little' else f'>{len(x_data)}e' # float16
  elif dtype == 'float32':
    format = f'<{len(x_data)}f' if endian == 'little' else f'>{len(x_data)}f' # float32    

  return struct.pack(format, *x_data)

def from_bytes(byte_data, dtype, endian):
  '''Converts from byte array to scalar array'''
  assert endian in ['little','big']
  if dtype == 'int8':
    return struct.unpack(f'{len(byte_data)}b', byte_data)
  if dtype == 'uint8':
    return struct.unpack(f'{len(byte_data)}B', byte_data)
  if dtype == 'int16':
    return struct.unpack(f'<{int(len(byte_data) / 2)}h', byte_data) if endian == 'little' \
        else struct.unpack(f'>{int(len(byte_data) / 2)}h', byte_data)
  if dtype == 'float16':
    return struct.unpack(f'<{int(len(byte_data) / 2)}e', byte_data) if endian == 'little' \
        else struct.unpack(f'>{int(len(byte_data) / 2)}e', byte_data)
  if dtype == 'float32':
    return struct.unpack(f'<{int(len(byte_data) / 4)}f', byte_data) if endian == 'little' \
        else struct.unpack(f'>{int(len(byte_data) / 4)}f', byte_data)

def byte_conversion_tests():
  '''Simple byte conversion unit tests'''  

  for endian in ['little', 'big']:
    src = np.array([1,2,3,4,5])   
    assert np.array_equal(from_bytes(to_bytes(src, 'int8', endian),'int8', endian), src), "INT8 failed"
    assert np.array_equal(from_bytes(to_bytes(src,'int16', endian),'int16', endian), src), "INT16 failed"
    assert np.array_equal(from_bytes(to_bytes(src,'uint8', endian),'uint8', endian), src), "UINT8 failed"
    assert np.array_equal(from_bytes(to_bytes(src,'float16', endian),'float16', endian), src), "float16 failed"
    assert np.array_equal(from_bytes(to_bytes(src,'float32', endian),'float32', endian), src), "float32 failed"
    
    src = np.array([1.0,-1.0,0.0])
    assert np.array_equal(from_bytes(to_bytes(src,'float16', endian),'float16', endian), src), "float16 failed"
    assert np.array_equal(from_bytes(to_bytes(src,'float32', endian),'float32', endian), src), "float32 failed"
    
    src = np.array([-1,0,1])
    assert np.array_equal(from_bytes(to_bytes(src, 'int8', endian),'int8', endian), src), "INT8 failed"
    assert np.array_equal(from_bytes(to_bytes(src,'int16', endian),'int16', endian), src), "INT16 failed"

def load_raw(source, dtype, endian = 'little', dest_path = 'data'):
  '''Loads RAW files from a directory path, as a data set'''
  raw_files = [f for f in os.listdir(dest_path) if f.endswith('.RAW')]  
  x_data = []
  index = 0
  crop = 0
  filename = lambda i,c: f'{i:04}_{c:03}.RAW'
  while filename(index,crop) in raw_files:
    with open(os.path.join(dest_path,filename(index,crop)), 'rb') as fp:
      values = from_bytes(fp.read(), dtype, endian)
      x_data.append(np.array(values))

    index += 1
    crop = math.floor(index / 10)
  
  return np.array(x_data)

def save_raw(x_data, dtype, endian = 'little', dest_path = 'data'):
  '''Saves a dataset into a RAW file format'''
  
  filename = lambda i,c: f'{i:04}_{c:03}.RAW'
  os.makedirs(os.path.dirname(dest_path), exist_ok=True)
  for i in range(len(x_data)):
    crop = math.floor(i/10)

    with open(os.path.join(dest_path,filename(i,crop)), 'wb') as fp:
      byte_data = to_bytes(x_data[i].flatten(), dtype, endian)
      fp.write(byte_data)

  print(f'Saved ({len(x_data)}) RAW files: {os.path.dirname(dest_path)}')

def save_scores(y_data, dtype, endian='little', dest_path = 'data'):
  '''Saves a simple representation of scores, for c++ use'''
  os.makedirs(os.path.dirname(dest_path), exist_ok=True)
  
  argmax_scores = np.argmax(y_data, axis = 1)

  print(y_data)
  print(y_data.shape)

  print(argmax_scores)
  print(argmax_scores.shape)  

  with open(os.path.join(dest_path, 'scores.bin'),'wb') as fp:
    byte_data = to_bytes(argmax_scores, dtype, endian)
    fp.write(byte_data)
  
  print(f'Saved expected scores: {dest_path}')

def load_data(data_path):
  '''Loads x and y arrays from numpy npz dataset'''
  with np.load(data_path, allow_pickle=True) as data:
    return data['x'], data['y']