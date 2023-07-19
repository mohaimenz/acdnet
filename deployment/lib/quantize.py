import tensorflow as tf
import numpy as np

def quantize_int8(x, axis):
  '''Quantization into int8_t precision, operating on x along axis'''

  scaling_factor_shape = tuple(np.append([len(x)],np.ones(x.ndim - 1, dtype = int)))
  epsilon = 0.000000001
  x_scaling_factor = (2 * np.max(np.abs(x), axis) / 255) + epsilon
  x_scaling_factor = x_scaling_factor.reshape(scaling_factor_shape)
  x_zero_offset = -0.5
  result = (x / x_scaling_factor) + x_zero_offset

  return np.rint(result).astype(np.int8)

def quantize_int16(x, axis):
  '''Quantization into int16_t precision, operating on x along axis'''
  
  scaling_factor_shape = tuple(np.append([len(x)],np.ones(x.ndim - 1, dtype = int)))
  epsilon = 0.00000000001
  x_scaling_factor = (2 * np.max(np.abs(x), axis) / 65535) + epsilon  
  x_scaling_factor = x_scaling_factor.reshape(scaling_factor_shape)  
  x_zero_offset = -0.5  
  result = (x / x_scaling_factor) + x_zero_offset
  
  return np.rint(result).astype(np.int16)

def quantize_uint8(x, axis):
  '''Quantization into uint8_t precision, operating on x along axis'''
  
  scaling_factor_shape = tuple(np.append([len(x)],np.ones(x.ndim - 1, dtype = int)))
  epsilon = 0.000000001
  x_scaling_factor = (2 * np.max(np.abs(x), axis) / 255) + epsilon  
  x_scaling_factor = x_scaling_factor.reshape(scaling_factor_shape)  
  x_zero_offset = 255 / 2.0  
  result = (x / x_scaling_factor) + x_zero_offset
  
  return np.rint(result).astype(np.uint8)

def quantize_float16(x, axis):
  '''Quantization into float16 precision, operating on x along axis'''
  
  scaling_factor_shape = tuple(np.append([len(x)],np.ones(x.ndim - 1, dtype = int)))
  epsilon = 0.000000001
  x_scaling_factor = (2 * np.max(np.abs(x), axis) / 2.0) + epsilon  
  x_scaling_factor = x_scaling_factor.reshape(scaling_factor_shape)  
  x_zero_offset = 0.0 
  result = (x / x_scaling_factor) + x_zero_offset

  return result.astype(np.float16)

def quantize_float32(x, axis):
  '''Quantization into float32 precision, operating on x along axis'''
  
  scaling_factor_shape = tuple(np.append([len(x)],np.ones(x.ndim - 1, dtype = int)))
  epsilon = 0.000000001
  x_scaling_factor = (2 * np.max(np.abs(x), axis) / 2.0) + epsilon  
  x_scaling_factor = x_scaling_factor.reshape(scaling_factor_shape)  
  x_zero_offset = 0.0 
  result = (x / x_scaling_factor) + x_zero_offset

  return result.astype(np.float32)

def get_cast(dtype):
  '''Helper to get a quantization method based on the target dtype'''

  if dtype == tf.float32 or dtype == np.float32 or dtype =='float32':
    print('Casting dataset to float32')
    return quantize_float32

  if dtype == tf.float16 or dtype == np.float16 or dtype =='float16':
    print('Casting dataset to float16')
    return quantize_float16

  if dtype == tf.int8 or dtype == int8 or dtype =='int8':
    print('Casting dataset to int8')
    return quantize_int8

  if dtype == tf.uint8 or dtype == np.uint8 or dtype =='uint8':
    print('Casting dataset to uint8')
    return quantize_uint8

  if dtype == tf.int16 or dtype == int16 or dtype =='int16':
    print('Casting dataset to int16')
    return quantize_int16

def quantization_tests():
  '''Simple unit tests for quantization'''
  test_x = np.array([[[[-0.3],[0.1],[0.0],[0.2]]],[[[0.4],[0.1],[0.0],[0.2]]], [[[0.4],[-0.4],[0.4],[-0.4]]], [[[0.0],[0.0],[0.0],[0.0]]]])

  quant_int8_actual= quantize_int8(test_x, axis = -2).flatten()
  quant_int8_expect = np.array([-128, 42, 0, 84, 127, 31, 0, 63, 127, -128, 127, -128, 0, 0, 0, 0], dtype=int)
  assert np.array_equal(quant_int8_actual,quant_int8_expect), "INT8 quantization failed"

  quant_uint8_actual= quantize_uint8(test_x, axis = -2).flatten()
  quant_uint8_expect = np.array([0, 170, 128, 212, 255, 159, 128, 191, 255, 0, 255, 0, 128, 128, 128, 128], dtype=np.uint8)
  assert np.array_equal(quant_uint8_actual,quant_uint8_expect), "UINT8 quantization failed"

  quant_int16_actual= quantize_int16(test_x, axis = -2).flatten()
  quant_int16_expect = np.array([-32768, 10922, 0, 21844, 32767, 8191, 0, 16383, 32767, -32768, 32767, -32768, 0, 0, 0, 0], dtype=np.int16)
  assert np.array_equal(quant_int16_actual,quant_int16_expect), "INT16 quantization failed"

  quant_float16_actual= quantize_float16(test_x, axis = -2).flatten()
  quant_float16_expect = np.array([-1., 0.3333, 0., 0.6665, 1., 0.25, 0., 0.5, 1., -1., 1., -1., 0., 0., 0., 0.], dtype=np.float16)
  assert np.array_equal(quant_float16_actual,quant_float16_expect), "FLOAT16 quantization failed"

  quant_float32_actual= quantize_float32(test_x, axis = -2).flatten()
  quant_float32_expect = np.array([-1., 0.33333334, 0., 0.6666667, 1., 0.25, 0., 0.5, 1., -1., 1., -1., 0., 0., 0., 0.], dtype=np.float32)
  assert np.array_equal(quant_float32_actual,quant_float32_expect), "FLOAT32 quantization failed"
