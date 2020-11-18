
import tensorflow as tf

data_support = {
    'nano-data-20khz' : {
        'path' : 'data/nano/20000', 
        'input_size' : 30225,
        'test' : {
            'file_name':'test4000.npz',            
            'url' : 'https://drive.google.com/uc?export=download&id=1lgUgMIqAS5mdelnQI1KdnketjtQJrkui',
        }        
    },
    'aug-data-20khz' : {
        'path' : 'data/esc50/20000',
        'input_size' : 30225,
        'test' : {
            'file_name':'test4000.npz',
            'url': 'https://drive.google.com/uc?export=download&id=1Grhy7jSZr6EOzfs0m_EybkQAg1Hy5x7y',
        },
        'representative' : {
            'file_name' : 'train1.npz',
            'url' : 'https://drive.google.com/uc?export=download&id=1-06FvgxTF1jTWhj9p3Z1MBd4OFohsFBE'
        }        
    },
    'aug-data-44.1khz' : {
        'path' : 'data/esc50/44100',
        'input_size' : 66650,
        'test' : {
            'file_name':'test4000.npz',
            'url': 'https://drive.google.com/uc?export=download&id=162amC8tZLaf05CqLOW1juTX731yprwXI',
        },
        'representative' : {
            'file_name':'train1.npz',
            'url' : 'https://drive.google.com/uc?export=download&id=1fd1bkuVoQQ-quoFKh7yQWZYQqbjdRBD6'
        }        
    }
}

quant_support = {
    'int8' : {
        'supported_ops' : [tf.lite.OpsSet.TFLITE_BUILTINS_INT8], 'input_type' : tf.int8, 'output_type' : tf.int8
    },
    'int16' : {
        'supported_ops' : [tf.lite.OpsSet.TFLITE_BUILTINS], 'supported_types' : [tf.int16], 'input_type' : tf.float32, 'output_type' : tf.float32
    },
    'int16_8' : {
        'supported_ops' : [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8], 'supported_types' : [tf.int16], 'input_type' : tf.int16, 'output_type' : tf.int16
    },
    'uint8' : {
        'supported_ops' : [tf.lite.OpsSet.TFLITE_BUILTINS_INT8], 'input_type' : tf.uint8, 'output_type' : tf.uint8
    },
    'float16' : {
        'supported_types' : [tf.float16]
    },
    'float32' : {
        'supported_ops' : [tf.lite.OpsSet.TFLITE_BUILTINS], 'supported_types' : [tf.float32], 'input_type' : tf.float32, 'output_type' : tf.float32
    },
}