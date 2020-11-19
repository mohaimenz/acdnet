import tensorflow as tf

crops = 10
feature_count = 4000

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