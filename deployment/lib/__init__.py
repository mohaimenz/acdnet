from lib.bytes import to_bytes, from_bytes, byte_conversion_tests, load_data, load_raw, save_raw, save_scores
from lib.constants import quant_support, crops, feature_count
from lib.quantize import quantization_tests, get_cast
from lib.opts import parse

def run_tests():
    byte_conversion_tests()
    quantization_tests()