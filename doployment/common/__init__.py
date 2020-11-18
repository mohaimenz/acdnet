from common.bytes import to_bytes, from_bytes, byte_conversion_tests, load_data, load_raw, save_raw, save_scores
from common.constants import data_support, quant_support
from common.gdrive import download,  get_data_path
from common.quantize import quantization_tests, get_cast

def run_tests():
    byte_conversion_tests()
    quantization_tests()