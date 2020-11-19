-include $(TOPDIR)/Make.defs
-include $(SDKDIR)/Make.defs

CUR_DIR=$(shell pwd)

# Check if libcxx is installed and is configured.
CHECK_DIR=$(shell ls $(TOPDIR)/include/libcxx)
ifeq ($(CHECK_DIR),)
$(error libcxx is not installed. please execute install_licxx.sh in $(CUR_DIR))
endif

ifeq ($(CONFIG_LIBCXX),)
$(error libcxx is needed for Tensorflow Lite Micro)
endif

PROGNAME = $(CONFIG_EXAMPLES_TFLITE_MICRO_PROGNAME)
PRIORITY = $(CONFIG_EXAMPLES_TFLITE_MICRO_PRIORITY)
STACKSIZE = $(CONFIG_EXAMPLES_TFLITE_MICRO_STACKSIZE)
MODULE = $(CONFIG_EXAMPLES_TFLITE_MICRO)

TENSORFLOW_GITHUB_URL = https://github.com/tensorflow/tensorflow.git
TENSORFLOW_TAG = $(CONFIG_EXAMPLES_TFLITE_MICRO_TENSORFLOW_VERSION)

# C++ file expression of NuttX default is .cxx but Tensorflow's is .cc,
# So CXXEXT should be changed to .cc
CXXEXT = .cc

# Tensorflow and related library PATHs.
TENSORFLOW_DIR = tensorflow
TENSORFLOW_LITE = $(TENSORFLOW_DIR)/tensorflow/lite
TENSORFLOW_LM = $(TENSORFLOW_LITE)/micro
TENSORFLOW_LM_DOWNLOADS = $(TENSORFLOW_LM)/tools/make/downloads
TENSORFLOW_LM_CMSIS = $(TENSORFLOW_LM_DOWNLOADS)/cmsis/CMSIS
TENSORFLOW_LM_GEMMLOWP = $(TENSORFLOW_LM_DOWNLOADS)/gemmlowp
TENSORFLOW_LM_FLATBUFFERS = $(TENSORFLOW_LM_DOWNLOADS)/flatbuffers/include
TENSORFLOW_LM_RUY = $(TENSORFLOW_LM_DOWNLOADS)/ruy 

#Models path
MODELS_DIR = ../models

# Tensorflow and related library's C++ sources.
TF_LM_CXXSRCS = \
	$(TENSORFLOW_LITE)/core/api/error_reporter.cc	\
	$(TENSORFLOW_LITE)/core/api/flatbuffer_conversions.cc	\
	$(TENSORFLOW_LITE)/core/api/op_resolver.cc	\
	$(TENSORFLOW_LITE)/core/api/tensor_utils.cc	\
	$(TENSORFLOW_LITE)/kernels/internal/quantization_util.cc	\
	$(TENSORFLOW_LITE)/kernels/kernel_util.cc	\
	$(TENSORFLOW_LITE)/micro/testing/test_utils.cc \
	$(TENSORFLOW_LM)/simple_memory_allocator.cc \
	$(TENSORFLOW_LM)/micro_error_reporter.cc \
	$(TENSORFLOW_LM)/all_ops_resolver.cc \
	$(TENSORFLOW_LM)/memory_helpers.cc \
	$(TENSORFLOW_LM)/micro_utils.cc \
	$(TENSORFLOW_LM)/recording_micro_allocator.cc \
	$(TENSORFLOW_LM)/micro_time.cc \
	$(TENSORFLOW_LM)/recording_simple_memory_allocator.cc \
	$(TENSORFLOW_LM)/micro_string.cc \
	$(TENSORFLOW_LM)/micro_profiler.cc \
	$(TENSORFLOW_LM)/micro_optional_debug_tools.cc \
	$(TENSORFLOW_LM)/micro_allocator.cc \
	$(TENSORFLOW_LM)/test_helpers.cc \
	$(TENSORFLOW_LM)/micro_interpreter.cc \
	$(TENSORFLOW_LM)/benchmarks/keyword_scrambled_model_data.cc \
	$(TENSORFLOW_LM)/kernels/circular_buffer.cc \
	$(TENSORFLOW_LM)/kernels/cmsis-nn/pooling.cc \
	$(TENSORFLOW_LM)/kernels/prelu.cc \
	$(TENSORFLOW_LM)/kernels/cmsis-nn/softmax.cc \
	$(TENSORFLOW_LM)/kernels/cmsis-nn/depthwise_conv.cc \
	$(TENSORFLOW_LM)/kernels/dequantize.cc \
	$(TENSORFLOW_LM)/kernels/pad.cc \
	$(TENSORFLOW_LM)/kernels/ethosu.cc \
	$(TENSORFLOW_LM)/kernels/l2norm.cc \
	$(TENSORFLOW_LM)/kernels/resize_nearest_neighbor.cc \
	$(TENSORFLOW_LM)/kernels/tanh.cc \
	$(TENSORFLOW_LM)/kernels/transpose.cc \
	$(TENSORFLOW_LM)/kernels/activations.cc \
	$(TENSORFLOW_LM)/kernels/arg_min_max.cc \
	$(TENSORFLOW_LM)/kernels/reduce.cc \
	$(TENSORFLOW_LM)/kernels/unpack.cc \
	$(TENSORFLOW_LM)/kernels/cmsis-nn/add.cc \
	$(TENSORFLOW_LM)/kernels/sub.cc \
	$(TENSORFLOW_LM)/kernels/floor.cc \
	$(TENSORFLOW_LM)/kernels/round.cc \
	$(TENSORFLOW_LM)/kernels/svdf.cc \
	$(TENSORFLOW_LM)/kernels/concatenation.cc \
	$(TENSORFLOW_LM)/kernels/cmsis-nn/conv.cc \
	$(TENSORFLOW_LM)/kernels/neg.cc \
	$(TENSORFLOW_LM)/kernels/quantize.cc \
	$(TENSORFLOW_LM)/kernels/elementwise.cc \
	$(TENSORFLOW_LM)/kernels/cmsis-nn/fully_connected.cc \
	$(TENSORFLOW_LM)/kernels/maximum_minimum.cc \
	$(TENSORFLOW_LM)/kernels/reshape.cc \
	$(TENSORFLOW_LM)/kernels/strided_slice.cc \
	$(TENSORFLOW_LM)/kernels/pack.cc \
	$(TENSORFLOW_LM)/kernels/logical.cc \
	$(TENSORFLOW_LM)/kernels/cmsis-nn/mul.cc \
	$(TENSORFLOW_LM)/kernels/comparisons.cc \
	$(TENSORFLOW_LM)/kernels/ceil.cc \
	$(TENSORFLOW_LM)/kernels/split.cc \
	$(TENSORFLOW_LM)/kernels/logistic.cc \
	$(TENSORFLOW_LM)/memory_planner/linear_memory_planner.cc \
	$(TENSORFLOW_LM)/memory_planner/greedy_memory_planner.cc \
	$(TENSORFLOW_LM)/testing/test_conv_model.cc

# Tensorflow and related library's C sources.
TF_LM_CSRCS = \
	$(TENSORFLOW_LITE)/c/common.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/BasicMathFunctions/arm_elementwise_mul_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/BasicMathFunctions/arm_elementwise_add_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7_opt.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/FullyConnectedFunctions/arm_fully_connected_q15_opt.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/FullyConnectedFunctions/arm_fully_connected_q15.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/FullyConnectedFunctions/arm_fully_connected_mat_q7_vec_q15.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/FullyConnectedFunctions/arm_fully_connected_mat_q7_vec_q15_opt.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConcatenationFunctions/arm_concatenation_s8_z.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConcatenationFunctions/arm_concatenation_s8_x.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConcatenationFunctions/arm_concatenation_s8_w.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConcatenationFunctions/arm_concatenation_s8_y.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_depthwise_conv_u8_basic_ver1.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_convolve_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_convolve_1x1_HWC_q7_fast_nonsquare.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7_nonsquare.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast_nonsquare.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_basic.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast_nonsquare.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16_reordered.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_depthwise_separable_conv_HWC_q7.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/PoolingFunctions/arm_pool_q7_HWC.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/PoolingFunctions/arm_avgpool_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/PoolingFunctions/arm_max_pool_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/PoolingFunctions/arm_max_pool_s8_opt.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ActivationFunctions/arm_relu_q15.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ActivationFunctions/arm_relu_q7.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ActivationFunctions/arm_nn_activations_q7.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ActivationFunctions/arm_relu6_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ActivationFunctions/arm_nn_activations_q15.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_nn_add_q7.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_nntables.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_nn_accumulate_q7_to_q15.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_nn_mult_q7.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_nn_mult_q15.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_with_offset.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/NNSupportFunctions/arm_q7_to_q15_no_shift.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/SoftmaxFunctions/arm_softmax_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/SoftmaxFunctions/arm_softmax_q15.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/SoftmaxFunctions/arm_softmax_u8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/SoftmaxFunctions/arm_softmax_q7.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/SoftmaxFunctions/arm_softmax_with_batch_q7.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ReshapeFunctions/arm_reshape_s8.c \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic_nonsquare.c


NO_FILE = \
	$(TENSORFLOW_LM)/kernels/kernel_runner.cc \
	$(TENSORFLOW_LM)/kernels/hard_swish.cc \
	$(TENSORFLOW_LM)/kernels/kernel_util.cc \
	$(TENSORFLOW_LM_CMSIS)/NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c

ifeq ($(CONFIG_EXAMPLES_TFLITE_MICRO_DEBUGLOG),y)
TF_DEBUG_FLAG := -DDEBUG
else
TF_DEBUG_FLAG := -DNDEBUG -DTF_LITE_STRIP_ERROR_STRINGS
endif

TF_COMPILE_FLAGS = \
	-DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK	\
	-D__ARM_FEATURE_MVE=0 \
	-DTF_LITE_STATIC_MEMORY	\
	-DTF_LITE_MCU_DEBUG_LOG \
	$(TF_DEBUG_FLAG) \
	-I$(TENSORFLOW_DIR) \
	-I$(TENSORFLOW_LM_DOWNLOADS) \
	-I$(TENSORFLOW_LM_CMSIS)/Core/Include \
	-I$(TENSORFLOW_LM_CMSIS)/NN/Include \
	-I$(TENSORFLOW_LM_GEMMLOWP) \
	-I$(TENSORFLOW_LM_FLATBUFFERS) \
	-I$(TENSORFLOW_LM_RUY)

CFLAGS += -nostdinc $(TF_COMPILE_FLAGS)
CXXFLAGS += -nostdinc++ $(TF_COMPILE_FLAGS)

ASRCS =
CSRCS = $(TF_LM_CSRCS)
CXXSRCS = $(TF_LM_CXXSRCS) \
					model.cc \
					debug_log.cc \
					nn_manager.cc \
					nn_score.cc

MAINSRC = tf_main.cc

clean:: tensorflow_lite_clean

distclean:: tensorflow_lite_clean

tensorflow_lite_clean:
	@rm -f `find $(TENSORFLOW_DIR) -name "*.o"`

delete_dfiles:
	@rm -f *.d

include $(APPDIR)/Application.mk
