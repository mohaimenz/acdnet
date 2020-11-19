#ifndef _MODEL_H
#define _MODEL_H

// We need to keep the data array aligned on some architectures.
#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif
#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif

extern const unsigned char g_model_tflite[];
extern const unsigned int g_model_tflite_len;
extern const char g_model_name[];
extern const unsigned int g_arena_size;
extern const unsigned int g_feature_size;

#ifndef FEATURE_PATH
#define FEATURE_PATH "/mnt/sd0/data"
#endif

#ifndef FEATURE_WIDTH
#define FEATURE_WIDTH 30225
#endif

#endif  // _MODEL_H
