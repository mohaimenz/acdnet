#ifndef _COMMON_H
#define _COMMON_H

#define _20khz_SP41_taylor_l0_full_80_85_25_h5_int8

#if defined(SP4_taylor_l0_full_80_84_int8)
    #include "models/SP4_taylor_l0_full_80_84_int8/model.h"
    #define FEATURE_WIDTH 66650
    #define FEATURE_PATH "data/mm/44100/int8"
    #define MODEL_NAME "SP4_taylor_l0_full_80_84_int8"
#elif defined(SP4_taylor_l0_full_80_84_int16)
    #include "models/SP4_taylor_l0_full_80_84_int16/model.h"
    #define FEATURE_WIDTH 66650
    #define FEATURE_PATH "data/mm/44100/int8" 
    #define MODEL_NAME "SP4_taylor_l0_full_80_84_int16"
#elif defined(SP4_taylor_l0_full_80_84_float16)
    #include "models/SP4_taylor_l0_full_80_84_float16/model.h"
    #define FEATURE_WIDTH 66650
    #define FEATURE_PATH "data/mm/44100/int8" 
    #define MODEL_NAME "SP4_taylor_l0_full_80_84_float16"
#elif defined(SP4_taylor_l0_full_80_84_float32)
    #include "models/SP4_taylor_l0_full_80_84_float32/model.h"
    #define FEATURE_WIDTH 66650
    #define FEATURE_PATH "data/mm/44100/int8" 
    #define MODEL_NAME "SP4_taylor_l0_full_80_84_float32"
#elif defined(SP4_20KHZ_taylor_l0_full_80_81_int8)
    #include "models/SP4_20KHZ_taylor_l0_full_80_81_int8/model.h"
    #define FEATURE_WIDTH 33025
    #define FEATURE_PATH "data/mm/20000/int8" 
    #define MODEL_NAME "SP4_20KHZ_taylor_l0_full_80_81_int8"
#elif defined(SP4_20KHZ_taylor_l0_full_80_81_int16)
    #include "models/SP4_20KHZ_taylor_l0_full_80_81_int16/model.h"
    #define FEATURE_WIDTH 33025
    #define FEATURE_PATH "data/mm/20000/int8" 
    #define MODEL_NAME "SP4_20KHZ_taylor_l0_full_80_81_int16"
#elif defined(SP4_20KHZ_taylor_l0_full_80_81_float16)
    #include "models/SP4_20KHZ_taylor_l0_full_80_81_float16/model.h"
    #define FEATURE_WIDTH 33025
    #define FEATURE_PATH "data/mm/20000/int8" 
    #define MODEL_NAME "SP4_20KHZ_taylor_l0_full_80_81_float16"
#elif defined(SP4_20KHZ_taylor_l0_full_80_81_float32)
    #include "models/SP4_20KHZ_taylor_l0_full_80_81_float32/model.h"
    #define FEATURE_WIDTH 66650
    #define FEATURE_PATH "data/mm/20000/int8" 
    #define MODEL_NAME "SP4_20KHZ_taylor_l0_full_80_81_float32"
#elif defined(SP4_20KHZ_taylor_l0_full_85_80_int8)
    #include "models/SP4_20KHZ_taylor_l0_full_85_80_int8/model.h"
    #define FEATURE_WIDTH 33025
    #define FEATURE_PATH "data/mm/20000/int8" 
    #define MODEL_NAME "SP4_20KHZ_taylor_l0_full_85_80_int8"
#elif defined(SP4_20KHZ_taylor_l0_full_85_80_int16)
    #include "models/SP4_20KHZ_taylor_l0_full_85_80_int16/model.h"
    #define FEATURE_WIDTH 33025
    #define FEATURE_PATH "data/mm/20000/int8" 
    #define MODEL_NAME "SP4_20KHZ_taylor_l0_full_85_80_int16"
#elif defined(SP4_20KHZ_taylor_l0_full_85_80_float16)
    #include "models/SP4_20KHZ_taylor_l0_full_85_80_float16/model.h"
    #define FEATURE_WIDTH 33025
    #define FEATURE_PATH "data/mm/20000/int8" 
    #define MODEL_NAME "SP4_20KHZ_taylor_l0_full_85_80_float16"
#elif defined(SP4_20KHZ_taylor_l0_full_85_80_int8)
    #include "models/SP4_20KHZ_taylor_l0_full_85_80_float32/model.h"
    #define FEATURE_WIDTH 33025
    #define FEATURE_PATH "data/mm/20000/int8"
    #define MODEL_NAME "SP4_20KHZ_taylor_l0_full_85_80_int8"
#elif defined(_44khz_SP4_taylor_l0_full_80_84_5_924)
    #include "models/44khz_SP4_taylor_l0_full_80_845_int8/model.h"
    #define FEATURE_WIDTH 66650
    #define FEATURE_PATH "data/mm/44100/int8"
    #define MODEL_NAME "44khz_SP4_taylor_l0_full_80_84_5_924"
#elif defined(_44khz_SP4_taylor_l0_full_85_80_int8)
    #include "models/44khz_SP4_taylor_l0_full_85_80_int8/model.h"
    #define FEATURE_WIDTH 66650
    #define FEATURE_PATH "data/mm/44100/int8"
    #define MODEL_NAME "44khz_SP4_taylor_l0_full_85_80"
#elif defined(SP4_taylor_l0_full_80_845_int8)
    #include "models/SP4_taylor_l0_full_80_845_int8/model.h"
    #define FEATURE_WIDTH 66650
    #define FEATURE_PATH "data/mm/44100/int8"
    #define MODEL_NAME "SP4_taylor_l0_full_80_845_int8"
#elif defined(_20khz_SP41_taylor_l0_full_85_83_int8)
    #include "models/20khz_SP41_taylor_l0_full_85_83_int8/model.h"
    #define FEATURE_WIDTH 33025
    #define FEATURE_PATH "data/ma/20000/int8"
    #define MODEL_NAME "20khz_SP41_taylor_l0_full_85_83_int8"
#elif defined(_20khz_SP41_taylor_l0_full_80_85_25_h5_int8)
    #include "models/20khz_SP41_taylor_l0_full_80_8525_int8/model.h"
    #define FEATURE_WIDTH 33025
    #define FEATURE_PATH "data/ma/20000/int8"
    #define MODEL_NAME "20khz_SP41_taylor_l0_full_80_8525_int8"
#endif

#endif