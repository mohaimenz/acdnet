AUTOMAKE_OPTIONS = subdir-objects
bin_PROGRAMS = pipeline

pipeline_SOURCES =  app/main.cpp \
                    app/model.cpp \
                    app/nn_manager.cpp \
                    app/nn_score.cpp \
                    app/debug_log.cpp

pipeline_CPPFLAGS =   -std=c++11 \
                    -DTF_LITE_STATIC_MEMORY \
                    -DDEBUG \
                    -DFEATURE_PATH=\"$(FEATURE_PATH)\" \
                    -DFEATURE_WIDTH=$(FEATURE_WIDTH) \
                    -g \                    
                    -I../../ext/tensorflow \
                    -I../../ext/flatbuffers/include

pipeline_LDADD =    ../../ext/tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a \
                    ../../ext/tensorflow/tensorflow/lite/micro/tools/make/gen/linux_x86_64/lib/libtensorflow-microlite.a

copy-model:
	@cp ${MODEL_PATH} app/model.cpp

clean-local:
	@rm config.status configure config.log
	@rm Makefile
	@rm -r autom4te.cache/
	@rm aclocal.m4
	@rm compile install-sh missing Makefile.in depcomp
