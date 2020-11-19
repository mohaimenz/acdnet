#!/usr/bin/env bash
(cd ../../ext/spresense && rm -rf examples/tf_helloworld && rm -rf sdk/configs/examples/tf_helloworld)
(cd ../../ext/spresense/sdk && source ~/spresenseenv/setup && ./tools/config.py examples/tflite_micro && make)