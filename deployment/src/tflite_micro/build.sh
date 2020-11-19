#!/usr/bin/env bash
(cd ../../ext/spresense/sdk && source ~/spresenseenv/setup && ./tools/config.py examples/tflite_micro && make)