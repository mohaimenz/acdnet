#!/usr/bin/env bash

# Uploads latest nuttx.spk to Spresense device
(cd ../../ext/spresense/sdk && ./tools/flash.sh -c /dev/ttyUSB0 -b 500000 nuttx.spk)