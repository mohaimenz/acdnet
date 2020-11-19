#!/usr/bin/env bash
(cd ../../ext/spresense/sdk && source ~/spresenseenv/setup && ./tools/flash.sh -c /dev/ttyUSB0 -b 500000 nuttx.spk)

screen /dev/ttyUSB0 115200