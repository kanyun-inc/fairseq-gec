#!/usr/bin/env bash
set -e
set -x

# generate data with noise
python noise_data.py -e 1 -s 9182
python noise_data.py -e 2 -s 78834
python noise_data.py -e 3 -s 5101
python noise_data.py -e 4 -s 33302
python noise_data.py -e 5 -s 781
python noise_data.py -e 6 -s 1092
python noise_data.py -e 7 -s 10688
python noise_data.py -e 8 -s 50245
python noise_data.py -e 9 -s 71187
