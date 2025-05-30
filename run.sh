#!/bin/bash

# Loop from 1 to 40



source ./pyenv-conda-activate.sh


for i in {1..40}
do

   python3 ./main_slam.py --tracker_config $i
done
