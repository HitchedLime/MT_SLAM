#!/bin/bash

# Loop from 1 to 40



source ./pyenv-conda-activate.sh


for i in {1..39}
do

   python3 ./main_vo.py --tracker_config $i --headless
done
