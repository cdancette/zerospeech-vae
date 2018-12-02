#!/bin/bash

date=`date '+%Y-%m-%d-%Hh%M.%S'`

python -u main.py --features-path data/mandarin-fbanks-training.h5f \
	--embedding-size 30 \
	--epochs 30 \
	--model-path model-$date | tee log-$date.txt
