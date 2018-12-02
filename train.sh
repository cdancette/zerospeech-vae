#!/bin/bash

date=`date '+%Y-%m-%d-%Hh%M.%S'`
mkdir -p experiments/$date/

cp train.sh experiments/$date

logs=experiments/$date/logs

python -u main.py --features-path data/mandarin-fbanks-training.h5f \
	--embedding-size 30 \
	--epochs 30 \
	--model-path experiments/$date/model/ \
	--learning-rate 1e-5 \
	| tee $logs
