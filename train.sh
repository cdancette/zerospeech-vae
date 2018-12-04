#!/bin/bash

date=`date '+%Y-%m-%d-%Hh%M.%S'`
mkdir -p experiments/$date/

echo "Folder : experiments/$date"

cp train.sh experiments/$date

logs=experiments/$date/logs

python -u main.py --features-path data/mandarin-fbanks-training-stacked \
	--embedding-size 100 \
	--epochs 40 \
	--model-path experiments/$date/model/ \
	--learning-rate 1e-6 \
	--input-size 280 \
	--loss bce \
	| tee $logs

echo "Folder : experiments/$date"
