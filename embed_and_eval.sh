#!/bin/bash

model=$1
models_dir=`dirname $model`
exp_dir=`dirname $models_dir`

zerospeech_vae="~/zerospeech-vae"

~/.conda/envs/zerospeech-vae/bin/python python embedder.py -f ~/zerospeech_vae/data/mandarin-fbanks-testing-10s.h5f --embedding-size 30 -p $model -o $exp_dir/features/mandarin/10s.h5f

~/.conda/envs/zerospeech/bin/python ./eval_track1.py --h5 -j 4 -n 1 mandarin 10 ~/zerospeech2017/data/  $exp_dir/features/mandarin/ $exp_dir/features/mandarin/10s-eval 
