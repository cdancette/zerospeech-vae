#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
	echo "Usage: script.sh model_path"
	exit 1
fi

model=$1
models_dir=`dirname $model`
exp_dir=`dirname $models_dir`

zerospeech_vae="~/zerospeech-vae"

#~/.conda/envs/zerospeech-vae/bin/python embedder.py -f ~/zerospeech-vae/data/mandarin-fbanks-testing-10s.h5f -p $model -o $exp_dir/features/mandarin/10s.h5f ${@:2}

~/.conda/envs/zerospeech-vae/bin/python embedder.py -f ~/zerospeech-vae/data/mandarin-fbanks-testing-10s-stacked.h5f -p $model -o $exp_dir/features/mandarin/10s.h5f ${@:2}


~/.conda/envs/zerospeech/bin/python ~/zerospeech2017/track1/eval/eval_track1.py --h5 -j 4 -n 1 mandarin 10 ~/zerospeech2017/data/  $exp_dir/features/mandarin/ $exp_dir/features/mandarin/10s-eval 

cat $exp_dir/features/mandarin/10s-eval/results.txt
