#!/bin/bash

cd ../

export CUDA_VISIBLE_DEVICES=0

conda activate multimae

python myEyeTenClassfication.py --config cfgs/combine_cls.yaml

python top_acc.py