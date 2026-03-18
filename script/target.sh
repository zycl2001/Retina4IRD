#!/bin/bash

cd ../

export CUDA_VISIBLE_DEVICES=0

conda activate multimae


python run_finetuning.py \
--config cfgs/image_cls.yaml \
--in_domains rgb

python run_finetuning.py \
--config cfgs/image_cls.yaml \
--in_domains oct

python top_acc.py
