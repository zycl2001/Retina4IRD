#!/bin/bash

cd ../

export CUDA_VISIBLE_DEVICES=0

conda activate multimae

for model in "weight-cfp"
do
  python run_finetuning.py --finetune $model\
  --config cfgs/image_cls.yaml \
  --in_domains rgb
done

for model in "weight-oct"
do
  python run_finetuning.py --finetune $model\
  --config cfgs/image_cls.yaml \
  --in_domains oct
done

python top_acc.py
