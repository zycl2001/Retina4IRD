#!/bin/bash

cd ../

export CUDA_VISIBLE_DEVICES=0

conda activate multimae

python myEyeTenClassfication.py \
--config cfgs/combine_cls.yaml \
--in_domains rgb

python myEyeTenClassfication.py \
--config cfgs/combine_cls.yaml \
--in_domains oct

 python top_acc.py