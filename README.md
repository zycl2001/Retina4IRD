# An AI-based Clinician-Decision Support System for Inherited Retinal Diseases: A Multicenter Clinical Validation Trial

## Contents

# Table of Contents

1. [Download the pretrained weights of Retina4IRD](#download-the-pretrained-weights-of-retina4ird)
2. [Prepare the environment](#prepare-the-environment)
3. [Fine-tuning with Retina4IRD Image Model Weights](#fine-tuning-with-retina4ird-image-model-weights)
4. [Fine-tuning with Retina4IRD Combined Model Weights](#fine-tuning-with-retina4ird-combined-model-weights)
5. [Prepare the datasets](#prepare-the-datasets)

## 📃Download  the pretrained weights of Retina4IRD
Obtain the Retina4IRD pre-training weights on (https://drive.google.com/drive/folders/1WxbkPx1npybuDoNuJZdyzIkbgInWReBJ?usp=drive_link)

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Source</th>
<!-- TABLE BODY -->
<tr><td align="left">Retina4IRD_imageModel_CFP</td>
<td align="center"><a href="https://drive.google.com/file/d/1nqUoso8IfCq1gKfLgHrfWiY8XehTza-5/view?usp=drive_link">access</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">Retina4IRD_imageModel_OCT</td>
<td align="center"><a href="https://drive.google.com/file/d/1jE-AGclX4dHLMGCFKisqQqCmufAlBLTL/view?usp=drive_link">access</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">Retina4IRD_combineModel_CFP</td>
<td align="center"><a href="https://drive.google.com/file/d/1jE-AGclX4dHLMGCFKisqQqCmufAlBLTL/view?usp=drive_link">access</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">Retina4IRD_combineModel_OCT</td>
<td align="center"><a href="https://drive.google.com/file/d/1FisgU_tzBqve6qlZfcvJXW1ayS8K0ku1/view?usp=drive_link">access</a></td>
</tr>
</tbody></table>


## 🔧Prepare the environment

### 1. Download the project

- Open the terminal in the system.

- Clone this repository to the home directory.

```
git clone https://github.com/zycl2001/Retina4IRD.git
```

- Change the current directory to the project directory

```
cd /Retina4IRD
```

### 2. Prepare the running environment

1. Create environment with conda:

```
conda create -n retina4IRD python=3.8 -y
conda activate retina4IRD
```

2. Install dependencies
- Install Pytorch 1.10.0 (Cuda 11.1)

```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- Install other Python packages

```
pip install -r requirements.txt
```

3. Download Weight

```
Please download Retina4IRD weights (https://drive.google.com/drive/folders/1WxbkPx1npybuDoNuJZdyzIkbgInWReBJ?usp=drive_link) and place them in the /weights directory
/weights/image_models/Retina4IRD_CFP_checkpoint.pth
/weights/image_models/Retina4IRD_OCT_checkpoint.pth
/weights/combine_models/Retina4IRD_combineModel_CFP.pkl
/weights/combine_models/Retina4IRD_combineModel_OCT.pkl
```

## 🌱Fine-tuning with Retina4IRD Image Model Weights

### 1. Configuration of Image Model
You can set image model training parameters in the `/cfgs/image_cls.yaml` file
```
nb_classes: number of classes
gene: 'gene'
csv_path: Your csv_path
data_path: Your image_path
.....
```
### 2. Run script file
```
cd /script
sh target.sh
```
- Training scripts support YAML configuration files and command-line parameters.
- Parameters specified in the configuration file override the default settings, while command-line parameters override both the default and configured parameters.

### 3. Inference
You can run the script below to test the image.
```
export CUDA_VISIBLE_DEVICES=0

conda activate retina4IRD

python run_finetuning.py \
--nb_classes 17 \
--gene gene \
--test \
--combine_eyes False \
--label_column gene_label \
--finetune in21k \
--weight_cfp /weights/image_models/Retina4IRD_CFP_checkpoint.pth \
--weight_oct /weights/image_models/Retina4IRD_OCT_checkpoint.pth \
--data_path YOUR_IMAGE_PATH \
--csv_path YOUR_CSV_PATH \
--output_dir YOUR_OUTPUT_DIR \
--epochs 50 \
--batch_size 32 \
--blr 0.005 \
--input_size 224 \
--use_mean_pooling True \
--in_domains rgb
```

## 🌱Fine-tuning with Retina4IRD Combined Model Weights
### 1. Configuration of Combine Model
You can set image model training parameters in the `/cfgs/combine_cls.yaml` file
```
csv_path: Your csv_path
output_dir: Your output path
....
```

### 2. Run script file
```
cd /script
sh run_myEye_parms.sh
```
- Training scripts support YAML configuration files and command-line parameters.
- Parameters specified in the configuration file override the default settings, while command-line parameters override both the default and configured parameters.

### 3. Inference
After running ImageModel, you can run the following script to test the combined model.
```
export CUDA_VISIBLE_DEVICES=0
conda activate retina4IRD

python myEyeTenClassfication.py \
--count 6 \
--weight 0.974 \
--test \
--seed_count 5 \
--csv_path YOUR_CSV_PATH \
--output_dir YOUR_OUTPUT_DIR \
--combine_weight_cfp /weights/combine_models/Retina4IRD_combineModel_CFP.pkl \
--combine_weight_oct /weights/combine_models/Retina4IRD_combineModel_OCT.pkl \
--label_column gene_label \
--in_domains rgb
```

## 📝Prepare the datasets

During fine-tuning, the images and labels of each modality are stored in CSV files: `train.csv`, `val.csv`, and `test.csv`, corresponding to the training, validation, and test sets, respectively. These files should be placed in the same directory. In each CSV file, `cfp` and `oct` indicate the image names for the corresponding modalities, `age` and other columns represent patient meta information, and `gene` denotes the associated hereditary eye disease. For paired-modality fine-tuning, each row contains paired data from two or more modalities.

```
├── csv_path
    ├──cfp
        ├──train.csv
        ├──val.csv
        ├──test.csv
    ├──oct
        ├──train.csv
        ├──val.csv
        ├──test.csv
```
```
├──image_path
    ├──cfp
    	├──1.png
    	├──2.png
    	...
    ├──oct
    	├──1.png
    	├──2.png
    	...
```
| ID  | cfp/oct | age | ... | gene |
|:---:|:-------:|:---:|:---:|:-----------:|
|  1  |  1.jpg  | 15  | ... |   gene_1    |
|  2  |  2.jpg  | 18  | ... |   gene_2    |
| ... |   ...   | ... | ... |    ....     |



