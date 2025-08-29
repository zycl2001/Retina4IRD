# An AI-based Clinician-Decision Support System for Inherited Retinal Diseases: A Multicenter Clinical Validation Trial

## Contents

1. Prepare the environment
2. Prepare the datasets
3. Setting up and running the Image model
4. Setting up and running the Combine model

## Prepare the environment
### 1. Download the project

- Open the terminal in the system.

- Clone this repo file to the home path.

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
Please download RETFound weights (https://github.com/rmaphoh/RETFound_MAE) and place them in the /weights directory
/weights/RETFound_cfp_weights.pth
/weights/RETFound_oct_weights.pth
```
## Prepare the datasets

During fine-tuning, the images and labels of each modality are stored in CSV files: `train.csv`, `val.csv`, and `test.csv`, corresponding to the training, validation, and test sets, respectively. These files should be placed in the same directory. In each CSV file, `cfp` and `oct` indicate the image names for the corresponding modalities, `age` and other columns represent patient meta information, and `mutant_gene` denotes the associated hereditary eye disease. For paired-modality fine-tuning, each row contains paired data from two or more modalities.

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
| ID  | Modality name | age | ... | mutant_gene |
|:---:|:-----:|:---:|:---:|:-----------:|
|  1  | 1.jpg | 15  | ... |   gene_1    |
|  2  | 2.jpg | 18  | ... |   gene_2    |
| ... |  ...  | ... | ... |    ....     |

## Setting up and running the Image model
### 1. Configuration of Image Model
You can set image model training parameters in the `/cfgs/image_cls.yaml` file
```
nb_classes: number of classes
mutant_gene: 'mutant_gene'
csv_path: Your csv_path
data_path: Your image_path
.....
```

### 2. Start running the script
```
cd /script
sh target.sh
```
- Training scripts support YAML configuration files and command-line parameters.
- Parameters specified in the configuration file override the default settings, while command-line parameters override both the default and configured parameters.


## Setting up and running the Combine model
### 1. Configuration of Combine Model
You can set image model training parameters in the `/cfgs/combine_cls.yaml` file
```
csv_path: Your csv_path
output_dir: Your output path
....
```

### 2. Start running the script
```
cd /script
sh run_myEye_parms.sh
```
- Training scripts support YAML configuration files and command-line parameters.
- Parameters specified in the configuration file override the default settings, while command-line parameters override both the default and configured parameters.

### 3. Evaluation
We adopt Top-5 accuracy as the core evaluation metric, while conducting systematic testing and evaluation in conjunction with data from clinical randomized controlled trials. The details of the tasks are shown in our paper. 

