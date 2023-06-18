# Sequence Region Specific Prediction of Protein Functions

In this study, the transformer model has been trained for regional protein function prediction problem. Also, a model that uses a classification head on top of the encoder of the [ProtT5-XL-UniRef50](https://github.com/agemagician/ProtTrans) model has been implemented for prediction of protein functions at protein level.

The Python script provided in this repository makes it possible to train both types of models and perform predictions using pretrained models.

## Getting Started
The recommended Python version is 3.8.10 since the script has been tested by using this version.

To create a clean virtual environment, use the following command:
```bash
$ python3 -m venv virtual_env
```

Then, activate the virtual environment by using the following command:
```bash
$ source virtual_env/bin/activate
```

After activating the virtual environment, in the directory where the requirements.txt file resides, execute the following command to install the requirements:

```bash
$ pip install -r requirements.txt
```

## Usage

The script has the following main usages:
 - Training the classification head model on non-regional datasets
 - Training the transformer model on regional datasets
 - Running a pretrained classification head model in inference mode
 - Running a pretrained transformer model in inference mode
 - Running a pretrained classification head model and a pretrained transformer model together in inference mode to produce both regional and non-regional predictions.

Note: For each of the main usages specified above, the [ProtT5-XL-UniRef50](https://github.com/agemagician/ProtTrans) encoder model is needed. You can download it using the following link: [https://drive.google.com/file/d/1PnPtgdzkopjdjNS6XMLhjyS9bioMT2Kq/view?usp=sharing](https://drive.google.com/file/d/1PnPtgdzkopjdjNS6XMLhjyS9bioMT2Kq/view?usp=sharing)

You can extract the folder of the encoder model from the archive using the following command:
```bash
$ tar xvf Rostlab_prot_t5_xl_uniref50.tar.xz
```

For each main usage specified above, a section exists below.

For seeing the usage of the script, the following command can be executed:
```bash
$ python3 main.py -h
```

The output of this command is as follows:
```bash
usage: main.py [-h] [-i] [-t] [-d DEVICE] -dp DATASET -mt {transformer,classification_head,merged} [-mp MODEL_PATH] -t5 PROT_T5_MODEL_PATH
               [-th THRESHOLD] [-bs BATCH_SIZE] [-e EPOCH] [-lr LEARNING_RATE] [-msd MODEL_SAVE_DIR] [-spe SAVE_PER_EPOCH] [-ml MAX_LENGTH]
               [-tdr TRAINING_DATASET_RATIO] [-tbld TENSORBOARD_LOG_DIR] [-chmp CLASSIFICATION_HEAD_MODEL_PATH] [-tmp TRANSFORMER_MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit

arguments:
  -i, --inference       If it is given, the model is set to run in inference mode
  -t, --train           If it is given, the model will be trained
  -d DEVICE, --device DEVICE
                        the device on which the model is to run
  -dp DATASET, --dataset DATASET
                        path to dataset (must be a pickle-saved binary file)
  -mt {transformer,classification_head,merged}, --model-type {transformer,classification_head,merged}
  -mp MODEL_PATH, --model-path MODEL_PATH
                        path to the directory where the model is saved
  -t5 PROT_T5_MODEL_PATH, --prot-t5-model-path PROT_T5_MODEL_PATH
                        path to the directory where ProtT5 model is stored
  -th THRESHOLD, --threshold THRESHOLD
                        threshold for classification head model. The less the threshold, the more the number of false positives, but the less the
                        number of false negatives. The more the threshold, the less the number of false positives, but the more the number of
                        false negatives.
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size
  -e EPOCH, --epoch EPOCH
                        number of epochs for training
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        learning rate for training
  -msd MODEL_SAVE_DIR, --model-save-dir MODEL_SAVE_DIR
                        the folder path to save the model(s) (must be non-existent)
  -spe SAVE_PER_EPOCH, --save-per-epoch SAVE_PER_EPOCH
                        if set to a positive value, each x epochs, the model is saved to the model save dir specified in the arguments
  -ml MAX_LENGTH, --max-length MAX_LENGTH
                        the maximum length for a protein sequence. If not set, automatically inferred from the dataset
  -tdr TRAINING_DATASET_RATIO, --training-dataset-ratio TRAINING_DATASET_RATIO
                        the ratio of the number of samples in training data to the number of samples in all dataset (must be between 0 and 1, 0
                        and 1 excluded)
  -tbld TENSORBOARD_LOG_DIR, --tensorboard-log-dir TENSORBOARD_LOG_DIR
                        the folder where the tensorboard logs are to be saved (must be non-existent)
  -chmp CLASSIFICATION_HEAD_MODEL_PATH, --classification-head-model-path CLASSIFICATION_HEAD_MODEL_PATH
                        path to the classification head model (only used in merged model type)
  -tmp TRANSFORMER_MODEL_PATH, --transformer-model-path TRANSFORMER_MODEL_PATH
                        path to the transformer model (only used in merged model type)
```

## Datasets
- For accessing datasets that have been used for training the classification head models, the following link can be used: [https://drive.google.com/drive/folders/1HKr7rp2E84jwvJn7Eje2t9zwVL4vFcF8?usp=sharing](https://drive.google.com/drive/folders/1HKr7rp2E84jwvJn7Eje2t9zwVL4vFcF8?usp=sharing)

- For accessing datasets that have been used for training the transformer models, the following link can be used: [https://drive.google.com/drive/folders/1M255UK-H72NCNrryh-UvdDGRXCXt_slJ?usp=sharing](https://drive.google.com/drive/folders/1M255UK-H72NCNrryh-UvdDGRXCXt_slJ?usp=sharing)




## Models
- For accessing pretrained classification head models, the following link can be used: [https://drive.google.com/drive/folders/1gRi7TBrwZfnfvXU21vtj8s5F4z5D9pXM?usp=sharing](https://drive.google.com/drive/folders/1gRi7TBrwZfnfvXU21vtj8s5F4z5D9pXM?usp=sharing)

- For accessing pretrained transformer models, the following link can be used: [https://drive.google.com/drive/folders/1ey8pZufHIV3bQAWjJ8yGnWABGkjFp4ta?usp=sharing](https://drive.google.com/drive/folders/1ey8pZufHIV3bQAWjJ8yGnWABGkjFp4ta?usp=sharing)

## Dataset Format
 - This script uses the datasets prepared in a specific format and saved as a binary object file using the pickle module.
 - Below, sample datasets consisting of toy examples are given for illustrating the required dataset formats. For viewing the dataset(s), you should download it and use the built-in pickle module in Python as follows (You can first open a Python shell and execute the following lines):
   ```python
   >>> import pickle
   >>> with open("test_nonregional_dataset.pkl", "rb") as f:
   >>>     dataset = pickle.load(f)
   >>> dataset
   ```
 - To examine the dataset format for training the classification head model, please use the following link: [test_nonregional_dataset.pkl](https://drive.google.com/file/d/1ycVX-Lx7rOmfoeZVwXAN72dMW-PqoblU/view?usp=sharing)
 - To examine the dataset format for training the transformer model, please use the following link: [test_regional_dataset.pkl](https://drive.google.com/file/d/1WyTfAkS_fXHclplaeOvxY7cW2WhP7_PQ/view?usp=sharing)
 - To examine the dataset format for running the model(s) in inference mode, please use the following link: [test_dataset.pkl](https://drive.google.com/file/d/1jeIiiYzPGLs-OL6_m1e5aVpThR6A2JPT/view?usp=sharing). Note that the formats of the dataset files named as test_nonregional_dataset.pkl and test_regional_dataset.pkl are also applicable for running the model(s) in inference mode. In the case where the GO term annotations are not known, the format of the file test_dataset.pkl must be used.

For getting acquainted with the usage of the script, it is recommended that you download the sample datasets specified above and use them to test the script by using the commands specified below.

## Using the Script to Train the Classification Head Model

A sample command for using the script to train the classification head model is as follows:
```bash
$ python3 main.py --train --device cuda:0 --dataset ../test_nonregional_dataset.pkl --model-type classification_head -t5 ../Rostlab_prot_t5_xl_uniref50 --model-save-dir ../classification_head_model_test --batch-size 8 --epoch 150 --learning-rate 0.005 --training-dataset-ratio 0.8
```

Executing this script by using the command above means that the **classification head** model will be **trained**
 - by using the device **cuda:0**
 - on the dataset found at relative path **../test_nonregional_dataset.pkl**
 - by using the T5Encoder model found at relative path **../Rostlab_prot_t5_xl_uniref50**
 - by saving the model at relative path **../classification_head_model_test** at the end of the training
 - by using the batch size as **8**
 - for **150** epochs
 - by using the learning rate as **0.005**
 - by using **80%** of the dataset as the training set, while the rest being the validation set.

## Using the Script to Train the Transformer Model

A sample command for using the script to train the transformer model is as follows:

```bash
$ python3 main.py --train --device cuda:0 --dataset ../test_regional_dataset.pkl -mt transformer -t5 ../Rostlab_prot_t5_xl_uniref50 --model-save-dir ../transformer_model_test --batch-size 8 --epoch 150 --learning-rate 0.05 --training-dataset-ratio 0.8 --tensorboard-log-dir ../transformer_model_test_tensorboard
```

Executing this script by using the command above means that the **transformer** model will be **trained**
 - by using the device **cuda:0**
 - on the dataset found at relative path **../test_regional_dataset.pkl**
 - by using the T5Encoder model found at relative path **../Rostlab_prot_t5_xl_uniref50**
 - by saving the model at relative path **../transformer_model_test** at the end of the training
 - by using the batch size as **8**
 - for **150** epochs
 - by using the learning rate as **0.05**
 - by using **80%** of the dataset as the training set, while the rest being the validation set
 - by saving tensorboard logs during training to the relative path **../transformer_model_test_tensorboard**

## Using the Script to Run the Classification Head Model in Inference Mode

A sample command for using the script to run the classification head model in inference mode is as follows:

```bash
$ python3 main.py --inference --model-type classification_head --model-path ../classification_head_model_test/end_of_training/ --threshold 0.20 --device cuda:0 -t5 ../Rostlab_prot_t5_xl_uniref50 --dataset ../test_dataset.pkl
```

Using the command above means that the **classification head** model found at relative path **../classification_head_model_test/end_of_training/** will be executed in **inference** mode
 - by using the device **cuda:0**
 - by using the threshold **0.20**
 - by using the T5Encoder model found at relative path **../Rostlab_prot_t5_xl_uniref50**
 - on the dataset found at relative path **../test_dataset.pkl** 

The sample outputs produced by the command above is as follows:
```
2023-06-17 07:04:57.145896: Loading ProtT5 tokenizer...
2023-06-17 07:04:57.716089: Done!
2023-06-17 07:04:57.716139: Loading ProtT5 encoder...
2023-06-17 07:05:09.797326: Done!
2023-06-17 07:05:09.797427: Loading classification model...
2023-06-17 07:05:17.368933: Done!
2023-06-17 07:05:18.234049: G: GO:0000001 GO:0000004 GO:0000005 GO:0000008 GO:0000009
2023-06-17 07:05:18.257649: QDRSMEN: GO:0000002 GO:0000004 GO:0000008
2023-06-17 07:05:18.280082: WLQT: GO:0000001 GO:0000003 GO:0000004 GO:0000009
2023-06-17 07:05:18.302514: W: GO:0000001 GO:0000004 GO:0000005 GO:0000008 GO:0000009 GO:0000010
2023-06-17 07:05:18.324971: KRD: GO:0000004 GO:0000009 GO:0000010
2023-06-17 07:05:18.347703: EVHGMEKGMI: GO:0000002 GO:0000003 GO:0000004 GO:0000007 GO:0000010
2023-06-17 07:05:18.371884: HKSTVWS: GO:0000001 GO:0000002 GO:0000004 GO:0000009 GO:0000010
2023-06-17 07:05:18.394303: DSTILAC: GO:0000003 GO:0000004 GO:0000009 GO:0000010
2023-06-17 07:05:18.419012: MEQGECPLMR: GO:0000002 GO:0000003 GO:0000004 GO:0000005 GO:0000006 GO:0000008 GO:0000010
2023-06-17 07:05:18.441450: AK: GO:0000002 GO:0000003 GO:0000004 GO:0000009 GO:0000010
2023-06-17 07:05:18.463799: FMAVREVLGH: GO:0000001 GO:0000004 GO:0000010
2023-06-17 07:05:18.486209: DALIDHWW: GO:0000003 GO:0000004 GO:0000007 GO:0000009 GO:0000010
2023-06-17 07:05:18.512298: QTQYN: GO:0000002 GO:0000004 GO:0000007 GO:0000008
2023-06-17 07:05:18.534839: G: GO:0000001 GO:0000004 GO:0000005 GO:0000008 GO:0000009
2023-06-17 07:05:18.557256: IQLHWCAAA: GO:0000003 GO:0000004 GO:0000006 GO:0000009 GO:0000010
2023-06-17 07:05:18.579784: T: GO:0000001 GO:0000004 GO:0000009
2023-06-17 07:05:18.603728: TGERVTPM: GO:0000003 GO:0000004 GO:0000007 GO:0000008 GO:0000009 GO:0000010
2023-06-17 07:05:18.625935: MITP: GO:0000001 GO:0000002 GO:0000003 GO:0000004 GO:0000005
2023-06-17 07:05:18.651647: LYDF: GO:0000003 GO:0000004 GO:0000010
2023-06-17 07:05:18.675184: LDGNCHFL: GO:0000002 GO:0000004 GO:0000008 GO:0000009 GO:0000010
2023-06-17 07:05:18.698682: P: GO:0000001 GO:0000004 GO:0000005 GO:0000008 GO:0000009
2023-06-17 07:05:18.721636: DAHF: GO:0000001 GO:0000002 GO:0000003 GO:0000004 GO:0000005 GO:0000007 GO:0000009
2023-06-17 07:05:18.745697: MCWDIS: GO:0000002 GO:0000003 GO:0000005 GO:0000006 GO:0000010
2023-06-17 07:05:18.770178: RGM: GO:0000003 GO:0000004 GO:0000005 GO:0000007 GO:0000008 GO:0000009 GO:0000010
2023-06-17 07:05:18.794179: NNTHNH: GO:0000004 GO:0000007 GO:0000008 GO:0000010
2023-06-17 07:05:18.816725: ML: GO:0000002 GO:0000003 GO:0000005 GO:0000006 GO:0000008 GO:0000010
2023-06-17 07:05:18.838899: YYCGCHIG: GO:0000001 GO:0000002 GO:0000010
2023-06-17 07:05:18.861319: KVLTI: GO:0000004 GO:0000005 GO:0000007 GO:0000010
2023-06-17 07:05:18.883900: R: GO:0000001 GO:0000004 GO:0000005 GO:0000009 GO:0000010
2023-06-17 07:05:18.908037: VNY: GO:0000002 GO:0000003 GO:0000004
2023-06-17 07:05:18.931543: LPN: GO:0000003 GO:0000004 GO:0000008
2023-06-17 07:05:18.954073: TKGPNDNA: GO:0000003 GO:0000004 GO:0000009
```

## Using the Script to Run the Transformer Model in Inference Mode

A sample command for using the script to run the transformer model in inference mode is as follows:

```bash
$ python3 main.py --inference --model-type transformer --model-path ../transformer_model_test/embed_size_1024.max_length_10000.lr_0.05_end_of_training/ --device cuda:0 -t5 ../Rostlab_prot_t5_xl_uniref50 --dataset ../test_dataset.pkl
```

Using the command above means that the **transformer** model found at relative path **../transformer_model_test/embed_size_1024.max_length_10000.lr_0.05_end_of_training/** will be executed in **inference** mode
 - by using the device **cuda:0**
 - by using the T5Encoder model found at relative path **../Rostlab_prot_t5_xl_uniref50**
 - on the dataset found at relative path **../test_dataset.pkl** 

The sample outputs produced by the command above is as follows:
```
2023-06-17 07:19:01.558596: Loading ProtT5 tokenizer...
2023-06-17 07:19:02.127968: Done!
2023-06-17 07:19:02.128011: Loading ProtT5 encoder...
2023-06-17 07:19:14.677408: Done!
2023-06-17 07:19:14.677520: Loading transformer model...
2023-06-17 07:19:24.017840: Done!
2023-06-17 07:19:24.866646: G: go:0000010: [[1, 1]] |
2023-06-17 07:19:25.115701: QDRSMEN: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 07:19:25.261224: WLQT: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] |
2023-06-17 07:19:25.319535: W: go:0000010: [[1, 1]] |
2023-06-17 07:19:25.442572: KRD: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] |
2023-06-17 07:19:25.770543: EVHGMEKGMI: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4], [9, 9]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] | go:0000002: [[10, 10]] |
2023-06-17 07:19:26.017410: HKSTVWS: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 07:19:26.260670: DSTILAC: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 07:19:26.611310: MEQGECPLMR: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4], [9, 9]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] | go:0000002: [[10, 10]] |
2023-06-17 07:19:26.708792: AK: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] |
2023-06-17 07:19:27.036395: FMAVREVLGH: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5], [10, 10]] | go:0000007: [[6, 6], [8, 8]] | go:0000003: [[7, 7], [9, 9]] |
2023-06-17 07:19:27.318858: DALIDHWW: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6], [8, 8]] | go:0000003: [[7, 7]] |
2023-06-17 07:19:27.491839: QTQYN: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] |
2023-06-17 07:19:27.546176: G: go:0000010: [[1, 1]] |
2023-06-17 07:19:27.854804: IQLHWCAAA: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6], [8, 8]] | go:0000003: [[7, 7], [9, 9]] |
2023-06-17 07:19:27.918508: T: go:0000010: [[1, 1]] |
2023-06-17 07:19:28.178690: TGERVTPM: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 07:19:28.328460: MITP: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] |
2023-06-17 07:19:28.492370: LYDF: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] |
2023-06-17 07:19:28.756567: LDGNCHFL: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 07:19:28.811352: P: go:0000010: [[1, 1]] |
2023-06-17 07:19:28.973147: DAHF: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] |
2023-06-17 07:19:29.186490: MCWDIS: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] |
2023-06-17 07:19:29.299019: RGM: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] |
2023-06-17 07:19:29.514377: NNTHNH: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] |
2023-06-17 07:19:29.612899: ML: go:0000010: [[1, 1]] | go:0000009: [[2, 2]] |
2023-06-17 07:19:29.876089: YYCGCHIG: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 07:19:30.055239: KVLTI: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] |
2023-06-17 07:19:30.120135: R: go:0000010: [[1, 1]] |
2023-06-17 07:19:30.238388: VNY: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] |
2023-06-17 07:19:30.349400: LPN: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] |
2023-06-17 07:19:30.612022: TKGPNDNA: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6], [8, 8]] | go:0000003: [[7, 7]] |
```

## Using the Script to Run the Merged Model in Inference Mode

A sample command for using the script to run the merged model in inference mode is as follows:

```bash
$ python3 main.py --inference --model-type merged --transformer-model-path ../transformer_model_test/embed_size_1024.max_length_10000.lr_0.05_end_of_training/ --classification-head-model-path ../classification_head_model_test/end_of_training/ --threshold 0.20 --device cuda:0 -t5 ../Rostlab_prot_t5_xl_uniref50 --dataset ../test_dataset.pkl
```

Using the command above means that the **merged** model consisting of the classification head model found at relative path **../classification_head_model_test/end_of_training/** and the transformer model found at relative path **../transformer_model_test/embed_size_1024.max_length_10000.lr_0.05_end_of_training/** will be executed in **inference** mode
 - by using the device **cuda:0**
 - by using the T5Encoder model found at relative path **../Rostlab_prot_t5_xl_uniref50**
 - on the dataset found at relative path **../test_dataset.pkl** 
 - by using the threshold **0.20** for the classification head model.

The sample outputs produced by the command above is as follows:

```
2023-06-17 04:44:08.914214: Loading ProtT5 tokenizer...
2023-06-17 04:44:09.456218: Done!
2023-06-17 04:44:09.456263: Loading ProtT5 encoder...
2023-06-17 04:44:21.329711: Done!
2023-06-17 04:44:21.334376: Loading classification model...
2023-06-17 04:44:28.720729: Done!
2023-06-17 04:44:29.453718: G: GO:0000001 GO:0000004 GO:0000005 GO:0000008 GO:0000009
2023-06-17 04:44:29.472584: QDRSMEN: GO:0000002 GO:0000004 GO:0000008
2023-06-17 04:44:29.490722: WLQT: GO:0000001 GO:0000003 GO:0000004 GO:0000009
2023-06-17 04:44:29.509122: W: GO:0000001 GO:0000004 GO:0000005 GO:0000008 GO:0000009 GO:0000010
2023-06-17 04:44:29.529871: KRD: GO:0000004 GO:0000009 GO:0000010
2023-06-17 04:44:29.550302: EVHGMEKGMI: GO:0000002 GO:0000003 GO:0000004 GO:0000007 GO:0000010
2023-06-17 04:44:29.570091: HKSTVWS: GO:0000001 GO:0000002 GO:0000004 GO:0000009 GO:0000010
2023-06-17 04:44:29.591253: DSTILAC: GO:0000003 GO:0000004 GO:0000009 GO:0000010
2023-06-17 04:44:29.614257: MEQGECPLMR: GO:0000002 GO:0000003 GO:0000004 GO:0000005 GO:0000006 GO:0000008 GO:0000010
2023-06-17 04:44:29.634674: AK: GO:0000002 GO:0000003 GO:0000004 GO:0000009 GO:0000010
2023-06-17 04:44:29.654129: FMAVREVLGH: GO:0000001 GO:0000004 GO:0000010
2023-06-17 04:44:29.675305: DALIDHWW: GO:0000003 GO:0000004 GO:0000007 GO:0000009 GO:0000010
2023-06-17 04:44:29.696014: QTQYN: GO:0000002 GO:0000004 GO:0000007 GO:0000008
2023-06-17 04:44:29.715333: G: GO:0000001 GO:0000004 GO:0000005 GO:0000008 GO:0000009
2023-06-17 04:44:29.732795: IQLHWCAAA: GO:0000003 GO:0000004 GO:0000006 GO:0000009 GO:0000010
2023-06-17 04:44:29.750773: T: GO:0000001 GO:0000004 GO:0000009
2023-06-17 04:44:29.773830: TGERVTPM: GO:0000003 GO:0000004 GO:0000007 GO:0000008 GO:0000009 GO:0000010
2023-06-17 04:44:29.791398: MITP: GO:0000001 GO:0000002 GO:0000003 GO:0000004 GO:0000005
2023-06-17 04:44:29.809347: LYDF: GO:0000003 GO:0000004 GO:0000010
2023-06-17 04:44:29.827405: LDGNCHFL: GO:0000002 GO:0000004 GO:0000008 GO:0000009 GO:0000010
2023-06-17 04:44:29.845105: P: GO:0000001 GO:0000004 GO:0000005 GO:0000008 GO:0000009
2023-06-17 04:44:29.863337: DAHF: GO:0000001 GO:0000002 GO:0000003 GO:0000004 GO:0000005 GO:0000007 GO:0000009
2023-06-17 04:44:29.882424: MCWDIS: GO:0000002 GO:0000003 GO:0000005 GO:0000006 GO:0000010
2023-06-17 04:44:29.900517: RGM: GO:0000003 GO:0000004 GO:0000005 GO:0000007 GO:0000008 GO:0000009 GO:0000010
2023-06-17 04:44:29.920050: NNTHNH: GO:0000004 GO:0000007 GO:0000008 GO:0000010
2023-06-17 04:44:29.938037: ML: GO:0000002 GO:0000003 GO:0000005 GO:0000006 GO:0000008 GO:0000010
2023-06-17 04:44:29.955522: YYCGCHIG: GO:0000001 GO:0000002 GO:0000010
2023-06-17 04:44:29.973980: KVLTI: GO:0000004 GO:0000005 GO:0000007 GO:0000010
2023-06-17 04:44:29.991912: R: GO:0000001 GO:0000004 GO:0000005 GO:0000009 GO:0000010
2023-06-17 04:44:30.009537: VNY: GO:0000002 GO:0000003 GO:0000004
2023-06-17 04:44:30.027109: LPN: GO:0000003 GO:0000004 GO:0000008
2023-06-17 04:44:30.044768: TKGPNDNA: GO:0000003 GO:0000004 GO:0000009
2023-06-17 04:44:30.438940: Loading transformer model...
2023-06-17 04:44:34.988885: Done!
2023-06-17 04:44:35.053637: G: go:0000010: [[1, 1]] |
2023-06-17 04:44:35.282410: QDRSMEN: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:35.423374: WLQT: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] |
2023-06-17 04:44:35.479875: W: go:0000010: [[1, 1]] |
2023-06-17 04:44:35.591468: KRD: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] |
2023-06-17 04:44:35.904460: EVHGMEKGMI: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4], [9, 9]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] | go:0000002: [[10, 10]] |
2023-06-17 04:44:36.133502: HKSTVWS: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:36.359336: DSTILAC: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:36.672005: MEQGECPLMR: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4], [9, 9]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] | go:0000002: [[10, 10]] |
2023-06-17 04:44:36.755867: AK: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] |
2023-06-17 04:44:37.070720: FMAVREVLGH: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5], [10, 10]] | go:0000007: [[6, 6], [8, 8]] | go:0000003: [[7, 7], [9, 9]] |
2023-06-17 04:44:37.326680: DALIDHWW: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6], [8, 8]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:37.494540: QTQYN: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] |
2023-06-17 04:44:37.548594: G: go:0000010: [[1, 1]] |
2023-06-17 04:44:37.832618: IQLHWCAAA: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6], [8, 8]] | go:0000003: [[7, 7], [9, 9]] |
2023-06-17 04:44:37.886289: T: go:0000010: [[1, 1]] |
2023-06-17 04:44:38.141513: TGERVTPM: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:38.280806: MITP: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] |
2023-06-17 04:44:38.420804: LYDF: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] |
2023-06-17 04:44:38.679136: LDGNCHFL: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:38.732383: P: go:0000010: [[1, 1]] |
2023-06-17 04:44:38.870777: DAHF: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] |
2023-06-17 04:44:39.067616: MCWDIS: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] |
2023-06-17 04:44:39.178638: RGM: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] |
2023-06-17 04:44:39.377095: NNTHNH: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] |
2023-06-17 04:44:39.460283: ML: go:0000010: [[1, 1]] | go:0000009: [[2, 2]] |
2023-06-17 04:44:39.714202: YYCGCHIG: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:39.882665: KVLTI: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] |
2023-06-17 04:44:39.937105: R: go:0000010: [[1, 1]] |
2023-06-17 04:44:40.047229: VNY: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] |
2023-06-17 04:44:40.157041: LPN: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] |
2023-06-17 04:44:40.410786: TKGPNDNA: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6], [8, 8]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:40.410914: G: GO:0000010 GO:0000009 GO:0000008 GO:0000001 GO:0000005 GO:0000004: go:0000010: [[1, 1]] |
2023-06-17 04:44:40.410983: QDRSMEN: GO:0000002 GO:0000009 GO:0000008 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:40.411062: WLQT: GO:0000009 GO:0000001 GO:0000006 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] |
2023-06-17 04:44:40.411103: W: GO:0000010 GO:0000009 GO:0000008 GO:0000001 GO:0000005 GO:0000004: go:0000010: [[1, 1]] |
2023-06-17 04:44:40.411150: KRD: GO:0000010 GO:0000009 GO:0000001 GO:0000006 GO:0000004: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] |
2023-06-17 04:44:40.411204: EVHGMEKGMI: GO:0000002 GO:0000010 GO:0000009 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4], [9, 9]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] | go:0000002: [[10, 10]] |
2023-06-17 04:44:40.411259: HKSTVWS: GO:0000002 GO:0000010 GO:0000009 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:40.411304: DSTILAC: GO:0000010 GO:0000009 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:40.411364: MEQGECPLMR: GO:0000002 GO:0000010 GO:0000009 GO:0000008 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4], [9, 9]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] | go:0000002: [[10, 10]] |
2023-06-17 04:44:40.411395: AK: GO:0000002 GO:0000010 GO:0000009 GO:0000006 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] |
2023-06-17 04:44:40.411443: FMAVREVLGH: GO:0000010 GO:0000009 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5], [10, 10]] | go:0000007: [[6, 6], [8, 8]] | go:0000003: [[7, 7], [9, 9]] |
2023-06-17 04:44:40.411499: DALIDHWW: GO:0000010 GO:0000009 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6], [8, 8]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:40.411542: QTQYN: GO:0000002 GO:0000009 GO:0000008 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] |
2023-06-17 04:44:40.411565: G: GO:0000010 GO:0000009 GO:0000008 GO:0000001 GO:0000005 GO:0000004: go:0000010: [[1, 1]] |
2023-06-17 04:44:40.411610: IQLHWCAAA: GO:0000010 GO:0000009 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6], [8, 8]] | go:0000003: [[7, 7], [9, 9]] |
2023-06-17 04:44:40.411632: T: GO:0000010 GO:0000001 GO:0000009 GO:0000004: go:0000010: [[1, 1]] |
2023-06-17 04:44:40.411677: TGERVTPM: GO:0000010 GO:0000009 GO:0000008 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:40.411711: MITP: GO:0000002 GO:0000009 GO:0000001 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] |
2023-06-17 04:44:40.411742: LYDF: GO:0000010 GO:0000009 GO:0000001 GO:0000006 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] |
2023-06-17 04:44:40.411786: LDGNCHFL: GO:0000002 GO:0000010 GO:0000009 GO:0000008 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:40.411808: P: GO:0000010 GO:0000009 GO:0000008 GO:0000001 GO:0000005 GO:0000004: go:0000010: [[1, 1]] |
2023-06-17 04:44:40.411842: DAHF: GO:0000002 GO:0000009 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] |
2023-06-17 04:44:40.411881: MCWDIS: GO:0000002 GO:0000010 GO:0000009 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] |
2023-06-17 04:44:40.411911: RGM: GO:0000010 GO:0000009 GO:0000008 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] |
2023-06-17 04:44:40.411950: NNTHNH: GO:0000010 GO:0000009 GO:0000008 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6]] |
2023-06-17 04:44:40.411976: ML: GO:0000010 GO:0000002 GO:0000009 GO:0000008 GO:0000006 GO:0000005 GO:0000003: go:0000010: [[1, 1]] | go:0000009: [[2, 2]] |
2023-06-17 04:44:40.412019: YYCGCHIG: GO:0000002 GO:0000010 GO:0000009 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5], [8, 8]] | go:0000007: [[6, 6]] | go:0000003: [[7, 7]] |
2023-06-17 04:44:40.412055: KVLTI: GO:0000010 GO:0000009 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] |
2023-06-17 04:44:40.412077: R: GO:0000010 GO:0000009 GO:0000001 GO:0000005 GO:0000004: go:0000010: [[1, 1]] |
2023-06-17 04:44:40.412103: VNY: GO:0000002 GO:0000001 GO:0000006 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] |
2023-06-17 04:44:40.412130: LPN: GO:0000008 GO:0000001 GO:0000006 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] |
2023-06-17 04:44:40.412173: TKGPNDNA: GO:0000009 GO:0000001 GO:0000007 GO:0000006 GO:0000005 GO:0000004 GO:0000003: go:0000006: [[1, 1]] | go:0000004: [[2, 2]] | go:0000001: [[3, 3]] | go:0000009: [[4, 4]] | go:0000005: [[5, 5]] | go:0000007: [[6, 6], [8, 8]] | go:0000003: [[7, 7]] |
```

In the output above, the script first loads and executes the classification head model. Then, the transformer model is loaded and executed. After that, the merged results are listed at the end.