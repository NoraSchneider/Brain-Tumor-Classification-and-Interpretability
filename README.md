# Interpretability Project

This repository contains different deep learning models for detecting tumors in brain MRI scans. 
Further different explainability methods are implemented and discussed for each model 
For a detailed discussion of the models and explainability methods we refer to the written report. 
We based our solution on the provided [skeleton](https://github.com/alain-ryser/interpretability-project).

## Setup
To use this template you can generate a Conda environment using `environment.yml` by running
```sh
conda env create -f project3_environment.yml  --name <custom_name>
```
## Dataset
This dataset contains a mix of samples from the Kaggle datasets [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) and [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) datasets.
Further for Transfer learning we use the following larger Kaggel dataset [Br35H :: Brain Tumor Detection 2020](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection?select=no&sort=votes).
Add these (our your customized data) to the `/data/images` and `/data/tl_dataset` folder. 

As mentioned in the report, we further filter the provided dataset for duplicates. To filter the dataset, run the script `make_unique.py`.

## Project3Lib
This package includes our models, functions for loading and preprocessing data and for training and evaluating our models. They are called from the respective notebooks.
To use the library in your code run, e.g.
```python
import project3Lib
model = project3Lib.CNN.BaslineClf()
```

## Training and Tesing Models
The code that we use for evaluating different models is in the respective notebooks. Further the code for the final
models are also included. All notebooks follow roughly the follwing structure:
1. Loading (and optionally preprocessing) data 
2. Train a classifier
3. Test classifier
4. Explain classifier using SHAP, Integrated Gradient and Grad Cam. 

The following gives a brief overview over the corresponding notebooks and their contents.
To reproduce our results from the report, follow the exact order of the tasks and their corresponding scripts
as they are described in the following.

|Notebook | Description |
| -------------- | --------- |
| Random_Forest_Baseline.ipynb | Task 1: Train a random forest classifier for the pyradiomics data and interpret it.| 
| Baseline.ipynb | Task 2: Train the provided baseline model on the image data and interpret it.| 
| CNN.ipynb | Task 3: Train an adjusted CNN on the image data and interpret it.| 
| CNN_TransferLearning.ipynb | Task 3: Train the provided baseline model on the image data and interpret it.| 
| VarAutoEncoder_Transfer_Learning.ipynb | Task 3: Train a VAE and classifier on the transfer learning und original data and then interpret it.| 
| TrainUnetFinal.ipynb | Task 3: Train a UNet and classifier on top and then interpret it.| 

## Authors
Mert Ertugrul, Johan Lokna, Nora Schneider
