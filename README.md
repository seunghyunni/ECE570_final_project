
# Image Generation using Continuous Filter Atoms

This repository is the official implementation for training and testing the 'Rotating Chair' experiment presented in "Image Generation using Continuous Filter Atoms".

Our codes adopted the official implementation of Pix2Pix and CycleGAN from [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 

The experiment is run on the RC-49 dataset. The dataset (h5 file) can be easily downloaded from [Continuous Conditional Generative Adversarial Networks](https://github.com/UBCDingXin/improved_CcGAN).


## Requirements

To install requirements:

go to 'scripts' folder and execute following lines.
```setup
bash install_deps.sh
bash conda_deps.sh
```

## Dataset

Download the RC-49 dataset (h5 file) and place it inside the 'dataset' folder.

Run this command to extract png files and split them into train and test datasets. 
```setup
bash init.sh
```

## Training

To train the model, run this command:

```train
bash train.sh
```

## Evaluation

To evaluate our model, run:

```eval
bash test.sh
```

## Pre-trained Models

The pre-trained models are included in the 'checkpoints/rc49' folder. 