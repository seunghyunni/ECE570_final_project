
# Repository for ECE 570 Final Checkpoint Code Submission.

This repository is the implementation for training and testing the experiment presented in the final report of the class ECE 570.

Note that codes are adopted from the official implementation of Pix2Pix from [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 

Also, the codes adopt the official implementation of DCF-Network from [DCFNet: Deep Neural Network with Decomposed Convolutional Filters](https://github.com/ZeWang95/DCFNet-Pytorch). 
The experiment is run on the Edges2Handbag dataset. The dataset can be easily downloaded from (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md).


## Requirements

To install requirements:

go to 'scripts' folder and execute following lines.
```setup
bash install_deps.sh
bash conda_deps.sh
```

## Dataset

Download the Edges2Handbags dataset and place it inside the 'dataset' folder.

Run this command to extract png files and split them into train and test datasets. 
```setup
bash ./datasets/create_dataset.sh edges2handbags
```

## Training

To train the model, run this command:

```train
bash train.sh
```

## Testing

To test our model, run:

```eval
bash test.sh
```

## Code references and guidelines to new codes

Entire structure of codes are borrowed from the official implementation of Pix2Pix from [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). However, codes related to model implementation are all self-produced. 

Specifically, `networks.py`, `pix2pix_model.py`, are modified from the official implementation of Pix2Pix and `test.py`, `DCF.py` are newly added (student's original code).

In `networks.py` and `pix2pix_model.py`, the generator of Pix2Pix model is modified to be decomposed over filter atoms and coefficients. 
Specifically, all the convolutional layers in the Resblocks of Resnet style generator of the Pix2Pix model are decomposed. Consequently, the original Conv2D filters in the Pix2Pix generator are all replaced into Decomposed-Conv2D filters. 

Code for decomposed convolutional filters can be found in `DCF.py`, under the name of `Conv_DCFre` class.
`Conv_DCFre` class has been modified from the official implementation of DCF-Network from [DCFNet: Deep Neural Network with Decomposed Convolutional Filters](https://github.com/ZeWang95/DCFNet-Pytorch). 
`rand_base_generator_dcfnet` class in the `DCF.py` script has been newly added in order to generate convolutional filters in a stochastic way. 

`test.py` provides a python script to reproduce the results provided in the project report.  
