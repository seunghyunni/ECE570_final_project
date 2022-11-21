set -ex
python train.py --dataroot ./dataset/RC49/ --dataset_mode unpaired --name rc49_train --model pix2pix --direction AtoB --netG resnet_9blocks --gray 1 --lambda_L1 0.0 --gpu_ids 2