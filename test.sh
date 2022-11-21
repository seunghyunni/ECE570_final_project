set -ex
python test.py --dataroot ./dataset/RC49/ --name rc49 --model pix2pix --direction AtoB --netG resnet_9blocks --gpu_ids 2 --num_test 4 --dataset_mode unpaired
