set -ex
python test.py --dataroot ./dataset/edges2handbags/ --name edges2handbags --model pix2pix --direction AtoB --netG resnet_9blocks --gpu_ids 0 --num_test 4 --dataset_mode aligned
