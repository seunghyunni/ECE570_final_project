import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class UnpairedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        if self.opt.isTrain:
            self.AB_paths = open(opt.dataroot + "train.txt", 'r').readlines() # test
        else:
            self.AB_paths = open(opt.dataroot + "test.txt", 'r').readlines() # test
        # random.seed(1234)
        # random.shuffle(self.AB_paths) # random shuffle
        self.transform = get_transform(opt)

        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index].split("\n")[0]
        
        image_folder = AB_path.split("_")[0]
        A_head = AB_path.split("_")[1]
        B_head = AB_path.split("_")[2] # train

        if self.opt.isTrain:
            A_path = os.path.join(self.opt.dataroot, "train", image_folder, A_head + ".png")
            B_path = os.path.join(self.opt.dataroot, "train", image_folder, B_head + ".png")

        else:
            A_path = os.path.join(self.opt.dataroot, "test", image_folder, A_head + ".png")
            B_path = os.path.join(self.opt.dataroot, "test", image_folder, B_head + ".png")

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)
        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B, 'R1': (float(A_head) * (1./90.)), 'R2': (float(B_head) * (1./90.)),
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'UnpairedDataset'
