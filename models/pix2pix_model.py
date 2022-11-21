import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pdb
from torch import nn
NUM_SAM = 1

class gray_L1(nn.Module):
    def __init__(self):
        super(gray_L1, self).__init__()
        self.cc = torch.nn.L1Loss()

    def forward(self, pred, target):
        pred = pred.mean(1, keepdim=True)
        target = target.mean(1, keepdim=True)
        return self.cc(pred, target)

class BatchNormalizeTensor(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        to_return = (tensor-self.mean)/self.std
        return to_return


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256')
        parser.set_defaults(dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--gray', type=int, default=0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'diverse_loss']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.n_downsampling, True)
        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if (self.opt.direction == 'AtoB' or self.opt.direction == 'BtoA') and not self.opt.noref:
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            print(self.netD)
            self.gray = opt.gray
        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            if self.gray == 0:
                self.criterionL1 = torch.nn.L1Loss()
            else:
                self.criterionL1 = gray_L1()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        # AtoB = self.opt.direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if "RC49" in self.opt.dataroot: # if rotation_chair exp
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            self.image_paths = input['A_paths']
            self.R1 = input['R1']
            self.R2 = input['R2']
        else:
            if self.opt.direction == 'AtoB':
                self.real_A = input['A'].to(self.device)
                self.real_B = input['B'].to(self.device)
                self.image_paths = input['A_paths']
            elif self.opt.direction == 'BtoA':
                self.real_A = input['B'].to(self.device)
                self.real_B = input['A'].to(self.device)
                self.image_paths = input['B_paths']
            elif self.opt.direction == 'AtoA':
                self.real_A = input['A'].to(self.device)
                self.real_B = input['A'].to(self.device)
                self.image_paths = input['A_paths']
            elif self.opt.direction == 'BtoB':
                self.real_A = input['B'].to(self.device)
                self.real_B = input['B'].to(self.device)
                self.image_paths = input['B_paths']

    def forward(self, R2=1.):
        if "RC49" in self.opt.dataroot:  
            if self.isTrain:
                self.fake_B = self.netG(self.real_A, self.R1, self.R2, True) # train time code
            else:
                self.fake_B = self.netG(self.real_A, self.R1, self.R1 + R2, True) # test time code 
        else:
            self.fake_B = self.netG(self.real_A, 0., R2, False)
            if self.opt.residual:
                self.fake_B = self.fake_B + self.real_A

    def compute_mean_loss(self):
        loss = torch.abs(self.fake_B - self.fake_mean).mean()
        return 2 - loss

    def preprocess_input_resnet(self, x):
        return BatchNormalizeTensor(torch.FloatTensor([0.485, 0.456, 0.406]).cuda().view([1,3,1,1]), 
                torch.FloatTensor([0.229, 0.224, 0.225]).cuda().view([1,3,1,1]))((x).expand([-1,3,-1,-1]))

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # if self.opt.direction == 'AtoB' or self.opt.direction == 'BtoA':            
        if (self.opt.direction == 'AtoB' or self.opt.direction == 'BtoA') and not self.opt.noref:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
            pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)

            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)

            # Combined loss
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

            self.loss_D = self.loss_D

            self.loss_D.backward()
            
        else:
            fake_AB = self.fake_AB_pool.query(self.fake_B)
            pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)

            # Real
            real_AB = self.real_B
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)

            # Combined loss
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

            self.loss_D = self.loss_D

            self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        if (self.opt.direction == 'AtoB' or self.opt.direction == 'BtoA') and not self.opt.noref:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        else:
            fake_AB = self.fake_B
        
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        # self.loss_diverse_loss = self.compute_mean_loss()
        self.loss_diverse_loss = torch.tensor([0]).cuda()

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + 0.01 * self.loss_G_L1# + 10*self.loss_diverse_loss

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
