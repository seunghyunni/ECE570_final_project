import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import pdb
import time
import numpy as np 
from matplotlib import pyplot as plt 
from PIL import Image
import torch 
from tqdm import tqdm 


if __name__ == '__main__':
    # pdb.set_trace()
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    data_loader = CreateDataLoader(opt) 
    dataset = data_loader.load_data()

    # model = create_model(opt, True)
    model = create_model(opt)
    model.setup(opt)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    ttt = []
    imtype = np.uint8

    # 0.0 ~ 45.0 (interval : 2.5/90)
    t_list = list (np.arange(2.5/90, 45/90, 2.5/90))

    fig, axes = plt.subplots(1, len(t_list)+1)

    save_dir = "./results/rc49/"

    start_index = 0

    # test
    for chair_type, data in enumerate(dataset):
        print("current iter: {}".format(chair_type))
        res = []
        if start_index + opt.num_test == chair_type:
            break

        # pdb.set_trace()
        model.set_input(data)
        
        if not os.path.exists(save_dir + str(chair_type)):
            os.mkdir(save_dir + str(chair_type))

        for idx, t in enumerate(t_list): # for every terminating point 't' 
            model.test(t)
            if idx == 0:
                realA = model.real_A[0].cpu().detach().numpy()
                realB = model.real_B[0].cpu().detach().numpy()  

                realB = (np.transpose(realB, (1, 2, 0)) + 1) / 2.0 * 255.0
                realB = realB.astype(imtype)
                
                realA = (np.transpose(realA, (1, 2, 0)) + 1) / 2.0 * 255.0
                realA = realA.astype(imtype)

                res.append(realA)

            fakeB = model.fake_B[0].cpu().detach().numpy()
            fakeB = (np.transpose(fakeB, (1, 2, 0)) + 1) / 2.0 * 255.0
            fakeB = fakeB.astype(imtype)
            fakeB_im  = Image.fromarray(fakeB)
            fakeB_im.save(save_dir + str(chair_type)+ "/fakeB_" + str(t) + ".png") # save fakeB with respect to terminating point 't'
            res.append(fakeB)

        for i,im in enumerate(res): 
            axes[i].imshow(res[i])
            axes[i].axis('off')
        
        plt.savefig(save_dir + str(chair_type) + "/result.png", pad_inches=0, dpi=300) # save overall plot
        plt.cla()
        realA_im  = Image.fromarray(realA)
        realA_im.save(save_dir + str(chair_type)+ "/realA.png") # save realA