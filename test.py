#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint

from torch.utils.tensorboard import SummaryWriter
from visualization import board_add_image, board_add_images, save_images

# mesllo
import wandb
from torchinfo import summary


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--warp_dir', type=str, default='', help='warp output')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default = 1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    # mesllo
    parser.add_argument('--rel', type=str, default='1.0', help='specify which relevance level we used')
    parser.add_argument('--fin', type=str, default='SHALLOW', help='specify which finetuning config we used')
    parser.add_argument('--plf', type=str, default='1TO5', help='specify which plf layers we used')
    parser.add_argument('--pln', type=str, default='vgg19', help='specify which pln we used')
    parser.add_argument('--dataset', type=str, default='viton_quick', help='specify which dataset we used for vton')

    opt = parser.parse_args()
    return opt

def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    ### WANDB STUFF FOR EXP TRACKING
    wandb.init(
        project="CP-VTON - Finetuned PLN",
        config = {
            "stage": opt.stage,
            "mode": opt.datamode,
            "dataset": opt.dataset
        }
    )

    wandb.run.name = opt.name
    ###

    base_name = opt.name + '_' + os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        c_names = inputs['c_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
            
        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        visuals = [ [im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        
        save_images(warped_cloth, c_names, warp_cloth_dir) 
        save_images(warped_mask*2-1, c_names, warp_mask_dir) 

        if (step+1) % opt.display_count == 0:
            #board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)

            visuals_new = []
            for i, images in enumerate(visuals):
                row = []
                for j, image in enumerate(images):
                    img = wandb.Image(image)
                    row.append(img)
                visuals_new.append(row)

            # log training loss on wandb for exp tracking
            wandb.log({
                "step": step+1,
                "epoch": ((step+1) // len(test_loader.dataset)) + 1,
                "warping1": visuals_new[0],
                "warping2": visuals_new[1],
                "warping3": visuals_new[2]
            })
        


def test_tom(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    ### WANDB STUFF FOR EXP TRACKING
    wandb.init(
        project="CP-VTON - Finetuned PLN",
        config = {
            "stage": opt.stage,
            "mode": opt.datamode,
            "relevance_level": opt.rel,
            "finetune_config": opt.fin,
            "plf_layers": opt.plf,
            "pln": opt.pln,
            "dataset": opt.dataset
        }
    )

    wandb.run.name = opt.name
    ###
    
    base_name = opt.name + '_' + os.path.basename(opt.checkpoint)
    #save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    save_dir = os.path.join(opt.result_dir, base_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        
        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, 2*cm-1, m_composite], 
                   [p_rendered, p_tryon, im]]
            
        save_images(p_tryon, im_names, try_on_dir) 
        if (step+1) % opt.display_count == 0:
            #board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)

            visuals_new = []
            for i, images in enumerate(visuals):
                row = []
                for j, image in enumerate(images):
                    img = wandb.Image(image)
                    row.append(img)
                visuals_new.append(row)

            # log training loss on wandb for exp tracking
            wandb.log({
                "step": step+1,
                "epoch": ((step+1) // len(test_loader.dataset)) + 1,
                "tryon1": visuals_new[0],
                "tryon2": visuals_new[1],
                "tryon3": visuals_new[2],
                "model_suffix": os.path.basename(opt.checkpoint),
            })



def main():
    opt = get_opt()
    print(opt)
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))
    
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    #board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
    board = None
    
    # create model & train
    if opt.stage == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, train_loader, model, board)
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, train_loader, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
  
    print('Finished test %s, named: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
