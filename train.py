#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, VGGLoss_new, load_checkpoint, save_checkpoint

from torch.utils.tensorboard import SummaryWriter
from visualization import board_add_image, board_add_images

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
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument('--result_dir', type=str, default='result', help='save result output')
    parser.add_argument('--warp_dir', type=str, default='', help='warp output')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    # mesllo
    parser.add_argument('--rel', type=str, default='baseline', help='specify which relevance level we used')
    parser.add_argument('--fin', type=str, default='ALL', help='specify which finetuning config we used')
    parser.add_argument('--plf_layers', type=str, default='DEFAULT', help='specify which plf layers we used')
    parser.add_argument('--pln', type=str, default='vgg19bn', help='specify which pln we used')
    parser.add_argument('--dataset', type=str, default='viton_quick', help='specify which dataset we used for vton')
    parser.add_argument('--pretrained', action='store_true', help='Specify whether we use a pretrained or randomly initialized PLN')
    parser.add_argument('--pln_path', type=str, default=None, help='path to finetuned pln model for perceptual loss')

    opt = parser.parse_args()
    return opt

def train_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

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

    # criterion
    criterionL1 = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    # view model tensors
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(count)
            print("\t",name,"\t",param.size())
            print(param.data)
        count = count + 1
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
            
        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        
        loss = criterionL1(warped_cloth, im_c)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            #board_add_images(board, 'combine', visuals, step+1)
            #board.add_scalar('metric', loss.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

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
                "epoch": ((step+1) // len(train_loader.dataset)) + 1,
                "warping1": visuals_new[0],
                "warping2": visuals_new[1],
                "warping3": visuals_new[2],
                "training_loss": loss.item()
            })

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))
    
    # view trained model tensors
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(count)
            print("\t",name,"\t",param.size())
            print(param.data)
        count = count + 1


def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()

    ### WANDB STUFF FOR EXP TRACKING
    wandb.init(
        project="CP-VTON - Finetuned PLN",
        config = {
            "stage": opt.stage,
            "mode": opt.datamode,
            "relevance_level": opt.rel,
            "finetune_config": opt.fin,
            "plf_layers": opt.plf_layers,
            "pln": opt.pln,
            "dataset": opt.dataset,
            "warp_dir": opt.warp_dir
        }
    )

    wandb.run.name = opt.name
    ###
    
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss(opt)
    #criterionVGG = VGGLoss_new(opt)
    criterionMask = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
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
        p_tryon = c * m_composite+ p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose],
                   [c, cm*2-1, m_composite*2-1], 
                   [p_rendered, p_tryon, im]]
            
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im, opt)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step+1) % opt.display_count == 0:
            #board_add_images(board, 'combine', visuals, step+1)
            #board.add_scalar('metric', loss.item(), step+1)
            #board.add_scalar('L1', loss_l1.item(), step+1)
            #board.add_scalar('VGG', loss_vgg.item(), step+1)
            #board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f' 
                    % (step+1, t, loss.item(), loss_l1.item(), 
                    loss_vgg.item(), loss_mask.item()), flush=True)

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
                "epoch": ((step+1) // len(train_loader.dataset)) + 1,
                "tryon1": visuals_new[0],
                "tryon2": visuals_new[1],
                "tryon3": visuals_new[2],
                "training_loss": loss.item(),
                "L1_loss": loss_l1.item(),
                "VGG_loss": loss_vgg.item(),
                "MaskL1_loss": loss_mask.item(),
            })

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))



def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    #board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
    board = None
    
    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        #print('Model details:')
        #summary(model, input_size=(opt.batch_size, 22, 256, 192), row_settings=("depth", "ascii_only"))

        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        #print('Model details:')
        #summary(model, input_size=(opt.batch_size, 22, 192, 256), row_settings=("depth", "ascii_only"))

        checkpoint_path = os.path.join(opt.checkpoint_dir, opt.checkpoint)
        print(checkpoint_path)
        if not opt.checkpoint == '' and os.path.exists(checkpoint_path):
            print(checkpoint_path)
            load_checkpoint(model, checkpoint_path)
        train_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
        
  
    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
