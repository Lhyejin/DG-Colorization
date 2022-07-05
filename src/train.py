# -*- coding: utf-8 -*-
# +
import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import time
import random
from PIL import Image

from model import *
from data_generator import *
from utils import *


def opt(param, lr, betas):
    return optim.Adam(param, lr=lr, betas=betas)

# Reference https://github.com/hyeonseobnam/sagnet
def randomize(x, mode='style', eps=1e-5):
    device = "cuda" if x.is_cuda else "cpu"
    sizes = x.shape
    alpha = torch.rand(sizes[0], 1).to(device)
    
    if len(sizes) == 4:
        x = x.view(sizes[0], sizes[1], -1)
        alpha = alpha.unsqueeze(-1)

    # channel-wise feature map mean, variance = style feature
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True)
    x = (x - mean) / (var + eps).sqrt() # normalize
    idx_swap = torch.randperm(sizes[0])
    if mode == 'style': # style randomization
        mean = alpha * mean + (1 - alpha) * mean[idx_swap]
        var = alpha * var + (1 - alpha) * var[idx_swap]
    else: # content randomization
        x = x[idx_swap].detach()
         
    x = x * (var + eps).sqrt() + mean
    return x.view(*sizes) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="yaml file path")
    parser.add_argument("--save_dir", type=str, default=None, help="model path to load")
    parser.add_argument("--w-adv", type=float, default=1.0, help='weight of adv loss')
    parser.add_argument("--gpu", type=str, default="" , help="gpu number")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config)
    print(args, config)

    if args.save_dir is not None:
        config['save_dir'] = args.save_dir
    
    epochs = config['epochs']
    lr = config['lr']
    weight_adv_s = args.w_adv
    save_freq = config['save_freq']
    domains = config['source_domain']
    src_domain_dict = {}
    domain_names = ""
    for i, d in enumerate(domains):
        domain_names += d
        src_domain_dict[d] = i 
    save_dir = os.path.join(config['save_dir'], domain_names)
    train_dir = config['train_dir']
    val_dir = config['val_dir']
    os.makedirs(save_dir, exist_ok=True)
    print('model save path: ', save_dir)
    print('source_domain:', src_domain_dict)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    # dataset load
    # source domain contain color domain
    trainset = []
    valset = []
    for d in domains:
        trainset.append(PacsImageDataset(os.path.join(train_dir, d), domain=d, domain_dict=src_domain_dict))
        valset.append(PacsImageDataset(os.path.join(val_dir, d), domain=d, domain_dict=src_domain_dict))
    source_train_dataset = torch.utils.data.ConcatDataset(trainset)
    source_val_dataset = torch.utils.data.ConcatDataset(valset)

    source_train_loader = torch.utils.data.DataLoader(source_train_dataset,
                                                     batch_size=config['batch_size'],
                                                     shuffle=True,
                                                     num_workers=4)
    
    # Initialize Model
    num_classes = len(trainset[0].classes)
    head_encoder = HeadEncoder(1, 64)
    tail_encoder = TailEncoder(64)
    net_class = ClassificationModel(num_classes=num_classes)
    net_domain_disc = StyleDiscriminator(64, len(domains))
    decoder = Decoder(64, 2)
    color_discriminator = Discriminator(3, 64, 1)

    loss_fn_GAN = torch.nn.MSELoss()
    loss_fn_l1 = torch.nn.L1Loss()

    # send to device
    head_encoder.to(device)
    tail_encoder.to(device)
    net_class.to(device)
    net_domain_disc.to(device)
    decoder.to(device)
    color_discriminator.to(device)
    loss_fn_GAN.to(device)
    loss_fn_l1.to(device)
    
    # Initialize optimizer
    optim_head = opt(head_encoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_tail = opt(tail_encoder.parameters(), lr=lr, betas=(0.5, 0.999))
    f_bn_param = [] # optimize only bathNorm in adversarial training
    for name, param in head_encoder.named_parameters():
        if 'bn' in name:
            f_bn_param += [param]
    optim_head_bn = opt(f_bn_param, lr=lr, betas=(0.5, 0.999))
    optim_class = opt(net_class.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_s = opt(net_domain_disc.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_decoder = opt(decoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_color_disc = opt(color_discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Initialize model parameter
    head_encoder.normal_weight_init()
    tail_encoder.normal_weight_init()
    net_domain_disc.normal_weight_init()
    decoder.normal_weight_init()
    color_discriminator.normal_weight_init()


    for epoch in range(epochs):
        running_G_loss = 0.0
        running_D_loss = 0.0
        running_style_loss = 0.0
        running_adv_style_loss = 0.0
        running_class_loss = 0.0
        start_time = time.time()
        
        for i, batch in enumerate(source_train_loader):
            x, real_ab, class_labels, domain_labels = batch
            x = x.to(device) # x: (-1, 1, 256, 256)
            real_ab = real_ab.to(device) # y: (-1, 2, 256, 256)
            class_labels = class_labels.to(device)
            domain_labels = domain_labels.to(device)

            # Training color discriminator (update optim_color_disc)
            optim_color_disc.zero_grad()
            z_head_set = head_encoder(x)  # z_head_set: [x1, x2, x3]
            z_tail_set = tail_encoder(z_head_set[-1])  # z_tail_set: [x4, ..., x8]
            z_set = z_head_set + z_tail_set
            fake_ab = decoder(z_set, adain_layer_idx=5)
            fake_lab = torch.cat((x, fake_ab), 1)
            pred_fake = color_discriminator(fake_lab)
            fake_label = torch.zeros_like(pred_fake).to(device)
            loss_D_fake = loss_fn_GAN(pred_fake, fake_label)

            real_lab = torch.cat((x, real_ab), 1)
            pred_real = color_discriminator(real_lab)
            real_label = torch.ones_like(pred_real).to(device)
            loss_D_real = loss_fn_GAN(pred_real, real_label)
            
            D_loss = (loss_D_fake + loss_D_real) * 0.5
            D_loss.backward()
            optim_color_disc.step()

            # class prediction (training semantic information)
            optim_head.zero_grad()
            optim_tail.zero_grad()
            optim_class.zero_grad()
            
            z_head_set = head_encoder(x)  # z_head_set: [x1, x2, x3]
            z_tail_set = tail_encoder(z_head_set[-1])  # z_tail_set: [x4, ..., x8]
            pred_class = net_class(z_tail_set[-1])
            loss_class = F.cross_entropy(pred_class, class_labels)
            loss_class.backward()
            optim_head.step()
            optim_tail.step()
            optim_class.step()

            # Training generator (update encoder, decoder)
            optim_head.zero_grad()
            optim_tail.zero_grad()
            optim_decoder.zero_grad()

            z_head_set = head_encoder(x)  # z_head_set: (x1, x2, x3)
            z_tail_set = tail_encoder(z_head_set[-1])   # z_tail_set: (x4, ..., x8)
            z_set = z_head_set + z_tail_set
            fake_ab = decoder(z_set, adain_layer_idx=5)
            fake_lab = torch.cat((x, fake_ab), 1)
            pred_fake = color_discriminator(fake_lab)
            real_label = torch.ones_like(pred_fake).to(device) # adversarial
            loss_G_fake = loss_fn_GAN(pred_fake, real_label)
            loss_l1 = loss_fn_l1(fake_ab, real_ab)
            G_loss = 100 * loss_l1 + loss_G_fake
            G_loss.backward()
            optim_head.step()
            optim_tail.step()
            optim_decoder.step()


            # texutre-biased domain discriminator prediction
            optim_s.zero_grad()
            z_head_set = head_encoder(x)  # z_head_set: (x1, x2, x3)
            z_cr = randomize(z_head_set[-1], mode='content')
            pred_domain_labels = net_domain_disc(z_cr)
            loss_s = F.cross_entropy(pred_domain_labels, domain_labels)
            loss_s.backward()
            optim_s.step()

            # 3. adversarial texutre-biased training (update optim_head_bn)
            optim_head_bn.zero_grad()
            z_head_set = head_encoder(x)
            z_cr = randomize(z_head_set[-1], mode='content')
            pred_domain_labels = net_domain_disc(z_cr)
            loss_adv = -F.cross_entropy(pred_domain_labels, domain_labels)
            loss_adv = loss_adv * weight_adv_s
            loss_adv.backward()
            optim_head_bn.step()
            
            
            # print loss
            running_G_loss += G_loss.item()
            running_D_loss += D_loss.item()
            running_style_loss += loss_s.item()
            running_adv_style_loss += loss_adv.item()
            print(f'progress epoch/batch {epoch+1}/{i}, G Loss: {G_loss}, D Loss: {D_loss}, '
                  f'style loss: {loss_s}, adv loss: {loss_adv}')
            
        if (epoch+1) % save_freq == 0:
            # save network_f, network_c, decoder
            torch.save(head_encoder.state_dict(), os.path.join(save_dir, 'F_%d.pth' % (epoch + 1)))
            torch.save(tail_encoder.state_dict(), os.path.join(save_dir, 'C_%d.pth' % (epoch + 1)))
            torch.save(decoder.state_dict(), os.path.join(save_dir, 'D_%d.pth'%(epoch+1)))

        running_time = time.time() - start_time
        
        print(f'current epoch : {epoch+1}, G Loss: {running_G_loss}, D Loss: {running_D_loss}'
              f'style loss: {running_style_loss}, adv loss: {running_adv_style_loss}, time: {running_time}')
        
            
            
            
    
    
    
