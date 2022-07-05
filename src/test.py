# +
import os
import random
import argparse
import cv2 as cv
import numpy as np
from skimage import color
import torch
from model import *
from data_generator import *
from PIL import Image
import torchvision.transforms as transforms


def lab2rgb(L, ab):
    """
    L : range: [-1, 1], torch tensor
    ab : range: [-1, 1], torch tensor
    """
    ab2 = ab * 100.0
    L2 = (L + 1.0) * 50.0
    Lab = torch.cat([L2, ab2], dim=1)
    Lab = Lab[0].detach().cpu().float().numpy()
    Lab = np.transpose(Lab.astype(np.float32), (1, 2, 0))
    rgb = color.lab2rgb(Lab) * 255
    return rgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--h_model_path", default='', type=str, help='head encoder model path')
    parser.add_argument("--t_model_path", default='', type=str, help='tail encoder model path')
    parser.add_argument("--d_model_path", default='', type=str, help='decoder model path')
    parser.add_argument("--source", type=str)
    parser.add_argument("--domain", default='photo', type=str)
    parser.add_argument("--data_dir", default='./pacs_data', type=str)
    parser.add_argument("--save_dir", default='./out', type=str)
    parser.add_argument("--gpu", default='', type=str)
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    domain = args.domain
    source = args.source

    
    test_dataset = PacsImageDataset(root=os.path.join(args.data_dir, domain), testing=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    transform = transforms.Compose([transforms.Resize((256, 256))])

    head_encoder = HeadEncoder(1, 64)  # feature extractor (partial encoder - layer3)
    tail_encoder = TailEncoder(64)  # content-biased encoder
    decoder = Decoder(64, 2)
    
    head_encoder.to(device)
    tail_encoder.to(device)
    decoder.to(device)
    head_encoder.load_state_dict(torch.load(args.h_model_path, map_location=torch.device(device)))
    tail_encoder.load_state_dict(torch.load(args.t_model_path, map_location=torch.device(device)))
    decoder.load_state_dict(torch.load(args.d_model_path, map_location=torch.device(device)))
    

    head_encoder.eval()
    tail_encoder.eval()
    decoder.eval()
    for i, batch in enumerate(test_loader):
        x, y, _, _ = batch
        x, y = x.to(device), y.to(device)

        z_head_set = head_encoder(x)
        z_tail_set = tail_encoder(z_head_set[-1])
        z_set = z_head_set + z_tail_set
        pred_ab = decoder(z_set, adain_layer_idx=5)

        pred_rgb = lab2rgb(x, pred_ab) # pred_rgb : (h, w, 3), range : [0, 1]
        real_rgb = lab2rgb(x, y)
        out_bgr = cv.cvtColor(pred_rgb.astype('uint8'), cv.COLOR_RGB2BGR)
        gray = cv.cvtColor(out_bgr, cv.COLOR_BGR2GRAY)
        bgr = cv.cvtColor(real_rgb.astype('uint8'), cv.COLOR_RGB2BGR)

        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'gray/%s2%s' % (source, domain)), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'gt/%s2%s' % (source, domain)), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'colorized/%s2%s' % (source, domain)), exist_ok=True)
        cv.imwrite(os.path.join(args.save_dir, 'gray/{}2{}/{}_image.jpg'.format(source, domain, i)), gray)
        cv.imwrite(os.path.join(args.save_dir, 'gt/{}2{}/{}.jpg'.format(source, domain, i)), bgr)
        cv.imwrite(os.path.join(args.save_dir, 'colorized/{}2{}/{}.jpg'.format(source, domain, i)), out_bgr)
