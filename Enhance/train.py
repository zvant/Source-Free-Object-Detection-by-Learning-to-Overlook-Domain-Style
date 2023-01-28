import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import net
from sampler import InfiniteSamplerWrapper
import numpy as np
import itertools

cudnn.benchmark = False
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform(scenario_name):
    if scenario_name=='voc2wc':
        transform_list = [
            transforms.Resize(600),
            transforms.RandomCrop(128),
        ]
    else:
        transform_list = [
            transforms.Resize(size=(600, 800)),
            transforms.RandomCrop(128),
        ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*.jpg'))
        self.transform = transform
        self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        img = np.array(img)
        img = img[:, :, ::-1]
        if np.random.rand() >= 0.5:
            img = img[:, ::-1, :]
        img = img.astype(np.float32, copy=False)
        img -= self.pixel_means
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(init_lr,optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = init_lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# python train.py --scenario_name voc2clipart --content_dir data/voc2clipart --style_dir data/voc2clipart --vgg pre_trained/vgg16_ori.pth --save_dir models/voc2clipart

parser = argparse.ArgumentParser()
parser.add_argument('--scenario_name', type=str, choices=['coco2scenes100', 'voc2clipart', 'voc2wc', 'city2foggy' ,'KC'],
                    help='choose one from voc2clipart, voc2wc, city2foggy and KC')
parser.add_argument('--id', type=str)
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')

parser.add_argument('--vgg', type=str, default='./models/vgg16.pth')

parser.add_argument('--save_dir', default='./models',
                    help='Directory to save the model')
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--save_model_interval', type=int, default=-1)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--lr_decoder', type=float, default=1e-4)
parser.add_argument('--lr_fcs', type=float, default=1e-4)
parser.add_argument('--style_weight', type=float, default=50)
parser.add_argument('--content_weight', type=float, default=1)
parser.add_argument('--content_style_weight', type=float, default=1)
parser.add_argument('--constrain_weight', type=float, default=1)
parser.add_argument('--before_fcs_steps', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda')

args.save_dir = os.path.join(os.path.dirname(__file__), 'models')
args.vgg = os.path.join(os.path.dirname(__file__), 'models', 'vgg16_ori.pth')
args.content_dir = args.style_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Intersections', 'images', 'train_lmdb', args.id, 'jpegs')

decoder = net.decoder
vgg = net.vgg
fc1 = net.fc1
fc2 = net.fc2

vgg.load_state_dict(torch.load(args.vgg)['model'])
vgg = nn.Sequential(*list(vgg.children())[:19])

network = net.Net(vgg, decoder,fc1,fc2)
network.train()
network.to(device)

content_tf = train_transform(None)
style_tf = train_transform(None)

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

# optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)
optimizer1 = torch.optim.Adam(itertools.chain(*[network.dec_1.parameters(),network.dec_2.parameters(), network.dec_3.parameters(), network.dec_4.parameters()]), lr=args.lr_decoder)
optimizer2 = torch.optim.Adam(itertools.chain(*[network.fc1.parameters(),network.fc2.parameters()]), lr=args.lr_fcs)

# for V in 001 003 005 006 007 008 009 011 012 013 014 015 016 017 019 020 023 025 027 034 036 039 040 043 044 046 048 049 050 051 053 054 055 056 058 059 060 066 067 068 069 070 071 073 074 075 076 077 080 085 086 087 088 090 091 092 093 094 095 098 099 105 108 110 112 114 115 116 117 118 125 127 128 129 130 131 132 135 136 141 146 148 149 150 152 154 156 158 159 160 161 164 167 169 170 171 172 175 178 179 ; do python train.py --id ${V} ; done

for i in tqdm(range(args.max_iter), ascii=True):
    adjust_learning_rate(args.lr_decoder,optimizer1, iteration_count=i)

    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)

    loss_c,loss_const = network(content_images, style_images,args.scenario_name,flag=0)
    loss_c = args.content_weight * loss_c
    loss_const=args.constrain_weight *loss_const
    loss = loss_c + loss_const

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss.backward()
    optimizer1.step()

    if i>=args.before_fcs_steps:
        adjust_learning_rate(args.lr_fcs,optimizer2, iteration_count=i-args.before_fcs_steps)
        loss_s_1,loss_s_2 = network(content_images, style_images,args.scenario_name,flag=1)
        loss_s_1 = args.style_weight * loss_s_1
        loss_s_2 = args.content_style_weight * loss_s_2
        loss =  loss_s_1 + loss_s_2

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

    save = False
    if args.save_model_interval != -1:
        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            save = True
    else:
        if (i + 1) == args.max_iter:
            save = True
    if save:
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, os.path.join(args.save_dir, 'decoder_%s_iter_%d.pth' % (args.id, i + 1)))

        state_dict = net.fc1.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, os.path.join(args.save_dir, 'fc1_%s_iter_%d.pth' % (args.id, i + 1)))

        state_dict = net.fc2.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, os.path.join(args.save_dir, 'fc2_%s_iter_%d.pth' % (args.id, i + 1)))
