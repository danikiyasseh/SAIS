# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

""" My Packages """
from sklearn.preprocessing import LabelEncoder

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    # ======= My Arguments ======= #
    parser.add_argument("--optical_flow_to_reps", default=False, action='store_true')
    parser.add_argument("--segmentation_to_reps", default=False, action='store_true')
    parser.add_argument('--optical_flow',default=False,action='store_true',help='generate and save optical flow images')
    parser.add_argument('--segmentation',default=False,action='store_true',help='generate and save segmentation images')
    parser.add_argument("--task", default='DINO')
    return parser

def getSets(df,dataset_name,phase,train_fraction,args):        
    if dataset_name == 'LapGyn4_v1.2':
        indices = [] #category is Instrument Count for downstream applications but all of them for pre-training
        if args.task == 'DINO':
            category_list = ['Anatomical_Structures','Actions_on_Anatomy','Instrument_Count','Surgical_Actions']
        else:
            category_list = ['Instrument_Count']
            
        new_df = pd.DataFrame()
        for category in category_list:
            cat_indices = np.where(df['category']==category)[0]
            curr_df = df.iloc[cat_indices,:]
            if category in ['Anatomical_Structures','Actions_on_Anatomy','Surgical_Actions']:
                curr_df['video'] = curr_df['path'].apply(lambda path:path.split('\\')[-1].split('_')[3])
            elif category == 'Instrument_Count': #DONE
                curr_df['video'] = curr_df['path'].apply(lambda path:path.split('\\')[-1].strip('.jpg'))
            all_videos = curr_df['video'].tolist()
            enc, curr_indices = getIndices(curr_df,dataset_name,all_videos,phase,train_fraction)
            #indices.extend(curr_indices)
            new_df = pd.concat((new_df,curr_df.iloc[curr_indices,:]),axis=0)
    elif dataset_name in ['Glenda_v1.0','Nephrec9','SurgicalActions160']:
        if dataset_name == 'Glenda_v1.0': #DONE
            df['video'] = df['path'].apply(lambda path:path.split('\\')[4].split('_')[1]) 
            if args.task != 'DINO':
                df = df.groupby(by=['video']).apply(lambda row:row.iloc[0:-1:10]).reset_index(drop=True) #downsample per video to avoid overfitting
        elif dataset_name == 'Nephrec9': #DONE
            #df['video'] = df['path'].apply(lambda path:path.split('\\')[-1].split('-')[1])
            df['video'] = df['path'].apply(lambda file:file.split('\\')[-1].split('_')[1][-1]) #should be 9 unique videos
            if args.task != 'DINO':
                df['segment'] = df['path'].apply(lambda file:file.split('\\')[-1].split('-')[1])
                df = df.groupby(by=['segment']).apply(lambda row:row.iloc[0:-1:10]).reset_index(drop=True) #downsample per segment to avoid overfitting
        elif dataset_name == 'SurgicalActions160': #DONE
            df['video'] = df['path'].apply(lambda path:path.split('\\')[2].split('_')[1]) #video level splits
        all_videos = df['video'].tolist()
        enc, indices = getIndices(df,dataset_name,all_videos,phase,train_fraction)
        new_df = df.iloc[indices,:] 
    elif dataset_name in ['cholec80']:
        df['video'] = df['category']
        if args.task != 'DINO':
            df = df.groupby(by=['video']).apply(lambda row:row.iloc[0:-1:25]).reset_index(drop=True)
        all_videos = df['video'].tolist()
        enc, indices = getIndices(df,dataset_name,all_videos,phase,train_fraction)
        new_df = df.iloc[indices,:] 
    elif dataset_name in ['NS','VUA','VUA_Gronau','VUA_HMH']:
        df['video'] = df['label'] #df['path'].apply(lambda path:path.split('\\')[2].split('_')[1])
        all_videos = df['video'].tolist()
        enc, indices = getIndices(df,dataset_name,all_videos,phase,train_fraction)
        new_df = df.iloc[indices,:] 
    
    if args.task != 'DINO': # DINO does not need labels 
        new_df['label'] = enc.transform(new_df['label'])
    new_df['dataset'] = dataset_name
    return new_df

def getIndices(df,dataset_name,all_videos,phase,train_fraction):
    if dataset_name in ['cholec80','Nephrec9','LapGyn4_v1.2','Glenda_v1.0','SurgicalActions160']:
        class_videos = df.groupby(by=['label'])['video'].unique()
        class_videos_df = pd.DataFrame(class_videos).reset_index()
        indices = defaultdict(list)
        for r,(label,videos) in class_videos_df.iterrows():
            nvideos = len(videos)
            random.seed(0)
            videos_shuffled = random.sample(list(videos),nvideos)
            train_frac,val_frac = 0.6,0.2
            ntrain, nval = int(train_frac*nvideos), int(val_frac*nvideos)
            train_videos, val_videos, test_videos = videos_shuffled[:ntrain], videos_shuffled[ntrain:ntrain+nval], videos_shuffled[ntrain+nval:]
            bool1 = df['label']==label
            train_indices, val_indices, test_indices = np.where((df['video'].isin(train_videos)) & bool1)[0], np.where((df['video'].isin(val_videos)) & bool1)[0], np.where((df['video'].isin(test_videos)) & bool1)[0]
            
            if train_fraction < 1:
                random.seed(0)
                tot_samples = len(train_indices)
                nsamples = int(train_fraction * tot_samples)
                train_indices = random.sample(list(train_indices),nsamples)
            
            indices['train'].extend(train_indices)
            indices['val'].extend(val_indices)
            indices['test'].extend(test_indices)
            
        train_indices, val_indices, test_indices = list(indices.values())
    elif dataset_name in ['NS','VUA','VUA_Gronau','VUA_HMH']:
        train_indices, val_indices, test_indices = list(range(df.shape[0])), [], []
    
    if phase == 'train':
        chosen_indices = train_indices
    elif phase == 'val':
        chosen_indices = val_indices
    elif phase == 'test':
        chosen_indices = test_indices
    
    enc = LabelEncoder()
    enc.fit(df.iloc[train_indices,:]['label'])
    
    return enc, chosen_indices

def getNewNSVids():
    return ['1']


def getNewVUAVids():
    return ['1']


class SurgDataset(torch.utils.data.Dataset):

    def __init__(self,phases,args,train_fraction,dataset_list,transform='',extract_only=False):
        data_path = args.data_path#.replace('md3','md2')
        df = pd.DataFrame()
        for dataset in dataset_list:
            print(dataset)
            if args.optical_flow_to_reps == True: #load optical flow paths
                curr_df = pd.read_csv(os.path.join(data_path,'paths','%s_FlowPaths.csv' % dataset),index_col=0)
                print('Loaded Optical Flow Paths!')
                #curr_df = curr_df.sample(n=10000,replace=False)
            elif args.segmentation_to_reps == True:
                curr_df = pd.read_csv(os.path.join(data_path,'paths','%s_SegPaths.csv' % dataset),index_col=0)    
                print('Loaded Segmentation Paths!')
            else: #load RGB paths
                curr_df = pd.read_csv(os.path.join(data_path,'paths','%s_Paths.csv' % dataset),index_col=0)
                print('Loaded RGB Paths!')
            
            if extract_only == False: #split into train / val / test
                for phase in phases:
                    phase_df = getSets(curr_df,dataset,phase,train_fraction,args)
                    phase_df['dataset'] = dataset
                    df = pd.concat((df,phase_df),axis=0)
            else: #extract all paths without shuffling
                curr_df['dataset'] = dataset
                df = pd.concat((df,curr_df),axis=0)
                
        print(df.shape)
        self.dataset = dataset_list[0]
        self.args = args
        self.df = df
        self.transform = transform
        self.data_path = data_path
        
    def __getitem__(self,idx):
        row = self.df.iloc[idx,:]
        dataset = row['dataset']
        if self.args.optical_flow_to_reps == True:
            framepath = row['flowpath']
        elif self.args.segmentation_to_reps == True:
            framepath = row['segpath']
        else:
            framepath = row['path']
        
        data_path = './SAIS' #/mnt/md2/kiyasseh/SurgicalDatasets' # md2
        
        framepath = framepath.replace('\\','/') #artefact of Windows to Linux
        framepath = os.path.join(data_path,framepath)
        with open(framepath, 'rb') as f:
            img = Image.open(f)
            img.convert('RGB')
            ## NEW - remove borders ##
            width,height = img.size
            height_frac, width_frac = self.getCropDims()
            img = transforms.CenterCrop((height_frac*height,width_frac*width))(img)
            
            if self.dataset in ['VUA_Lab','VUA_AFB']: # new for lab study
                new_height,new_width = height_frac*height, width_frac*width
                img = transforms.functional.crop(img,0,0,new_height,new_width-130) # to remove timer on right-hand side
            ## END ##
            if self.args.segmentation == True:
                framename = 'frame_' if 'frame_' in framepath else 'frames_'
                meta = int(framepath.split(framename)[-1].strip('.jpg'))
            else:
                meta = row['dataset']
                
        sample = self.transform(img)
        label = row['label']
        #dataset_name = row['dataset']
        return sample, label, meta

    def getCropDims(self):
        if self.dataset in ['NS_Gronau','VUA_Gronau']:
            height_frac,width_frac = 0.8, 0.7 #original videos have more padding width-wise, so chop more off 
        else:
            height_frac,width_frac = 0.8, 0.8
        return height_frac,width_frac
    
    def __len__(self):
        return self.df.shape[0]

def train_dino(args):
    utils.init_distributed_mode(args)
    print(args.gpu)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    #dataset = datasets.ImageFolder(args.data_path, transform=transform)
    """ SurgicalVideoNet Path """
    #dataset_list = ['cholec80','Glenda_v1.0','Nephrec9','SurgicalActions160','LapGyn4_v1.2'] # ['NS'] #
    #phases = ['train','val']
    """ NS Path """
    dataset_list = ['VUA','VUA_Gronau','VUA_HMH']
    phases = ['train']
    train_fraction = 1
    dataset = SurgDataset(phases,args,train_fraction,dataset_list,transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
    
        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        #teacher = nn.DataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    #student = nn.DataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ freeze some parameters ... =========== #
#     names_to_train = ['blocks.11.norm1.weight', 'blocks.11.norm1.bias', 'blocks.11.attn.qkv.weight', 'blocks.11.attn.qkv.bias', 'blocks.11.attn.proj.weight', 'blocks.11.attn.proj.bias', 'blocks.11.norm2.weight', 'blocks.11.norm2.bias', 'blocks.11.mlp.fc1.weight', 'blocks.11.mlp.fc1.bias', 'blocks.11.mlp.fc2.weight', 'blocks.11.mlp.fc2.bias', 'norm.weight', 'norm.bias']
#     for name,param in student.named_parameters():
#         if name not in names_to_train:
#             param.requires_grad_(False)
    
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    print('# of GPUs: %i' % utils.get_world_size())
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 2}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "dino_deitsmall16_pretrain_VUA_epoch2.pth"), #dino_deitsmall16_pretrain_surgicalvideonet_epoch1.pth
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            #param_group["lr"] = 1e-5
            #print('LR: %.8f' % lr_schedule[it])
            #print(list(student.parameters())[0][0])
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC), #224
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC), #224
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC), #96
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
