import torch
import os
from tqdm import tqdm
tqdm.pandas()
import argparse
import sys
sys.path.append('./SAIS/scripts/dino-main')
#sys.path.append('/mnt/md2/kiyasseh/Scripts/segmentation')
from main_dino import *
#from ternaus import UNet11, UNet16
import timm
import h5py
import numpy as np
from operator import itemgetter

import cv2 as cv
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
from collections import OrderedDict

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from moviepy.editor import VideoFileClip, ImageSequenceClip
import torchvision # new
import time

class OpticalFlowDataset(torch.utils.data.Dataset):

    def __init__(self,rank,world_size,data_path,dataset_list,pid,jump_size,transform='',extract_only=False):
        model = ptlflow.get_model('raft', pretrained_ckpt='things')
        self.model = model #just for IO Adapter stuff
        #model.to(rank)
        #model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        df = pd.DataFrame()
        for dataset in dataset_list:
            print(dataset)
            curr_df = pd.read_csv(os.path.join(data_path,'paths','%s_FlowPaths.csv' % dataset)) #.replace('md3','md2')
            curr_df = curr_df[curr_df['label']==pid]
            if extract_only == False: #split into train / val / test
                curr_df = getTrainValSets(curr_df)
            df = pd.concat((df,curr_df),axis=0)
            
        print(df.shape)
        self.jump_size = jump_size
        self.dataset = dataset_list[0]
        self.df = df
        self.data_path = data_path
        self.transform = transform
        #self.model = model
        #self.rank = rank

    def process_inputs(self,path1,path2):
        images = [
            cv.imread(path1),
            cv.imread(path2)
        ]
        #print(path1,path2)
        # A helper to manage inputs and outputs of the model
        io_adapter = IOAdapter(self.model, images[0].shape[:2], cuda=torch.cuda.is_available())
        
        # inputs is a dict {'images': torch.Tensor}
        # The tensor is 5D with a shape BNCHW. In this case, it will have the shape:
        # (1, 2, 3, H, W)
        inputs = io_adapter.prepare_inputs(images)
        inputs['images'] = inputs['images'].squeeze(0) # in order to allow for batch setup
        #self.io_adapter = io_adapter
        return inputs 

#     def obtain_flow(self,inputs):
#         inputs['images'] = inputs['images'] #.to(self.rank)
#         predictions = self.model(inputs)

#         # Remove extra padding that may have been added to the inputs
#         predictions = self.io_adapter.unpad_and_unscale(predictions)

#         # The output is a dict with possibly several keys,
#         # but it should always store the optical flow prediction in a key called 'flows'.
#         flows = predictions['flows']

#         # flows will be a 5D tensor BNCHW.
#         # This example should print a shape (1, 1, 2, H, W).
#         #print(flows.shape)

#         # Create an RGB representation of the flow to show it on the screen
#         flow_rgb = flow_utils.flow_to_rgb(flows)
#         # Make it a numpy array with HWC shape
#         flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
#         flow_rgb_npy = flow_rgb.detach().cpu().numpy()
#         flow_im = Image.fromarray(np.uint8(flow_rgb_npy*255))
#         #flow_im.save('/mnt/md2/kiyasseh/SurgicalDatasets/NS/sample_flow.jpg')
#         #print('flow saved!')
#         return flow_im
            
    def __getitem__(self,idx):
        row = self.df.iloc[idx,:]

        framepath1 = row['path1']
        framepath2 = row['path2']
        #frame = int(framepath2.split('frame_')[-1].strip('.jpg'))
        framepath1 = framepath1.replace('\\','/') #artefact of Windows to Linux
        framepath2 = framepath2.replace('\\','/')
        framepath1 = os.path.join(self.data_path,framepath1)
        framepath2 = os.path.join(self.data_path,framepath2)

        if 'frames' in framepath1:
            name = 'frames_'
        elif 'frame' in framepath1:
            name = 'frame_'
        nflow = int(framepath1.split(name)[-1].strip('.jpg')) // self.jump_size
        
        #if 'frames' in framepath1: # some datasets are saved as frameS (with an 's')
        #    nflow = int(framepath1.split('frames_')[-1].strip('.jpg')) // 15 # flow number (for ordering later on)
        #elif 'frame' in framepath1:
        #    nflow = int(framepath1.split('frame_')[-1].strip('.jpg')) // 15 # flow number (for ordering later on)
        #with open(framepath, 'rb') as f:
        #    img = Image.open(f)
        #    img.convert('RGB')
        flow_inputs = self.process_inputs(framepath1,framepath2)
        if self.dataset == 'CinVivo': # b/c the images are huge
            flow_inputs['images'] = torchvision.transforms.functional.resize(flow_inputs['images'],[216,384])

        #flow_img = self.obtain_flow(flow_inputs)
        #sample = self.transform(flow_img)
#         """ New - Resize if Too Large (Computational Reasons) """
#         print(flow_inputs.keys())
#         n,c,h,w = flow_inputs['images'].shape
#         if w > 1000:
#             new_inputs = dict()
#             im1 = transforms.Resize((540,960))(flow_inputs['images'][0,:])
#             im2 = transforms.Resize((540,960))(flow_inputs['images'][1,:])
#             print(torch.stack([im1,im2]).shape)
#             new_inputs['images'] = torch.stack([im2,im2])
#             flow_inputs = new_inputs
#         """ End """
        
        label = row['label']
        return flow_inputs, label, nflow

    def __len__(self):
        return self.df.shape[0]

def prepareDataloader(rank,world_size,dataset_list,args,pid=None,jump_size=None,train_fraction=1):
    if 'SelfSupervised' in args.model_type:
        if args.arch == 'vit_small': #from DINO
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif 'Supervised' in args.model_type:
        if 'vit' in args.arch: #from TIMM
            mean, std = (0.500, 0.500, 0.500), (0.500, 0.500, 0.500)
#     transform = transforms.Compose([
#             transforms.Resize(248),
#             transforms.CenterCrop(224), #put center crop first and in the dataset loader __getitem__
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std),
#             ])
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
    if args.optical_flow == True:
        dataset = OpticalFlowDataset(rank,world_size,args.data_path,dataset_list,pid,jump_size,transform=transform,extract_only=True)
    elif args.segmentation == True:
        transform = transforms.Compose([
                #transforms.CenterCrop((0.8*height,0.8*width)), # moved to getitem in dataset loader
                transforms.Resize((224,224)), # 1024,1280 # b/c segmentation network expects this dimensionality
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        phases = 'all' #placeholder - I always extract all paths irrespective of phase
        dataset = SurgDataset(phases,args,train_fraction,dataset_list,transform=transform,extract_only=True)
    else:
        phases = 'all' #placeholder - I always extract all paths irrespective of phase
        dataset = SurgDataset(phases,args,train_fraction,dataset_list,transform=transform,extract_only=True)
    #dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size_per_gpu,pin_memory=True,shuffle=False)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size_per_gpu,shuffle=False)
    return dataloader

def loadModel(rank,world_size,dataset,args):
    """ model_type (str): options: ViT_SelfSupervised_ImageNet """
    #print(args.model_type)
    if 'SelfSupervised' in args.model_type:
        #path_to_model = os.path.join('/mnt/md2/kiyasseh/Scripts/dino-main/outputs/')
        path_to_model = './SAIS/scripts/dino-main/outputs'
        if args.model_type == 'ViT_SelfSupervised_ImageNet':
            """ Self Supervised on ImageNet Path """
            params = torch.load(os.path.join(path_to_model,'dino_deitsmall%i_pretrain.pth' % args.patch_size)) 
        elif args.model_type in ['ViT_SelfSupervised_SurgicalVideoNet','ViT_SelfSupervised_VUAVideoNet']:
            if args.model_type == 'ViT_SelfSupervised_SurgicalVideoNet':
                loaded_params = torch.load(os.path.join(path_to_model,'dino_deitsmall%i_pretrain_surgicalvideonet_epoch10.pth' % args.patch_size))
            elif args.model_type == 'ViT_SelfSupervised_VUAVideoNet':
                loaded_params = torch.load(os.path.join(path_to_model,'dino_deitsmall%i_pretrain_VUA_epoch7.pth' % args.patch_size),map_location=torch.device('cpu'))                
            loaded_params = OrderedDict(list(loaded_params['student'].items())[:-8]) # remove final MLP heads and so forth (make compatible with ImageNet arch)
            params = OrderedDict() # modify names of parameters to make them compatible with model later
            for name,param in loaded_params.items():
                new_name = '.'.join(name.split('.')[2:])
                params[new_name] = param

        student = vits.__dict__[args.arch](patch_size=args.patch_size,drop_path_rate=args.drop_path_rate)
        student.load_state_dict(params)
        model = student        
    elif 'Supervised' in args.model_type:
        """ Supervised on ImageNet Path """
        #path_to_model = os.path.join(savepath,dataset,'Results')
        #model = torch.load(os.path.join(path_to_model,args.model_type))
        model = timm.create_model('%s_patch%i_224_in21k' % (args.arch,args.patch_size), pretrained=True, num_classes=0)
    
    #if torch.cuda.device_count() > 1:
    #    print('Lets use',torch.cuda.device_count(),'GPUs!')
    #    model = nn.DataParallel(model)
    
    # if torch.cuda.is_available():
    #     model.to(rank)
    #     model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    # else:
    # model = DDP(model, device_ids=None, find_unused_parameters=True)
    return model

def obtain_flow(model,inputs):
    predictions = model(inputs)
    
    key1,key2 = list(predictions.keys()) #key1 = flows
    flow_im_list = []
    for prediction,aux,sample in zip(predictions['flows'],predictions[key2],inputs['images']):
        prediction, sample = prediction.unsqueeze(0), sample.unsqueeze(0)
        #print(prediction.shape,sample.shape)
        io_adapter = IOAdapter(model, sample.shape[-2:], cuda=torch.cuda.is_available()) #images[0].shape[:2])
        prediction = {'flows':prediction,key2:aux}
        # Remove extra padding that may have been added to the inputs
        prediction = io_adapter.unpad_and_unscale(prediction)

        # The output is a dict with possibly several keys,
        # but it should always store the optical flow prediction in a key called 'flows'.
        flows = prediction['flows']

        # flows will be a 5D tensor BNCHW.
        # This example should print a shape (1, 1, 2, H, W).
        #print(flows.shape)

        # Create an RGB representation of the flow to show it on the screen
        flow_rgb = flow_utils.flow_to_rgb(flows)
        # Make it a numpy array with HWC shape
        flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
        flow_rgb_npy = flow_rgb.detach().cpu().numpy()
        flow_im = Image.fromarray(np.uint8(flow_rgb_npy*255))
        
        flow_im_list.append(flow_im)
    #flow_im.save('/mnt/md2/kiyasseh/SurgicalDatasets/NS/sample_flow.jpg')
    #print('flow saved!')    
    return flow_im_list

def saveFlows(flow_ims,labels,nflows,dataset_list,args):
    dataset_name = dataset_list[0]
    for im,label,nflow in zip(flow_ims,labels,nflows):
        nflow = nflow.item()
        savename = 'flows' + '_' + '0'*(8 - len(str(nflow))) + str(nflow) + '.jpg'
        savepath = os.path.join(args.data_path,'flows',label) #label is flow folder (video id)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        im.save(os.path.join(savepath,savename))

def extractFlows(rank,world_size,dataset_list,pid,jump_size,args):
    #dist.init_process_group("nccl", rank=rank, world_size=world_size) 
    dataloader = prepareDataloader(rank,world_size,dataset_list,args,pid=pid,jump_size=jump_size)
    model = ptlflow.get_model('raft', pretrained_ckpt='things')
    for param in model.parameters(): 
        param.requires_grad_(False)
    #model.to(rank)
    #model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    """ Data Parallel Stuff """
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # print('Lets use',torch.cuda.device_count(),'GPUs!')
    # model = nn.DataParallel(model)
    # model.to(device)
    """ End """
    model.eval()
    reps_list = []
    labels_list = []
    for inputs,labels,nflows in tqdm(dataloader):
        with torch.set_grad_enabled(False):
            #inputs['images'] = inputs['images'].to(rank)
            inputs['images'] = inputs['images'].to(device)
            flow_im_list = obtain_flow(model,inputs)
            saveFlows(flow_im_list,labels,nflows,dataset_list,args)
    print('All Flows Saved!')

def obtainSegmentations(model,inputs):
    outputs = model(inputs)
    seg_im_list = []
    for output in outputs:
        probs = torch.sigmoid(output)
        preds = probs > 0.5
        preds = preds.type(torch.int)
        imPreds = Image.fromarray((255*preds).view(224,224).byte().cpu().detach().numpy()) #1024,1280
        #predsA = transforms.Resize((height,width))(imPreds) #resize back to original image size # no need b/c will be resized later anyway
        predsA = imPreds.convert('RGB')
        seg_im_list.append(predsA)
    return seg_im_list

def saveSegmentations(seg_ims,labels,nframes,dataset_list,args):
    dataset_name = dataset_list[0]
    for im,label,nframe in zip(seg_ims,labels,nframes):
        nframe = nframe.item()
        savename = 'segs' + '_' + '0'*(8 - len(str(nframe))) + str(nframe) + '.jpg'
        savepath = os.path.join(args.data_path,dataset_name,'Segmentations',label) #label is flow folder (video id)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        im.save(os.path.join(savepath,savename))

# def getSegmentationModel(rank):
#     mode = 'binary' #'binary' | 'instruments'
#     num_classes = 1 # 1 | 8
#     params = torch.load('/mnt/md2/kiyasseh/Scripts/segmentation/unet16_%s_20/unet16_%s_20/model_0.pt' % (mode,mode))
#     params_dict = OrderedDict()
#     for key in params['model'].keys():
#         name = '.'.join(key.split('.')[1:])
#         params_dict[name] = params['model'][key]
#     model = UNet16(num_classes=num_classes)
#     model.load_state_dict(params_dict)
#     print('Pretrained Parameters Loaded!')
#     model.to(rank)
#     model = DDP(model, device_ids=[rank], find_unused_parameters=True)
#     return model

def extractSegmentations(rank,world_size,dataset_list,args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size) 
    dataloader = prepareDataloader(rank,world_size,dataset_list,args)
    model = getSegmentationModel(rank)
    for param in model.parameters(): 
        param.requires_grad_(False)
    """ Data Parallel Stuff """
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('Lets use',torch.cuda.device_count(),'GPUs!')
    #model = nn.DataParallel(model)
    #model.to(device)
    """ End """
    model.eval()
    reps_list = []
    labels_list = []
    for inputs,labels,nframes in tqdm(dataloader):
        with torch.set_grad_enabled(False):
            inputs = inputs.to(rank)
            seg_im_list = obtainSegmentations(model,inputs)
            saveSegmentations(seg_im_list,labels,nframes,dataset_list,args)
    print('All Segmentations Saved!')


def extractFeatures(rank,world_size,dataset_list,args): 
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # if torch.cuda.is_available():
    #     backend = 'nccl'
    # else:
    #     backend = 'gloo'
    # dist.init_process_group(backend, rank=rank, world_size=world_size) 
    dataloader = prepareDataloader(rank,world_size,dataset_list,args)
    model = loadModel(rank,world_size,dataset_list[0],args)
    #model = model.to(device)
    model.eval()
    reps_list = []
    labels_list = []
    for inputs,labels,dataset in tqdm(dataloader):
        with torch.set_grad_enabled(False):
            # if torch.cuda.is_available():
            # inputs = inputs.to(rank)
            inputs = inputs.to(device)
            reps = model(inputs)
        reps_list.extend(reps.cpu().detach().numpy())
        labels_list.extend(labels)
        
    if args.save_type == 'dict':
        saveDict(dataset_list,args,reps_list,labels_list) #use this for public datasets = easier to work with
    elif args.save_type == 'h5':
        saveH5(dataset_list,args,reps_list,labels_list) #use this for NS and VUA due to large size of files
    print('Info Saved!')

def saveDict(dataset_list,args,reps_list,labels_list):
    dataset_name = dataset_list[0]
    path = os.path.join(args.data_path,dataset_name,'Results')
    filename = '%s_RepsAndFeatures' % args.model_type
    info = {'reps':reps_list,'labels':labels_list}
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(info,os.path.join(path,filename))
    
def saveH5(dataset_list,args,reps_list,labels_list):
    dataset_name = dataset_list[0]
    path = os.path.join(args.data_path,'results')
    if not os.path.exists(path):
        os.makedirs(path)
        
    if args.optical_flow_to_reps == True:
        suffix = '%s_FlowRepsAndLabels.h5' % args.model_type
    elif args.segmentation_to_reps == True:
        suffix = '%s_SegRepsAndLabels.h5' % args.model_type
    else:
        suffix = '%s_RepsAndLabels.h5' % args.model_type

    with h5py.File(os.path.join(path,suffix),'w') as hf:
        unique_labels = np.unique(labels_list)
        for label in unique_labels:
            indices = np.where(np.in1d(labels_list,label))[0]
            reps = np.vstack(list(itemgetter(*indices)(reps_list)))
            hf.create_dataset(label,data=reps)
        
           
def getArgParser():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--model_type', type=str, choices=['ViT_SelfSupervised_ImageNet','ViT_Supervised_ImageNet','ViT_SelfSupervised_SurgicalVideoNet','ViT_SelfSupervised_USC_NSVideoNet','ViT_SelfSupervised_VUAVideoNet'])
    parser.add_argument('--batch_size_per_gpu', type=int)
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--data_list',nargs='+',help='choose dataset e.g., VUA | NS | ...')
    parser.add_argument('--save_type',type=str,choices=['dict','h5'])
    parser.add_argument('--optical_flow',default=False,action='store_true',help='generate and save optical flow images')
    parser.add_argument('--segmentation',default=False,action='store_true',help='generate and save segmentation images')
    parser.add_argument('--optical_flow_to_reps',default=False,action='store_true',help='convert optical flow images to reps')
    parser.add_argument('--segmentation_to_reps',default=False,action='store_true',help='convert segmentation images to reps')
    parser.add_argument('--local_rank',type=int)
    return parser
    
fps_dict = {'HMV1': 30.0,
		 'HMV10': 59.94,
		 'HMV11': 59.94,
		 'HMV12': 59.94,
		 'HMV2': 30.0,
		 'HMV3': 30.0,
		 'HMV4': 60.0,
		 'HMV5': 60.0,
		 'HMV6': 60.0,
		 'HMV7': 60.0,
		 'HMV8': 59.94,
		 'HMV9': 59.94,
         'HMV13':59.94,
         'HMV14':59.94,
         'HMV15':59.94,
         'HMV16':59.94,
         'HMV17':59.94,
         'HMV18':59.94,
         'HMV19':59.94,
         'HMV20':59.94,
         'HMV21':59.94,
         'HMV22':59.94,
         'HMV23':29.97,
         'HMV24':29.97,
         'HMV25':29.97,
         'HMV26':29.97,
        }
    
if __name__ == '__main__':
    starttime = time.time()

    parser = getArgParser()
    args = parser.parse_args()
    #assert args.optical_flow != args.segmentation #one or the other 
    
    if args.optical_flow == True: #to generate optical flow images
        list_of_datasets = args.data_list #['NS'] #['Glenda_v1.0','LapGyn4_v1.2','Nephrec9','SurgicalActions160']
        for dataset_name in list_of_datasets:
            dataset_list = [dataset_name]
            assert dataset_name in ['Custom','NS','VUA','NS_Gronau','VUA_Gronau','RAPN','VUA_COH','VUA_HMH','VUA_Lab','JIGSAWS_Suturing','DVC_UCL']
            df = pd.read_csv(os.path.join(args.data_path,'paths','%s_FlowPaths.csv' % dataset_name))
            pids = df['label'].unique()
            #pids = ['HMV23','HMV24','HMV25','HMV26']#,'HMV17','HMV18','HMV19','HMV20','HMV21','HMV22']
            #pids = getNewNSVids() #from main_dino.py
            #fps_dict = getFPS(dataset_name) # only for VUA and VUA_HMH
            #pids = getNewVUAVids()
            #pids = sorted(list(set(pids) - set(getNewVUAVids()))) #from main_dino.py
            for pid in pids: #iterate over patient IDs
                if not os.path.exists(os.path.join(args.data_path,dataset_name,'Flows',pid)): #avoid recreating the optical flows
                    print(pid)
                    if dataset_name in ['VUA','VUA_HMH']:
                        jump_size = int(fps_dict[pid] // 2) 
                    elif dataset_name in ['Custom','NS','NS_Gronau','VUA_Gronau','RAPN','VUA_COH','JIGSAWS_Suturing']:
                        jump_size = 15
                    elif dataset_name in ['VUA_Lab','DVC_UCL']:
                        jump_size = 30
                    world_size = 1
                    extractFlows(args.local_rank,world_size,dataset_list,pid,jump_size,args)
                    #mp.spawn(extractFeatures,args=(world_size,dataset_list,pid,args),nprocs=world_size,join=True)
    elif args.segmentation == True: #to generate segmentation images 
        list_of_datasets = args.data_list #['NS'] #['Glenda_v1.0','LapGyn4_v1.2','Nephrec9','SurgicalActions160']
        for dataset_name in list_of_datasets:
            dataset_list = [dataset_name]
            world_size = 4 #num of GPUs
            mp.spawn(extractSegmentations,args=(world_size,dataset_list,args),nprocs=world_size,join=True)
    else:
        list_of_datasets = args.data_list #['NS'] #['Glenda_v1.0','LapGyn4_v1.2','Nephrec9','SurgicalActions160']
        for dataset_name in list_of_datasets:
            dataset_list = [dataset_name]
            world_size = 1
            # extractFeatures(args.local_rank,world_size,dataset_list,args)
            mp.spawn(extractFeatures,args=(world_size,dataset_list,args),nprocs=world_size,join=True)

    diff = time.time() - starttime
    print("Time taken (s): %.3f" % diff)