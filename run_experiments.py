# -*- coding: utf-8 -*-
# %%
"""
Created on Tue Oct  5 12:01:13 2021

@author: DaniK
"""

import os
import argparse
from train import trainModel
import torch.multiprocessing as mp

def runExperiment(rank,world_size,root_path,savepath,dataset_name,data_type,batch_size,nclasses,domain,phases,lr,modalities,freeze_encoder_params,inference,task,balance,balance_groups,single_group,group_info,self_attention,importance_loss,encoder_type,encoder_params,snippetLength,frameSkip,overlap,rep_dim,nepochs,fold,training_fraction):
        trainModel(rank,world_size,root_path,savepath,dataset_name,data_type,batch_size,nclasses,domain,phases,lr,modalities,freeze_encoder_params,inference,task,balance,balance_groups,single_group,group_info,self_attention,importance_loss,encoder_type,encoder_params,snippetLength,frameSkip,overlap,rep_dim,nepochs,fold,training_fraction)

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-p','--path',type=str)
parser.add_argument('-data','--dataset_name',type=str,choices=['VUA_EASE','VUA_EASE_Stitch','NS_DART','NS_Gestures_Classification','VUA_Gestures_Classification','DVC_UCL_Gestures_Classification','JIGSAWS_Suturing_Gestures_Classification','NS_vs_VUA'])
parser.add_argument('-d','--domain_name',type=str,help='choose domain e.g., VUA')
parser.add_argument('-m','--model',type=str,default='R3D',help='choose network architecture')
parser.add_argument('-enc','--encoder_params',type=str,default='ViT_SelfSupervised_ImageNet',help='choose encoder params',choices=['ViT_SelfSupervised_ImageNet','ViT_SelfSupervised_SurgicalVideoNet','ViT_SelfSupervised_USC_NSVideoNet','Kinetics','ViT_SelfSupervised_VUAVideoNet'])
parser.add_argument('-dim','--rep_dim',type=int,default=512,help='choose dimension of representations')
parser.add_argument('-mod','--modalities',type=str,help='choose type of modalities')
parser.add_argument('-bs','--batch_size',type=int,default=1,help='choose batch size')
parser.add_argument('-lr','--learning_rate',type=float,help='choose learning rate')
parser.add_argument('-tf','--training_fraction',type=float,default=1,help='choose training fraction [0-1]')
parser.add_argument('-fe','--freeze_encoder',default=False,action='store_true')
parser.add_argument('-t','--task',type=str,help='choose task to achieve')
parser.add_argument('-nc','--nclasses',type=int,help='choose number of classes')
parser.add_argument('-bc','--balance_classes',default=False,action='store_true',help='balance classes')
parser.add_argument('-bg','--balance_groups',default=False,action='store_true',help='balance groups')
parser.add_argument('-sg','--single_group',default=False,action='store_true',help='train on single group')
parser.add_argument('-sa','--self_attention',default=False,action='store_true',help='include self attention (Transformer Encoder)')
parser.add_argument('-il','--importance_loss',default=False,action='store_true',help='include importance loss per frame')
parser.add_argument('-domains','--domains',nargs='+',type=str,help='choose domains in each task e.g., r_vs_c, r_vs_h, etc.')
parser.add_argument('-ph','--phases',nargs='+',help='choose phases (e.g., train, val, test)')
parser.add_argument('-dt','--data_type',type=str,help='choose type of data (raw vs. reps)')
parser.add_argument('-e','--nepochs',type=int,help='choose number of epochs')
parser.add_argument('-f','--nfolds',type=int,help='choose number of folds')
#parser.add_argument('-folds','--folds',nargs='+',type=int,help='choose number of folds')
parser.add_argument('-i','--inference',default=False,action='store_true')
parser.add_argument('--local_rank',type=int)
args = parser.parse_args()

# %%
root_path = args.path #'/mnt/md2/kiyasseh/SurgicalDatasets'
dataset_name = args.dataset_name #_Recommendation' #dataset to train on - options: 'NS' | 'SOCAL' | 'NS_Gestures_Classification' | 'NS_Gestures' | 'NS_Gestures_Recommendation' | 'VUA_EASE_Stitch' | 'VUA_EASE'
data_type = args.data_type #'raw' #options: raw (images) | reps (extracted features)
modalities = args.modalities # RGB | Flow | RGB-Flow
batch_size = args.batch_size #1
nclasses = args.nclasses #3 # 10 for top gestures in NS Gestures dataset_name
domains = args.domains  # r_vs_c | Top9          #label definition - options: Gesture | tr | th | gs | ESI_12M for NS, 'success' for SOCAL
phases = args.phases #['train','val']
#print(phases)
lr = args.learning_rate
freeze_encoder_params = args.freeze_encoder #options: True (default) | False (i.e., update params)
inference = args.inference #False #options: False (i.e., do not perform inference) | True (perform inference on test set, for example) - loads saved weights
task = args.task #'MIL' #options: AoT (self-supervision) | 'MIL' (multiple instance learning) | 'FeatureExtraction' (extract snippet features)
balance = args.balance_classes
balance_groups = args.balance_groups
single_group = args.single_group
self_attention = args.self_attention
importance_loss = args.importance_loss
encoder_type = args.model #'R3D' #options: R3D | ViT 
encoder_params = args.encoder_params #options: ViT_SelfSupervised_ImageNet
overlap = 0 #fraction overlap across snippets [0,1] #0 means no overlap
snippetLength = 5 #number of frames in the video snippet # 30 for NS # 5 for SOCAL
frameSkip = 1 #number of frames to skip (i.e., downsampling factor) # 30 for NS # 1 for SOCAL
rep_dim = args.rep_dim #512 #dimension of intermediate representation # 512 for R3D | 768 for ViT
nepochs = args.nepochs 
nfolds = args.nfolds #1
#folds = args.folds
training_fraction = args.training_fraction #1
print('Modalities: %s' % modalities)
print('Self Attention: %s' % str(self_attention))
print('Balance Groups: %s' % str(balance_groups))
print('Predict Importance: %s' % str(importance_loss))

if __name__ == '__main__':
    """ Iterate Over Distinct Experiments """
    for domain in domains:
        print('Domain: %s' % domain)
        print('Balance Classes: %s' % str(balance))
        for fold in range(nfolds):
            print('Fold: %i' % fold)
            #if torch.cuda.is_available():
            if balance_groups == True: # path for bias mitigation with group balancing strategy
                group_info = 'None'
                if importance_loss == True:
                    savepath = os.path.join(args.path,args.domain_name,'Results',task,domain,encoder_params,'Balance_%s' % str(balance),'BalanceGroups_%s' % str(balance_groups),'PredictImportance_%s' % str(importance_loss),'SelfAttention_%s' % str(self_attention),modalities,'Fold_%i' % fold) #'/mnt/md2/kiyasseh/SurgicalDatasets/NS/Results' #Neurosurgery/SOCAL/Results' #'C:/Users/DaniK/OneDrive/Desktop'
                else:
                    savepath = os.path.join(args.path,args.domain_name,'Results',task,domain,encoder_params,'Balance_%s' % str(balance),'BalanceGroups_%s' % str(balance_groups),'SelfAttention_%s' % str(self_attention),modalities,'Fold_%i' % fold) #'/mnt/md2/kiyasseh/SurgicalDatasets/NS/Results' #Neurosurgery/SOCAL/Results' #'C:/Users/DaniK/OneDrive/Desktop'
            elif single_group == True: # path for bias mitigation with group-specific classifiers
                group_info = {'group_name':'Prostate Volume Group','group_val':'ProstateLarge60ml'} #'ProstateLarge60ml'
                group_val = group_info['group_val']
                savepath = os.path.join(args.path,args.domain_name,'Results',task,domain,encoder_params,'Balance_%s' % str(balance),'SingleGroup_%s' % str(single_group),group_val,'SelfAttention_%s' % str(self_attention),modalities,'Fold_%i' % fold) #'/mnt/md2/kiyasseh/SurgicalDatasets/NS/Results' #Neurosurgery/SOCAL/Results' #'C:/Users/DaniK/OneDrive/Desktop'
            elif importance_loss == True:
                group_info = 'None'
                savepath = os.path.join(args.path,args.domain_name,'Results',task,domain,encoder_params,'Balance_%s' % str(balance),'PredictImportance_%s' % str(importance_loss),'SelfAttention_%s' % str(self_attention),modalities,'Fold_%i' % fold) #'/mnt/md2/kiyasseh/SurgicalDatasets/NS/Results' #Neurosurgery/SOCAL/Results' #'C:/Users/DaniK/OneDrive/Desktop'                
            else: # traditional path
                group_info = 'None'
                savepath = os.path.join(args.path,args.domain_name,'Results',task,domain,encoder_params,'Balance_%s' % str(balance),'SelfAttention_%s' % str(self_attention),modalities,'Fold_%i' % fold) #'/mnt/md2/kiyasseh/SurgicalDatasets/NS/Results' #Neurosurgery/SOCAL/Results' #'C:/Users/DaniK/OneDrive/Desktop'
            print('***** \n Savepath: %s \n *****' % savepath)
            world_size = 1 # more leads to hanging of process after experiment completion (might need to do a cleanup of processes)
            mp.spawn(runExperiment,
                    args=(world_size,root_path,savepath,dataset_name,data_type,batch_size,nclasses,domain,phases,lr,modalities,freeze_encoder_params,inference,task,balance,balance_groups,single_group,group_info,self_attention,importance_loss,encoder_type,encoder_params,snippetLength,frameSkip,overlap,rep_dim,nepochs,fold,training_fraction),
                    nprocs=world_size,
                    join=True)
            #else:
            #    runExperiment(root_path,savepath,dataset_name,data_type,batch_size,nclasses,domain,phases,lr,modalities,freeze_encoder_params,inference,task,balance,self_attention,encoder_type,snippetLength,frameSkip,overlap,rep_dim,nepochs,fold,training_fraction)
