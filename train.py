# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 08:22:39 2021

@author: DaniK
"""
from prepare_dataset import loadDataloader
from prepare_model import loadModel
from perform_training import single_epoch, single_epoch_feature_extraction
from prepare_miscellaneous import printMetrics, trackMetrics
from collections import defaultdict
import torch.distributed as dist
import torch.nn as nn
import copy
import torch
import os
#from tensorboardX import SummaryWriter
#writer = SummaryWriter()

# def set_bn_training(m):
#       if isinstance(m, nn.BatchNorm2d):
#               m.eval()

# def set_bn_training(m):
#       for i,child in enumerate(m.children()):
#               if i == 0:
#                       child.eval()
#               
# def stop_tracking(model):
#       for ii in list(model.children())[0]:
#               if isinstance(ii,nn.BatchNorm3d):
#                       ii.track_running_stats = False
#       return model

def trainModel(rank,world_size,root_path,savepath,dataset_name,data_type,batch_size,nclasses,domain,phases,lr,modalities,freeze_encoder_params,inference,task,balance,balance_groups,single_group,group_info,self_attention,importance_loss,encoder_type,encoder_params,snippetLength,frameSkip,overlap,rep_dim,nepochs,fold,training_fraction):
        if torch.cuda.is_available():
            backend = 'nccl'
        else:
            backend = 'gloo'
        dist.init_process_group(backend, rank=rank, world_size=world_size) #"nccl"
        
        """ Prototypes Stuff """
        best_params_dict = dict()
        reps_and_labels_dict = dict()
        best_prototypes_dict = dict()
        metrics_dict = defaultdict(list)
        attention_dict = dict()
        
        snippets_dict = dict()
        videonames_dict = dict()
        labels_dict = dict()

        """ Load Model """
        print('Loading Model...')
        model,optimizer,device = loadModel(rank,world_size,savepath,data_type,nclasses,domain,rep_dim,encoder_type,task,fold,lr=lr,modalities=modalities,freeze_encoder_params=freeze_encoder_params,self_attention=self_attention,importance_loss=importance_loss,inference=inference)

        """ Load Data """
        print('Loading Data...')
        dataloaderClass = loadDataloader(root_path,dataset_name,data_type,batch_size,nclasses,domain,phases,task,balance,balance_groups,single_group,group_info,importance_loss,encoder_type,encoder_params,snippetLength,frameSkip,overlap,fold,training_fraction)
        dataloader = dataloaderClass.load()

        """ Perform Training """
        min_loss = float('inf')
        epoch_count = 1
        max_patience = 5
        patience_count = 1
        while epoch_count <= nepochs and patience_count <= max_patience:
                print('\n **** Epoch %i ****' % epoch_count)

                for phase in phases:
                        print(phase)
                        if phase == 'train':
                                if task == 'FeatureExtraction' or inference == True:
                                        model['model'].eval() #switch to .eval() when self.task = FeatureExtraction
                                else:
                                        model['model'].train()
                        elif phase in ['val','test']:
                                model['model'].eval()
                        elif 'inference' in phase:
                                model['model'].eval()

                        if task == 'FeatureExtraction':
                                snippets, videonames, labels = single_epoch_feature_extraction(dataloader,model,optimizer,device,phase,nclasses)
                                snippets_dict[phase] = snippets
                                videonames_dict[phase] = videonames
                                labels_dict[phase] = labels
                        else:
                                metrics, snippets, labels, videonames, attention, importance, logits = single_epoch(rank,world_size,dataloader,model,optimizer,device,phase,nclasses,task,importance_loss)
                                printMetrics(phase,metrics)
                                #writer.add_scalar('loss/%s' % phase,metrics['loss'],epoch_count)


                        if task != 'FeatureExtraction':
                            if inference == False:
                                if phase == 'val':
                                        loss = metrics['loss']
                                        metrics_dict = trackMetrics(metrics,metrics_dict)
                                        #torch.save(metrics_dict,os.path.join(savepath,'metrics_%s_nfold%i' % (task,fold)))
                                        if loss < min_loss:
                                                min_loss = loss
                                                patience_count = 1
                                                best_params_dict = copy.deepcopy(model['model'].state_dict())
                                                reps_and_labels_dict = {'reps':snippets,'labels':labels,'videonames':videonames,'logits':logits}
                                                best_prototypes_dict = copy.deepcopy(model['prototypes'])
                                        else:
                                                patience_count += 1
                            elif inference == True:
                                reps_and_labels_dict = {'reps':snippets,'labels':labels,'videonames':videonames,'logits':logits}
                                attention_dict = attention
                                importance_dict = importance
                                #if phase in ['val','test']:
                                #    attention_dict[fold] = attention
                epoch_count += 1

        if rank == 0: # save once to avoid weird multi-processing stuff
            """ Save Results For Current Fold """ 
            if task == 'FeatureExtraction':
                all_info_dict = {'snippets':snippets_dict,'videonames':videonames_dict,'labels':labels_dict}
                torch.save(all_info_dict,os.path.join(savepath,'all_info_dict_%s' % encoder_type))
                print('All Info Saved!')
            else:
                if inference == False:
                    if not os.path.exists(savepath):
                        os.makedirs(savepath) 
                    torch.save(best_params_dict,os.path.join(savepath,'params'))
                    torch.save(metrics_dict,os.path.join(savepath,'metrics'))
                    torch.save(best_prototypes_dict,os.path.join(savepath,'prototypes'))
                    torch.save(reps_and_labels_dict,os.path.join(savepath,'reps_and_labels'))
                    print('All Info Saved!')
                elif inference == True:
                    if task == 'MIL':
                        torch.save(attention_dict,os.path.join(savepath,'attention_%s' % (task)))
                    elif task == 'Prototypes':
                        torch.save(reps_and_labels_dict,os.path.join(savepath,'reps_and_labels_%s' % phases[0]))
                        torch.save(attention_dict,os.path.join(savepath,'attention_%s' % phases[0]))
                        torch.save(importance_dict,os.path.join(savepath,'importance_%s' % phases[0]))
                    elif task == 'ClassificationHead':
                        torch.save(reps_and_labels_dict,os.path.join(savepath,'reps_and_labels_%s' % phases[0]))
                        
        """ Added New - February 12th - Might Prevent Hanging """
        dist.destroy_process_group()
