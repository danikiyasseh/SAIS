# -*- coding: utf-8 -*-
# %%
"""
Created on Tue Oct  5 11:37:15 2021

@author: DaniK
"""
import torch
import numpy as np
import torch.nn as nn
from tabulate import tabulate
from sklearn.metrics import roc_auc_score, precision_score, recall_score

def calcNCELoss(rank,snip_sequence,labels,videoname,gesture_prototypes,domains):
    """ Prepare Prototypes """
    p = torch.vstack(list(gesture_prototypes.values())) # nprototypes x D
    norm = torch.norm(p,dim=1).unsqueeze(1).repeat(1,p.shape[1])        
    p_norm = p / norm
    # if torch.cuda.is_available():
    #     p_norm = p_norm.to(rank)
    # else:
    p_norm = p_norm.to('cpu')
    p_labels = list(gesture_prototypes.keys()) # 0, 1, ... , nclasses,
    p_labels = np.repeat(np.expand_dims(np.array(p_labels),0),snip_sequence.shape[0],axis=0) # nbatch x nprototypes
    """ Prepare Video Representations """
    norm = torch.norm(snip_sequence,dim=1).unsqueeze(1).repeat(1,snip_sequence.shape[1]) # nbatch x D
    s_norm = snip_sequence / norm
    """ Calculate Similarities Between Prototypes and Representations """
    cols = []
    sim = torch.matmul(s_norm,p_norm.T) # nbatch x nprototypes
    sim_exp = torch.exp(sim)
    sides = list(map(lambda video:video.split('_')[-1],videoname))
    labels = list(map(lambda label:str(label.cpu().detach().numpy().item()),labels))
    s_labels = list(map(lambda tup:tup[1],zip(sides,labels))) #e.g. 0L, 2R, etc
    s_labels = np.repeat(np.expand_dims(np.array(s_labels),1),p.shape[0],axis=1) # nbatch x nprototypes

    cols = np.argmax(p_labels == s_labels,1)
    rows = list(range(len(cols)))
    nums = sim_exp[rows,cols] # nbatch
    #dens = torch.sum(sim_exp,1) # nbatch 
    #if len(np.unique(domains)) > 1: # multi-task setting (only consider subset of prototypes)
    #    dens = torch.stack([torch.sum(sim_exp[i,[0,1]]) if domain == 'NH_02' else torch.sum(sim_exp[i,[2,3]]) for i,domain in enumerate(domains)])
    #else:
    dens = torch.sum(sim_exp,1) # nbatch 
    loss = -torch.mean(torch.log(nums/dens)) # scalar
    return loss

def calcImportanceLoss(output_importances,importances,ipad,labels):
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    output_importances = output_importances[:,:,1:,0] # B x 1 x MAX_FRAMES (avoid cls_token importance)
    #print(output_importances.shape,importances.shape) # output_importances[:,1:] to avoid the cls_token importance
    loss = criterion(output_importances,importances) # B x MAX_FRAMES
    loss = torch.mean(loss)
    ipad = ~ipad # to invert elements from TRUE to FALSE and vice versa
    ipad = ipad[:,:,:-1] # not sure why mask has an additional entry (i.e., MAX_FRAMES + 1)
    loss = loss * ipad # apply mask to loss elements to ONLY consider the frames of interest (not those padded or cls token)
    low_skill_indices = np.where(labels.cpu() == torch.tensor(0,dtype=torch.long))[0]
    loss = loss[low_skill_indices,:] # only consider loss incurred on low-skill video segments
    loss = torch.mean(loss)
    return loss

def calcLoss(output_logits,labels,nclasses,future_reps,snip_reps,include_ss_loss=False):
        """ Calculate Loss for MIL 
        
        Args:
                output_logits (torch.Tensor): dim = B x nclasses
        """
        #if nclasses == 2:
#       criterion = nn.BCEWithLogitsLoss()
#       labels = labels.type(torch.float)
#       labels = labels.unsqueeze(1)
        #elif nclasses > 2:
        #print(output_logits,labels)
        #print(output_logits.shape,labels.shape)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output_logits,labels)
        #print(loss)
        if include_ss_loss == True:
                ss_loss = calcSSLoss(future_reps,snip_reps)
                loss += ss_loss
                loss = loss / 2
        
        return loss

def calcSSLoss(future_reps,snip_reps):
        """ Calculate Self-supervised Loss = predict next representation
        Args:
                snip_sequence (torch.Tensor): before transformer
                snip_reps (torch.Tensor): after transformer
        """
        pred_reps = snip_reps[:,:-1,:] # B x nsnippets-1 x D
        future_reps = future_reps[:,1:,:] # B x nsnippets-1 x D
        criterion = nn.MSELoss()
        loss = criterion(pred_reps,future_reps)
        return loss

def calcNCEMetrics(rank,snip_sequence_list,labels_list,videoname_list,gesture_prototypes):
    labels = torch.stack(labels_list)
    videoname = videoname_list

    """ Prtotype-Specific Stuff """
    p = torch.vstack(list(gesture_prototypes.values())) # nprototypes x D
    norm = torch.norm(p,dim=1).unsqueeze(1).repeat(1,p.shape[1])
    p_norm = p / norm
    # if torch.cuda.is_available():
    #     p_norm = p_norm.to(rank)
    # else:
    p_norm = p_norm.to('cpu')
    p_labels = list(gesture_prototypes.keys())

    def getProbs(snip_sequence,labels,videoname,p_norm,p_labels):

        """ Reps-Specific Stuff """
        norm = torch.norm(snip_sequence,dim=1).unsqueeze(1).repeat(1,snip_sequence.shape[1]) # nbatch x D
        s_norm = snip_sequence / norm

        sim = torch.matmul(s_norm,p_norm.T) # nbatch x nprototypes
        sim_exp = torch.exp(sim)
        sides = list(map(lambda video:video.split('_')[-1],videoname))
        labels = list(map(lambda label:str(label.cpu().detach().numpy().item()),labels))
        s_labels = list(map(lambda tup:tup[1],zip(sides,labels))) #e.g. 0L, 2R, etc
        s_labels = np.repeat(np.expand_dims(np.array(s_labels),1),p.shape[0],axis=1) # nbatch x nprototypes
        #print(p_labels,s_labels)
        labels = torch.tensor(np.argmax(p_labels == s_labels,1))
        probs = sim_exp / torch.sum(sim_exp,1).unsqueeze(1).repeat(1,sim_exp.shape[1]) # nbatch x nprototype
        return probs, labels
    
    if isinstance(snip_sequence_list,tuple):
        nitems = torch.stack(snip_sequence_list[0]).shape[0]
        p_labels = np.repeat(np.expand_dims(np.array(p_labels),0),nitems,axis=0) # nbatch x nprototypes
        probs = torch.zeros(nitems,p_norm.shape[0])
        for snip_sequence_list_el in snip_sequence_list:
            snip_sequence = torch.stack(snip_sequence_list_el) 
            curr_probs, curr_labels = getProbs(snip_sequence,labels,videoname,p_norm,p_labels)
            probs += curr_probs.cpu().detach()
        probs = probs/len(snip_sequence_list)
        labels = curr_labels
    else:
        snip_sequence = torch.stack(snip_sequence_list)
        p_labels = np.repeat(np.expand_dims(np.array(p_labels),0),snip_sequence.shape[0],axis=0) # nbatch x nprototypes
        probs, labels = getProbs(snip_sequence,labels,videoname,p_norm,p_labels)

    preds = torch.argmax(probs,1) 
    preds = preds.cpu().detach()
    acc = (torch.sum(preds == labels) / preds.shape[0]).item()

    labels,probs = labels.cpu().detach().numpy(), probs.cpu().detach().numpy()
    preds = preds.numpy()
    #print(labels,preds)
    prec = precision_score(labels,preds,average='macro')
    rec = recall_score(labels,preds,average='macro')
    nclasses = len(gesture_prototypes)
    if nclasses == 2:
        probs = probs[:,-1]

    try:
        auc = roc_auc_score(labels,probs,multi_class='ovr')
    except:
        auc = np.nan

    return acc, auc, prec, rec


def calcMetrics(output_logits_list,labels_list,nclasses):
        """ Calculate Accuracy for MIL 
        
        Args:
                output_logits (torch.Tensor): dim = B x nclasses
                labels (torch.Tensor): dim = B x 1
        """     
        if isinstance(output_logits_list,tuple):
            output_logits_list = torch.stack([torch.vstack(output_logits_el) for output_logits_el in output_logits_list]) # 
            output_logits = torch.mean(output_logits_list,0) # average logits across the TTA augments
        else:
            output_logits = torch.vstack(output_logits_list)
        
        labels = torch.vstack(labels_list).squeeze(1)
        if nclasses == 1:
            output_probs = torch.sigmoid(output_logits)
            preds = (output_probs > 0.5).to(torch.long).squeeze()
        else:
            output_probs = torch.softmax(output_logits,1)
            preds = torch.argmax(output_probs,1)
        #print(preds,labels)
        acc = (torch.sum(preds == labels) / preds.shape[0]).item()
        
        preds = preds.cpu().detach().numpy()
        labels, output_probs = labels.cpu().detach().numpy(),output_probs.cpu().detach().numpy()
        prec = precision_score(labels,preds,average='macro')
        rec = recall_score(labels,preds,average='macro')
        #if nclasses == 2:
        #    output_probs = output_probs[:,-1]
            
        auc = roc_auc_score(labels,output_probs,multi_class='ovr')
        return acc, auc, prec, rec

def printMetrics(phase,metrics):
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        metric_names = list(map(lambda name: phase + '_' + name,metric_names))
        metric_values = list(map(lambda value: '%.3f' % value,metric_values))
        print(tabulate([metric_names,metric_values],headers='firstrow'))

def trackMetrics(metrics,metrics_dict):
        for name,value in metrics.items():
                metrics_dict[name].append(value)
        return metrics_dict

# %%

def calcTemporalCoherenceLoss(output_logits,output_logits_flipped):
        """ Calculate the Temporal Coherence Loss
        
        Args:
                output_logits: logits for video snippet which is NOT flipped
                output_logits_flipped: logits for video snippet which IS flipped
        """
#       criterion = nn.BCEWithLogitsLoss()
#       lossA = criterion(output_logits,torch.zeros(output_logits.shape[0],1,dtype=torch.float))
#       lossB = criterion(output_logits_flipped,torch.ones(output_logits.shape[0],1,dtype=torch.float))

        criterion = nn.CrossEntropyLoss()       
        lossA = criterion(output_logits,torch.zeros(output_logits.shape[0],dtype=torch.long))
        lossB = criterion(output_logits_flipped,torch.ones(output_logits.shape[0],dtype=torch.long))
        
        loss = (lossA + lossB) / 2
        
        return loss

def calcTemporalCoherenceAcc(output_logits_list,output_logits_flipped_list):
        output_logits = torch.vstack(output_logits_list)
        output_logits_flipped = torch.vstack(output_logits_flipped_list)

        predsA = torch.argmax(output_logits,1)
        labelsA = torch.zeros(output_logits.shape[0])
#       predsA = (torch.sigmoid(output_logits) > 0.5).type(torch.int)
#       labelsA = torch.zeros(output_logits.shape[0],1)

        predsB = torch.argmax(output_logits_flipped,1)
        labelsB = torch.ones(output_logits_flipped.shape[0])
#       predsB = (torch.sigmoid(output_logits_flipped) > 0.5).type(torch.int)
#       labelsB = torch.ones(output_logits_flipped.shape[0],1)
        
        countA = torch.sum(predsA == labelsA)
        countB = torch.sum(predsB == labelsB)
        acc = (countA + countB ) / (predsA.shape[0] + predsB.shape[0]) 
        return acc


