# -*- coding: utf-8 -*-
# %%
"""
Created on Tue Oct  5 11:28:26 2021

@author: DaniK
"""
import torch.nn as nn
import torch
from prepare_miscellaneous import calcLoss, calcTemporalCoherenceLoss, \
                                    calcTemporalCoherenceAcc, calcMetrics, \
                                        calcNCELoss, calcNCEMetrics, calcImportanceLoss
from tqdm import tqdm

def single_epoch_feature_extraction(dataloader,model,optimizer,device,phase,nclasses):
        #snippets_dict = dict()
        snippets_list = []
        videoname_list = []
        labels_list = []
        batch = 1
        for videoname, snippets, labels in tqdm(dataloader[phase]):
                
                snippets = snippets.to(device)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(False):
                        snip_sequence = model.extractFeatures(snippets)

                #snippets_dict[videoname] = snip_sequence
                
                if snippets.shape[0] == 1:
                    snippets_list.append(snip_sequence)
                    videoname_list.append(videoname)
                    labels_list.append(labels)
                else:
                    snippets_list.extend(snip_sequence)
                    videoname_list.extend(videoname)
                    labels_list.extend(labels)

                batch += 1
                
                #if batch == 3:
                #    break

        return snippets_list, videoname_list, labels_list #snippets_dict

# %%

def single_epoch(rank,world_size,dataloader,model_dict,optimizer,device,phase,nclasses,task,importance_loss):
        model = model_dict['model']
        snip_sequence_list = []
        snip_sequence2_list = []
        snip_sequence3_list = []
        
        output_logits_list = []
        output_logits2_list = []
        output_logits3_list = []
        
        attention_list = []
        importance_list = []
        labels_list = []
        videoname_list = []

        snippets_dict = dict()
        output_logits = []
        labels_list = []
        attention_dict = dict()
        ave_loss = 0
        running_loss = 0
        batch = 1
        for videoname, snippets, flows, importances, labels, xlens, flens, xpad, fpad, ipad, domains in tqdm(dataloader[phase]):
                #print([s.shape for s in snippets])
                # if torch.cuda.is_available():
                #     """ GPU """
                #     if isinstance(snippets,dict):
                #         snippets = [snippet.to(rank) for snippet in snippets.values()]
                #         xpad = [xpad_el.to(rank) for xpad_el in xpad.values()]
                #         xlens = [xlens_el for xlens_el in xlens.values()]

                #         flows = [flow.to(rank) for flow in flows.values()]
                #         fpad = [fpad_el.to(rank) for fpad_el in fpad.values()]
                #         flens = [flens_el for flens_el in flens.values()]
                        
                #         #importances = [importance.to(rank) for importance in importances.values()]
                #         #ipad = [ipad_el.to(rank) for ipad_el in ipad.values()]
                #     else:
                #         snippets = snippets.to(rank)
                #         xpad = xpad.to(rank)
                #         fpad = fpad.to(rank)
                #         ipad = ipad.to(rank)
                #         flows = flows.to(rank)
                #         importances = importances.to(rank)
                #     labels = labels.to(rank)
                # else:
                """ CPU """
                if isinstance(snippets,dict):
                    snippets = [snippet.to(device) for snippet in snippets.values()]
                    xpad = [xpad_el.to(device) for xpad_el in xpad.values()]
                    xlens = [xlens_el for xlens_el in xlens.values()]

                    flows = [flow.to(device) for flow in flows.values()]
                    fpad = [fpad_el.to(device) for fpad_el in fpad.values()]
                    flens = [flens_el for flens_el in flens.values()]
                else:
                    snippets = snippets.to(device)
                    xpad = xpad.to(device)
                    fpad = fpad.to(device)
                    flows = flows.to(device)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(phase=='train'):
                        if task == 'MIL':
                            snip_sequence, snip_reps, output_logits, attention = model(snippets,flows,xlens,flens,task,xpad,fpad)
                            loss = calcLoss(output_logits,labels,nclasses,snip_sequence,snip_reps)
                        elif task == 'Prototypes':
                            if importance_loss == True:
                                output_importances, snip_sequence, snip_attn = model(snippets,flows,xlens,flens,task,xpad,fpad,domains)
                            else:
                                snip_sequence, snip_attn = model(snippets,flows,xlens,flens,task,xpad,fpad,domains)
                                
                            if 'inference' in phase:
                                loss = torch.tensor(0)
                            else:
                                if isinstance(snip_sequence,list):
                                    loss = torch.mean(torch.tensor([calcNCELoss(rank,snip_sequence_el,labels,videoname,model_dict['prototypes'],domains) for snip_sequence_el in snip_sequence]))
                                else:
                                    loss = calcNCELoss(rank,snip_sequence,labels,videoname,model_dict['prototypes'],domains)
                                    if phase == 'train':
                                        if importance_loss == True:
                                            iloss = calcImportanceLoss(output_importances,importances,ipad,labels)
                                            loss = loss + iloss
                                            #print(loss,iloss)
                        elif task in 'ClassificationHead':
                            snip_sequence, output_logits = model(snippets,flows,xlens,flens,task,xpad,fpad,domains)
                            if 'inference' in phase:
                                loss = torch.tensor(0)
                            else:
                                if isinstance(snip_sequence,list):
                                    if nclasses == 1:
                                        criterion = nn.BCEWithLogitsLoss()
                                        labels = labels.to(torch.float)
                                        loss = torch.mean(torch.tensor([criterion(output_logits_el.view_as(labels),labels) for output_logits_el in output_logits]))
                                    else:
                                        criterion = nn.CrossEntropyLoss()
                                        loss = torch.mean(torch.tensor([criterion(output_logits_el,labels) for output_logits_el in output_logits]))
                                else:
                                    if nclasses == 1:
                                        criterion = nn.BCEWithLogitsLoss()
                                        labels = labels.to(torch.float)
                                        loss = criterion(output_logits.view_as(labels),labels)
                                    else:
                                        criterion = nn.CrossEntropyLoss()
                                        loss = criterion(output_logits,labels)
                                
                if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                #print(model_dict['prototypes']['0'])
                if task == 'MIL':
                    """ MIL-related Stuff """
                    attention_dict[videoname[0]] = attention #works for single item batch
                    output_logits_list.append(output_logits)
                    labels_list.append(labels.unsqueeze(1))
                elif task == 'Prototypes':
                    """ Prototype-relate Stuff """
                    if isinstance(snip_sequence,list):
                        snip_sequence_list.extend(snip_sequence[0])
                        snip_sequence2_list.extend(snip_sequence[1])
                        snip_sequence3_list.extend(snip_sequence[2])
                    else:
                        snip_sequence_list.extend(snip_sequence)
                    attention_list.append(snip_attn)
                    labels_list.extend(labels)
                    videoname_list.extend(videoname)
                    if importance_loss == True:
                        #print(output_importances.shape,len(xlens))
                        if isinstance(snip_sequence,list):
                            #output_importances = output_importances[0] # first TTA element
                            xlens = xlens[0] # first TTA element 
                            output_importances = [importance[:,1:xlen+1,:].squeeze() for importance,xlen in zip(output_importances,xlens)]
                        else:
                            output_importances = [importance[:,1:xlen+1,:].squeeze() for importance,xlen in zip(output_importances,xlens)]
                        importance_list.append(output_importances)
                elif task == 'ClassificationHead':
                    if isinstance(snip_sequence,list):
                        output_logits_list.extend(output_logits[0])
                        output_logits2_list.extend(output_logits[1])
                        output_logits3_list.extend(output_logits[2])
                    else:
                        output_logits_list.extend(output_logits)
                    labels_list.extend(labels)
                        
                #snippets_dict[videoname] = snip_sequence
                if isinstance(snip_sequence,list):
                    curr_loss = loss.item() * snippets[0].shape[0]    
                else:
                    curr_loss = loss.item() * snippets.shape[0]
                running_loss += curr_loss
                batch += 1
                
                #if batch == 3:
                #    break
        
        ave_loss = running_loss / (len(dataloader[phase].dataset))
        if task == 'MIL':
            acc, auc, prec, rec = calcMetrics(output_logits_list,labels_list,nclasses)
        elif task == 'Prototypes':
            if phase == 'inference':
                acc, auc, prec, rec = 0, 0, 0, 0
            else:
                if isinstance(snip_sequence,list):
                    snip_sequence_list = (snip_sequence_list,snip_sequence2_list,snip_sequence3_list)
                acc, auc, prec, rec = calcNCEMetrics(rank,snip_sequence_list,labels_list,videoname_list,model_dict['prototypes'])
        elif task == 'ClassificationHead':
            if phase in ['inference','USC_inference']:
                if isinstance(snip_sequence,list):
                    output_logits_list = (output_logits_list,output_logits2_list,output_logits3_list)
                acc, auc, prec, rec = 0, 0, 0, 0
            else:
                if isinstance(snip_sequence,list):
                    output_logits_list = (output_logits_list,output_logits2_list,output_logits3_list)
                acc, auc, prec, rec = calcMetrics(output_logits_list,labels_list,nclasses)
        
        metrics = {'loss':ave_loss,'acc':acc,'auc':auc,'precision':prec,'recall':rec}
        return metrics, snip_sequence_list, labels_list, videoname_list, attention_list, importance_list, output_logits_list #snippets_dict, attention_dict


