"""
Created on Tue Oct  5 09:47:23 2021

@author: DaniK
"""
import os
import timm
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
# from pytorch_i3d import InceptionI3d
#from R3D import generate_model

class fullModel(nn.Module):
        
        def __init__(self,data_type='raw',nclasses=2,domain='NH_02',rep_dim=512,encoder_type='R3D',modalities='RGB-Flow',encoder_depth=18,load_pretrained_params=True,freeze_encoder_params=True,self_attention=True,importance_loss=False):
                super(fullModel,self).__init__()

#               encoder = generate_model(encoder_depth,n_classes=700)
#               if load_pretrained_params == True:
#                       encoder = self.loadPretrainedParams(encoder)
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                self.device = device
                if encoder_type == 'I3D':
                        pass
                        # encoder = InceptionI3d(num_classes=400, in_channels=3)
                        # encoder.load_state_dict(torch.load('rgb_imagenet.pt'))
                        #encoder.replace_logits(384) # only activate for ND_02
                        #encoder = nn.Sequential(*list(encoder.children())[:-1])
                        # print('I3D Loaded...')
                elif encoder_type == 'R3D':
                        encoder = torchvision.models.video.r3d_18(pretrained=load_pretrained_params)
                        encoder = nn.Sequential(*list(encoder.children())[:-1]) # to return 512 representation
                        print('R3D Loaded...')
                elif encoder_type == 'ViT':
                        encoder = timm.create_model('vit_base_patch16_224_in21k', pretrained=load_pretrained_params, num_classes=0) #num_classes=0 auto turns it into feature extractor
                        print('ViT Loaded...')
                
                if freeze_encoder_params == True:
                        self.freezeParams(encoder)
                        print('Encoder Params Frozen...')
                
                self.linear = nn.Linear(rep_dim,256)
                if '+' in domain:
                    self.linearB = nn.Linear(rep_dim,256) # only for multi-task setting
                self.linear2 = nn.Linear(256,3)#nclasses)
                
                if data_type == 'raw':
                    self.cls_head = nn.Linear(rep_dim,nclasses) # 400 for I3D, 512 for R3D
                
                if importance_loss == True:
                    self.importance_function = nn.Linear(rep_dim,1)
                
                self.data_type = data_type
                self.encoder_type = encoder_type
                self.encoder = encoder

                self.frame_cls = nn.Parameter(torch.rand(1,rep_dim,device=device))
                self.clip_cls = nn.Parameter(torch.rand(1,rep_dim,device=device))
                """ Embeddings Stuff """
                def createPosEmbeddings():
                    frame_pos_embeddings = nn.ParameterDict()
                    for pos in range(2000): # 2000
                        frame_pos_embeddings[str(pos)] = nn.Parameter(torch.rand(1,rep_dim,device=device)) # max number of snippets in single video, might need be to changed
                    return frame_pos_embeddings
                self.frame_pos_embeddings = createPosEmbeddings()
                self.clip_pos_embeddings = createPosEmbeddings()
                
                """ Transformer Stuff """
                def createTransEncoder(rep_dim):
                    transEncLayerFrame = nn.TransformerEncoderLayer(d_model=rep_dim,nhead=4) #nheads = 4
                    transEncoderFrame = nn.TransformerEncoder(transEncLayerFrame,num_layers=4) #nlayers = 4, # inputs must be S x B x D
                    return transEncoderFrame
                transEncoderFrame = createTransEncoder(rep_dim)
                self.transEncoderFrame = transEncoderFrame
                transEncoderClip = createTransEncoder(rep_dim)
                self.transEncoderClip = transEncoderClip
        
                """ Attention Stuff """
                self.attentionA = nn.Linear(rep_dim,256)
                self.attentionB = nn.Linear(rep_dim,256)
                self.attentionModules = nn.ModuleDict()
                self.finalModules = nn.ModuleDict()
                for category in range(3):# nclasses
                        name = str(category) #+ 'A', str(category) + 'B'
                        self.attentionModules[name] = nn.Linear(256,1)
                        self.finalModules[name] = nn.Linear(rep_dim,1)

                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
                self.tanh = nn.Tanh()
                self.nclasses = nclasses
                self.domain = domain
                self.rep_dim = rep_dim
                self.modalities = modalities 
                self.self_attention = self_attention
                self.importance_loss = importance_loss
        
        def loadPretrainedParams(self,encoder):
                params_dict = torch.load('C:/Users/DaniK/OneDrive/Desktop/r3d18_K_200ep.pth')['state_dict']
                encoder.load_state_dict(params_dict)
#               for name,param in encoder.state_dict().items():
#                   if name in params_dict.keys():
#                       if param.shape == params_dict[name].shape:
#                           param.data.copy_(params_dict[name])
                return encoder
        
        def freezeParams(self,encoder,encoder_type='ViT'):
                if encoder_type == 'I3D':
                    for param in encoder.parameters():
                        param.requires_grad = False
                    for param in encoder.logits.parameters():
                        param.requires_grad = True
                    for param in encoder.Mixed_5c.parameters():
                        param.requires_grad = True # True for NH
                    for param in encoder.Mixed_5b.parameters():
                        param.requires_grad = True # True for NH
                else:
                    for name,param in encoder.named_parameters():
                        param.requires_grad_(False)
        
        def src_mask(self,sz):
                mask = (torch.triu(torch.ones(sz, sz, device=self.device)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                return mask
        
        def calcAttention(self,snip_reps,category):
                name = str(category)# + 'A', str(category) + 'B'
                snip_repsA = self.tanh(self.attentionA(snip_reps)) # B x nsnippets x E
                snip_repsB = self.sigmoid(self.attentionB(snip_reps)) # B x nsnippets x E
                snips_gated = snip_repsA * snip_repsB # B x nsnippets x E
                attention = torch.softmax(self.attentionModules[name](snips_gated),1) # B x nsnippets x 1
                attention = attention.squeeze(-1) # B x nsnippets
                return attention
        
        def obtainVideoRep(self,snip_reps,category):
                attention = self.calcAttention(snip_reps,category)
                video_rep = torch.bmm(attention.unsqueeze(1),snip_reps) # (B x 1 x nsnippets) X (B x nsnippets x D) = B x 1 x D
                return video_rep, attention
        
        def obtainVideoScore(self,video_rep,category):
                name = str(category)# + 'A', str(category) + 'B'
                score = self.finalModules[name](video_rep) # B x 1 x 1
                return score

        def extractFeatures(self,x):
                nbatch, nsnippets, nchannels, nframes, height, width = x.shape
                
                if self.encoder_type in ['R3D','I3D']:
                        snip_rep_list = []
                        for idx in range(nsnippets):
                                snip_rep = self.encoder(x[:,idx,:])
                                snip_rep = snip_rep.view(nbatch,1,self.rep_dim) # B x 1 x D
                                snip_rep_list.append(snip_rep)
                        snip_sequence = torch.cat(snip_rep_list,1) # B x nsippets x D
                elif self.encoder_type == 'ViT':
                        snip_rep_list = []
                        for idxSnippet in range(nsnippets):
                                #frame_rep_list = []
                                snippet = x[:,idxSnippet,:] # B x nchannels x snippetLength x height x width
                                snippet_reshaped = snippet.view(nbatch*nframes,nchannels,height,width) # B*snippetLength x nchannels x height x width
#                               for idxFrame in range(nframes):
#                                       frame_rep = self.encoder(x[:,idxSnippet,idxFrame,:]) # B x D 
#                                       frame_rep = frame_rep.view(nbatch,1,-1) # B x 1 x D
#                                       frame_rep_list.append(frame_rep)
                                frame_sequence = self.encoder(snippet_reshaped) # B*snippetLength x D 
                                frame_sequence = frame_sequence.view(nbatch,1,nframes,-1) # B x 1 x snippetLength x D
                                #frame_sequence = torch.cat(frame_rep_list,1) # B x nframes x D
                                #frame_sequence = frame_sequence.view(nbatch,1,nframes,-1) # B x 1 x nframes x D
                                snip_rep_list.append(frame_sequence)
                        snip_sequence = torch.cat(snip_rep_list,1) # B x nsnippets x nframes x D

                return snip_sequence            
        
        def prepareInputForTransformer(self,x,modality='rgb'):
            #if modality == 'rgb':
            #    mode = 'frames'
            #    pos_embedding_fn = self.frame_pos_embeddings
            #elif modality == 'flows':
            #    mode = 'flows'
            #    pos_embedding_fn = self.flow_pos_embeddings
            snip_sequence = x # B x nsnippets x nframes x D
            future_reps = snip_sequence
            nbatch, nsnippets, nframes, ndim = x.shape
            pos_embeddings = torch.vstack([self.frame_pos_embeddings[str(i)] for i in range(nframes)]) # nframes x D
            #pos_embeddings = torch.stack([self.frame_pos_embeddings[str(i)] for i in range(nframes)]).squeeze() # nframes x D # USE THIS FOR ONNX CONVERSION (COMMENT OUT ABOVE LINE)
            pos_embeddings = pos_embeddings.view(1,1,nframes,ndim).repeat(nbatch,nsnippets,1,1) # nbatch x nsnippets x nframes x D
            snip_sequence += pos_embeddings # nbatch x nsnippets x nframes x D
            cls_tokens = self.frame_cls.expand(nbatch,nsnippets,1,-1) # nbatch x nsnippet x 1 x D
            snip_sequence = torch.cat((cls_tokens,snip_sequence),2) # nbatch x nsnippets x nframes+1 x D
            return snip_sequence
        
        def aggregateInputs(self,snip_sequence,lens,pad,modality='rgb'):
            #if modality == 'rgb':
            #    mode = 'frames'
            #elif modality == 'flows':
            #    mode = 'flows'
            #print(snip_sequence.shape)
            nbatch, nsnippets, nframes, ndim = snip_sequence.shape
            #nframes += 1 # b/c of the cls_tokens
            snip_sequence = snip_sequence.view(nbatch*nsnippets,nframes,-1) # B*nsnippets x nframes x D
            snip_sequence = snip_sequence.permute(1,0,2) # nframes x B*nsnippets x D
            ### NEW ###
            #key_padding_mask = self.createPaddingMask(snip_sequence,lens)
            #print(pad.shape) # reshaping is NEW
            key_padding_mask = pad.view(nbatch*nsnippets,nframes) # B*nsnippets x nframes (different from snip_sequence dimension order on purpose) 
            #print(snip_sequence.shape,key_padding_mask.shape) 
            ### END ###
            snip_reps, attn = self.transEncoderFrame(snip_sequence,src_key_padding_mask=key_padding_mask) # nframes x B*nsnippets x D
            #attn = torch.ones(10) # dummy variable for now (until I figure out versioning on Windows)
            snip_reps = self.relu(snip_reps)
            snip_reps = snip_reps.permute(1,0,2) # B*nsnippets x nframes x D
            snip_reps = snip_reps.view(nbatch,nsnippets,nframes,-1) # B x nsnippets x nframes x D
            full_snip_sequence = snip_reps
            #print(snip_reps.shape)
            snip_sequence = snip_reps[:,:,0,:] # take representation of first 'token' # B x nsnippets x D
            return full_snip_sequence, snip_sequence, attn #[:,0,:] #how cls token attends to all other frames
        
#         def createPaddingMask(self,x,lens):
#             nframes, nbatch, ndim = x.shape
#             key_padding_mask = torch.zeros(nbatch,nframes,device=self.device).type(torch.bool) #B*snippets x nframes 
#             for row,xlen in zip(range(key_padding_mask.shape[0]),lens):
#                 key_padding_mask[row,xlen:] = True
#             key_padding_mask = key_padding_mask.permute(1,0)
#             print(key_padding_mask.shape)
#             return key_padding_mask
        
        def getR3Dreps(self,inputs):
            nbatch, nsnippets, nchannels, nframes, height, width = inputs.shape
            snip_rep_list = []
            for idx in range(nsnippets):
                snip_rep = self.encoder(inputs[:,idx,:])
                #snip_rep = torch.mean(snip_rep,-1) # only for when nframes > 16
                #print(inputs.shape)
                snip_rep = snip_rep.view(nbatch,1,self.rep_dim) # B x 1 x D
                pos_embedding = self.frame_pos_embeddings[str(idx)].view(1,1,self.rep_dim).repeat(snip_rep.shape[0],1,1)
                snip_rep = snip_rep + pos_embedding
                snip_rep_list.append(snip_rep)
            snip_sequence = torch.cat(snip_rep_list,1) # B x nsippets x D
            return snip_sequence
    
        def forward(self,x,f,xlens,flens,task,xpad,fpad,domains):
                """ Forward Pass of Data Through Network 
                Args:
                        x (torch.Tensor): dim = nsnippets x 1 x channels x snippetLength x new_height x new_width
                """
                                
                if self.data_type == 'raw':
                        if isinstance(x,list): # list means we have multiple input augments 
                            if self.modalities == 'RGB':
                                snip_sequence = [self.getR3Dreps(xel) for xel in x]
                            elif self.modalities == 'Flow':
                                flow_sequence = [self.getR3Dreps(fel) for fel in f]
                            elif self.modalities == 'RGB-Flow':
                                snip_sequence = [self.getR3Dreps(xel) for xel in x]
                                flow_sequence = [self.getR3Dreps(fel) for fel in f]
                        else:
                            if self.modalities == 'RGB':
                                snip_sequence = self.getR3Dreps(x)
                            elif self.modalities == 'Flow':
                                flow_sequence = self.getR3Dreps(f) 
                            elif self.modalities == 'RGB-Flow':
                                snip_sequence = self.getR3Dreps(x)
                                flow_sequence = self.getR3Dreps(f)                
                elif self.data_type == 'reps':
                        if self.encoder_type == 'R3D':
                                snip_sequence = x # B x nsnippets x D
                                future_reps = snip_sequence # used for ss loss only
                                #snip_reps = 'None'
                                nbatch, nsnippets, ndim = x.shape
                                pos_embeddings = torch.vstack([self.pos_embeddings[str(i)] for i in range(nsnippets)]) # nsnippets x D
                                pos_embeddings = pos_embeddings.view(1,nsnippets,ndim).repeat(nbatch,1,1) # nbatch x nsnippets x D
                                snip_sequence += pos_embeddings # nbatch x nsnippets x D
                        elif self.encoder_type == 'ViT':
                                if self.self_attention == True:
                                    if isinstance(x,list): # list means we have multiple input augments 
                                        if self.modalities == 'RGB':
                                            snip_sequence = [self.prepareInputForTransformer(xel,'rgb') for xel in x]
                                        elif self.modalities == 'Flow':
                                            flow_sequence = [self.prepareInputForTransformer(fel,'flows') for fel in f]
                                        elif self.modalities == 'RGB-Flow':
                                            snip_sequence = [self.prepareInputForTransformer(xel,'rgb') for xel in x]
                                            flow_sequence = [self.prepareInputForTransformer(fel,'flows') for fel in f]
                                    else:
                                        if self.modalities == 'RGB':
                                            snip_sequence = self.prepareInputForTransformer(x,'rgb')
                                        elif self.modalities == 'Flow':
                                            flow_sequence = self.prepareInputForTransformer(f,'flows') 
                                        elif self.modalities == 'RGB-Flow':
                                            snip_sequence = self.prepareInputForTransformer(x,'rgb')
                                            flow_sequence = self.prepareInputForTransformer(f,'flows')
                                elif self.self_attention == False:
                                    if isinstance(x,list):
                                        snip_sequence = [torch.mean(xel,2) for xel in x]
                                        flow_sequence = [torch.mean(fel,2) for fel in f]
                                    else:
                                        snip_sequence = torch.mean(x,2)
                                        flow_sequence = torch.mean(f,2)

                if self.encoder_type == 'ViT':
                    if self.self_attention == True:
                        if self.modalities == 'RGB':
                            if isinstance(snip_sequence,list): # list means we have multiple input augments
                                """ Iterate Over TTA Augments (RGB) """
                                new_snip_sequence = []
                                for i,(snip_sequence_el,xlens_el,xpad_el) in enumerate(zip(snip_sequence,xlens,xpad)):
                                    full_snip_sequence, curr_snip_sequence, curr_snip_attn = self.aggregateInputs(snip_sequence_el,xlens_el,xpad_el,'rgb')
                                    new_snip_sequence.append(curr_snip_sequence)
                                    if i == 0:
                                        snip_attn = curr_snip_attn #only consider attn from one of the TTA augments
                                snip_sequence = new_snip_sequence
                            else:
                                full_snip_sequence, snip_sequence, snip_attn = self.aggregateInputs(snip_sequence,xlens,xpad,'rgb')
                        elif self.modalities == 'Flow':
                            if isinstance(flow_sequence,list): # list means we have multiple input augments
                                """ Iterate Over TTA Augments (Flow) """
                                new_flow_sequence = []
                                for i,(flow_sequence_el,flens_el,fpad_el) in enumerate(zip(flow_sequence,flens,fpad)):
                                    full_flow_sequence, curr_flow_sequence, curr_flow_attn = self.aggregateInputs(flow_sequence_el,flens_el,fpad_el,'flows')
                                    new_flow_sequence.append(curr_flow_sequence)
                                    if i == 0:
                                        snip_attn = curr_flow_attn
                                flow_sequence = new_flow_sequence
                            else:
                                full_flow_sequence, flow_sequence, snip_attn = self.aggregateInputs(flow_sequence,flens,fpad,'flows')
                        elif self.modalities == 'RGB-Flow':
                            if isinstance(snip_sequence,list): # list means we have multiple input augments
                                """ Iterate Over TTA Augments (RGB) """
                                new_snip_sequence = []
                                for i,(snip_sequence_el,xlens_el,xpad_el) in enumerate(zip(snip_sequence,xlens,xpad)):
                                    full_curr_snip_sequence, curr_snip_sequence, curr_snip_attn = self.aggregateInputs(snip_sequence_el,xlens_el,xpad_el,'rgb')
                                    new_snip_sequence.append(curr_snip_sequence)
                                    if i == 0:
                                        snip_attn = curr_snip_attn #only consider attn from one of the TTA augments
                                        full_snip_sequence = full_curr_snip_sequence #.copy()
                                snip_sequence = new_snip_sequence
                                """ Iterate Over TTA Augments (Flow) """
                                new_flow_sequence = []
                                for i,(flow_sequence_el,flens_el,fpad_el) in enumerate(zip(flow_sequence,flens,fpad)):
                                    full_curr_flow_sequence, curr_flow_sequence, curr_flow_attn = self.aggregateInputs(flow_sequence_el,flens_el,fpad_el,'flows')
                                    new_flow_sequence.append(curr_flow_sequence)
                                flow_sequence = new_flow_sequence
                                #snip_sequence = [self.aggregateInputs(snip_sequence_el,xlens_el,xpad_el,'rgb')[0] for snip_sequence_el,xlens_el,xpad_el in zip(snip_sequence,xlens,xpad)]
                                #flow_sequence = [self.aggregateInputs(flow_sequence_el,flens_el,fpad_el,'flows')[0] for flow_sequence_el,flens_el,fpad_el in zip(flow_sequence,flens,fpad)]
                            else:
                                full_snip_sequence, snip_sequence, snip_attn = self.aggregateInputs(snip_sequence,xlens,xpad,'rgb')
                                full_flow_sequence, flow_sequence, flow_attn = self.aggregateInputs(flow_sequence,flens,fpad,'flows')
                    elif self.self_attention == False:
                        snip_sequence = snip_sequence
                        flow_sequence = flow_sequence
                        snip_attn = torch.ones(1,1) #placeholder b/c self_attention is disabled 
                
                # snip_sequence ----- B x nsnippets x D
                
                if task == 'MIL':
                    """ Option 1 - MIL Pathway """
                    snip_sequence, snip_reps = self.getClipReps(snip_sequence)
                    flow_sequence, flow_reps = self.getClipReps(flow_sequence)
                    output_logits, attention_dict = self.MIL_Head(snip_reps,flow_reps=None)
                elif task == 'Prototypes':
                    """ Option 2 - Obtain Modality-Specific Video Representations """
                    if self.modalities == 'RGB':
                        if isinstance(snip_sequence,list):
                            snip_sequence = [torch.mean(snip_sequence_el,1) for snip_sequence_el in snip_sequence]
                        else:
                            snip_sequence = torch.mean(snip_sequence,1) # B x D
                    elif self.modalities == 'Flow':
                        if isinstance(flow_sequence,list):
                            flow_sequence = [torch.mean(flow_sequence_el,1) for flow_sequence_el in flow_sequence]
                        else:
                            flow_sequence = torch.mean(flow_sequence,1) # B x D
                    elif self.modalities == 'RGB-Flow':
                        if isinstance(snip_sequence,list):
                            snip_sequence = [torch.mean(snip_sequence_el,1) for snip_sequence_el in snip_sequence]
                            flow_sequence = [torch.mean(flow_sequence_el,1) for flow_sequence_el in flow_sequence]
                        else:
                            snip_sequence = torch.mean(snip_sequence,1) # B x D
                            flow_sequence = torch.mean(flow_sequence,1) # B x D
                    
                    """ Obtain Single Video Representation """
                    if self.modalities == 'RGB':
                        if isinstance(snip_sequence,list):
                            snip_sequence = snip_sequence
                            snip_sequence = [self.linear(self.relu(snip_sequence_el)) for snip_sequence_el in snip_sequence] # B x E
                            output_logits = [self.linear2(self.relu(snip_sequence_el)) for snip_sequence_el in snip_sequence] 
                        else:
                            snip_sequence = snip_sequence
                            snip_sequence = self.linear(self.relu(snip_sequence)) # B x E
                            output_logits = self.linear2(self.relu(snip_sequence)) 
                    elif self.modalities == 'Flow':
                        if isinstance(flow_sequence,list):
                            snip_sequence = flow_sequence
                            snip_sequence = [self.linear(self.relu(snip_sequence_el)) for snip_sequence_el in snip_sequence] # B x E
                            output_logits = [self.linear2(self.relu(snip_sequence_el)) for snip_sequence_el in snip_sequence] 
                        else:
                            snip_sequence = flow_sequence       
                            snip_sequence = self.linear(self.relu(snip_sequence)) # B x E
                            output_logits = self.linear2(self.relu(snip_sequence))
                    elif self.modalities == 'RGB-Flow':
                        if isinstance(snip_sequence,list):
                            snip_sequence = [snip_sequence_el + flow_sequence_el for snip_sequence_el,flow_sequence_el in zip(snip_sequence,flow_sequence)] #add optical flow    
                            if '+' in self.domain:
                                snip_sequence = [torch.stack([self.linear(self.relu(snip_sequence_element)) if domain == 'NH_02' else self.linearB(self.relu(snip_sequence_element)) for snip_sequence_element,domain in zip(snip_sequence_el,domains)]) for snip_sequence_el in snip_sequence] # only for multi-task setting with task-specific parameters
                            else:
                                snip_sequence = [self.linear(self.relu(snip_sequence_el)) for snip_sequence_el in snip_sequence] # B x E # original implemention - no multi-task setting
                            output_logits = [self.linear2(self.relu(snip_sequence_el)) for snip_sequence_el in snip_sequence] 
                        else:
                            snip_sequence += flow_sequence #add optical flow
                            if '+' in self.domain:
                                snip_sequence = torch.stack([self.linear(self.relu(snip_sequence_element)) if domain == 'NH_02' else self.linearB(self.relu(snip_sequence_element)) for snip_sequence_element,domain in zip(snip_sequence,domains)]) # only for multi-task setting with task-specific parameters
                            else:
                                snip_sequence = self.linear(self.relu(snip_sequence)) # B x E # original implemention - no multi-task setting
                            output_logits = self.linear2(self.relu(snip_sequence))            
                
                    if self.importance_loss == True:
                        """ Obtain Importance Prediction """
                        output_importances = self.importance_function(full_snip_sequence) # B x nframes x 1
                        #print(output_importances)
                elif task == 'ClassificationHead':
                    if self.modalities == 'RGB':
                        if isinstance(snip_sequence,list):
                            output_logits = [self.cls_head(self.relu(snip_sequence_el)) for snip_sequence_el in snip_sequence]
                        else:
                            output_logits = self.cls_head(self.relu(snip_sequence))
                    elif self.modalities == 'Flow':
                        if isinstance(flow_sequence,list):
                            output_logits = [self.cls_head(self.relu(flow_sequence_el)) for flow_sequence_el in flow_sequence]
                        else:
                            output_logits = self.cls_head(self.relu(flow_sequence))
                    elif self.modalities == 'RGB-Flow':
                        if isinstance(snip_sequence,list):
                            snip_sequence = [snip_sequence_el + flow_sequence_el for snip_sequence_el,flow_sequence_el in zip(snip_sequence,flow_sequence)]
                            output_logits = [self.cls_head(self.relu(torch.mean(snip_sequence_el,1))) for snip_sequence_el in snip_sequence]
                        else:
                            snip_sequence = snip_sequence + flow_sequence
                            output_logits = self.cls_head(self.relu(torch.mean(snip_sequence,1)))
                
                if task == 'MIL':
                    return snip_sequence, snip_reps, output_logits, attention_dict # snip_sequence (for Prototypes Pathway)
                elif task == 'Prototypes':
                    if self.importance_loss == True:
                        return output_importances, snip_sequence, snip_attn
                    else:
                        return snip_sequence, snip_attn
                elif task == 'ClassificationHead':
                    return snip_sequence, output_logits
        
        def getClipReps(self,snip_sequence):
            #causal_mask = self.src_mask(nsnippets)
            """ Bidrectional Attention Over Clips """
            nbatch, nsnippets, ndim = snip_sequence.shape
            pos_embeddings = torch.vstack([self.clip_pos_embeddings[str(i)] for i in range(nsnippets)]) # nsnippets x D
            pos_embeddings = pos_embeddings.view(1,nsnippets,ndim).repeat(nbatch,1,1) # B x nsnippets x D
            snip_sequence += pos_embeddings # B x nsnippets x D

            #cls_tokens = self.clip_cls.expand(nbatch,1,-1) # B x 1 x D
            #snip_sequence = torch.cat((cls_tokens,snip_sequence),1) # B x nsnippets+1 x D # no need for cls token here b/c we are considered all reps (not a summary rep)

            snip_sequence = snip_sequence.permute(1,0,2) # nsnippets x B x D
            snip_reps, attn = self.transEncoderClip(snip_sequence)#,mask=causal_mask) # nsnippets x B x D
            snip_reps = self.relu(snip_reps)
            snip_reps = snip_reps.permute(1,0,2) # B x nsnippets x D
            return snip_sequence, snip_reps

        def MIL_Head(self,snip_reps,flow_reps=None):
            """ Attention Based MIL """
            #print(snip_reps.shape)
            video_scores = []
            attention_dict = dict()
            for category in range(self.nclasses):
                video_rep, attention = self.obtainVideoRep(snip_reps,category) # B x 1 x D #you could also use this to attract to prototypes
                #flow_rep, attention = self.obtainVideoRep(flow_reps,category) # B x 1 x D #you could also use this to attract to prototypes
                #video_rep += flow_rep # B x 1 x D
                #print(video_rep.shape)
                video_score = self.obtainVideoScore(video_rep,category) # B x 1 x 1 
                video_scores.append(video_score)
                attention_dict[category] = attention

            """ Output Logits """
            video_scores = torch.cat(video_scores,1) # B x nclasses x 1
            #print(video_scores.shape)
            output_logits = video_scores.squeeze(-1) # B x nclasses
            
            return output_logits, attention_dict
            
        def calcNCELoss(snip_sequence,labels,videoname):
            gesture_prototypes = self.gesture_prototypes
            p = torch.vstack(list(gesture_prototypes.values())) # nprototypes x D
            norm = torch.norm(p,dim=1).unsqueeze(1).repeat(1,p.shape[1])
            p_norm = p / norm
            p_labels = list(gesture_prototypes.keys())
            p_labels = np.repeat(np.expand_dims(np.array(p_labels),0),snip_sequence.shape[0],axis=0) # nbatch x nprototypes
            #print(snip_sequence)
            norm = torch.norm(snip_sequence,dim=1).unsqueeze(1).repeat(1,snip_sequence.shape[1]) # nbatch x D
            s_norm = snip_sequence / norm

            sim = torch.matmul(s_norm,p_norm.T) # nbatch x nprototypes
            sim_exp = torch.exp(sim)
            sides = list(map(lambda video:video.split('_')[-1],videoname))
            labels = list(map(lambda label:str(label.cpu().detach().numpy().item()),labels))
            s_labels = list(map(lambda tup:tup[1],zip(sides,labels))) #e.g. 0L, 2R, etc
            s_labels = np.repeat(np.expand_dims(np.array(s_labels),1),p.shape[0],axis=1) # nbatch x nprototypes
            #print(p_labels)
            #print('\n',s_labels)
            cols = np.argmax(p_labels == s_labels,1)
            rows = list(range(len(cols)))
            nums = sim_exp[rows,cols] # nbatch
            dens = torch.sum(sim_exp,1) # nbatch
            #print(nums,dens)
            loss = -torch.mean(torch.log(nums/dens)) # scalar
            return loss

def loadModel(rank,world_size,savepath,data_type,nclasses,domain,rep_dim,encoder_type,task,fold,lr=0.001,modalities='RGB-Flow',freeze_encoder_params=True,self_attention=True,importance_loss=False,inference=False):
        
        #dist.init_process_group("nccl", rank=rank, world_size=world_size)
        model = fullModel(data_type,nclasses,domain,rep_dim,encoder_type,modalities=modalities,freeze_encoder_params=freeze_encoder_params,self_attention=self_attention,importance_loss=importance_loss)
        if inference == True:
            params = torch.load(os.path.join(savepath,'params.zip'),map_location='cpu')
            """ Rename Params for Compatability """
            new_params = dict()
            for param_name,param in params.items():
                new_name = param_name.split('module.')[1]
                new_params[new_name] = param
            print('# of Loaded Params: %i' % len(new_params.keys()),'# of Model Params: %i' % len(dict(model.named_parameters()).keys()))
            model.load_state_dict(new_params)
            print('Params Loaded...')
 
         # Load Pre-Trained Params From Other Task (e.g., VUA Phase Recognition)
#         parampath = os.path.join('/'.join(savepath.split('/')[:-1]),'Phase')
#         params = torch.load(os.path.join(parampath,'params'),map_location='cpu')
#         """ Rename Params for Compatability """
#         new_params = dict()
#         for param_name,param in params[fold].items():
#             new_name = param_name.split('module.')[1]
#             new_params[new_name] = param
#         print('# of Loaded Params: %i' % len(new_params.keys()),'# of Model Params: %i' % len(dict(model.named_parameters()).keys()))
#         model.load_state_dict(new_params)
#         print('Pre-Trained Params Loaded...')
    
        device = torch.device('cpu')
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # if torch.cuda.is_available():
        #     """ GPU """
        #     model.to(rank)
        #     model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        # else:
        #     """ CPU """
        model.to(device)
        # model = DDP(model, device_ids=[], find_unused_parameters=True)
        
        if inference == False:
            gesture_prototypes = nn.ParameterDict()
            nclasses = [str(i) for i in range(nclasses)]
            for gesture in nclasses: #['0','1']:#,'2']:  #'c','p','r'
                name = gesture
                gesture_prototypes[name] = nn.Parameter(torch.rand(1,256,device=device))
        else:
            gesture_prototypes = torch.load(os.path.join(savepath,'prototypes.zip'),map_location=device) #'cpu'
            gesture_prototypes = gesture_prototypes #[fold]
            print('Prototypes Loaded!')
        
        params = list(model.parameters()) + list(gesture_prototypes.values())
        optimizer = optim.SGD(params,lr=lr)
        model_dict = {'model':model,'prototypes':gesture_prototypes}

        return model_dict, optimizer, device
