"""
Created on Tue Oct  5 08:25:24 2021

@author: DaniK
"""
import torch
import os
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
#import pims
import random
import numpy as np
import pandas as pd
# from prepare_paths import obtainPaths
from operator import itemgetter
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
import h5py

# fps
fps_dict = {'VUA':
		{'1': 20.0} 
	}

class VideoDataset(Dataset):
        
        def __init__(self,root_path,dataset_name,data_type,nclasses,domain,phase,task,balance,balance_groups,single_group,group_info,importance_loss,encoder_type='R3D',encoder_params='ViT_SelfSupervised_ImageNet',frameSkip=30,snippetLength=10,overlap=0,fold=1,training_fraction=1):
                """ Load Video Dataset on the Fly 
                
                Args:
                        overlap (float): percentage overlap of the snippet length [0,1]
                
                """
                self.root_path = root_path 
                self.dataset_name = dataset_name
                self.data_type = data_type
                self.nclasses = nclasses
                self.domain = domain
                self.phase = phase
                self.task = task
                self.balance = balance
                self.balance_groups = balance_groups
                self.single_group = single_group
                self.group_info = group_info
                self.importance_loss = importance_loss
                self.encoder_type = encoder_type
                self.encoder_params = encoder_params
                self.frameSkip = frameSkip
                self.snippetLength = snippetLength
                self.overlap = overlap
                self.fold = fold
                self.training_fraction = training_fraction
                
                if self.encoder_type == 'R3D':
                        self.width = 112 # b/c network was pre-trained with this
                elif self.encoder_type in ['ViT','I3D']:
                        self.width = 224 # b/c network was pre-trained with this
                
                ### VUA EASE Dataset (for EASE prediction) ... ###
                if self.dataset_name in ['VUA_EASE']:

                    def loadExplanations(domain,inference_set='test'):
                        hospital = 'USC' #if inference_set == 'test' else inference_set.split('_')[0]
                        explain_dfA = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','EASE_Explanations.csv'))
                        explain_dfB = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','EASE_Explanations_Training.csv'))
                        explain_dfC = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','EASE_Explanations_Training_MediumSkill.csv'))
                        explain_df = pd.concat((explain_dfA,explain_dfB,explain_dfC),axis=0)
                        bool1 = explain_df['Suturing Phase'] == domain.split('_')[0]
                        bool2 = explain_df['Hospital'] == hospital
                        boolcomb = bool1 & bool2
                        curr_explain_df = explain_df[boolcomb]
                        curr_explain_df.columns = [el.replace('File Number','File') for el in curr_explain_df.columns]
                        return curr_explain_df

                    def returnFrameIndices(curr_df):
                        race = curr_df['RACE']
                        if race == 'Needle Withdrawal':
                            colStartName = 'Needle Withdrawal Start Frame'
                            colEndName = 'Needle Withdrawal End Frame'
                        elif race == 'Needle Handling':
                            colStartName = 'Needle Handling Start Frame'
                            colEndName = 'Needle Entry Start Frame'
                        elif race == 'Needle Driving':
                            colStartName = 'Needle Entry Start Frame'
                            colEndName = 'Needle Withdrawal Start Frame'
                        startIdx = curr_df[colStartName] #-1 # here, I care about frame number and NOT index
                        endIdx = curr_df[colEndName]#-1 # here, I care about frame number and NOT index

                        if race == 'Needle Withdrawal':
                            jump_size = int((endIdx - startIdx)//10)
                            start, end = startIdx, endIdx
                            indices = np.arange(start,end,jump_size)
                            #indices = np.arange(startIdx-40,startIdx+40,10) #looking back is fine to ensure you capture actual withdrawal
                        elif race == 'Needle Handling':
                            diff = endIdx - startIdx
                            frames_to_drop = int(diff * 0.20)
                            start, end = startIdx, endIdx-frames_to_drop
                            indices = np.arange(start,end,10)
                            #indices = np.arange(startIdx,endIdx-20,10) #do not look back, which may leak into withdrawal from previous stitch
                        elif race == 'Needle Driving':
                            diff = endIdx - startIdx
                            frames_to_drop = int(diff * 0.20)
                            start, end = startIdx, endIdx-frames_to_drop
                            indices = np.arange(start,end,10)

                        indices = indices - startIdx # reset indices to start from 0 (to enable easy comparison to ground-truth explanations)
                        return indices

                    def getFrameImportance(curr_df):
                        frame_numbers = curr_df['frame indices']
                        nspans = 6
                        frameImportance_list = []
                        for frame_number in frame_numbers:
                            for n in range(1,nspans+1):
                                startFrame, endFrame = curr_df['Start%i Frame' % n], curr_df['End%i Frame' % n]
                                if frame_number <= endFrame and frame_number >= startFrame:
                                    frameImportance = 1
                                    break # once frame number is found to be in some interval (only occurs once), then break to avoid overwriting
                                else:
                                    frameImportance = 0
                            frameImportance_list.append(frameImportance)
                        return frameImportance_list

                    def getImportance(df_train):
                        low_skill_vals = list(set(df_train['maj'].unique().tolist()) - set([2])) # 2 is high skill
                        high_skill_df = df_train[df_train['maj']==2]
                        low_skill_df = df_train[df_train['maj'].isin(low_skill_vals)] # ==0 (default) | .isin([0,1]) for multi-class skill assessment (assuming low skill is 0 and 1)
                        curr_explain_df = loadExplanations(domain)#,inference_set=inference_set)
                        
                        fold_df = low_skill_df.copy()
                        fold_df['frame indices'] = fold_df.apply(returnFrameIndices,axis=1)
                        fold_df.columns = fold_df.columns.str.replace('CaseID','File') #to allow for merging on CaseID
                        fold_df = fold_df.merge(curr_explain_df,how='left',on=['File','Stitch'])
                        #""" NEW - August 10th """
                        #fold_df = fold_df[fold_df['Start1'].notna()] # remove rows with potential ground-truth EASE label noise (based on how I annotated explanations)
                        #""" END """
                        fold_df = fold_df[fold_df['frame indices'].notna()]
                        fold_df['frame importance'] = fold_df.apply(getFrameImportance,axis=1)
                        fold_df['frame importance'] = fold_df['frame importance'].apply(lambda attn:list(attn))

                        df_train = pd.merge(high_skill_df,fold_df,indicator=True,how='outer')
                        return df_train

                    def durFilterFunc(row):
                        #dur = 10 #frames = 10/20fps = 0.5 seconds
                        if row['RACE'] == 'Needle Handling':
                            dur = 20 
                            dur_bool = (row['Needle Entry Start Frame'] - row['Needle Handling Start Frame']) > dur
                        elif row['RACE'] == 'Needle Withdrawal':
                            dur = 80
                            dur_bool = (row['Needle Withdrawal End Frame'] - row['Needle Withdrawal Start Frame']) > dur
                        elif row['RACE'] == 'Needle Driving':
                            startIdx, endIdx = row['Needle Entry Start Frame'], row['Needle Withdrawal Start Frame']
                            diff = endIdx - startIdx
                            frames_to_drop = int(diff * 0.20)
                            dur_bool = diff > frames_to_drop
                        return dur_bool

                    def RaceAndEaseFilter(row,race):
                        val = False
                        if race == 'NW':
                            if row['RACE'] == 'Needle Withdrawal':
                                if row['EASE'] == 'Wrist Rotation':
                                    val = True
                        elif race == 'NH':
                            if row['RACE'] == 'Needle Handling':
                                if row['EASE'] == '# Repositions':
                                    val = True
                        elif race == 'ND':
                            if row['RACE'] == 'Needle Driving':
                                if row['EASE'] == 'Driving Sequence':
                                    val = True
                        return val

                    def balance_scores(df,maj_labels):
                        """ Balance Classes """
                        min_class_amount = df['maj'].value_counts().min()
                        balanced_df = pd.DataFrame()
                        for maj_label in maj_labels:
                            curr_df = df[df['maj']==maj_label].sample(n=min_class_amount,replace=False,random_state=0)
                            balanced_df = pd.concat((balanced_df,curr_df),axis=0)
                        df = balanced_df.copy()
                        return df

                    def balanceGroups(df,group='Caseload Group',inference_set='test'):
                        """ Load Meta Information """
                        meta_df = loadMetaInfo(inference_set)
                        meta_df.drop_duplicates(subset=['CaseID','TaskID'],keep='first',inplace=True)
                        """ Include Meta Information """
                        df['TaskID'] = df['Anatomy'].apply(lambda side:11 if side == 'Posterior' else 12) #to allow for merging on TaskID
                        df.columns = df.columns.astype(str)
                        df.columns = df.columns.str.replace('File','CaseID') #to allow for merging on CaseID
                        df.columns = [int(col) if col.isdigit() else col for col in df.columns]
                        """ Merge Meta Information """
                        df = df.merge(meta_df,how='left',on=['CaseID','TaskID'])
                        #print(group)
                        """ Balance Groups in Each Class """
                        unique_labels = df['maj'].unique().tolist()
                        final_df = pd.DataFrame()
                        for label in unique_labels:
                            curr_df = df[df['maj']==label]
                            min_amount = curr_df[group].value_counts().min()
                            groups = curr_df[curr_df[group].notna()][group].unique()
                            for group_val in groups:
                                group_df = curr_df[curr_df[group]==group_val].sample(n=min_amount,replace=False,random_state=0)
                                final_df = pd.concat((final_df,group_df),axis=0)
                        return final_df

                    def getSingleGroup(df,group_info,inference_set='test'): # ≤60
                        group = group_info['group_name']
                        group_val_name = group_info['group_val']
                        group_val = '>60' if group_val_name == 'ProstateLarge60ml' else '≤60'
                        """ Load Meta Information """
                        meta_df = loadMetaInfo(inference_set)
                        meta_df.drop_duplicates(subset=['CaseID','TaskID'],keep='first',inplace=True)
                        """ Include Meta Information """
                        df['TaskID'] = df['Anatomy'].apply(lambda side:11 if side == 'Posterior' else 12) #to allow for merging on TaskID
                        df.columns = df.columns.astype(str)
                        df.columns = df.columns.str.replace('File','CaseID') #to allow for merging on CaseID
                        df.columns = [int(col) if col.isdigit() else col for col in df.columns]
                        """ Merge Meta Information """
                        df = df.merge(meta_df,how='left',on=['CaseID','TaskID'])

                        """ Only Consider the Subset of the Group Satisfying Group Val (e.g., Prostate Vol. > 60ml)"""
                        curr_df = df[df[group]==group_val]
                        return curr_df

                    def loadMetaInfo(inference_set):
                        if inference_set == 'Gronau_inference': # Gronau Path
                            meta_df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','Meta','surgeon_and_patient_case_meta_gronau.csv'),index_col=0)
                            meta_df['CaseID'] = meta_df['CaseID'].apply(lambda case:int(case.split('GP-')[1]))
                            meta_df['Prostate Volume Group'] = meta_df['Prostate volume'].apply(lambda vol:'≤49' if vol <= 49 else '>49')
                            meta_df['Patient Age Group'] = meta_df['Age'].apply(lambda age:'≤66' if age <= 66 else '>66')
                            meta_df['Patient BMI Group'] = meta_df['BMI'].apply(lambda bmi:'≤28' if bmi <= 28 else '>28')
                            meta_df['Preop Gleason'] = meta_df['Preop Gleason'].apply(lambda num:num if np.isnan(num) else str(int(num))) #convert to strings
                        else: # USC Path 
                            meta_df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','Meta','surgeon_and_patient_case_meta.csv'),index_col=0) #CaseID
                            meta_df['TaskID'] = meta_df['TaskID'].astype(int)
                            meta_df['Surgery Date'] = pd.to_datetime(meta_df['Surgery Date'])
                            meta_df['Surgery Year'] = meta_df['Surgery Date'].apply(lambda date:int(date.year) if not pd.isnull(date) else date)
                            meta_df['Caseload'] = meta_df.apply(lambda row:row['%i Robotic Cases' % row['Surgery Year']] if row['Surgery Year'] in [2016,2017,2018,2019] else np.nan,axis=1)
                            meta_df['Caseload'] = meta_df['Caseload'].fillna(-1)
                            meta_df['Caseload'] = meta_df['Caseload'].astype(int)
                            meta_df['Caseload Group'] = pd.cut(meta_df['Caseload'],[0,100,float('inf')],labels=['novice','expert'])
                            meta_df['Prostate Volume Group'] = pd.qcut(meta_df['Prostate volume'],[0,0.5,1],labels=['≤49','>49']) # nans stay as nans
                            meta_df['Patient Age Group'] = pd.qcut(meta_df['Age'],[0,0.5,1],labels=['≤66','>66'])
                            meta_df['Patient BMI Group'] = pd.qcut(meta_df['BMI'],[0,0.5,1],labels=['≤28','>28'])
                            meta_df['Preop Gleason'] = meta_df['Preop Gleason'].replace('-9','9') # correct input error
                            meta_df['Preop Gleason'] = meta_df['Preop Gleason'].replace('3',np.nan)
                            meta_df['Preop Gleason'] = meta_df['Preop Gleason'].replace('UTC',np.nan)
                            meta_df['Preop Gleason'] = meta_df['Preop Gleason'].replace('UTA',np.nan)
                        #Rank, Preop Gleason
                        return meta_df

                    def obtain_train_val_split(df,maj_labels,kind='Video',balance=True,balance_groups=False,single_group=False,group_info=dict(),importance_loss=False):
                        """ Split Data for Training and Validation
                        Args:
                            kind (str): type of split. Options: 'Instance' | 'Video'
                            balance (bool): balance both training and validation sets
                        Output:
                            train_df (pd.DataFrame)
                            val_df (pd.DataFrame)
                        """

                        if kind == 'Video':
                            cases = df['Video'].unique().tolist()
                            ncases = len(cases)
                            random.seed(fold)
                            train_cases = random.sample(cases,int(0.9*ncases))
                            val_cases = random.sample(train_cases,int(0.1*len(train_cases)))
                            train_cases = list(set(train_cases) - set(val_cases))
                            test_cases = list(set(cases) - set(train_cases) - set(val_cases))
                            assert set(train_cases).intersection(set(val_cases)).intersection(set(test_cases)) == set()
                            train_indices, val_indices, test_indices = np.where(df['Video'].isin(train_cases))[0], np.where(df['Video'].isin(val_cases))[0], np.where(df['Video'].isin(test_cases))[0]
                            df_train, df_val, df_test = df.iloc[train_indices,:], df.iloc[val_indices,:], df.iloc[test_indices,:]

                            if balance == True:
                                if balance_groups == True:
                                    assert single_group == False
                                    df_train = balanceGroups(df_train)

                                if single_group == True:
                                    assert balance_groups == False
                                    df_train = getSingleGroup(df_train,group_info)
                                
                                if importance_loss == True:
                                    df_train = getImportance(df_train)#,inference_set)
                                
                                df_train = balance_scores(df_train,maj_labels)
                                df_val = balance_scores(df_val,maj_labels)
                                df_test = balance_scores(df_test,maj_labels)
                            elif balance == False: # do not balance train, BUT balance val/test sets
                                if importance_loss == True:
                                    df_train = getImportance(df_train)
                                
                                df_val = balance_scores(df_val,maj_labels)
                                df_test = balance_scores(df_test,maj_labels)

                        return df_train, df_val, df_test
                    
                    if 'inference' in phase: # path for datasets used during inference
                        if 'Gronau' in phase:
                            downstream_dataset = 'VUA_Gronau'
                        elif 'COH' in phase:
                            downstream_dataset = 'VUA_COH'
                        elif 'HMH' in phase:
                            downstream_dataset = 'VUA_HMH'
                        elif 'Lab' in phase:
                            downstream_dataset = 'VUA_Lab'
                        elif 'AFB' in phase:
                            downstream_dataset = 'VUA_AFB'
                        elif 'USC' in phase: # December 2022
                            downstream_dataset = 'VUA'
                        hf_rgb = h5py.File(os.path.join(self.root_path,downstream_dataset,'Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                        #hf_rgb = h5py.File(os.path.join(self.root_path,downstream_dataset,'Results','ViT_SelfSupervised_ImageNet_RepsAndLabels.h5'),'r')
                        hf_of = h5py.File(os.path.join(self.root_path,downstream_dataset,'Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')
                        df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','%s_EASE_Stitch_Paths.csv' % downstream_dataset),index_col=0) #.replace('md3','md2')
                        df['Video'] = df['Path'].apply(lambda path:path.split('\\')[-1] if '\\' in path else path.split('/')[-1])
                        race = domain.split('_')[0]

                        """ Only Use Videos Available in Directory """
                        videos_to_keep = list(hf_rgb.keys())
                        df = df[df['Video'].isin(videos_to_keep)]
                        bool1 = df[['RACE','EASE']].apply(lambda row: RaceAndEaseFilter(row,race),axis=1)
                        #bool3 = df['maj'].isin(maj_labels)
                        #boolcomb = bool1 & bool3
                        df = df[bool1]
                        df['maj'] = df['maj'].map({0:0,1:0,2:2}) # to increase number of negative cases (DEFAULT)
                        self.label_encoder = label_encoder.fit(df['maj']) 
                            
                        if phase == 'Gronau_full_inference':
                            """ Perform inference on entire dataset (no filters) """
                            final_df = df.copy()
                        elif phase == 'Lab_inference':
                            final_df = df.copy()
                            final_df['Domain'] = domain # filler
                        elif phase == 'AFB_inference':
                            """ Balance Classes """
                            maj_labels = [0,2]
                            final_df = balance_scores(df,maj_labels)
                            final_df['Domain'] = domain # filler
                        elif phase == 'USC_inference': # December 2022
                            df = df[~df['File'].isin([102,372])]
                            df = df[~df['videoname'].str.contains('P-')] # these files are causing issues with indices (might be fps mismatch)
                            dur_bool = df[['RACE','Needle Handling Start Frame','Needle Entry Start Frame','Needle Withdrawal Start Frame','Needle Withdrawal End Frame']].apply(durFilterFunc,axis=1)
                            df = df[dur_bool] 
                            final_df = df.copy()
                            final_df['Domain'] = domain # filler
                        else:
                            """ Downsample Majority Group - Equal # of Instances from Each PID """
                            if downstream_dataset == 'VUA_Gronau':
                                nsamples = 5
                            elif downstream_dataset == 'VUA_COH':
                                nsamples = 5
                            elif downstream_dataset == 'VUA_HMH':
                                nsamples = 5 #5
                            high_df = df[df['maj']==2]
                            low_df = df[df['maj']==0]
                            high_sampled_df = pd.DataFrame()
                            for pid in high_df['Video'].unique():
                                pid_df = high_df[high_df['Video']==pid].sample(nsamples,random_state=0)
                                high_sampled_df = pd.concat((high_sampled_df,pid_df),axis=0)

                            """ Balance Classes """
                            df = pd.concat((high_sampled_df,low_df),axis=0)
                            min_count = df['maj'].value_counts().min()
                            scores = df['maj'].unique()
                            final_df = pd.DataFrame()
                            for score in scores:
                                curr_df = df[df['maj']==score].sample(min_count,random_state=0)
                                final_df = pd.concat((final_df,curr_df),axis=0)
                            final_df['Domain'] = domain # filler
                        print(final_df['maj'].value_counts())
                        data = {phase:final_df}
                    else:
                        hf_rgb = h5py.File(os.path.join(self.root_path,'VUA','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                        #hf_rgb = h5py.File(os.path.join(self.root_path,'VUA','Results','ViT_SelfSupervised_ImageNet_RepsAndLabels.h5'),'r')
                        hf_of = h5py.File(os.path.join(self.root_path,'VUA','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')
                        df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','VUA_EASE_Stitch_Paths.csv'),index_col=0)
                        df = df[~df['File'].isin([102,372])] #problematic videos that need further inspection (fps discrepancy)
                        df['Video'] = df['Path'].apply(lambda path:path.split('\\')[-1])

                        def get_train_val_data_one_domain(domain,df):
                            race = domain.split('_')[0] # e.g., 'NH_02'
                            #print(race)
                            dur_bool = df[['RACE','Needle Handling Start Frame','Needle Entry Start Frame','Needle Withdrawal Start Frame','Needle Withdrawal End Frame']].apply(durFilterFunc,axis=1)
                            df = df[dur_bool] 

                            maj_labels = list(map(lambda label:int(label),list(domain.split('_')[1]))) #[0,2] #start with extremes NH_02
                            bool1 = df[['RACE','EASE']].apply(lambda row: RaceAndEaseFilter(row,race),axis=1)
                            bool3 = df['maj'].isin(maj_labels)
                            boolcomb = bool1 & bool3
                            extreme_df = df[boolcomb]
                            #df['maj'] = df['maj'].map({0:0,1:0,2:1})
                            self.label_encoder = label_encoder.fit(extreme_df['maj'])
                            split = 'Video'

                            df_train, df_val, df_test = obtain_train_val_split(extreme_df,maj_labels,kind=split,balance=balance,balance_groups=balance_groups,single_group=single_group,group_info=group_info,importance_loss=importance_loss)
                        
                            print(df_train['maj'].value_counts())
                            train_data = df_train.copy() #_in.copy()
                            val_data = df_val.copy() #pd.concat((df_val_in,df_val_out),0)
                            test_data = df_test.copy()
                            # new to track domain of data point
                            train_data['Domain'] = domain 
                            val_data['Domain'] = domain
                            test_data['Domain'] = domain
                            return train_data, val_data, test_data

                        if '+' in domain: # multi-task learning paradigm
                            race_domains = domain.split('+') # e.g., domain = 'NH_02+ND_02'
                            train_data, val_data, test_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                            for domain in race_domains:
                                curr_train_data, curr_val_data, curr_test_data = get_train_val_data_one_domain(domain,df)
                                train_data, val_data, test_data = pd.concat((train_data,curr_train_data),axis=0), pd.concat((val_data,curr_val_data),axis=0), pd.concat((test_data,curr_test_data),axis=0) 
                        else:
                            train_data, val_data, test_data = get_train_val_data_one_domain(domain,df) # domain = 'NH_02'
                            
                        data = {'train':train_data,'val':val_data,'test':test_data}

                    self.data = data
                    self.hf_rgb = hf_rgb
                    self.hf_of = hf_of
                ### VUA EASE Dataset (for prototypes) ... ###
                elif self.dataset_name in ['VUA_EASE_Stitch']: # I am using this for phase recognition

                    def durFilterFunc(row):
                        #dur = 10 #frames = 10/20fps = 0.5 seconds
                        if row['RACE'] == 'Needle Handling':
                            dur = 20 
                            dur_bool = (row['Needle Entry Start Frame'] - row['Needle Handling Start Frame']) > dur
                        elif row['RACE'] == 'Needle Withdrawal':
                            dur = 80
                            dur_bool = (row['Needle Withdrawal End Frame'] - row['Needle Withdrawal Start Frame']) > dur
                        elif row['RACE'] == 'Needle Driving':
                            #dur = 100
                            startIdx, endIdx = row['Needle Entry Start Frame'], row['Needle Withdrawal Start Frame']
                            diff = endIdx - startIdx
                            frames_to_drop = int(diff * 0.20)
                            dur_bool = diff > frames_to_drop
                        return dur_bool                

                    def RaceAndEaseFilter(row):
                        val = False
                        if row['RACE'] == 'Needle Withdrawal':
                            if row['EASE'] == 'Wrist Rotation':
                                val = True
                        elif row['RACE'] == 'Needle Handling':
                            if row['EASE'] == '# Repositions':
                                val = True
                        elif row['RACE'] == 'Needle Driving':
                            if row['EASE'] == 'Driving Sequence':
                                val = True
                        return val

                    if phase == 'USC_inference': # perform inference on VUA videos as if no time-stamps were available
                        #hf_rgb = h5py.File(os.path.join(self.root_path,'VUA','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                        hf_rgb = h5py.File(os.path.join(self.root_path,'VUA','Results','ViT_SelfSupervised_ImageNet_RepsAndLabels.h5'),'r')
                        hf_of = h5py.File(os.path.join(self.root_path,'VUA','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')
                        df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','VUA_Paths.csv'),index_col=0)
                        to_exclude = (df['label'].str.contains('Log')) | (df['label'].str.contains('SESSION'))
                        subdf = df[~to_exclude] # these all have fps = 20 (makes life easier to work with)
                        countdf = subdf.groupby(by=['category','label']).count().reset_index()
                        countdf.columns = ['category','label','count']
                        
                        duration = 10 # seconds
                        hop = 5 # seconds
                        fps = 20 # fps
                        duration_frames = duration * fps
                        hop_frames = hop * fps
                        inference_df = pd.DataFrame()
                        for idx,(category,label,total_frames) in tqdm(countdf.iterrows()):
                            nsamples = (total_frames - duration_frames)//hop_frames + 1
                            startframes = [n * hop_frames for n in range(nsamples)]
                            endframes = [startframe + duration_frames for startframe in startframes]
                            frames_df = pd.DataFrame([startframes,endframes]).T
                            frames_df.columns = ['StartFrame','EndFrame']
                            frames_df[['category','label']] = [category,label]
                            frames_df[['Video','Domain']] = [label,'NH_vs_ND_vs_NW']
                            inference_df = pd.concat((inference_df,frames_df),axis=0)
                        df = inference_df.copy()
                        #df = df.iloc[14800:]
                        data = {'USC_inference':df}
                    elif 'inference' in phase:
                        if 'Gronau' in phase:
                            downstream_dataset = 'VUA_Gronau'
                        elif 'COH' in phase:
                            downstream_dataset = 'VUA_COH'
                        elif 'HMH' in phase:
                            downstream_dataset = 'VUA_HMH'
                        hf_rgb = h5py.File(os.path.join(self.root_path,downstream_dataset,'Results','ViT_SelfSupervised_ImageNet_RepsAndLabels.h5'),'r')
                        hf_of = h5py.File(os.path.join(self.root_path,downstream_dataset,'Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')
                        df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','%s_EASE_Stitch_Paths.csv' % downstream_dataset),index_col=0)
                        df['Video'] = df['Path'].apply(lambda path:path.split('\\')[-1])

                        ease_domains = ['Wrist Rotation','# Repositions','Driving Sequence']
                        maj_labels = [0,1,2] # 2
                        bool1 = df[['RACE','EASE']].apply(RaceAndEaseFilter,axis=1)
                        bool3 = df['maj'].isin(maj_labels)
                        boolcomb = bool1 & bool3
                        df = df[boolcomb]
                        df['Domain'] = 'NH_vs_ND_vs_NW'
                        
                        """ Only Use Videos Available in Directory """
                        videos_to_keep = list(hf_rgb.keys())
                        df = df[df['Video'].isin(videos_to_keep)]

                        self.label_encoder = label_encoder.fit(ease_domains)
                        data = {phase:df}
                    else:
                        #hf_rgb = h5py.File(os.path.join(self.root_path,'VUA','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                        hf_rgb = h5py.File(os.path.join(self.root_path,'VUA','Results','ViT_SelfSupervised_ImageNet_RepsAndLabels.h5'),'r')
                        hf_of = h5py.File(os.path.join(self.root_path,'VUA','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')
                        df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','VUA_EASE_Stitch_Paths.csv'),index_col=0)
                        df = df[~df['File'].isin([102,372])] #problematic videos that need further inspection (fps discrepancy)
                        df['Video'] = df['Path'].apply(lambda path:path.split('\\')[-1])

                        dur_bool = df[['RACE','Needle Handling Start Frame','Needle Entry Start Frame','Needle Withdrawal Start Frame','Needle Withdrawal End Frame']].apply(durFilterFunc,axis=1)
                        df = df[dur_bool] 

                        #race_domains = ['Needle Withdrawal','Needle Handling','Needle Driving']
                        ease_domains = ['Wrist Rotation','# Repositions','Driving Sequence']
                        maj_labels = [0,1,2] # 2
                        bool1 = df[['RACE','EASE']].apply(RaceAndEaseFilter,axis=1)
                        bool3 = df['maj'].isin(maj_labels)
                        boolcomb = bool1 & bool3
                        df = df[boolcomb]

                        self.label_encoder = label_encoder.fit(ease_domains)

                        cases = df['Video'].unique().tolist()
                        ncases = len(cases)
                        random.seed(fold)
                        train_cases = random.sample(cases,int(0.9*ncases))
                        val_cases = random.sample(train_cases,int(0.1*len(train_cases)))
                        train_cases = list(set(train_cases) - set(val_cases))
                        test_cases = list(set(cases) - set(train_cases) - set(val_cases))
                        assert set(train_cases).intersection(set(val_cases)).intersection(set(test_cases)) == set()
                        train_indices, val_indices, test_indices = np.where(df['Video'].isin(train_cases))[0], np.where(df['Video'].isin(val_cases))[0], np.where(df['Video'].isin(test_cases))[0]
                        df_train, df_val, df_test = df.iloc[train_indices,:], df.iloc[val_indices,:], df.iloc[test_indices,:]
                        domain = 'NH_vs_ND_vs_NW'
                        df_train['Domain'] = domain
                        df_val['Domain'] = domain
                        df_test['Domain'] = domain
                        
                        if self.training_fraction < 1 and self.phase == 'train':
                            nsamples = int(df_train.shape[0] * self.training_fraction)
                            df_train = df_train.sample(n=nsamples,random_state=0)

                        #scores = [0,1]
                        #bool3 = df['maj'].isin(scores)
                        #boolcomb = bool1 & bool3
                        #df_val_out = df[boolcomb]

                        train_data = df_train.copy() #_in.copy()
                        val_data = df_val.copy() #pd.concat((df_val_in,df_val_out),0)
                        test_data = df_test.copy()
                        data = {'train':train_data,'val':val_data,'test':test_data}

                        print(train_data.groupby(by=['RACE','EASE'])['maj'].value_counts())
                        print(val_data.groupby(by=['RACE','EASE'])['maj'].value_counts())

                    self.data = data
                    self.hf_rgb = hf_rgb
                    self.hf_of = hf_of
                elif self.dataset_name in ['NS_vs_VUA']:
                    """ Part 1 - VUA Data """
                    def durFilterFunc(row):
                        #dur = 10 #frames = 10/20fps = 0.5 seconds
                        if row['RACE'] == 'Needle Handling':
                            dur = 20 
                            dur_bool = (row['Needle Entry Start Frame'] - row['Needle Handling Start Frame']) > dur
                        elif row['RACE'] == 'Needle Withdrawal':
                            dur = 80
                            dur_bool = (row['Needle Withdrawal End Frame'] - row['Needle Withdrawal Start Frame']) > dur
                        elif row['RACE'] == 'Needle Driving':
                            #dur = 100
                            startIdx, endIdx = row['Needle Entry Start Frame'], row['Needle Withdrawal Start Frame']
                            diff = endIdx - startIdx
                            frames_to_drop = int(diff * 0.20)
                            dur_bool = diff > frames_to_drop
                        return dur_bool                

                    def RaceAndEaseFilter(row):
                        val = False
                        if row['RACE'] == 'Needle Withdrawal':
                            if row['EASE'] == 'Wrist Rotation':
                                val = True
                        elif row['RACE'] == 'Needle Handling':
                            if row['EASE'] == '# Repositions':
                                val = True
                        elif row['RACE'] == 'Needle Driving':
                            if row['EASE'] == 'Driving Sequence':
                                val = True
                        return val
                    
                    def getStartAndEndFrame(row):
                        val = False
                        if row['RACE'] == 'Needle Withdrawal':
                            if row['EASE'] == 'Wrist Rotation':
                                startframe,endframe = row['Needle Withdrawal Start Frame']-40, row['Needle Withdrawal Start Frame']+40
                        elif row['RACE'] == 'Needle Handling':
                            if row['EASE'] == '# Repositions':
                                startframe,endframe =  row['Needle Handling Start Frame'], row['Needle Entry Start Frame']
                        elif row['RACE'] == 'Needle Driving':
                            if row['EASE'] == 'Driving Sequence':
                                startframe,endframe = row['Needle Entry Start Frame'], row['Needle Withdrawal Start Frame']
                        return pd.Series([startframe,endframe])

                    if phase == 'USC_inference': # perform inference on VUA videos as if no time-stamps were available
                        #hf_rgb = h5py.File(os.path.join(self.root_path,'VUA','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                        hf_rgb = h5py.File(os.path.join(self.root_path,'VUA','Results','ViT_SelfSupervised_ImageNet_RepsAndLabels.h5'),'r')
                        hf_of = h5py.File(os.path.join(self.root_path,'VUA','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')
                        df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','VUA_Paths.csv'),index_col=0)
                        to_exclude = (df['label'].str.contains('Log')) | (df['label'].str.contains('SESSION'))
                        subdf = df[~to_exclude] # these all have fps = 20 (makes life easier to work with)
                        countdf = subdf.groupby(by=['category','label']).count().reset_index()
                        countdf.columns = ['category','label','count']
                        
                        duration = 10 # seconds
                        hop = 5 # seconds
                        fps = 20 # fps
                        duration_frames = duration * fps
                        hop_frames = hop * fps
                        inference_df = pd.DataFrame()
                        for idx,(category,label,total_frames) in tqdm(countdf.iterrows()):
                            nsamples = (total_frames - duration_frames)//hop_frames + 1
                            startframes = [n * hop_frames for n in range(nsamples)]
                            endframes = [startframe + duration_frames for startframe in startframes]
                            frames_df = pd.DataFrame([startframes,endframes]).T
                            frames_df.columns = ['StartFrame','EndFrame']
                            frames_df[['category','label']] = [category,label]
                            frames_df[['Video','Domain']] = [label,'NH_vs_ND_vs_NW']
                            inference_df = pd.concat((inference_df,frames_df),axis=0)
                        df = inference_df.copy()
                        #df = df.iloc[14800:]
                        data = {'USC_inference':df}
                    elif 'inference' in phase:
                        if 'Gronau' in phase:
                            downstream_dataset = 'VUA_Gronau'
                        elif 'COH' in phase:
                            downstream_dataset = 'VUA_COH'
                        elif 'HMH' in phase:
                            downstream_dataset = 'VUA_HMH'
                        hf_rgb = h5py.File(os.path.join(self.root_path,downstream_dataset,'Results','ViT_SelfSupervised_ImageNet_RepsAndLabels.h5'),'r')
                        hf_of = h5py.File(os.path.join(self.root_path,downstream_dataset,'Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')
                        df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','%s_EASE_Stitch_Paths.csv' % downstream_dataset),index_col=0)
                        df['Video'] = df['Path'].apply(lambda path:path.split('\\')[-1])

                        ease_domains = ['Wrist Rotation','# Repositions','Driving Sequence']
                        maj_labels = [0,1,2] # 2
                        bool1 = df[['RACE','EASE']].apply(RaceAndEaseFilter,axis=1)
                        bool3 = df['maj'].isin(maj_labels)
                        boolcomb = bool1 & bool3
                        df = df[boolcomb]
                        df['Domain'] = 'NH_vs_ND_vs_NW'
                        
                        """ Only Use Videos Available in Directory """
                        videos_to_keep = list(hf_rgb.keys())
                        df = df[df['Video'].isin(videos_to_keep)]

                        self.label_encoder = label_encoder.fit(ease_domains)
                        data = {phase:df}
                    else:
                        #hf_rgb = h5py.File(os.path.join(self.root_path,'VUA','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                        hf_rgb = h5py.File(os.path.join(self.root_path,'VUA','Results','ViT_SelfSupervised_ImageNet_RepsAndLabels.h5'),'r')
                        hf_of = h5py.File(os.path.join(self.root_path,'VUA','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')
                        df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','VUA_EASE_Stitch_Paths.csv'),index_col=0)
                        df = df[~df['File'].isin([102,372])] #problematic videos that need further inspection (fps discrepancy)
                        df['Video'] = df['Path'].apply(lambda path:path.split('\\')[-1])

                        dur_bool = df[['RACE','Needle Handling Start Frame','Needle Entry Start Frame','Needle Withdrawal Start Frame','Needle Withdrawal End Frame']].apply(durFilterFunc,axis=1)
                        df = df[dur_bool] 

                        #race_domains = ['Needle Withdrawal','Needle Handling','Needle Driving']
                        ease_domains = ['Wrist Rotation','# Repositions','Driving Sequence']
                        maj_labels = [0,1,2] # 2
                        bool1 = df[['RACE','EASE']].apply(RaceAndEaseFilter,axis=1)
                        bool3 = df['maj'].isin(maj_labels)
                        boolcomb = bool1 & bool3
                        df = df[boolcomb]
                        domain = 'VUA'
                        df['Domain'] = domain
                        
                        #self.label_encoder = label_encoder.fit(ease_domains)
                        df[['StartFrame','EndFrame']] = df.apply(getStartAndEndFrame,axis=1) # CHECK

                        cases = df['Video'].unique().tolist()
                        ncases = len(cases)
                        random.seed(fold)
                        train_cases = random.sample(cases,int(0.9*ncases))
                        val_cases = random.sample(train_cases,int(0.1*len(train_cases)))
                        train_cases = list(set(train_cases) - set(val_cases))
                        test_cases = list(set(cases) - set(train_cases) - set(val_cases))
                        assert set(train_cases).intersection(set(val_cases)).intersection(set(test_cases)) == set()
                        train_indices, val_indices, test_indices = np.where(df['Video'].isin(train_cases))[0], np.where(df['Video'].isin(val_cases))[0], np.where(df['Video'].isin(test_cases))[0]
                        df = df[['StartFrame','EndFrame','Domain','Video']]
                        df_train, df_val, df_test = df.iloc[train_indices,:], df.iloc[val_indices,:], df.iloc[test_indices,:]
                        
                        if self.training_fraction < 1 and self.phase == 'train':
                            nsamples = int(df_train.shape[0] * self.training_fraction)
                            df_train = df_train.sample(n=nsamples,random_state=0)

                        train_data_vua = df_train.copy() #_in.copy()
                        val_data_vua = df_val.copy() #pd.concat((df_val_in,df_val_out),0)
                        test_data_vua = df_test.copy()
                        
                        hf_rgb_vua = hf_rgb
                        hf_of_vua = hf_of
                        
                    """ Part 2 - NS Data """
                    def filter_examples(df):
                        min_nframes = 10
                        #max_nframes = 120 # 2 seconds
                        df['Diff'] = df['EndFrame'] - df['StartFrame']
                        bool1 = df['Diff'] > min_nframes
                        #bool2 = df['Diff'] <= max_nframes 
                        boolcomb = bool1 #& bool2
                        df = df[boolcomb]
                        return df

                    def filter_gestures(df):
                        min_count = 100
                        gesture_counts = df['Gesture'].value_counts()
                        gestures = gesture_counts[gesture_counts > min_count].index.tolist()
                        df = df[df['Gesture'].isin(gestures)]
                        #self.label_encoder = label_encoder.fit(gestures)
                        return df

                    def balance_gestures(df,category='Gesture'):
                        gestures = df[category].unique().tolist()
                        min_class_amount = df[category].value_counts().min()
                        balanced_df = pd.DataFrame()
                        for gesture in gestures:
                            bool1 = df[category] == gesture
                            curr_df = df[bool1].sample(n=min_class_amount,replace=False,random_state=1)
                            balanced_df = pd.concat((balanced_df,curr_df),axis=0)
                        df = balanced_df.copy()
                        return df

                    def obtain_train_val_split(df,kind='Video',balance=True):
                        """ Split Data for Training and Validation
                        Args:
                            kind (str): type of split. Options: 'Instance' | 'Video'
                            balance (bool): balance both training and validation sets
                        Output:
                            train_df (pd.DataFrame)
                            val_df (pd.DataFrame)
                        """

                        if kind == 'Video':
                            train_df = pd.DataFrame()
                            val_df = pd.DataFrame()
                            test_df = pd.DataFrame()
                            for gesture in sorted(df['Gesture'].unique().tolist()):
                                curr_df = df[df['Gesture'] == gesture]
                                vids = curr_df['Video'].unique().tolist()
                                nvids = len(vids)
                                random.seed(fold)
                                train_vids = random.sample(vids,int(0.9*nvids))
                                val_vids = random.sample(train_vids,int(0.1*nvids))
                                #train_vids = list(set(train_vids) - set(val_vids)) #added most recently Feb 26 2022
                                test_vids = list(set(vids) - set(train_vids) - set(val_vids))
                                assert set(train_vids).intersection(set(val_vids)).intersection(set(test_vids)) == set()
                                curr_train_df = curr_df[curr_df['Video'].isin(train_vids)]
                                curr_val_df = curr_df[curr_df['Video'].isin(val_vids)]
                                curr_test_df = curr_df[curr_df['Video'].isin(test_vids)]
                                train_df = pd.concat((train_df,curr_train_df),axis=0)
                                val_df = pd.concat((val_df,curr_val_df),axis=0)
                                test_df = pd.concat((test_df,curr_test_df),axis=0)

                            if balance == True:
                                train_df = balance_gestures(train_df)
                                val_df = balance_gestures(val_df)
                                test_df = balance_gestures(test_df)
                            else: # do not balance Training Set
                                val_df = balance_gestures(val_df)
                                test_df = balance_gestures(test_df)

                        return train_df, val_df, test_df

                    hf_rgb = h5py.File(os.path.join(self.root_path,'NS','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                    hf_of = h5py.File(os.path.join(self.root_path,'NS','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')

                    df = pd.read_csv(os.path.join(self.root_path,'NS','NS_gestures_timestamps.csv'),index_col=0)
                    df['Video'] = df['Path'].apply(lambda path:path.split('\\')[-1])              
                    df = df[~df['Video'].str.contains('P-129')] #video P-129 has frame number mismatch - needs further inspection
                    df['Gesture'] = df['Gesture'].apply(lambda gest:str(gest).strip())

                    if phase == 'Gronau_inference':
                        hf_rgb = h5py.File(os.path.join(self.root_path,'NS_Gronau','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                        hf_of = h5py.File(os.path.join(self.root_path,'NS_Gronau','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')

                        df = pd.read_csv(os.path.join(self.root_path,'NS_Gronau','NS_Gronau_gestures_timestamps.csv'),index_col=0)
                        df['Video'] = df['Path'].apply(lambda path:path.split('\\')[-1])              
                        df['Gesture'] = df['Gesture'].apply(lambda gest:str(gest).strip())

                        """ Only Use Videos Available in Directory """
                        videos_to_keep = list(hf_rgb.keys())
                        df = df[df['Video'].isin(videos_to_keep)]

                        """ Remove Short Videos """
                        df = filter_examples(df)

                        """ Sample Gestures Per PID (Breadth) for Evaluation """
                        gestures = ['p', 'h', 'c', 'r', 'm', 'k']
                        df = df[df['Gesture'].isin(gestures)]

                        all_df = pd.DataFrame()
                        for pid in sorted(df['PID'].unique().tolist()):
                            for gesture in sorted(df['Gesture'].unique().tolist()):
                                bool1 = df['PID']==pid
                                bool2 = df['Gesture']==gesture
                                boolcomb = bool1 & bool2
                                if boolcomb.sum() >= 3:
                                    curr_df = df[boolcomb]
                                    gest_df = curr_df.sample(n=3,replace=False,random_state=1)
                                    all_df = pd.concat((all_df,gest_df),axis=0)

                        self.label_encoder = label_encoder.fit(sorted(gestures)) # b/c not all gestures might be available
                        data = {'Gronau_inference':all_df}
                    else:
                        """ Only Use Videos Available in Directory """
                        videos_to_keep = list(hf_rgb.keys())
                        df = df[df['Video'].isin(videos_to_keep)]
                        df['Domain'] = 'NS'

                        """ Remove Short Videos """
                        df = filter_examples(df)

                        """ Only Keep Gestures with At Least min_count Examples """
                        df = filter_gestures(df)

                        """ Remove Gestures with Notes (Abnormal Behaviour) """
                        df = df[df['Note'].isna()]

                        """ Binary Classification (Retractions vs. Cold Cut) """
                        domain = 'Top6'
                        if 'vs' in domain: #e.g., 'r_vs_c'
                            gestures = domain.split('_vs_')
                            df = df[df['Gesture'].isin(gestures)]
                            split = 'Video'
                        elif domain == 'Top6':
                            gestures = ['p', 'h', 'c', 'r', 'm', 'k'] #enough samples from as many videos as possible
                            df = df[df['Gesture'].isin(gestures)]
                            split = 'Video'
                        elif domain == 'coarse_gestures': #multi-class high-level gesture groups
                            gestures = ['p', 'h', 'c', 'r', 'm', 'g', 'k']
                            df = df[df['Gesture'].isin(gestures)]
                            df['Gesture'] = df['Gesture'].replace(['s', 'p', 'h'],'blunt')
                            df['Gesture'] = df['Gesture'].replace(['c', 'e'],'sharp')
                            df['Gesture'] = df['Gesture'].replace(['r', 'm', 'g', 'k'],'supporting')
                            split = 'Video'

                        """ Obtain Train/Val Split """
                        df_train, df_val, df_test = obtain_train_val_split(df,kind=split,balance=balance)

                        df_train = df_train[['StartFrame','EndFrame','Domain','Video']]
                        df_val = df_val[['StartFrame','EndFrame','Domain','Video']]
                        df_test = df_test[['StartFrame','EndFrame','Domain','Video']]
                        
                        train_data_ns = df_train.copy()
                        val_data_ns = df_val.copy()
                        test_data_ns = df_test.copy()
                        
                        hf_rgb_ns = hf_rgb
                        hf_of_ns = hf_of
                        
                        """ Combine Datasets """
                        train_data = pd.concat((train_data_vua,train_data_ns),axis=0)
                        val_data = pd.concat((val_data_vua,val_data_ns),axis=0)
                        test_data = pd.concat((test_data_vua,test_data_ns),axis=0)
                        
                        train_data = balance_gestures(train_data,category='Domain')
                        val_data = balance_gestures(val_data,category='Domain')
                        test_data = balance_gestures(test_data,category='Domain')
                        print(train_data['Domain'].value_counts())
                        
                        data = {'train':train_data,'val':val_data,'test':test_data}

                    self.data = data
                    self.hf_rgb = {'VUA':hf_rgb_vua,'NS':hf_rgb_ns}
                    self.hf_of = {'VUA':hf_of_vua,'NS':hf_of_ns}
                    
                ### Nerve Sparing Dataset ... ###
                elif self.dataset_name in ['NS_DART']:#,'NS_Gestures_Recommendation']:
                    hf_rgb = h5py.File(os.path.join(self.root_path,'NS','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                    hf_of = h5py.File(os.path.join(self.root_path,'NS','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')
                    dfA = pd.read_csv(os.path.join(self.root_path,'NS','NS_gestures_timestamps.csv'),index_col=0)
                    dfB = pd.read_csv(os.path.join(self.root_path,'NS','NS_USC_inference_gestures.csv'),index_col=0)

                    dfA['Video'] = dfA['Path'].apply(lambda path:path.split('\\')[-1])
                    dfA['Gesture'] = dfA['Gesture'].apply(lambda gest:str(gest).strip())
                    """ Concatenate Gesture Timestamp Dataframes """
                    cols = ['Video','Path','StartFrame','EndFrame','Gesture']
                    dfA = dfA[cols]
                    dfB = dfB[cols]
                    df = pd.concat((dfA,dfB),axis=0)

                    """ Retrieve DART Scores """
                    cols = ['Video','GS1', 'GS2', 'GS3', 'GS4', 'GS5', 'IVA1', 'IVA2', 'IVA3',
                           'IVA4', 'IVA5', 'RTP1', 'RTP2', 'RTP3', 'RTP4', 'RTP5', 'TH1', 'TH2',
                           'TH3', 'TH4', 'TH5', 'TR1', 'TR2', 'TR3', 'TR4', 'TR5', 'E1', 'E2',
                           'E3', 'E4', 'E5']

                    """ DART Batch 1 Scores """
                    dart1 = pd.read_csv(os.path.join(self.root_path,'NS','DART.csv'),index_col=0)
                    dart1['Video'] = dart1.index
                    dart1 = dart1.drop_duplicates(subset=['Video'],keep='last') #2 cases are repeateds
                    dart1['Video'] = dart1['Video'].apply(lambda name:name.split('_')[0] + '_NVB_' + name.split('_')[1] if '_' in name else name.split(' ')[0] + '_NVB_' + name.split(' ')[1])
                    cols_lower = [el.lower() if el != 'Video' else el for el in cols]
                    dart1 = dart1[[col for col in dart1.columns if col in cols_lower]]
                    dart1 = dart1[cols_lower]
                    dart1.columns = cols

                    """ DART Batch 2 Scores """
                    dart2 = pd.read_csv(os.path.join(self.root_path,'NS','DARTv2.csv'),index_col=0)       
                    dart2 = dart2[~dart2.index.str.contains('training')]
                    dart2['Video'] = dart2['Video'].apply(lambda name:name.split('_')[0] + '_NVB_' + name.split('_')[1] if '_' in name else name.split(' ')[0] + '_NVB_' + name.split(' ')[1])
                    dart2 = dart2[cols]

                    """ Merge DART Scores Across Batches """
                    dart = pd.concat((dart1,dart2),axis=0)

                    """ Aggregate DART Scores Across Raters """
                    aggregation_style = 'mean'

                    if aggregation_style == 'maj':
                        dart['AVE_TR'] = dart[['TR%i' % i for i in range(1,6,1)]].apply(lambda scores:list(dict(sorted(Counter(scores[scores.notna()]).items(),key=lambda el:el[1])).keys())[-1],axis=1)
                        dart['AVE_TH'] = dart[['TH%i' % i for i in range(1,6,1)]].apply(lambda scores:list(dict(sorted(Counter(scores[scores.notna()]).items(),key=lambda el:el[1])).keys())[-1],axis=1)
                        dart['AVE_IVA'] = dart[['IVA%i' % i for i in range(1,6,1)]].apply(lambda scores:list(dict(sorted(Counter(scores[scores.notna()]).items(),key=lambda el:el[1])).keys())[-1],axis=1)
                        dart['AVE_RTP'] = dart[['RTP%i' % i for i in range(1,6,1)]].apply(lambda scores:list(dict(sorted(Counter(scores[scores.notna()]).items(),key=lambda el:el[1])).keys())[-1],axis=1)
                        dart['AVE_E'] = dart[['E%i' % i for i in range(1,6,1)]].apply(lambda scores:list(dict(sorted(Counter(scores[scores.notna()]).items(),key=lambda el:el[1])).keys())[-1],axis=1)
                        dart['AVE_GS'] = dart[['GS%i' % i for i in range(1,6,1)]].apply(lambda scores:list(dict(sorted(Counter(scores[scores.notna()]).items(),key=lambda el:el[1])).keys())[-1],axis=1)
                    elif aggregation_style == 'mean':
                        dart['AVE_TR'] = dart[['TR%i' % i for i in range(1,6,1)]].mean(axis=1)
                        dart['AVE_TH'] = dart[['TH%i' % i for i in range(1,6,1)]].mean(axis=1)
                        dart['AVE_IVA'] = dart[['IVA%i' % i for i in range(1,6,1)]].mean(axis=1)
                        dart['AVE_RTP'] = dart[['RTP%i' % i for i in range(1,6,1)]].mean(axis=1)
                        dart['AVE_E'] = dart[['E%i' % i for i in range(1,6,1)]].mean(axis=1)
                        dart['AVE_GS'] = dart[['GS%i' % i for i in range(1,6,1)]].mean(axis=1)

                    def group_dart(score):
                        if score <= 2.6:
                            score = 0
                        elif score <= 2.8:
                            score = 1
                        elif score <= 3.0:
                            score = 2
                        return score

                    """ Group the DART Domain Scores """
                    dart_domain = 'AVE_TR'
                    self.dart_domain = dart_domain
                    dart[dart_domain] = dart[dart_domain].apply(group_dart)
                    dart = dart[dart[dart_domain].isin([0,2])] #only consider extremes
                    label_encoder.fit(dart[dart_domain])
                    self.label_encoder = label_encoder
                    #print(dart['Video'])

                    def filter_examples(df):
                        min_nframes = 10 
                        #max_nframes = 120 # 4 seconds #introducing upper limit may filter out long noisy gestures
                        df['Diff'] = df['EndFrame'] - df['StartFrame']
                        bool1 = df['Diff'] > min_nframes
                        #bool2 = df['Diff'] <= max_nframes 
                        boolcomb = bool1 #& bool2
                        df = df[boolcomb]
                        return df

                    """ Only Use Videos Available in Directory """
                    videos_to_keep = list(hf_rgb.keys())
                    df = df[df['Video'].isin(videos_to_keep)]

                    """ Remove Gestures with Few Frames """
                    df = filter_examples(df)

                    """ Only Consider Specific Gestures """
                    gestures = ['r']
                    df = df[df['Gesture'].isin(gestures)]

                    #if self.training_fraction < 1 and self.phase == 'train':
                    #    nsamples = int(dfg.shape[0] * self.training_fraction)
                    #    dfg = dfg.sample(n=nsamples,random_state=0)

                    """ Only Consider DART Videos with Gestures """
                    dart = dart[dart['Video'].isin(df['Video'].unique())]

                    """ Obtain Train/Val/Test Split of Data """
                    split = 'Video'
                    cases = dart[split].unique().tolist()
                    ncases = len(cases)
                    random.seed(fold)
                    train_cases = random.sample(cases,int(0.9*ncases))
                    val_cases = random.sample(train_cases,int(0.1*ncases))
                    train_cases = list(set(train_cases) - set(val_cases))
                    test_cases = list(set(cases) - set(train_cases) - set(val_cases))
                    assert set(train_cases).intersection(set(val_cases)).intersection(set(test_cases)) == set()
                    train_indices, val_indices, test_indices = np.where(dart[split].isin(train_cases))[0], np.where(dart[split].isin(val_cases))[0], np.where(dart[split].isin(test_cases))[0]
                    dart_train, dart_val, dart_test = dart.iloc[train_indices,:], dart.iloc[val_indices,:],  dart.iloc[test_indices,:]

                    train_data = dart_train.copy()
                    val_data = dart_val.copy()
                    test_data = dart_test.copy()
                    data = {'train':train_data,'val':val_data,'test':test_data}

                    print(dart_train[dart_domain].value_counts())
                    print(dart_val[dart_domain].value_counts())

                    self.df = df
                    self.data = data
                    self.hf_rgb = hf_rgb
                    self.hf_of = hf_of
                elif self.dataset_name in ['NS_Gestures_Classification']:

                    def filter_examples(df):
                        min_nframes = 10
                        #max_nframes = 120 # 2 seconds
                        df['Diff'] = df['EndFrame'] - df['StartFrame']
                        bool1 = df['Diff'] > min_nframes
                        #bool2 = df['Diff'] <= max_nframes 
                        boolcomb = bool1 #& bool2
                        df = df[boolcomb]
                        return df

                    def filter_gestures(df):
                        min_count = 100
                        gesture_counts = df['Gesture'].value_counts()
                        gestures = gesture_counts[gesture_counts > min_count].index.tolist()
                        df = df[df['Gesture'].isin(gestures)]
                        #self.label_encoder = label_encoder.fit(gestures)
                        return df

                    def balance_gestures(df):
                        gestures = df['Gesture'].unique().tolist()
                        min_class_amount = df['Gesture'].value_counts().min()
                        balanced_df = pd.DataFrame()
                        for gesture in gestures:
                            bool1 = df['Gesture'] == gesture
                            curr_df = df[bool1].sample(n=min_class_amount,replace=False,random_state=1)
                            balanced_df = pd.concat((balanced_df,curr_df),axis=0)
                        df = balanced_df.copy()
                        return df

                    def obtain_train_val_split(df,kind='Video',balance=True):
                        """ Split Data for Training and Validation
                        Args:
                            kind (str): type of split. Options: 'Instance' | 'Video'
                            balance (bool): balance both training and validation sets
                        Output:
                            train_df (pd.DataFrame)
                            val_df (pd.DataFrame)
                        """

                        if kind == 'Video':
                            train_df = pd.DataFrame()
                            val_df = pd.DataFrame()
                            test_df = pd.DataFrame()
                            for gesture in sorted(df['Gesture'].unique().tolist()):
                                curr_df = df[df['Gesture'] == gesture]
                                vids = curr_df['Video'].unique().tolist()
                                nvids = len(vids)
                                random.seed(fold)
                                train_vids = random.sample(vids,int(0.9*nvids))
                                val_vids = random.sample(train_vids,int(0.1*nvids))
                                #train_vids = list(set(train_vids) - set(val_vids)) #added most recently Feb 26 2022
                                test_vids = list(set(vids) - set(train_vids) - set(val_vids))
                                assert set(train_vids).intersection(set(val_vids)).intersection(set(test_vids)) == set()
                                curr_train_df = curr_df[curr_df['Video'].isin(train_vids)]
                                curr_val_df = curr_df[curr_df['Video'].isin(val_vids)]
                                curr_test_df = curr_df[curr_df['Video'].isin(test_vids)]
                                train_df = pd.concat((train_df,curr_train_df),axis=0)
                                val_df = pd.concat((val_df,curr_val_df),axis=0)
                                test_df = pd.concat((test_df,curr_test_df),axis=0)

                            if balance == True:
                                train_df = balance_gestures(train_df)
                                val_df = balance_gestures(val_df)
                                test_df = balance_gestures(test_df)
                            else: # do not balance Training Set
                                val_df = balance_gestures(val_df)
                                test_df = balance_gestures(test_df)

                        elif kind == 'Instance':
                            if balance == True:
                                df = balance_gestures(df)

                            nsamples = df.shape[0]
                            ntrain = int(0.8*nsamples)
                            random.seed(0)
                            indices = random.sample(list(range(nsamples)),nsamples)
                            train_indices, val_indices = indices[:ntrain], indices[ntrain:]

                            df_train, df_val = df.iloc[train_indices,:], df.iloc[val_indices,:]
                            train_df = df_train.copy()
                            val_df = df_val.copy()
                            test_df = df_val.copy() # placeholder for now

                        self.label_encoder = label_encoder.fit(train_df['Gesture'])
                        return train_df, val_df, test_df

                    hf_rgb = h5py.File(os.path.join(self.root_path,'NS','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                    hf_of = h5py.File(os.path.join(self.root_path,'NS','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')

                    df = pd.read_csv(os.path.join(self.root_path,'NS','NS_gestures_timestamps.csv'),index_col=0)
                    df['Video'] = df['Path'].apply(lambda path:path.split('\\')[-1])              
                    df = df[~df['Video'].str.contains('P-129')] #video P-129 has frame number mismatch - needs further inspection
                    df['Gesture'] = df['Gesture'].apply(lambda gest:str(gest).strip())

                    if phase == 'USC_inference':
                        all_df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','NS_Paths.csv'),index_col=0)
                        all_df['Video'] = all_df['label']
                        videos_to_exclude = df['Video'].unique() # all videos with gestures
                        videos_to_include = list(set(all_df['Video']) - set(videos_to_exclude)) #remove videos with gestures & only consider videos without gestures
                        df = all_df[all_df['Video'].isin(videos_to_include)]

                        """ Only Use Videos Available in Directory """
                        videos_to_keep = list(hf_rgb.keys())
                        videos_to_keep = sorted(list(set(df['Video']).intersection(set(videos_to_keep))))
                        #single_video_to_keep = [videos_to_keep[0]]
                        df = df[df['Video'].isin(videos_to_keep)]

                        """ Get Start and End Frame (1 second = 30 frame sliding clips) """
                        df = df.groupby(by=['category','label']).apply(lambda rows:rows[0:-1:30]).reset_index(drop=True)
                        new_paths = df.groupby(by=['category','label'])['path'].apply(lambda rows:rows[1:]).reset_index(drop=True) #
                        df = df.groupby(by=['category','label']).apply(lambda rows:rows[:-1]).reset_index(drop=True) #drop last row per case
                        df.index = new_paths.index
                        df['StartFrame'] = df['path'].apply(lambda path:int(path.split('frame_')[1].strip('.jpg')))
                        df['EndFrame'] = new_paths.apply(lambda path:int(path.split('frame_')[1].strip('.jpg')))
                        print(df.shape)
                        data = {'USC_inference':df}
                    elif phase == 'Gronau_inference':
                        hf_rgb = h5py.File(os.path.join(self.root_path,'NS_Gronau','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                        hf_of = h5py.File(os.path.join(self.root_path,'NS_Gronau','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')

                        df = pd.read_csv(os.path.join(self.root_path,'NS_Gronau','NS_Gronau_gestures_timestamps.csv'),index_col=0)
                        df['Video'] = df['Path'].apply(lambda path:path.split('\\')[-1])              
                        df['Gesture'] = df['Gesture'].apply(lambda gest:str(gest).strip())

                        """ Only Use Videos Available in Directory """
                        videos_to_keep = list(hf_rgb.keys())
                        df = df[df['Video'].isin(videos_to_keep)]

                        """ Remove Short Videos """
                        df = filter_examples(df)

                        """ Sample Gestures Per PID (Breadth) for Evaluation """
                        gestures = ['p', 'h', 'c', 'r', 'm', 'k']
                        df = df[df['Gesture'].isin(gestures)]

                        all_df = pd.DataFrame()
                        for pid in sorted(df['PID'].unique().tolist()):
                            for gesture in sorted(df['Gesture'].unique().tolist()):
                                bool1 = df['PID']==pid
                                bool2 = df['Gesture']==gesture
                                boolcomb = bool1 & bool2
                                if boolcomb.sum() >= 3:
                                    curr_df = df[boolcomb]
                                    gest_df = curr_df.sample(n=3,replace=False,random_state=1)
                                    all_df = pd.concat((all_df,gest_df),axis=0)

                        self.label_encoder = label_encoder.fit(sorted(gestures)) # b/c not all gestures might be available
                        data = {'Gronau_inference':all_df}
                    elif phase == 'RAPN_inference':
                        hf_rgb = h5py.File(os.path.join(self.root_path,'RAPN','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                        hf_of = h5py.File(os.path.join(self.root_path,'RAPN','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')

                        df = pd.read_csv(os.path.join(self.root_path,'RAPN','RAPN_gestures_timestamps.csv'),index_col=0)
                        df['Video'] = df['Path'].apply(lambda path:path.split('\\')[-1])            
                        df['Gesture'] = df['Gesture'].apply(lambda gest:str(gest).strip())

                        """ Only Use Videos Available in Directory """
                        videos_to_keep = list(hf_rgb.keys())
                        df = df[df['Video'].isin(videos_to_keep)]

                        """ Remove Short Videos """
                        df = filter_examples(df)

                        """ Sample Gestures Per PID (Breadth) for Evaluation """
                        gestures = ['p', 'h', 'c', 'r', 'm', 'k']
                        df = df[df['Gesture'].isin(gestures)]

                        all_df = pd.DataFrame()
                        for pid in sorted(df['PID'].unique().tolist()):
                            for gesture in sorted(df['Gesture'].unique().tolist()):
                                bool1 = df['PID']==pid
                                bool2 = df['Gesture']==gesture
                                boolcomb = bool1 & bool2
                                if boolcomb.sum() >= 3:
                                    curr_df = df[boolcomb]
                                    gest_df = curr_df.sample(n=3,replace=False,random_state=0)
                                    all_df = pd.concat((all_df,gest_df),axis=0)

                        self.label_encoder = label_encoder.fit(sorted(gestures)) # b/c not all gestures might be available
                        data = {'RAPN_inference':all_df}
                    elif phase == 'CinVivo_inference':
                        df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','CinVivo_Paths.csv'),index_col=0)

                        hf_rgb = h5py.File(os.path.join(self.root_path,'CinVivo','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                        hf_of = h5py.File(os.path.join(self.root_path,'CinVivo','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')

                        countdf = df.groupby(by=['category','label']).count().reset_index()
                        countdf.columns = ['category','label','count']
                        
                        duration = 0.5 # seconds
                        hop = 0.5 # seconds
                        fps = 30 # fps
                        duration_frames = int(duration * fps)
                        hop_frames = int(hop * fps)
                        inference_df = pd.DataFrame()
                        for idx,(category,label,total_frames) in tqdm(countdf.iterrows()):
                            nsamples = (total_frames - duration_frames)//hop_frames + 1
                            startframes = [n * hop_frames for n in range(nsamples)]
                            endframes = [startframe + duration_frames for startframe in startframes]
                            frames_df = pd.DataFrame([startframes,endframes]).T
                            frames_df.columns = ['StartFrame','EndFrame']
                            frames_df[['category','label']] = [category,label]
                            frames_df[['Video','Domain']] = [label,'Gesture']
                            inference_df = pd.concat((inference_df,frames_df),axis=0)
                        df = inference_df.copy()
                        
                        print(df.shape)
                        data = {'CinVivo_inference':df}
                    else:
                        """ Only Use Videos Available in Directory """
                        videos_to_keep = list(hf_rgb.keys())
                        df = df[df['Video'].isin(videos_to_keep)]

                        """ Remove Short Videos """
                        df = filter_examples(df)

                        """ Only Keep Gestures with At Least min_count Examples """
                        df = filter_gestures(df)

                        """ Remove Gestures with Notes (Abnormal Behaviour) """
                        df = df[df['Note'].isna()]

                        """ Binary Classification (Retractions vs. Cold Cut) """
                        print(domain)
                        if 'vs' in domain: #e.g., 'r_vs_c'
                            gestures = domain.split('_vs_')
                            df = df[df['Gesture'].isin(gestures)]
                            #balance = False
                            split = 'Video'
                        elif domain == 'Top6':
                            gestures = ['p', 'h', 'c', 'r', 'm', 'k'] #enough samples from as many videos as possible
                            df = df[df['Gesture'].isin(gestures)]
                            #balance = True
                            split = 'Video'
                        elif domain == 'coarse_gestures': #multi-class high-level gesture groups
                            gestures = ['p', 'h', 'c', 'r', 'm', 'g', 'k']
                            df = df[df['Gesture'].isin(gestures)]
                            df['Gesture'] = df['Gesture'].replace(['s', 'p', 'h'],'blunt')
                            df['Gesture'] = df['Gesture'].replace(['c', 'e'],'sharp')
                            df['Gesture'] = df['Gesture'].replace(['r', 'm', 'g', 'k'],'supporting')
                            #balance = True
                            split = 'Video'

                        """ Obtain Train/Val Split """
                        df_train, df_val, df_test = obtain_train_val_split(df,kind=split,balance=balance)

                        """ Reduce Training Set Size for Faster Iteration """
                        #if self.training_fraction < 1 and self.phase == 'train':
                        #    nsamples = int(df_train.shape[0] * self.training_fraction)
                        #    df_train = df_train.sample(n=nsamples,random_state=0)

                        print(df_train['Gesture'].value_counts())
                        print('# Classes: %i' % len(df_train['Gesture'].unique()))
                        print(df.shape,df_train.shape,df_val.shape)

                        train_data = df_train.copy()
                        val_data = df_val.copy()
                        train_val_data = pd.concat((train_data,val_data),axis=0)
                        test_data = df_test.copy()
                        data = {'train':train_data,'val':val_data,'train+val':train_val_data,'test':test_data}

                    self.data = data
                    self.hf_rgb = hf_rgb
                    self.hf_of = hf_of
                elif self.dataset_name in ['VUA_Gestures_Classification']:

                    def balance_gestures(df):
                        gestures = df['Gesture'].unique().tolist()
                        min_class_amount = df['Gesture'].value_counts().min()
                        balanced_df = pd.DataFrame()
                        for gesture in gestures:
                            bool1 = df['Gesture'] == gesture
                            curr_df = df[bool1].sample(n=min_class_amount,replace=False,random_state=1)
                            balanced_df = pd.concat((balanced_df,curr_df),axis=0)
                        df = balanced_df.copy()
                        return df

                    def obtain_train_val_split(df,kind='Video',balance=True):
                        """ Split Data for Training and Validation
                        Args:
                            kind (str): type of split. Options: 'Instance' | 'Video'
                            balance (bool): balance both training and validation sets
                        Output:
                            train_df (pd.DataFrame)
                            val_df (pd.DataFrame)
                        """

                        if kind == 'Video':
                            train_df = pd.DataFrame()
                            val_df = pd.DataFrame()
                            test_df = pd.DataFrame()
                            for gesture in sorted(df['Gesture'].unique().tolist()):
                                curr_df = df[df['Gesture'] == gesture]
                                vids = curr_df['Video'].unique().tolist()
                                nvids = len(vids)
                                random.seed(fold)
                                train_vids = random.sample(vids,int(0.9*nvids))
                                val_vids = random.sample(train_vids,int(0.1*nvids))
                                #train_vids = list(set(train_vids) - set(val_vids)) #added most recently Feb 26 2022
                                test_vids = list(set(vids) - set(train_vids) - set(val_vids))
                                assert set(train_vids).intersection(set(val_vids)).intersection(set(test_vids)) == set()
                                curr_train_df = curr_df[curr_df['Video'].isin(train_vids)]
                                curr_val_df = curr_df[curr_df['Video'].isin(val_vids)]
                                curr_test_df = curr_df[curr_df['Video'].isin(test_vids)]
                                train_df = pd.concat((train_df,curr_train_df),axis=0)
                                val_df = pd.concat((val_df,curr_val_df),axis=0)
                                test_df = pd.concat((test_df,curr_test_df),axis=0)

                            if balance == True:
                                train_df = balance_gestures(train_df)
                                val_df = balance_gestures(val_df)
                                test_df = balance_gestures(test_df)
                            else: # do not balance Training Set
                                val_df = balance_gestures(val_df)
                                test_df = balance_gestures(test_df)

                        elif kind == 'Instance':
                            if balance == True:
                                df = balance_gestures(df)

                            nsamples = df.shape[0]
                            ntrain = int(0.8*nsamples)
                            random.seed(0)
                            indices = random.sample(list(range(nsamples)),nsamples)
                            train_indices, val_indices = indices[:ntrain], indices[ntrain:]

                            df_train, df_val = df.iloc[train_indices,:], df.iloc[val_indices,:]
                            train_df = df_train.copy()
                            val_df = df_val.copy()
                            test_df = df_val.copy() # placeholder for now

                        self.label_encoder = label_encoder.fit(train_df['Gesture'])
                        return train_df, val_df, test_df

                    hf_rgb = h5py.File(os.path.join(self.root_path,'VUA','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                    hf_of = h5py.File(os.path.join(self.root_path,'VUA','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')

                    df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','VUA_gestures_timestamps.csv'),index_col=0)
                    df['Video'] = df['Path'].apply(lambda path:path.split('\\')[-1])              
                    df['Gesture'] = df['Gesture'].apply(lambda gest:str(gest).strip())
                    df = df[df['Gesture']!='nan']
                    df = df[df['Gesture'].apply(lambda gest: len(gest.split(','))==1)]
                    df['Gesture'] = df['Gesture'].astype(int)

                    """ Only Use Videos Available in Directory """
                    videos_to_keep = list(hf_rgb.keys())
                    df = df[df['Video'].isin(videos_to_keep)]

                    """ Binary Classification (Retractions vs. Cold Cut) """
                    #print(domain)
                    if 'vs' in domain: #e.g., 'r_vs_c'
                        gestures = domain.split('_vs_')
                        df = df[df['Gesture'].isin(gestures)]
                        #balance = False
                        split = 'Video'
                    elif domain == 'Top4':
                        gestures = [1,2,7,13] #enough samples from as many videos as possible
                        df = df[df['Gesture'].isin(gestures)]
                        #balance = True
                        split = 'Video'

                    """ Obtain Train/Val Split """
                    df_train, df_val, df_test = obtain_train_val_split(df,kind=split,balance=balance)

                    print(df_train['Gesture'].value_counts())
                    print('# Classes: %i' % len(df_train['Gesture'].unique()))
                    print(df.shape,df_train.shape,df_val.shape)

                    train_data = df_train.copy()
                    val_data = df_val.copy()
                    train_val_data = pd.concat((train_data,val_data),axis=0)
                    test_data = df_test.copy()
                    data = {'train':train_data,'val':val_data,'train+val':train_val_data,'test':test_data}

                    self.data = data
                    self.hf_rgb = hf_rgb
                    self.hf_of = hf_of
                elif self.dataset_name in ['DVC_UCL_Gestures_Classification']:

                    def balance_gestures(df):
                        gestures = df['Gesture'].unique().tolist()
                        min_class_amount = df['Gesture'].value_counts().min()
                        balanced_df = pd.DataFrame()
                        for gesture in gestures:
                            bool1 = df['Gesture'] == gesture
                            curr_df = df[bool1].sample(n=min_class_amount,replace=False,random_state=1)
                            balanced_df = pd.concat((balanced_df,curr_df),axis=0)
                        df = balanced_df.copy()
                        return df

                    def obtain_train_val_split(df,kind='Video',balance=True):
                        """ Split Data for Training and Validation
                        Args:
                            kind (str): type of split. Options: 'Instance' | 'Video'
                            balance (bool): balance both training and validation sets
                        Output:
                            train_df (pd.DataFrame)
                            val_df (pd.DataFrame)
                        """

                        if kind == 'Video':
                            train_df = pd.DataFrame()
                            val_df = pd.DataFrame()
                            test_df = pd.DataFrame()
                            for gesture in sorted(df['Gesture'].unique().tolist()):
                                curr_df = df[df['Gesture'] == gesture]
                                vids = curr_df['Video'].unique().tolist()
                                nvids = len(vids)
                                random.seed(fold)
                                train_vids = random.sample(vids,int(0.9*nvids))
                                val_vids = random.sample(train_vids,int(0.1*nvids))
                                train_vids = list(set(train_vids) - set(val_vids)) #added most recently Feb 26 2022
                                test_vids = list(set(vids) - set(train_vids) - set(val_vids))
                                assert set(train_vids).intersection(set(val_vids)).intersection(set(test_vids)) == set()
                                curr_train_df = curr_df[curr_df['Video'].isin(train_vids)]
                                curr_val_df = curr_df[curr_df['Video'].isin(val_vids)]
                                curr_test_df = curr_df[curr_df['Video'].isin(test_vids)]
                                train_df = pd.concat((train_df,curr_train_df),axis=0)
                                val_df = pd.concat((val_df,curr_val_df),axis=0)
                                test_df = pd.concat((test_df,curr_test_df),axis=0)

                            if balance == True:
                                train_df = balance_gestures(train_df)
                                val_df = balance_gestures(val_df)
                                test_df = balance_gestures(test_df)
                            else: # do not balance Training Set
                                val_df = balance_gestures(val_df)
                                test_df = balance_gestures(test_df)

                        elif kind == 'Instance':
                            if balance == True:
                                df = balance_gestures(df)

                            nsamples = df.shape[0]
                            ntrain = int(0.8*nsamples)
                            random.seed(0)
                            indices = random.sample(list(range(nsamples)),nsamples)
                            train_indices, val_indices = indices[:ntrain], indices[ntrain:]

                            df_train, df_val = df.iloc[train_indices,:], df.iloc[val_indices,:]
                            train_df = df_train.copy()
                            val_df = df_val.copy()
                            test_df = df_val.copy() # placeholder for now

                        self.label_encoder = label_encoder.fit(train_df['Gesture'])
                        return train_df, val_df, test_df

                    hf_rgb = h5py.File(os.path.join(self.root_path,'DVC_UCL','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                    hf_of = h5py.File(os.path.join(self.root_path,'DVC_UCL','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')

                    df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','DVC_UCL_gestures_timestamps.csv'),index_col=0)
                    df['Video'] = df['Path'].apply(lambda path:path.split('\\')[-1])              
                    df['Gesture'] = df['Gesture'].apply(lambda gest:str(gest).strip())
                    df = df[df['Gesture']!='nan']
                    df = df[df['Gesture'].apply(lambda gest: len(gest.split(','))==1)]
                    df['Gesture'] = df['Gesture'].astype(int)

                    """ Only Use Videos Available in Directory """
                    videos_to_keep = list(hf_rgb.keys())
                    df = df[df['Video'].isin(videos_to_keep)]

                    """ Multi-Class Classification """
                    gestures = [0,1,2,3,4,6,7] # removed 5 b/c not enough samples
                    df = df[df['Gesture'].isin(gestures)]
                    split = 'Video'

                    """ Obtain Train/Val Split """
                    df_train, df_val, df_test = obtain_train_val_split(df,kind=split,balance=balance)

                    print(df_train['Gesture'].value_counts())
                    print('# Classes: %i' % len(df_train['Gesture'].unique()))
                    print(df.shape,df_train.shape,df_val.shape)

                    train_data = df_train.copy()
                    val_data = df_val.copy()
                    train_val_data = pd.concat((train_data,val_data),axis=0)
                    test_data = df_test.copy()
                    data = {'train':train_val_data,'val':test_data,'test':test_data}

                    self.data = data
                    self.hf_rgb = hf_rgb
                    self.hf_of = hf_of
                elif self.dataset_name in ['JIGSAWS_Suturing_Gestures_Classification']:
                    def balance_gestures(df):
                        gestures = df['Gesture'].unique().tolist()
                        min_class_amount = df['Gesture'].value_counts().min()
                        balanced_df = pd.DataFrame()
                        for gesture in gestures:
                            bool1 = df['Gesture'] == gesture
                            curr_df = df[bool1].sample(n=min_class_amount,replace=False,random_state=1)
                            balanced_df = pd.concat((balanced_df,curr_df),axis=0)
                        df = balanced_df.copy()
                        return df

                    def obtain_train_val_split(df,kind='User',balance=True):
                        """ Split Data for Training and Validation
                        Args:
                            kind (str): type of split. Options: 'Instance' | 'Video'
                            balance (bool): balance both training and validation sets
                        Output:
                            train_df (pd.DataFrame)
                            val_df (pd.DataFrame)
                        """
                        if kind == 'User':
                            users = df['Subject'].unique().tolist()
                            test_user = users[fold] # number of folds = number of users
                            other_users = set(users) - set([test_user])
                            random.seed(fold)
                            shuffled_other_users = random.sample(sorted(other_users),len(other_users))
                            val_user = shuffled_other_users[-1]
                            train_users = shuffled_other_users[:-1]
                            #train_users = set(users) - set([val_user]) - set([test_user])
                            assert set(train_users).union(set([val_user])).intersection(set([test_user])) == set()
                            print('Train Users:',train_users)
                            print('Val User:',val_user)
                            print('Test User:',test_user)
                            train_df = df[df['Subject'].isin(train_users)]
                            val_df = df[df['Subject'].isin([val_user])]
                            test_df = df[df['Subject'].isin([test_user])]
                        elif kind == 'SuperTrial':
                            trials = df['SuperTrial'].unique().tolist()
                            test_trial = trials[fold] # number of folds = number of users
                            other_trials = set(trials) - set([test_trial])
                            random.seed(fold)
                            val_trial = random.sample(other_trials,1)[0]
                            train_trials = set(trials) - set([val_trial]) - set([test_trial])
                            train_df = df[df['SuperTrial'].isin(train_trials)]
                            val_df = df[df['SuperTrial'].isin([val_trial])]
                            test_df = df[df['SuperTrial'].isin([test_trial])]
                        elif kind == 'Video':
                            train_df = pd.DataFrame()
                            val_df = pd.DataFrame()
                            test_df = pd.DataFrame()
                            for gesture in sorted(df['Gesture'].unique().tolist()):
                                curr_df = df[df['Gesture'] == gesture]
                                vids = curr_df['Video'].unique().tolist()
                                nvids = len(vids)
                                random.seed(fold)
                                train_vids = random.sample(vids,int(0.9*nvids))
                                val_vids = random.sample(train_vids,int(0.1*nvids))
                                #train_vids = list(set(train_vids) - set(val_vids)) #added most recently Feb 26 2022
                                test_vids = list(set(vids) - set(train_vids) - set(val_vids))
                                assert set(train_vids).intersection(set(val_vids)).intersection(set(test_vids)) == set()
                                curr_train_df = curr_df[curr_df['Video'].isin(train_vids)]
                                curr_val_df = curr_df[curr_df['Video'].isin(val_vids)]
                                curr_test_df = curr_df[curr_df['Video'].isin(test_vids)]
                                train_df = pd.concat((train_df,curr_train_df),axis=0)
                                val_df = pd.concat((val_df,curr_val_df),axis=0)
                                test_df = pd.concat((test_df,curr_test_df),axis=0)

                            if balance == True:
                                train_df = balance_gestures(train_df)
                                val_df = balance_gestures(val_df)
                                test_df = balance_gestures(test_df)
                            else: # do not balance Training Set
                                val_df = balance_gestures(val_df)
                                test_df = balance_gestures(test_df)

                        elif kind == 'Instance':
                            if balance == True:
                                df = balance_gestures(df)

                            nsamples = df.shape[0]
                            ntrain = int(0.8*nsamples)
                            random.seed(0)
                            indices = random.sample(list(range(nsamples)),nsamples)
                            train_indices, val_indices = indices[:ntrain], indices[ntrain:]

                            df_train, df_val = df.iloc[train_indices,:], df.iloc[val_indices,:]
                            train_df = df_train.copy()
                            val_df = df_val.copy()
                            test_df = df_val.copy() # placeholder for now

                        self.label_encoder = label_encoder.fit(train_df['Gesture'])
                        return train_df, val_df, test_df

                    hf_rgb = h5py.File(os.path.join(self.root_path,'JIGSAWS_Suturing','Results','%s_RepsAndLabels.h5' % encoder_params),'r')
                    hf_of = h5py.File(os.path.join(self.root_path,'JIGSAWS_Suturing','Results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')

                    df = pd.read_csv(os.path.join(self.root_path,'SurgicalPaths','JIGSAWS_Suturing_gestures_timestamps.csv'),index_col=0)
                    df['Video'] = df['Path'].apply(lambda path:path.split('\\')[-1])              

                    """ Only Use Videos Available in Directory """
                    videos_to_keep = list(hf_rgb.keys())
                    df = df[df['Video'].isin(videos_to_keep)]

                    """ Multi-Class Classification """
                    gestures = ['G2', 'G3', 'G6', 'G4', 'G8', 'G11', 'G5', 'G1', 'G9', 'G10'] # removed G10 b/c not enough samples
                    df = df[df['Gesture'].isin(gestures)]
                    split = 'User' #options: User | SuperTrial

                    """ Obtain Train/Val Split """
                    df_train, df_val, df_test = obtain_train_val_split(df,kind=split,balance=balance)

                    print(df_train['Gesture'].value_counts())
                    print('# Classes: %i' % len(df_train['Gesture'].unique()))
                    print(df.shape,df_train.shape,df_val.shape)

                    train_data = df_train.copy()
                    val_data = df_val.copy()
                    train_val_data = pd.concat((train_data,val_data),axis=0)
                    test_data = df_test.copy()
                    data = {'train':train_val_data,'val':test_data,'test':test_data} #,'train+val':train_val_data,'test':test_data}

                    self.data = data
                    self.hf_rgb = hf_rgb
                    self.hf_of = hf_of
                elif self.dataset_name in ['Custom_Gestures']:
                    def balance_scores(df,maj_labels):
                        """ Balance Classes """
                        min_class_amount = df['maj'].value_counts().min()
                        balanced_df = pd.DataFrame()
                        for maj_label in maj_labels:
                            curr_df = df[df['maj']==maj_label].sample(n=min_class_amount,replace=False,random_state=0)
                            balanced_df = pd.concat((balanced_df,curr_df),axis=0)
                        df = balanced_df.copy()
                        return df

                    def obtain_train_val_split(df,maj_labels,kind='Video',balance=True):
                        """ Split Data for Training and Validation
                        Args:
                            kind (str): type of split. Options: 'Instance' | 'Video'
                            balance (bool): balance both training and validation sets
                        Output:
                            train_df (pd.DataFrame)
                            val_df (pd.DataFrame)
                        """

                        if kind in ['Sample','Video','File']:
                            cases = df[kind].unique().tolist()
                            ncases = len(cases)
                            random.seed(fold)
                            train_cases = random.sample(cases,int(0.7*ncases))
                            val_cases = random.sample(train_cases,int(0.2*len(train_cases)))
                            train_cases = list(set(train_cases) - set(val_cases))
                            test_cases = list(set(cases) - set(train_cases) - set(val_cases))
                            assert set(train_cases).intersection(set(val_cases)) == set()
                            assert set(train_cases).intersection(set(test_cases)) == set()        
                            train_indices, val_indices, test_indices = np.where(df[kind].isin(train_cases))[0], np.where(df[kind].isin(val_cases))[0], np.where(df[kind].isin(test_cases))[0]
                            df_train, df_val, df_test = df.iloc[train_indices,:], df.iloc[val_indices,:], df.iloc[test_indices,:]

                            if balance == True:
                                df_train = balance_scores(df_train,maj_labels)
                                df_val = balance_scores(df_val,maj_labels)
                                df_test = balance_scores(df_test,maj_labels)
                            elif balance == False: # do not balance train, BUT balance val/test sets
                                df_val = balance_scores(df_val,maj_labels)
                                df_test = balance_scores(df_test,maj_labels)

                        return df_train, df_val, df_test
                    
                    def get_valid_inview_frames(df,frame_dur=60):
                        df['OOV Frames'] = df.apply(lambda row: list(range(row['StartFrame'],row['EndFrame'])),axis=1)
                        oov_frames = df.groupby(by=['Video']).apply(lambda rows:set(rows['OOV Frames'].sum())).reset_index()
                        oov_frames.columns = ['Video','Video OOV Frames']
                        df = df.merge(oov_frames,how='inner',on='Video')
                        df['In Frames'] = df.apply(lambda row:list(set(range(0,row['MaxFrame'])) - row['Video OOV Frames']),axis=1)
                        df.drop(labels=['OOV Frames','Video OOV Frames'],axis=1,inplace=True)

                        from collections import defaultdict
                        dict_of_frames = defaultdict(list)
                        for video in df['Video'].unique():
                            in_frames = df[df['Video']==video]['In Frames'].iloc[0]
                            inflection_indices = list(np.where(pd.Series(in_frames).diff()>1)[0])
                            inflection_indices = [0] + inflection_indices    
                            for i in range(len(inflection_indices)-1):
                                curr_frames = in_frames[inflection_indices[i]:inflection_indices[i+1]]
                                if len(curr_frames) > frame_dur: # only keep contiguous frames that are longer than some duration
                                    dict_of_frames[video].append(curr_frames)
                        df['Contiguous In Frames'] = df['Video'].apply(lambda video:dict_of_frames[video])
                        return df
                    
                    def get_inview_samples(df,frame_dur=60,seed=0):
                        np.random.seed(seed)
                        start_frames = df['Contiguous In Frames'].apply(lambda list_of_frames:np.random.choice(list_of_frames[np.random.choice(list(range(len(list_of_frames))),1).item()][:-frame_dur],1).item())
                        end_frames = start_frames.apply(lambda start_frame:start_frame + frame_dur)
                        new_samples = pd.DataFrame([start_frames,end_frames]).T
                        new_samples.columns = ['StartFrame','EndFrame']
                        new_samples['Gesture'] = 'in-view'
                        cols = ['Video','Path','File','Sample']
                        new_samples[cols] = df[cols]
                        return new_samples
                    
                    hf_rgb = h5py.File(os.path.join(self.root_path,'results','%s_RepsAndLabels.h5' % encoder_params),'r')
                    hf_of = h5py.File(os.path.join(self.root_path,'results','ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5'),'r')
                    
                    if phase == 'Custom_inference':
                        df = pd.read_csv(os.path.join(self.root_path,'paths','Custom_Paths.csv'),index_col=0)
                        
                        countdf = df.groupby(by=['category','label']).count().reset_index()
                        countdf.columns = ['category','label','count']
                        
                        duration = 0.5 # seconds
                        hop = 0.5 # seconds
                        fps = 30 # fps
                        duration_frames = int(duration * fps)
                        hop_frames = int(hop * fps)
                        inference_df = pd.DataFrame()
                        for idx,(category,label,total_frames) in tqdm(countdf.iterrows()):
                            nsamples = (total_frames - duration_frames)//hop_frames + 1
                            startframes = [n * hop_frames for n in range(nsamples)]
                            endframes = [startframe + duration_frames for startframe in startframes]
                            frames_df = pd.DataFrame([startframes,endframes]).T
                            frames_df.columns = ['StartFrame','EndFrame']
                            frames_df[['category','label']] = [category,label]
                            frames_df[['Video','Domain']] = [label,'Gesture']
                            inference_df = pd.concat((inference_df,frames_df),axis=0)
                        df = inference_df.copy()
                        print(df.shape)
                        data = {'Custom_inference':df}

                    self.data = data
                    self.hf_rgb = hf_rgb
                    self.hf_of = hf_of
                
                
        # def load_data(self):
        #         df = obtainPaths(self.root_path,self.dataset_name,self.nclasses,self.domain,self.fold)
        #         if isinstance(df,tuple):
        #             df,df_gestures = df #used with NS dataset
        #             self.df_gestures = df_gestures
        #         df_subset = df[df['Phase'] == self.phase]
        #         #print(self.training_fraction,self.phase)
        #         if self.training_fraction < 1 and self.phase == 'train':
        #             nsamples = int(df_subset.shape[0] * self.training_fraction)
        #             df_subset = df_subset.sample(n=nsamples,random_state=0)
        #         return df_subset
                        
        def __getitem__(self,idx):
                if self.dataset_name == 'SOCAL':
                    if self.data_type == 'raw':
                        videopath = self.df_subset.iloc[idx,:]['VideoPath']
                        domainName = 'success'
                        label = self.df_subset.iloc[idx,:][domainName]
                        label = torch.tensor(label,dtype=torch.long) 
                        snippets, _, _ = self.getData(videopath,idx)
                        videoname = videopath.split('/')[-1]
                    elif self.data_type == 'reps':
                        videopath = '' #placeholders
                        snippets, videoname, label = self.getData(videopath,idx)
                        #videoname = videopath
                elif self.dataset_name == 'NS':
                        videopath = self.df_subset.iloc[idx,:]['VideoPath']
                        if self.domain in ['ESI_12M','ESI_6M','ESI_3M']:
                                domainName = self.domain
                        else:
                                domainName = 'ave_' + self.domain + '_dt'
                        label = self.df_subset.iloc[idx,:][domainName]
                        label = torch.tensor(label,dtype=torch.long)
                        snippets, _, _ = self.getData(videopath,idx)
                        videoname = videopath.split('/')[-1]
                elif self.dataset_name in ['VUA_EASE','VUA_EASE_Stitch',
                                           'NS_Gestures_Classification','VUA_Gestures_Classification',
                                           'NS_DART','NS_Gestures_Recommendation','DVC_UCL_Gestures_Classification',
                                           'JIGSAWS_Suturing_Gestures_Classification','NS_vs_VUA','CinVivo_OutView','Custom_Gestures']:
                    if self.data_type == 'raw':
                        videopath = '' #self.df_subset.iloc[idx,:]['Path']
#                         videopath = videopath.replace('\\','/')
#                         videopath = os.path.join(self.root_path,'VUA',videopath)
#                         domainName = self.domain

#                         label = self.df_subset.iloc[idx,:][domainName]
#                         #if self.nclasses > 2:
#                         label = torch.tensor(label,dtype=torch.long) # [0,1,2,...]
#                         #else:
#                         #       label = torch.tensor(label,dtype=torch.float)
                        snippets, flows, videoname, label, frames_importance, domain = self.getData(videopath,idx)
                        #videoname = videopath.split('/')[-1]
                    elif self.data_type == 'reps':
                        videopath = '' #placeholder
                        snippets, flows, videoname, label, frames_importance, domain = self.getData(videopath,idx)
                        #videoname = videopath.split('/')[-1]
                #print(snippets.shape,label)
                #print(flows.shape)
                return videoname, snippets, flows, label, frames_importance, domain

        def getData(self,videopath,idx):
                
                if self.data_type == 'raw':
                        if self.dataset_name in ['VUA_EASE','VUA_EASE_Stitch']:
                            df = self.data[self.phase]
                            curr_df = df.iloc[idx,:] #df with timestamps, videopath, etc.
                            videoname = curr_df['Video']
                            domain = curr_df['Domain'] #July26
                            curr_dataset = 'VUA' if self.phase in ['train','val','test','USC_inference'] else 'VUA_%s' % (self.phase.split('_inference')[0])
                            
                            if self.dataset_name == 'VUA_EASE':
                                label = self.label_encoder.transform([curr_df['maj']]).item() # 0, 1, nclasses
                                # new for multi-task learning
                                if '+' in self.domain: # multi-task paradigm, self.domain = 'NH_02+ND_02'
                                    label = label + 2 if curr_df['Domain'] == 'ND_02' else label # increment labels by 2 if ND (assuming in second order)
                                    
                                label = torch.tensor(label,dtype=torch.long)
                                race = curr_df['RACE']
                                if race == 'Needle Withdrawal':
                                    colStartName = 'Needle Withdrawal Start Frame'
                                    colEndName = 'Needle Withdrawal End Frame'
                                elif race == 'Needle Handling':
                                    colStartName = 'Needle Handling Start Frame'
                                    colEndName = 'Needle Entry Start Frame'
                                elif race == 'Needle Driving':
                                    colStartName = 'Needle Entry Start Frame'
                                    colEndName = 'Needle Withdrawal Start Frame'
                                startIdx = curr_df[colStartName]#-1 b/c dealing with frames not indices
                                endIdx = curr_df[colEndName]#-1 b/c dealing with frames not indices
                            elif self.dataset_name == 'VUA_EASE_Stitch':
                                if self.phase == 'USC_inference': #b/c we do not have ground-truth labels here
                                    label = torch.tensor(0,dtype=torch.long) # placeholder - I just need some integer
                                    colStartName, colEndName = 'StartFrame', 'EndFrame'
                                    startIdx = curr_df[colStartName]#-1 if curr_df[colStartName] != 0 else curr_df[colStartName]
                                    endIdx = curr_df[colEndName]-10 #-1
                                else:
                                    label = self.label_encoder.transform([curr_df['EASE']]).item()
                                    label = torch.tensor(label,dtype=torch.long)
                                    race = curr_df['RACE']
                                    if race == 'Needle Withdrawal':
                                        colStartName = 'Needle Withdrawal Start Frame'
                                        colEndName = 'Needle Withdrawal End Frame'
                                    elif race == 'Needle Handling':
                                        colStartName = 'Needle Handling Start Frame'
                                        colEndName = 'Needle Entry Start Frame'
                                    elif race == 'Needle Driving':
                                        colStartName = 'Needle Entry Start Frame'
                                        colEndName = 'Needle Withdrawal Start Frame'
                                    startIdx = curr_df[colStartName]#-1 b/c dealing with frames not indices
                                    endIdx = curr_df[colEndName]#-1 b/c dealing with frames not indices
            
                            if self.phase == 'USC_inference':
                                diff = endIdx - startIdx
                                jump_size = diff//16
                                indices = np.arange(startIdx,endIdx,jump_size)[:16]
                                offset2 = 3
                                offset3 = 6
                                indices2 = list(np.arange(startIdx+offset2,endIdx,jump_size))[:16]
                                indices3 = list(np.arange(startIdx+offset3,endIdx,jump_size))[:16]
                            elif self.phase in ['val','test'] or 'inference' in self.phase: # e.g., HMH_inference
                                if race == 'Needle Withdrawal':
                                    start, end = startIdx-40, startIdx+40
                                    diff = end - start
                                    jump_size = diff//16
                                    indices = np.arange(start,end,jump_size)
                                    #jump_size = int((endIdx - startIdx)//10)
                                    #start, end = startIdx, endIdx
                                    #indices = np.arange(start,end,jump_size)
                                    #indices = np.arange(startIdx-40,startIdx+40,10) #looking back is fine to ensure you capture actual withdrawal
                                elif race == 'Needle Handling':
                                    diff = endIdx - startIdx
                                    # ORIGINAL
                                    frames_to_drop = int(diff * 0.20)
                                    start, end = startIdx, endIdx-frames_to_drop
                                    diff = end - start
                                    # MODIFIED to take only 16 frames (similar to SOTA paper)
                                    jump_size = diff//16
                                    indices = np.arange(start,end,jump_size)
                                    #indices = np.arange(startIdx,endIdx-20,10) #do not look back, which may leak into withdrawal from previous stitch
                                elif race == 'Needle Driving':
                                    diff = endIdx - startIdx
                                    # ORIGINAL
                                    frames_to_drop = int(diff * 0.20)
                                    start, end = startIdx, endIdx-frames_to_drop
                                    diff = end - start
                                    # MODIFIED to take only 16 frames (similar to SOTA paper)
                                    jump_size = diff//16
                                    indices = np.arange(startIdx,endIdx,jump_size) #do not look too forward, which may leak into withdrawal of same stitch
                                offset2 = 3
                                offset3 = 6
                                indices2 = list(np.arange(startIdx+offset2,endIdx+offset2,jump_size))[:16]
                                indices3 = list(np.arange(startIdx+offset3,endIdx+offset3,jump_size))[:16]
                            elif self.phase == 'train':
                                if race == 'Needle Withdrawal':
                                    start, end = startIdx-40, startIdx+40
                                    diff = end - start
                                    jump_size = diff//16
                                    indices = np.arange(start,end,jump_size)
                                    #jump_size = int((endIdx - startIdx)//10)
                                    #indices = np.arange(startIdx,endIdx,jump_size)
                                    #indices = np.arange(startIdx-40,startIdx+40,10) #looking back is fine to ensure you capture actual withdrawal
                                elif race == 'Needle Handling':
                                    diff = endIdx - startIdx
                                    # ORIGINAL
                                    frames_to_drop = int(diff * 0.20)
                                    start, end = startIdx, endIdx-frames_to_drop
                                    diff = end - start
                                    # MODIFIED to take only 16 frames (similar to SOTA paper)
                                    jump_size = diff//16
                                    indices = np.arange(start,end,jump_size)
                                    #indices = np.arange(startIdx,endIdx-20,10) #do not look back, which may leak into withdrawal from previous stitch
                                elif race == 'Needle Driving':
                                    diff = endIdx - startIdx
                                    # ORIGINAL
                                    frames_to_drop = int(diff * 0.20)
                                    start, end = startIdx, endIdx-frames_to_drop
                                    diff = end - start
                                    # MODIFIED to take only 16 frames (similar to SOTA paper)
                                    jump_size = diff//16
                                    indices = np.arange(start,end,jump_size) #do not look too forward, which may leak into withdrawal of same stitch                       
                            indices = indices[:16] # to avoid off by 1
                            if self.encoder_type == 'R3D':
                                mean = [0.43216, 0.394666, 0.37645]
                                std = [0.22803, 0.22145, 0.216989]
                            elif self.encoder_type == 'I3D':
                                mean = [0.485, 0.456, 0.406]
                                std = [0.229, 0.224, 0.225]
                            
                            def loadImages(indices,modality='RGB'):
                                """ Load RGB Images """
                                foldername = 'Images' if modality == 'RGB' else 'Flows'
                                framename = 'frames' if modality == 'RGB' else 'flows'
                                frame_numbers = list(map(lambda idx:idx+1,indices)) # offset by 1
                                paths = list(map(lambda frame:os.path.join(self.root_path,curr_dataset,foldername,videoname,'%s_%s' % (framename,('0'*(8-len(str(frame))) + str(frame) + '.jpg'))),frame_numbers))
                                frames = [np.asarray(Image.open(path)) for path in paths] # height x width x channels
                                snippets = torch.stack([torchvision.transforms.ToTensor()(frame) for frame in frames]) # nframes x channels x height x width
                                #print(snippets[0,0,:].max())
                                #frames = np.stack(frames)
                                #snippets = torch.tensor(frames,dtype=torch.float) # nframes x height x width x channels
                                return snippets
                            
                            def processImages(snippets):
                                """ Process RGB Images """
                                im_height, im_width = snippets.shape[2], snippets.shape[3]
                                height, width = int(0.8*im_height), int(0.8*im_width)
                                snippets = snippets.permute(1,0,2,3) # channels x nframes x height x width
                                snippets = torchvision.transforms.functional.center_crop(snippets,[height,width])
                                snippets = torchvision.transforms.functional.resize(snippets,[self.width,self.width]) # channels x nframes x new_height x new_width
                                snippets = snippets.permute(1,0,2,3) # nframes x nchannels x new_height x new_width
                                snippets = torchvision.transforms.functional.normalize(snippets,mean,std)
                                snippets = snippets.permute(1,0,2,3) # channels x nframes x new_height x new_width
                                snippets = snippets.unsqueeze(0)
                                return snippets

                            snippets = loadImages(indices,'RGB')
                            snippets = processImages(snippets)
                            """ Placeholder - Not Used with C3D """
                            frames_importance = torch.zeros(1,snippets.shape[1],dtype=torch.float)
                            
                            """ Load Flow Images """
                            if 'inference' in self.phase:
                                if self.phase == 'Gronau_inference':
                                    jump_size = 15 #Gronau VUA videos are of a single fps (30)
                                elif self.phase == 'HMH_inference':
                                    dataset = 'VUA_HMH'
                                    jump_size = int(fps_dict[dataset][videoname]//2) 
                                elif self.phase in ['Lab_inference','AFB_inference']:
                                    jump_size = 30 
                                elif self.phase == 'USC_inference':
                                    #dataset = 'VUA'
                                    #jump_size = int(fps_dict[dataset][videoname]//2)
                                    jump_size = 10 # I only looked at videos with fps = 20
                            else:
                                dataset = 'VUA'
                                jump_size = int(fps_dict[dataset][videoname]//2)
                            #jump_size = int(fps_dict[videoname]//2) # each video had a slightly different fps, which was used to generate the flows
                            flow_indices = list(map(lambda idx:idx//jump_size,indices)) # you need fps_dict here to make sure right loading of flows
                            # REMOVED TO ENABLE STACKING oF SAMPLES WITHIN BATCH
                            #flow_indices = np.unique(flow_indices) #to avoid repeating frames and thus prevent overfitting
                            flows = loadImages(flow_indices,'Flow')
                            flows = processImages(flows)
                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                """ Snippets TTA  """
                                snippets2 = loadImages(indices2,'RGB')
                                snippets2 = processImages(snippets2)
                                snippets3 = loadImages(indices3,'RGB')
                                snippets3 = processImages(snippets3)
                                """ Flows TTA """
                                flow_indices2 = list(map(lambda idx:idx//jump_size,indices2)) # you need fps_dict here to make sure right loading of flows
                                flows2 = loadImages(flow_indices2,'Flow')
                                flows2 = processImages(flows2)
                                """ Load Flows """
                                flow_indices3 = list(map(lambda idx:idx//jump_size,indices3)) # you need fps_dict here to make sure right loading of flows
                                flows3 = loadImages(flow_indices3,'Flow')
                                flows3 = processImages(flows3) # nsnippets x channels x nframes x new_height x new_width 

                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                return (snippets, snippets2, snippets3), (flows, flows2, flows3), videoname, label, frames_importance, domain
                        
                        if self.dataset_name in ['SOCAL','NS']:
                        
                                if self.dataset_name == 'SOCAL':
                                        nframes = self.df_subset.iloc[idx,:]['nframes']
                                elif self.dataset_name == 'NS':
                                        """ Path 1 - Naive = Take All Clips from Video """
                                        nframes = self.df_subset.iloc[idx,:]['nframes']
                                        """ Path 2 - Take Gesture-Specific Clips from Video """
                                        df_gestures = self.df_gestures
                                        paths_df = df_gestures['Path'].apply(lambda path:path.split('/')[-1].split('\\')[-1])
                                        videoname = videopath.split('/')[-1]
                                        #print(paths_df)
                                        #print('hello')
                                        #print(videoname)
                                        df_gestures = df_gestures[paths_df.isin([videoname])]
                                        df_gestures = df_gestures[df_gestures['Gesture'].isin(['s','r','m','c','p'])]
                                        #######
                                        # print(df_gestures)
                                        #print(videoname)
                                        snippets = []
                                        """ Iterative Over Gestures """
                                        for index,row in df_gestures.iterrows():
                                                startFrame = row['StartFrame'] # with original sampling rate
                                                endFrame = row['EndFrame']
                                                startFrameIdx = startFrame
                                                endFrameIdx = endFrame
                                                """ Iterate Within Gesture """
                                                for startFrame in range(startFrameIdx,endFrameIdx-30,30):
                                                        endFrame = startFrame + 30
                                                        frames = []
                                                        """ Iterate Over Frames within Gesture Clip """
                                                        for frameNumber in range(startFrame,endFrame,2): # 2 means downsampling 2x #range(1,self.snippetLength+1):
                                                                #frameNumber += 1 #b/c it starts with 1 in the data directory
                                                                framename = 'frame_' + ('0' * (8-len(str(frameNumber)))) + str(frameNumber) + '.jpg' 
                                                                framepath = os.path.join(videopath,framename)
                                                                frame = np.asarray(Image.open(framepath)) # height x width x channels
                                                                frames.append(frame)
                                                        frames = np.stack(frames)
                                                        snippet = torch.tensor(frames,dtype=torch.float) # snippetLength x height x width x channels
                                                        
                                                        im_height, im_width = snippet.shape[1], snippet.shape[2]
                                                        height, width = int(0.8*im_height), int(0.8*im_width)
                                                        snippet = snippet.permute(3,0,1,2) # channels x snippetLength x height x width
                                                        snippet = torchvision.transforms.functional.center_crop(snippet,[height,width])
                                                        snippet = torchvision.transforms.functional.resize(snippet,[self.width,self.width]) # channels x snippetLength x new_height x new_width
                                                        
                                                        snippets.append(snippet)

                                """ Path 1 Cntd. - Deactivate if Part 2 is On """
                                frameIndices = np.arange(0,nframes,self.frameSkip)
                                nframes = len(frameIndices)
                                if self.overlap == 0:
                                        jump = self.snippetLength
                                else:
                                        jump = int(self.snippetLength * self.overlap)
                                startIndices = np.arange(0,nframes,jump) #list of indices for start frame
                                
                                if self.task == 'AoT':
                                        if len(startIndices) > 1:
                                                sampleIdx = random.sample(list(range(len(startIndices)-1)),1)[0]
                                                startIndices = startIndices[sampleIdx:sampleIdx+1]
                                        else:
                                                startIndices = startIndices
                                elif self.task == 'MIL':
                                        startIndices = startIndices
                                elif self.task == 'FeatureExtraction':
                                        startIndices = startIndices
                                
                                snippets = []
                                #snippets_flipped = []
                                for startIndex in startIndices:
                                        videoIndices = frameIndices[startIndex:startIndex+self.snippetLength]
                                        if len(videoIndices) == self.snippetLength: #avoids final snippet if it is shorter than snippetLength
                                                if self.dataset_name == 'SOCAL':
                                                        frames = []
                                                        for frameNumber in videoIndices: #range(1,self.snippetLength+1):
                                                                frameNumber += 1 #b/c it starts with 1 in the data directory
                                                                framename = '_frame_' + ('0' * (8-len(str(frameNumber)))) + str(frameNumber) + '.jpeg' 
                                                                framepath = videopath + framename
                                                                frame = np.asarray(Image.open(framepath))
                                                                frames.append(frame)
                                                        frames = np.stack(frames)
                                                        snippet = torch.tensor(frames,dtype=torch.float) # snippetLength x height x width x channels
                                                        #top,left,height,width = 0,400,1080,1200
                                                        im_height, im_width = snippet.shape[1], snippet.shape[2]
                                                        height, width = int(0.8*im_height), int(0.5*im_width)
                                                elif self.dataset_name == 'NS':
                                                        #snippet = torch.tensor(np.asarray(video[videoIndices]),dtype=torch.float) # snippetLength x height x width x channels
                                                        frames = []
                                                        for frameNumber in videoIndices: #range(1,self.snippetLength+1):
                                                                frameNumber += 1 #b/c it starts with 1 in the data directory
                                                                framename = 'frame_' + ('0' * (8-len(str(frameNumber)))) + str(frameNumber) + '.jpg' 
                                                                framepath = os.path.join(videopath,framename)
                                                                frame = np.asarray(Image.open(framepath)) # height x width x channels
                                                                frames.append(frame)
                                                        frames = np.stack(frames)
                                                        snippet = torch.tensor(frames,dtype=torch.float) # snippetLength x height x width x channels
                                                        #top,left,height,width = 50,100,450,750
                                                        im_height, im_width = snippet.shape[1], snippet.shape[2]
                                                        height, width = int(0.8*im_height), int(0.8*im_width)
                                                
                                                snippet = snippet.permute(3,0,1,2) # channels x snippetLength x height x width
                                                #snippet = torchvision.transforms.functional.crop(snippet,top,left,height,width)
                                                snippet = torchvision.transforms.functional.center_crop(snippet,[height,width])
                                                snippet = torchvision.transforms.functional.resize(snippet,[self.width,self.width]) # channels x snippetLength x new_height x new_width
                                                snippets.append(snippet)
                                                
                                                #snippet_flipped = torchvision.transforms.functional.rotate(snippet,180)
                                                #snippets_flipped.append(snippet_flipped)
                                """ End of Path 1 Cntd. """

                                snippets = self.normalizeSnippets(snippets)
                                videoname = 'None'
                                label = 'None'

                        elif self.dataset_name in ['NS_Gestures_Classification','NS_DART','NS_Gestures_Recommendation']:
                                startFrame = self.df_subset.iloc[idx,:]['StartFrame'] # with original sampling rate
                                endFrame = self.df_subset.iloc[idx,:]['EndFrame']
                                startFrameIdx = startFrame
                                endFrameIdx = endFrame

                                #print(startFrame,endFrame) 
                                #rangeFrame = list(range(startFrame,endFrame-30))
                                #startFrame = random.sample(rangeFrame,1)[0]
                                #endFrame = startFrame + 30 # ensures 1 second of video at fps=30 

                                snippets = []
                                for startFrame in range(startFrameIdx,endFrameIdx,30):
                                        endFrame = startFrame + 30 
                                        frames = []
                                        for frameNumber in range(startFrame,endFrame,2): #2 means downsample 2x   #range(1,self.snippetLength+1):
                                                #frameNumber += 1 #b/c it starts with 1 in the data directory
                                                framename = 'frame_' + ('0' * (8-len(str(frameNumber)))) + str(frameNumber) + '.jpg' 
                                                framepath = os.path.join(videopath,framename)
                                                frame = np.asarray(Image.open(framepath)) # height x width x channels
                                                frames.append(frame)
                                        frames = np.stack(frames)
                                        snippet = torch.tensor(frames,dtype=torch.float) # snippetLength x height x width x channels
                                        #top,left,height,width = 50,100,450,750
                                        
                                        im_height, im_width = snippet.shape[1], snippet.shape[2]
                                        height, width = int(0.8*im_height), int(0.8*im_width)
                                
                                        snippet = snippet.permute(3,0,1,2) # channels x snippetLength x height x width
        #                               snippet = torchvision.transforms.functional.crop(snippet,top,left,height,width)
                                        snippet = torchvision.transforms.functional.center_crop(snippet,[height,width])
                                        snippet = torchvision.transforms.functional.resize(snippet,[self.width,self.width]) # channels x snippetLength x new_height x new_width                                 
                                        snippets.append(snippet)
                                        
                                snippets = self.normalizeSnippets(snippets)
                                videoname = 'None'
                                label = 'None'
                
                elif self.data_type == 'reps': 
                        if self.dataset_name == 'VUA_EASE':
                            df = self.data[self.phase]
                            curr_df = df.iloc[idx,:] #df with timestamps, videopath, etc.
                            videoname = curr_df['Video']
                            domain = curr_df['Domain'] #July26
                            
                            label = self.label_encoder.transform([curr_df['maj']]).item() # 0, 1, nclasses
                            # new for multi-task learning
                            if '+' in self.domain: # multi-task paradigm, self.domain = 'NH_02+ND_02'
                                label = label + 2 if curr_df['Domain'] == 'ND_02' else label # increment labels by 2 if ND (assuming in second order)
                                
                            label = torch.tensor(label,dtype=torch.long)
                            race = curr_df['RACE']
                            if race == 'Needle Withdrawal':
                                colStartName = 'Needle Withdrawal Start Frame'
                                colEndName = 'Needle Withdrawal End Frame'
                            elif race == 'Needle Handling':
                                colStartName = 'Needle Handling Start Frame'
                                colEndName = 'Needle Entry Start Frame'
                            elif race == 'Needle Driving':
                                colStartName = 'Needle Entry Start Frame'
                                colEndName = 'Needle Withdrawal Start Frame'
                            startIdx = curr_df[colStartName]-1
                            endIdx = curr_df[colEndName]-1
                            
                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                if race == 'Needle Withdrawal':
                                    jump_size = int((endIdx - startIdx)//10)
                                    start, end = startIdx, endIdx
                                    indices = np.arange(start,end,jump_size)
                                    #indices = np.arange(startIdx-40,startIdx+40,10) #looking back is fine to ensure you capture actual withdrawal
                                elif race == 'Needle Handling':
                                    diff = endIdx - startIdx
                                    frames_to_drop = int(diff * 0.20)
                                    start, end = startIdx, endIdx-frames_to_drop
                                    if self.phase == 'AFB_inference':
                                        jump_size = 120 # low-pass filter to get rid of high frequency movements
                                    else:
                                        jump_size = 10
                                    indices = np.arange(start,end,jump_size) # 10
                                    #indices = np.arange(startIdx,endIdx-20,10) #do not look back, which may leak into withdrawal from previous stitch
                                elif race == 'Needle Driving':
                                    diff = endIdx - startIdx
                                    frames_to_drop = int(diff * 0.20)
                                    start, end = startIdx, endIdx-frames_to_drop
                                    jump_size = 10
                                    indices = np.arange(start,end,jump_size) #do not look too forward, which may leak into withdrawal of same stitch # 10
                                offset2 = 3
                                offset3 = 6
                                indices2 = list(np.arange(start+offset2,end+offset2,jump_size)) # 10
                                indices3 = list(np.arange(start+offset3,end+offset3,jump_size)) # 10
                            elif self.phase == 'train':
                                if race == 'Needle Withdrawal':
                                    jump_size = int((endIdx - startIdx)//10)
                                    indices = np.arange(startIdx,endIdx,jump_size)
                                    #indices = np.arange(startIdx-40,startIdx+40,10) #looking back is fine to ensure you capture actual withdrawal
                                elif race == 'Needle Handling':
                                    diff = endIdx - startIdx
                                    frames_to_drop = int(diff * 0.20)
                                    indices = np.arange(startIdx,endIdx-frames_to_drop,10)
                                    #indices = np.arange(startIdx,endIdx-20,10) #do not look back, which may leak into withdrawal from previous stitch
                                elif race == 'Needle Driving':
                                    diff = endIdx - startIdx
                                    frames_to_drop = int(diff * 0.20)
                                    indices = np.arange(startIdx,endIdx-frames_to_drop,10) #do not look too forward, which may leak into withdrawal of same stitch                         

                            """ Load RGB Representations """
                            video_reps = np.array(self.hf_rgb.get(videoname))
                            snippets = video_reps[indices,:] # nIndices x D
                            snippets = torch.tensor(snippets,dtype=torch.float)
                            snippets = snippets.unsqueeze(0) # 1 x nframes x D
                            #print(curr_df,startIdx,endIdx)
                            """ Frame Importance Stuff """
                            if self.phase in ['train']:
                                if label == torch.tensor(0,dtype=torch.long): # only applies to low-skill activity
                                    if self.importance_loss == True:
                                        frames_importance = curr_df['frame importance'] # list of frame importances
                                        frames_importance = torch.tensor(frames_importance,dtype=torch.float) # convert to tensor of ints
                                        frames_importance = frames_importance.unsqueeze(0) # 1 x nframes
                                    else:
                                        frames_importance = torch.zeros(1,snippets.shape[1],dtype=torch.float) # placeholder    
                                else:
                                    frames_importance = torch.zeros(1,snippets.shape[1],dtype=torch.float) # placeholder # change to snippets length for max-length purposes
                            else: # we do not use frames_importance other than during training
                                frames_importance = torch.zeros(1,snippets.shape[1],dtype=torch.float)
                            
                            """ Load Flow Representations """
                            if 'inference' in self.phase:
                                if self.phase == 'Gronau_inference':
                                    jump_size = 15 #Gronau VUA videos are of a single fps (30)
                                elif self.phase == 'HMH_inference':
                                    dataset = 'VUA_HMH'
                                    jump_size = int(fps_dict[dataset][videoname]//2) 
                                elif self.phase in ['Lab_inference','AFB_inference']:
                                    jump_size = 30 
                            else:
                                dataset = 'VUA'
                                jump_size = int(fps_dict[dataset][videoname]//2)
                            #jump_size = int(fps_dict[videoname]//2) # each video had a slightly different fps, which was used to generate the flows
                            flow_indices = list(map(lambda idx:idx//jump_size,indices)) # you need fps_dict here to make sure right loading of flows
                            flow_indices = np.unique(flow_indices) #to avoid repeating frames and thus prevent overfitting
                            flow_reps = np.array(self.hf_of.get(videoname))
                            #print(videoname,curr_df,len(flow_reps),flow_indices)
                            flows = flow_reps[flow_indices,:]
                            flows = torch.tensor(flows,dtype=torch.float)
                            flows = flows.unsqueeze(0) # 1 x nflows x D
                    
                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                snippets2 = video_reps[indices2,:] # nframes x D  # [startIdx:endIdx,:] (def)
                                snippets2 = torch.tensor(snippets2,dtype=torch.float)
                                snippets2 = snippets2.unsqueeze(0) # 1 x nframes x D

                                snippets3 = video_reps[indices3,:] # nframes x D  # [startIdx:endIdx,:] (def)
                                snippets3 = torch.tensor(snippets3,dtype=torch.float)
                                snippets3 = snippets3.unsqueeze(0) # 1 x nframes x D
                    
                                flow_indices2 = list(map(lambda idx:idx//jump_size,indices2)) # 15 b/c of 30 fps of NS videos
                                flow_indices2 = np.unique(flow_indices2) 
                                flows2 = flow_reps[flow_indices2,:] # 1 x D
                                flows2 = torch.tensor(flows2,dtype=torch.float)
                                flows2 = flows2.unsqueeze(0)
                            
                                flow_indices3 = list(map(lambda idx:idx//jump_size,indices3)) # 15 b/c of 30 fps of NS videos
                                flow_indices3 = np.unique(flow_indices3)
                                flows3 = flow_reps[flow_indices3,:] # 1 x D
                                flows3 = torch.tensor(flows3,dtype=torch.float)
                                flows3 = flows3.unsqueeze(0)
                        
                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                return (snippets, snippets2, snippets3), (flows, flows2, flows3), videoname, label, frames_importance, domain
                            
                        elif self.dataset_name == 'VUA_EASE_Stitch':
                            df = self.data[self.phase]
                            curr_df = df.iloc[idx,:] #df with timestamps, videopath, etc.
                            videoname = curr_df['Video']
                            domain = curr_df['Domain'] #July26
                            
                            if self.phase == 'USC_inference': #b/c we do not have ground-truth labels here
                                label = torch.tensor(0,dtype=torch.long) # placeholder - I just need some integer
                                colStartName, colEndName = 'StartFrame', 'EndFrame'
                                startIdx = curr_df[colStartName]-1 if curr_df[colStartName] != 0 else curr_df[colStartName]
                                endIdx = curr_df[colEndName]-1
                            else:
                                label = self.label_encoder.transform([curr_df['EASE']]).item()
                                label = torch.tensor(label,dtype=torch.long)

                                race = curr_df['RACE']
                                if race == 'Needle Withdrawal':
                                    colStartName = 'Needle Withdrawal Start Frame'
                                    colEndName = 'Needle Withdrawal End Frame'
                                elif race == 'Needle Handling':
                                    colStartName = 'Needle Handling Start Frame'
                                    colEndName = 'Needle Entry Start Frame'
                                elif race == 'Needle Driving':
                                    colStartName = 'Needle Entry Start Frame'
                                    colEndName = 'Needle Withdrawal Start Frame'
                                startIdx = curr_df[colStartName]-1
                                endIdx = curr_df[colEndName]-1
                            
                            if self.phase in ['val','test']:
                                if race == 'Needle Withdrawal':
                                    start, end = startIdx-40, startIdx+40
                                    indices = np.arange(start,end,10) #looking back is fine to ensure you capture actual withdrawal
                                elif race == 'Needle Handling':
                                    start, end = startIdx, endIdx-20
                                    indices = np.arange(start,end,10) #do not look back, which may leak into withdrawal from previous stitch
                                elif race == 'Needle Driving':
                                    diff = endIdx - startIdx
                                    frames_to_drop = int(diff * 0.20)
                                    start, end = startIdx, endIdx-frames_to_drop
                                    indices = np.arange(start,end,10) #do not look too forward, which may leak into withdrawal of same stitch
                                offset2 = 3
                                offset3 = 6
                                indices2 = list(np.arange(start+offset2,end+offset2,10))
                                indices3 = list(np.arange(start+offset3,end+offset3,10))
                            elif self.phase == 'USC_inference':
                                indices = np.arange(startIdx,endIdx,10)
                                offset2 = 3
                                offset3 = 6
                                indices2 = list(np.arange(startIdx+offset2,endIdx+offset2,10))
                                indices3 = list(np.arange(startIdx+offset3,endIdx+offset3,10))
                            elif 'inference' in self.phase:
                                if race == 'Needle Withdrawal':
                                    start, end = startIdx, startIdx+60 #startIdx-40, startIdx+40
                                    indices = np.arange(start,end,10) #looking back is fine to ensure you capture actual withdrawal
                                elif race == 'Needle Handling':
                                    start, end = startIdx, endIdx#-20
                                    indices = np.arange(start,end,10) #do not look back, which may leak into withdrawal from previous stitch
                                elif race == 'Needle Driving':
                                    diff = endIdx - startIdx
                                    #frames_to_drop = int(diff * 0.20)
                                    start, end = startIdx, endIdx#-frames_to_drop
                                    indices = np.arange(start,end,10) #do not look too forward, which may leak into withdrawal of same stitch
                                offset2 = 3
                                offset3 = 6
                                indices2 = list(np.arange(start+offset2,end+offset2,10))
                                indices3 = list(np.arange(start+offset3,end+offset3,10))
                            elif self.phase == 'train':
                                if race == 'Needle Withdrawal':
                                    indices = np.arange(startIdx-40,startIdx+40,10) #looking back is fine to ensure you capture actual withdrawal
                                elif race == 'Needle Handling':
                                    indices = np.arange(startIdx,endIdx-20,10) #do not look back, which may leak into withdrawal from previous stitch
                                elif race == 'Needle Driving':
                                    diff = endIdx - startIdx
                                    frames_to_drop = int(diff * 0.20)
                                    indices = np.arange(startIdx,endIdx-frames_to_drop,10) #do not look too forward, which may leak into withdrawal of same stitch
                            
                            """ Load RGB Representations """
                            video_reps = np.array(self.hf_rgb.get(videoname))
                            snippets = video_reps[indices,:] # nIndices x D
                            snippets = torch.tensor(snippets,dtype=torch.float)
                            snippets = snippets.unsqueeze(0) # 1 x nframes x D
                            
                            """ Load Flow Representations """
                            if 'inference' in self.phase:
                                if self.phase == 'Gronau_inference':
                                    jump_size = 15 #Gronau VUA videos are of a single fps (30)
                                elif self.phase == 'HMH_inference':
                                    dataset = 'VUA_HMH'
                                    jump_size = int(fps_dict[dataset][videoname]//2)
                            else:
                                jump_size = int(fps_dict[videoname]//2)
                            flow_indices = list(map(lambda idx:idx//jump_size,indices)) # you need fps_dict here to make sure right loading of flows
                            flow_indices = np.unique(flow_indices)
                            flow_reps = np.array(self.hf_of.get(videoname)) 
                            flows = flow_reps[flow_indices,:]
                            flows = torch.tensor(flows,dtype=torch.float)
                            flows = flows.unsqueeze(0) # 1 x nflows x D
                                                        
                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                snippets2 = video_reps[indices2,:] # nframes x D  # [startIdx:endIdx,:] (def)
                                snippets2 = torch.tensor(snippets2,dtype=torch.float)
                                snippets2 = snippets2.unsqueeze(0) # 1 x nframes x D

                                snippets3 = video_reps[indices3,:] # nframes x D  # [startIdx:endIdx,:] (def)
                                snippets3 = torch.tensor(snippets3,dtype=torch.float)
                                snippets3 = snippets3.unsqueeze(0) # 1 x nframes x D
                    
                                flow_indices2 = list(map(lambda idx:idx//jump_size,indices2)) # 15 b/c of 30 fps of NS videos
                                flow_indices2 = np.unique(flow_indices2) 
                                flows2 = flow_reps[flow_indices2,:] # 1 x D
                                flows2 = torch.tensor(flows2,dtype=torch.float)
                                flows2 = flows2.unsqueeze(0)
                            
                                flow_indices3 = list(map(lambda idx:idx//jump_size,indices3)) # 15 b/c of 30 fps of NS videos
                                flow_indices3 = np.unique(flow_indices3)
                                flows3 = flow_reps[flow_indices3,:] # 1 x D
                                flows3 = torch.tensor(flows3,dtype=torch.float)
                                flows3 = flows3.unsqueeze(0)
                        
                            domain = self.domain
                            frames_importance = torch.zeros(1,snippets.shape[1],dtype=torch.float) # placeholder
                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                return (snippets, snippets2, snippets3), (flows, flows2, flows3), videoname, label, frames_importance, domain
                        elif self.dataset_name in ['NS_vs_VUA']:
                            df = self.data[self.phase]
                            curr_df = df.iloc[idx,:] #df with timestamps, videopath, etc.
                            videoname = curr_df['Video']
                            label = 0 if curr_df['Domain'] == 'NS' else 1 #July26
                            domain = curr_df['Domain']
                            
                            if self.phase == 'USC_inference': #b/c we do not have ground-truth labels here
                                label = torch.tensor(0,dtype=torch.long) # placeholder - I just need some integer
                                colStartName, colEndName = 'StartFrame', 'EndFrame'
                                startIdx = curr_df[colStartName]-1 if curr_df[colStartName] != 0 else curr_df[colStartName]
                                endIdx = curr_df[colEndName]-1
                            else:
                                #label = self.label_encoder.transform([curr_df['EASE']]).item()
                                label = torch.tensor(label,dtype=torch.long)

                                colStartName = 'StartFrame'
                                colEndName = 'EndFrame'
                                startIdx = curr_df[colStartName]-1
                                endIdx = curr_df[colEndName]-1
                            
                            #if domain == 'VUA':
                            jump_size = 10
                            #elif domain == 'NS':
                            #    diff = endIdx - startIdx
                            #    jump_size = diff//10
                                
                            if self.phase in ['val','test']:
                                indices = np.arange(startIdx,endIdx,jump_size)
                                offset2 = 3
                                offset3 = 6
                                indices2 = list(np.arange(startIdx+offset2,endIdx+offset2,jump_size))
                                indices3 = list(np.arange(startIdx+offset3,endIdx+offset3,jump_size))
                            elif self.phase == 'train':
                                indices = np.arange(startIdx,endIdx,jump_size)
                            indices = indices[:2000] # in the event of an outlier segment which is too long
                            """ Load RGB Representations """
                            video_reps = np.array(self.hf_rgb[domain].get(videoname))
                            snippets = video_reps[indices,:] # nIndices x D
                            snippets = torch.tensor(snippets,dtype=torch.float)
                            snippets = snippets.unsqueeze(0) # 1 x nframes x D
                            
                            """ Load Flow Representations """
                            #print(domain)
                            if 'inference' in self.phase:
                                if self.phase == 'Gronau_inference':
                                    jump_size = 15 #Gronau VUA videos are of a single fps (30)
                                elif self.phase == 'HMH_inference':
                                    dataset = 'VUA_HMH'
                                    jump_size = int(fps_dict[dataset][videoname]//2)
                            else:
                                if domain == 'NS':
                                    jump_size = 15 # all NS videos are 30 Hz
                                elif domain == 'VUA':
                                    dataset = 'VUA'
                                    jump_size = int(fps_dict[dataset][videoname]//2)
                            flow_indices = list(map(lambda idx:idx//jump_size,indices)) # you need fps_dict here to make sure right loading of flows
                            flow_indices = np.unique(flow_indices)
                            flow_reps = np.array(self.hf_of[domain].get(videoname)) 
                            flows = flow_reps[flow_indices,:]
                            flows = torch.tensor(flows,dtype=torch.float)
                            flows = flows.unsqueeze(0) # 1 x nflows x D
                                                        
                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                snippets2 = video_reps[indices2,:] # nframes x D  # [startIdx:endIdx,:] (def)
                                snippets2 = torch.tensor(snippets2,dtype=torch.float)
                                snippets2 = snippets2.unsqueeze(0) # 1 x nframes x D

                                snippets3 = video_reps[indices3,:] # nframes x D  # [startIdx:endIdx,:] (def)
                                snippets3 = torch.tensor(snippets3,dtype=torch.float)
                                snippets3 = snippets3.unsqueeze(0) # 1 x nframes x D
                    
                                flow_indices2 = list(map(lambda idx:idx//jump_size,indices2)) # 15 b/c of 30 fps of NS videos
                                flow_indices2 = np.unique(flow_indices2) 
                                flows2 = flow_reps[flow_indices2,:] # 1 x D
                                flows2 = torch.tensor(flows2,dtype=torch.float)
                                flows2 = flows2.unsqueeze(0)
                            
                                flow_indices3 = list(map(lambda idx:idx//jump_size,indices3)) # 15 b/c of 30 fps of NS videos
                                flow_indices3 = np.unique(flow_indices3)
                                flows3 = flow_reps[flow_indices3,:] # 1 x D
                                flows3 = torch.tensor(flows3,dtype=torch.float)
                                flows3 = flows3.unsqueeze(0)
                        
                            #domain = self.domain
                            frames_importance = torch.zeros(1,snippets.shape[1],dtype=torch.float) # placeholder
                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                return (snippets, snippets2, snippets3), (flows, flows2, flows3), videoname, label, frames_importance, domain
    
                        elif self.dataset_name in ['NS_DART']:
                            dart = self.data[self.phase]
                            curr_dart = dart.iloc[idx,:] #df with DART domain scores, videoname, etc. 
                            videoname = curr_dart['Video']
                            label = self.label_encoder.transform([curr_dart[self.dart_domain]]).item()
                            label = torch.tensor(label,dtype=torch.long)
                            
                            """ Obtain Video-Specific Gesture Segments """
                            df = self.df
                            bool1 = (df['Video'] == videoname)
                            gest_df = df[bool1]
                            #print(sorted(df['Video'].unique()))
                            #print(videoname)
                            
                            def getSingleClip(curr_df,videoname):
                                startIdx = curr_df['StartFrame']-1
                                endIdx = curr_df['EndFrame']-1
                                if self.phase == 'val':
                                    diff = endIdx - startIdx
                                    jump_size = diff // 10
                                    indices = list(np.arange(startIdx,endIdx,jump_size)) #[:10] #15
                                elif self.phase == 'train':
                                    diff = endIdx - startIdx
                                    jump_size = diff // 10
                                    indices = list(np.arange(startIdx,endIdx,jump_size)) #[:10] #15
                                
                                """ Get RGB Clip """
                                video_reps = np.array(self.hf_rgb.get(videoname))
                                snippets = video_reps[indices,:] # nframes x D  # [startIdx:endIdx,:] (def)
                                snippets = torch.tensor(snippets,dtype=torch.float)
                                snippets = snippets.unsqueeze(0) # 1 x nframes x D
                            
                                """ Get Flows Clip """
                                flow_indices = list(map(lambda idx:idx // 15,indices))
                                flow_indices = np.unique(flow_indices) #to avoid repeating frames and thus prevent overfitting
                                flow_reps = np.array(self.hf_of.get(videoname))
                                flows = flow_reps[flow_indices,:] # nflows x D
                                flows = torch.tensor(flows,dtype=torch.float)
                                flows = flows.unsqueeze(0) # 1 x nflows x D
                                return snippets, flows
                            
                            snippets, flows = [], []
                            #print('# of Clips: %i' % len(gest_df))
                            for index,curr_df in gest_df.iterrows():
                                curr_snippets, curr_flows = getSingleClip(curr_df,videoname) #same video, different gesture segments # 1 x nframes x D
                                snippets.append(curr_snippets)
                                flows.append(curr_flows)
                            
                            """ Prototypes Path - Concatenate to use prototypes """
                            snippets = torch.cat(snippets,1) # 1 x nframes*nsnippets x D
                            flows = torch.cat(flows,1)
                            #snippets = torch.stack(snippets) # nsnippetsA x nframes x dim
                            #flows = torch.stack(flows) # nsnippetsB x nflows x dim
                            
                        elif 'Gestures_Classification' in self.dataset_name:# in ['NS_Gestures_Classification']: #,'NS_Gestures_Recommendation','SOCAL']:
                            df = self.data[self.phase]
                            curr_df = df.iloc[idx,:] #df with gesture, timestamps, videopath, etc.
                            videoname = curr_df['Video']
                            
                            if self.phase in ['CinVivo_inference','USC_inference']: #b/c we do not have ground-truth labels here
                                label = torch.tensor(0,dtype=torch.long) # placeholder - I just need some integer
                            else:
                                label = self.label_encoder.transform([curr_df['Gesture']]).item()
                                label = torch.tensor(label,dtype=torch.long)
                                
                            startIdx = curr_df['StartFrame']-1
                            endIdx = curr_df['EndFrame']-1
                            #nframes = endIdx - startIdx
                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                diff = endIdx - startIdx
                                jump_size = diff//10 #10 for NS
                                """ Done - introduce offset so that indices do NOT overlap """
                                indices = list(np.arange(startIdx,endIdx,jump_size)) #[:10] #15
                                indices2 = list(np.arange(startIdx+3,endIdx,jump_size)) #endIdx+3
                                indices3 = list(np.arange(startIdx+6,endIdx,jump_size)) #endIdx+3
                            elif self.phase in ['train','train+val']:
                                diff = endIdx - startIdx
                                jump_size = diff//10 #10 for NS
                                #offset = random.sample(list(np.arange(10)),1)[0]
                                indices = list(np.arange(startIdx,endIdx,jump_size)) #[:10] #15
                            
                            video_reps = np.array(self.hf_rgb.get(videoname))
                            snippets = video_reps[indices,:] # nframes x D  # [startIdx:endIdx,:] (def)
                            snippets = torch.tensor(snippets,dtype=torch.float)
                            snippets = snippets.unsqueeze(0) # 1 x nframes x D

                            if 'NS' in self.dataset_name:
                                jump_factor = 15
                            elif 'VUA' in self.dataset_name:
                                jump_factor = 10
                            elif 'JIGSAWS' in self.dataset_name:
                                jump_factor = 15
                            elif 'DVC_UCL' in self.dataset_name:
                                jump_factor = 30
                            
                            flow_indices = list(map(lambda idx:idx//jump_factor,indices)) # 15 b/c of 30 fps of NS videos
                            flow_indices = np.unique(flow_indices) #to avoid repeating frames and thus prevent overfitting
                            flow_reps = np.array(self.hf_of.get(videoname))
                            # new to avoid choosing out-of-bound reps
                            flow_indices = [idx for idx in flow_indices if idx < len(flow_reps)] ## NEW - July 29th 
                            flows = flow_reps[flow_indices,:] # 1 x D
                            flows = torch.tensor(flows,dtype=torch.float)
                            flows = flows.unsqueeze(0)
                            
                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                snippets2 = video_reps[indices2,:] # nframes x D  # [startIdx:endIdx,:] (def)
                                snippets2 = torch.tensor(snippets2,dtype=torch.float)
                                snippets2 = snippets2.unsqueeze(0) # 1 x nframes x D

                                snippets3 = video_reps[indices3,:] # nframes x D  # [startIdx:endIdx,:] (def)
                                snippets3 = torch.tensor(snippets3,dtype=torch.float)
                                snippets3 = snippets3.unsqueeze(0) # 1 x nframes x D
                    
                                flow_indices2 = list(map(lambda idx:idx//jump_factor,indices2)) # 15 b/c of 30 fps of NS videos
                                flow_indices2 = np.unique(flow_indices2) 
                                flow_indices2 = [idx for idx in flow_indices2 if idx < len(flow_reps)] # new  - July 29th
                                flows2 = flow_reps[flow_indices2,:] # 1 x D
                                flows2 = torch.tensor(flows2,dtype=torch.float)
                                flows2 = flows2.unsqueeze(0)
                            
                                flow_indices3 = list(map(lambda idx:idx//jump_factor,indices3)) # 15 b/c of 30 fps of NS videos
                                flow_indices3 = np.unique(flow_indices3)
                                flow_indices3 = [idx for idx in flow_indices3 if idx < len(flow_reps)] # new  - July 29th
                                flows3 = flow_reps[flow_indices3,:] # 1 x D
                                flows3 = torch.tensor(flows3,dtype=torch.float)
                                flows3 = flows3.unsqueeze(0)
                        
                            domain = self.domain
                            frames_importance = torch.zeros(1,snippets.shape[1],dtype=torch.float) # placeholder
                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                return (snippets, snippets2, snippets3), (flows, flows2, flows3), videoname, label, frames_importance, domain
                            
                            #snippets = self.all_info_dict['snippets'][idx]
                            #snippets = snippets.squeeze(0)
                            #snippets = snippets[-1,:]
                            #snippets = snippets.unsqueeze(0)
                            #snippets = snippets.squeeze(0)
                            #snippets = torch.mean(snippets,1) # 1 x 512
                            #videoname = self.all_info_dict['videonames'][idx]
                            #label = self.all_info_dict['labels'][idx] #[0]
                        elif self.dataset_name in ['Custom_Gestures']:
                            df = self.data[self.phase]
                            curr_df = df.iloc[idx,:] #df with gesture, timestamps, videopath, etc.
                            videoname = curr_df['Video']
                            
                            if self.phase in ['Custom_inference']: #b/c we do not have ground-truth labels here
                                label = torch.tensor(0,dtype=torch.long) # placeholder - I just need some integer
                            else:
                                label = self.label_encoder.transform([curr_df['Gesture']]).item()
                                label = torch.tensor(label,dtype=torch.long)
                                
                            startIdx = curr_df['StartFrame']-1
                            endIdx = curr_df['EndFrame']-1
                            #nframes = endIdx - startIdx
                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                diff = endIdx - startIdx
                                jump_size = diff//10 #10 for NS
                                """ Done - introduce offset so that indices do NOT overlap """
                                indices = list(np.arange(startIdx,endIdx,jump_size)) #[:10] #15
                                indices2 = list(np.arange(startIdx+3,endIdx,jump_size)) #endIdx+3
                                indices3 = list(np.arange(startIdx+6,endIdx,jump_size)) #endIdx+3
                            elif self.phase in ['train','train+val']:
                                diff = endIdx - startIdx
                                jump_size = diff//10 #10 for NS
                                #offset = random.sample(list(np.arange(10)),1)[0]
                                indices = list(np.arange(startIdx,endIdx,jump_size)) #[:10] #15
                            
                            video_reps = np.array(self.hf_rgb.get(videoname))
                            snippets = video_reps[indices,:] # nframes x D  # [startIdx:endIdx,:] (def)
                            snippets = torch.tensor(snippets,dtype=torch.float)
                            snippets = snippets.unsqueeze(0) # 1 x nframes x D

                            jump_factor = 15
                            
                            flow_indices = list(map(lambda idx:idx//jump_factor,indices)) # 15 b/c of 30 fps of NS videos
                            flow_indices = np.unique(flow_indices) #to avoid repeating frames and thus prevent overfitting
                            flow_reps = np.array(self.hf_of.get(videoname))
                            # new to avoid choosing out-of-bound reps
                            flow_indices = [idx for idx in flow_indices if idx < len(flow_reps)] ## NEW - July 29th 
                            flows = flow_reps[flow_indices,:] # 1 x D
                            flows = torch.tensor(flows,dtype=torch.float)
                            flows = flows.unsqueeze(0)
                            
                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                snippets2 = video_reps[indices2,:] # nframes x D  # [startIdx:endIdx,:] (def)
                                snippets2 = torch.tensor(snippets2,dtype=torch.float)
                                snippets2 = snippets2.unsqueeze(0) # 1 x nframes x D

                                snippets3 = video_reps[indices3,:] # nframes x D  # [startIdx:endIdx,:] (def)
                                snippets3 = torch.tensor(snippets3,dtype=torch.float)
                                snippets3 = snippets3.unsqueeze(0) # 1 x nframes x D
                    
                                flow_indices2 = list(map(lambda idx:idx//jump_factor,indices2)) # 15 b/c of 30 fps of NS videos
                                flow_indices2 = np.unique(flow_indices2) 
                                flow_indices2 = [idx for idx in flow_indices2 if idx < len(flow_reps)] # new  - July 29th
                                flows2 = flow_reps[flow_indices2,:] # 1 x D
                                flows2 = torch.tensor(flows2,dtype=torch.float)
                                flows2 = flows2.unsqueeze(0)
                            
                                flow_indices3 = list(map(lambda idx:idx//jump_factor,indices3)) # 15 b/c of 30 fps of NS videos
                                flow_indices3 = np.unique(flow_indices3)
                                flow_indices3 = [idx for idx in flow_indices3 if idx < len(flow_reps)] # new  - July 29th
                                flows3 = flow_reps[flow_indices3,:] # 1 x D
                                flows3 = torch.tensor(flows3,dtype=torch.float)
                                flows3 = flows3.unsqueeze(0)
                        
                            domain = self.domain
                            frames_importance = torch.zeros(1,snippets.shape[1],dtype=torch.float) # placeholder
                            if self.phase in ['val','test'] or 'inference' in self.phase:
                                return (snippets, snippets2, snippets3), (flows, flows2, flows3), videoname, label, frames_importance, domain
                        else:
                            """ New Path """
                            videonames = self.all_info_dict['videonames']
                            videoname = videopath.split('/')[-1]
                            indices = np.where(np.in1d(videonames,videoname))[0]
                            #print(videonames)
                            #print('hello')
                            #print(videoname)
                            snippets = list(itemgetter(*indices)(self.all_info_dict['snippets'])) 
                            snippets = torch.cat(snippets,1)
                            snippets = snippets.squeeze(0)
                            
                            label = ''
                            """ End New Path """

                            """ Old Path """
                            #videoname = video'path.split('\\')[-1]
                            #phase_patients = list(self.snippets_dict.keys())
                            #idx = np.where([videoname in name[0] for name in phase_patients])[0][0]
                            #key = phase_patients[idx]
                            ##print(key,videoname)
                            #snippets = self.snippets_dict[key] # 1 x nsnippets x D
                            #snippets = snippets.squeeze(0) # nsnippets x D
                            # indices = random.sample(list(range(snippets.shape[0]-5)),2)
                            #snippets_original = snippets[:2,:]
                            #snippets_flipped = snippets[-2:,:]
                            """ End of Old Path """

                return snippets, flows, videoname, label, frames_importance, domain
        
#         def getSingleSnippet(self,video_reps,startIdx,EndIdx):
#             snippet = video_reps[startIdx:endIdx,:]
#             return snippet
        
        def normalizeSnippets(self,snippets):
                snippets = torch.stack(snippets) # nsnippets x channels x snippetLength x new_height x new_width
                nsnippets,nchannels,nframes,height,width = snippets.shape

#               snippets_mean, snippets_std = torch.mean(snippets,[3,4]), torch.std(snippets,[3,4]) # nsnippets x channels x snippetLength
#               snippets_mean, snippets_std = snippets_mean.view(nsnippets,nchannels,nframes,1,1).repeat(1,1,1,height,width), snippets_std.view(nsnippets,nchannels,nframes,1,1).repeat(1,1,1,height,width) 
#               snippets = (snippets - snippets_mean) / snippets_std # nsnippets x channels x snippetLength x height x width    

                #snippets_mean, snippets_std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])         
                if self.encoder_type == 'I3D':
                        snippets_mean, snippets_std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
                elif self.encoder_type == 'R3D':
                        snippets_mean, snippets_std = torch.tensor([0.43216, 0.394666, 0.37645]), torch.tensor([0.22803, 0.22145, 0.216989])            
                elif self.encoder_type == 'ViT':
                        snippets_mean, snippets_std = torch.tensor([0.500,0.500,0.500]), torch.tensor([0.500,0.500,0.500])                                      
                
                snippets_mean, snippets_std = snippets_mean.view(1,nchannels,1,1,1).repeat(nsnippets,1,nframes,height,width), snippets_std.view(1,nchannels,1,1,1).repeat(nsnippets,1,nframes,height,width)             
                snippets = (snippets/255 - snippets_mean) / snippets_std # nsnippets x channels x snippetLength x height x width        
                
                return snippets         
                
        def __len__(self):
            if self.data_type == 'raw':
                n = len(self.data[self.phase]) #self.df_subset.shape[0]
            elif self.data_type == 'reps':
                #if self.dataset_name in ['VUA_EASE','VUA_EASE_Stitch','NS_Gestures_Classification','VUA_Gestures_Classification','NS_DART','NS_Gestures_Recommendation','SOCAL','DVC_UCL_Gestures_Classification','JIGSAWS_Suturing_Gestures_Classification']:
                    #n = len(self.all_info_dict['labels'])
                n = len(self.data[self.phase])
                #elif self.dataset_name == 'NS':
                #    n = self.df_subset.shape[0]
            return n

class loadDataloader(object):
    
    def __init__(self,root_path,dataset_name,data_type,batch_size,nclasses,domain,phases,task,balance,balance_groups,single_group,group_info,importance_loss,encoder_type='R3D',encoder_params='ViT_SelfSupervised_ImageNet',snippetLength=30,frameSkip=30,overlap=0,fold=1,training_fraction=1):
        self.root_path = root_path
        self.dataset_name = dataset_name
        self.data_type = data_type
        self.batch_size = batch_size
        self.nclasses = nclasses
        self.domain = domain
        self.phases = phases
        self.task = task
        self.balance = balance
        self.balance_groups = balance_groups
        self.single_group = single_group
        self.group_info = group_info
        self.importance_loss = importance_loss
        self.encoder_type = encoder_type
        self.encoder_params = encoder_params
        self.snippetLength = snippetLength
        self.frameSkip = frameSkip
        self.overlap = overlap
        self.fold = fold
        self.training_fraction = training_fraction
    
    def load(self):
        shuffle_dict = {'train':True,'train+val':True,'val':False,'test':False,'inference':False,'Gronau_inference':False,'Gronau_full_inference':False,'RAPN_inference':False,'USC_inference':False,'HMH_inference':False,'Lab_inference':False,'AFB_inference':False,'CinVivo_inference':False,'CinVivo_inference_labelled':False,'Custom_inference':False}        
        DatasetPhases = {phase: VideoDataset(self.root_path,self.dataset_name,self.data_type,self.nclasses,self.domain,phase,self.task,self.balance,self.balance_groups,self.single_group,self.group_info,self.importance_loss,self.encoder_type,self.encoder_params,self.frameSkip,self.snippetLength,self.overlap,self.fold,self.training_fraction) for phase in self.phases}
        DataloaderPhases = {phase: DataLoader(DatasetPhases[phase],batch_size=self.batch_size,shuffle=shuffle_dict[phase],drop_last=False,collate_fn=self.pad_collate) for phase in self.phases}
        
        return DataloaderPhases

    def createPaddingMask(self,x,lens):
        nsnippets = max([el.shape[1] for el in x]) # nframes x nsnippets x ndim (max number of snippets) # NEW
        nbatch = len(x)
        nframes = max(lens)+1 # +1 to account for frame_cls token that is prepended (max number of frames)

        key_padding_mask = torch.zeros(nbatch,nsnippets,nframes).type(torch.bool) #nsnippets is NEW
        for row,xlen in zip(range(key_padding_mask.shape[0]),lens):
            key_padding_mask[row,:,xlen+1:] = True # +1 to also account for last admissable token (given inclusion of frame_cls)
        return key_padding_mask

    def pad_collate(self,batch):
        videoname, snippets, flows, label, frames_importance, domains = zip(*batch)
        if self.task == 'MIL':
            #if self.batch_size == 1:
            snippets = snippets[0] #extra
            flows = flows[0] #extra
            #else:
            #    pass
                #snippets = torch.vstack([torch.tensor(el) for el in snippets])
                #flows = torch.vstack(flows)
            
            snippets_lens = [s.shape[1] for s in snippets] #nsnippets x nframes x dim
            flows_lens = [f.shape[1] for f in flows]

            snippets, flows = [s.permute(1,0,2) for s in snippets], [f.permute(1,0,2) for f in flows] #variable length dimension must be second (in this case it is nframes) # nframes x nsnippets x ndim
            snippets_mask, flows_mask = self.createPaddingMask(snippets,snippets_lens), self.createPaddingMask(flows,flows_lens)

            snippets_padded = pad_sequence(snippets, batch_first=True, padding_value=0) 
            flows_padded = pad_sequence(flows, batch_first=True, padding_value=0)
            
            #if self.dataset_name == 'VUA_EASE':
            #    """ New """
            #    snippets_padded, flows_padded = snippets_padded.permute(0,2,1,3), flows_padded.permute(0,2,1,3) #repermute since model expects a particular order of dimensions - B x nsnippets x nframes x D
            #else:
            snippets_padded, flows_padded = snippets_padded.permute(2,0,1,3), flows_padded.permute(2,0,1,3) #repermute since model expects a particular order of dimensions - B x nsnippets x nframes x D
            #print(snippets_padded.shape)
            """ TODO: you can also try padding along the snippets dimension here, if need be --> also requires a new len/mask pair"""

            videoname = [v for v in videoname]
            label = torch.stack([l for l in label])
                        
        elif self.task == 'Prototypes':
            if isinstance(snippets[0],tuple):
                snippets_lens = dict()
                snippets_padded = dict()
                snippets_mask = dict()
                
                flows_lens = dict()
                flows_padded = dict()
                flows_mask = dict()
                
                nbatch = len(snippets)
                nversions = len(snippets[0]) # 3 versions
                for i in range(nversions): #snippets is tuple # I want to iterate over different versions NOT batches
                    snippet = [snippets[n][i] for n in range(nbatch)]
                    flow = [flows[n][i] for n in range(nbatch)]
                    #print(snippet[0].shape)
                    snippet_lens = [s.shape[1] for s in snippet]
                    snippet = [s.permute(1,0,2) for s in snippet]
                    snippet_mask = self.createPaddingMask(snippet,snippet_lens)
                    snippet_padded = pad_sequence(snippet, batch_first=True, padding_value=0) 
                    snippet_padded = snippet_padded.permute(0,2,1,3) #repermute since model expects a particular order of dimensions - B x nsnippets x nframes x D
                    snippets_lens[i] = snippet_lens
                    snippets_padded[i] = snippet_padded
                    snippets_mask[i] = snippet_mask
                    
                    flow_lens = [f.shape[1] for f in flow]
                    flow = [f.permute(1,0,2) for f in flow]
                    flow_mask = self.createPaddingMask(flow,flow_lens)
                    flow_padded = pad_sequence(flow, batch_first=True, padding_value=0)
                    flow_padded = flow_padded.permute(0,2,1,3) 
                    flows_lens[i] = flow_lens
                    flows_padded[i] = flow_padded
                    flows_mask[i] = flow_mask
                
                frames_importance_padded = torch.zeros(1,1) # placeholder
                frames_importance_mask = torch.zeros(1,1)
                #flows_lens = [f.shape[1] for f in flows]
                #flows = [f.permute(1,0,2) for f in flows]
                #flows_mask = self.createPaddingMask(flows,flows_lens)
                #flows_padded = pad_sequence(flows, batch_first=True, padding_value=0)
                #flows_padded = flows_padded.permute(0,2,1,3)
            else:
                snippets_lens = [s.shape[1] for s in snippets] #nsnippets x nframes x dim (for s)
                flows_lens = [f.shape[1] for f in flows]
                
                frames_importance = [i.permute(1,0) for i in frames_importance] # nframes x 1
                snippets, flows = [s.permute(1,0,2) for s in snippets], [f.permute(1,0,2) for f in flows] #variable length dimension must be second (in this case it is nframes) # nframes x nsnippets x ndim
                
                snippets_mask, flows_mask = self.createPaddingMask(snippets,snippets_lens), self.createPaddingMask(flows,flows_lens)
                frames_importance_mask = self.createPaddingMask(frames_importance,snippets_lens) # same as number of frames so no need to change # B x MAX_FRAMES x MAX_FRAMES (bool) # CAUTION: TRUE element is an element to MASK
                
                snippets_padded = pad_sequence(snippets, batch_first=True, padding_value=0) 
                flows_padded = pad_sequence(flows, batch_first=True, padding_value=0)
                frames_importance_padded = pad_sequence(frames_importance, batch_first=True, padding_value=0) # B x MAX_FRAMES x 1 (actual values)
                #print(frames_importance_padded.shape,snippets_padded.shape) # B x 1 x MAX_FRAMES (actual values)

                snippets_padded, flows_padded = snippets_padded.permute(0,2,1,3), flows_padded.permute(0,2,1,3) #repermute since model expects a particular order of dimensions - B x nsnippets x nframes x D
                frames_importance_padded = frames_importance_padded.permute(0,2,1) # B x 1 x MAX_FRAMES
                
            videoname = [v for v in videoname]
            label = torch.stack([l for l in label])
            #print(snippets.shape)
            #print(label)
        elif self.task == 'ClassificationHead': # padding won't be dealt with by R3D (might need to be changed)
            if isinstance(snippets[0],tuple):
                frames_importance = torch.zeros(1,1) # placeholder
                frames_importance_mask = torch.zeros(1,1) # placeholder 
                snippets_lens = dict()
                snippets_padded = dict()
                snippets_mask = dict()
                flows_lens = dict()
                flows_padded = dict()
                flows_mask = dict()
                
                nbatch = len(snippets)
                nversions = len(snippets[0]) # 3 versions
                for i in range(nversions): #snippets is tuple # I want to iterate over different versions NOT samples
                    snippet = torch.stack([snippets[n][i] for n in range(nbatch)])
                    flow = torch.stack([flows[n][i] for n in range(nbatch)])
                    snippets_padded[i] = snippet
                    flows_padded[i] = flow
                snippets = snippets_padded
                flows = flows_padded
            else:
                snippets = torch.stack(snippets)
                flows = torch.stack(flows)
                frames_importance = torch.stack(frames_importance)
                snippets_lens = []
                snippets_mask = torch.zeros(1,1)
                flows_lens = []
                flows_mask = torch.zeros(1,1)
                frames_importance_mask = torch.zeros(1,1)
            videoname = [v for v in videoname]
            label = torch.stack([l for l in label])
            return videoname, snippets, flows, frames_importance, label, snippets_lens, flows_lens, snippets_mask, flows_mask, frames_importance_mask, domains
            
        return videoname, snippets_padded, flows_padded, frames_importance_padded, label, snippets_lens, flows_lens, snippets_mask, flows_mask, frames_importance_mask, domains

        
                
                
