import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from tqdm import tqdm
import argparse
import torch
import time
label_encoder = LabelEncoder()

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path',type=str)
args = parser.parse_args()
rootpath = args.path

def loadCustomInferenceData(inference_set):
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
    
    if inference_set == 'Custom_inference':
        df = pd.read_csv(os.path.join(rootpath,'paths','Custom_Paths.csv'),index_col=0)

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
    
    gestures = ['in-view','out-of-view']
    mapping = dict(zip(np.arange(len(gestures)),sorted(gestures)))
    return df, mapping

def calcProbs(info,p,augment):
    #augment = 0 # TTA version
    reps = torch.stack(info['reps'][augment])
    reps = reps.cpu().detach()
    pros = torch.vstack(list(p.values()))
    pros = pros.cpu().detach()
    
    norm = torch.norm(pros,dim=1).unsqueeze(1).repeat(1,pros.shape[1])
    p_norm = pros / norm
    norm = torch.norm(reps,dim=1).unsqueeze(1).repeat(1,reps.shape[1]) # nbatch x D
    s_norm = reps / norm

    sim = torch.matmul(s_norm,p_norm.T) # nbatch x nprototypes
    sim_exp = torch.exp(sim)
    probs = sim_exp / torch.sum(sim_exp,1).unsqueeze(1).repeat(1,sim_exp.shape[1]) # nbatch x nprototype
    return reps, sim, probs

def getSavepath(args,fold): # where params are saved
    return os.path.join(os.path.join(args.path,'params/Fold_%i' % fold))

def getResults(savepath,inference_set='test'):
    """ Get Output Probability Scores For All TTA Augments """
    info = torch.load(os.path.join(savepath,'reps_and_labels_%s' % inference_set),map_location=device)
    p = torch.load(os.path.join(savepath,'prototypes.zip'),map_location=device)      
    augment_versions = 3
    probs_df = pd.DataFrame()
    for augment in range(augment_versions):
        _, _, probs = calcProbs(info,p,augment)
        probs = probs.cpu().detach().numpy()
        curr_probs_df = pd.DataFrame(probs)
        curr_probs_df['TTA'] = augment
        curr_probs_df['ID'] = np.arange(curr_probs_df.shape[0])
        probs_df = pd.concat((probs_df,curr_probs_df),axis=0)
    return probs_df

def getMetaInfo(probs_df,df,mapping,dataset='Custom',domain='in_vs_out',class_cols=[0,1],inference_set='Custom_inference'):
    if len(class_cols) > 1: # prototypes path
        if dataset == 'VUA' and inference_set == 'AFB': # this is threshold we used for trial
            probs_df['pred'] = probs_df[1].apply(lambda probs:1 if probs > 0.4 else 0)
        else: # equivalent to threshold of 0.5 for binary classification
            probs_df['pred'] = probs_df[class_cols].apply(lambda probs:np.argmax(probs),1) #still applies in binary b/c 2 prototypes
    else: # classification head path
        probs_df['pred'] = probs_df[0].apply(lambda prob:1 if prob > 0.5 else 0) 
    probs_df.index = df.index
    probs_df['Video'] = df['Video']
    
    if domain == 'in_vs_out':
        if inference_set == 'Custom_inference':
            cols = ['StartFrame','EndFrame'] # no gesture label
            probs_df[cols] = df[cols]
        
    probs_df['pred'] = probs_df['pred'].map(mapping)
    return probs_df

def getPreds(ensemble_df,mapping,threshold=None):
    ensemble_df['Entropy'] = ensemble_df[class_cols].apply(lambda probs:-np.sum(probs*np.log(probs)),axis=1)
    if threshold == None:
        ensemble_df['pred'] = ensemble_df[class_cols].apply(lambda probs:np.argmax(probs),1) 
    else:
        ensemble_df['pred'] = ensemble_df[class_cols[-1]].apply(lambda prob:int(prob>threshold),1)
    ensemble_df['pred'] = ensemble_df['pred'].map(mapping)
    return ensemble_df

def groupPredictionIntervals(curr_df,seconds=2):
    cumCount = 0
    startIndices = []
    endIndices = []
    
    """ Edge Case With 1 Entry """
    if len(curr_df) == 1:
        startIndices.append(curr_df.index[0])
        endIndices.append(curr_df.index[0])
    
    startIdx = curr_df.index[0]
    prevIdx = startIdx
    for index,row in curr_df.iloc[1:,:].iterrows():
        if index - prevIdx > seconds:
            startIndices.append(startIdx)
            endIndices.append(prevIdx)
            startIdx = index
            cumCount = 0

        """ Edge Conditions """
        if index == curr_df.index[-1]:
            if cumCount == 0: #final single entry
                startIndices.append(index)
                endIndices.append(index)
            else:
                startIndices.append(startIdx)
                endIndices.append(index)
                
        cumCount += 1
        prevIdx = index
    return startIndices, endIndices

def getGestures(curr_df,startIndices,endIndices):
    probs_df = pd.DataFrame()
    for startIdx,endIdx in zip(startIndices,endIndices):
        startRow, endRow = curr_df.loc[startIdx,:], curr_df.loc[endIdx,:] 
        startFrame = int(startRow['StartFrame'])
        endFrame = int(endRow['EndFrame'])   
        curr_probs_df = curr_df.loc[startIdx:endIdx,class_cols].mean() #average probability across time
        curr_probs_df = pd.DataFrame(curr_probs_df).T
        curr_probs_df.columns = class_cols
        curr_probs_df[['StartFrame','EndFrame']] = [startFrame,endFrame]
        probs_df = pd.concat((probs_df,curr_probs_df))
    gest_df = getPreds(probs_df,mapping).reset_index(drop=True)
    return gest_df

def FramesToTime(gest_df,col='StartFrame',fps=30):
    seconds = gest_df[col].apply(lambda frame:frame//fps)
    mins = seconds.apply(lambda sec:sec//60)
    hours = mins.apply(lambda mins:mins//60)
    time_df = pd.DataFrame([hours,mins,seconds]).T
    time_df.columns = ['hour','min','sec']
    time_cols = ['hour','min','sec']
    time_df.columns = time_cols
    for col in time_cols:
        time_df[col] = time_df.apply(lambda row:row[col] % 60,axis=1)
    time_df['time'] = time_df.astype(str).agg('-'.join,axis=1)
    time_df['time'] = pd.to_datetime(time_df['time'],format='%H-%M-%S')
    return time_df['time']

#%%
# Load in inference results
if __name__ == '__main__':
    starttime = time.time()

    device = 'cpu'
    binarizer = LabelBinarizer()
    class_cols = [0,1] # needs to be more robust --> function
    max_class = len(class_cols)
    df = pd.DataFrame()
    dataset = 'Custom' # NS | CinVivo
    domain = 'in_vs_out' # Top6 | in_vs_out
    inference_set = 'Custom_inference' # USC_inference | CinVivo_inference
    folds = [0]
    df_inf, mapping = loadCustomInferenceData(inference_set)
    for fold in tqdm(folds):
        #savepath = getSavepath(rootpath,dataset,task,domain,encoder_params,balance,self_attention,modalities,fold,balance_groups,single_group,group_info,importance_loss)
        savepath = getSavepath(args,fold)
        probs_df = getResults(savepath,inference_set=inference_set)
        probs_df = probs_df.groupby(by=['ID']).mean() # aggregate across test-time augments 
        probs_df = getMetaInfo(probs_df,df_inf,mapping,dataset=dataset,domain=domain,class_cols=class_cols,inference_set=inference_set)
        probs_df['ID'] = np.arange(len(probs_df))
        probs_df['Fold'] = fold
        df = pd.concat((df,probs_df),axis=0)

    # Ensemble the results
    meta_cols = [col for col in df.columns if col not in class_cols]
    ensemble_probs_df = df.groupby(by=['ID'])[class_cols].mean() # aggregate across folds
    ensemble_meta_df = df[df['Fold']==0][meta_cols].reset_index(drop=True)
    ensemble_df = pd.concat((ensemble_probs_df,ensemble_meta_df),axis=1)
    print(ensemble_df)
    ensemble_df = getPreds(ensemble_df,mapping,threshold=0.515)

    # Define hyperparameters for post-processing of ensembled results
    seconds = 3 # time between valid gesture predictions which imply SAME gesture # 2
    entropy_thresh = 0.66 # 1.735 for 6-way gesture classification

    # Process ensembled results to get predictions
    all_gest_df = pd.DataFrame()
    for video in tqdm(ensemble_df['Video'].unique()):
        bool1 = ensemble_df['Video']==video
        for gesture in list(mapping.values()):
            bool2 = ensemble_df['pred']==gesture
            boolcomb = bool1 & bool2
            curr_df = ensemble_df[boolcomb]
            curr_df = curr_df[curr_df['Entropy'] <= entropy_thresh] # remove uncertain predictions
            if len(curr_df) > 0: #if we have gesture predictions left
                startIndices, endIndices = groupPredictionIntervals(curr_df,seconds)
                gest_df = getGestures(curr_df,startIndices,endIndices)
                gest_df['StartTime'] = FramesToTime(gest_df,col='StartFrame')
                gest_df['EndTime'] = FramesToTime(gest_df,col='EndFrame')
                gest_df['Gesture'] = gesture
                gest_df['Video'] = video
                gest_df['Path'] = os.path.join('images',video)
                all_gest_df = pd.concat((all_gest_df,gest_df),axis=0)

    all_gest_df['Path'] = all_gest_df['Video'].apply(lambda video:os.path.join('images',video))
    if not os.path.exists(os.path.join(args.path,'results')):
        os.makedir(os.path.join(args.path,'results'))
    all_gest_df.to_csv(os.path.join(args.path,'results/Custom_inference_gestures.csv'))

    diff = time.time() - starttime
    print("Time taken (s): %.3f" % diff)