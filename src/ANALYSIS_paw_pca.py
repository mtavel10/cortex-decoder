import numpy as np
import matplotlib.pyplot as plt
import os
import src.IO
from sklearn.decomposition import PCA
from scipy.interpolate import PchipInterpolator

def forward_backward_fill(y):
    first_valid = np.min(np.argwhere(~np.isnan(y)))
    last_valid = np.max(np.argwhere(~np.isnan(y)))
    y[:first_valid] = y[first_valid]
    y[last_valid + 1:] = y[last_valid]
    return y

def interpolate_missing_keypts(y):
    #performs interpolation for 1 event, 1 bodypart, 1 dimension
    #input is a 1d masked array of length nframes in event
    #output is same size but with missing values interpolated
    #does pchip interpolation with no extrapolation and then forward or 
    #backward fills values at beginning or end
    x = np.arange(y.shape[0])
    # fig,ax = plt.subplots(3,1)
    # ax[0].plot(x,y)
    interp = PchipInterpolator(x[~y.mask],y[~y.mask],extrapolate=False)
    x_missing = x[y.mask]
    y_interp = interp(x_missing)
    y_filled = y.copy()
    y_filled[y.mask] = y_interp
    #need a different strategy for missing values at the beginning or end
    #--> right or left fill at beginning or end
    # ax[1].plot(x,y_filled)
    y_final = forward_backward_fill(y_filled)
    # ax[2].plot(x,y_final)
    # plt.show()
    # if np.sum(np.isnan(y_filled))>0:
    #     print(np.sum(np.isnan(y_filled)))
    #     print(np.argwhere(np.isnan(y_filled)))
    return y_filled

mice = ['mouse25','mouse22','mouse39','mouse35','mouse51','mouse46','mouse549']
behavior_list = ['all_reach','reach','grasp','carry','eating']
bodyparts = ['d1middle','d2tip','d2middle','d2knuckle','d3tip','d3middle',
'd3knuckle','d4tip','d4middle','wrist','wrist_outer']#,'elbow','elbow_crook']#,'pellet','pedestal','p2d1tip']
chunk_length = 29550 #TODO: handle shorter videos
pcutoff = 0.4
duration = 50
pca_all = {}
for mouseID in mice:
    pca_all[mouseID] = {}
    print(mouseID)
    days = src.IO.get_days(mouseID)
    for day in days:
        print(day)
        s2p_fld = src.IO.get_s2p_fld(mouseID,day)
        if not os.path.exists(f'{s2p_fld}/cam_event_times.pkl'):
            print(f'skipping {day}')
            continue
        pca_all[mouseID][day] = {}
        cam_event_times = src.IO.load_cam_event_times(mouseID,day)
        event_labels = np.load(f'{s2p_fld}/event_labels.npy')
        keys = sorted(cam_event_times.keys())
        X = []
        Y = []
        starts = []
        count_so_far = 0
        for key in keys:
            df_cam1, df_cam2 = src.IO.load_kinematics_df(key,mouseID,day)
            #X should have shape nsamples (timepoints) x nfeatures (bodyparts x and y)
            #keep x and y separate until after I've indexed timepoints
            x_bp = []
            y_bp = []
            paw_avgx,paw_avgy = src.utils.get_spatial_bp_avg(df_cam2,
                bodyparts,pcutoff)
            for b,bp in enumerate(bodyparts):
                x,y = src.IO.get_x_y(df_cam2,bp,pcutoff)
                if x[~x.mask].shape[0]<2:
                    print(f'{bp} obscured for entire event {key}')
                    x = paw_avgx
                    y = paw_avgy
                x = interpolate_missing_keypts(x)
                y = interpolate_missing_keypts(y)
                #print(np.sum(np.isnan(x)),np.sum(np.isnan(y)))
                x_bp.append(x-paw_avgx)
                y_bp.append(y-paw_avgy)
            starts += list(cam_event_times[key]+count_so_far)
            count_so_far += x.shape[0] #ntimepoints this event
            X.append(np.ma.stack(x_bp))
            Y.append(np.ma.stack(y_bp))
        X = np.ma.concatenate(X,axis=1)
        Y = np.ma.concatenate(Y,axis=1) 
        #now split timepoints by behavior and do PCA
        for beh in behavior_list:
            labels = src.utils.get_labels_for_behavior(event_labels,beh)
            starts = np.array(starts)
            starts_beh = starts[labels]
            indices = np.concatenate([np.arange(s,s+duration) for s in starts_beh])
            X_beh = X[:,indices]
            Y_beh = Y[:,indices]
            X_all = np.ma.concatenate([X_beh,Y_beh],axis=0)
            #subtract paw avg before pca?            
            pca = PCA()
            X_new = pca.fit_transform(X_all)
            pca_all[mouseID][day][beh] = pca
            # print(pca.explained_variance_ratio_.shape)
            # plt.figure()
            # plt.plot(pca.explained_variance_ratio_)
            # plt.title('variance explained')
            # plt.show()
    src.IO.save_pickle('results/paw_pca_per_day.pickle',pca_all)