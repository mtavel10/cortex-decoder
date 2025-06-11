import numpy as np
import pandas as pd
import glob
import src.utils
import src.IO
import matplotlib.pyplot as plt
import os

cam = 2
duration = 50
behavior_list = ['reach','grasp','carry','eating']
pcutoff = 0.4
# mice = ['mouse25','mouse22','mouse39']
mice = ['mouse51','mouse46','mouse549','mouse35']
if cam==1:
    mouth_pos = {'mouse51':(1220,472),'mouse46':(1248,370),
    'mouse549':(1156,381),'mouse35':(1253,476)}
elif cam==2:
    mouth_pos = {'mouse51':(1152,488),'mouse46':(1205,374),
    'mouse549':(1078,346),'mouse35':(1161,496)}
#mouth_pos = {'mouse25':(1210,393),'mouse39':(1205,484),'mouse22':(1212,564)}
for beh in behavior_list:
    fig,ax = plt.subplots(len(mice),12,figsize=(20,15),sharey=True)
    for m,mouseID in enumerate(mice):
        mouth_x,mouth_y = mouth_pos[mouseID]
        mouth_y = 1080-mouth_y
        print(mouseID)
        ax[m,0].set_ylabel(mouseID)
        days = src.IO.get_days(mouseID)
        for d,day in enumerate(days):
            print(day)
            kin_all_trials_cam1 = []
            kin_all_trials_cam2 = []    
            s2p_fld = src.IO.get_s2p_fld(mouseID,day)
            if not os.path.exists(f"{s2p_fld}/cam_event_times.pkl"):
                continue
            else:
                cam_event_times = src.IO.load_cam_event_times(mouseID,day)
            labels = src.IO.load_event_labels(mouseID,day)
            x_paw,y_paw = src.utils.get_paw_avg_mat(mouseID,day,cam_event_times,duration,pcutoff)
            y_paw = 1080 - y_paw #since 0 is from top in the coordinates
            #get just the beh trials
            beh_labels = src.utils.get_labels_for_behavior(labels,beh)
            ax[m,d].plot(x_paw[beh_labels].T,y_paw[beh_labels].T,color='gray',alpha=0.2)
            ax[m,d].plot(np.mean(x_paw[beh_labels],axis=0),np.mean(y_paw[beh_labels],axis=0),color='black')
            ax[m,d].plot(np.mean(x_paw[beh_labels],axis=0)[0],np.mean(y_paw[beh_labels],axis=0)[0],'o',color='red')
            ax[m,d].plot(np.mean(x_paw[beh_labels],axis=0)[-1],np.mean(y_paw[beh_labels],axis=0)[-1],'o',color='blue')
            ped_x,ped_y = src.utils.get_pedestal_avg(mouseID,day,cam_event_times,cam,pcutoff)
            ped_y = 1080-ped_y
            ax[m,d].plot(ped_x,ped_y,'o',color='purple')
            ax[m,d].plot(mouth_x,mouth_y,'o',color='pink')
            ax[m,d].set_aspect('equal')
    fig.suptitle(beh)
    plt.savefig(f'figs/paw_avg_spaghetti_plots_SST_mice_cam{cam}_{beh}')
    plt.show()
    