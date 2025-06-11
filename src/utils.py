import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import butter, lfilter, freqz
import src.IO as io

def curate_spks(spks,dF):
    '''first test: if the spk count estimates contain huge values, that 
    usually means the dF is super noisy with random spks --> 
    '''
def get_reg_ind(reg_file):
    import mat73
    reg = mat73.loadmat(reg_file)
    reg = reg['cell_registered_struct']['cell_to_index_map']
    reg_list = []
    for i in range(reg.shape[1]):
        reg_list.append(reg[:,i]>0)
    cells_in_all = np.all(np.array(reg_list),axis=0)
    print('no. registered cells: ',np.sum(cells_in_all))
    ind_list = []
    for i in range(reg.shape[1]):
        ind_list.append(reg[cells_in_all,i].astype(int)-1)#minus one because matlab using 1-indexing
    return ind_list 

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def plot_PETH(F,reach_starts,t_pre=15,t_post=30,sort_ind=None,norm=False):
    sum_F = np.zeros([F.shape[0],t_pre+t_post])
    for t in reach_starts.astype(int):
        if (t+t_post < (F.shape[1]-32)) and (t-t_pre > 32):
            sum_F = sum_F + F[:,(t-t_pre):(t+t_post)]
    avg_F = sum_F/len(reach_starts)
    #normalize ea cell's avg activity
    if norm:
        for neuron in range(avg_F.shape[0]):
            avg_F[neuron] = normalize(avg_F[neuron])
    #now sort
    if sort_ind is not None:
        #sort by given sorting indices
        avg_F_sorted = avg_F[sort_ind,:]
        return avg_F_sorted,sort_ind
    else:
        #sort by latency
        ind_maxF = np.argmax(avg_F,axis=1)
        avg_F_sorted = avg_F[np.argsort(ind_maxF),:]
        return avg_F_sorted, np.argsort(ind_maxF)

def get_labels_for_behavior(event_labels,beh):
    '''boolean inds for different behaviors to index event times'''
    if beh=='reach':
        labels = event_labels==0 
    elif beh == 'all_reach':
        labels = np.logical_or(np.logical_or(event_labels==0,event_labels==1),event_labels==2)
    elif beh=='grasp':
        labels = event_labels==1
    elif beh=='carry':
        labels = event_labels==2
    elif beh=='eating':
        labels = event_labels==5                    
    elif beh=='grooming':
        labels = event_labels==6
    elif beh=='fidget':
        labels = event_labels==4 
    elif beh == 'non_movement':
        labels = event_labels==3
    elif beh == 'reach_grasp':
        labels = np.logical_or(event_labels==0,event_labels==1)
    elif beh == 'non_reach':
        labels = np.logical_or(np.logical_or(event_labels==3,event_labels==4),
            np.logical_or(event_labels==5,event_labels==6))
    return labels 

def anova_reach_nonreach_PETH(spks,event_times,event_labels,t_pre=10,t_post=15):
    behavior_list = ['all_reach','reach','grasp','carry','non_movement','eating','grooming','fidget']

    P = np.zeros([len(behavior_list),spks.shape[0],t_pre+t_post]) #n behaviors x n cells x n timepts considered
    for i,beh in enumerate(behavior_list):
        labels = get_labels_for_behavior(event_labels,beh)
        sort_ind = np.arange(0,spks.shape[0],1)
        if np.any(event_times[labels]):
            P[i],ind = plot_PETH(spks,event_times[labels],t_pre=t_pre,t_post=t_post,sort_ind=sort_ind)

    F = stats.f_oneway(np.hstack((P[1],P[2],P[3])),np.hstack((P[4],P[5],P[6],P[7])),axis=1) #this should give me a stat for each cell
    return F

def anova_eating_PETH(spks,event_times,event_labels,t_pre=10,t_post=15):
    behavior_list = ['all_reach','reach','grasp','carry','non_movement','eating','grooming','fidget']

    P = np.zeros([len(behavior_list),spks.shape[0],t_pre+t_post]) #n behaviors x n cells x n timepts considered
    for i,beh in enumerate(behavior_list):
        labels = get_labels_for_behavior(event_labels,beh)
        sort_ind = np.arange(0,spks.shape[0],1)
        if np.any(event_times[labels]):
            P[i],ind = plot_PETH(spks,event_times[labels],t_pre=t_pre,t_post=t_post,sort_ind=sort_ind)

    F = stats.f_oneway(P[5],np.hstack((P[1],P[2],P[3],P[4],P[6],P[7])),axis=1) #this should give me a stat for each cell
    return F

def get_fr_for_behavior(spks, reach_starts,t_pre,t_post):
    #returns avg spks with shape n_cells x n_timepoints
    sum_spks = np.zeros([spks.shape[0],t_pre+t_post])
    for t in reach_starts.astype(int):
        if (t+t_post < (spks.shape[1]-32)) and (t-t_pre > 32):
            sum_spks = sum_spks + spks[:,(t-t_pre):(t+t_post)]
    avg_spks = sum_spks/len(reach_starts)
    return avg_spks

def get_trial_fr_for_behavior(spks, reach_starts,t_pre,t_post):
    #returns avg spks with shape n_cells x n_trials
    avg_spks = np.zeros([spks.shape[0],reach_starts.shape[0]])
    tr_time = (t_pre+t_post)/30
    for i,t in enumerate(reach_starts.astype(int)):
        if (t+t_post < (spks.shape[1]-32)) and (t-t_pre > 32):
            avg_spks[:,i] = np.nansum(spks[:,(t-t_pre):(t+t_post)],axis=1)/tr_time
    return avg_spks

def get_cat_spks_for_behavior(spks,reach_starts,t_pre,t_post):
    reach_starts = reach_starts[(reach_starts+t_post)<(spks.shape[1]-32)]
    reach_starts = reach_starts[(reach_starts-t_pre)>32]
    duration = t_pre+t_post
    spks_cat = np.zeros([spks.shape[0],reach_starts.shape[0]*duration])
    ind = np.arange(0,spks_cat.shape[1]+1,duration).astype(int)
    for i,t in enumerate(reach_starts.astype(int)):
        spks_cat[:,ind[i]:ind[i+1]] = spks[:,(t-t_pre):(t+t_post)]
    return spks_cat

def get_outlier_ind(reach_trials):
    #input in shape ncells x ntrials
    thresh = 1.5*(np.nanpercentile(reach_trials.flatten(),75)-np.nanpercentile(reach_trials.flatten(),25))
    ind = np.where(reach_trials>thresh)
    return ind

def get_outlier_masked_array(reach_trials):
    #input in shape ncells x ntrials
    thresh = 1.5*(np.nanpercentile(reach_trials.flatten(),75)-np.nanpercentile(reach_trials.flatten(),25))
    new_array = np.ma.masked_where(reach_trials>thresh,reach_trials)
    return new_array

def plot_residuals_and_std(reach_trials,other_trials):
    reach_trials = np.log10(reach_trials+1e-6)
    other_trials = np.log10(other_trials+1e-6)
    mean_reach = np.mean(reach_trials,axis=1)
    mean_other = np.mean(other_trials,axis=1)
    residuals_reach = np.zeros_like(reach_trials)
    residuals_other = np.zeros_like(other_trials)
    for n in range(reach_trials.shape[0]):
        residuals_reach[n] = reach_trials[n,:]-mean_reach[n]
        residuals_other[n] = other_trials[n,:]-mean_other[n]
    print(residuals_reach.shape)
    print(residuals_reach.T.shape)
    plt.figure(figsize=(20,15))
    plt.violinplot(residuals_reach.T)
    plt.title('Test Behavior')
    plt.figure(figsize=(20,15))
    plt.violinplot(residuals_other.T)
    plt.title('Other Behavior')
    stdev_reach = np.nanstd(reach_trials,axis=1)
    stdev_other = np.nanstd(other_trials,axis=1)
    plt.figure()
    plt.plot(np.zeros(stdev_reach.shape[0]),stdev_reach,'o',color='purple',alpha=0.2)
    plt.plot(np.ones(stdev_other.shape[0]),stdev_other,'o',color='red',alpha=0.2)
    plt.show()


def anova(reach_trials,other_trials,n_bootstraps = 100,subsample=False,remove_outliers=False,log=True):
    if remove_outliers:
        reach_trials = get_outlier_masked_array(reach_trials)
        other_trials = get_outlier_masked_array(other_trials)
    #plot_residuals_and_std(reach_trials,other_trials)
    if log:
        reach_trials = np.log10(reach_trials+1e-6)
        other_trials = np.log10(other_trials+1e-6)
    
    if subsample:
        p = np.zeros((reach_trials.shape[0],n_bootstraps))
        for n in range(n_bootstraps):
            if reach_trials.shape[1]<other_trials.shape[1]:
                subsample_ind = np.random.choice(other_trials.shape[1],size=reach_trials.shape[1],replace=False)
                other_trials = other_trials[:,subsample_ind]
            elif reach_trials.shape[1]>other_trials.shape[1]:
                subsample_ind = np.random.choice(reach_trials.shape[1],size=other_trials.shape[1],replace=False)
                reach_trials = reach_trials[:,subsample_ind]
            F = stats.f_oneway(reach_trials,other_trials,axis=1,nan_policy='omit') #this should give me a stat for each cell
            p[:,n] = F.pvalue
        mean_p = np.mean(p,axis=1)
        return mean_p
    else:
        F = stats.f_oneway(reach_trials,other_trials,axis=1,nan_policy='omit') #this should give me a stat for each cell
        return F.pvalue

def anova_reach_nonreach(spks,event_times,event_labels,t_pre=10,t_post=15):
    behavior_list = ['all_reach','reach','grasp','carry','non_movement','eating','grooming','fidget']
    fr = []
    for i,beh in enumerate(behavior_list):
        labels = get_labels_for_behavior(event_labels,beh)
        fr.append(get_trial_fr_for_behavior(spks,event_times[labels],t_pre,t_post))
    reach_trials = fr[0]
    other_trials =np.hstack((fr[4],fr[5],fr[6],fr[7]))
    print('n reach trials: ',reach_trials.shape[1])
    print('n other trials: ',other_trials.shape[1])
    p = anova(reach_trials,other_trials) #function corrects sample size
    return p 


def anova_eating(spks,event_times,event_labels,t_pre=10,t_post=15):
    behavior_list = ['all_reach','reach','grasp','carry','non_movement','eating','grooming','fidget']
    fr = []
    for i,beh in enumerate(behavior_list):
        labels = get_labels_for_behavior(event_labels,beh)
        fr.append(get_trial_fr_for_behavior(spks,event_times[labels],t_pre,t_post))
    eating_trials = fr[5]
    other_trials =np.hstack((fr[0],fr[4],fr[6],fr[7]))
    print('n eating trials: ',eating_trials.shape[1])
    print('n other trials: ',other_trials.shape[1])
    p = anova(eating_trials,other_trials)
    return p

def anova_reach_nonreach_inst_fr(spks,event_times,event_labels,t_pre,t_post):
    labels_reach = get_labels_for_behavior(event_labels,'all_reach')
    spks_reach = get_cat_spks_for_behavior(spks,event_times[labels_reach],t_pre,t_post)
    labels_non_reach = get_labels_for_behavior(event_labels,'non_reach')
    spks_non_reach = get_cat_spks_for_behavior(spks,event_times[labels_non_reach],t_pre,t_post)
    print('n time points reach/nonreach: ',spks_reach.shape[1],spks_non_reach.shape[1])
    p = anova(spks_reach,spks_non_reach)
    return p

def anova_eating_inst_fr(spks,event_times,event_labels,t_pre,t_post):
    labels_eating = get_labels_for_behavior(event_labels,'eating')
    spks_eating = get_cat_spks_for_behavior(spks,event_times[labels_eating],t_pre,t_post)
    labels_reach = get_labels_for_behavior(event_labels,'all_reach')
    spks_reach = get_cat_spks_for_behavior(spks,event_times[labels_reach],t_pre,t_post)
    print('n time points reach/nonreach: ',spks_reach.shape[1],spks_eating.shape[1])
    p = anova(spks_eating,spks_reach)
    return p

def get_reach_mod_cells(spks,event_times,event_labels,t_pre=10,t_post=15):
    #spks,event_times,event_labels = src.IO.load_spks_and_events(mouseID,day)
    F = anova_reach_nonreach_PETH(spks,event_times,event_labels,t_pre=t_pre,t_post=t_post)
    bool_ind = F.pvalue<(0.05/spks.shape[0])#with Bonferroni
    return bool_ind

def get_eating_mod_cells(spks,event_times,event_labels,t_pre=10,t_post=15):
    F = anova_eating_PETH(spks,event_times,event_labels,t_pre=t_pre,t_post=t_post)
    bool_ind = F.pvalue<(0.05/spks.shape[0])#with Bonferroni
    return bool_ind

def get_success_rate(event_labels):
    #these ind are of the first eating after a successful reach
    # Find indices where a 5 is preceded by 0, 1, or 2
    event_labels = event_labels.astype(int) #should already be ints, but just in case
    indices = np.where((event_labels[1:] == 5) & np.isin(event_labels[:-1], [0, 1, 2]))[0] + 1
    #print('ind of successes: ', indices)

    n_successes = indices.shape[0]
    #n_reaches_total = np.sum(np.where(np.isin(event_labels, [0, 1, 2]))[0])
    #success_rate = n_successes/(n_reaches_total-n_successes)
    #this is only an estimate, since sometimes a reach/grasp/carry sequence takes a while and is separated into
    #its components, and sometimes a single carry instance contains all of it
    #maybe in the future, count 0,1,2 sequences as a single reach as in: 

    # Find indices where the array has a sequence of [0, 1, 2] (returning the index of 2)
    seq_indices = np.where((event_labels[:-2] == 0) & (event_labels[1:-1] == 1) & (event_labels[2:] == 2))[0] + 2
    # Find indices where the array is equal to 0, 1, or 2 but not part of the sequence [0, 1, 2]
    individual_indices = np.where(np.isin(event_labels, [0, 1, 2]))[0]
    # Remove indices that are part of the sequence [0, 1, 2]
    individual_indices = np.setdiff1d(individual_indices, seq_indices)
    n_reaches = seq_indices.shape[0]+individual_indices.shape[0]
    print(n_reaches)
    success_rate = n_successes/(n_reaches-n_successes)
    return [success_rate, n_successes,n_reaches]


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def low_pass_filt(data,cutoff):
    # Filter requirements.
    order = 6
    fs = 200.0       # sample rate, Hz
    #cutoff = 10  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)
    y = butter_lowpass_filter(data, cutoff, fs, order)

    #as a hacky fix to filter edge effect, replace beginning datapts with original
    y[0:30] = data[0:30]
    return y
#LFILTER DOESN'T HANDLE MASKED DATA WELL; TRY SOMETHING LIKE THIS IF I DECIDE TO FILTER
def masked_lfilter(b, a, x_masked, fill_value=0):
    x_filled = x_masked.filled(fill_value)
    y = lfilter(b, a, x_filled)
    return np.ma.array(y, mask=x_masked.mask)

def get_spatial_bp_avg(df,bodyparts,pcutoff):
    x, y = src.IO.get_x_y(df, bodyparts[0], pcutoff)#just to get shape
    x_all_bp = np.ma.masked_all((len(bodyparts),x.shape[0]))
    y_all_bp = np.ma.masked_all((len(bodyparts),y.shape[0]))
    for b,bp in enumerate(bodyparts):
        x, y = src.IO.get_x_y(df, bp, pcutoff)
        x_all_bp[b,:] = x
        y_all_bp[b,:] = y
    x_avg = np.ma.mean(x_all_bp, axis=0)
    y_avg = np.ma.mean(y_all_bp, axis=0)
    assert np.all(x_avg.shape==x.shape)
    assert (np.ma.max(x_avg)<1450) and (np.ma.max(y_avg)<1090),f'{np.max(x_avg)},{np.max(y_avg)}'
    return x_avg,y_avg

def get_kinematic_trials_xy(x,y,cam_event_times,duration):
    #from vectors x and y, get matrices n_trials x duration
    assert cam_event_times.size >0,f'{cam_event_times}'
    x_mat = np.ma.masked_all((cam_event_times.shape[0],duration))
    y_mat = np.ma.masked_all((cam_event_times.shape[0],duration))
    assert (np.max(x)<1450) and (np.max(y)<1090),f'{np.max(x)} {np.max(y)}'
    for s,start in enumerate(cam_event_times):
        x_mat[s,:] = x[start:start+duration]
        y_mat[s,:] = y[start:start+duration]
    assert (np.ma.max(x_mat)<1440) and (np.ma.max(y_mat)<1080),f'{np.max(x_mat)} {np.max(y_mat)}'
        
    return x_mat,y_mat

def get_paw_avg_mat(mouseID,day,cam_event_times,duration,pcutoff):
    x_paw_all = []
    y_paw_all = []
    for key in cam_event_times.keys():
        if cam_event_times[key].size >0:
            df_cam1, df_cam2 = src.IO.load_kinematics_df(key,mouseID,day)
            x_paw,y_paw = get_spatial_bp_avg(df_cam1,['d1middle',
                'd2tip','d2middle','d2knuckle','d3tip','d3middle',
                'd3knuckle','d4tip','d4middle','wrist','wrist_outer'],pcutoff)
            x_paw_mat,y_paw_mat = get_kinematic_trials_xy(x_paw,y_paw,cam_event_times[key],duration)
            assert (np.max(x_paw_mat)<1440) and (np.max(y_paw_mat)<1080),f'{np.max(x_paw_mat)} {np.max(y_paw_mat)}'
            x_paw_all.append(x_paw_mat)
            y_paw_all.append(y_paw_mat)
    paw_avg_mat_x = np.ma.concatenate(x_paw_all,axis=0)
    paw_avg_mat_y = np.ma.concatenate(y_paw_all,axis=0)
    assert (np.max(paw_avg_mat_x)<1440) and (np.max(paw_avg_mat_y)<1080),f'{np.max(paw_avg_mat_x)} {np.max(paw_avg_mat_y)}'
    return paw_avg_mat_x, paw_avg_mat_y

def get_pedestal_avg(mouseID,day,cam_event_times,cam,pcutoff):
    x_all = 0
    y_all = 0
    for key in cam_event_times.keys():
        df_cam1,df_cam2 = src.IO.load_kinematics_df(key,mouseID,day)
        if cam==1:
            x,y = src.IO.get_x_y(df_cam1,'pedestal',pcutoff)
        elif cam==2:
            x,y = src.IO.get_x_y(df_cam2,'pedestal',pcutoff)
        x_all+=np.ma.mean(x)
        y_all+=np.ma.mean(y)
    x_mean = x_all/len(cam_event_times.keys())
    y_mean = y_all/len(cam_event_times.keys())
    return x_mean,y_mean

def get_flat_upper_tri(matrix):
    return matrix[np.triu_indices(matrix.shape[0],k=1)].flatten()

def plot_adaptive_binning(spks,log_spks,bins,binned_spks,dF):
    fig,ax = plt.subplots(5,1,tight_layout=True,figsize=(20,15))
    ax[2].ecdf(log_spks)
    ax[2].set_xlabel('log of est spk count',fontsize=16)
    ax[2].set_ylabel('cdf',fontsize=16)
    ax[3].hist(spks[32:-32],bins=bins)
    ax[3].vlines(bins,0,9000,color='red')
    ax[3].set_xlabel('est spk count',fontsize=16)
    ax[3].set_ylabel('counts',fontsize=16)
    ax[0].plot(dF[32:-32],color='purple')
    ax[0].plot(spks[32:-32],color='teal')
    ax[0].set_xlabel('frame',fontsize=16)
    ax[0].set_ylabel('dF/F',fontsize=16)
    ax[1].plot(spks[32:-32],color='teal')
    ax[1].set_xlabel('frame',fontsize=16)
    ax[1].set_ylabel('est spk count',fontsize=16)
    ax[4].plot(binned_spks,color='teal')
    ax[4].set_xlabel('frame',fontsize=16)
    ax[4].set_ylabel('bin',fontsize=16)
    ax[0].tick_params(labelsize=16)
    ax[1].tick_params(labelsize=16)
    ax[2].tick_params(labelsize=16)
    ax[3].tick_params(labelsize=16)
    ax[4].tick_params(labelsize=16)
    plt.show()

def bin_spks_adaptive(spks,dF,thresh,plot=False):
    #takes 1 spike train at a time, so an array of length n_timepoints
    spks[spks<thresh] = 0
    spks_nonzero = spks[spks>1e-6]
    log_spks = np.log10(spks_nonzero)
    res = stats.ecdf(log_spks)
    quant = res.cdf.quantiles
    prob = res.cdf.probabilities
    prob_bins = [np.min(prob),0.2,0.4,0.6,0.8,1]
    atol=np.max(np.diff(prob))
    bins = [quant[np.argwhere(np.isclose(prob,prob_bins[i],atol=atol))[0]][0] for i in range(len(prob_bins))]
    bins = 10**np.array(bins)
    bins = np.insert(bins,0,0)
    bins[-1] = np.max(spks)+1e-6
    binned_spks = np.digitize(spks,bins)
    binned_spks = binned_spks-1 #0 indexing will make things easier
    if plot:
        plot_adaptive_binning(spks,log_spks,bins,binned_spks,dF)

    return binned_spks

# def compute_MI_graph_og(binned_spks):
#     n_neurons = binned_spks.shape[0]
#     MI_graph = np.zeros([n_neurons,n_neurons]) 
#     binned_values = np.unique(binned_spks[0])
#     p_joint = np.zeros([binned_values.shape[0],binned_values.shape[0]])
#     for x in range(n_neurons):
#         for y in range(n_neurons):
#             print(x,y)
#             for i in binned_values:
#                 for j in binned_values:
#                     p_x = np.sum(binned_spks[x]==i)/binned_spks.shape[1]
#                     p_y = np.sum(binned_spks[y]==j)/binned_spks.shape[1]
#                     p_joint[int(i),int(j)] = np.sum(np.logical_and(binned_spks[x]==i,binned_spks[y]==j))/binned_spks.shape[1]
#                     MI_graph[x,y] += p_joint[int(i),int(j)]*np.log2(1e-6+p_joint[int(i),int(j)]/(p_x*p_y))
#     return MI_graph


def compute_MI_graph(binned_spks):
    n_neurons = binned_spks.shape[0]
    n_timepoints = binned_spks.shape[1]
    MI_graph = np.zeros([n_neurons,n_neurons])  
    binned_values = np.unique(binned_spks[0])
    n_bins = binned_values.shape[0]
    # Precompute marginal probabilities
    p_all = np.zeros((n_neurons, binned_values.shape[0]))
    for n in range(n_neurons):
        counts, _ = np.histogram(binned_spks[n], bins=np.arange(n_bins + 1) - 0.5)
        p_all[n] = counts / n_timepoints
    for x in range(n_neurons):
        for y in range(n_neurons):
            # Joint histogram
            joint_hist, _, _ = np.histogram2d(
                binned_spks[x],
                binned_spks[y],
                bins=(np.arange(n_bins + 1) - 0.5, np.arange(n_bins + 1) - 0.5)
            )
            p_joint = joint_hist / n_timepoints
            # Outer product of marginals
            p_product = np.outer(p_all[x], p_all[y]) + 1e-12
            # Mask zeros to avoid log(0)
            nonzero = p_joint > 0
            MI_graph[x, y] = np.sum(
                p_joint[nonzero] * np.log2(p_joint[nonzero] / p_product[nonzero])
            )
    return MI_graph 

def compute_MI_graph_for_behavior(binned_spks,s2p_fld,beh,t_pre,t_post):
    event_labels = np.load(f"{s2p_fld}/event_labels.npy")
    reach_starts = np.load(f'{s2p_fld}/calcium_event_times.npy')
    labels_beh = get_labels_for_behavior(event_labels,beh)
    spks_beh = get_cat_spks_for_behavior(binned_spks,reach_starts[labels_beh],
        t_pre,t_post)
    MI_graph = compute_MI_graph(spks_beh)
    return MI_graph