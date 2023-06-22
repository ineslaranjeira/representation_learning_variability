
## All functions from brainbox

import numpy as np
from one.api import ONE
one = ONE() 


def get_dlc_XYs(eid, video_type, query_type='remote'):

    #video_type = 'left'    
    Times = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.times.npy',
                             query_type=query_type) 
    cam = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.dlc.pqt', 
                           query_type=query_type)
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])

    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array(
            [x, y])    

    return Times, XYs  


def get_licks(XYs):

    '''
    define a frame as a lick frame if
    x or y for left or right tongue point
    change more than half the sdt of the diff
    '''  
    
    licks = []
    for point in ['tongue_end_l', 'tongue_end_r']:
        for c in XYs[point]:
           thr = np.nanstd(np.diff(c))/4
           licks.append(set(np.where(abs(np.diff(c))>thr)[0]))
    return sorted(list(set.union(*licks))) 


def get_lick_times(eid, combine=False, video_type='left'):
    
    if combine:    
        # combine licking events from left and right cam
        lick_times = []
        for video_type in ['right','left']:
            times, XYs = get_dlc_XYs(eid, video_type)
            r = get_licks(XYs)
            # cover case that there are less times than DLC points            
            idx = np.where(np.array(r)<len(times))[0][-1]            
            lick_times.append(times[r[:idx]])
        
        lick_times = sorted(np.concatenate(lick_times))
        
    else:
        times, XYs = get_dlc_XYs(eid, video_type)    
        r = get_licks(XYs)
        # cover case that there are less times than DLC points
        idx = np.where(np.array(r)<len(times))[0][-1]              
        lick_times = times[r[:idx]]

    return lick_times
        