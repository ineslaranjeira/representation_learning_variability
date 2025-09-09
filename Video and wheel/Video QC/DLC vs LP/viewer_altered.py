import numpy as np
from one.api import ONE
from pathlib import Path
import cv2
import os,fnmatch
import matplotlib
import pandas as pd
# conda install -c conda-forge pyarrow
import os
from ibldsp.smooth import smooth_interpolate_savgol
from brainbox.io.one import SessionLoader
from copy import deepcopy


#one = ONE(base_url='https://openalyx.internationalbrainlab.org',
#      password='international', silent=True)

one = ONE()

def Find(pattern, path):

    '''
    find a local video like so:
    flatiron='/home/mic/Downloads/FlatIron'      
    vids = Find('*.mp4', flatiron)
    '''
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# def load_lp(eid, cam, masked=True, paws=True,
#             reso='128x102_128x128', flav='multi'):

#     '''
#     for a given session and cam, load all lp tracked points;
#     that's paw specific now; 
#     flav is either single or multi view EKS
#     '''
    
#     print(f'loading LP, {reso}, {cam}')
#     print(f'{flav}, paws:{paws}, {eid}')
    
#     if paws:
    
#         pth = ('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/lp_ens'
#               f'/{reso}/{eid}/ensembles_{cam}Camera/'
#               f'_iblrig_{cam}Camera.raw.paws.eks_{flav}.csv') 

#         d0 = pd.read_csv(pth, low_memory=False)


#         if reso[:7] == '128x102':
#             scale = 10 if cam == 'left' else 5
#         else:    
#             scale = 4 if cam == 'left' else 2

#         print('scale', scale)
       
#         # concat column keys
#         d = {}
#         for k in d0:
#             if (d0[k][1] in ['x','y']):
#                 d[d0[k][0]+'_'+d0[k][1]] = scale * np.array(
#                                                d0[k][2:].values, 
#                                                dtype=np.float32)
#             else:
#                 d[d0[k][0]+'_'+d0[k][1]] = np.array(
#                                                d0[k][2:].values, 
#                                                dtype=np.float32)                
          
             
#         del d['bodyparts_coords']
        
# #        k0 = list(d.keys())
# #        for k in k0:
# #            if 'likelihood' in k:
# #                del d[k]    

    
#     d['times'] = np.load(one.eid2path(eid) / 'alf'
#                     / f'_ibl_leftCamera.times.npy')
                    

#     ls = [len(d[x]) for x in d]
#     if not all(ls == np.mean(ls)):
#         lsd = {x:len(d[x]) for x in d}
#         print(f'length mismatch: {lsd}')
#         print(eid, cam)
#         print('cutting times')
#         d['times'] = d['times'][:ls[0]]

#     if (not paws and reso == '128x102_128x128'):
#         # load old complete lp file        
#         pth = ('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/lp_ens'
#               f'/{reso}/{eid}/_ibl_{cam}Camera.lightningPose.pqt') 

#         d = pd.read_parquet(pth)    

#         if masked:
#             points = np.unique(['_'.join(x.split('_')[:-1]) 
#                                 for x in d.keys()])[1:]
        
#             for point in points:
#                 cond = d[f'{point}_likelihood'] < 0.9
#                 d.loc[cond, [f'{point}_x', f'{point}_y']] = np.nan

#     return d


# def load_dlc(eid, cam, smoothed=False, manual=True):

#     '''
#     cam in left, right, body 
#     '''

#     if manual:
#         pth = one.eid2path(eid)    
#         d = pd.read_parquet(pth / 'alf' / f'_ibl_{cam}Camera.dlc.pqt')
#         d['times'] = np.load(one.eid2path(eid) / 'alf'
#                     / f'_ibl_{cam}Camera.times.npy')
                    
#         ls = [len(d[x]) for x in d]
#         if not all(ls == np.mean(ls)):
#             lsd = {x:len(d[x]) for x in d}
#             print(f'length mismatch: {lsd}')
#             print(eid, cam)
#             print('cutting times')
#             d['times'] = d['times'][:ls[0]]            

#     else:
#         # load DLC
#         sess_loader = SessionLoader(one, eid)
#         sess_loader.load_pose(views=[cam])
#         d = sess_loader.pose[f'{cam}Camera']
    
#     if smoothed:
#         print('smoothing dlc traces')
#         window = 13 if cam == 'right' else 7
#         sers = [x for x in d.keys() if (x[-1] in ['x','y'])]# and 'paw' in x
#         for ser in sers:
#             d[ser] = smooth_interpolate_savgol(
#                 d[ser].to_numpy(),
#                 window=window,order=3, interp_kind='linear')   

#     return d




def Viewer(eid, video_type, cam, frame_start, frame_stop, save_video=True, 
           eye_zoom=False, masked=True, paws_only=False):

# def Viewer(eid, video_type, cam, frame_start, frame_stop, save_video=True, 
#            eye_zoom=False, lp=False, ens=False,
#            res = '128x102_128x128', masked=True, paws_only=False,
#            smooth_dlc = False):
           
    '''
    eid: session id, e.g. '3663d82b-f197-4e8b-b299-7b803a155b84'
    video_type: one of 'left', 'right', 'body'
    save_video: video is saved this local folder

    Example usage to view and save labeled video with wheel angle:
    Viewer('3663d82b-f197-4e8b-b299-7b803a155b84', 'left', [5,7])
    3D example: 'cb2ad999-a6cb-42ff-bf71-1774c57e5308', [5,7]
    
    Different resolutions:
    128x102_128x128
    320x256_128x128
    320x256_256x256
    
    
    paws: paws only
    '''

    save_vids_here = Path.home()


    alf_path = one.eid2path(eid)

    # Download a single video
    video_path = (alf_path / 
        f'raw_video_data/_iblrig_{video_type}Camera.raw.mp4')
    
    if not os.path.isfile(video_path):
        print('mp4 not found locally, downloading it ...')
        video_path = one.load_dataset(eid,
            f'raw_video_data/_iblrig_{video_type}Camera.raw.mp4',
            download_only=True)

    # Download DLC traces and stamps
    Times = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.times.npy')

                                         
    # get video info
    # cap = cv2.VideoCapture(video_path.as_uri())
    cap = cv2.VideoCapture(video_path.as_uri()[7:])  # Ines edited
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(3)), int(cap.get(4)))


    print(eid,
          ', ',
          video_type,
          ', fsp:',
          fps,
          ', #frames:',
          length,
          ', #stamps:',
          len(Times),
          ', #frames - #stamps = ',
          length - len(Times))

    # # pick trial range for which to display stuff
    # trials = one.load_object(eid, 'trials', download_only=True)

    print('frame start stop', frame_start, frame_stop)

    '''
    load wheel
    '''

    wheel = one.load_object(eid, 'wheel')
    import brainbox.behavior.wheel as wh
    try:
        pos, t = wh.interpolate_position(
            wheel['timestamps'], wheel['position'], freq=1000)
    except BaseException:
        pos, t = wh.interpolate_position(
            wheel['times'], wheel['position'], freq=1000)

    w_start = find_nearest(t, Times[frame_start])
    w_stop = find_nearest(t, Times[frame_stop])

    # confine to interval
    pos_int = pos[w_start:w_stop]
    t_int = t[w_start:w_stop]

    # alignment of cam stamps and interpolated wheel stamps
    wheel_pos = []
    kk = 0
    for wt in Times[frame_start:frame_stop]:
        wheel_pos.append(pos_int[find_nearest(t_int, wt)])
        kk += 1
        if kk % 3000 == 0:
            print('iteration', kk)

    '''
    DLC related stuff
    '''
    Times = Times[frame_start:frame_stop]
    Frames = np.arange(frame_start, frame_stop)
    
    points = [x[:-2] for x in cam.keys() if x[-1] == 'x']
    points = np.array(points)
          
    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.array(cam[point + '_x'])
        y = np.array(cam[point + '_y'])
        XYs[point] = np.array(
            [x[frame_start:frame_stop], y[frame_start:frame_stop]])        
        

    # Zoom at eye
    if eye_zoom:
        pivot = np.nanmean(XYs['pupil_top_r'], axis=1)
        x0 = int(pivot[0]) - 33
        x1 = int(pivot[0]) + 33
        y0 = int(pivot[1]) - 28
        y1 = int(pivot[1]) + 38
        size = (66, 66)
        dot_s = 1  # [px] for painting DLC dots

    else:
        x0 = 0
        x1 = size[0]
        y0 = 0
        y1 = size[1]
        if video_type == 'left':
            dot_s = 10  # [px] for painting DLC dots
        else:
            dot_s = 5

    if save_video:
    
        # rr = f'_{res}' if ens else ''
        loc = (save_vids_here / 
        f'{eid}_{video_type}_frames_{frame_start}_{frame_stop}.mp4')

        out = cv2.VideoWriter(str(loc),
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps,
                              size)  # put , 0 if grey scale

    # writing stuff on frames
    font = cv2.FONT_HERSHEY_SIMPLEX

    if video_type == 'left':
        bottomLeftCornerOfText = (20, 1000)
        fontScale = 4
    else:
        bottomLeftCornerOfText = (10, 500)
        fontScale = 2

    lineType = 2

    # assign a color to each DLC point (now: all points red)
    cmap = matplotlib.cm.get_cmap('Set1')
    CR = np.arange(len(points)) / len(points)

    block = np.ones((2 * dot_s, 2 * dot_s, 3))

    # set start frame
    cap.set(1, frame_start)

    k = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = frame

        # print wheel angle
        fontColor = (255, 255, 255)
        Angle = round(wheel_pos[k], 2)
        Time = round(Times[k], 3)
        cv2.putText(gray,
                    'Wheel angle: ' + str(Angle),
                    bottomLeftCornerOfText,
                    font,
                    fontScale / 2,
                    fontColor,
                    lineType)

        a, b = bottomLeftCornerOfText
        bottomLeftCornerOfText0 = (int(a * 10 + b / 2), b)
        cv2.putText(gray,
                    '  time: ' + str(Time),
                    bottomLeftCornerOfText0,
                    font,
                    fontScale / 2,
                    fontColor,
                    lineType)
                    
                    
        bottomLeftCornerOfText1 = (a, b - 3* a)
        cv2.putText(gray,
                    'Frame: ' + str(Frames[k]),
                    bottomLeftCornerOfText1,
                    font,
                    fontScale / 2,
                    fontColor,
                    lineType)                    
                    
                    

        # print DLC dots
        ll = 0
        for point in points:

            # Put point color legend
            fontColor = (np.array([cmap(CR[ll])]) * 255)[0][:3]
            a, b = bottomLeftCornerOfText
            if video_type == 'right':
                bottomLeftCornerOfText2 = (a, a * 2 * (1 + ll))
            else:
                bottomLeftCornerOfText2 = (b, a * 2 * (1 + ll))
            fontScale2 = fontScale / 4
            if np.isin(point, ['paw_l', 'rec_paw_l']):
                fontColor = (np.array([cmap(CR[0])]) * 255)[0][:3]
            elif np.isin(point, ['paw_r', 'rec_paw_r']):
                fontColor = (np.array([cmap(CR[1])]) * 255)[0][:3]

            cv2.putText(gray, point,
                        bottomLeftCornerOfText2,
                        font,
                        fontScale2,
                        fontColor,
                        lineType)

            X0 = XYs[point][0][k]
            Y0 = XYs[point][1][k]
            # transform for opencv?
            X = Y0
            Y = X0

            if not np.isnan(X) and not np.isnan(Y):
                col = (np.array([cmap(CR[ll])]) * 255)[0][:3]
                if np.isin(point, ['paw_l', 'rec_paw_l']):
                    col = (np.array([cmap(CR[0])]) * 255)[0][:3]
                elif np.isin(point, ['paw_r', 'rec_paw_r']):
                    col = (np.array([cmap(CR[1])]) * 255)[0][:3]


                # col = np.array([0, 0, 255]) # all points red
                X = X.astype(int)
                Y = Y.astype(int)
                
                if np.isin(point, ['paw_l', 'paw_r']):
                    gray[X - dot_s:X + dot_s, Y - 
                        dot_s:Y + dot_s] = block * col
                elif np.isin(point, ['rec_paw_l', 'rec_paw_r']):
                    cv2.circle(gray, (Y, X), dot_s, col.tolist(), -1)            

            ll += 1

        gray = gray[y0:y1, x0:x1]
        if save_video:
            out.write(gray)
        #cv2.imshow('frame', gray)
        #cv2.waitKey(1)
        k += 1
        if k == (frame_stop - frame_start) - 1:
            break

    if save_video:
        out.release()
    cap.release()
    #cv2.destroyAllWindows()
    
    print(eid, video_type, frame_stop, frame_start)
    #return XYs, Times

prefix = '/home/ines/repositories/'
path = prefix + 'representation_learning_variability/Video and wheel/Video QC/DLC vs LP/'
cam = pd.read_parquet(path+'rightCamera_keypoints')
Viewer('5b49aca6-a6f4-4075-931a-617ad64c219c', 'right', cam, 10000, 13920, 
       save_video=True, eye_zoom=False, masked=True, paws_only=False)