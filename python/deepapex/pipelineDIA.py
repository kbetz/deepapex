import numpy as np
import pandas as pd
import sqlite3 as sql
import tensorflow as tf

pd.options.mode.chained_assignment = None#default='warn'

from tqdm import tqdm_notebook as tqdm
from datetime import datetime
import sys, gc

from deepapex import model
from proteolizarddata.data import PyTimsDataHandleDIA
from proteolizardalgo.clustering import cluster_precursors_hdbscan, cluster_precursors_dbscan
from proteolizardalgo.hashing import TimsHasher

BATCH_SIZE = 1
MAX_N = 1024
ATT_SIZE = 2
POINT_DIM = 4
NUM_SCORES = 6

HASHER = TimsHasher(trials=32, len_trial=32, seed=42, num_dalton=10, resolution=1)

RTs = np.linspace(0.0, 48.0, 24 + 1) * 60
RT_epsilon = (RTs[1] - RTs[0]) / 2
MZs = np.linspace(0.0, 1700.0, 170*5 + 1)
MZ_epsilon = (MZs[1] - MZs[0]) / 2

RT_loadsize = 2
RT_slices = [(max(i-1, 0), min(i+RT_loadsize+1, len(RTs)-1)) for i in range(0,len(RTs)-RT_loadsize, RT_loadsize)]

CLASS_SHARES = np.array([0.99, 0.002, 0.0045, 0.003, 0.0003, 0.0003])
CLASS_WEIGHT = dict(zip(range(len(CLASS_SHARES)), 1-CLASS_SHARES))


def hash_slice(slic):
    points = pd.DataFrame()
    #for fid in frames:
    for frame in slic.get_precursor_frames():
        #frame = cursor.get_frame(fid).filter_ranged(scan_min=0, scan_max=500, mz_min=MZs[j], mz_max=MZs[j+1])
        try:
            hash_res = HASHER.filter_frame_auto_correlation(frame).data()
        except IndexError:
            continue
        points = pd.concat([points, hash_res])
        
    return points

def run(data_path, feature_path, isodet, train=True):
    
    cursor = PyTimsDataHandleDIA(data_path)
    with sql.connect(feature_path) as con:
        peak_data = pd.read_sql_query("SELECT * from LcTimsMsFeature", con)

    INT_MIN = 100
    INT_MAX = peak_data.Intensity.max()
    
    if train:
        log_path = "../../proteologs/" + str(datetime.now()) + ".log"
    else:
        y_true = np.array([])
        y_pred = np.array([])

    # load larger slices
    for rt_idx_lower, rt_idx_upper in RT_slices:
        
        if rt_idx_lower > 0:
            rt_idx_lower += 1
        if rt_idx_upper < len(RTs)-1:
            rt_idx_upper -= 1
            
        slic = cursor.get_slice_rt_range(rt_min=RTs[rt_idx_lower], rt_max=RTs[rt_idx_upper])
        

        temp = slic.get_precursor_points()
        print("NUM_POINTS:",len(temp))
        print("BYTE_SIZE(MB):",sys.getsizeof(temp)/2**20)
        print("----------")
        
        del slic
        gc.collect()
        continue

        #del temp
        #del slic

        scan2iim = slic.get_precursor_points()[["inv_ion_mob","scan"]].values

        # hashing
        points = hash_slice(slic)

        if len(points) == 0:
            continue

        # intensity filter
        points["rt"] = cursor.frames_to_rts(points["frame"])
        points = points[["rt","scan","mz","intensity"]]
        points_filt = points[points["intensity"] > INT_MIN]

        if len(points_filt) == 0:
                continue

        # all rt windows in current slice
        for i in tqdm(np.random.permutation(range(rt_idx_lower, rt_idx_upper))):

            # all mz windows
            for j in tqdm(np.random.permutation(range(len(MZs)-1))):

                # extract points in window
                points_window = points_filt.query(f"{RTs[i]} <= rt and rt < {RTs[i+1]} \
                                                    and {MZs[j]} <= mz and mz < {MZs[j+1]}")
                
                if len(points_window) == 0:
                    continue

                points_north = points_filt.query(f"{RTs[i] - RT_epsilon} <= rt and rt < {RTs[i]} \
                                                   and {MZs[j]} <= mz and mz < {MZs[j+1]}")

                points_south = points_filt.query(f"{RTs[i+1]} <= rt and rt < {RTs[i+1] + RT_epsilon} \
                                                   and {MZs[j]} <= mz and mz < {MZs[j+1]}")

                points_east = points_filt.query(f"{RTs[i]} <= rt and rt < {RTs[i+1]} \
                                                  and {MZs[j] - MZ_epsilon} <= mz and mz < {MZs[j]}")

                points_west = points_filt.query(f"{RTs[i]} <= rt and rt < {RTs[i+1]} \
                                                  and {MZs[j+1]} <= mz and mz < {MZs[j+1] + MZ_epsilon}")

                points_att = pd.concat([points_north, points_south, points_east, points_west])

                # create bounding boxes
                peak_filt = peak_data.query(f"{RTs[i]} <= RT and RT < {RTs[i+1]} \
                                          and {MZs[j]} <= MZ and MZ < {MZs[j+1]}")

                boxes = peak_filt[["RT_lower","RT_upper","Mobility_lower","Mobility_upper","MZ_lower","MZ","MZ_upper","Charge"]]
                boxes.loc[:,"Mobility_lower"] = boxes["Mobility_lower"].apply(lambda x: scan2iim[min(range(len(scan2iim)), key = lambda i: abs(scan2iim[i][0]-x))][1])
                boxes.loc[:,"Mobility_upper"] = boxes["Mobility_upper"].apply(lambda x: scan2iim[min(range(len(scan2iim)), key = lambda i: abs(scan2iim[i][0]-x))][1])
                boxes[["Mobility_lower","Mobility_upper"]] = boxes[["Mobility_upper","Mobility_lower"]]  

                if len(boxes) == 0:
                    continue

                # label points according to boxes
                points_window["charge"] = 0

                for _, box in boxes.iterrows():
                    box_query = f"{box.RT_lower} < rt and rt < {box.RT_upper} \
                              and {box.Mobility_lower} < scan and scan < {box.Mobility_upper} \
                              and {box.MZ_lower} < mz and mz < {box.MZ_upper} \
                              and {box.Charge} > charge"
                    points_window.loc[points_window.eval(box_query), "charge"] = box.Charge

                # scaling
                points_window["rt"] = points_window["rt"].subtract(RTs[i])
                points_window["mz"] = points_window["mz"].subtract(MZs[j])

                points_att["rt"] = points_att["rt"].subtract(RTs[i])
                points_att["mz"] = points_att["mz"].subtract(MZs[j])

                points_window.intensity

                #print("TOTAL_POINTS:", len(points_window))

                # cutoff or pad
                if len(points_window) >= MAX_N:
                    #all_values = points_window.values[:MAX_N,:]
                    nlargest = points_window.nlargest(MAX_N, "intensity")
                    #print("SMALLEST_INT:", nlargest.intensity.min())
                    values_window = nlargest.values
                else:
                    pad_len = MAX_N - len(points_window)
                    values_window = np.pad(points_window, ((0,pad_len), (0,0)), "edge")

                # cutoff or pad for sorrounding points
                if len(points_att) >= MAX_N * ATT_SIZE:
                    #all_values = points_window.values[:MAX_N,:]
                    nlargest = points_att.nlargest(MAX_N * ATT_SIZE, "intensity")
                    values_att = nlargest.values
                else:
                    pad_len = MAX_N * ATT_SIZE - len(points_att)
                    values_att = np.pad(points_att, ((0,pad_len), (0,0)), "constant")


                # separate features and labels
                features = values_window[:,:-1]
                labels = values_window[:,-1]

                # charge numbers to one hot vectors
                labels_onehot = np.eye(NUM_SCORES)[labels.astype(int)] # tf.one_hot()
                
                # class weights
                #shares = np.sum(labels_onehot, axis=0)/MAX_N
                #class_weight = dict(zip(range(len(shares)), 1-shares))
                sample_weight = np.vectorize(CLASS_WEIGHT.__getitem__)(labels).reshape(BATCH_SIZE, MAX_N)

                x_train = np.concatenate([features, values_att]).reshape(BATCH_SIZE, MAX_N*(ATT_SIZE+1), POINT_DIM)

                # intensity scaling
                x_train[:,3] *= 255/INT_MAX

                y_train = labels_onehot.reshape(BATCH_SIZE, MAX_N, NUM_SCORES)

                #class_dist = np.sum(y_train, axis=-2)[0]
                #if class_dist[0] < 1024:
                #    print(class_dist)

                #print(y_train.shape)

                if train:
                    isodet.fit(x_train,
                               y_train,
                               batch_size=BATCH_SIZE,
                               verbose=0,
                               epochs=1,
                               callbacks=model.AddOns.get_callbacks(log_path),
                               sample_weight=sample_weight)
                else:
                    yp = isodet.predict(x_train,
                                        batch_size=BATCH_SIZE,
                                        verbose=0)
                    y_true = np.append(y_true, labels)
                    y_pred = np.append(y_pred, yp)

    if not train:
        return y_true, y_pred
        
def run_train(data_paths, feature_paths, isodet):
    
    for dp, fp in zip(data_paths, feature_paths):
        run(dp, fp, isodet, True)
        print("=== FINISHED FILE ===")
        
def run_test(data_paths, feature_paths, isodet):
    
    for dp, fp in zip(data_paths, feature_paths):
        run(dp, fp, isodet, False)
        print("=== FINISHED FILE ===")