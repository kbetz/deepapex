import numpy as np
import pandas as pd
import tensorflow as tf

pd.options.mode.chained_assignment = None#default='warn'

from tqdm import tqdm_notebook as tqdm
from datetime import datetime
from scipy.sparse import csr_array
from sklearn.metrics.pairwise import cosine_similarity

from deepapex import model
from proteolizarddata.data import PyTimsDataHandleDDA, MzSpectrum
from proteolizardalgo.hashing import TimsHasher
from proteolizardalgo.isotopes import generate_pattern

# CONSTANTS
BATCH_SIZE = 1
MAX_N = 1024
ATT_SIZE = 1
POINT_DIM = 3
NUM_SCORES = 6

NUM_PEAKS = 8
BUFFER = 0.1
STEP_SIZE = 1e-5

SIMIL_THOLD = 0.5
NUM_MZ_BINS = int(2e6)
SCAN_RANGE = 15

MZs = np.linspace(0.0, 1700.0, 20 + 1)
MZ_epsilon = (MZs[1] - MZs[0]) / 2

CLASS_SHARES = np.array([0.99, 0.002, 0.0045, 0.003, 0.0003, 0.0003])
CLASS_WEIGHT = dict(zip(range(len(CLASS_SHARES)), 1-CLASS_SHARES))

def run(path, isodet, train=True):
    
    cursor = PyTimsDataHandleDDA(path)
    selected = cursor.get_selected_precursors()
    hasher = TimsHasher(trials=32, len_trial=32, seed=42, num_dalton=10, resolution=1)
    
    selected = cursor.get_selected_precursors()

    INT_MIN = 100
    INT_MAX = selected.Intensity.max()
    
    
    if train:
        log_path = "../../proteologs/" + str(datetime.now()) + ".log"
    else:
        y_true = np.array([])
        y_pred = np.array([])

    for fid in tqdm(np.random.permutation(cursor.precursor_frames)):
        feature = selected[selected.Parent == fid].reset_index(drop=True)
        spec = feature.apply(lambda prec: generate_pattern(prec.MonoisotopicMz - BUFFER,
                                                           prec.MonoisotopicMz + NUM_PEAKS/prec.Charge + BUFFER,
                                                           STEP_SIZE,
                                                           prec.MonoisotopicMz*prec.Charge,
                                                           prec.Charge,
                                                           prec.Intensity,
                                                           NUM_PEAKS),
                             axis=1).values

        frame = cursor.get_frame(fid)
        try:
            hashed_frame = hasher.filter_frame_auto_correlation(frame)
        except IndexError:
            #print("EMPTY FRAME")
            continue

        points = hashed_frame.data()[["mz", "scan", "intensity"]]
        points_filt = points[points.intensity >= INT_MIN].reset_index(drop=True)
        labels = np.zeros((len(points_filt), NUM_SCORES))

        # get labels from features
        for idx, row in feature.iterrows():

            lower_mz = row.MonoisotopicMz - BUFFER
            upper_mz = row.MonoisotopicMz + NUM_PEAKS/row.Charge + BUFFER

            hashed_slice = hashed_frame.filter_ranged(scan_min=int(row.ScanNumber) - SCAN_RANGE,
                                                      scan_max=int(row.ScanNumber) + SCAN_RANGE,
                                                      mz_min=lower_mz,
                                                      mz_max=upper_mz)


            hashed_specs = hashed_slice.get_spectra()
            vals = np.concatenate([sp.vectorize(3).values()  for sp in hashed_specs])
            rows = np.concatenate([[i]*len(sp.mz())          for i, sp in enumerate(hashed_specs)])
            idxs = np.concatenate([sp.vectorize(3).indices() for sp in hashed_specs])

            if idxs[0] < 0:
                continue

            data_specs = csr_array((vals, (rows, idxs)), shape=(rows[-1]+1, NUM_MZ_BINS))

            fsp = MzSpectrum(None, -1, -1, spec[idx][0], spec[idx][1]).to_resolution(2)

            target_spec = csr_array((fsp.vectorize(3).values(),
                                     ([0]*len(fsp.mz()), fsp.vectorize(3).indices())),
                                     shape=(1, NUM_MZ_BINS))

            simil = cosine_similarity(data_specs, target_spec)

            if np.max(simil) < SIMIL_THOLD:
                continue

            best_idx = np.argmax(simil)
            best_sim = np.max(simil)
            best_spec = hashed_slice.get_spectra()[best_idx]

            mask = np.logical_and.reduce([points_filt.mz >= lower_mz,
                                          points_filt.mz <= upper_mz,
                                          points_filt.scan >= best_spec.scan_id() - SCAN_RANGE,
                                          points_filt.scan <= best_spec.scan_id() + SCAN_RANGE])

            #print("MASK", np.sum(mask))
            labels[mask, int(row.Charge)] = best_sim


            break

        hardmax = labels.argmax(axis=-1)

        labels.fill(0.0)
        labels[np.arange(len(labels)),hardmax] = 1

        labels_df = pd.DataFrame(labels, columns=[str(i) for i in range(NUM_SCORES)])
        points_labs = pd.concat([points_filt, labels_df], axis=1)

        # training loop
        for i in np.random.permutation(range(len(MZs)-1)):

            points_window = points_labs.query(f"{MZs[i]} <= mz and mz < {MZs[i+1]}")
            #print(len(points_window))
            if len(points_window) == 0:
                continue

            points_east = points_labs.query(f"{MZs[i] - MZ_epsilon} <= mz and mz < {MZs[i]}")

            points_west = points_labs.query(f"{MZs[i+1]} <= mz and mz < {MZs[i+1] + MZ_epsilon}")

            points_att = pd.concat([points_east, points_west])[["mz", "scan", "intensity"]]

            if len(points_att) == 0:
                continue

            # scaling
            points_window["mz"] = points_window["mz"].subtract(MZs[i])
            points_att["mz"] = points_att["mz"].subtract(MZs[i])

            # cutoff or pad
            if len(points_window) >= MAX_N:
                nlargest = points_window.nlargest(MAX_N, "intensity")
                values_window = nlargest.values
            else:
                pad_len = MAX_N - len(points_window)
                values_window = np.pad(points_window, ((0,pad_len), (0,0)), "edge")

            # cutoff or pad for sorrounding points
            if len(points_att) >= MAX_N * ATT_SIZE:
                nlargest = points_att.nlargest(MAX_N * ATT_SIZE, "intensity")
                values_att = nlargest.values
            else:
                pad_len = MAX_N * ATT_SIZE - len(points_att)
                values_att = np.pad(points_att, ((0,pad_len), (0,0)), "constant")

            # separate features and labels
            features = values_window[:,:POINT_DIM]
            labels_onehot = values_window[:,POINT_DIM:]
            
            # class weights
            #shares = np.sum(labels_onehot, axis=0)/MAX_N
            #class_weight = dict(zip(range(len(shares)), 1-shares))
            labels = np.apply_along_axis(lambda x: np.where(x==1), 1, labels_onehot).flatten()
            sample_weight = np.vectorize(CLASS_WEIGHT.__getitem__)(labels).reshape(BATCH_SIZE, MAX_N)

            x_train = np.concatenate([features, values_att]).reshape(BATCH_SIZE, MAX_N*(ATT_SIZE+1), POINT_DIM)

            # intensity scaling
            x_train[:,2] *= 255/INT_MAX
            y_train = labels_onehot.reshape(BATCH_SIZE, MAX_N, NUM_SCORES)
            
            #class_dist = np.sum(y_train, axis=-2)[0]
            #if class_dist[0] < 1024:
            #    print(class_dist)

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
                
                yt = labels_onehot.argmax(axis=-1)
                
                if len(points_window) < MAX_N:
                    yt = yt[:len(points_window)]
                    
                y_true = np.append(y_true, yt)
                y_pred = np.append(y_pred, yp.argmax(axis=-1))
                
                
        if not train and len(y_true) >= 2**17:
            return y_true, y_pred
                
    
    if not train:
        return y_true, y_pred
                               
        
def run_train(paths, isodet):
    
    for p in paths:
        run(p, isodet, True)
        print("=== FINISHED FILE ===")
        
def run_test(paths, isodet):
    y_true = np.array([])
    y_pred = np.array([])
    
    for p in paths:
        yt, yp = run(p, isodet, False)
        y_true = np.append(y_true, yt)
        y_pred = np.append(y_pred, yp)
        print("=== FINISHED FILE ===")
    
    return y_true, y_pred