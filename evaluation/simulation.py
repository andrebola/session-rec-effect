import os
import struct
import time
import tqdm
import numpy as np
import pickle
from collections import defaultdict
from scipy import sparse
from lightfm import LightFM

def load_feats(feat_fname, meta_only=False, nrz=False):
    # Aux method to save the trained model
    with open(feat_fname, 'rb') as fin:
        keys = fin.readline().strip().split()
        R, C = struct.unpack('qq', fin.read(16))
        if meta_only:
            return keys, (R, C)
        feat = np.fromstring(fin.read(), count=R * C, dtype=np.float32)
        feat = feat.reshape((R, C))
        if nrz:
            feat = feat / np.sqrt((feat ** 2).sum(-1) + 1e-8)[..., np.newaxis]
    return keys, feat

def save(keys, feats, out_fname):
    # Aux method to load the trained model
    feats = np.array(feats, dtype=np.float32)
    with open(out_fname + '.tmp', 'w') as fout:
        fout.write(' '.join([str(k) for k in keys]))
        fout.write('\n')
        R, C = feats.shape
        fout.write(struct.pack('qq', *(R, C)))
        fout.write(feats.tostring())
    os.rename(out_fname + '.tmp', out_fname)

def load(model_folder):
    user_ids, user_vecs_reg = load_feats(os.path.join(model_folder,'out_user_features.feats'))
    item_ids, item_vecs_reg = load_feats(os.path.join(model_folder, 'out_item_features.feats'))
    return item_ids, item_vecs_reg, user_ids, user_vecs_reg

def train(impl_train_data, config, user_ids, item_ids, model_folder, save_res=True):
    # In this method we train the MF algorithm
    model = LightFM(loss='warp', no_components=config['dims'], learning_rate=config['lr'])
    model = model.fit(impl_train_data, epochs=50, num_threads=8)

    user_biases, user_embeddings = model.get_user_representations()
    item_biases, item_embeddings = model.get_item_representations()
    item_vecs_reg = np.concatenate((item_embeddings, np.reshape(item_biases, (1, -1)).T), axis=1)
    user_vecs_reg = np.concatenate((user_embeddings, np.ones((1, user_biases.shape[0])).T), axis=1)
    print("USER FEAT:", user_vecs_reg.shape)
    print("ITEM FEAT:", item_vecs_reg.shape)
    if save_res==True:
        save(item_ids, item_vecs_reg, os.path.join(model_folder, 'out_item_features.feats'))
        save(user_ids, user_vecs_reg, os.path.join(model_folder, 'out_user_features.feats'))
    return item_ids, item_vecs_reg, user_ids, user_vecs_reg

def load_previous_data(dataset, item_key, session_key, time_key, model_folder, data_filename):
    # Here we load all the train set in a dictionary that is used in the simulation.
    # We also count the co-ocurrence of the artists and we train a MF algorithm to generate seeds.
    session_ids = []
    item_ids = []
    items_dict = {}
    sessions_dict = {}
    data_train = []
    row_train = []
    col_train = []
    count = 0
    actions = len(dataset)

    offset_sessions = np.zeros(dataset[session_key].nunique()+1, dtype=np.int32)
    length_session = np.zeros(dataset[session_key].nunique(), dtype=np.int32)
    offset_sessions[1:] = dataset.groupby(session_key).size().cumsum()
    length_session[0:] = dataset.groupby(session_key).size()

    current_session_idx = 0
    pos = offset_sessions[current_session_idx]
    position = 0
    finished = False

    #sessions will contain all the information for a session in the dataset
    sessions = {}
    global_first = 1999999999999
    global_last = 0
    tracks_pop = defaultdict(lambda:1)

    artists_simil = {}
    previous_session = None
    artists_items = {}
    items = []
    session_pos = {}
    count = 0
    for line in tqdm.tqdm(open(data_filename)):
        hists = line.split('\t')
        artist_id = hists[4][:-1]
        item_id = hists[2]
        artists_items[item_id] = artist_id
        session_pos[hists[1]] = count
        count += 1


    count = 1 # We skip the first line
    items = {}
    for line in tqdm.tqdm(open(data_filename)):
        hist = line.split('\t')
        tracks_counter = {}
        session = hist[1]
        if session == 'SessionId':
            continue
        item_id = hist[2]
        c_time = hist[3]
        if count == session_pos[session]:
            # For each session we count: first and last date of interaction, total interactions, we also list all the tracks and artist
            sessions[session] = {'first': 1999999999999, 'last': 0, 'total': 0, 'all': set(), 'times': [], 'tracks':[], 'repeats':0}
            artists = []
            for track_id, time_v  in items[session]:
                if session not in sessions_dict:
                    sessions_dict[session] = len(session_ids)
                    session_ids.append(session)
                if track_id not in items_dict:
                    items_dict[track_id] = len(item_ids)
                    item_ids.append(track_id)
                if track_id not in tracks_counter:
                    tracks_counter[track_id] = 0
                tracks_counter[track_id] += 1

                if track_id not in tracks_pop:
                    tracks_pop[track_id] = 0
                tracks_pop[track_id] += 1
                artists.append(artists_items[track_id])

                if int(time_v) > sessions[session]['last']:
                    sessions[session]['last'] = int(time_v)
                if int(time_v) < sessions[session]['first']:
                    sessions[session]['first'] = int(time_v)
                sessions[session]['total'] = sessions[session]['total']+1
                sessions[session]['all'].add(track_id)
                if sessions[session]['last'] > global_last:
                    global_last = sessions[session]['last']
                if sessions[session]['first'] < global_first:
                    global_first = sessions[session]['first']
                sessions[session]['times'].append(int(time_v))
                sessions[session]['tracks'].append(track_id)
            sessions[session]['repeats'] = len(sessions[session]['all'])/sessions[session]['total']
            most_listened = None
            most_listened_count = 0
            for track in tracks_counter.keys():
                col_train.append(items_dict[track])
                row_train.append(sessions_dict[session])
                data_train.append(tracks_counter[track])
                if most_listened_count < tracks_counter[track]:
                    most_listened_count = tracks_counter[track]
                    most_listened = track

            # We count all the artists co-ocurrence in artists_simil
            sessions[session]['most_listened'] = most_listened
            for artist in artists:
                for artist2 in artists:
                    if artist not in artists_simil:
                        artists_simil[artist] = {}
                    if artist2 not in artists_simil:
                        artists_simil[artist2] = {}
                    if artist2 not in artists_simil[artist]:
                        artists_simil[artist][artist2] = 0
                    if artist not in artists_simil[artist2]:
                        artists_simil[artist2][artist] = 0
                    artists_simil[artist][artist2] += 1
                    artists_simil[artist2][artist] += 1

            del items[session]
        else:
            if c_time.isdigit():
                if session not in items:
                    items[session] = []
                items[session].append((item_id,c_time))
            else:
                print (c_time)
        count += 1

    # Now we train a MF algorithm with the training data
    train_params = {'dims':200, 'lr':0.05}
    train_data = sparse.coo_matrix((data_train, (row_train, col_train)), dtype=np.float32)
    item_ids, item_vecs_reg, session_ids, user_vecs_reg = train(train_data, train_params, session_ids, item_ids, model_folder, False)
    #item_ids, item_vecs_reg, session_ids, user_vecs_reg = load(model_folder)
    return sessions, item_ids, item_vecs_reg, session_ids, user_vecs_reg, tracks_pop, sessions_dict, items_dict, global_last, global_first, artists_simil, artists_items
