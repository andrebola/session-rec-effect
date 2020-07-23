import os
import time
import datetime
import pickle
import random
import json
import numpy as np
import evaluation.simulation
from matplotlib import pyplot as plt
from collections import Counter
import pandas as pd


def predict_test(pr, test_data, session_key, item_key, time_key, time, metrics, items_to_predict, offset_sessions, length_session):
    # Make recommendations for the test set to compute accuracy metrics
    for m in metrics:
        m.reset();

    current_session_idx = 0
    pos = offset_sessions[current_session_idx]
    position = 0
    finished = False

    while not finished:

        crs = time.clock();
        trs = time.time();

        current_item = test_data[item_key][pos]
        current_session = test_data[session_key][pos]
        ts = test_data[time_key][pos]
        rest = test_data[item_key][pos+1:offset_sessions[current_session_idx]+length_session[current_session_idx]].values

        for m in metrics:
            if hasattr(m, 'start_predict'):
                m.start_predict( pr )

        preds = pr.predict_next(current_session, current_item, items_to_predict, timestamp=ts)

        for m in metrics:
            if hasattr(m, 'start_predict'):
                m.stop_predict( pr )

        preds[np.isnan(preds)] = 0
        preds.sort_values( ascending=False, inplace=True )

        for m in metrics:
            if hasattr(m, 'add_multiple'):
                m.add_multiple( preds, rest, for_item=current_item, session=current_session, position=position)

        pos += 1
        position += 1

        if pos + 1 == offset_sessions[current_session_idx]+length_session[current_session_idx]:
            current_session_idx += 1

            if current_session_idx == test_data[session_key].nunique():
                finished = True

            pos = offset_sessions[current_session_idx]
            position = 0

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


def evaluate(iteration_tracks, seeds, items_dict, tracks_pop):
    # Compute the evaluation for the given recommendations
    all_songs = {}
    rel_popularity = []
    popularity = []
    for user in iteration_tracks.keys():
        if len(iteration_tracks[user]):
            curr_pop = 0
            curr_pop_rel = 0
            seed_popularity = 0
            if seeds[user][0][0] in tracks_pop:
                seed_popularity = tracks_pop[seeds[user][0][0]]
            for track in iteration_tracks[user]:
                if track in tracks_pop:
                    curr_pop += tracks_pop[track]
                curr_track_pop = 0
                if track in tracks_pop:
                    curr_track_pop = tracks_pop[track]
                curr_pop_rel += (curr_track_pop - seed_popularity)
                if track not in all_songs:
                    all_songs[track] = 0
                all_songs[track] += 1
            popularity.append(curr_pop/len(iteration_tracks[user]))
            rel_popularity.append(curr_pop_rel/len(iteration_tracks[user]))

    popularity = np.mean(popularity)
    rel_popularity = np.mean(rel_popularity)
    different_songs = len(all_songs)
    if different_songs > len(items_dict):
        np_counts = np.zeros(different_songs, np.dtype('float64'))
    else:
        np_counts = np.zeros(len(items_dict), np.dtype('float64'))
    np_counts[:different_songs] = np.array(list(all_songs.values())) 
    return gini(np_counts), different_songs, popularity, rel_popularity

def generate_cagh(history, n_recs, artists_simil, artists_items, tracks_pop, prevs=None, lambda1=10, rerank=False, rerank_type='popular'):
    # Here we generate recommendations using the cagh algorithm
    if prevs == None:
        prevs = {}
    artists_simil2 = {}
    artists_totals = {}
    for artist in artists_simil.keys():
        artists_totals[artist] = np.sum([v for a,v in artists_simil[artist].items()])
    for artist in artists_simil.keys():
        for artist2 in artists_simil[artist].keys():
            if artist not in artists_simil2:
                artists_simil2[artist] = {}
            artists_simil2[artist][artist2] = artists_simil[artist][artist2]/np.sqrt(artists_totals[artist]*artists_totals[artist2])

    ret = {}
    most_similar = {}
    for artist in artists_simil2.keys():
        most_similar[artist] = [artist2 for artist2, artist_score in Counter(artists_simil2[artist]).most_common(n_recs+30) if artist2 != artist]
    items_artists = {}
    for item in artists_items.keys():
        if artists_items[item] not in items_artists:
            items_artists[artists_items[item]] = []
        items_artists[artists_items[item]].append(item)
    most_popular = {}
    for artist in items_artists:
        curr_artist = {}
        for item in items_artists[artist]:
            if item in tracks_pop:
                curr_artist[item] = tracks_pop[item]
        most_popular[artist] = [track for track, track_score in Counter(curr_artist).most_common(n_recs+30)]
    for user in history.keys():
        prev_items = dict(Counter([song for song,time in history[user]]))
        curr_score = {}
        artists = set()
        prev_tracks = {t:v for t,v in history[user] }
        for item, timestamp in history[user]:
            artists.add(artists_items[item])
        more_added = True
        while (len(curr_score) <(n_recs+30)) and more_added:
            more_added = False
            artists_to_add = []
            for artist in artists:
                artists_similar = most_similar[artist]
                for artist_similar in artists_similar:
                    for track in most_popular[artist_similar]:
                        #if track not in prev_tracks:
                            if track not in curr_score:
                                curr_score[track] = 0
                                for artist2 in artists:
                                    if artist_similar in artists_simil2[artist2]:
                                        curr_score[track] += tracks_pop[track]*artists_simil2[artist2][artist_similar]
                                artists_to_add.append(artists_items[track])
            pre_len = len(artists)
            artists.update(artists_to_add)
            if pre_len < len(artists):
                more_added = True

        # If rerank is enable we penalize the recommendations according to the number times that was already recommended
        if not rerank:
            ret[user] = [track for track, track_score in Counter(curr_score).most_common(n_recs)]
        else:
            recs_dict = {str(track):i for i, (track, track_score) in enumerate(Counter(curr_score).most_common(n_recs+30))}
            for track in recs_dict.keys():
                if rerank_type == 'popular':
                    if track in prevs:
                        recs_dict[track] += lambda1*np.log1p(prevs[track])
                    else:
                        recs_dict[track] -= lambda1
                else:
                    if track in prev_items:
                        recs_dict[track] += lambda1*prev_items[track]
                    else:
                        recs_dict[track] -= lambda1
 
            ret[user] = [k for k,v in sorted(recs_dict.items(), key=lambda x: x[1])][:n_recs]
        if len(ret[user]) == 0:
            print (artists_simil2[artist], artist)
            print ('USER', user, len(artists), len(curr_score), len(most_similar[artist]), len(items_artists[artist]))
    return ret 


def generate_random_recs(items_dict, n_recs):
    recs = random.sample(items_dict.keys(), n_recs)
    return recs

def retrain_cagh(curr_session_tracks, artists_items, artists_simil):
    # We retrain the cagh algorithm
    artists_simil2 = artists_simil.copy()
    items_pop = {}
    for user in curr_session_tracks.keys():
        artists = set()
        for item,time in curr_session_tracks[user]:
            if item not in items_pop:
                items_pop[item] = 0
            artists.add(artists_items[item])
            items_pop[item] += 1
        for artist in artists:
            for artist2 in artists:
                if artist not in artists_simil2:
                    artists_simil2[artist] = {}
                if artist2 not in artists_simil2:
                    artists_simil2[artist2] = {}
                if artist2 not in artists_simil2[artist]:
                    artists_simil2[artist][artist2] = 0
                if artist not in artists_simil2[artist2]:
                    artists_simil2[artist2][artist] = 0
                artists_simil2[artist][artist2] += 1
                artists_simil2[artist2][artist] += 1

    return artists_simil2, items_pop


def retrain_algorithm(algorithm, curr_session_tracks, epoch, train_data, session_key='SessionId', item_key='ItemId', time_key='Time'):
    # In this method we add the new sessions and retrain the algorithm
    sessions = []
    items = []
    times = []

    for user in curr_session_tracks.keys():
        if len(curr_session_tracks[user]) > 0:
            for item,time in curr_session_tracks[user]:
                sessions.append(user)
                times.append(time)
                items.append(int(item))

    df = pd.DataFrame.from_dict({session_key: sessions, time_key: times, item_key: items})
    train = pd.concat([train_data,df])
    print ("ADDED", len(train_data), len(df), len(train))
    algorithm.n_epochs = 1
    algorithm.epochs = 1
    algorithm.fit(train)
    return train

def get_tracks(reco_tracks, number_listened, tracks_pop, alpha):
    # Select 10 tracks from the recommendations
    selected = []
    if alpha == 1:
        perc_listened =  number_listened/len(reco_tracks)
        for i, track in enumerate(reco_tracks):
            if random.uniform(0,1) < perc_listened:
                selected.append(track)
    else:
        selected = reco_tracks[:number_listened]
    
    return selected


def generate_recs(sessions, n_iter, last_epoch, items_to_predict, pr, item_ids, prevs=None, lambda1=10, rerank=False, rerank_type='popular'):
    # In this method we generate the recommendations for a given session
    moved = []
    if prevs == None:
        prevs = {}
    ret = {}
    for s in sessions:
        u_moved = []
        prev_items = dict(Counter([song for song,time in sessions[s]]))
        previous_tracks = sessions[s][-1]
        ts = previous_tracks[1] + 3
        preds = pr.predict_next(s, int(previous_tracks[0]), items_to_predict, timestamp=ts)
        preds[np.isnan(preds)] = 0
        preds.sort_values( ascending=False, inplace=True )
        curr_pred = preds[:n_iter+30]
        if not rerank:
            ret[s] = [str(t) for t in curr_pred.keys().tolist()][:n_iter]
        else:
            # If rerank is enabled the use the previous occurence of the track to penalize
            recs_dict = {str(t):i for i,t in enumerate(curr_pred.keys().tolist())}
            for track in recs_dict.keys():
                if rerank_type == 'popular':
                    if track in prevs:
                        recs_dict[track] += lambda1*np.log1p(prevs[track])
                        u_moved.append(lambda1*np.log1p(prevs[track]))
                    else:
                        recs_dict[track] -= lambda1
                else:
                    if track in prev_items:
                        recs_dict[track] += lambda1*prev_items[track]
                        u_moved.append(lambda1*prev_items[track])
                    else:
                        recs_dict[track] -= lambda1
            ret[s] = [k for k,v in sorted(recs_dict.items(), key=lambda x: x[1])][:n_iter]
            if len(u_moved):
                moved.append(np.mean(u_moved))
            else:
                moved.append(0)
    print ("MODIFIED: ", np.mean(moved))
    print ("MODIFIED USERS", len([i for i in moved if i!=0]))
    return ret

def evaluate_sessions(pr, metrics, test_data, train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time', conf=None, key='def'): 
    '''
    Evaluates the baselines wrt. recommendation accuracy measured by recall@N and MRR@N. Has no batch evaluation capabilities. Breaks up ties.

    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    metrics : list
        A list of metric classes providing the proper methods
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out :  list of tuples
        (metric_name, value)
    
    '''
    model_folder = conf['data']['model']
    data_folder = conf['data']['folder']
    data_prefix = conf['data']['prefix']
    data_filename = data_folder + data_prefix + '_train_full.txt'
    conf_opt = conf['simulation']['opts']
    choose_seed = conf_opt['choose_seed']
    n_recs = int(conf_opt['n_recs'])
    alpha = float(conf_opt['alpha'])
    use_cagh = False
    if 'use_cagh' in conf_opt:
        use_cagh = conf_opt['use_cagh'] == 'True'
    if use_cagh:
        key = 'cagh'
    rerank = False
    if 'rerank' in conf_opt:
        rerank= conf_opt['rerank'] == 'True'
    rerank_type = 'popular'
    if 'rerank_type' in conf_opt:
        rerank_type = conf_opt['rerank_type']
    print ("RERANK", rerank_type)
    iter_days = int(conf_opt['iter_days'])
    iter_secs = iter_days*60*60*24
    actions = len(test_data)
    sessions = len(test_data[session_key].unique())
    count = 0
    print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')
    
    sc = time.clock();
    st = time.time();
    
    time_sum = 0
    time_sum_clock = 0
    time_count = 0

    for m in metrics:
        m.reset();
    # The following method loads the dataset and returns the MF trained model
    users, item_ids, item_vecs_reg, user_ids, user_vecs_reg, tracks_pop, users_dict, items_dict, global_last,global_first, orig_artists_simil, artists_items = evaluation.simulation.load_previous_data(train_data, item_key, session_key, time_key, model_folder, data_filename)
    print (len(item_ids))
    print (len(user_ids))

    pickle.dump(artists_items, open(os.path.join(model_folder, 'artists_items.pkl'), 'wb')) 
    pickle.dump(orig_artists_simil, open(os.path.join(model_folder, 'artists_simil.pkl'), 'wb')) 
    pickle.dump(users, open(os.path.join(model_folder, 'users.pkl'), 'wb'))
    pickle.dump(item_ids, open(os.path.join(model_folder, 'item_ids.pkl'), 'wb'))
    pickle.dump(user_ids, open(os.path.join(model_folder, 'user_ids.pkl'), 'wb'))
    pickle.dump(dict(tracks_pop), open(os.path.join(model_folder, 'tracks_pop.pkl'), 'wb'))
    pickle.dump(users_dict, open(os.path.join(model_folder, 'users_dict.pkl'), 'wb'))
    pickle.dump(items_dict, open(os.path.join(model_folder, 'items_dict.pkl'), 'wb'))
    pickle.dump(global_last, open(os.path.join(model_folder, 'global_last.pkl'), 'wb'))
    pickle.dump(global_first, open(os.path.join(model_folder, 'global_first.pkl'), 'wb'))
    np.save(open(os.path.join(model_folder, 'user_vecs_reg.npz'), 'wb'), user_vecs_reg)
    np.save(open(os.path.join(model_folder, 'item_vecs_reg.npz'), 'wb'), item_vecs_reg)
    """
    artists_items = pickle.load(open(os.path.join(model_folder, 'artists_items.pkl'), 'rb')) 
    orig_artists_simil = pickle.load(open(os.path.join(model_folder, 'artists_simil.pkl'), 'rb')) 
    users = pickle.load(open(os.path.join(model_folder, 'users.pkl'), 'rb'))
    item_ids = pickle.load(open(os.path.join(model_folder, 'item_ids.pkl'), 'rb'))
    user_ids = pickle.load(open(os.path.join(model_folder, 'user_ids.pkl'), 'rb'))
    tracks_pop = pickle.load(open(os.path.join(model_folder, 'tracks_pop.pkl'), 'rb'))
    users_dict = pickle.load(open(os.path.join(model_folder, 'users_dict.pkl'), 'rb'))
    items_dict = pickle.load(open(os.path.join(model_folder, 'items_dict.pkl'), 'rb'))
    global_last = pickle.load(open(os.path.join(model_folder, 'global_last.pkl'), 'rb'))
    global_first = pickle.load(open(os.path.join(model_folder, 'global_first.pkl'), 'rb'))
    user_vecs_reg = np.load(open(os.path.join(model_folder, 'user_vecs_reg.npz'), 'rb'))
    item_vecs_reg = np.load(open(os.path.join(model_folder, 'item_vecs_reg.npz'), 'rb'))
    """

    listened_tracks = {u:t['tracks'] for u,t in users.items()}
    test_data.sort_values([session_key, time_key], inplace=True)
    test_data = test_data.reset_index(drop=True)

    items_to_predict = train_data[item_key].unique()

    offset_sessions = np.zeros(test_data[session_key].nunique()+1, dtype=np.int32)
    length_session = np.zeros(test_data[session_key].nunique(), dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    length_session[0:] = test_data.groupby(session_key).size()

    current_session_idx = 0
    pos = offset_sessions[current_session_idx]
    position = 0
    finished = False
    session_tracks = {}

    n_secs = global_last - global_first
    different_songs = []
    rnd_different_songs = []
    algorithm_results = []
    users_interacted = []
    last_epoch=0
    crs = time.clock();
    trs = time.time();

    counter = 0
    curr_sec = global_first
    n_users = []
    iter_recs = {}
    # Generate seeds for each session, options are: MF, pop, rnd
    for user in users.keys():
        listened = users[user]['times']
        if len(listened) > 0:
            user_pos = users_dict[user]

            curr_time = curr_sec
            if choose_seed == 'MF':
                if user not in session_tracks:
                    session_tracks[user] = []
                    sims = user_vecs_reg[user_pos].dot(item_vecs_reg.T)
                    top = np.argsort(-sims)[:1]
                    for i in top:
                        session_tracks[user].append((item_ids[i], curr_time))
                        curr_time += 3
            elif choose_seed == 'rnd':
                recs = random.sample(items_dict.keys(), 1)
                session_tracks[user] = []
                for i in recs:
                    session_tracks[user].append((i, curr_time))
            elif choose_seed == 'pop':
                session_tracks[user] = []
                if 'most_listened' in users[user]:
                    session_tracks[user].append((users[user]['most_listened'], curr_time))
                else:
                    counter += 1

            n_users.append(user)
    print ('MISSING', counter)
    print ('USERS', n_users)

    # START SIMULATION
    original_seeds = {}
    tracks_pop_new = tracks_pop
    artists_simil = orig_artists_simil
    res_prev_iter = {}
    # For each seed we generate 30 tracks and then select 10, this is repeated 30 times
    for i in range(10):
        for j in range(3):
            curr_sessions = {s:session_tracks[s] for s in n_users}
            if i==0 and j==0:
                original_seeds = curr_sessions.copy()
            print ('START RECOMMENDING')
            if use_cagh:
                iter_recs = generate_cagh(session_tracks, n_recs, artists_simil, artists_items, tracks_pop_new, prevs=res_prev_iter, rerank=rerank, rerank_type=rerank_type)
            else:
                iter_recs = generate_recs(curr_sessions, n_recs, last_epoch, items_to_predict, pr, item_ids, prevs=res_prev_iter, rerank=rerank, rerank_type=rerank_type)

            # We only consider the recommendations in the last round for the rerank, this is not used if strategy is 'user'
            if j == 0:
                res_prev_iter = {}
            for user in iter_recs.keys():
                for t in iter_recs[user]:
                    if t not in res_prev_iter:
                        res_prev_iter[t] = 0
                    res_prev_iter[t] += 1
            print ('FINISHED RECOMMENDING')
            if i==0 and j==0:
                gini_val, different_songs, popularity, rel_pop= evaluate(iter_recs, session_tracks, items_dict, tracks_pop)
                predict_test(pr, test_data, session_key, item_key, time_key, time, metrics, items_to_predict, offset_sessions, length_session)
                res = []
                for m in metrics:
                    if type(m).__name__ == 'Time_usage_testing':
                        res.append(m.result_second(time_sum_clock/time_count))
                        res.append(m.result_cpu(time_sum_clock / time_count))
                    else:
                        res.append( m.result() )
                algorithm_results.append({'gini': gini_val, 'avg_pop': popularity, 'different_items': different_songs, 'rel_pop': rel_pop, 'retrain': i, 'results': res})
            number_listened = 10
            tracks_length = []
            for user in n_users:
                simul_tracks = get_tracks(iter_recs[user], number_listened, tracks_pop, alpha)
                tracks_length.append(len(simul_tracks))
                session_tracks[user] += [(t, (curr_sec+3*p)) for p,t in enumerate(simul_tracks)]
            print ("LENGTH", np.mean(tracks_length))

        # Now we retrain the algorithms with the added interactions and generate a recommendation with the original seeds
        if use_cagh:
            artists_simil, tracks_pop_new =retrain_cagh(session_tracks, artists_items, orig_artists_simil)
            iter_recs = generate_cagh(original_seeds, n_recs, artists_simil, artists_items, tracks_pop_new, prevs=res_prev_iter, rerank=rerank, rerank_type=rerank_type)
        else:
            ret_data = retrain_algorithm(pr, session_tracks, last_epoch, train_data, session_key, item_key, time_key)
            items_to_predict = ret_data[item_key].unique()
            iter_recs = generate_recs(original_seeds, n_recs, last_epoch, items_to_predict, pr, item_ids, prevs=res_prev_iter, rerank=rerank, rerank_type=rerank_type)
        # We evaluate the recommendations generated with the original seeds
        gini_val, different_songs, popularity, rel_pop= evaluate(iter_recs, original_seeds, items_dict, tracks_pop)
        predict_test(pr, test_data, session_key, item_key, time_key, time, metrics, items_to_predict, offset_sessions, length_session)
        res = []
        for m in metrics:
            if type(m).__name__ == 'Time_usage_testing':
                res.append(m.result_second(time_sum_clock/time_count))
                res.append(m.result_cpu(time_sum_clock / time_count))
            else:
                res.append( m.result() )
        algorithm_results.append({'gini': gini_val, 'avg_pop': popularity, 'different_items': different_songs, 'rel_pop': rel_pop, 'retrain': i+1, 'results': res})
        str_rerank = ''
        if rerank:
            if rerank_type=='popular':
                str_rerank = '_rerank_nonusers'
            else:
                str_rerank = '_rerank'
        # Save the results after each evaluation
        json.dump( algorithm_results, open( "models_2/10_retrain_{}_{}seed_eval_{}_lastschoose{}_fix.json".format(key, choose_seed, data_prefix, str_rerank), "w" ) ) 

    time_sum_clock += time.clock()-crs
    time_sum += time.time()-trs
    time_count += 1
    print( 'END iteration in ', (time.clock()-sc), 'c / ', (time.time()-st), 's' )
    print( '    avg rt ', (time_sum/time_count), 's / ', (time_sum_clock/time_count), 'c' )

    res = []
    for m in metrics:
        if type(m).__name__ == 'Time_usage_testing':
            res.append(m.result_second(time_sum_clock/time_count))
            res.append(m.result_cpu(time_sum_clock / time_count))
        else:
            res.append( m.result() )

    return res

