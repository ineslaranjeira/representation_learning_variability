"""
IMPORTS
"""

prefix = '/home/ines/repositories/'
prefix = '/Users/ineslaranjeira/Documents/Repositories/'

import autograd.numpy as np
import os 
import pandas as pd

# --Machine learning and statistics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.cluster import KMeans
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn import mixture
from scipy.stats import entropy
from kneed import KneeLocator

# Get my functions
functions_path =  prefix + 'representation_learning_variability/Models/Sub-trial//3_postprocess_results/'
os.chdir(functions_path)
from postprocessing_functions import trial_relative_frequency

""" Cluster assessment """

def GMM_neg_log_likelihood(embedding, components):
    
    LL = np.zeros(len(components)) * np.nan
    
    for i, k in enumerate(components):
        # g = mixture.GaussianMixture(n_components=k)
        # generate random sample, two components
        np.random.seed(0)

        # concatenate the two datasets into the final training set
        cutoff = int(np.shape(embedding)[0]*0.8)
        train_indices = np.random.choice(embedding.shape[0], cutoff, replace=False)
        X_train = np.vstack([embedding[train_indices, 0], embedding[train_indices, 1]]).T

        # fit a Gaussian Mixture Model with two components
        clf = mixture.GaussianMixture(n_components=k, covariance_type='full')
        clf.fit(X_train)

        all_indices = np.arange(0, embedding.shape[0], 1)
        test_indices = [idx for idx in all_indices if idx not in train_indices]
        X_test = np.vstack([embedding[test_indices, 0], embedding[test_indices, 1]])
        LL[i] = -clf.score(X_test.T)
        
    return LL


def revert_to_original(use_data):
    # Revert to original states
    use_data['original_states'] = use_data['identifiable_states'].copy()

    state = use_data.loc[use_data['original_states'].str[0]=='L', 'original_states']
    use_data.loc[use_data['original_states'].str[0]=='L', 'original_states'] = '1' + state.str[1:]

    state = use_data.loc[use_data['original_states'].str[0]=='l', 'original_states']
    use_data.loc[use_data['original_states'].str[0]=='l', 'original_states'] = '1' + state.str[1:]

    state = use_data.loc[use_data['original_states'].str[0]=='R', 'original_states']
    use_data.loc[use_data['original_states'].str[0]=='R', 'original_states'] = '1' + state.str[1:]

    state = use_data.loc[use_data['original_states'].str[0]=='r', 'original_states']
    use_data.loc[use_data['original_states'].str[0]=='r', 'original_states'] = '1' + state.str[1:]


    state = use_data.loc[use_data['original_states'].str[0]=='n', 'original_states']
    use_data.loc[use_data['original_states'].str[0]=='n', 'original_states'] = '1' + state.str[1:]
    
    return use_data


def get_ballistic(use_data):
    use_data['bal_state'] = use_data['identifiable_states'].copy()
    use_data.loc[use_data['identifiable_states'].str[0]=='L', 'bal_state'] = 'balistic'
    use_data.loc[use_data['identifiable_states'].str[0]=='R', 'bal_state'] = 'balistic'
    use_data.loc[use_data['identifiable_states'].str[0]=='l', 'bal_state'] = 'non_balistic'
    use_data.loc[use_data['identifiable_states'].str[0]=='r', 'bal_state'] = 'non_balistic'
    use_data.loc[use_data['identifiable_states'].str[0]=='n', 'bal_state'] = 'non_balistic'
    use_data.loc[use_data['identifiable_states'].str[0]=='0', 'bal_state'] = np.nan

    return use_data


def get_no_resp(use_data):
    use_data['resp'] = use_data['identifiable_states'].copy()
    use_data.loc[use_data['identifiable_states'].str[0]=='L', 'resp'] = 'response'
    use_data.loc[use_data['identifiable_states'].str[0]=='R', 'resp'] = 'response'
    use_data.loc[use_data['identifiable_states'].str[0]=='l', 'resp'] = 'response'
    use_data.loc[use_data['identifiable_states'].str[0]=='r', 'resp'] = 'response'
    use_data.loc[use_data['identifiable_states'].str[0]=='n', 'resp'] = 'non_response'
    use_data.loc[use_data['identifiable_states'].str[0]=='0', 'resp'] = np.nan

    return use_data


def find_best_k(embedding, Ks):
    # Assuming `X` is your data
    inertia_values = []
    
    for k in Ks:
        kmeans = KMeans(n_clusters=k)
        cutoff = int(np.shape(embedding)[0]*0.8)
        X_train = np.vstack([embedding[:cutoff, 0], embedding[:cutoff, 1]]).T
        kmeans.fit(X_train)
        inertia_values.append(kmeans.inertia_)
        
    kneedle = KneeLocator(Ks, inertia_values, curve="convex", direction="decreasing")
    optimal_k = kneedle.knee
    return optimal_k

def Ks_iter (ori_X, repeats, Ks):
    optimal_ks = np.zeros((repeats))
    
    # Embedd
    for r in range(repeats):
        reducer = umap.UMAP(n_components=2)
        embedding = reducer.fit_transform(ori_X)
        embedding.shape
        
        # Find ideal k
        optimal_k = find_best_k(embedding, Ks)
        optimal_ks[r] = optimal_k
        
    return optimal_ks


def cluster_consensus(ori_X, optimal_k, shuffle=False, repeats=100):
       
    # Initialize consensus matrix

    n_samples, _ = ori_X.shape
    consensus_matrix = np.zeros((n_samples, n_samples))
    
    # Embedd
    for r in range(repeats):
        reducer = umap.UMAP(n_components=2)

        if shuffle:
            shuffled_arr = np.apply_along_axis(np.random.permutation, 1, ori_X)
            part_embedding = reducer.fit_transform(shuffled_arr)
        else:
            part_embedding = reducer.fit_transform(ori_X)
        
        # Cluster
        kmeans = KMeans(n_clusters=optimal_k)
        kmeans.fit(part_embedding)
        labels = kmeans.predict(part_embedding)
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                if labels[i] == labels[j]:
                    consensus_matrix[i, j] += 1
                    if i != j:
                        consensus_matrix[j, i] += 1  # symmetry
                            
    # Normalize to [0, 1]
    consensus_matrix /= repeats
    
    return consensus_matrix


def repeated_splits(trial_clusters, vars, n_parts, rng, optimal_k, reps=10):

    sessions = trial_clusters.session.unique()
    session_num = len(sessions)
    all_cluster_repeats = pd.DataFrame(columns=['mouse_name', 'session', 'sample', 'y_kmeans', 'repeat'])
    # all_cluster_repeats = np.zeros((session_num * n_parts, reps)) * np.nan
    
    for r in range(reps):
        all_cluster = pd.DataFrame(columns=['mouse_name', 'session', 'sample', 'y_kmeans', 'repeat'], index=range(session_num * n_parts))
        use_df = trial_clusters.copy()
        use_df['session_part'] = use_df['response'] * np.nan

        for s, session in enumerate(sessions):
            session_df = use_df.loc[use_df['session']==session]
            n_trials = len(session_df)
            # Create shuffled labels for parts
            parts = np.tile(np.arange(n_parts), int(np.ceil(n_trials / n_parts)))[:n_trials]
            rng.shuffle(parts)  # Shuffle to randomize assignment
            use_df.loc[use_df['session']==session, 'session_part'] = parts

        # Assign to new column
        use_df['session_part'] = use_df['session_part'].astype(str)
        use_df = use_df.rename(columns={"sample": "old_sample"})
        use_df['sample'] = use_df[['session', 'session_part']].agg(' '.join, axis=1)
        
        # Prepare design matrix
        count, freq_df = trial_relative_frequency(use_df, vars)
        # keys = freq_df.reset_index()['sample']
        var_names = freq_df.keys()
        ori_X = np.array(freq_df[var_names])

        # Cluster and save
        consensus_matrix = cluster_consensus(ori_X, optimal_k, shuffle=False, repeats=10)
        # Perform final clustering on consensus matrix
        final_clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage="average")
        mouse_y_kmeans = final_clustering.fit_predict(1 - consensus_matrix)  # Convert to similarity
        
        # Save
        info = use_df[['sample', 'mouse_name', 'session']].drop_duplicates().reset_index()
        all_cluster['mouse_name'] = info['mouse_name']
        all_cluster['session'] = info['session']
        all_cluster['sample'] = info['sample']
        all_cluster['y_kmeans'] = mouse_y_kmeans
        all_cluster['repeat'] = r
        all_cluster_repeats = pd.concat([all_cluster_repeats, all_cluster], ignore_index=True)

    
    return all_cluster_repeats


def agreement_ratio(all_cluster_repeats):
    cluster_entropy = pd.DataFrame(columns=['mouse_name', 'True', 'Shuffled', 'session_len'], index=range(len(all_cluster_repeats['mouse_name'].unique())))
    for m, mouse in enumerate(all_cluster_repeats['mouse_name'].unique()):
        cluster_entropy['mouse_name'][m] = mouse
        mouse_session_parts = all_cluster_repeats.loc[all_cluster_repeats['mouse_name']==mouse][['mouse_name', 'sample', 'y_kmeans', 'repeat']]
        mouse_repeats = mouse_session_parts['repeat'].unique()

        agree = []
        shuffle_agree = []
        for r, rep in enumerate(mouse_repeats):

            repeat_data = mouse_session_parts.loc[mouse_session_parts['repeat']==r, 'y_kmeans']
            # Count cluster label occurrences for sample i
            same = 1 if np.array(repeat_data)[0] == np.array(repeat_data)[1] else 0
            agree.append(same)

            # Test for random cluster assignment
            shuffle = np.random.randint(np.min(all_cluster_repeats['y_kmeans']), np.max(all_cluster_repeats['y_kmeans']), len(repeat_data))
            same_shuffle = 1 if shuffle[0]==shuffle[1] else 0
            shuffle_agree.append(same_shuffle)


        cluster_entropy['True'][m] = np.mean(agree)
        cluster_entropy['Shuffled'][m] = np.mean(shuffle_agree)
        cluster_entropy['session_len'][m] = np.floor(len(mouse_session_parts['sample'].unique())/2)

    melted_df = pd.melt(cluster_entropy, id_vars=['mouse_name', 'session_len'], value_vars=['True', 'Shuffled'])
    return melted_df

def calculate_entropy(all_cluster_repeats):
    
    cluster_entropy = pd.DataFrame(columns=['mouse_name', 'True', 'Shuffled', 'session_len'], index=range(len(all_cluster_repeats['mouse_name'].unique())))
    for m, mouse in enumerate(all_cluster_repeats['mouse_name'].unique()):
        cluster_entropy['mouse_name'][m] = mouse
        mouse_session_parts = all_cluster_repeats.loc[all_cluster_repeats['mouse_name']==mouse][['mouse_name', 'sample', 'y_kmeans', 'repeat']]
        mouse_repeats = mouse_session_parts['repeat'].unique()

        agree = []
        shuffle_agree = []
        
        for r, rep in enumerate(mouse_repeats):

            repeat_data = mouse_session_parts.loc[mouse_session_parts['repeat']==r, 'y_kmeans']
            # Count cluster label occurrences for sample i
            labels, counts = np.unique(np.array(repeat_data), return_counts=True)
            prob = counts / counts.sum()
            ent = entropy(prob)  # Shannon entropy
            # same = 1 if np.array(repeat_data)[0] == np.array(repeat_data)[1] else 0
            # agree.append(same)
            agree.append(ent)

            # Test for random cluster assignment
            shuffle = np.random.randint(np.min(all_cluster_repeats['y_kmeans']), np.max(all_cluster_repeats['y_kmeans']), len(repeat_data))
            # same_shuffle = 1 if shuffle[0]==shuffle[1] else 0
            # shuffle_agree.append(same_shuffle)
            _, shuffle_counts = np.unique(shuffle, return_counts=True)
            shuff_prob = shuffle_counts / shuffle_counts.sum()
            shuff_ent = entropy(shuff_prob)  # Shannon entropy
            shuffle_agree.append(shuff_ent)

        cluster_entropy['True'][m] = np.mean(agree)
        cluster_entropy['Shuffled'][m] = np.mean(shuffle_agree)
        cluster_entropy['session_len'][m] = np.floor(len(mouse_session_parts['sample'].unique())/2)

    melted_df = pd.melt(cluster_entropy, id_vars=['mouse_name', 'session_len'], value_vars=['True', 'Shuffled'])
    return melted_df