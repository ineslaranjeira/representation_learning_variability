"""
IMPORTS
"""
import autograd.numpy as np
import os 

# --Machine learning and statistics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.cluster import KMeans
import umap
from sklearn import mixture


""" Cluster assessment """

def GMM_log_likelihood(embedding, components):
    
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