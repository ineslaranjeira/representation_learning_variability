import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1"
import pandas as pd
import pickle
from joblib import Parallel, delayed
import numpy as np
import time

import jax
from jax import devices, default_device
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from scipy.stats import zscore
from jax import vmap
from pprint import pprint
import jax.numpy as jnp
import jax.random as jr
from collections import defaultdict
import gc
from dynamax.hidden_markov_model import LinearAutoregressiveHMM, PoissonHMM
from dynamax.hidden_markov_model import PoissonHMM 

# Get my functions
functions_path =  '/home/ines/repositories/representation_learning_variability/Models/Sub-trial//2_fit_models/'
#functions_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability//Models/Sub-trial//2_fit_models/'
os.chdir(functions_path)
from preprocessing_functions import idxs_from_files

"""" AR-HMM FITTING FUNCTIONS """

def cross_validate_armodel(model, key, train_emissions, train_inputs, method_to_use, num_train_batches, method, num_iters=100):
    # Initialize the parameters using K-Means on the full training set
    #init_params, props = model.initialize(key=key, method="kmeans", emissions=train_emissions)
    init_params, props = model.initialize(key=key, method=method_to_use, emissions=train_emissions)

    # Split the training data and the training inputs matrix_all[ses][0]into folds.
    # Note: this is memory inefficient but it highlights the use of vmap.
    folds = jnp.stack([
        jnp.concatenate([train_emissions[:i], train_emissions[i+1:]])
        for i in range(num_train_batches)])
    
    inpt_folds = jnp.stack([
        jnp.concatenate([train_inputs[:i], train_inputs[i+1:]])
        for i in range(num_train_batches)])
    
    # Baseline model has the same number of states but random initialization
    def _fit_fold_baseline(y_val, inpts):
        return model.marginal_log_prob(init_params, y_val, inpts) # np.shape(y_val)[1]

    baseline_val_lls = vmap(_fit_fold_baseline)(train_emissions, train_inputs)
    
    # Then actually fit the model to data
    if method == 'em':
        def _fit_fold(y_train, y_val, inpt_folds, inpts):
            fit_params, train_lps = model.fit_em(init_params, props, y_train, inpt_folds, 
                                                num_iters=num_iters, verbose=False)
            return model.marginal_log_prob(fit_params, y_val, inpts) , fit_params 
    elif method == 'sgd':
        def _fit_fold(y_train, y_val, inpt_folds, inpts):
            fit_params, train_lps = model.fit_sgd(init_params, props, y_train, inpt_folds, 
                                                num_epochs=num_iters)
            return model.marginal_log_prob(fit_params, y_val, inpts) , fit_params  
    
    val_lls, fit_params = vmap(_fit_fold)(folds, train_emissions, inpt_folds, train_inputs)
    
    return val_lls, fit_params, init_params, baseline_val_lls


def compute_inputs(emissions, num_lags, emission_dim):
    """Helper function to compute the matrix of lagged emissions.

    Args:
        emissions: $(T \times N)$ array of emissions
        prev_emissions: $(L \times N)$ array of previous emissions. Defaults to zeros.

    Returns:
        $(T \times N \cdot L)$ array of lagged emissions. These are the inputs to the fitting functions.
    """
    prev_emissions = jnp.zeros((num_lags, emission_dim))

    padded_emissions = jnp.vstack((prev_emissions, emissions))
    num_timesteps = len(emissions)
    return jnp.column_stack([padded_emissions[lag:lag+num_timesteps]
                                for lag in reversed(range(num_lags))])
    
"""" POISSON-HMM FITTING FUNCTIONS """

def cross_validate_poismodel(model, key, train_emissions, num_train_batches, fit_method, num_iters=100):
    # Initialize the parameters using K-Means on the full training set
    init_params, props = model.initialize(key=key)

    # Split the training data into folds.
    # Note: this is memory inefficient but it highlights the use of vmap.
    folds = jnp.stack([
        jnp.concatenate([train_emissions[:i], train_emissions[i+1:]])
        for i in range(num_train_batches)])

    # Baseline model has the same number of states but random initialization
    def _fit_fold_baseline(y_train, y_val):
        return model.marginal_log_prob(init_params, y_val) # np.shape(y_val)[1]
    
    # Then actually fit the model to data
    if fit_method == 'em':
        def _fit_fold(y_train, y_val):
            fit_params, train_lps = model.fit_em(init_params, props, y_train, 
                                                num_iters=num_iters, verbose=False)
            return model.marginal_log_prob(fit_params, y_val) , fit_params  
    elif fit_method == 'sgd':
        def _fit_fold(y_train, y_val):
            fit_params, train_lps = model.fit_sgd(init_params, props, y_train, 
                                                num_epochs=num_iters)
            return model.marginal_log_prob(fit_params, y_val) , fit_params  
    
    val_lls, fit_params = vmap(_fit_fold)(folds, train_emissions)
    
    baseline_val_lls = vmap(_fit_fold_baseline)(folds, train_emissions)

    return val_lls, fit_params, init_params, baseline_val_lls

"""" GRID SEARCH FUNCTIONS """

# Function to perform grid search for a single session
def grid_search_versatile(id, var_interest, zsc, fixed_states, truncate,
                          params, sticky, save_path, data_path, num_train_batches, method, fit_method):
    print(params)
    print(jax.devices())
    # Time job
    start = time.time()
    
    # Load mouse/session data
    mouse_name, session = id
    filename = data_path + "design_matrix_" + str(session) + '_'  + mouse_name
    original_design_matrix = pd.read_parquet(filename)

    # Zscore if needed (paws)
    if zsc == True:
      array_matrix = zscore(np.array(original_design_matrix), axis=0, nan_policy='omit')
    else:
      array_matrix = np.array(original_design_matrix)

    # Remove NaNs
    filtered_matrix = array_matrix[~np.isnan(array_matrix).any(axis=1)]
    keys = np.where(original_design_matrix.keys().isin(var_interest))
    if truncate:
        design_matrix = filtered_matrix[:100000, keys]  # Truncate session
    else:
        design_matrix = filtered_matrix[:, keys] 
    
    design_matrix = np.reshape(design_matrix,(np.shape(design_matrix)[0], np.shape(design_matrix)[2]))

    fit_id = str(mouse_name + session)

    print(f"Fitting session {fit_id}...")

    num_timesteps = np.shape(design_matrix)[0]
    emission_dim = np.shape(design_matrix)[1]
    shortened_array = np.array(design_matrix[:(num_timesteps // num_train_batches) * num_train_batches])
    train_emissions = jnp.stack(jnp.split(shortened_array, num_train_batches))

    # Initialize result containers
    all_lls, all_baseline_lls = defaultdict(dict), defaultdict(dict)
    all_init_params, all_fit_params = defaultdict(dict), defaultdict(dict)

    if var_interest == ['Lick count']:
        num_states, _, kappas = params
        if fixed_states:
            # Grid search
            for kappa in kappas:
                test_poishmm = PoissonHMM(num_states, emission_dim, transition_matrix_stickiness=kappa)
                all_val_lls, fit_params, init_params, baseline_lls = cross_validate_poismodel(test_poishmm, jr.PRNGKey(0),
                                                                                            train_emissions, num_train_batches, fit_method, num_iters=100)

                all_lls[kappa] = all_val_lls
                all_baseline_lls[kappa] = baseline_lls
                all_init_params[kappa] = init_params
                all_fit_params[kappa] = fit_params
        else:
            for state in num_states:
                # Grid search
                for kappa in kappas:
                    test_poishmm = PoissonHMM(state, emission_dim, transition_matrix_stickiness=kappa)
                    all_val_lls, fit_params, init_params, baseline_lls = cross_validate_poismodel(test_poishmm, jr.PRNGKey(0),
                                                                                               train_emissions, num_train_batches, fit_method, num_iters=100)

                    all_lls[state][kappa] = all_val_lls
                    all_baseline_lls[state][kappa] = baseline_lls
                    all_init_params[state][kappa] = init_params
                    all_fit_params[state][kappa] = fit_params

    else:
        num_states, all_num_lags, kappas = params
        if fixed_states:
            # Grid search
            for lag in all_num_lags:
                for kappa in kappas:
                    test_arhmm = LinearAutoregressiveHMM(num_states, emission_dim, num_lags=lag, transition_matrix_stickiness=kappa)
                    my_inputs = compute_inputs(shortened_array, lag, emission_dim)
                    train_inputs = jnp.stack(jnp.split(my_inputs, num_train_batches))

                    all_val_lls, fit_params, init_params, baseline_lls = cross_validate_armodel(
                        test_arhmm, jr.PRNGKey(0), train_emissions, train_inputs, method, num_train_batches, fit_method
                    )

                    all_lls[lag][kappa] = all_val_lls
                    all_baseline_lls[lag][kappa] = baseline_lls
                    all_init_params[lag][kappa] = init_params
                    all_fit_params[lag][kappa] = fit_params
        else:
            # Grid search
            for state in num_states:
                for lag in all_num_lags:
                    for kappa in kappas:
                        test_arhmm = LinearAutoregressiveHMM(state, emission_dim, num_lags=lag, transition_matrix_stickiness=kappa)
                        my_inputs = compute_inputs(shortened_array, lag, emission_dim)
                        train_inputs = jnp.stack(jnp.split(my_inputs, num_train_batches))

                        all_val_lls, fit_params, init_params, baseline_lls = cross_validate_armodel(
                            test_arhmm, jr.PRNGKey(0), train_emissions, train_inputs, method, num_train_batches, fit_method
                        )

                        all_lls[state][lag][kappa] = all_val_lls
                        all_baseline_lls[state][lag][kappa] = baseline_lls
                        all_init_params[state][lag][kappa] = init_params
                        all_fit_params[state][lag][kappa] = fit_params

    # Save results
    mouse_results = all_lls, all_baseline_lls, all_init_params, all_fit_params, design_matrix, params
    result_filename = os.path.join(save_path, f"{'best_sticky' if sticky else 'best'}_results_{var_interest[0]}_{fit_id}")
    with open(result_filename, "wb") as f:
        pickle.dump(mouse_results, f)

    print(f"Session {fit_id} completed.")
    print("Time:", time.time() - start)
    del mouse_results, all_lls, all_baseline_lls, all_init_params, all_fit_params
    gc.collect()


# Main function for parallel processing
def run_grid_search_parallel(idxs, var_interest, zsc, fixed_states, truncate, params, sticky,
                             save_path, data_path, num_train_batches, method, fit_method, n_jobs):

    # Create folder where to dump data
    if not os.path.exists(save_path):
        # Create a new directory because it does not exist
        os.makedirs(save_path)
   
    # Identify sessions to process
    sessions_to_process = []
    for m, mat in enumerate(idxs):
        mouse_name = mat[37:]
        session = mat[:36]
        fit_id = str(mouse_name + session)
        result_filename = os.path.join(save_path, f"{'best_sticky' if sticky else 'best'}_results_{var_interest[0]}_{fit_id}")
        if not os.path.exists(result_filename):
            sessions_to_process.append((mouse_name, session))
    # sessions_to_process = sessions_to_process

    print(f"Found {len(sessions_to_process)} sessions to process.")

    # Run grid search in parallel
    Parallel(n_jobs=n_jobs)(
        delayed(grid_search_versatile)(
            id, var_interest, zsc,
            fixed_states, truncate, params, sticky, save_path, data_path, num_train_batches, method, fit_method
        ) for id in sessions_to_process
    )

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""" 'RUN CODE' """"""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Parameters
n_jobs = 2  # Number of CPU cores to use
bin_size = 0.017
# LOAD DATA
data_path ='/home/ines/repositories/representation_learning_variability/DATA/Sub-trial/Design matrix/' + 'v6_21Jul2025/' + str(bin_size) + '/'
all_files = os.listdir(data_path)
design_matrices = [item for item in all_files if 'design_matrix' in item and 'standardized' not in item]
idxs, mouse_names = idxs_from_files(design_matrices, bin_size)


var_interest = ['l_paw_x', 'l_paw_y']
var_interest = ['r_paw_x', 'r_paw_y']
var_interest = ['whisker_me']
# var_interest = ['Lick count']

""" GRID """
all_num_lags = [1, 3, 5, 7, 10, 15, 20, 30]
all_num_lags=list(range(1, 20, 2))
kappas=[0, 0.5, 1, 2, 3, 4, 5, 10, 20, 100]

fixed_states = True
num_states = 2

if var_interest == ['whisker_me']:
    n_jobs = 1
    all_num_lags=[1, 10, 20, 40, 60] 
    # all_num_lags=list(range(1, 60, 10))
    kappas=[0, 5, 50]
    num_states = 2
elif var_interest == ['avg_wheel_vel']:
    n_jobs = 2
    all_num_lags=[1, 5, 7, 10, 15, 20] 
    kappas=[0, 1, 5, 50, 100]
    num_states = 3
elif var_interest == ['Lick count']:
    n_jobs = 3
    kappas=[0, 1, 5, 10, 50, 100, 500, 1000]
    num_states = 2
elif var_interest == ['l_paw_x', 'l_paw_y'] or var_interest == ['r_paw_x', 'r_paw_y']:
    n_jobs = 2
    all_num_lags=list(range(1, 40, 5))
    kappas=[0, 1, 5, 50, 100]
    num_states = 2

params = num_states, all_num_lags, kappas

""" FITTING PARAMETERS """
num_train_batches = 5
method='prior'
fit_method='em'
zsc = False
truncate = False

if var_interest == ['whisker_me']:
    zsc = False
elif var_interest == ['avg_wheel_vel']:
    zsc = True
elif var_interest == ['Lick count']:
    zsc = False
elif var_interest == ['l_paw_x', 'l_paw_y'] or var_interest == ['r_paw_x', 'r_paw_y']:
    zsc = True

fitting_params = str(num_train_batches)+'_'+method+'_'+fit_method+'_zsc_'+str(zsc)+'/'
if truncate:
    fitting_params = str(num_train_batches)+'_'+method+'_'+fit_method+'_zsc_'+str(zsc)+'_truncated/'
if fixed_states == True:
    save_path = '/home/ines/repositories/representation_learning_variability/DATA/Sub-trial/Results/'  + str(bin_size) + '/'+str(num_states)+'_states/grid_search/individual_sessions/v6_21Jul2025/'+fitting_params
else:
    save_path = '/home/ines/repositories/representation_learning_variability/DATA/Sub-trial/Results/'  + str(bin_size) + '/grid_search/individual_sessions/v6_21Jul2025/'+fitting_params
    num_states = list(range(1, 10, 1))

# Run the grid search
run_grid_search_parallel(
    idxs, var_interest, zsc,
    fixed_states=fixed_states, truncate=truncate, params=params, sticky=False,
    save_path=save_path,  data_path=data_path, num_train_batches=num_train_batches, method=method, fit_method=fit_method, n_jobs=n_jobs)

