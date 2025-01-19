import os

import pickle
from joblib import Parallel, delayed
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from collections import defaultdict
import gc
from dynamax.hidden_markov_model import LinearAutoregressiveHMM

# # Get my functions
# functions_path =  '/home/ines/repositories/representation_learning_variability/Models/Sub-trial//2_fit_models/'
# #functions_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability//Models/Sub-trial//2_fit_models/'
# os.chdir(functions_path)
# from preprocessing_functions import concatenate_sessions
# from fitting_functions import cross_validate_armodel, compute_inputs

# Joblib breaks when I import functions, so I copied them here from fitting_functions
from functools import partial
from jax import vmap
from pprint import pprint
import jax.numpy as jnp
import jax.random as jr

def cross_validate_armodel(model, key, all_emissions, train_emissions, train_inputs, method_to_use, num_train_batches, num_iters=100):
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
    def _fit_fold(y_train, y_val, inpt_folds, inpts):
        fit_params, train_lps = model.fit_em(init_params, props, y_train, inpt_folds, 
                                             num_iters=num_iters, verbose=False)
        return model.marginal_log_prob(fit_params, y_val, inpts) , fit_params  # np.shape(y_val)[1]
    
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
    
    
    
# Function to perform grid search for a single session
def grid_search_lag_kappa_session(id, matrix_all, var_interest, var_interest_map, 
                                  idx_init_list, idx_end_list, use_sets, num_states, 
                                  all_num_lags, kappas, sticky, save_path, num_train_batches, method):
    # Load mouse/session data
    index_var = np.where(np.array(var_interest_map) == var_interest)[0][0]
    idx_init = idx_init_list[index_var]
    idx_end = idx_end_list[index_var]
    var_names = use_sets[index_var]

    mouse_name, session = id
    design_matrix = matrix_all[mouse_name][session][:, idx_init:idx_end]
    fit_id = str(mouse_name + session)

    if len(np.shape(design_matrix)) > 2:
        design_matrix = design_matrix[0]

    print(f"Fitting session {fit_id}...")

    num_timesteps = np.shape(design_matrix)[0]
    emission_dim = np.shape(design_matrix)[1]
    shortened_array = np.array(design_matrix[:(num_timesteps // num_train_batches) * num_train_batches])
    train_emissions = jnp.stack(jnp.split(shortened_array, num_train_batches))

    # Initialize result containers
    all_lls, all_baseline_lls = defaultdict(dict), defaultdict(dict)
    all_init_params, all_fit_params = defaultdict(dict), defaultdict(dict)

    # Grid search
    for lag in all_num_lags:
        for kappa in kappas:
            test_arhmm = LinearAutoregressiveHMM(num_states, emission_dim, num_lags=lag, transition_matrix_stickiness=kappa)
            my_inputs = compute_inputs(shortened_array, lag, emission_dim)
            train_inputs = jnp.stack(jnp.split(my_inputs, num_train_batches))

            all_val_lls, fit_params, init_params, baseline_lls = cross_validate_armodel(
                test_arhmm, jr.PRNGKey(0), shortened_array, train_emissions, train_inputs, method, num_train_batches
            )

            all_lls[lag][kappa] = all_val_lls
            all_baseline_lls[lag][kappa] = baseline_lls
            all_init_params[lag][kappa] = init_params
            all_fit_params[lag][kappa] = fit_params

    # Save results
    mouse_results = all_lls, all_baseline_lls, all_init_params, all_fit_params, design_matrix, kappas, all_num_lags
    result_filename = os.path.join(save_path, f"{'best_sticky' if sticky else 'best'}_results_{var_names[0]}_{fit_id}.pkl")
    with open(result_filename, "wb") as f:
        pickle.dump(mouse_results, f)

    print(f"Session {fit_id} completed.")
    del mouse_results, all_lls, all_baseline_lls, all_init_params, all_fit_params
    gc.collect()

# Main function for parallel processing
def run_grid_search_parallel(matrix_all, idxs, var_interest, var_interest_map, idx_init_list,
                             idx_end_list, use_sets, num_states, all_num_lags, kappas, sticky,
                             save_path, num_train_batches, method, n_jobs):
    # Identify sessions to process
    sessions_to_process = []
    for m, mat in enumerate(idxs):
        mouse_name = mat[37:]
        session = mat[:36]
        fit_id = str(mouse_name + session)
        result_filename = os.path.join(save_path, f"{'best_sticky' if sticky else 'best'}_results_{var_interest}_{fit_id}.pkl")
        if not os.path.exists(result_filename):
            sessions_to_process.append((mouse_name, session))

    print(f"Found {len(sessions_to_process)} sessions to process.")

    # Run grid search in parallel
    Parallel(n_jobs=n_jobs)(
        delayed(grid_search_lag_kappa_session)(
            id, matrix_all, var_interest, var_interest_map, idx_init_list, idx_end_list, use_sets,
            num_states, all_num_lags, kappas, sticky, save_path, num_train_batches, method
        ) for id in sessions_to_process
    )

# Parameters
n_jobs = 6  # Number of CPU cores to use
bin_size = 0.017
num_states=2
var_interest = 'whisker_me'
save_path = '/home/ines/repositories/representation_learning_variability/DATA/Sub-trial/Results/'  + str(bin_size) + '/'+str(num_states)+'_states/grid_search/individual_sessions/'

# Load data
data_file = "preprocessed_data_v5_01-17-2025"
if data_file == "preprocessed_data_v4_170724":
    use_sets = [['avg_wheel_vel'], ['Lick count'], ['whisker_me'],
                ['left_X', 'left_Y', 'right_X', 'right_Y'], ['nose_X', 'nose_Y']]
    var_interest_map = ['avg_wheel_vel', 'Lick count', 'whisker_me', 'left_X', 'nose_X']
    idx_init_list = [0, 1, 2, 3, 7]
    idx_end_list = [1, 2, 3, 7, 9]
    
    # Load preprocessed data
    prepro_results_path =  '/home/ines/repositories/representation_learning_variability/DATA/Sub-trial/Results/90_trials/' + str(bin_size) + '/'
    idxs, mouse_names, matrix_all, matrix_all_unnorm, session_all = pickle.load(open(prepro_results_path + data_file, "rb"))
    # collapsed_matrices, collapsed_unnorm, collapsed_trials = concatenate_sessions (mouse_names, matrix_all, matrix_all_unnorm, session_all)
elif data_file == "preprocessed_data_v4_171224_alltrials":
    use_sets = [['avg_wheel_vel'], ['whisker_me'], ['Lick count']]
    var_interest_map = ['avg_wheel_vel', 'whisker_me', 'Lick count']
    idx_init_list = [0, 1, 2]
    idx_end_list = [1, 2, 3]
    
    # Load preprocessed data
    prepro_results_path =  '/home/ines/repositories/representation_learning_variability/DATA/Sub-trial/Results/' + str(bin_size) + '/'
    idxs, mouse_names, matrix_all, matrix_all_unnorm, session_all = pickle.load(open(prepro_results_path + data_file, "rb"))
    # collapsed_matrices, collapsed_unnorm, collapsed_trials = concatenate_sessions (mouse_names, matrix_all, matrix_all_unnorm, session_all)
elif data_file == "preprocessed_data_v5_01-17-2025":
    use_sets = [['avg_wheel_vel'], ['whisker_me'], ['Lick count'], ['0.25', '0.5',
       '1.0', '2.0', '4.0', '8.0', '16.0']]
    var_interest_map = ['avg_wheel_vel', 'whisker_me', 'Lick count', '0.25', '0.5',
       '1.0', '2.0', '4.0', '8.0', '16.0']
    idx_init_list = [0, 1, 2, 3]
    idx_end_list = [1, 2, 3, 9]
    
    # Load preprocessed data
    prepro_results_path =  '/home/ines/repositories/representation_learning_variability/DATA/Sub-trial/Results/' + str(bin_size) + '/'
    filename = prepro_results_path + data_file
    idxs, mouse_names, matrix_all, matrix_all_unnorm, session_all = pickle.load(open(filename, "rb"))
    # collapsed_matrices, collapsed_unnorm, collapsed_trials = concatenate_sessions (mouse_names, matrix_all, matrix_all_unnorm, session_all)


# Run the grid search
run_grid_search_parallel(
    matrix_all, idxs, var_interest, var_interest_map, idx_init_list, idx_end_list, use_sets,
    num_states=num_states, all_num_lags=list(range(1, 20, 2)), kappas=[0, 1, 10, 100], sticky=False,
    save_path=save_path, num_train_batches=5, method='kmeans', n_jobs=n_jobs
)