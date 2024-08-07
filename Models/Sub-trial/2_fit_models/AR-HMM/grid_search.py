











# %%

import os
import sys
import time
import pickle
from IPython.display import display, Javascript

# Save the state before restarting
def save_state(state, state_filename='state.pkl'):
    with open(state_filename, 'wb') as f:
        pickle.dump(state, f)

# Load the state after restarting
def load_state(state_filename='state.pkl'):
    with open(state_filename, 'rb') as f:
        return pickle.load(f)

# Restart the kernel
def restart_kernel():
    display(Javascript('Jupyter.notebook.kernel.restart()'))

# Restart the program
def restart_program():
    """Restarts the current program."""
    print("Restarting the program...")
    os.execv(sys.executable, ['python'] + sys.argv)

##################################################
################### MY CODE ######################
##################################################

import os
import autograd.numpy as np
import pickle
from collections import defaultdict
import tracemalloc
import gc
from one.api import ONE
import jax.numpy as jnp
import jax.random as jr
from dynamax.hidden_markov_model import LinearAutoregressiveHMM
import sys
import jax

# Get my functions
functions_path =  '/home/ines/repositories/representation_learning_variability/Models/Sub-trial//2_fit_models/'
#functions_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability//Models/Sub-trial//2_fit_models/'
os.chdir(functions_path)
from preprocessing_functions import concatenate_sessions
from fitting_functions import cross_validate_armodel, compute_inputs
one = ONE(mode="remote")

""" 
PARAMETERS
"""
bin_size = 0.1
multiplier = 1/bin_size
num_iters = 100
num_train_batches = 5
method = 'kmeans'
threshold = 0.05

""" 
FITTING PARAMETERS
"""
# Load preprocessed data
prepro_results_path =  '/home/ines/repositories/representation_learning_variability/DATA/Sub-trial/Results/' + str(bin_size) + '/'
os.chdir(prepro_results_path)
idxs, mouse_names, matrix_all, matrix_all_unnorm, session_all = pickle.load(open(prepro_results_path + "preprocessed_data_v4_170724", "rb"))
collapsed_matrices, collapsed_unnorm, collapsed_trials = concatenate_sessions (mouse_names, matrix_all, matrix_all_unnorm, session_all)

var_interest = 'whisker_me'
concatenate = False
num_states = 2
last_lag = 20
lag_step = 2
start_lag = 1
all_num_lags = list(range(start_lag, last_lag, lag_step))

sticky = False
if sticky == True:
    kappas = [1, 5, 10, 100, 500, 1000, 2000, 5000, 7000, 10000]
else:
    kappas = [0, 0.2, 0.5, 0.7, 1, 5, 10, 100, 500, 1000, 2000, 5000, 7000, 10000]
    kappas = [0, 0.2, 0.5, 0.7, 1, 5, 10, 100, 500, 1000, 2000, 5000, 7000, 10000]
    kappas = [0, 0.5, 1, 5, 10, 100, 1000, 5000, 10000]
    
use_sets = [['avg_wheel_vel'], ['Lick count'], ['whisker_me'],
            ['left_X', 'left_Y', 'right_X', 'right_Y'], ['nose_X', 'nose_Y']]
var_interest_map = ['avg_wheel_vel', 'Lick count', 'whisker_me', 'left_X', 'nose_X']
idx_init_list = [0, 1, 2, 3, 7]
idx_end_list = [1, 2, 3, 7, 9]


# Loop through mice to find next one
for m, mat in enumerate(idxs):
    if len(mat) > 35:
        mouse_name = mat[37:]
        session = mat[0:36]
        fit_id = str(mouse_name+session)
        if sticky:
            filename = "best_sticky_results_" + var_interest + '_' + fit_id
        else:
            filename = "best_results_" + var_interest + '_' + fit_id
        data_path =  '/home/ines/repositories/representation_learning_variability/DATA/Sub-trial/Results/' + str(bin_size) + '/grid_search/individual_sessions/'
        os.chdir(data_path)
        files = os.listdir()

    
""" 
FUNCTIONS
"""
# The good one
def grid_search_lag_kappa(id, matrix_all, collapsed_matrices, var_interest, var_interest_map, idx_init_list, idx_end_list, use_sets, concatenate, num_states, all_num_lags, kappas, sticky):
    
    index_var = np.where(np.array(var_interest_map)==var_interest)[0][0]
    idx_init = idx_init_list[index_var]
    idx_end = idx_end_list[index_var]
    var_names = use_sets[index_var]
    
    # Initialize vars for saving results
    all_init_params = defaultdict(list)
    all_fit_params = defaultdict(list)
    all_lls = defaultdict(list)
    all_baseline_lls = defaultdict(list)
    
    # Get mouse data
    if concatenate == True:
        mouse_name = id
        design_matrix = collapsed_matrices[mouse_name][:,idx_init:idx_end]
        fit_id = mouse_name
    else:
        mouse_name, session = id
        design_matrix = matrix_all[mouse_name][session][:,idx_init:idx_end]
        fit_id = str(mouse_name+session)

    if len(np.shape(design_matrix)) > 2:
        design_matrix = design_matrix[0]

    print('Fitting mouse ' + mouse_name)
        
    " Fit model with cross-validation"
    # Prepare data for cross-validationfrom jax.interpreters import xla

    num_timesteps = np.shape(design_matrix)[0]
    emission_dim = np.shape(design_matrix)[1]
    shortened_array = np.array(design_matrix[:(num_timesteps // num_train_batches) * num_train_batches])
    train_emissions = jnp.stack(jnp.split(shortened_array, num_train_batches))
    
    " Fit model with cross-validation across kappas and lags "
    for lag in all_num_lags:
        
        # print(f"fitting model with {lag} lags")
        # Initialize lag 
        all_lls[lag] = {}
        all_baseline_lls[lag] = {}
        all_init_params[lag] = {}
        all_fit_params[lag] = {}
        
        for kappa in kappas:
            
            # print(f"fitting model with kappa {kappa}")
            # Initialize stickiness 
            all_lls[lag][kappa] = {}
            all_baseline_lls[lag][kappa] = {}
            all_init_params[lag][kappa] = {}
            all_fit_params[lag][kappa] = {}
        
            # Make a range of Gaussian HMMs
            test_arhmm = LinearAutoregressiveHMM(num_states, emission_dim, num_lags=lag, transition_matrix_stickiness=kappa)
            # Compute inputs for required timelags
            my_inputs = compute_inputs(shortened_array, lag, emission_dim)
            train_inputs = jnp.stack(jnp.split(my_inputs, num_train_batches))
        
            all_val_lls, fit_params, init_params, baseline_lls = cross_validate_armodel(test_arhmm, jr.PRNGKey(0), shortened_array, 
                                                train_emissions, train_inputs, method, num_train_batches)
        
            # Save results
            all_lls[lag][kappa] = all_val_lls
            all_baseline_lls[lag][kappa] = baseline_lls
            all_init_params[lag][kappa] = init_params
            all_fit_params[lag][kappa] = fit_params
            
            gc.collect()
            # pdb.set_trace()
                
    mouse_results = all_lls, all_baseline_lls, all_init_params, all_fit_params, design_matrix, kappas, all_num_lags
    
    # Save design matrix
    data_path =  '/home/ines/repositories/representation_learning_variability/DATA/Sub-trial/Results/' + str(bin_size) + '/grid_search/individual_sessions/'
    os.chdir(data_path)
    
    if sticky == True:
        pickle.dump(mouse_results, open("best_sticky_results_" + var_names[0] + '_' + fit_id , "wb"))
    else:
        pickle.dump(mouse_results, open("best_results_" + var_names[0] + '_' + fit_id , "wb"))

    del mouse_results, all_lls, all_baseline_lls, all_init_params, all_fit_params
    
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')

    # print("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #     print(stat)
    
    # tracemalloc.clear_traces()
    # Clear caches and collect garbage
    jax.clear_caches()
    gc.collect()
                
                
# Example state
state = {'iteration': 0}

# Loop with kernel restarts
for m, mat in enumerate(idxs[40:]):
    state['iteration'] = 0
    print(f"Loop iteration {0}")
    
    # Your processing code here

    print(mat)
    mouse_name = mat[37:]
    session = mat[0:36]
    fit_id = str(mouse_name+session)
    # Check if session has been computed
    if sticky:
        filename = "best_sticky_results_" + var_interest + '_' + fit_id
    else:
        filename = "best_results_" + var_interest + '_' + fit_id
    if filename not in files:
        id = mouse_name, session
        print(filename)
        grid_search_lag_kappa(id, matrix_all, collapsed_matrices, var_interest, var_interest_map, idx_init_list, idx_end_list, use_sets, concatenate, num_states, all_num_lags, kappas, sticky)

    save_state(state)
    # restart_kernel()
    restart_program()
    time.sleep(10)  # Adjust sleep time if needed

    state = load_state()
    import os, sys, time, pickle  # Re-import modules
    
    print(f"Restored state: {state}")

print("Processing completed.")

# %%
