""" 
IMPORTS
"""
import os
import sys
import autograd.numpy as np
import pickle
import pandas as pd

from itertools import count
from one.api import ONE
import brainbox.behavior.wheel as wh

from functools import partial
from jax import vmap
from pprint import pprint
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Custom functions
functions_path =  '/home/ines/repositories/representation_learning_variability/Functions/'
#functions_path = '/Users/ineslaranjeira/Documents/Repositories/representation_learning_variability//Models/Sub-trial//2_fit_models/'
os.chdir(functions_path)
from one_functions_generic import prepro


""" DYNAMAC FUNCTIONS """

def cross_validate_model(model, key, all_emissions, train_emissions, method_to_use, num_train_batches, num_iters=100):
    # Initialize the parameters using K-Means on the full training set
    #init_params, props = model.initialize(key=key, method="kmeans", emissions=train_emissions)
    init_params, props = model.initialize(key=key, method=method_to_use, emissions=train_emissions)

    # Split the training data into folds.
    # Note: this is memory inefficient but it highlights the use of vmap.
    folds = jnp.stack([
        jnp.concatenate([train_emissions[:i], train_emissions[i+1:]])
        for i in range(num_train_batches)])

    # Baseline model has the same number of states but random initialization
    def _fit_fold_baseline(y_train, y_val):
        return model.marginal_log_prob(init_params, y_val) # np.shape(y_val)[1]
    
    # Then actually fit the model to data
    def _fit_fold(y_train, y_val):
        fit_params, train_lps = model.fit_em(init_params, props, y_train, 
                                             num_iters=num_iters, verbose=False)
        return model.marginal_log_prob(fit_params, y_val) , fit_params  # np.shape(y_val)[1]
    
    # Get log likelihood without cross-validation
    def _train_ll(all_emissions):
        fit_params_non_cv, _ = model.fit_em(init_params, props, all_emissions,
                                            num_iters=num_iters, verbose=False)
        ll_train = model.marginal_log_prob(fit_params_non_cv, all_emissions)/len(all_emissions)
        return ll_train

    ll_train = _train_ll(all_emissions)
    
    val_lls, fit_params = vmap(_fit_fold)(folds, train_emissions)
    
    baseline_val_lls = vmap(_fit_fold_baseline)(folds, train_emissions)

    return val_lls, fit_params, init_params, baseline_val_lls, ll_train


def find_permutation(z1, z2):
    K1 = z1.max() + 1
    K2 = z2.max() + 1

    perm = []
    for k1 in range(K1):
        indices = jnp.where(z1 == k1)[0]
        counts = jnp.bincount(z2[indices])
        perm.append(jnp.argmax(counts))

    return jnp.array(perm)

def plot_transition_matrix(transition_matrix):
    plt.imshow(transition_matrix, vmin=0, vmax=1, cmap="Greys")
    plt.xlabel("next state")
    plt.ylabel("current state")
    plt.colorbar()
    plt.show()


def compare_transition_matrix(true_matrix, test_matrix):
    # latexify(width_scale_factor=1, fig_height=1.5)
    figsize = (10, 5)
    if is_latexify_enabled():
        figsize = None
    latexify(width_scale_factor=1, fig_height=1.5)
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    out = axs[0].imshow(true_matrix, vmin=0, vmax=1, cmap="Greys")
    axs[1].imshow(test_matrix, vmin=0, vmax=1, cmap="Greys")
    axs[0].set_title("True Transition Matrix")
    axs[1].set_title("Test Transition Matrix")
    cax = fig.add_axes(
        [
            axs[1].get_position().x1 + 0.07,
            axs[1].get_position().y0,
            0.02,
            axs[1].get_position().y1 - axs[1].get_position().y0,
        ]
    )
    plt.colorbar(out, cax=cax)
    plt.show()


def plot_posterior_states(Ez, states, perm):
    # latexify(width_scale_factor=1, fig_height=1.5)
    figsize = (25, 5)
    if is_latexify_enabled():
        figsize = None
    plt.figure(figsize=figsize)
    plt.imshow(Ez.T[perm], aspect="auto", interpolation="none", cmap="Greys")
    plt.plot(states, label="True State", linewidth=1)
    plt.plot(Ez.T[perm].argmax(axis=0), "--", label="Predicted State", linewidth=1)
    plt.xlabel("time")
    plt.ylabel("latent state")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title("Predicted vs. Ground Truth Latent State")
    

def softmax(x, class_num, num_timesteps):
    # Initialize probabilities for each outcomes (as many as class_num)
    pi = np.zeros((class_num, num_timesteps))
    for i in range(class_num):
        pi[i] = np.exp(x[:, i]) / np.sum(np.exp(x[:, :]), axis=1)
    return pi


""" INES FUNCTIONS """
