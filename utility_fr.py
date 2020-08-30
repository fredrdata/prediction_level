"""
Imports
"""

import numpy as np
import lightgbm as lgbm

from hyperopt import hp, tpe, fmin, Trials

import tensorflow as tf

from keras import optimizers, layers
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense

from functools import partial


'''
Prediction Models
'''


class mdl_naive():
    # Naive Predictor

    def __init__(self):
        """
        Initializer
        """

    def fit(self, params=[]):
        self.fit = True
        return self

    def predict(self, X, y=None):
        return X[:, -1]


class mdl_mean():
    # Mean Predictor: Average of n Last Samples

    def __init__(self, nsteps=1):
        # Initializer
        self.nsteps = nsteps

    def fit(self, params=[]):
        self.fit = True
        return self

    def predict(self, X, y=None):
        ns = np.min([X.shape[1], self.nsteps])
        return X[:, -ns:].mean(1)


class mdl_trend():
    # Trend Predictor: Average of n Differences

    def __init__(self, ndiff=1, T=1):
        # Initializer
        self.ndiff = ndiff
        self.T = T

    def fit(self, params=[]):
        self.fit = True
        return self

    def predict(self, X, y=None):
        df = np.min([X.shape[1]-1, self.ndiff])
        return X[:, -1] + self.T*np.diff(X[:, -df-1:], axis=1).mean(1)


class mdl_ar():
    """
    AR Predictor: Straight-Forward Least-Squares Estimation of AR-Coefficients
    (Data-Dimension Determines No. of Coefficients)
    """

    def __init__(self):
        """
        Initializer
        """

    def fit(self, params):
        # LS-fit for AR-Coefficients (Using Pseudoinverse)
        self.mdl = np.matmul(np.linalg.pinv(params["X_train"]), params["y_train"])
        self.fit = True
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "fit")
        except AttributeError:
            raise RuntimeError("Train before prediction.")

        return np.hstack([np.matmul(self.mdl, Xk) for Xk in X])


class mdl_FCNN():
    # Keras Fully Connected NN Based on Model Specification
    # The Structure of NN is Cone-Like {Wide (input) to Narrow (out)}

    def __init__(self, **params):
        # Initializer
        for arg, val in params.items():
            setattr(self, arg, val)
        
        self.spec_mdl = params

    def fit(self, params):
        # Fit Regressor

        # Random Seeds for Reproducibility
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        # Early Stopping
        cback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        # Get The Cone-Like FCNN-Model
        self.spec_mdl["no_feat"] = params["X_train"].shape[1]
        self.mdl = get_NN_cone(self.spec_mdl)

        # Train Model, Stop When Validation Error (MSE) Does Not Decrease
        self.mdl.fit(params["X_train"],
                       params["y_train"],
                       epochs=self.n_epoch,
                       batch_size=self.bSize,
                       validation_data=(params["X_val"], params["y_val"]),
                       callbacks=cback,
                       verbose=0)

        self.fit = True

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "fit")
        except AttributeError:
            raise RuntimeError("Train before prediction.")

        return self.mdl.predict(X, verbose=0)


def get_NN_cone(spec):
    # Create Cone-Like Keras Model
    K.clear_session()

    # Parameters
    n_layers = spec["n_layers"]
    le_y0 = spec["le_y0"]
    le_out = spec["out"]
    activs = [spec["activ"]]*n_layers
    biases = [spec["bias"]]*n_layers

    # If Only One Activation (Except for Linear) in Total: Place in Middle
    if spec["mid_act_only"] and n_layers > 2:
        m_layer = int(n_layers/2)
        for im in range(n_layers):
            activs[im] = 'linear' if im!=m_layer else activs[im]

    # Adjust Output Size of Each Layer
    k = (le_y0 - le_out)/(n_layers-1)
    outs = np.arange(le_y0, le_out-k, -k).round().astype(int)
    if len(outs) > n_layers:
        outs = outs[:-1]
        outs[-1] = le_out
    elif len(outs) < n_layers:
        noMiss = len(outs) - n_layers
        outs = outs + [le_out]*noMiss

    # Build Network Based On Specification
    layer_list = [
        layers.Dense(outs[0], input_dim=spec["no_feat"], activation=activs[0], use_bias=biases[0])
    ]

    for it, lsize in enumerate(outs[1:-1]):
        layer_list.append(layers.Dense(lsize, activation=activs[it], use_bias=biases[it]))

    layer_list.append(layers.Dense(outs[-1], use_bias=biases[-1]))

    # Create Sequential Cone-Like Model
    mdl_out = Sequential(layer_list)
    mdl_out.compile(optimizer=optimizers.Adam(spec["lr"]), loss='mean_squared_error')

    return mdl_out


class mdl_lgbm():
    # LGBM Regressor

    def __init__(self, **params):
        # Initializer
        self.mdl = lgbm.LGBMRegressor(**params)

    def fit(self, params):
        # Train Model
        self.mdl.fit(params["X_train"], params["y_train"])
        self.mdl.fit = True
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "fit")
        except AttributeError:
            raise RuntimeError("Train before prediction.")

        return self.mdl.predict(X)


'''
Hyperparameter Optimization
'''


def mdl_fit_pred(params, mdl_name, data, eval_suffix, ids=[]):

    # Train (fit) & Evaluate Model given Model Name & Parameters

    # Training
    p_model = globals()[mdl_name](**params)
    p_model.fit(data)

    # Prediction
    y_true = data[f'y_{eval_suffix}']  # True Values
    if not len(ids):
        y_pred = p_model.predict(data[f'X_{eval_suffix}']).reshape(y_true.shape)
    else:
        y_true = y_true[ids]
        y_pred = p_model.predict(data[f'X_{eval_suffix}'][ids]).reshape(y_true.shape)
    
    return y_true, y_pred


def mdl_eval(params, mdl_name, data, eval_suffix='val', ids=[], metric='rmse'):

    # Train Model and Get Prediction
    y_true, y_pred = mdl_fit_pred(params, mdl_name, data, eval_suffix, ids)

    # Return Evaluation Result
    return globals()[metric](y_true, y_pred)


def mdl_hypopt(mdl_name, para_space, data_ML, maxIter=1000, verb=False, time_stop=None, rstate=0):

    # Hyperparameter Optimization of Model

    # Create Training Function for 'mdl_name' including ML-data
    opt_func = partial(
        mdl_eval,
        mdl_name=mdl_name,
        data=data_ML
    )

    # Hyperparameter Optimization
    hOpt_trials = Trials()
    best = fmin(
        fn=opt_func,
        space=para_space,
        algo=tpe.suggest,
        return_argmin=False,
        trials=hOpt_trials,
        timeout=time_stop,
        verbose=verb,
        rstate=np.random.RandomState(rstate),
        max_evals=maxIter)

    # Return the best set of parameters
    return best


'''
Functions for Preparing ML Data
'''


def get_xy_data(data, length, lead):

    N = len(data)

    id_end = list(range(N-1, lead+length-2, -1))

    X, y = [], []
    for id_ in reversed(id_end):
        y.append(data[id_])
        X.append(data[id_-lead-length+1:id_-lead+1])

    return np.vstack(X), np.hstack(y)


def kfold_win(N, k, size=0.8):

    # Outputs max k sets of indices, each of size: size*N

    end_start = np.ceil((1-size)*N)
    len_segment = N - end_start

    id_start = np.arange(0, end_start+1, np.max([np.floor(end_start/(k-1)), 1]))
    id_start = id_start[-k:] if len(id_start)>k else id_start
    id_kfold = np.vstack([np.arange(ids, ids+len_segment) for ids in id_start])

    return id_kfold.astype(int)


'''
Misc Statistics
'''


def rmse(y_true, y_pred):
    # Root Mean Square of Errors
    return np.sqrt(((y_true - y_pred)**2).mean())
