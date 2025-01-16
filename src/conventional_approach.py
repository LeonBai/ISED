### The script hub for conventional approachs in ISED manuscript


##############################					#############################
##############################	General setup	##############################
##############################				    ##############################


import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import GPy
import phate
import tphate
from statsmodels.tsa.ar_model import AutoReg
from tensorflow.keras import layers, Model
import tensorflow as tf
#from cebra import CEBRA  ## Inconsistency with python version >3.7


##############################					#############################
##############################	Methods     	##############################
##############################				    ##############################


def load_and_preprocess_data(filepath, alpha_index=0):
    """
    Load and preprocess the data from the given file path.

    Parameters:
    - filepath (str): Path to the data file.
    - alpha_index (int): Index to select specific alpha data.

    Returns:
    - train_data (ndarray): Training data normalized.
    - test_data (ndarray): Test data normalized.
    """
    with open(filepath, 'rb') as file:
        raw_data = pickle.load(file)

    X_normalized = preprocessing.MinMaxScaler().fit_transform(raw_data[alpha_index])
    train_data = X_normalized[:500]  # First half for training
    test_data = X_normalized[500:820]  # Second half for testing

    return train_data, test_data


def perform_pca(train_data, test_data, n_components=2):
    """
    Perform Principal Component Analysis (PCA) on the training data.

    Parameters:
    - train_data (ndarray): Training data.
    - test_data (ndarray): Test data.
    - n_components (int): Number of principal components to keep.

    Returns:
    - pca_embedding (ndarray): Transformed test data using PCA.
    """
    pca = PCA(n_components=n_components)
    pca.fit(train_data)
    pca_embedding = pca.transform(test_data)
    return pca_embedding


def perform_pca_smooth(train_data, test_data, n_components=2, ar_order=150):
    """
    Perform PCA followed by smoothing using AutoRegressive (AR) models.

    Parameters:
    - train_data (ndarray): Training data.
    - test_data (ndarray): Test data.
    - n_components (int): Number of principal components.
    - ar_order (int): Order of the AR model for smoothing.

    Returns:
    - Z_pca_smooth (ndarray): PCA-smoothed embedding dynamics.
    """
    pca = PCA(n_components=n_components)
    pca.fit(train_data)
    Z_pca = preprocessing.MinMaxScaler().fit_transform(pca.transform(test_data))

    ar_models = []
    ar_params = []
    for i in range(n_components):
        model = AutoReg(Z_pca[:, i], lags=ar_order, old_names=False)
        model_fitted = model.fit()
        ar_models.append(model_fitted)
        ar_params.append(model_fitted.params)

    Z_ar = np.zeros(Z_pca.shape)
    for i in range(n_components):
        model_fitted = ar_models[i]
        predictions = model_fitted.predict(start=ar_order, end=len(Z_pca) - 1, dynamic=False)
        Z_ar[:, i] = np.concatenate((Z_pca[:ar_order, i], predictions))

    Z_pca_smooth = preprocessing.MinMaxScaler().fit_transform(Z_ar[ar_order:])
    return Z_pca_smooth


def perform_tsne(test_data, n_components=2):
    """
    Perform t-distributed Stochastic Neighbor Embedding (t-SNE) on test data.

    Parameters:
    - test_data (ndarray): Test data.
    - n_components (int): Number of dimensions for the embedding.

    Returns:
    - tsne_embedding (ndarray): t-SNE embedding of the test data.
    """
    tsne = TSNE(n_components=n_components)
    tsne_embedding = tsne.fit_transform(test_data)
    return tsne_embedding


def perform_umap(train_data, test_data, n_components=2):
    """
    Perform Uniform Manifold Approximation and Projection (UMAP) on training data and transform test data.

    Parameters:
    - train_data (ndarray): Training data.
    - test_data (ndarray): Test data.
    - n_components (int): Number of dimensions for the embedding.

    Returns:
    - umap_embedding (ndarray): UMAP embedding of the test data.
    """
    umap = UMAP(n_components=n_components)
    umap.fit(train_data)
    umap_embedding = umap.transform(test_data)
    return umap_embedding


def train_gp_lvm(train_data, test_data, num_latent_dimensions=2, num_iterations=1000):
    """
    Train a Gaussian Process Latent Variable Model (GP-LVM) and predict latent variables for the test data.

    Parameters:
    - train_data (ndarray): Training data.
    - test_data (ndarray): Test data.
    - num_latent_dimensions (int): Number of latent dimensions.
    - num_iterations (int): Number of iterations for model optimization.

    Returns:
    - model: Trained GP-LVM model.
    - reduced_train_data (ndarray): Latent positions for training data.
    - reduced_test_data (ndarray): Latent positions for test data.
    """
    kernel = GPy.kern.RBF(num_latent_dimensions, ARD=True)
    model = GPy.models.GPLVM(train_data, input_dim=num_latent_dimensions, kernel=kernel)
    model.optimize(messages=True, max_iters=num_iterations)
    reduced_train_data = model.X.mean
    reduced_test_data = model.predict(test_data)[0]
    return model, reduced_train_data, reduced_test_data


def perform_phate(test_data, n_components=2):
    """
    Perform PHATE embedding on test data.

    Parameters:
    - test_data (ndarray): Test data.
    - n_components (int): Number of dimensions for the embedding.

    Returns:
    - phate_embedding (ndarray): PHATE embedding of the test data.
    """
    phate_op = phate.PHATE(n_components=n_components)
    phate_embedding = phate_op.fit_transform(test_data)
    return phate_embedding


def perform_tphate(test_data, n_components=2):
    """
    Perform T-PHATE embedding on test data.

    Parameters:
    - test_data (ndarray): Test data.
    - n_components (int): Number of dimensions for the embedding.

    Returns:
    - tphate_embedding (ndarray): T-PHATE embedding of the test data.
    """
    tphate_op = tphate.TPHATE(n_components=n_components)
    tphate_embedding = tphate_op.fit_transform(test_data)
    return tphate_embedding


def train_ae_rnn(train_data, latent_dim=2, timesteps=500, input_dim=100, epochs=500, batch_size=100):
    """
    Train an Autoencoder (AE) with RNN on the training data and reduce the dimensionality.

    Parameters:
    - train_data (ndarray): Training data.
    - latent_dim (int): Number of latent dimensions.
    - timesteps (int): Number of time steps.
    - input_dim (int): Number of input features.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.

    Returns:
    - encoded_data (ndarray): Reduced dimensional representation of test data.
    """
    inputs = layers.Input(shape=(timesteps, input_dim))
    encoded = layers.GRU(latent_dim)(inputs)
    decoded = layers.RepeatVector(timesteps)(encoded)
    decoded = layers.GRU(input_dim, return_sequences=True)(decoded)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    autoencoder.fit(train_data, train_data, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[callback], verbose=0)
    encoder = Model(inputs, encoded)
    encoded_data = preprocessing.MinMaxScaler().fit_transform(encoder.predict(train_data))
    return encoded_data


def perform_cebra(train_data, test_data, output_dimension=2, max_iterations=1000, batch_size=32, learning_rate=3e-4):
    """
    Perform Contrastive Embedding via Realignment and Alignment (CEBRA) on the data.

    Parameters:
    - train_data (ndarray): Training data.
    - test_data (ndarray): Test data.
    - output_dimension (int): Dimensionality of the output embedding.
    - max_iterations (int): Maximum number of training iterations.
    - batch_size (int): Batch size for training.
    - learning_rate (float): Learning rate for optimization.

    Returns:
    - cebra_embedding (ndarray): Transformed test data using CEBRA.
    """
    cebra_model = CEBRA(
        model_architecture='offset10-model',
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=1.12,
        output_dimension=output_dimension,
        max_iterations=max_iterations,
        distance='cosine',
        conditional='time',
        verbose=True,
        time_offsets=10
    )

    cebra_model.fit(train_data)
    cebra_embedding = cebra_model.transform(test_data)
    return cebra_embedding
