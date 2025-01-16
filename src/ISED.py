import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from scipy.signal import periodogram
from skimage.util import view_as_windows
import skdim
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.losses import mse,binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import sklearn.linear_model as linear_model


initializer = tf.keras.initializers.Orthogonal()
mydtype = tf.float64

import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import periodogram
from skimage.util import view_as_windows
import skdim
from sklearn.model_selection import GridSearchCV

class DataProcessor:
    def __init__(self, filepath, alpha=0.5):
        """
        Initialize the DataProcessor with a file path and optional alpha value.

        Parameters:
        - filepath (str): Path to the data file (supports .pkl or .npy).
        - alpha (float): Specific alpha value for the dataset (used only for pickle files).
        """
        self.filepath = filepath
        self.alpha = alpha
        self.X_normalized = None
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None

    def load_and_preprocess_data(self, method='sliding', latent_dim=None, window_length=None):
        """
        Load, preprocess, and prepare subsequences from data.

        Supports both pickle and numpy file formats.

        Parameters:
        - method (str): Method to use for generating subsequences ('buffering', 'appending', or 'sliding').
        - latent_dim (int): The width of each window, set to the latent dimension.
        - window_length (int): User-defined window length for generating subsequences.

        Returns:
        - X_train (ndarray): Training subsequences.
        - X_test (ndarray): Test subsequences.
        - X_normalized (ndarray): Full normalized dataset.
        """
        # Load data based on file extension
        if self.filepath.endswith('.pkl'):
            with open(self.filepath, 'rb') as file:
                raw_data = pickle.load(file)
            data = raw_data[int(self.alpha * 2)]  # alpha=0.5 is index 1
        elif self.filepath.endswith('.npy'):
            data = np.load(self.filepath)
        else:
            raise ValueError("Unsupported file format. Use .pkl or .npy files.")

        # Normalize the data
        self.X_normalized = MinMaxScaler().fit_transform(data)

        # Determine window length
        if window_length is None:
            # Calculate the dominant period to determine the default window length (k)
            dominant_period = self.calculate_dominant_period()
            window_length = int(dominant_period) if dominant_period else 100  # Default to 100 if no dominant period is found

        # Choose the subsequencing method
        if method == 'buffering':
            # Generate circular buffer windows
            self.X_train, self.X_test = self.generate_circular_buffer_windows(window_length)
        elif method == 'appending':
            # Generate subsequences using the appending method
            self.X_train, self.X_test = self.generate_appending_subsequences(window_length)
        elif method == 'sliding':
            # Generate sliding windows
            latent_dim = latent_dim or self.X_normalized.shape[1]
            self.X_train, self.X_test = self.generate_sliding_windows(window_length, latent_dim)
        else:
            raise ValueError("Unsupported method. Choose 'buffering', 'appending', or 'sliding'.")

        return self.X_train, self.X_test, self.X_normalized

    def generate_circular_buffer_windows(self, window_length):
        """
        Generate circular buffer windows.

        Parameters:
        - window_length (int): The length of each window.

        Returns:
        - X_train (ndarray): Training subsequences.
        - X_test (ndarray): Test subsequences.
        """
        n_samples = self.X_normalized.shape[0]
        X_circular = np.concatenate((self.X_normalized, self.X_normalized[:window_length - 1]), axis=0)
        X_subsequence = np.squeeze(view_as_windows(X_circular, window_shape=(window_length, self.X_normalized.shape[-1]), step=1))
        X_subsequence = X_subsequence[:n_samples]  # Keep only the necessary number of windows

        # Split into train and test subsequences
        midpoint = X_subsequence.shape[0] // 2
        return X_subsequence[:midpoint], X_subsequence[midpoint:]

    def generate_appending_subsequences(self, window_length):
        """
        Generate subsequences using an appending method to maintain the original number of time steps.

        Parameters:
        - window_length (int): The length of each subsequence window.

        Returns:
        - X_train (ndarray): Training subsequences.
        - X_test (ndarray): Test subsequences.
        """
        n_samples, n_features = self.X_normalized.shape

        # Initialize an empty array for storing the subsequences
        X_appended = np.zeros((n_samples, window_length, n_features))

        # Generate subsequences by sliding window and appending
        for i in range(n_samples):
            # Determine the start and end indices for each window
            start_idx = max(0, i - window_length + 1)
            end_idx = i + 1

            # Fill the subsequence with appropriate data
            X_window = self.X_normalized[start_idx:end_idx]

            # If the window is smaller than the window_length, pad it with zeros at the beginning
            if X_window.shape[0] < window_length:
                X_appended[i] = np.pad(X_window, ((window_length - X_window.shape[0], 0), (0, 0)), mode='constant')
            else:
                X_appended[i] = X_window

        # Split into train and test subsequences
        midpoint = X_appended.shape[0] // 2
        return X_appended[:midpoint], X_appended[midpoint:]

    def generate_sliding_windows(self, window_length, latent_dim):
        """
        Generate sliding windows.

        Parameters:
        - window_length (int): The length of each window.
        - latent_dim (int): The width of each window, set to the latent dimension.

        Returns:
        - X_train (ndarray): Training subsequences.
        - X_test (ndarray): Test subsequences.
        """
        X_subsequence = np.squeeze(view_as_windows(self.X_normalized, window_shape=(window_length, latent_dim)))

        # Split into train and test subsequences
        midpoint = X_subsequence.shape[0] // 2
        return X_subsequence[:midpoint], X_subsequence[midpoint:]

    def estimate_intrinsic_dimension(self, n_neighbors=20, n_jobs=1):
        """
        Estimate the intrinsic dimensionality of the data using local PCA.

        Parameters:
        - n_neighbors (int): Number of neighbors to consider.
        - n_jobs (int): Number of jobs to run in parallel.

        Returns:
        - int: Estimated intrinsic dimensionality.
        """
        lpca = skdim.id.lPCA().fit_pw(self.X_normalized, n_neighbors=n_neighbors, n_jobs=n_jobs)
        return int(np.mean(lpca.dimension_pw_))

    def calculate_dominant_period(self):
        """
        Calculate the dominant period of the dataset.

        Returns:
        - float: Optimal subsequence length.
        """
        wavelengths, power_values = [], []

        for i in range(self.X_normalized.shape[1]):
            f, Pxx = periodogram(self.X_normalized[:, i], fs=1)
            peak_idx = np.argmax(Pxx)
            peak_frequency = f[peak_idx]
            if peak_frequency > 0:
                wavelength = 1 / peak_frequency
                wavelengths.append(wavelength)
                power_values.append(Pxx[peak_idx])

        return (wavelengths[np.argmax(power_values)] / 4) if wavelengths else None

    def perform_window_length_grid_search(self, min_length, max_length, step=1, scoring='accuracy', cv=3):
        """
        Perform grid search for optimal window length.

        Parameters:
        - min_length (int): Minimum window length to consider.
        - max_length (int): Maximum window length to consider.
        - step (int): Step size for window length.
        - scoring (str): Scoring metric for grid search.
        - cv (int): Number of cross-validation folds.

        Returns:
        - dict: Best window length and corresponding score.
        """
        best_score = -np.inf
        best_window_length = None

        for window_length in range(min_length, max_length + 1, step):
            self.load_and_preprocess_data(window_length=window_length)
            # Here, you would train a model and evaluate it (e.g., using cross-validation)
            # For demonstration purposes, we'll assume a dummy scoring function
            score = self.dummy_model_score()
            if score > best_score:
                best_score = score
                best_window_length = window_length

        return {'best_window_length': best_window_length, 'best_score': best_score}

tf.keras.backend.set_floatx('float64')
mydtype = tf.float64

initializer = tf.keras.initializers.Orthogonal()

class Jacobian(layers.Layer):

  def __init__(self, func,  **kwargs):
    ''' input-output shapes of func should be (batch, dim) -> (batch, dim)
    '''
    self.func = func
    super(Jacobian, self).__init__(**kwargs)

  def call(self, X, t_length=None):
    # x have (batch, timestep, dim)
    if t_length == None:
      t_length = X.shape[1]
    batch_size = X.shape[0]

    X = tf.reshape(X, [batch_size*t_length, X.shape[2]])
    #X = tf.reshape(X, [-1, 2])
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tape:
      tape.watch(X)
      X_next, _ = self.func(X,[X])

    Jxs = tape.batch_jacobian(X_next, X)

    Jxs = tf.reshape(Jxs, [batch_size, t_length, X.shape[1], X.shape[1]])
    return Jxs

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = backend.dot(inputs, self.kernel)
        output = h + backend.dot(prev_output, self.recurrent_kernel)
        #output = tf.stack([output])
        return output, [output]

class QRDcell(layers.Layer):
  '''
  performing successive QR decomposition.
  This class can be used as a RNN cell in Keras RNN Layer
  '''

  def __init__(self, dim, **kwargs):
    super(QRDcell, self).__init__(**kwargs)
    self.dim = dim
    # d x d dimension (d is a dimension of dynamical systems)
    self.state_size = tf.constant([dim, dim])
    self.output_size = tf.constant(dim)

  def get_initial_state(self, inputs=None, batch_size=None, dtype=mydtype):
    ''' return identity matrices'''

    return tf.linalg.eye(self.dim, self.dim, batch_shape=[batch_size], dtype=mydtype)

  def call(self, inputs, states):
    # inputs is  J_{n} (batch, dim, dim)
    # states is Q_{n-1} (batch, dim,dim). Q_{0} is identity matrix
    # Q_{n}R_{n} = J_{n}Q_{n-1}
    # Q_{n} is the next state. (Q_new)
    # R_{n} is the output. (R_new)

    J = inputs
    Q = states[0]
    Q_new, R_new = tf.linalg.qr(J@Q)
    return R_new, [Q_new]

class ISEDModel:
    def __init__(self,
                 input_dim=100,
                 seq_length=125,
                 latent_dim=20,
                 batch_size=50,
                 epochs=300,
                 use_early_stopping=True,
                 encoder_layers=[(50, 'relu'), (20, 'relu')],
                 rnn_layers=[(30, 'gru'), (20, 'gru')],
                 decoder_layers=[(20, 'relu'), (30, 'relu')],
                 optimizer='adam',
                 loss_weights=None,
                 verbose = 1):
        """
        Initialize the ISED model with the given parameters.

        Parameters:
        - input_dim (int): Dimensionality of the input features.
        - seq_length (int): Length of the input sequences.
        - latent_dim (int): Dimensionality of the latent space.
        - batch_size (int): Batch size used in training.
        - epochs (int): Number of training epochs.
        - use_early_stopping (bool): Whether to use early stopping.
        - encoder_layers (list of tuples): List of encoder layers with (units, activation).
        - rnn_layers (list of tuples): List of RNN layers with (units, type ('gru' or 'lstm')).
        - decoder_layers (list of tuples): List of decoder layers with (units, activation).
        - optimizer (str): Optimizer for model compilation.
        - loss_weights (dict): Weights for different losses.
        """
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_early_stopping = use_early_stopping
        self.encoder_layers = encoder_layers
        self.rnn_layers = rnn_layers
        self.decoder_layers = decoder_layers
        self.optimizer = optimizer
        self.loss_weights = loss_weights or {'MI_loss': 1.0, 'GSM_loss': 1.0, 'LSM_loss': 1.0, 'kl_loss': 1.0}
        self.verbose = verbose
        self.model = None
        self.embedding_transformer = None

    def _build_model(self):
        """
        Build the ISED model with customizable layers.

        Returns:
        - Model: Compiled Keras model ready for training.
        """
        # Input layer
        x_inputs = layers.Input((self.seq_length, self.input_dim), batch_size=self.batch_size)
        encoder_input = Input(self.input_dim)

        # Encoder network
        x = encoder_input
        for units, activation in self.encoder_layers:
            x = Dense(units=units, activation=activation)(x)
            x = BatchNormalization()(x)

        encoder_output = Dense(self.latent_dim, activation=None, name='encoder_embedding',
                              kernel_initializer = initializer)(x)
        encoder_model = Model(encoder_input, encoder_output, name='encoder')

        x_encoded_sequence = TimeDistributed(encoder_model)(x_inputs)

        # RNN Layers for Temporal Learning
        x = x_encoded_sequence
        for units, layer_type in self.rnn_layers:
            if layer_type.lower() == 'gru':
                x = RNN(GRUCell(units), return_sequences=True)(x)
            elif layer_type.lower() == 'lstm':
                x = RNN(LSTMCell(units), return_sequences=True)(x)

        # Jacobian Layer
        jl = Jacobian(layers.GRUCell(self.latent_dim))
        js = jl(x, t_length=self.seq_length)

        # QR Decomposition RNN
        qrd_rnn = layers.RNN(QRDcell(dim=self.latent_dim), return_sequences=True)
        rs = qrd_rnn(js)

        # Latent space processing
        z = GRU(self.latent_dim)(x)
        z_mean = Dense(self.latent_dim)(z)
        z_log_sigma = Dense(self.latent_dim)(z)

        # Reparameterization trick
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=0.1)
            return z_mean + K.exp(z_log_sigma) * epsilon

        z_all = Lambda(sampling)([z_mean, z_log_sigma])

        decoded = self._network_prediction(z_mean)

        # Decoder network
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = latent_inputs
        for units, activation in self.decoder_layers:
            x = Dense(units=units, activation=activation)(x)
            x = BatchNormalization()(x)
        Outputs = Dense(self.input_dim, activation='linear')(x)
        decoder = Model(latent_inputs, [Outputs], name='decoder')

        Output = decoder(z_mean)

        # Additional inputs
        y_inputs = layers.Input((self.seq_length, self.input_dim), batch_size=self.batch_size)
        y_input_ = layers.Input(self.input_dim)

        # Encode y_inputs using the encoder model
        y_encoded = TimeDistributed(encoder_model)(y_inputs)

        # Loss functions
        dot_product = K.mean(y_encoded * decoded, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)

        MI_loss = mse(y_input_, Output) * self.loss_weights['MI_loss']
        GSM_loss = K.sigmoid(dot_product) * self.loss_weights['GSM_loss']
        LSM_loss = K.sigmoid(K.sum(tf.math.reduce_mean(tf.math.log(tf.math.abs(tf.linalg.diag_part(rs))), axis=1))) * self.loss_weights['LSM_loss']
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1) * -0.5 * self.loss_weights['kl_loss']

        # Build the model
        training_model = Model([x_inputs, y_inputs, y_input_], [decoded, Output])
        total_loss = MI_loss  + GSM_loss + LSM_loss + kl_loss
        training_model.add_loss(total_loss)
        training_model.compile(optimizer=self.optimizer)

        embedding_transformer = Model(x_inputs, z_mean, name='embedding_transformer')

        return training_model, embedding_transformer, decoder

    def _network_prediction(self, z_mean):
        """
        Predict future steps from the latent space.

        Parameters:
        - z_mean (Tensor): Mean representation in the latent space.

        Returns:
        - Tensor: Predicted output.
        """
        outputs = [Dense(self.latent_dim, activation="linear")(z_mean) for _ in range(self.seq_length)]
        if len(outputs) == 1:
            output =Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
        else:
            output =Lambda(lambda x: K.stack(x, axis=1))(outputs)

        return output
        #return Lambda(lambda x: K.stack(x, axis=1))(outputs)

    def fit(self, X_train, X_data):
        """
        Fit the ISED model on the training data.

        Parameters:
        - X_train (ndarray): Training data, shaped as (n_samples, seq_length, input_dim).
        - X_data (ndarray): Non-subsequenced training data.
        """
        self.model, self.embedding_transformer, self.decoder = self._build_model()
        callbacks = []
        if self.use_early_stopping:
            callbacks.append(EarlyStopping(monitor='loss', patience=10, restore_best_weights=True))

        x_inputs = X_train  # Input for x_inputs
        y_inputs = X_train  # Input for y_inputs (same shape as x_inputs)
        y_input_ = X_data   # Input for y_input_ (different shape)

        # Fit the model
        self.model.fit([x_inputs, y_inputs, y_input_],
                       epochs=self.epochs, batch_size=self.batch_size,
                       shuffle=True, verbose=self.verbose, callbacks=callbacks)

        #self.embedding_transformer.set_weights(self.model.get_layer('embedding_transformer').get_weights())


    def transform(self, X):
        """
        Apply the learned encoder model to transform new data.

        Parameters:
        - X (ndarray): Input data to transform, shaped as (n_samples, seq_length, input_dim).

        Returns:
        - ndarray: Transformed data (embedding dynamics) with learned latent representations.
        """
        #self.model = self._build_model()
        # Use the encoder model to transform the input data
        return self.embedding_transformer.predict(X, verbose=self.verbose)

    def decode(self, X):

        return self.decoder.predict(X, verbose=self.verbose)

    def fit_transform(self, X_train, X_data):
        """
        Fit the model to the training data and transform it in one step.

        Parameters:
        - X_train (ndarray): Training data, shaped as (n_samples, seq_length, input_dim).
        - X_data (ndarray): Non-subsequenced data, same shape as train_data.

        Returns:
        - ndarray: Transformed data after fitting the model.
        """
        self.fit(X_train, X_data)
        return self.transform(X_train)
#################### Code implementation #####################
"""
train_data, test_data, X_normalized = load_and_preprocess_data('../manuscript/code/data/Simulation_data/Xs.pkl')
dominant_period = calculate_dominant_period(X_normalized)
X_train, X_test = prepare_subsequences(X_normalized, int(dominant_period))

# Instantiate the ISED learner with customizable hyperparameters
ised_learner = ISEDLearner(
    latent_dim=20,             # Dimensionality of the latent space
    input_dim=100,             # Dimensionality of the input features
    seq_length=125,            # Length of the input sequences
    batch_size=50,             # Batch size used in training
    epochs=300,                # Number of epochs for training
    use_early_stopping=True    # Whether to use early stopping
)

# Fit the model on training data
ised_learner.fit(X_train,train_data)

# Transform the test data
z_transformed = ised_learner.transform(X_test)

# Alternatively, use fit_transform to fit the model and transform the data in one step
z_transformed_full = ised_learner.fit_transform(X_train, train_data)

# Evaluate smoothness and reconstruction score
smoothness = compute_smoothness(z_transformed)
recon_score = reconstruction_score(z_transformed, test_data)

print(f"Smoothness of the transformed data: {smoothness:.4f}")
print(f"Reconstruction score (R²) on the test data: {recon_score:.4f}")
"""
