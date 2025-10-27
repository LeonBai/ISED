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
    #batch_size = X.shape[0]

    #X = tf.reshape(X, [batch_size*t_length, X.shape[2]])
    shape = tf.shape(X)
    batch_size = shape[0]
    t_length  = t_length or shape[1]
    feature_dim = shape[2]
   
    X = tf.reshape(X, [batch_size * t_length, feature_dim])
    
    
    #X = tf.reshape(X, [-1, 2])
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tape:
      tape.watch(X)
      X_next, _ = self.func(X,[X])

    Jxs = tape.batch_jacobian(X_next, X)

    Jxs = tf.reshape(Jxs, [batch_size, t_length, feature_dim, feature_dim])
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
                 encoder_layers=[(30, 'relu'), (20, 'relu')],
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
        self.loss_weights = loss_weights or {'MI_loss': 1.0, 'GSM_loss': 1.0, 'LSM_loss': 1.0, 'time_loss':1.0}
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
        x_inputs = layers.Input((self.seq_length, self.input_dim))
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
        

        # 2) Measure difference via MSE
        G_x = K.dot(y_input_, K.transpose(y_input_)) # Shape: (batch_size, batch_size)
        G_z = K.dot(z, K.transpose(z)) 
        
        time_loss_raw = mse(G_x, G_z)
        time_loss =K.sigmoid(K.log(tf.reduce_mean(time_loss_raw)))  # scalar

        # 3) Weight it
        time_loss = self.loss_weights['time_loss'] * time_loss

        MI_loss = (K.sigmoid(tf.reduce_mean(mse(y_input_, Output)))) * self.loss_weights['MI_loss']
        GSM_loss = K.sigmoid(dot_product) * self.loss_weights['GSM_loss']
        LSM_loss = K.sigmoid(K.sum(tf.reduce_mean(tf.math.log(tf.math.abs(tf.linalg.diag_part(rs))), axis=1))) * self.loss_weights['LSM_loss']
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sigmoid(K.mean(K.sum(kl_loss, axis=-1) * -0.5)) 

        # Build the model
        training_model = Model([x_inputs, y_inputs, y_input_], [decoded, Output])
        total_loss = MI_loss  + GSM_loss + LSM_loss + kl_loss + time_loss
        training_model.add_loss(total_loss)
        training_model.compile(optimizer=self.optimizer)

        embedding_transformer = Model(x_inputs, z_mean, name='embedding_transformer')

        return training_model, embedding_transformer, decoder, encoder_model

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
        self.model, self.embedding_transformer, self.decoder, self.encoder_model = self._build_model()
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
        #return self.encoder_model.predict(X, verbose = self.verbose)

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

############ Example #########
"""

latent_dim = 35
    
model = ISEDModel(
    input_dim=X_train.shape[2],
    seq_length=X_train.shape[1],
    latent_dim=latent_dim, ## self-defined latent dimension
    batch_size=100,
    epochs=10000,
    encoder_layers=[(30, 'relu'), (latent_dim, 'relu')],
    rnn_layers=[(30, 'gru'), (latent_dim, 'gru')],
    decoder_layers=[(latent_dim, 'relu'), (30, 'relu')],
    optimizer='adam',
    use_early_stopping=True,
    loss_weights={'MI_loss': 1.0, 'GSM_loss': 1.0, 'LSM_loss': 1.0, 'time_loss':1.0},
    verbose = 0   ### tensorflow and keras print-out options: set to 1 or 2 for full results printout during training
)
# Train ISEDModel on train data

#model.fit(np.vstack([X_train[:,:,:], X_test[:,:,:]]),X_norm[:500,:])
model.fit(X_train[:,:,:],X_norm[:500,:])
"""

