from ISED import utils

# Load simulation data file
data_file = './ISED/data/Simulation_data/Xs.pkl'  ## Or your self-defined data path, currently accepting both .pkl and .npy files

dp = utils.DataProcessor(
        data_source=data_file,
        alpha=1,
        method='buffering',
        window_length=None,
    )
X_train, X_test, X_norm = dp.process(
    length_mode='quarter',
    x_norm_type='standard', 
    id_method=None
)

from ISED import ISED_learner

latent_dim = 2

model = ISED_learner.ISEDModel(
    input_dim=X_train.shape[-1],
    seq_length=X_train.shape[1],
    latent_dim=latent_dim,
    batch_size=50,
    epochs=500,    # each .fit() runs this many epochs
    encoder_layers=[(50, 'relu'), (latent_dim, 'relu')],
    rnn_layers=[(20, 'gru'), (latent_dim, 'gru')],
    decoder_layers=[(latent_dim, 'relu'), (30, 'relu')],
    optimizer='adam',
    use_early_stopping=True,
    loss_weights={
        'MI_loss': 1.0,
        'GSM_loss': 1.0,
        'LSM_loss': 1.0,
        'time_loss': 0.0, ### This part loss is not included in manuscript. Can set to 1 for extra embedding performance up, 
    },
    verbose=0
) 
model.fit(X_train[:500],X_norm[:500])

from sklearn import preprocessing
from sklearn.decomposition import PCA

z_y = model.transform(X_test[:352])
#encoded_data = preprocessing.MinMaxScaler().fit_transform(z_y)
#pca = PCA(n_components=2)
encoded_data =  preprocessing.MinMaxScaler().fit_transform(z_y)

print (encoded_data.shape)