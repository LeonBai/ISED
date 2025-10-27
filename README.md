# ISED: Intrinsic Smooth Embedding Learning Framework 

ISED is a Python library that provides temporal smoothness embedding learning through a unique approach for dimensionality reduction of high-dimensional dynamical systems in order to attain smoothned trajectories.

## Version update

### v0.1.0 --> 0.2.0 

## Features

- **Preprocess the high-dimensional time series for subsequences**: Includes the attainment of optimal subsequence length and dimensionality, we offer buffering/appending/sliding window approaches to make subsequences.
- **Embedding learning (ISED core)**: Learn and transform unseen data into low-dimensional smooth trajectories. It requires both aligned time series and subsequences.
- **(Optional) Post analysis**: We offer several optional post analyses ranging from the estimation of the temporal smoothness of learned embeddings to behvaior analysis for real neural datasets. 


## Installation

To use ISED, you can either clone the repository or simply running $pip install ISED$ and ensure that the required dependencies are installed. Dependencies include NumPy, SciPy, Scikit-learn, and Matplotlib, among others.

```sh
# Create a virtual env (currently on python 3.9) for running ISED code
conda create -n ISED_running python=3.9

# (Recommanded Option) Pypi install, note if you have old version installed (v0.1) please uninstall and reinstall v=0.2 
pip install ISED 

# Or clone this repository
git clone https://github.com/LeonBai/ISED_learner.git

# Then Install required packages
pip install -r requirements.txt
```


### Requirements
- Python 3.6+
- NumPy >= 1.18.0
- TensorFlow >= 2.0.0
- SciPy >= 1.4.0
- scikit-learn >= 0.22.0
- scikit-dimension >= 0.3.4
- scikit-image >= 0.16.0
- GPy >= 1.9.0


## Notebook (`Test_run.ipynb`)

The notebook `Test_run.ipynb` provides a structured example to test the ISED model workflow:
- **Loading Functions**: Import all relevant functions from the `ISED.py` script.
- **Loading Data**: Simulation data is loaded, and a preprocessing method is applied.
- **Determine Latent Dimension**: The latent dimension (`latent_dim`) is set for the analysis.
- **Training and Evaluation**: Train the `ISEDModel` and analyze the embedding and decoded dynamics.


## Usage

Below is a simple code snippet demonstrating how to use the ISED model with pre-processing method:

```python
from ISED_learner import ISED
from sklearn import preprocessing

# Load simulation data file
data_file = './ISED/data/Simulation_data/Xs.pkl'  ## Or your self-defined data path, currently accepting both .pkl and .npy files

# Preprocess data, normalization, and subsequencing with opted subsequence methods (choose from sliding window, buffering and appending methods)
## New in 0.2.0 version 

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

# Train ISEDModel on train data

model.fit(X_train[:500],X_norm[:500])


# Attain trajectories on test data
length = 320

from sklearn import preprocessing
from sklearn.decomposition import PCA

z_y = model.transform(X_test[:352])
encoded_data =  preprocessing.MinMaxScaler().fit_transform(z_y)
```

## Additonal Analyses 

Additional analyses for learned embeddings can be found in (https://github.com/LeonBai/ISED/tree/main).

## License
This project is licensed under the MIT License.

## Author
- Wenjun Bai (wjbai@atr.jp)

For more details, please visit the [GitHub Repository](https://github.com/LeonBai/ISED).

## Contributing
Contributions are welcome! Please submit a pull request or file an issue to help improve ISED.
