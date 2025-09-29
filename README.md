# Smoothness-Enhanced Embedding Learning  

------------------------------------------------------------------------------------
## A official code hub for algorithm smoothness-enhanced embedding learning 
### 1. Intrinsic Smoothness-enhanced Embedding Learning (ISED)
### 2. Structure-aware neural analysis {Viz/E/D} [TBA]
------------------------------------------------------------------------------------

## [1] ISED

### Short Description: ISED (Intrinsic Smoothness Embedding Dynamics Learner) is a Python library that provides temporal smoothness embedding learning through a unique approach for dimensionality reduction of high-dimensional dynamical systems in order to attain smoothned trajectories. It caters various of neural datasets: electrophysiological/MEG/EEG/MRI/calicum imaging data. 

### Version: 0.2
### Updates (0.1 -> 0.2)
+ Nicer print-out option for data preprocessing to determine subsequence length k and latent dimensionality d
+ More subsequence methods {buffering; sliding windoer, appending} with length determination k {self-defined or data-driven} available in data preprocessing in util.py
+ A new analysis + plotting script uploaded to replicate some of the key result plots in Fig.4 of main text. (see Analysis folder here)
+ Add post-hoc smoothing operation in util.py
+ Add plot.py for plotting the trajectory 
- Remove the alternative approach from pip installable ISED package for less library-wise conflicts
- Remove the main_analysis from pip installable ISED package for less library-wise conflicts
  

## Features
- **Smoothed Embedding Learning**: Implement ISED on high-dimensional neural dynamics to learn its smoothed low-dimnensional embeddings.
- **Equipped with adaptable modelling choices**: Multiple options on data-driven/pre-determined $k$/$d$ and structures of embedding and projection functions.
- **Demo Data Available** Simulation data and CA1 cell data from one mouse are available.

## Installation

To use ISED, clone the repository and ensure that the required dependencies are installed. Dependencies include NumPy, SciPy, Scikit-learn, and Matplotlib, among others.

```sh
#(Prefered option) pip install
pip install ISED

# Manual installation 
## Clone this repository
git clone https://github.com/LeonBai/ISED.git

## Install required packages
pip install -r requirements.txt

## Or using Conda
conda env create -f environment.yml
```


### Requirements
- Python 3.6+
- NumPy >= 1.25 < 2.0
- TensorFlow >= 2.0.0 <=2.13.0 ## This is essential for not getting .shape issue in later model training 
- SciPy >= 1.4.0
- scikit-learn >= 0.22.0
- scikit-dimension >= 0.3.4
- scikit-image >= 0.16.0

- GPy >= 1.9.0 (optional)
- PHATE >= 1.0.0 (optional)
- tphate >= 0.1.0 (optional)


## Notebook (`Test_run.ipynb`)

The notebook `Test_running_synthetic.ipynb` provides a structured example to test the ISED model workflow:
- **Loading Data**: Simulation data is loaded, and a preprocessing method is applied.
- **Determine Latent Dimension**: The latent dimension (`latent_dim`) is set for the analysis.
- **Training and Evaluation**: Train the `ISED_learner.ISEDModel` and analyze the temporal smoothness of embedding dynamics.


## Usage

Below is a simple code snippet demonstrating how to use the ISED model with pre-processing data processor method:

```python
import numpy as np
from ISED import utils
from ISED import ISED_learner
from sklearn import preprocessing

# Load simulation data file
data_file = '../data/Simulation_data/Xs.pkl'  ## Or your self-defined data path, currently accepting both .pkl and .npy files

# Preprocess data, normalization, and subsequencing with opted subsequence methods (choose from sliding window, buffering and appending methods)
## New in v0.2, now we can 
dp = utils.DataProcessor(
        data_source=data_file,
        alpha=1,
        method='buffering',
        window_length=None
    )
X_train, X_test, X_norm = dp.process(
    length_mode='quarter',
    x_norm_type='raw',
    id_method=None
)

# Instanitate ISED model
from ISED import ISED_learner

latent_dim = 10

model = ISED_learner.ISEDModel(
    input_dim=X_train.shape[-1],
    seq_length=X_train.shape[1],
    latent_dim=latent_dim,
    batch_size=50,
    epochs=300,    
    encoder_layers=[(50, 'relu'), (latent_dim, 'relu')], ## Embedding function 'f_emb' in main text
    rnn_layers=[(20, 'gru'), (latent_dim, 'gru')],       ## Projection funtion  'f_pro' in main text
    decoder_layers=[(latent_dim, 'relu'), (30, 'relu')], ## Decoding function  'g' in main text
    optimizer='adam',
    use_early_stopping=True,
    loss_weights={
        'MI_loss': 1.0,    #
        'GSM_loss': 1.0,   #  Set = 0 if you donot need global smoothness loss
        'LSM_loss': 1.0,   #  Set = 0 if you donot need local smoothness loss 
     },
    verbose=2
) 

# Train ISEDModel on train data

model.fit(x_train,x[:500])

# Attain trajectories on test data
## Keep this identical length to all embedding learning methods
length = 320

z_y = model.transform_2(x_test[:length])

encoded_data = preprocessing.MinMaxScaler().fit_transform(z_y) ## Learned trajectories

```


## License
This project is licensed under the MIT License.
see License.md

## PyPI repository
[https://pypi.org/project/ISED/0.2.0/]

## Author(s)
- Wenjun Bai (wjbai@atr.jp) Advanced Telecommunication Research Institute International (ATR)

  
## Contributing
Contributions are welcome! Please submit a pull request or file an issue to help improve ISED.

