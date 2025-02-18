# ISED 

### A official code hub for algorithm smoothness-enhanced embedding learning 
### 1. ISED (current)
### 2. i-Rastermap(Upcoming)

## ISED (Information-based smoothness-enhanced embedding learning model) is a Python library that provides temporal smoothness embedding learning through a unique approach for dimensionality reduction of high-dimensional dynamical systems in order to attain smoothned trajectories.

## Version: 0.1

## Features
- **Smoothed Embedding Learning**: Implement ISED on high-dimensional neural dynamics to learn its smoothed low-dimnensional embeddings.
- **Equipped with adaptable modelling choices**: Multiple options on data-driven/pre-determined $k$/$d$ and structures of embedding and projection functions.
- **Demo Data Available** Simulation data and CA1 cell data from one mouse are available.
- **Integration with Other Embedding Methods**: Works seamlessly with embedding techniques such as UMAP, PHATE, and others.

## Installation

To use ISED, clone the repository and ensure that the required dependencies are installed. Dependencies include NumPy, SciPy, Scikit-learn, and Matplotlib, among others.

```sh
#(Prefered option) pip install
pip install ISED

## Clone this repository
git clone https://github.com/LeonBai/ISED.git

## Install required packages
pip install -r requirements.txt
```


### Requirements
- Python 3.6+
- NumPy >= 1.18.0 & < 2.2.0
- TensorFlow >= 2.0.0
- SciPy >= 1.4.0
- scikit-learn >= 0.22.0
- scikit-dimension >= 0.3.4
- scikit-image >= 0.16.0

- GPy >= 1.9.0 (optional)
- PHATE >= 1.0.0 (optional)
- tphate >= 0.1.0 (optional)


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
data_file = '../data/Simulation_data/Xs.pkl'  ## Or your self-defined data path, currently accepting both .pkl and .npy files

# Preprocess data, normalization, and subsequencing with opted subsequence methods (choose from sliding window, buffering and appending methods)

processor = ISED.DataProcessor(filepath=data_file, alpha=0.5)## alpha = 0, 0.5, 1 === [0,1,2]
x_train, x_test, x = processor.load_and_preprocess_data(method = 'buffering') ## Number of ways to attain subsequences 

# Instanitate ISED model
model = ISED.ISEDModel(
    input_dim=x_train.shape[2],
    seq_length=x_train.shape[1],
    latent_dim=latent_dim, ## self-defined latent dimension
    batch_size=50,
    epochs=300,
    encoder_layers=[(50, 'relu'), (latent_dim, 'relu')],
    rnn_layers=[(30, 'gru'), (latent_dim, 'gru')],
    decoder_layers=[(latent_dim, 'relu'), (30, 'relu')],
    optimizer='adam',
    use_early_stopping=True,
    loss_weight = None, ## ISED-sublearner choice 
    verbose = 0   ### tensorflow and keras print-out options: set to 1 or 2 for full results printout during training
)
# Train ISEDModel on train data

model.fit(x_train,x[:500])

# Attain trajectories on test data
## Keep this identical length to all embedding learning methods
length = 320

z_y = model.transform(x_test[:length])

encoded_data = preprocessing.MinMaxScaler().fit_transform(z_y) ## Learned trajectories

decoded_data =  preprocessing.MinMaxScaler().fit_transform(model.decode(encoded_data)) ## Reconstructed feature dynamics
```


## License
This project is licensed under the MIT License.
see License.md

## PyPI repository
[https://pypi.org/project/ISED/0.1.0/]

## Author(s)
- Wenjun Bai (wjbai@atr.jp) Advanced Telecommunication Research Institute International (ATR)

  
## Contributing
Contributions are welcome! Please submit a pull request or file an issue to help improve ISED.

