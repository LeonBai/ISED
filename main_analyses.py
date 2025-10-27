###########  Analyses that in accordance to TABLE 2. in main manuscript    #########
####### Uploaded v0.2
####### Uploaded v0.3 ++ adding more used analyses
####### Now aligned for all data and systems.


####################################################################################
############## Analyses for all systems: temporal smoothness measures ##############
####################################################################################

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import preprocessing
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tensorflow.keras.utils import to_categorical ## TF version >=2.1.0 < 2.13.0
from sklearn.model_selection import LeaveOneOut

###### Temporal smoothness ######

def compute_smoothness(data):
    diff = np.diff(data, axis=0)
    bending_energy = np.sum(np.linalg.norm(diff, axis=1)**2)  ## conventional 1-degree time difference
    return 1/bending_energy   ### to ensure it is monotonic increasing properity 

####################################################################################
############## Analyses for simulated system: reconstruction accuracy  ##############
####################################################################################

import sklearn.linear_model

def reconstruction_score(x, y):

    def _linear_fitting(x, y):
        lin_model = sklearn.linear_model.LinearRegression()
        #lin_model = SVR(kernel='rbf')
        lin_model.fit(x, y)
        return lin_model.score(x, y)

    return _linear_fitting(x, y)

#with open('simulation/z_true.pkl', 'rb') as file:
    
    #true_z = pickle.load(file)
    
### Usage: 

#print ('Reconstruction error on latent:', reconstruction_score(z_transformed, true_z[0][500:500+320,:]))



####################################################################################
##################    Analyses for mouse hippocampal system          ##############
####################################################################################


###### Intra-mouse behavior interpretability ######

datas = {
    'achilles': achilles_data['position'][:10000,0], 
    'buddy': buddy_data['position'][:6500,0], 
    'cicero': cicero_data['position'][:47000,0],
    'gatsby': gatsby_data['position'][:16000,0]
}

# Set-up your learned embeddings here .npy is prefered 

embeddings = '.npy'

X_train, X_test, y_train, y_test = train_test_split(embeddings[0], datas['achilles'], test_size=0.3, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compute the R^2 score
r2 = r2_score(y_test, y_pred)

r2

###### Behavior-aligned inter-mouse consistency ######
# Step 1: downsampling and matched with behavior construe, from Cebra part

import numpy as np
from scipy.interpolate import interp1d
np.random.seed(0)


behavior_data = [achilles_data['position'][:10000,0], 
    buddy_data['position'][:6500,0], 
    cicero_data['position'][:47000,0],
    gatsby_data['position'][:16000,0]]

results = {}

# Loop over each method in the dictionary
for method, datasets in z.items():
    embeddings = [datasets[dataset_name] for dataset_name in datasets]
    embeddings_masked = []

    # Quantization of data
    for i, j in zip(datas, embeddings):
        quantized_data = []
        for bin_idx in range(1, 100):
            for difference in [0, 1, 2]:
                digitized_labels = np.digitize(i, np.linspace(0, 1.6, 100))
                mask = abs(digitized_labels - bin_idx) <= difference
                quantized_data_ = np.mean(j[mask], axis=0)
            quantized_data.append(quantized_data_)
        quantized_data = np.squeeze(quantized_data)
        embeddings_masked.append(quantized_data)

    # Pairwise linear regression
    sco_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i != j:
                #model = LinearRegression()
                #model = make_pipeline(PolynomialFeatures(2), LinearRegression())
                #model.fit(embeddings_masked[i], embeddings_masked[j])
                #score = model.score(embeddings_masked[i], embeddings_masked[j])
                score,_ =fastdtw(embeddings_masked[i], embeddings_masked[j])
                sco_matrix[i, j] = score
    sco_matrix = 1- preprocessing.MinMaxScaler().fit_transform(sco_matrix)

    # Compute means of non-zero values
    means = []
    for row in sco_matrix:
        non_zero_values = [value for value in row if value != 0]
        mean_value = sum(non_zero_values) / len(non_zero_values) if non_zero_values else 0
        means.append(mean_value)

    results[method] = {
        'sco_matrix': sco_matrix,
        'means': means
    }

print(results)

###### Inter-subject representational similarity ######

# See conventional approach.py for PSID implementation 

###### Inter-subject representational similarity ######

# Step 1: compute brain and behavior inter-subject matricies

def compute_dtw_distance_matrix(data_list):
 
    num_series = len(data_list)
    distance_matrix = np.zeros((num_series, num_series))
    
    for i in range(num_series):
        for j in range(i, num_series):
            distance, _ = fastdtw(data_list[i], data_list[j], dist=euclidean)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric matrix
    
    return distance_matrix

# Compute the DTW distance matrix for the behavior data
dtw_behav_matrix = compute_dtw_distance_matrix(behavior_data)

# Compute the DTW distance matrix for the brain/trajectory data

dtw_emb = compute_dtw_distance_matrix(ised_emb)


print("DTW Distance Matrix:")
print(normalize_matrix(dtw_behav_matrix))

#### Step 2: Mantel test with 10k permutations

def mantel_test(matrix1, matrix2, permutations=10000):
    triu_indices = np.triu_indices_from(matrix1, k=1)
    vec1 = matrix1[triu_indices]
    vec2 = matrix2[triu_indices]

    # Compute the correlation between the two vectors
    mantel_r, _ = pearsonr(vec1, vec2)

    # Permutation test
    permuted_r = np.zeros(permutations)
    for i in range(permutations):
        np.random.shuffle(vec2)
        permuted_r[i], _ = pearsonr(vec1, vec2)
    
    # Calculate p-value
    p_value = np.sum(permuted_r >= mantel_r) / permutations

    return mantel_r, p_value

# Compute the Mantel's test

mantel_r, p_value = mantel_test(dtw_emb,dtw_behav_matrix)


######  Rolling variation  ######

# each embedding is NX3 time series 

#embeddings = 
differences = np.diff(embeddings, axis=0)
euclidean_norms = np.linalg.norm(differences, axis=1)

# Convert to a pandas Series for easy rolling computation
euclidean_series = pd.Series(euclidean_norms)
window_size = 500

# Compute rolling variation 
rolling_euclidean = euclidean_series.rolling(window=window_size).mean()

###### Recurrence ######

def create_recurrence_plot(distance_matrix, threshold):
    """
    Create a recurrence plot given a distance matrix and a threshold.
    """
    recurrence_plot = distance_matrix < threshold
    return recurrence_plot

def calculate_recurrence_rate(recurrence_plot):
    """
    Calculate the recurrence rate (RR) for the recurrence plot.
    RR is the fraction of points in the plot that are recurrences.
    """
    total_points = recurrence_plot.size
    recurrences = np.sum(recurrence_plot)
    return recurrences / total_points

rr = calculate_recurrence_rate(recurrence_plot)

print("Recurrence Rate (RR):", rr)

def calculate_determinism(recurrence_plot, min_line_length=3):
    """
    Calculate the determinism (DET) of a recurrence plot.
    DET is the proportion of recurrent points that form diagonal lines of at least a certain length.
    """
    num_points = recurrence_plot.shape[0]
    diagonal_counts = 0
    total_diagonal_points = 0

    # Iterate over all possible diagonals (excluding the main diagonal)
    for k in range(-num_points + 1, num_points):
        diagonal = np.diag(recurrence_plot, k=k)

        if len(diagonal) >= min_line_length:
            line_lengths = np.diff(np.where(np.concatenate(([0], diagonal, [0])))[0]) - 1
            valid_lines = line_lengths[line_lengths >= min_line_length]
            total_diagonal_points += np.sum(valid_lines)
            diagonal_counts += len(valid_lines)

    # DET is the total number of points in diagonal lines divided by total recurrent points
    total_recurrent_points = np.sum(recurrence_plot)
    det = total_diagonal_points / total_recurrent_points if total_recurrent_points != 0 else 0

    return det

det = calculate_determinism(recurrence_plot)
print("Determinism (DET):", det)


####################################################################################
##################    Analyses for Human cortical system          ##############
####################################################################################


###### Inter-subject consistency ######

# assume all embeddings/original data are subject-wise stored in emb variable

emb = '.npy' ### Subject x Embedding matrix, in our case 119 x Embedding

dtw_matrix = np.zeros((119,119))

#matrix = np.concatenate([c_align[:46], a_align[:46]], axis = 0)
                       
for i in range(119):
        for j in range(119):
            if i != j:
                distance,_= fastdtw(emb[i]**1, emb[j]**1)
                dtw_matrix[i, j] = (distance)
                dtw_matrix[j, i] = (distance)
dtw_matrix = 1 - preprocessing.MinMaxScaler().fit_transform(dtw_matrix)

means = []
for row in dtw_matrix:
    non_zero_values = [value for value in row if value != 0]
    mean_value = sum(non_zero_values) / len(non_zero_values) if non_zero_values else 0
    means.append(mean_value)

###### Phenotype decoding ######

# Assume emd stores all learned embedding/original data
emb = '.npy' ### Subject x Embedding matrix, in our case 119 x Embedding

x = np.asarray(emb)[:,:250,:]

##### This is for original feature space, we randomly select three features. 

indices = np.random.choice(116, 3, replace=False)  
x = ori[:,:250,indices]


y = np.concatenate([np.zeros(73), np.ones(46)])
X_flat = x.reshape(x.shape[0], -1)  
loo = LeaveOneOut()
lr_accuracies = []

for train_index, test_index in loo.split(X_flat):
    X_train, X_test = X_flat[train_index], X_flat[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    lr_model = LogisticRegression(max_iter=400)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    lr_accuracies.append(accuracy_score(y_test, y_pred))

print("Average LOOCV Accuracy for Logistic Regression:", np.mean(lr_accuracies))

