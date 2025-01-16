import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import periodogram
from skimage.util import view_as_windows
import skdim

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

    def load_and_preprocess_data(self, method='sliding', latent_dim=None):
        """
        Load, preprocess, and prepare subsequences from data.

        Supports both pickle and numpy file formats.

        Parameters:
        - method (str): Method to use for generating subsequences ('buffering', 'appending', or 'sliding').
        - latent_dim (int): The width of each window, set to the latent dimension.

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

        # Calculate the dominant period to determine the window length (k)
        dominant_period = self.calculate_dominant_period()
        window_length = int(dominant_period) if dominant_period else 20  # Default to 20 if no dominant period is found

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

        # Flatten the appended subsequences back to the original shape
        #X_appended_flattened = X_appended.reshape(n_samples, -1)

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
