import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import periodogram
from skimage.util import view_as_windows
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import skdim

import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

class DataProcessor:
    def __init__(self, data_source, alpha=0,
                 method='sliding',
                 window_length=None,
                 verbose = True):
        """
        data_source: np.ndarray or path to .pkl/.npy file
        alpha: float, used when loading from .pkl
        method: 'buffering', 'appending', or 'sliding'
        window_length: manually specify window length
        """
        self.data_source = data_source
        self.alpha = alpha
        self.method = method
        self.user_length = window_length
        self.X_normalized = None
        self.X_train = None
        self.X_test = None
        self.verbose = verbose
  
        # New attributes for length estimation
        self.length_range = None
        self.optimal_midpoint = None
        self.optimal_quarter = None
        self.chosen_length = None
        self.x_norm_type = None
        # intrinsic dimension
        self.id_method = None
        self.intrinsic_dim = None

    def load_data(self):
        """Load time-series data from various file types into a numpy array."""
        # Raw numpy array
        if isinstance(self.data_source, np.ndarray):
            return self.data_source
        # File paths
        if isinstance(self.data_source, str):
            if self.data_source.endswith('.pkl'):
                with open(self.data_source, 'rb') as f:
                    raw = pickle.load(f)
                return raw[int(self.alpha)]
            if self.data_source.endswith('.npy'):
                return np.load(self.data_source)
            if self.data_source.endswith(('.h5', '.hdf5')):
                # assume single dataset or first dataset in file
                with h5py.File(self.data_source, 'r') as f:
                    # pick the first dataset found
                    key = next(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
                    data = f[key][()]
                return np.array(data)
        raise ValueError("data_source must be a numpy array or a .pkl/.npy/.h5/.hdf5 file path.")

    def normalize(self):
        data = self.load_data()
        self.X_normalized = MinMaxScaler().fit_transform(data)
        return self.X_normalized

    def estimate_tau(self, x, max_lag=1000, thresh=1/np.e):
        xz = x - x.mean(); var = np.var(xz); N = len(xz)
        for lag in range(1, max_lag+1):
            R = np.dot(xz[:N-lag], xz[lag:]) / ((N-lag)*var)
            if R <= thresh:
                return lag
        return max_lag

    def false_nearest_fraction(self, x, tau, m, R_tol=10.0):
        N = len(x); K = N - m*tau
        if K <= 1:
            return 1.0
        X_m = np.column_stack([x[i*tau:i*tau+K] for i in range(m)])
        X_m1 = np.column_stack([x[i*tau:i*tau+K] for i in range(m+1)])
        nbrs = NearestNeighbors(n_neighbors=2).fit(X_m)
        dist, idx = nbrs.kneighbors(X_m)
        dm, nn = dist[:,1], idx[:,1]
        dm1 = np.linalg.norm(X_m1 - X_m1[nn], axis=1)
        valid = dm > 1e-8
        return np.sum(dm1[valid]/dm[valid] > R_tol) / np.sum(valid)

    def determine_optimal_range(self):
        """Estimate the min and max window lengths based on embedding geometry."""
        X = self.X_normalized
        pc1 = PCA(n_components=1).fit_transform(X).ravel()
        tau = self.estimate_tau(pc1)
        m = next((m for m in range(1, 101)
                  if self.false_nearest_fraction(pc1, tau, m) < 0.05), 100)
        L_min = (m-1)*tau + 1
        lag0 = self.estimate_tau(pc1, max_lag=100, thresh=0.05)
        L_max = min((m-1)*lag0 + 1, len(X)//2)
        self.length_range = (L_min, L_max)
        self.optimal_midpoint = (L_min + L_max) // 2

    def compute_quarter_period(self):
        """Compute dominant period and return one-quarter of it as integer."""
        X = self.X_normalized
        wavelengths = []
        power_values = []
        for i in range(X.shape[1]):
            f, Pxx = periodogram(X[:, i], fs=1)
            idx = np.argmax(Pxx)
            if f[idx] > 0:
                wavelengths.append(1 / f[idx])
                power_values.append(Pxx[idx])
        if not wavelengths:
            raise RuntimeError("Unable to compute dominant period.")
        dominant_period = wavelengths[np.argmax(power_values)]
        self.optimal_quarter = int(np.round(dominant_period * 0.25))

    def generate_windows(self, L):
        X = self.X_normalized
        if self.method == 'buffering':
            circ = np.vstack([X, X[:L-1]])
            win = np.squeeze(view_as_windows(circ, (L, X.shape[1]), step=1))[:len(X)]
            mid = len(win)//2
            return win[:mid], win[mid:]
        if self.method == 'appending':
            n, f = X.shape
            app = np.zeros((n, L, f))
            for i in range(n):
                seg = X[max(0,i-L+1):i+1]
                if seg.shape[0] < L:
                    seg = np.pad(seg, ((L-seg.shape[0],0),(0,0)))
                app[i] = seg
            mid = n//2
            return app[:mid], app[mid:]
        if self.method == 'sliding':
            d =  X.shape[1]
            win = np.squeeze(view_as_windows(X, (L, d)))
            mid = win.shape[0]//2
            return win[:mid], win[mid:]
        raise ValueError(f"Unknown method: {self.method}")
        
    def estimate_intrinsic_dimension(self, data, method='two_nn'):
        """Estimate intrinsic dimension of `data` using skdim.id methods."""
        from skdim.id import TwoNN, MLE
        self.id_method = method
        if method == 'two_nn':
            est = TwoNN()
        elif method == 'mle':
            est = MLE()
        else:
            raise ValueError("method must be 'two_nn' or 'mle'.")
        ids = est.fit_transform(data)
        # global estimate: mean of local dims
        self.intrinsic_dim = float(np.mean(ids))
        return self.intrinsic_dim

    def process(self, length_mode='midpoint', x_norm_type='minmax', id_method=None):
        """
        Full pipeline: normalize, determine/gather length, generate windows, normalize
        and estimate intrinsic dimension.

        length_mode: 'midpoint', 'quarter', or 'both'
        x_norm_type: 'minmax', 'standard', or 'raw'
        id_method: 'two_nn' or 'mle'
        """
        # normalize and length estimates
        self.normalize()
        if self.user_length is not None:
            self.optimal_midpoint = self.user_length
            self.length_range = (self.user_length, self.user_length)
            self.optimal_quarter = self.user_length
        else:
            self.determine_optimal_range()
            self.compute_quarter_period()

        if length_mode == 'midpoint':
            L = self.optimal_midpoint
        elif length_mode == 'quarter':
            L = self.optimal_quarter
        elif length_mode == 'information':
            return (self.length_range,
                    self.optimal_midpoint,
                    self.optimal_quarter)
        else:
            raise ValueError("length_mode must be 'midpoint', 'quarter', or 'information'.")
        self.chosen_length = L
        self.X_train, self.X_test = self.generate_windows(L)

        # build X_norm per choice
        self.x_norm_type = x_norm_type
        if x_norm_type == 'minmax':
            scaler = MinMaxScaler()
        elif x_norm_type == 'standard':
            scaler = StandardScaler()
        elif x_norm_type == 'raw':
            self.X_norm = self.X_normalized
        else:
            raise ValueError("x_norm_type must be 'minmax', 'standard', or 'raw'.")

        if x_norm_type != 'raw':
            stacked = np.vstack([
                np.mean(self.X_train, axis=1),
                np.mean(self.X_test, axis=1)
            ])
            self.X_norm = scaler.fit_transform(stacked)

        # optionally estimate intrinsic dimension
        if id_method is not None:
            self.estimate_intrinsic_dimension(self.X_norm, method=id_method)
        else:
            self.id_method = None
            self.intrinsic_dim = None

        if self.verbose:
            self._print_summary()
        return self.X_train, self.X_test, self.X_norm
    
    def _print_summary(self):
        """Nicely print shapes and chosen parameters."""
        print("=== DataProcessor Summary ===")
        print(f"Subsequence Method: {self.method}")
        print(f"Chosen window length: {self.chosen_length}")
        print(f"Length range: {self.length_range}")
        print(f"Optimal midpoint: {self.optimal_midpoint}, quarter: {self.optimal_quarter}")
        print(f"X_norm type: {self.x_norm_type}")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"X_norm shape: {self.X_norm.shape}")
        if self.id_method:
            print(f"Intrinsic dimension ({self.id_method}): {self.intrinsic_dim:.2f}")
        else:
            print("Intrinsic dimension: skipped")
        print("==============================")





def post_hoc_smoothing(data, filter_type='gaussian', window_size=11, polynomial_order=6, sigma=3):
    """
    Applies post-hoc smoothing to a multivariate time series.

    Args:
        data (pd.DataFrame or np.ndarray): The multivariate time series data,
                                            with time as rows and variables as columns.
        filter_type (str, optional): The type of filter to use.
                                     Options: 'moving_average', 'gaussian', 'savitzky_golay'.
                                     Defaults to 'gaussian'.
        window_size (int, optional): The size of the window for the moving average and
                                   Savitzky-Golay filters. Must be an odd integer.
                                   Defaults to 11.
        polynomial_order (int, optional): The order of the polynomial for the
                                          Savitzky-Golay filter. Must be less than window_size.
                                          Defaults to 3.
        sigma (float, optional): The standard deviation for the Gaussian filter.
                                 Defaults to 3.

    Returns:
        pd.DataFrame or np.ndarray: The smoothed multivariate time series,
                                    in the same format as the input data.
    """
    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise TypeError("Input data must be a pandas DataFrame or a NumPy array.")

    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer.")

    is_pandas = isinstance(data, pd.DataFrame)
    if is_pandas:
        original_index = data.index
        original_columns = data.columns
        data_np = data.values
    else:
        data_np = data

    smoothed_data = np.zeros_like(data_np)

    for i in range(data_np.shape[1]):
        if filter_type == 'moving_average':
            # Using pandas rolling mean for simplicity and to handle edges well
            series = pd.Series(data_np[:, i])
            smoothed_data[:, i] = series.rolling(window=window_size, center=True, min_periods=1).mean()
        elif filter_type == 'gaussian':
            smoothed_data[:, i] = gaussian_filter1d(data_np[:, i], sigma=sigma)
        elif filter_type == 'savitzky_golay':
            if polynomial_order >= window_size:
                raise ValueError("polynomial_order must be less than window_size.")
            smoothed_data[:, i] = savgol_filter(data_np[:, i], window_length=window_size, polyorder=polynomial_order)
        else:
            raise ValueError("Invalid filter_type. Choose from 'moving_average', 'gaussian', or 'savitzky_golay'.")

    if is_pandas:
        return pd.DataFrame(smoothed_data, index=original_index, columns=original_columns)
    else:
        return smoothed_data


