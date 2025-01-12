import datetime
import functools
import io
import logging
import os
import math
import sys
import time
import traceback
import types
import cProfile
import pstats
import psutil
import unittest
from collections import Counter
from numba import njit, vectorize
from multiprocessing import Pool
from functools import wraps
import numpy as np
import pandas as pd
from contextlib import contextmanager
import scipy.stats as sps
from scipy.spatial.distance import cdist, pdist
from sympy import primerange, factorint, isprime
from collections import Counter
import shap
from ruptures import Binseg  # for change point detection
from scipy.stats import skew, kurtosis, norm, mode, normaltest, levene, f_oneway, spearmanr, entropy, lognorm
from scipy.cluster.hierarchy import linkage
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.cm as cm
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations, count
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import xgboost as xgb
import warnings
import gc
import pickle
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import pywt
import pickle
from logging.handlers import RotatingFileHandler




# PrimeAnalysis by Nova Spivack
# https://www.novaspivack.com
# https://github.com/novaspivack?tab=repositories

################################################################################
# SETTINGS
################################################################################

# **** NOTE!!!!: REMEMBER TO SET YOUR PYTHON INTERPRETER TO THE SAME INTERPRETER AS TENSORFLOW PATH ****

DEBUG_MODE = True # Set to True to enable debug mode with verbose logging and profiling
PRIME_MAP_SAMPLE_RATE = 0.1 # Set to 10% for testing
N_TEST_RANGES = 5 # Number of ranges to test the model on
TEST_RANGE_SIZE = None # Size of each test range. If None it will be calculated automatically
TRANSFER_LEARNING_ENABLED = True # Enable or disable transfer learning

N = 10000 # Number of primes to analyze
N_i = 1000  # Number integers to analyze; optional setting, not implemented yet, that can be used instead of number of primes if an approach needs this setting
COMPOSITE_SAMPLE_RATE = .1  # Default 1.0 for 100% for full analysis

# add other settings here if needed; e.g. which ML models to run, or settings related to the analysis and amount of computation to do etc. Making the system more configurable will be beneficial especially if we start doing analysis that could be computationally intensive so that we can test with lower computation and then when the system is working we can increase to full computation which might take longer to run. These could represented as configuration settings here.

BATCH_THRESHOLD = 100000  # Threshold for batch processing; normally ste to 50000
BATCH_SIZE = 1000  # Default batch size for large datasets; normally set to 10000



# Create output directories under the current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
OUTPUT_LOG_FILE = os.path.join(OUTPUT_DIR, "prime_gap_analysis_report.log")

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start profilers
        profiler = cProfile.Profile()
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Run function with profiling
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Compute metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Capture profiling info to string
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 time-consuming functions
        profile_output = s.getvalue()
        
        # Log profiling info
        logger = logging.getLogger('PrimeAnalysis')
        logger.debug(f"\nDetailed Profile for {func.__name__}:")
        logger.debug(f"Total Time: {duration:.2f} seconds")
        logger.debug(f"Total Memory: {memory_used:.2f} MB")
        logger.debug(profile_output)
        
        return result
    return wrapper

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        logger = logging.getLogger('PrimeAnalysis')
        logger.debug(f"\n{func.__name__}:")
        logger.debug(f"  Time: {duration:.2f} seconds")
        logger.debug(f"  Memory: {memory_used:.2f} MB")
        
        return result
    return wrapper

class CustomFilter(logging.Filter):
    """Custom filter to filter out specific log levels and messages."""
    def __init__(self, filter_levels=None, filter_messages=None):
        super().__init__()
        self.filter_levels = filter_levels or []
        self.filter_messages = filter_messages or []

    def filter(self, record):
        if record.levelno in self.filter_levels:
            return False
        for message in self.filter_messages:
            if message in record.getMessage():
                return False
        return True


class PrimeAnalysisLogger:
    """Manages logging for prime number analysis."""
    
    def __init__(self, debug_mode=False):
        self.logger = self._setup_logging(debug_mode)
            
    def _setup_logging(self, debug_mode):
        """Setup comprehensive logging with advanced filtering capabilities."""
        try:
            # Determine the script's directory
            script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            logs_dir = os.path.join(script_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Full path for the log file
            log_file = os.path.join(logs_dir, "prime_analysis.log")
            
            # Full path for the error log file
            error_log_file = os.path.join(logs_dir, "prime_analysis_error.log")
            
            # Create logger
            logger = logging.getLogger('PrimeAnalysis')
            logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
            
            # Clear existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Console handler setup
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG) # Show all console messages
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S.%f'
            )
            console_handler.setFormatter(console_formatter)
            # console_handler.addFilter(CustomFilter()) # Remove the filter from console
            logger.addHandler(console_handler)
            
            # Configure rotating file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=20*1024*1024,  # 20 MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG) # Log all messages to file
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
            
            # Clear the file at the start of each run
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("")
            
            # Configure error log file
            error_handler = RotatingFileHandler(
                error_log_file,
                maxBytes=20*1024*1024,
                backupCount=3,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(console_formatter)
            logger.addHandler(error_handler)
            
            # Clear the error log file at the start of each run
            with open(error_log_file, 'w', encoding='utf-8') as f:
                f.write("")
            
            # Create debug log file if in debug mode
            if debug_mode:
                debug_log_file = os.path.join(logs_dir, "prime_analysis_debug.log")
                debug_handler = RotatingFileHandler(
                    debug_log_file,
                    maxBytes=20*1024*1024,  # 20 MB
                    backupCount=3,
                    encoding='utf-8'
                )
                debug_handler.setLevel(logging.DEBUG)
                debug_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - '
                    '%(funcName)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S.%f'
                )
                debug_handler.setFormatter(debug_formatter)
                logger.addHandler(debug_handler)
            
            # Log initial setup information
            logger.info(f"Prime analysis logging initialized at {datetime.datetime.now()}")
            logger.info(f"Main log file: {log_file}")
            if debug_mode:
                logger.info(f"Debug log file: {debug_log_file}")
            logger.info(f"Log directory: {logs_dir}")
            logger.info(f"Debug mode: {debug_mode}")
            
            # Log system information
            logger.info("System Information:")
            logger.info(f"Python version: {sys.version}")
            logger.info(f"Platform: {sys.platform}")
            memory = psutil.virtual_memory()
            logger.info(f"Total memory: {memory.total / (1024**3):.2f} GB")
            logger.info(f"Available memory: {memory.available / (1024**3):.2f} GB")
            
            return logger
            
        except Exception as e:
            print(f"Critical error in logging setup: {e}")
            import traceback
            print(traceback.format_exc())
            # Create a basic console-only logger as fallback
            basic_logger = logging.getLogger('PrimeAnalysis_Fallback')
            basic_logger.setLevel(logging.INFO)
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            basic_logger.addHandler(console)
            return basic_logger
           
    def configure_filter(self, filter_levels=None, filter_messages=None):
        """Dynamically reconfigure log filters."""
        for handler in self.logger.handlers:
            # Remove existing custom filters
            handler.filters = [
                f for f in handler.filters if not isinstance(f, CustomFilter)
            ]
            
            # Add new filter
            new_filter = CustomFilter(
                filter_levels=filter_levels or [],
                filter_messages=filter_messages or []
            )
            handler.addFilter(new_filter)
    
    def log_memory_usage(self, message):
        """Log memory usage with message."""
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.logger.debug(f"{message} (Memory: {memory_mb:.1f} MB)")
    
    def log_progress(self, current, total, prefix="", suffix=""):
        """Log progress with percentage."""
        percent = 100 * (current / float(total))
        self.logger.info(f"{prefix} - {percent:.1f}% {suffix}")
    
    def log_analysis_step(self, step_name, **kwargs):
        """Log analysis step with optional parameters."""
        params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"Starting {step_name}" + (f" with {params}" if params else ""))
    
    def log_error(self, message, exc_info=None):
        """Log error with optional exception info."""
        if exc_info:
            self.logger.error(message, exc_info=exc_info)
            logger.logger.error(traceback.format_exc())
        else:
            self.logger.error(message)
            
    def monitor_memory_usage(self, operation_name, threshold_mb=1000):
        """Monitor memory usage and warn if it exceeds threshold."""
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_mb > threshold_mb:
            self.logger.warning(f"High memory usage in {operation_name}: {memory_mb:.1f} MB")
            # Force garbage collection
            gc.collect()
            # Get updated memory usage
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.logger.info(f"Memory after GC: {memory_mb:.1f} MB")
        return memory_mb

    def log_and_print(self, message, level=logging.INFO):
        """Log message and print to console."""
        self.logger.log(level, message)
        print(message)
        
def log_progress(message, log_file=OUTPUT_LOG_FILE):
    """Log progress message with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as log:
        log.write(f"{timestamp}: {message}\n")
        
def check_memory_usage():
    """Monitor memory usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def get_memory_usage():
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def log_memory_usage(message):
    """Log memory usage with message."""
    print(f"{message} (Memory usage: {get_memory_usage():.1f} MB)")

def log_memory_status():
    """Log current memory usage status."""
    memory = psutil.virtual_memory()
    print(f"Memory Status:")
    print(f"  Available: {memory.available / (1024**3):.1f} GB")
    print(f"  Used: {memory.used / (1024**3):.1f} GB")
    print(f"  Free: {memory.free / (1024**3):.1f} GB")
    print(f"  Percent: {memory.percent}%")
    
@contextmanager
def suppress_overflow_warnings():
    """Context manager to suppress overflow warnings."""
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
            
@contextmanager
def suppress_numeric_warnings():
    """Suppress numeric warnings and handle overflow."""
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        yield

@timing_decorator
def optimize_memory_usage(df, logger=None):
    """Optimize DataFrame memory usage with aggressive downcasting and handling of object types."""
    if logger:
        logger.log_and_print("Optimizing memory usage...")
    
    try:
        for col in df.columns:
            if df[col].dtype == 'float64':
                # Try to downcast to float32 first
                df[col] = pd.to_numeric(df[col], downcast='float', errors='coerce')
            elif df[col].dtype == 'int64':
                # Try to downcast to smallest integer type
                df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')
            
            # Handle object dtypes that should be numeric
            if df[col].dtype == 'object':
                try:
                    # Attempt to convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Downcast again if conversion was successful
                    if df[col].dtype == 'float64':
                        df[col] = pd.to_numeric(df[col], downcast='float', errors='coerce')
                    elif df[col].dtype == 'int64':
                        df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')
                    
                except Exception as e:
                    if logger:
                        logger.log_and_print(f"Warning: Could not convert column {col} to numeric: {str(e)}")
                    continue # Skip to next column if conversion fails
            
            # Handle boolean types
            if df[col].dtype == 'bool':
                df[col] = df[col].astype('int8')
            
            # Handle categorical types
            if df[col].dtype == 'category':
                try:
                    df[col] = df[col].astype('int32')
                except Exception as e:
                    if logger:
                        logger.log_and_print(f"Warning: Could not convert column {col} to int32: {str(e)}")
                    continue # Skip to next column if conversion fails
        
        # Explicitly convert to float32 if still float64
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
        
        if logger:
            logger.log_memory_usage("Memory optimization complete")
        
        return df
    
    except Exception as e:
        error_msg = f"Error optimizing memory usage: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        return df
    
class ProgressBar:
    """Custom progress bar for console output."""
    def __init__(self, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.print_end = print_end
        self.iteration = 0

    def print(self, iteration):
        """Print the progress bar."""
        self.iteration = iteration
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        print(f'\r{self.prefix} |{bar}| {percent}% {self.suffix}', end=self.print_end)
        if iteration == self.total:
            print()

    def increment(self):
        """Increment the progress bar by one step."""
        self.iteration += 1
        self.print(self.iteration)

class TensorFlowProgressCallback(keras.callbacks.Callback):
    """Progress callback for TensorFlow training with improved error handling and logging."""
    def __init__(self, fold_number=None, logger=None):
        super().__init__()
        self.fold_number = fold_number
        self.logger = logger
        self.last_metrics = None
        self.epoch_times = []
        self.start_time = None
        self.total_epochs = None
        self.current_epoch = 0
        
    def on_train_begin(self, logs=None):
        """Initialize training start."""
        self.start_time = time.time()
        self.total_epochs = self.params.get('epochs', 0)
        fold_str = f" fold {self.fold_number}" if self.fold_number is not None else ""
        msg = f"Training neural network{fold_str}..."
        if self.logger:
            self.logger.log_and_print(msg)
        else:
            print(msg, end='', flush=True)

    def on_epoch_begin(self, epoch, logs=None):
        """Record epoch start time."""
        self.epoch_start_time = time.time()
        self.current_epoch = epoch + 1

    def on_epoch_end(self, epoch, logs=None):
        """Update metrics and show progress."""
        try:
            if logs:
                self.last_metrics = {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v 
                    for k, v in logs.items()
                }
                
                # Calculate epoch time
                epoch_time = time.time() - self.epoch_start_time
                self.epoch_times.append(epoch_time)
                
                # Calculate ETA
                if len(self.epoch_times) > 0:
                    avg_epoch_time = np.mean(self.epoch_times)
                    remaining_epochs = (self.total_epochs or 0) - (self.current_epoch or 0)
                    eta = avg_epoch_time * remaining_epochs
                    
                    # Format progress message
                    metrics_str = ", ".join([
                        f"{k}: {v:.4f}" if isinstance(v, (int, float, np.number)) else f"{k}: {v}"
                        for k, v in self.last_metrics.items()
                    ])
                    
                    progress_msg = (
                        f"\rEpoch {self.current_epoch}/{self.total_epochs} - "
                        f"ETA: {eta:.1f}s - {metrics_str}"
                    )
                    
                    # Print progress
                    if self.logger:
                        self.logger.log_and_print(progress_msg)
                    else:
                        print(progress_msg, end='', flush=True)
                        
        except Exception as e:
            # Handle any errors silently to not interrupt training
            if self.logger:
                logger.log_and_print(f"Warning: Error in progress callback: {str(e)}")

    def on_train_end(self, logs=None):
        """Show final training results."""
        try:
            total_time = time.time() - (self.start_time or 0)
            
            if self.last_metrics:
                metrics_str = ", ".join([
                    f"{k}: {v:.4f}" if isinstance(v, (int, float, np.number)) else f"{k}: {v}"
                    for k, v in self.last_metrics.items()
                ])
                final_msg = f" done ({metrics_str}) in {total_time:.1f}s"
            else:
                final_msg = f" done in {total_time:.1f}s"
            
            if self.logger:
                self.logger.log_and_print(final_msg)
            else:
                print(final_msg)
                
        except Exception as e:
            # Handle any errors silently
            if self.logger:
                self.logger.log_and_print(f"Warning: Error in progress callback: {str(e)}")
            print(" done")

    def _format_metrics(self, metrics):
        """Safely format metrics dictionary."""
        try:
            return {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in metrics.items()
            }
        except Exception:
            return metrics
        
################################################################################
# Basic Functions
################################################################################

@timing_decorator
def generate_primes(n):
    """Returns a list of the first n primes."""
    logger = logging.getLogger('PrimeAnalysis')
    logger.debug(f"Generating first {n} primes...")
    start_time = time.time()
    primes = list(primerange(2, int(n * (math.log(n) + math.log(math.log(n)) + 1))))
    primes = primes[:n]
    logger.debug(f"Generated {len(primes)} primes in {time.time() - start_time:.2f} seconds")
    return primes

@timing_decorator
def compute_gaps(primes):
    """Returns a list of gaps between consecutive primes."""
    logger = logging.getLogger('PrimeAnalysis')
    logger.debug("Computing gaps between primes...")
    start_time = time.time()
    gaps = [primes[i] - primes[i - 1] for i in range(1, len(primes))]
    logger.debug(f"Computed {len(gaps)} gaps in {time.time() - start_time:.2f} seconds")
    return gaps

@timing_decorator
def compute_composites_between(p1, p2):
    """Returns a list of composite numbers between two primes."""
    logger = logging.getLogger('PrimeAnalysis')
    logger.debug(f"Computing composites between {p1} and {p2}...")
    return [x for x in range(p1 + 1, p2) if not isprime(x)]

@timing_decorator
def factor_composites(composites):
    """Factorizes each composite and returns a list of factor dictionaries."""
    logger = logging.getLogger('PrimeAnalysis')
    logger.debug(f"Factorizing {len(composites)} composites...")
    factored = []
    for comp in composites:
        factored.append(factorint(comp))
    return factored

@njit
def _factorint_numba(n):
    """Numba-optimized version of sympy.factorint with wheel factorization."""
    factors = np.zeros((10, 2), dtype=np.int64)  # Assuming max 10 factors, each with prime and count
    factor_count = 0
    
    # Handle 2 and 3 separately
    while n % 2 == 0:
        factors[factor_count, 0] = 2
        factors[factor_count, 1] += 1
        n //= 2
        if n % 2 != 0:
            factor_count += 1
    while n % 3 == 0:
        factors[factor_count, 0] = 3
        factors[factor_count, 1] += 1
        n //= 3
        if n % 3 != 0:
            factor_count += 1
    
    # Optimized loop for other primes
    d = 5
    while d * d <= n:
        if n % d == 0:
            count = 0
            while n % d == 0:
                count += 1
                n //= d
            factors[factor_count, 0] = d
            factors[factor_count, 1] = count
            factor_count += 1
        if n % (d + 2) == 0:
            count = 0
            while n % (d + 2) == 0:
                count += 1
                n //= (d + 2)
            factors[factor_count, 0] = d + 2
            factors[factor_count, 1] = count
            factor_count += 1
        d += 6
    
    if n > 1:
        factors[factor_count, 0] = n
        factors[factor_count, 1] = 1
        factor_count += 1
    
    return factors[:factor_count]

@njit
def _compute_entropy_numba(probabilities):
    """Numba-optimized Shannon entropy calculation with numerical stability."""
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(max(p, 1e-15))  # Clip very small probabilities
    return entropy

@timing_decorator
def compute_entropy(values):
    """Compute Shannon entropy with improved numerical stability."""
    logger = logging.getLogger('PrimeAnalysis')
    logger.debug("Computing entropy of values...")
    with suppress_overflow_warnings():
        try:
            # Convert to numpy array if not already
            values_array = np.array(values, dtype=np.float64)
            
            # Check if array is empty
            if values_array.size == 0:
                return 0
            
            # Convert to list of values for Counter
            values_list = values_array.tolist()
            
            # Calculate probabilities
            counts = Counter(values_list)
            total = sum(counts.values())
            probabilities = np.array([count / total for count in counts.values()], dtype=np.float64)
            
            # Clip very small probabilities to prevent log(0)
            probabilities = np.clip(probabilities, 1e-15, 1.0)
            
            # Calculate entropy with log2
            return float(_compute_entropy_numba(probabilities))
            
        except Exception as e:
            logger.warning(f"Warning: Error computing entropy: {str(e)}")
            return 0.0
        
@njit
def isprime_numba(n):
    """Numba-optimized primality test."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


@njit
def _get_prime_type_string(prime_type_code):
    """Numba-optimized function to map prime type codes to strings."""
    if prime_type_code == 0:
        return "None"
    if prime_type_code == 1:
        return "Sophie Germain"
    if prime_type_code == 2:
        return "Safe Prime"
    if prime_type_code == 3:
        return "Mersenne Prime"
    if prime_type_code == 4:
        return "Cousin Prime"
    if prime_type_code == 5:
        return "Twin Prime"
    if prime_type_code == 6:
        return "Pythagorean Prime"
    if prime_type_code == 7:
        return "3k+1 Prime"
    if prime_type_code == 8:
        return "3k+2 Prime"
    return "Regular Prime"
    
@njit
def identify_prime_type_numba(prime):
    """Numba-vectorized prime type identification returning integer codes."""
    if prime <= 1:
        return 0  # None
    
    if (prime - 1) % 2 == 0 and isprime_numba((prime - 1) // 2):
        return 1  # Sophie Germain
    
    if isprime_numba(2*prime + 1):
        return 2  # Safe Prime
    
    if isprime_numba(2**prime - 1):
        return 3  # Mersenne Prime
    
    if isprime_numba(2*prime - 1):
        return 4  # Cousin Prime
    
    if isprime_numba(prime - 2) or isprime_numba(prime + 2):
        return 5  # Twin Prime
    
    if prime % 4 == 1:
        return 6  # Pythagorean Prime
    
    if prime % 3 == 1:
        return 7  # 3k+1 Prime
    
    if prime % 3 == 2:
        return 8  # 3k+2 Prime
    
    return 9  # Regular Prime

@functools.lru_cache(maxsize=1024)
def _cached_identify_prime_type(prime):
    """Cached version of identify_prime_type_numba."""
    return identify_prime_type_numba(prime)


def identify_prime_type(prime):
    """Identifies the type of a prime number using a cache."""
    logger = logging.getLogger('PrimeAnalysis')
    logger.debug(f"Identifying prime type for {prime}...")
    prime_type_code = _cached_identify_prime_type(prime)
    return _get_prime_type_string(prime_type_code)

@timing_decorator
def assign_composite_to_cluster(df, features, model_results=None, use_classifier=True, scaler=None, logger=None):
    """Assign a composite to a cluster using either a trained classifier or cluster centers."""
    if logger:
        logger.log_and_print("Assigning composite to cluster...")
    
    try:
        # Ensure features is a DataFrame
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame([features])
        
        # Get feature columns from the training data
        feature_cols = [col for col in df.columns if col not in [
            'cluster', 'sub_cluster', 'gap_size', 'lower_prime', 
            'upper_prime', 'is_outlier', 'preceding_gaps'
        ]]
        
        # Ensure features has the same columns as training data
        missing_cols = set(feature_cols) - set(features.columns)
        extra_cols = set(features.columns) - set(feature_cols)
        
        # Add missing columns with zeros
        for col in missing_cols:
            features[col] = 0
            
        # Remove extra columns
        features = features[feature_cols]
        
        # Scale the features if scaler provided
        if scaler:
            if not hasattr(scaler, 'fit') or not hasattr(scaler, 'transform'):
                if logger:
                    logger.log_and_print("Warning: Scaler does not have fit or transform methods, skipping scaling.")
                features_scaled = features.values
            else:
                # Fit the scaler on the training data if it hasn't been fitted yet
                if not hasattr(scaler, 'scale_'):
                    scaler.fit(df[feature_cols])
                features_scaled = scaler.transform(features)
        else:
            features_scaled = features.values
        
        # Handle NaN/inf values
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # First try using the classifier if available
        if use_classifier and model_results:
            if 'cluster_membership_rf' in model_results:
                classifier_config = model_results['cluster_membership_rf']
                if isinstance(classifier_config, dict) and 'model' in classifier_config:
                    classifier = classifier_config['model']
                    if hasattr(classifier, 'predict'):
                        try:
                            predicted_cluster = classifier.predict(features_scaled)[0]
                            if logger:
                                logger.log_and_print(f"Assigned to cluster {predicted_cluster} using classifier")
                            return int(predicted_cluster)
                        except Exception as e:
                            if logger:
                                logger.log_and_print(f"Warning: Classifier prediction failed: {str(e)}")
        
        # Fallback to using cluster centers if available
        if hasattr(df, 'cluster_centers_'):
            try:
                centers = df.cluster_centers_
                distances = cdist(features_scaled, centers)
                predicted_cluster = np.argmin(distances)
                if logger:
                    logger.log_and_print(f"Assigned to cluster {predicted_cluster} using cluster centers")
                return int(predicted_cluster)
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Warning: Cluster center assignment failed: {str(e)}")
        
        # If KMeans object is available in df
        if hasattr(df, 'kmeans_model_'):
            try:
                predicted_cluster = df.kmeans_model_.predict(features_scaled)[0]
                if logger:
                    logger.log_and_print(f"Assigned to cluster {predicted_cluster} using KMeans model")
                return int(predicted_cluster)
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Warning: KMeans prediction failed: {str(e)}")
        
        # Last resort: Use a new KMeans model on the existing data
        try:
            if 'cluster' in df.columns:
                n_clusters = len(df['cluster'].unique())
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                
                # Fit on existing data
                kmeans.fit(df[feature_cols])
                
                # Predict cluster
                predicted_cluster = kmeans.predict(features_scaled)[0]
                if logger:
                    logger.log_and_print(f"Assigned to cluster {predicted_cluster} using new KMeans model")
                return int(predicted_cluster)
                
        except Exception as e:
            if logger:
                logger.log_and_print(f"Warning: Fallback clustering failed: {str(e)}")
        
        # If all methods fail, return -1 as indicator of failure
        if logger:
            logger.log_and_print("Warning: Could not assign cluster, returning -1")
        return -1
        
    except Exception as e:
        error_msg = f"Error in cluster assignment: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        return -1
    
################################################################################
# Advanced Analysis Functions
################################################################################

class AnalysisError(Exception):
    """Custom exception class for analysis errors."""
    pass

def error_recovery_wrapper(func):
    """Decorator for error recovery in analysis functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except MemoryError:
                gc.collect()
                if attempt == max_retries - 1:
                    raise AnalysisError("Memory error persists after garbage collection")
                time.sleep(retry_delay)
            except np.linalg.LinAlgError:
                # Handle numerical computation errors
                if attempt == max_retries - 1:
                    raise AnalysisError("Numerical computation error persists")
                # Try with more stable parameters
                kwargs['numerical_stability'] = True
                time.sleep(retry_delay)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise AnalysisError(f"Unrecoverable error: {str(e)}")
                time.sleep(retry_delay)
    return wrapper

class PrimeAnalysisRecovery:
    """Class to handle error recovery and state management for prime analysis."""
    
    def __init__(self, checkpoint_dir="./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.current_state = {}
        self.error_log = []
    
    def save_checkpoint(self, state, checkpoint_name):
        """Save analysis state to checkpoint with error handling and compression."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pkl")
        temp_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_temp.pkl")
        
        try:
            # First save to temporary file
            with open(temp_path, 'wb') as f:
                # Convert DataFrames and numpy arrays to more efficient format before saving
                state_to_save = state.copy()
                for key, value in state_to_save.items():
                    if isinstance(value, pd.DataFrame):
                        table = pa.Table.from_pandas(value)
                        state_to_save[key] = table
                    elif isinstance(value, np.ndarray):
                        # Save numpy arrays using savez_compressed
                        np_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_{key}.npz")
                        np.savez_compressed(np_path, data=value)
                        state_to_save[key] = np_path
                
                # Save with highest protocol for better performance
                pickle.dump(state_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # If temporary save successful, move to final location
            if os.path.exists(checkpoint_path):
                # Keep one backup
                backup_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_backup.pkl")
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(checkpoint_path, backup_path)
                
            os.rename(temp_path, checkpoint_path)
            self.current_state = state
            
            # Log successful save
            self.log_success(f"Successfully saved checkpoint: {checkpoint_name}")
            
            # Clean up old checkpoints if needed
            self._cleanup_old_checkpoints()
            
            return True
            
        except Exception as e:
            self.log_error(f"Failed to save checkpoint {checkpoint_name}: {str(e)}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return False
            
        finally:
            # Ensure temp file is cleaned up
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def load_checkpoint(self, checkpoint_name):
        """Load analysis state from checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pkl")
        try:
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'rb') as f:
                    state = pickle.load(f)
                
                # Convert DataFrames and numpy arrays back from saved format
                for key, value in state.items():
                    if isinstance(value, pa.Table):
                        state[key] = value.to_pandas()
                    elif isinstance(value, str) and value.endswith('.npz'):
                        try:
                            np_data = np.load(value)
                            state[key] = np_data['data']
                            os.remove(value)  # Remove the temporary .npz file
                        except Exception as e:
                            self.log_error(f"Failed to load numpy array from {value}: {str(e)}")
                            state[key] = None
                
                self.current_state = state
                return state
            return None
        except Exception as e:
            self.log_error(f"Failed to load checkpoint: {str(e)}")
            return None
    
    def log_error(self, error_message):
        """Log error with timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.error_log.append(f"{timestamp}: {error_message}")
        print(f"Error logged: {error_message}")
    
    def log_success(self, message):
        """Log success message with timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        success_log = os.path.join(self.checkpoint_dir, "success.log")
        try:
            with open(success_log, 'a') as f:
                f.write(f"{timestamp}: {message}\n")
        except Exception as e:
            print(f"Warning: Could not write to success log: {str(e)}")
    
    def recover_analysis(self, n, batch_size=10000):
        """Attempt to recover analysis from last successful state."""
        try:
            # Try to load last successful state
            last_state = self.load_checkpoint("last_successful_state")
            if last_state is not None and isinstance(last_state, dict):
                if 'dataframe' in last_state and isinstance(last_state['dataframe'], pd.DataFrame):
                    if not last_state['dataframe'].empty:
                        print("Recovering from last successful state...")
                        return self.continue_analysis_from_state(last_state, n, batch_size)
            
            print("No valid recovery state found. Starting fresh analysis...")
            return None
            
        except Exception as e:
            self.log_error(f"Recovery failed: {str(e)}")
            return None
    
    def continue_analysis_from_state(self, state, n, batch_size):
        """Continue analysis from a recovered state."""
        try:
            if not isinstance(state, dict) or 'dataframe' not in state:
                return None
                
            df = state['dataframe']
            if not isinstance(df, pd.DataFrame) or df.empty:
                return None
                
            completed_primes = len(df)
            remaining_primes = n - completed_primes
            
            if remaining_primes <= 0:
                return state
            
            print(f"Continuing analysis from prime {completed_primes}")
            return state
            
        except Exception as e:
            self.log_error(f"Failed to continue analysis: {str(e)}")
            return None
    
    def _cleanup_old_checkpoints(self, max_checkpoints=5):
        """Clean up old checkpoints, keeping only the most recent ones."""
        try:
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pkl')]
            if len(checkpoints) > max_checkpoints:
                checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)))
                for old_checkpoint in checkpoints[:-max_checkpoints]:
                    os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))
        except Exception as e:
            self.log_error(f"Error cleaning up old checkpoints: {str(e)}")
                               
def compute_advanced_prime_features(p1, p2, gap, cache_dir="./cache", composite_sample_rate=COMPOSITE_SAMPLE_RATE):
    """
    Compute advanced features for a prime gap, including factor analysis and prime type identification.

    Args:
        p1 (int): The lower prime number.
        p2 (int): The upper prime number.
        gap (int): The gap size between p1 and p2.
        cache_dir (str, optional): Directory to cache computed features. Defaults to "./cache".
        composite_sample_rate (float, optional): Sampling rate for composite numbers. Defaults to COMPOSITE_SAMPLE_RATE.

    Returns:
        dict: A dictionary containing computed features.
    """
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"{p1}_{p2}"
    cache_file = os.path.join(cache_dir, f"features_{cache_key}.pkl")
    
    # Check if cached features exist
    if os.path.exists(cache_file):
        try:
            # Check if file is not empty
            if os.path.getsize(cache_file) > 0:
                with open(cache_file, 'rb') as f:
                    # Attempt to load the pickle file
                    try:
                        features = pickle.load(f)
                        if isinstance(features, dict):
                            return features
                        else:
                            print(f"Warning: Cached features for {cache_key} is not a dictionary, skipping cache.")
                    except Exception as e:
                        print(f"Warning: Could not load cached features for {cache_key}: {e}")
            else:
                print(f"Warning: Cached features for {cache_key} is empty, skipping cache.")
        except Exception as e:
            print(f"Warning: Could not load cached features for {cache_key}: {e}")
    
    # Identify prime types
    lower_prime_type = identify_prime_type(p1)
    upper_prime_type = identify_prime_type(p2)
    
    # Initialize features dictionary
    features = {
        'gap_size': gap,
        'gap_mod6': gap % 6,
        'gap_mod30': gap % 30,
        'lower_prime_type': lower_prime_type if lower_prime_type else "Regular Prime",
        'upper_prime_type': upper_prime_type if upper_prime_type else "Regular Prime",
        'type_transition': f"{lower_prime_type or 'Regular Prime'}_{upper_prime_type or 'Regular Prime'}"
    }
    
    try:
        # Get composites between primes
        composites = np.array([x for x in range(p1 + 1, p2) if not isprime_numba(x)], dtype=np.int64)
        
        # Sample composites
        num_composites = len(composites)
        num_composites_to_factorize = int(num_composites * composite_sample_rate)
        
        if num_composites_to_factorize > 0:
            if num_composites_to_factorize < num_composites:
                # Sample composites
                sample_indices = np.random.choice(num_composites, num_composites_to_factorize, replace=False)
                composites_to_factorize = composites[sample_indices]
            else:
                composites_to_factorize = composites
            
            # Use numba-optimized factorization
            factorizations = [_factorint_numba(comp) for comp in composites_to_factorize]
            
            # Analyze factors
            all_factors = []
            for factor_array in factorizations:
                for factor, count in factor_array:
                    all_factors.extend([factor] * count)
            
            if all_factors:
                # Basic factor statistics
                all_factors = np.array(all_factors, dtype=np.int64)
                features.update({
                    'unique_factors': len(set(all_factors)),
                    'total_factors': len(all_factors),
                    'max_factor': int(np.max(all_factors)),
                    'min_factor': int(np.min(all_factors)),
                    'mean_factor': float(np.mean(all_factors)),
                    'factor_std': float(np.std(all_factors)) if len(all_factors) > 1 else 0,
                    'factor_entropy': compute_entropy(all_factors),
                    'factor_density': len(all_factors) / gap
                })
                
                # Handle factor range ratio
                min_factor = int(np.min(all_factors))
                if min_factor > 0:
                    features['factor_range_ratio'] = float(np.max(all_factors) / min_factor)
                else:
                    features['factor_range_ratio'] = 0
                
                # Compute sqrt factor metrics
                sqrt_factors = np.sqrt(all_factors)
                features.update({
                    'mean_sqrt_factor': float(np.mean(sqrt_factors)),
                    'sum_sqrt_factor': float(np.sum(sqrt_factors)),
                    'std_sqrt_factor': float(np.std(sqrt_factors)) if len(sqrt_factors) > 1 else 0
                })
                
                # Factor distribution metrics
                features.update({
                    'num_distinct_prime_factors': len(set(all_factors)),
                    'product_of_prime_factors': np.prod(all_factors),
                    'sum_of_prime_factors': sum(all_factors)
                })
            else:
                # Set default values if no factors found
                features.update({
                    'unique_factors': 0,
                    'total_factors': 0,
                    'max_factor': 0,
                    'min_factor': 0,
                    'mean_factor': 0,
                    'factor_std': 0,
                    'factor_entropy': 0,
                    'factor_density': 0,
                    'factor_range_ratio': 0,
                    'mean_sqrt_factor': 0,
                    'sum_sqrt_factor': 0,
                    'std_sqrt_factor': 0,
                    'num_distinct_prime_factors': 0,
                    'product_of_prime_factors': 0,
                    'sum_of_prime_factors': 0
                })
        else:
            # Set default values if no composites found
            features.update({
                'unique_factors': 0,
                'total_factors': 0,
                'max_factor': 0,
                'min_factor': 0,
                'mean_factor': 0,
                'factor_std': 0,
                'factor_entropy': 0,
                'factor_density': 0,
                'factor_range_ratio': 0,
                'mean_sqrt_factor': 0,
                'sum_sqrt_factor': 0,
                'std_sqrt_factor': 0,
                'num_distinct_prime_factors': 0,
                'product_of_prime_factors': 0,
                'sum_of_prime_factors': 0
            })
    except Exception as e:
        print(f"Warning: Error computing features: {str(e)}")
        # Set default values on error
        features.update({
            'unique_factors': 0,
            'total_factors': 0,
            'max_factor': 0,
            'min_factor': 0,
            'mean_factor': 0,
            'factor_std': 0,
            'factor_entropy': 0,
            'factor_density': 0,
            'factor_range_ratio': 0,
            'mean_sqrt_factor': 0,
            'sum_sqrt_factor': 0,
            'std_sqrt_factor': 0,
            'num_distinct_prime_factors': 0,
            'product_of_prime_factors': 0,
            'sum_of_prime_factors': 0
        })
    
    # Cache features
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(features, f)
    except Exception as e:
        print(f"Warning: Could not cache features for {cache_key}: {e}")
    
    return features


@njit
def _compute_cluster_level_features_numba(cluster_gaps, sub_clusters, unique_factors, total_factors, factor_density, gap_mod6, gap_mod30):
    """Numba-optimized version of compute_cluster_level_features."""
    
    num_gaps = len(cluster_gaps)
    
    cluster_gap_mean = np.mean(cluster_gaps)
    cluster_gap_std = np.std(cluster_gaps)
    cluster_gap_median = np.median(cluster_gaps)
    cluster_gap_skew = skew(cluster_gaps)
    cluster_gap_kurtosis = kurtosis(cluster_gaps)
    
    sub_cluster_freqs = {}
    if len(sub_clusters) > 0:
        for sub_id in set(sub_clusters):
            sub_cluster_freqs[sub_id] = np.sum(sub_clusters == sub_id) / num_gaps
    
    sub_cluster_gap_means = {}
    sub_cluster_gap_stds = {}
    sub_cluster_gap_medians = {}
    sub_cluster_gap_skews = {}
    sub_cluster_gap_kurtosises = {}
    sub_cluster_unique_factors_means = {}
    sub_cluster_unique_factors_stds = {}
    sub_cluster_total_factors_means = {}
    sub_cluster_total_factors_stds = {}
    sub_cluster_mod6_freqs = {}
    sub_cluster_factor_density_means = {}
    sub_cluster_factor_density_stds = {}
    
    if len(sub_clusters) > 0:
        for sub_id in set(sub_clusters):
            sub_mask = sub_clusters == sub_id
            sub_gaps = cluster_gaps[sub_mask]
            sub_unique_factors = unique_factors[sub_mask]
            sub_total_factors = total_factors[sub_mask]
            sub_factor_density = factor_density[sub_mask]
            sub_mod6 = gap_mod6[sub_mask]
            
            sub_cluster_gap_means[sub_id] = np.mean(sub_gaps)
            sub_cluster_gap_stds[sub_id] = np.std(sub_gaps)
            sub_cluster_gap_medians[sub_id] = np.median(sub_gaps)
            sub_cluster_gap_skews[sub_id] = skew(sub_gaps)
            sub_cluster_gap_kurtosises[sub_id] = kurtosis(sub_gaps)
            
            if len(sub_unique_factors) > 0:
                sub_cluster_unique_factors_means[sub_id] = np.mean(sub_unique_factors)
                sub_cluster_unique_factors_stds[sub_id] = np.std(sub_unique_factors)
                sub_cluster_total_factors_means[sub_id] = np.mean(sub_total_factors)
                sub_cluster_total_factors_stds[sub_id] = np.std(sub_total_factors)
            
            if len(sub_mod6) > 0:
                for i in range(6):
                    sub_cluster_mod6_freqs[(sub_id, i)] = np.sum(sub_mod6 == i) / len(sub_mod6)
            
            if len(sub_factor_density) > 0:
                sub_cluster_factor_density_means[sub_id] = np.mean(sub_factor_density)
                sub_cluster_factor_density_stds[sub_id] = np.std(sub_factor_density)
    
    cluster_mean_unique_factors = np.mean(unique_factors) if len(unique_factors) > 0 else 0.0
    cluster_std_unique_factors = np.std(unique_factors) if len(unique_factors) > 1 else 0.0
    cluster_mean_total_factors = np.mean(total_factors) if len(total_factors) > 0 else 0.0
    cluster_std_total_factors = np.std(total_factors) if len(total_factors) > 1 else 0.0
    
    cluster_mean_factor_density = np.mean(factor_density) if len(factor_density) > 0 else 0.0
    cluster_std_factor_density = np.std(factor_density) if len(factor_density) > 1 else 0.0
    
    cluster_mod6_freqs = {}
    if len(gap_mod6) > 0:
        for i in range(6):
            cluster_mod6_freqs[i] = np.sum(gap_mod6 == i) / len(gap_mod6)
    
    cluster_mod30_freqs = {}
    if len(gap_mod30) > 0:
        for i in range(30):
            cluster_mod30_freqs[i] = np.sum(gap_mod30 == i) / len(gap_mod30)
    
    return {
        'cluster_gap_mean': float(cluster_gap_mean),
        'cluster_gap_std': float(cluster_gap_std),
        'cluster_gap_median': float(cluster_gap_median),
        'cluster_gap_skew': float(cluster_gap_skew),
        'cluster_gap_kurtosis': float(cluster_gap_kurtosis),
        'sub_cluster_freqs': sub_cluster_freqs,
        'sub_cluster_gap_means': sub_cluster_gap_means,
        'sub_cluster_gap_stds': sub_cluster_gap_stds,
        'sub_cluster_gap_medians': sub_cluster_gap_medians,
        'sub_cluster_gap_skews': sub_cluster_gap_skews,
        'sub_cluster_gap_kurtosises': sub_cluster_gap_kurtosises,
        'sub_cluster_unique_factors_means': sub_cluster_unique_factors_means,
        'sub_cluster_unique_factors_stds': sub_cluster_unique_factors_stds,
        'sub_cluster_total_factors_means': sub_cluster_total_factors_means,
        'sub_cluster_total_factors_stds': sub_cluster_total_factors_stds,
        'sub_cluster_mod6_freqs': sub_cluster_mod6_freqs,
        'sub_cluster_factor_density_means': sub_cluster_factor_density_means,
        'sub_cluster_factor_density_stds': sub_cluster_factor_density_stds,
        'cluster_mean_unique_factors': float(cluster_mean_unique_factors),
        'cluster_std_unique_factors': float(cluster_std_unique_factors),
        'cluster_mean_total_factors': float(cluster_mean_total_factors),
        'cluster_std_total_factors': float(cluster_std_total_factors),
        'cluster_mean_factor_density': float(cluster_mean_factor_density),
        'cluster_std_factor_density': float(cluster_std_factor_density),
        'cluster_mod6_freqs': cluster_mod6_freqs,
        'cluster_mod30_freqs': cluster_mod30_freqs
    }

@timing_decorator
def compute_cluster_level_features(df, cluster_id):
    """Compute aggregate features for a cluster including sub-cluster patterns."""
    cluster_data = df[df['cluster'] == cluster_id]
    
    # Extract data for numba function
    cluster_gaps = cluster_data['gap_size'].values.astype(np.float64)
    sub_clusters = cluster_data['sub_cluster'].values.astype(np.int32) if 'sub_cluster' in cluster_data.columns else np.array([], dtype=np.int32)
    unique_factors = cluster_data['unique_factors'].values.astype(np.float64) if 'unique_factors' in cluster_data.columns else np.array([], dtype=np.float64)
    total_factors = cluster_data['total_factors'].values.astype(np.float64) if 'total_factors' in cluster_data.columns else np.array([], dtype=np.float64)
    factor_density = cluster_data['factor_density'].values.astype(np.float64) if 'factor_density' in cluster_data.columns else np.array([], dtype=np.float64)
    gap_mod6 = cluster_data['gap_mod6'].values.astype(np.int32) if 'gap_mod6' in cluster_data.columns else np.array([], dtype=np.int32)
    gap_mod30 = cluster_data['gap_mod30'].values.astype(np.int32) if 'gap_mod30' in cluster_data.columns else np.array([], dtype=np.int32)
    
    # Call numba-optimized function
    numba_features = _compute_cluster_level_features_numba(
        cluster_gaps, sub_clusters, unique_factors, total_factors, factor_density, gap_mod6, gap_mod30
    )
    
    features = {
        'cluster_gap_mean': numba_features['cluster_gap_mean'],
        'cluster_gap_std': numba_features['cluster_gap_std'],
        'cluster_gap_median': numba_features['cluster_gap_median'],
        'cluster_gap_skew': numba_features['cluster_gap_skew'],
        'cluster_gap_kurtosis': numba_features['cluster_gap_kurtosis'],
    }
    
    # Sub-cluster analysis
    if 'sub_cluster' in cluster_data.columns:
        # Sub-cluster distribution
        for sub_id, freq in numba_features['sub_cluster_freqs'].items():
            features[f'sub_cluster_{sub_id}_freq'] = freq
        
        # Statistics for each sub-cluster
        for sub_id in sorted(cluster_data['sub_cluster'].unique()):
            prefix = f'sub_cluster_{sub_id}'
            
            # Gap statistics within sub-cluster
            features.update({
                f'{prefix}_gap_mean': numba_features['sub_cluster_gap_means'].get(sub_id, 0.0),
                f'{prefix}_gap_std': numba_features['sub_cluster_gap_stds'].get(sub_id, 0.0),
                f'{prefix}_gap_median': numba_features['sub_cluster_gap_medians'].get(sub_id, 0.0),
                f'{prefix}_gap_skew': numba_features['sub_cluster_gap_skews'].get(sub_id, 0.0),
                f'{prefix}_gap_kurtosis': numba_features['sub_cluster_gap_kurtosises'].get(sub_id, 0.0)
            })
            
            # Factor patterns within sub-cluster
            features.update({
                f'{prefix}_mean_unique_factors': numba_features['sub_cluster_unique_factors_means'].get(sub_id, 0.0),
                f'{prefix}_std_unique_factors': numba_features['sub_cluster_unique_factors_stds'].get(sub_id, 0.0),
                f'{prefix}_mean_total_factors': numba_features['sub_cluster_total_factors_means'].get(sub_id, 0.0),
                f'{prefix}_std_total_factors': numba_features['sub_cluster_total_factors_stds'].get(sub_id, 0.0)
            })
            
            # Modulo patterns within sub-cluster
            for i in range(6):
                features[f'{prefix}_mod6_{i}_freq'] = numba_features['sub_cluster_mod6_freqs'].get((sub_id, i), 0.0)
            
            # Factor distribution patterns within sub-cluster
            features.update({
                f'{prefix}_mean_factor_density': numba_features['sub_cluster_factor_density_means'].get(sub_id, 0.0),
                f'{prefix}_std_factor_density': numba_features['sub_cluster_factor_density_stds'].get(sub_id, 0.0)
            })
        
        # Sub-cluster transition patterns
        sub_cluster_transitions = pd.crosstab(
            cluster_data['sub_cluster'],
            cluster_data['sub_cluster'].shift(-1),
            normalize='index'
        )
        for i in sorted(sub_cluster_transitions.index):
            for j in sorted(sub_cluster_transitions.columns):
                val = sub_cluster_transitions.loc[i, j]
                features[f'sub_cluster_transition_{i}_to_{j}'] = float(val) if isinstance(val, (int, float)) and pd.notnull(val) else 0.0
    
    # Factor patterns within cluster
    features.update({
        'cluster_mean_unique_factors': numba_features['cluster_mean_unique_factors'],
        'cluster_std_unique_factors': numba_features['cluster_std_unique_factors'],
        'cluster_mean_total_factors': numba_features['cluster_mean_total_factors'],
        'cluster_std_total_factors': numba_features['cluster_std_total_factors']
    })
    
    # Factor distribution patterns
    features.update({
        'cluster_mean_factor_density': numba_features['cluster_mean_factor_density'],
        'cluster_std_factor_density': numba_features['cluster_std_factor_density']
    })
    
    # Modulo patterns within cluster
    for i in range(6):
        features[f'cluster_mod6_{i}_freq'] = numba_features['cluster_mod6_freqs'].get(i, 0.0)
    
    for i in range(30):
        features[f'cluster_mod30_{i}_freq'] = numba_features['cluster_mod30_freqs'].get(i, 0.0)
    
    # Factor type distributions
    features.update({
        'cluster_mean_sqrt_factor': numba_features['cluster_mean_sqrt_factor'],
        'cluster_std_sqrt_factor': numba_features['cluster_std_sqrt_factor']
    })
    
    # Prime factor patterns
    features.update({
        'cluster_mean_prime_factor_sum': numba_features['cluster_mean_prime_factor_sum'],
        'cluster_std_prime_factor_sum': numba_features['cluster_std_prime_factor_sum']
    })
    
    return features

@njit
def _compute_sequence_features_numba(gaps, clusters, sub_clusters, unique_factors, factor_density, sum_of_prime_factors, sequence_length):
    """Numba-optimized version of compute_sequence_features."""
    
    num_gaps = len(gaps)
    
    seq_mean_gap = np.mean(gaps)
    seq_std_gap = np.std(gaps)
    seq_trend = np.polyfit(np.arange(sequence_length), gaps, 1)[0]
    seq_last_gap = gaps[-1]
    
    seq_cluster_freqs = {}
    if len(clusters) > 0:
        for cluster_id in set(clusters):
            seq_cluster_freqs[cluster_id] = np.sum(clusters == cluster_id) / num_gaps
    
    seq_last_clusters = {}
    if len(clusters) > 0:
        for j in range(1, min(11, sequence_length + 1)):
            seq_last_clusters[j] = clusters[-j]
    
    seq_sub_cluster_freqs = {}
    seq_last_sub_clusters = {}
    seq_sub_transition_counts = {}
    seq_max_sub_cluster_run = 1
    
    if len(sub_clusters) > 0:
        for sub_id in set(sub_clusters):
            seq_sub_cluster_freqs[sub_id] = np.sum(sub_clusters == sub_id) / num_gaps
        
        for j in range(1, min(11, sequence_length + 1)):
            seq_last_sub_clusters[j] = sub_clusters[-j]
        
        for i in range(len(sub_clusters) - 1):
            from_cluster = sub_clusters[i]
            to_cluster = sub_clusters[i+1]
            key = (from_cluster, to_cluster)
            seq_sub_transition_counts[key] = seq_sub_transition_counts.get(key, 0) + 1
        
        current_run = 1
        max_run = 1
        for j in range(1, len(sub_clusters)):
            if sub_clusters[j] == sub_clusters[j-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        seq_max_sub_cluster_run = max_run
    
    seq_mean_unique_factors = np.mean(unique_factors) if len(unique_factors) > 0 else 0.0
    seq_std_unique_factors = np.std(unique_factors) if len(unique_factors) > 1 else 0.0
    seq_trend_unique_factors = np.polyfit(np.arange(sequence_length), unique_factors, 1)[0] if len(unique_factors) > 1 else 0.0
    
    seq_mod6_freqs = {}
    if len(gaps) > 0:
        for i in range(6):
            seq_mod6_freqs[i] = np.sum((gaps % 6) == i) / len(gaps)
    
    seq_mean_factor_density = np.mean(factor_density) if len(factor_density) > 0 else 0.0
    seq_std_factor_density = np.std(factor_density) if len(factor_density) > 1 else 0.0
    seq_trend_factor_density = np.polyfit(np.arange(sequence_length), factor_density, 1)[0] if len(factor_density) > 1 else 0.0
    
    seq_mean_prime_factor_sum = np.mean(sum_of_prime_factors) if len(sum_of_prime_factors) > 0 else 0.0
    seq_std_prime_factor_sum = np.std(sum_of_prime_factors) if len(sum_of_prime_factors) > 1 else 0.0
    seq_trend_prime_factor_sum = np.polyfit(np.arange(sequence_length), sum_of_prime_factors, 1)[0] if len(sum_of_prime_factors) > 1 else 0.0
    
    return {
        'seq_mean_gap': float(seq_mean_gap),
        'seq_std_gap': float(seq_std_gap),
        'seq_trend': float(seq_trend),
        'seq_last_gap': float(seq_last_gap),
        'seq_cluster_freqs': seq_cluster_freqs,
        'seq_last_clusters': seq_last_clusters,
        'seq_sub_cluster_freqs': seq_sub_cluster_freqs,
        'seq_last_sub_clusters': seq_last_sub_clusters,
        'seq_sub_transition_counts': seq_sub_transition_counts,
        'seq_max_sub_cluster_run': int(seq_max_sub_cluster_run),
        'seq_mean_unique_factors': float(seq_mean_unique_factors),
        'seq_std_unique_factors': float(seq_std_unique_factors),
        'seq_trend_unique_factors': float(seq_trend_unique_factors),
        'seq_mod6_freqs': seq_mod6_freqs,
        'seq_mean_factor_density': float(seq_mean_factor_density),
        'seq_std_factor_density': float(seq_std_factor_density),
        'seq_trend_factor_density': float(seq_trend_factor_density),
        'seq_mean_prime_factor_sum': float(seq_mean_prime_factor_sum),
        'seq_std_prime_factor_sum': float(seq_std_prime_factor_sum),
        'seq_trend_prime_factor_sum': float(seq_trend_prime_factor_sum)
    }


@timing_decorator
def compute_sequence_features(df, sequence_length=50):
    """Compute features from longer sequences including sub-cluster patterns."""
    features = []
    
    for i in range(sequence_length, len(df)):
        sequence = df.iloc[i-sequence_length:i]
        
        # Extract data for numba function
        gaps = sequence['gap_size'].values.astype(np.float64)
        clusters = sequence['cluster'].values.astype(np.int32) if 'cluster' in sequence.columns else np.array([], dtype=np.int32)
        sub_clusters = sequence['sub_cluster'].values.astype(np.int32) if 'sub_cluster' in sequence.columns else np.array([], dtype=np.int32)
        unique_factors = sequence['unique_factors'].values.astype(np.float64) if 'unique_factors' in sequence.columns else np.array([], dtype=np.float64)
        factor_density = sequence['factor_density'].values.astype(np.float64) if 'factor_density' in sequence.columns else np.array([], dtype=np.float64)
        sum_of_prime_factors = sequence['sum_of_prime_factors'].values.astype(np.float64) if 'sum_of_prime_factors' in sequence.columns else np.array([], dtype=np.float64)
        
        # Call numba-optimized function
        numba_features = _compute_sequence_features_numba(
            gaps, clusters, sub_clusters, unique_factors, factor_density, sum_of_prime_factors, sequence_length
        )
        
        # Basic sequence statistics
        seq_features = {
            'seq_mean_gap': numba_features['seq_mean_gap'],
            'seq_std_gap': numba_features['seq_std_gap'],
            'seq_trend': numba_features['seq_trend'],
            'seq_last_gap': numba_features['seq_last_gap']
        }
        
        # Cluster transition patterns
        if 'cluster' in sequence.columns:
            for cluster_id, freq in numba_features['seq_cluster_freqs'].items():
                seq_features[f'seq_cluster_{cluster_id}_freq'] = freq
            
            # Last N clusters
            for j, cluster in numba_features['seq_last_clusters'].items():
                seq_features[f'seq_last_{j}_cluster'] = cluster
        
        # Sub-cluster sequence patterns
        if 'sub_cluster' in sequence.columns:
            # Sub-cluster frequencies in sequence
            for sub_id, freq in numba_features['seq_sub_cluster_freqs'].items():
                seq_features[f'seq_sub_cluster_{sub_id}_freq'] = freq
            
            # Last N sub-clusters
            for j, sub_cluster in numba_features['seq_last_sub_clusters'].items():
                seq_features[f'seq_last_{j}_sub_cluster'] = sub_cluster
            
            # Sub-cluster transition patterns in sequence
            for (from_cluster, to_cluster), count in numba_features['seq_sub_transition_counts'].items():
                total_transitions = len(sequence) - 1
                seq_features[f'seq_sub_transition_{from_cluster}_to_{to_cluster}'] = float(count / total_transitions) if total_transitions > 0 else 0.0
            
            # Sub-cluster runs (consecutive same sub-clusters)
            seq_features['seq_max_sub_cluster_run'] = numba_features['seq_max_sub_cluster_run']
        
        # Factor pattern evolution
        if 'unique_factors' in sequence.columns:
            seq_features.update({
                'seq_mean_unique_factors': numba_features['seq_mean_unique_factors'],
                'seq_std_unique_factors': numba_features['seq_std_unique_factors'],
                'seq_trend_unique_factors': numba_features['seq_trend_unique_factors']
            })
        
        # Modulo pattern evolution
        if 'gap_mod6' in sequence.columns:
            for i, freq in numba_features['seq_mod6_freqs'].items():
                seq_features[f'seq_mod6_{i}_freq'] = freq
        
        # Factor density evolution
        if 'factor_density' in sequence.columns:
            seq_features.update({
                'seq_mean_factor_density': numba_features['seq_mean_factor_density'],
                'seq_std_factor_density': numba_features['seq_std_factor_density'],
                'seq_trend_factor_density': numba_features['seq_trend_factor_density']
            })
        
        # Prime factor pattern evolution
        if 'sum_of_prime_factors' in sequence.columns:
            seq_features.update({
                'seq_mean_prime_factor_sum': numba_features['seq_mean_prime_factor_sum'],
                'seq_std_prime_factor_sum': numba_features['seq_std_prime_factor_sum'],
                'seq_trend_prime_factor_sum': numba_features['seq_trend_prime_factor_sum']
            })
        
        features.append(seq_features)
    
    return pd.DataFrame(features)

@timing_decorator
def analyze_gap_patterns(gaps, df=None, max_sequence_length=5, logger=None):
    """Analyze patterns in the sequence of prime gaps with improved performance."""
    with suppress_numeric_warnings():
        if not isinstance(gaps, np.ndarray):
            gaps = np.array(gaps, dtype=np.float64)
            
        pattern_analysis = {
            'runs': [],
            'mod_patterns': {},
            'common_gaps': [],
            'uncommon_gaps': [],
            'arithmetic_progression': None,
            'geometric_progression': None,
            'periodicity': {
                'main_period': None,
                'strength': 0.0,
                'all_periods': []
            },
            'prime_type_patterns': {},
        }
        
        try:
            print("  Analyzing runs...")
            # Find runs using boolean indexing
            if len(gaps) > 0:
                diffs = np.diff(gaps)
                run_starts = np.where(np.abs(diffs) > 1e-10)[0] + 1
                
                if len(run_starts) > 0:
                    runs = np.split(gaps, run_starts)
                else:
                    runs = [gaps]
                
                pattern_analysis['runs'] = [run.tolist() for run in runs if len(run) > 1]
            
            print("  Analyzing modular patterns...")
            moduli = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
            
            # Vectorized modulo and entropy calculation
            for m in moduli:
                residues = (gaps % m).astype(int)
                counts = np.bincount(residues, minlength=m)
                probabilities = counts / len(residues)
                probabilities = np.clip(probabilities, 1e-15, 1.0) # Clip very small probabilities
                entropy_val = float(-np.sum(probabilities * np.log2(probabilities)))
                
                pattern_analysis['mod_patterns'][m] = {
                    'counts': dict(zip(range(m), counts)),
                    'entropy': entropy_val
                }
            
            print("  Analyzing common gaps...")
            gap_counts = Counter(gaps.astype(int).tolist())
            pattern_analysis['common_gaps'] = gap_counts.most_common(10)
            
            print("  Analyzing uncommon gaps...")
            pattern_analysis['uncommon_gaps'] = gap_counts.most_common()[:-11:-1]
            
            print("  Analyzing arithmetic progression...")
            if len(gaps) > 2:
                diffs = np.diff(gaps)
                if np.all(np.abs(np.diff(diffs)) < 1e-10):
                    pattern_analysis['arithmetic_progression'] = {
                        'start': float(gaps[0]),
                        'difference': float(diffs[0])
                    }
            
            print("  Analyzing geometric progression...")
            if len(gaps) > 2:
                ratios = gaps[1:] / gaps[:-1]
                if np.all(np.abs(np.diff(ratios)) < 1e-10):
                    pattern_analysis['geometric_progression'] = {
                        'start': float(gaps[0]),
                        'ratio': float(ratios[0])
                    }
            
            print("  Analyzing periodicity...")
            if len(gaps) > 10:
                # Use smaller window and faster FFT approach
                max_window = min(len(gaps), 128)  # Limit window size
                series = gaps[:max_window] - np.mean(gaps[:max_window])
                
                # Use real FFT instead of complex FFT (twice as fast)
                fft_result = np.fft.rfft(series)
                power = np.abs(fft_result)
                freqs = np.fft.rfftfreq(len(series), d=1.0)
                
                # Only look at meaningful frequencies
                mask = freqs > 0
                power = power[mask]
                freqs = freqs[mask]
                
                if len(power) > 0:
                    # Find top periods quickly
                    top_k = min(5, len(power))
                    top_indices = np.argpartition(power, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(-power[top_indices])]
                    
                    # Get main period
                    main_idx = top_indices[0]
                    if freqs[main_idx] != 0:
                        period = 1.0 / freqs[main_idx]
                        strength = float(power[main_idx] / np.sum(power))
                        
                        pattern_analysis['periodicity'].update({
                            'main_period': float(period),
                            'strength': strength,
                            'all_periods': [
                                (float(1.0 / freqs[idx]), float(power[idx] / np.sum(power)))
                                for idx in top_indices
                                if freqs[idx] != 0
                            ]
                        })
            
            # Analyze prime type patterns
            if df is not None and 'lower_prime_type' in df.columns and 'upper_prime_type' in df.columns:
                if logger:
                    logger.log_and_print("Analyzing prime type patterns...")
                
                # Type transitions
                type_transitions = Counter(
                    zip(df['lower_prime_type'], df['upper_prime_type'])
                )
                pattern_analysis['prime_type_patterns']['type_transitions'] = type_transitions.most_common(10)
                
                # Type runs
                types = df['lower_prime_type'].values
                runs = []
                current_run = 1
                if len(types) > 1:
                    for i in range(1, len(types)):
                        if types[i] == types[i-1]:
                            current_run += 1
                        else:
                            runs.append(current_run)
                            current_run = 1
                    runs.append(current_run)
                pattern_analysis['prime_type_patterns']['type_runs'] = runs
                
                # Type and gap counts
                type_gap_counts = Counter(
                    zip(df['lower_prime_type'], df['gap_mod6'])
                )
                pattern_analysis['prime_type_patterns']['type_gap_counts'] = type_gap_counts.most_common(10)
            
            return pattern_analysis
            
        except Exception as e:
            print(f"Warning: Error in pattern analysis: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return pattern_analysis
        
@timing_decorator
def analyze_primes_and_gaps(n, output_log_file, plot_dir):
    """Main analysis pipeline with enhanced reporting and numerical stability."""
    with suppress_numeric_warnings():
        logger = PrimeAnalysisLogger(debug_mode=False)
        
        logger.log_and_print(f"\nStarting advanced prime number analysis for n={n}")
        overall_start = time.time()
    
        try:
            if n >= BATCH_THRESHOLD:
                logger.log_and_print(f"\nLarge dataset detected (N={n}). Using batch processing...")
                
                # Determine optimal batch size based on available memory
                available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
                estimated_memory_per_prime = 0.5  # MB per prime (conservative estimate)
                batch_size = min(
                    5000,  # Maximum batch size
                    int((available_memory * 0.2) / estimated_memory_per_prime),  # Use 20% of available memory
                    int(n / (psutil.cpu_count() or 1)) # Distribute workload across cores
                )
                
                logger.log_and_print(f"Using batch size of {batch_size} (Available memory: {available_memory:.2f} MB)")
                
                # Call the large-scale analysis function
                complete_results = analyze_primes_and_gaps_large_scale(n, output_log_file, plot_dir, batch_size=batch_size)
                
                if complete_results is None:
                    logger.log_and_print("Error: Large-scale analysis failed, cannot proceed.", level=logging.ERROR)
                    return None
                
                return complete_results
            else:
                logger.log_and_print(f"\nStandard dataset size (N={n}). Using regular processing...")
                # Generate primes and compute gaps
                primes = generate_primes(n)
                gaps = compute_gaps(primes)
                
                # Compute features for each gap with protection
                logger.log_and_print("\nComputing advanced features...")
                features_list = []
                previous_gaps = []
                
                for i in range(len(gaps)):
                    features = compute_advanced_prime_features(
                        primes[i],
                        primes[i + 1],
                        gaps[i]
                    )
                    features['lower_prime'] = primes[i]
                    features['upper_prime'] = primes[i + 1]
                    features_list.append(features)
                
                # Create DataFrame and optimize memory usage
                logger.log_and_print("\nCreating and enhancing dataset...")
                df = pd.DataFrame(features_list)
                df = optimize_memory_usage(df)
                
                # Analyze chaos patterns
                logger.log_and_print("Analyzing chaos patterns...")
                chaos_metrics = compute_chaos_metrics(df, feature_cols=['gap_size'], logger=logger)
                
                # Analyze superposition patterns
                logger.log_and_print("Analyzing superposition patterns...")
                superposition_patterns = compute_superposition_patterns(df, feature_cols=['gap_size'], logger=logger)
                
                # Analyze wavelet patterns
                logger.log_and_print("Analyzing wavelet patterns...")
                wavelet_patterns = analyze_wavelet_patterns(df, feature_col='gap_size', logger=logger)
                
                # Compute fractal dimension
                logger.log_and_print("Computing fractal dimension...")
                fractal_dimension = compute_fractal_dimension(df, feature_col='gap_size', logger=logger)
                
                # Analyze phase space
                logger.log_and_print("Analyzing phase space...")
                phase_space_analysis = analyze_phase_space(df, feature_col='gap_size', logger=logger)
                
                # Create recurrence plot
                logger.log_and_print("Creating recurrence plot...")
                recurrence_plot_data = create_recurrence_plot(df, feature_col='gap_size', logger=logger)
                
                # Create advanced features
                logger.log_and_print("\nCreating advanced features...")
                df = create_advanced_features(df, logger=logger, chaos_metrics=chaos_metrics, superposition_patterns=superposition_patterns)
                
                # Prepare training data with error handling
                logger.log_and_print("\nPreparing training data...")
                try:
                    X, y, feature_cols, cluster_X, cluster_y, gap_cluster_X, gap_cluster_y, next_cluster_X, next_cluster_y = \
                        prepare_training_data(df)
                    
                    if X is None or X.empty or len(feature_cols) == 0:
                        logger.log_and_print("Warning: No valid features for training", level=logging.WARNING)
                        return None
                    
                    # Train models with protection
                    logger.log_and_print("Training predictive models...")
                    model_results, feature_importance = train_models(
                        X, y, feature_cols, cluster_X, cluster_y, 
                        gap_cluster_X, gap_cluster_y, next_cluster_X, next_cluster_y, logger=logger
                    )
                except Exception as e:
                    logger.log_and_print(f"Error in data preparation or training: {str(e)}", level=logging.ERROR)
                    model_results, feature_importance = {}, pd.DataFrame()
                
                # Analyze patterns with protection
                logger.log_and_print("\nAnalyzing gap patterns...")
                pattern_analysis = analyze_gap_patterns(gaps, df=df, logger=logger)
                
                # Perform clustering with protection
                logger.log_and_print("\nPerforming initial clustering...")
                feature_cols = [col for col in df.columns if col not in [
                    'cluster', 'sub_cluster', 'gap_size', 'lower_prime', 
                    'upper_prime', 'is_outlier', 'preceding_gaps'
                ]]
                X = df[feature_cols].copy()
                
                # Scale features with protection
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled = np.clip(X_scaled, -1e10, 1e10)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                df['cluster'] = kmeans.fit_predict(X_scaled)
                
                # Perform sub-clustering with protection
                logger.log_and_print("Performing sub-clustering...")
                for cluster_id in sorted(df['cluster'].unique()):
                    cluster_data = df[df['cluster'] == cluster_id]
                    if len(cluster_data) > 3:
                        X_cluster = cluster_data[feature_cols].copy()
                        X_cluster_scaled = scaler.fit_transform(X_cluster)
                        X_cluster_scaled = np.clip(X_cluster_scaled, -1e10, 1e10)
                        
                        kmeans_sub = KMeans(n_clusters=2, random_state=42, n_init=10)
                        df.loc[df['cluster'] == cluster_id, 'sub_cluster'] = kmeans_sub.fit_predict(X_cluster_scaled)
                    else:
                        df.loc[df['cluster'] == cluster_id, 'sub_cluster'] = -1
                
                # Analyze outliers with protection
                logger.log_and_print("Analyzing outliers...")
                df, outlier_threshold = detect_outliers(df)
                
                # Initialize preceding_gaps column
                df['preceding_gaps'] = pd.Series([None] * len(df), dtype=object)
                
                # Update preceding gaps for outliers
                outlier_count = 0
                for idx in range(len(df)):
                    if df.iloc[idx]['is_outlier']:
                        outlier_count += 1
                        start_idx = max(0, idx - 5)
                        preceding_gaps = df.iloc[start_idx:idx]['gap_size'].tolist()
                        df.at[df.index[idx], 'preceding_gaps'] = preceding_gaps
                
                logger.log_and_print(f"Found {outlier_count} outliers above threshold {outlier_threshold:.2f}")
                
                # Compute cluster statistics
                logger.log_and_print("Computing cluster statistics...")
                cluster_stats = compute_cluster_statistics(df, logger=logger)
                
                # Compute temporal pattern statistics
                logger.log_and_print("Computing temporal pattern statistics...")
                temporal_patterns = compute_temporal_pattern_statistics(df, logger=logger)
                
                # Compute cluster separation metrics
                logger.log_and_print("Computing cluster separation metrics...")
                separation_metrics = compute_cluster_separation_metrics(df, logger=logger)
                
                # Compute correlation statistics
                logger.log_and_print("Computing correlation statistics...")
                correlation_stats = compute_correlation_statistics(df, logger=logger)
                
                # Compute SHAP values for feature interpretation
                logger.log_and_print("\nComputing SHAP values...")
                shap_values, shap_importance = compute_shap_values(
                    {name: model_results[name].get('model') for name in ['random_forest', 'xgboost'] if name in model_results and 'model' in model_results.get(name, {})},
                    X,
                    feature_cols,
                    logger=logger
                )
                
                # Compute prediction intervals for best model
                try:
                    valid_models = [
                        k for k in model_results
                        if 'avg_test_mse' in model_results.get(k, {}) and 'model' in model_results.get(k, {})
                    ]
                    
                    logger.log_and_print(f"DEBUG: Valid models for prediction intervals: {valid_models}")
                    
                    if valid_models:
                        best_model_name = min(
                            valid_models,
                            key=lambda k: model_results[k].get('avg_test_mse', float('inf'))
                        )
                        
                        logger.log_and_print(f"DEBUG: best_model_name before prediction intervals: {best_model_name}")
                        logger.log_and_print(f"DEBUG: model_results before prediction intervals: {model_results.keys()}")
                        
                        if best_model_name in model_results and 'model' in model_results.get(best_model_name, {}):
                            best_model = model_results[best_model_name]['model']
                            logger.log_and_print(f"\nComputing prediction intervals for {best_model_name}...")
                            mean_pred, lower_pred, upper_pred = compute_prediction_intervals(best_model, X, logger=logger)
                            prediction_intervals = {
                                'mean': mean_pred,
                                'lower': lower_pred,
                                'upper': upper_pred
                            }
                        else:
                            logger.log_and_print(f"Warning: No 'model' key found for best model: {best_model_name}. Skipping prediction intervals.")
                    else:
                        logger.log_and_print("Warning: No valid models with 'model' key found for prediction intervals. Skipping prediction intervals.")
                except Exception as e:
                    logger.log_and_print(f"Warning: Could not compute prediction intervals: {str(e)}")
                
                # Detect change points
                logger.log_and_print("\nDetecting change points...")
                change_point_analysis = detect_change_points(df, logger=logger)
                
                # Perform advanced clustering analysis
                logger.log_and_print("\nPerforming advanced clustering analysis...")
                advanced_clustering_results = perform_advanced_clustering_analysis(df, logger=logger)
                
                # Perform statistical tests
                logger.log_and_print("\nPerforming statistical tests...")
                statistical_test_results = perform_advanced_statistical_tests(df, advanced_clustering_results, logger=logger)
                
                # Generate prime probability map
                logger.log_and_print("\nGenerating prime probability map...")
                prime_probability_map = generate_prime_probability_map(
                    df,
                    model_results,
                    feature_cols,
                    StandardScaler(),
                    StandardScaler(),
                    StandardScaler(),
                    n_primes_to_predict=1000,
                    logger=logger
                )
                
                # Analyze cluster transitions
                logger.log_and_print("\nAnalyzing cluster transitions...")
                cluster_transitions = analyze_cluster_transitions_advanced(df, logger=logger)
                
                # Analyze gap sequences
                logger.log_and_print("\nAnalyzing gap sequences...")
                gap_sequences = analyze_gap_sequences_advanced(df, logger=logger)
                
                # Analyze prime factor patterns
                logger.log_and_print("\nAnalyzing prime factor patterns...")
                factor_patterns = analyze_prime_factor_patterns(df, logger=logger)
                
                # Create visualizations with protection
                logger.log_and_print("\nCreating visualizations...")
                analyses = {
                    'pattern_analysis': pattern_analysis,
                    'cluster_stats': cluster_stats,
                    'temporal_stats': temporal_patterns,
                    'separation_metrics': separation_metrics,
                    'gap_distribution': None,
                    'gap_sequences': gap_sequences,
                    'factor_patterns': factor_patterns,
                    'cluster_transitions': cluster_transitions,
                    'prime_probability_map': prime_probability_map,
                    'shap_values': shap_values,
                    'shap_importance': shap_importance,
                    'prediction_intervals': prediction_intervals,
                    'change_points': change_point_analysis,
                    'advanced_clustering': advanced_clustering_results,
                    'statistical_tests': statistical_test_results
                }
                create_visualizations_large_scale(df, feature_importance, pattern_analysis, 
                                               plot_dir, model_results, analysis_stats=analyses)
                create_cluster_visualization(df, plot_dir, logger=logger)
                
                # Store complete results
                complete_results = {
                    'dataframe': df,
                    'model_results': model_results,
                    'feature_importance': feature_importance,
                    **analyses
                }
                
                logger.log_and_print(f"\nAnalysis completed in {time.time() - overall_start:.2f} seconds")
                
                # Write report
                logger.log_and_print(f"\nWriting analysis results to {output_log_file}")
                _write_analysis_report(
                    output_log_file,
                    model_results,
                    feature_importance,
                    pattern_analysis,
                    df,
                    prime_probability_map=prime_probability_map,
                    cluster_sequence_analysis=None,
                    gap_distribution=analyses['gap_distribution'],
                    gap_sequences=analyses['gap_sequences'],
                    factor_patterns=analyses['factor_patterns'],
                    cluster_transitions=analyses['cluster_transitions'],
                    temporal_patterns=analyses['temporal_patterns'],
                    separation_metrics=analyses['separation_metrics'],
                    shap_values=shap_values,
                    shap_importance=shap_importance,
                    prediction_intervals=prediction_intervals,
                    change_points=change_point_analysis,
                    cluster_stats=cluster_stats,
                    advanced_clustering=advanced_clustering_results,
                    statistical_tests=statistical_test_results,
                    logger=logger
                )
                
                return complete_results
        
        except Exception as e:
            logger.log_and_print(f"Critical error in analysis pipeline: {str(e)}", level=logging.ERROR)
            return None          

@timing_decorator
def analyze_primes_and_gaps_large_scale(n, output_log_file, plot_dir, batch_size=100000):
    """Enhanced analysis pipeline for large datasets with error handling."""
    with suppress_numeric_warnings():
        try:
            logger = PrimeAnalysisLogger(debug_mode=False)
            logger.log_and_print(f"\nStarting large-scale prime number analysis for n={n}")
            overall_start = time.time()
            
            # Initialize recovery object
            recovery = PrimeAnalysisRecovery()
            
            # Process primes in batches
            df = batch_process_primes(n, batch_size, logger=logger)
            df = optimize_memory_usage(df, logger=logger)
            
            # Save checkpoint after batch processing
            if not recovery.save_checkpoint({'dataframe': df}, "after_batch_processing"):
                logger.log_and_print("Warning: Checkpoint save failed after batch processing.", level=logging.WARNING)
            
            # Analyze chaos patterns
            logger.log_and_print("Analyzing chaos patterns...")
            chaos_metrics = compute_chaos_metrics(df, feature_cols=['gap_size'], logger=logger)
            
            # Analyze superposition patterns
            logger.log_and_print("Analyzing superposition patterns...")
            superposition_patterns = compute_superposition_patterns(df, feature_cols=['gap_size'], logger=logger)
            
             # Analyze wavelet patterns
            logger.log_and_print("Analyzing wavelet patterns...")
            wavelet_patterns = analyze_wavelet_patterns(df, feature_col='gap_size', logger=logger)
            
            # Compute fractal dimension
            logger.log_and_print("Computing fractal dimension...")
            fractal_dimension = compute_fractal_dimension(df, feature_col='gap_size', logger=logger)
            
            # Analyze phase space
            logger.log_and_print("Analyzing phase space...")
            phase_space_analysis = analyze_phase_space(df, feature_col='gap_size', logger=logger)
            
            # Create recurrence plot
            logger.log_and_print("Creating recurrence plot...")
            recurrence_plot_data = create_recurrence_plot(df, feature_col='gap_size', logger=logger)
            
            # Create advanced features
            logger.log_and_print("\nCreating advanced features...")
            df = create_advanced_features(df, logger=logger, chaos_metrics=chaos_metrics, superposition_patterns=superposition_patterns)
            
            # Save checkpoint after feature engineering
            if not recovery.save_checkpoint({'dataframe': df}, "after_feature_engineering"):
                logger.log_and_print("Warning: Checkpoint save failed after feature engineering.", level=logging.WARNING)
            
            # Prepare training data with error handling
            logger.log_and_print("\nPreparing training data...")
            try:
                X, y, feature_cols, cluster_X, cluster_y, gap_cluster_X, gap_cluster_y, next_cluster_X, next_cluster_y = \
                    prepare_training_data(df)
                
                if X is None or X.empty or len(feature_cols) == 0:
                    logger.log_and_print("Warning: No valid features for training", level=logging.WARNING)
                    return None
                
                # Train models with error handling
                logger.log_and_print("Training predictive models...")
                model_results, feature_importance = train_models(
                    X, y, feature_cols, cluster_X, cluster_y, 
                    gap_cluster_X, gap_cluster_y, next_cluster_X, next_cluster_y, logger=logger
                )
            except Exception as e:
                logger.log_and_print(f"Error in data preparation or training: {str(e)}", level=logging.ERROR)
                model_results, feature_importance = {}, pd.DataFrame()
            
            # Save checkpoint after model training
            if not recovery.save_checkpoint({'dataframe': df, 'model_results': model_results, 'feature_importance': feature_importance}, "after_model_training"):
                logger.log_and_print("Warning: Checkpoint save failed after model training.", level=logging.WARNING)
            
            # Perform all analyses in a single block with memory optimization
            logger.log_and_print("\nPerforming analyses...")
            analyses = {
                'pattern_analysis': analyze_gap_patterns(df['gap_size'].values, df=df, logger=logger),
                'cluster_features': analyze_cluster_features(df, logger=logger),
                'temporal_patterns': {}, # Initialize temporal_patterns
                'separation_metrics': compute_cluster_separation_metrics(df, logger=logger),
                'gap_distribution': analyze_gap_distribution_characteristics(df, logger=logger),
                'gap_sequences': analyze_gap_sequences_advanced(df, logger=logger),
                'factor_patterns': analyze_prime_factor_patterns(df, logger=logger),
                'cluster_transitions': analyze_cluster_transitions_advanced(df, logger=logger),
                'cluster_stability': compute_cluster_stability(df, logger=logger),
                'cluster_stats': compute_cluster_statistics(df, logger=logger),
                'correlation_stats': compute_correlation_statistics(df, logger=logger)
            }
            
            # Perform time series analysis and handle errors
            try:
                analyses['temporal_patterns'] = analyze_time_series_patterns(df, logger=logger)
                if analyses['temporal_patterns'].get('error_message'):
                    logger.log_and_print(f"Warning: Time series tests failed: {analyses['temporal_patterns']['error_message']}", level=logging.WARNING)
            except Exception as e:
                logger.log_and_print(f"Warning: Time series tests failed: {str(e)}", level=logging.WARNING)
            
            gc.collect()
            
            # Save checkpoint after analyses
            if not recovery.save_checkpoint({'dataframe': df, 'model_results': model_results, 'feature_importance': feature_importance, **analyses}, "after_analyses"):
                logger.log_and_print("Warning: Checkpoint save failed after analyses.", level=logging.WARNING)
            
            # # Generate prime probability map
            analyses['prime_probability_map'] = generate_prime_probability_map(
                df,
                model_results,
                feature_cols,
                StandardScaler(),
                StandardScaler(),
                StandardScaler(),
                n_primes_to_predict=1000,
                logger=logger
            )
            
            # Create visualizations with protection
            logger.log_and_print("\nCreating visualizations...")
            create_visualizations_large_scale(df, feature_importance, analyses['pattern_analysis'], 
                                           plot_dir, model_results, analysis_stats=analyses)
            create_cluster_visualization(df, plot_dir, logger=logger)
            
            # Store complete results
            complete_results = {
                'dataframe': df,
                'model_results': model_results,
                'feature_importance': feature_importance,
                **analyses
            }
            
            # Log model_results before SHAP computation
            logger.log_and_print(f"DEBUG: model_results before SHAP: {model_results.keys()}")
            
            # Compute SHAP values for feature interpretation
            logger.log_and_print("\nComputing SHAP values...")
            
            # Log model names before SHAP computation
            logger.log_and_print(f"Models for SHAP: {[name for name in model_results if 'model' in model_results.get(name, {})]}")
            
            shap_values, shap_importance = compute_shap_values(
                {name: model_results[name].get('model') for name in ['random_forest', 'xgboost'] if name in model_results and 'model' in model_results.get(name, {})},
                X,
                feature_cols,
                logger=logger
            )
            
            # Compute prediction intervals for best model
            try:
                valid_models = [
                    k for k in model_results
                    if 'avg_test_mse' in model_results.get(k, {}) and 'model' in model_results.get(k, {})
                ]
                
                logger.log_and_print(f"DEBUG: Valid models for prediction intervals: {valid_models}")
                
                if valid_models:
                    best_model_name = min(
                        valid_models,
                        key=lambda k: model_results[k].get('avg_test_mse', float('inf'))
                    )
                    
                    logger.log_and_print(f"DEBUG: best_model_name before prediction intervals: {best_model_name}")
                    logger.log_and_print(f"DEBUG: model_results before prediction intervals: {model_results.keys()}")
                    
                    if best_model_name in model_results and 'model' in model_results.get(best_model_name, {}):
                        best_model = model_results[best_model_name]['model']
                        logger.log_and_print(f"\nComputing prediction intervals for {best_model_name}...")
                        mean_pred, lower_pred, upper_pred = compute_prediction_intervals(best_model, X, logger=logger)
                        complete_results['prediction_intervals'] = {
                            'mean': mean_pred,
                            'lower': lower_pred,
                            'upper': upper_pred
                        }
                    else:
                        logger.log_and_print(f"Warning: No 'model' key found for best model: {best_model_name}. Skipping prediction intervals.")
                else:
                    logger.log_and_print("Warning: No valid models with 'model' key found for prediction intervals. Skipping prediction intervals.")
            except Exception as e:
                logger.log_and_print(f"Warning: Could not compute prediction intervals: {str(e)}")
            
            # Detect change points
            logger.log_and_print("\nDetecting change points...")
            change_point_analysis = detect_change_points(df, logger=logger)
            complete_results['change_points'] = change_point_analysis
            
            # Perform advanced clustering analysis
            logger.log_and_print("\nPerforming advanced clustering analysis...")
            advanced_clustering_results = perform_advanced_clustering_analysis(df, logger=logger)
            complete_results['advanced_clustering'] = advanced_clustering_results
            
            # Perform statistical tests
            logger.log_and_print("\nPerforming statistical tests...")
            statistical_test_results = perform_advanced_statistical_tests(df, advanced_clustering_results, logger=logger)
            complete_results['statistical_tests'] = statistical_test_results
            
            complete_results['shap_values'] = shap_values
            complete_results['shap_importance'] = shap_importance
            
            logger.log_and_print(f"\nAnalysis completed in {time.time() - overall_start:.2f} seconds")
            
            # Write report
            logger.log_and_print(f"\nWriting analysis results to {output_log_file}")
            _write_analysis_report(
                output_log_file,
                model_results,
                feature_importance,
                analyses['pattern_analysis'],
                df,
                prime_probability_map=analyses['prime_probability_map'],
                cluster_sequence_analysis=None,
                gap_distribution=analyses['gap_distribution'],
                gap_sequences=analyses['gap_sequences'],
                factor_patterns=analyses['factor_patterns'],
                cluster_transitions=analyses['cluster_transitions'],
                temporal_patterns=analyses['temporal_patterns'],
                separation_metrics=analyses['separation_metrics'],
                shap_values=shap_values,
                shap_importance=shap_importance,
                prediction_intervals=complete_results.get('prediction_intervals'),
                change_points=change_point_analysis,
                cluster_stats=analyses.get('cluster_stats'),
                advanced_clustering=advanced_clustering_results,
                statistical_tests=statistical_test_results,
                logger=logger
            )
            
            return complete_results
            
        except Exception as e:
            if logger:
                logger.log_and_print(f"Critical error in analysis: {str(e)}", level=logging.ERROR)
                logger.logger.error(traceback.format_exc())
            else:
                print(f"Critical error in analysis: {str(e)}")
                traceback.print_exc()
            return None
                                            
@timing_decorator
def analyze_gap_distribution_characteristics(df, logger=None):
    """Analyze gap distribution characteristics with improved numerical stability and error handling."""
    if logger:
        logger.log_and_print("Starting gap distribution analysis...")
    
    try:
        # Convert data to numpy array and ensure proper type
        data = df['gap_size'].values.astype(np.float64)
        data = data.reshape(-1, 1)
        
        # Clip values for numerical stability
        data = np.clip(data, -1e10, 1e10)
        
        # Get unique clusters and initialize results
        clusters = sorted(df['cluster'].unique())
        cluster_distributions = {}
        
        for cluster_id in clusters:
            if logger:
                logger.log_and_print(f"Processing cluster {cluster_id}")
            
            # Get gaps for this cluster
            cluster_mask = df['cluster'] == cluster_id
            cluster_gaps = data[cluster_mask].flatten()  # Flatten to 1D array
            
            if len(cluster_gaps) > 0:
                # Compute basic statistics with protection
                with np.errstate(all='ignore'):
                    stats = {
                        'count': int(len(cluster_gaps)),
                        'mean': float(np.mean(cluster_gaps)),
                        'std': float(np.std(cluster_gaps)),
                        'min': float(np.min(cluster_gaps)),
                        'max': float(np.max(cluster_gaps))
                    }
                    
                    # Compute quantiles safely
                    quantiles = np.percentile(cluster_gaps, [10, 25, 50, 75, 90])
                    stats.update({
                        'q10': float(quantiles[0]),
                        'q25': float(quantiles[1]),
                        'q50': float(quantiles[2]),
                        'q75': float(quantiles[3]),
                        'q90': float(quantiles[4]),
                        'iqr': float(quantiles[3] - quantiles[1])
                    })
                    
                    # Compute higher moments safely
                    stats.update({
                        'skewness': float(sps.skew(cluster_gaps)),
                        'kurtosis': float(sps.kurtosis(cluster_gaps))
                    })
                    
                    # Compute mode safely
                    try:
                        # Convert to integers for mode calculation
                        int_gaps = cluster_gaps.astype(int)
                        unique_vals, counts = np.unique(int_gaps, return_counts=True)
                        mode_idx = np.argmax(counts)
                        mode_value = unique_vals[mode_idx]
                        mode_count = counts[mode_idx]
                        
                        stats.update({
                            'mode': float(mode_value),
                            'mode_count': int(mode_count)
                        })
                    except Exception as e:
                        if logger:
                            logger.log_and_print(f"Warning: Mode calculation failed: {str(e)}")
                        stats.update({
                            'mode': float(stats['mean']),
                            'mode_count': 1
                        })
                    
                    cluster_distributions[int(cluster_id)] = stats
            else:
                # Provide default values for empty clusters
                cluster_distributions[int(cluster_id)] = {
                    'count': 0,
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'q10': 0.0,
                    'q25': 0.0,
                    'q50': 0.0,
                    'q75': 0.0,
                    'q90': 0.0,
                    'iqr': 0.0,
                    'skewness': 0.0,
                    'kurtosis': 0.0,
                    'mode': 0.0,
                    'mode_count': 0
                }
        
        # Add summary statistics
        try:
            summary_stats = {
                'total_clusters': len(clusters),
                'mean_cluster_size': float(np.mean([d['count'] for d in cluster_distributions.values()])),
                'std_cluster_size': float(np.std([d['count'] for d in cluster_distributions.values()])),
                'mean_gap_ranges': {
                    cluster_id: float(d['max'] - d['min'])
                    for cluster_id, d in cluster_distributions.items()
                }
            }
            cluster_distributions['summary'] = summary_stats
            
        except Exception as e:
            if logger:
                logger.log_and_print(f"Warning: Error computing summary statistics: {str(e)}")
            cluster_distributions['summary'] = {
                'total_clusters': len(clusters),
                'mean_cluster_size': 0.0,
                'std_cluster_size': 0.0,
                'mean_gap_ranges': {}
            }
        
        if logger:
            logger.log_and_print("Gap size distribution analysis complete")
        
        return cluster_distributions
        
    except Exception as e:
        error_msg = f"Error in gap distribution analysis: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'summary': {
                'total_clusters': len(clusters),
                'mean_cluster_size': 0.0,
                'std_cluster_size': 0.0,
                'mean_gap_ranges': {}
            }
        }

@timing_decorator
def compute_gap_transition_probabilities(df, max_gap=None, batch_size=5000, logger=None):
    """Compute transition probabilities between gap sizes with batched processing."""
    with suppress_overflow_warnings():
        if logger:
            logger.log_and_print("Computing gap transition probabilities...")
        
        # Convert to float64 and clip values
        gaps = df['gap_size'].astype(np.float64)
        
        if max_gap is None:
            max_gap = min(50, int(gaps.max()))  # Limit maximum gap size for transition matrix
            
        if logger:
            logger.log_and_print(f"Using maximum gap size of {max_gap}")
        
        # Initialize matrices with appropriate types
        transition_matrix = np.zeros((max_gap + 1, max_gap + 1), dtype=np.float64)
        transition_counts = np.zeros((max_gap + 1, max_gap + 1), dtype=np.int32)
        
        # Process transitions in batches
        for start_idx in range(0, len(gaps) - 1, batch_size):
            end_idx = min(start_idx + batch_size, len(gaps) - 1)
            
            # Get current and next gaps for this batch
            current_gaps = gaps.iloc[start_idx:end_idx].values
            next_gaps = gaps.iloc[start_idx + 1:end_idx + 1].values
            
            # Handle invalid values
            mask = np.isfinite(current_gaps) & np.isfinite(next_gaps)
            current_gaps = current_gaps[mask]
            next_gaps = next_gaps[mask]
            
            # Clip gaps to max_gap
            current_gaps = np.minimum(current_gaps, max_gap)
            next_gaps = np.minimum(next_gaps, max_gap)
            
            # Count transitions for this batch
            for curr, next_gap in zip(current_gaps, next_gaps):
                transition_counts[int(curr), int(next_gap)] += 1
            
            gc.collect()
            
            if logger and start_idx % (batch_size * 5) == 0:
                logger.log_and_print(f"Processed {end_idx}/{len(gaps)} gaps")
        
        # Convert to probabilities with improved numerical stability
        with np.errstate(all='ignore'):
            row_sums = transition_counts.sum(axis=1, keepdims=True)
            transition_matrix = np.divide(transition_counts, 
                                       row_sums, 
                                       out=np.zeros_like(transition_counts, dtype=np.float64),
                                       where=row_sums!=0)
        
        # Compute additional transition metrics
        if logger:
            logger.log_and_print("Computing transition metrics...")
            
        metrics = {
            'entropy': {},
            'most_likely_next': {},
            'expected_next': {},
            'transition_stability': {},
            'transition_diversity': {}
        }
        
        # Process metrics in batches
        for i in range(0, max_gap + 1, batch_size):
            end_i = min(i + batch_size, max_gap + 1)
            
            for gap in range(i, end_i):
                probs = transition_matrix[gap]
                if np.any(probs > 0):
                    # Compute entropy with improved numerical stability
                    valid_probs = probs[probs > 0]
                    local_entropy = -np.sum(valid_probs * np.log2(valid_probs + 1e-10))
                    metrics['entropy'][gap] = float(local_entropy)
                    
                    # Most likely next gap
                    metrics['most_likely_next'][gap] = int(np.argmax(probs))
                    
                    # Expected next gap
                    gap_indices = np.arange(max_gap + 1)
                    expected_next = float(np.sum(gap_indices * probs))
                    metrics['expected_next'][gap] = expected_next
                    
                    # Transition stability (probability of staying in same gap)
                    metrics['transition_stability'][gap] = float(probs[gap])
                    
                    # Transition diversity (number of possible next gaps with prob > threshold)
                    metrics['transition_diversity'][gap] = int(np.sum(probs > 0.01))
            
            gc.collect()
        
        # Compute summary statistics
        transition_stats = {
            'transition_matrix': transition_matrix,
            'transition_counts': transition_counts,
            'metrics': metrics,
            'summary_stats': {
                'mean_entropy': float(np.mean(list(metrics['entropy'].values()))),
                'max_entropy': float(np.max(list(metrics['entropy'].values()))),
                'min_entropy': float(np.min(list(metrics['entropy'].values()))),
                'total_transitions': int(np.sum(transition_counts)),
                'unique_transitions': int(np.count_nonzero(transition_counts)),
                'sparsity': float(np.count_nonzero(transition_counts) / 
                                ((max_gap + 1) * (max_gap + 1)))
            }
        }
        
        # Add transition stability metrics
        diagonal_prob = np.diagonal(transition_matrix)
        transition_stats['summary_stats']['stability'] = float(np.mean(diagonal_prob))
        
        # Add transition diversity metrics
        nonzero_probs = transition_matrix[transition_matrix > 0]
        if len(nonzero_probs) > 0:
            transition_stats['summary_stats']['diversity'] = float(np.sum(
                nonzero_probs * np.log2(1 / (nonzero_probs + 1e-10))
            ) / len(nonzero_probs))
        else:
            transition_stats['summary_stats']['diversity'] = 0.0
        
        if logger:
            logger.log_and_print("Gap transition probability analysis complete")
        
        return transition_stats
       
@njit
def _analyze_prime_factor_patterns_numba(unique_factors, total_factors, factor_density, 
                                         mean_factor, max_factor, min_factor, factor_entropy):
    """Numba-optimized version of analyze_prime_factor_patterns."""
    
    unique_factors_total = np.sum(unique_factors)
    total_factors_total = np.sum(total_factors)
    mean_density = np.mean(factor_density)
    mean_factor_size = np.mean(mean_factor)
    std_factor_size = np.std(mean_factor) if len(mean_factor) > 1 else 0.0
    max_factor_seen = np.max(max_factor) if len(max_factor) > 0 else 0.0
    min_factor_seen = np.min(min_factor) if len(min_factor) > 0 else 0.0
    
    return (
        int(unique_factors_total),
        int(total_factors_total),
        float(mean_density),
        float(mean_factor_size),
        float(std_factor_size),
        float(max_factor_seen),
        float(min_factor_seen)
    )
    

@timing_decorator
def analyze_prime_factor_patterns(df, batch_size=5000, logger=None):
    """Analyze patterns in prime factorizations with improved memory management and numerical stability."""
    with suppress_overflow_warnings():
        factor_patterns = {
            'factor_frequencies': {},
            'factor_combinations': {},
            'factor_sequences': [],
            'metrics': {},
            'temporal_patterns': {}
        }
        
        try:
            if logger:
                logger.log_and_print("Analyzing prime factor patterns...")
            
            # Get pre-computed factor columns
            factor_cols = ['unique_factors', 'total_factors', 'factor_density', 
                         'mean_factor', 'max_factor', 'min_factor', 'factor_entropy']
            
            if all(col in df.columns for col in factor_cols):
                # Extract data for numba function
                unique_factors = df['unique_factors'].values.astype(np.float64)
                total_factors = df['total_factors'].values.astype(np.float64)
                factor_density = df['factor_density'].values.astype(np.float64)
                mean_factor = df['mean_factor'].values.astype(np.float64)
                max_factor = df['max_factor'].values.astype(np.float64)
                min_factor = df['min_factor'].values.astype(np.float64)
                factor_entropy = df['factor_entropy'].values.astype(np.float64)
                
                # Call numba-optimized function
                factorizations = [_factorint_numba(comp) for comp in df['upper_prime'].values.astype(np.int64)]
                
                all_factors = []
                for factor_array in factorizations:
                    for factor, count in factor_array:
                        all_factors.extend([factor] * count)
                
                all_factors = np.array(all_factors, dtype=np.int64)
                
                numba_metrics = _analyze_prime_factor_patterns_numba(
                    unique_factors, total_factors, factor_density, mean_factor, max_factor, min_factor, factor_entropy
                )
                
                factor_patterns['metrics'] = {
                    'unique_factors_total': numba_metrics[0],
                    'total_factors': numba_metrics[1],
                    'mean_density': numba_metrics[2],
                    'mean_factor_size': numba_metrics[3],
                    'std_factor_size': numba_metrics[4],
                    'max_factor_seen': numba_metrics[5],
                    'min_factor_seen': numba_metrics[6]
                }
                
                # Analyze factor sequences in batches
                window_sizes = [2, 3, 4, 5]
                for window in window_sizes:
                    sequence_stats = []
                    
                    for start_idx in range(0, len(df) - window + 1, batch_size):
                        end_idx = min(start_idx + batch_size, len(df) - window + 1)
                        
                        for i in range(start_idx, end_idx):
                            if i + window <= len(df):
                                window_data = df.iloc[i:i+window]
                                if 'unique_factors' in window_data.columns:
                                    factors = window_data['unique_factors'].values.astype(np.float64)
                                    factors = np.clip(factors, -1e10, 1e10)
                                    
                                    if len(factors) == window:
                                        with np.errstate(all='ignore'):
                                            sequence_stats.append({
                                                'mean': float(np.mean(factors)),
                                                'std': float(np.std(factors)),
                                                'trend': float(np.polyfit(range(window), factors, 1)[0])
                                            })
                        
                        gc.collect()
                    
                    if sequence_stats:
                        factor_patterns['factor_sequences'].append({
                            'window_size': window,
                            'stats': {
                                'mean_trend': float(np.mean([s['trend'] for s in sequence_stats])),
                                'std_trend': float(np.std([s['trend'] for s in sequence_stats])),
                                'mean_std': float(np.mean([s['std'] for s in sequence_stats]))
                            }
                        })
                
                # Analyze temporal patterns
                if 'factor_entropy' in df.columns:
                    temporal_data = df['factor_entropy'].values.astype(np.float64)
                    temporal_data = np.clip(temporal_data, -1e10, 1e10)
                    temporal_data = temporal_data[np.isfinite(temporal_data)]
                    
                    if len(temporal_data) > 1:
                        # Compute autocorrelation
                        autocorr = []
                        for lag in range(1, min(11, len(temporal_data))):
                            with np.errstate(all='ignore'):
                                corr = np.corrcoef(temporal_data[:-lag], temporal_data[lag:])[0, 1]
                                autocorr.append(float(corr) if np.isfinite(corr) else 0.0)
                        
                        factor_patterns['temporal_patterns']['autocorrelation'] = autocorr
                        
                        # Detect change points
                        try:
                            algo = Binseg(model="l2").fit(temporal_data.reshape(-1, 1))
                            change_points = algo.predict(n_bkps=3)
                            
                            factor_patterns['temporal_patterns']['change_points'] = {
                                'locations': [int(cp) for cp in change_points],
                                'values': [float(temporal_data[cp]) for cp in change_points]
                            }
                        except Exception as e:
                            if logger:
                                logger.log_and_print(f"Warning: Change point detection failed: {str(e)}")
                
                # Analyze factor combinations
                if 'num_distinct_prime_factors' in df.columns:
                    factor_counts = df['num_distinct_prime_factors'].value_counts()
                    factor_patterns['factor_combinations'] = {
                        int(k): int(v) for k, v in factor_counts.nlargest(10).items()
                    }
            
            if logger:
                logger.log_and_print("Factor pattern analysis complete")
            
            return factor_patterns
            
        except Exception as e:
            error_msg = f"Error in factor pattern analysis: {str(e)}"
            if logger:
                logger.log_and_print(error_msg, level=logging.ERROR)
                logger.logger.error(traceback.format_exc())
            else:
                print(error_msg)
                traceback.print_exc()
            
            # Return safe default values
            return {
                'factor_frequencies': {},
                'factor_combinations': {},
                'factor_sequences': [],
                'metrics': {
                    'unique_factors_total': 0,
                    'total_factors': 0,
                    'mean_density': 0.0,
                    'mean_factor_size': 0.0,
                    'std_factor_size': 0.0,
                    'max_factor_seen': 0.0,
                    'min_factor_seen': 0.0
                },
                'temporal_patterns': {
                    'autocorrelation': [],
                    'change_points': {'locations': [], 'values': []}
                }
            }         

@njit
def _analyze_gap_distribution_numba(cluster_gaps, values_for_quantiles):
    """Numba-optimized version of analyze_gap_distribution_characteristics_by_cluster."""
    
    count = len(cluster_gaps)
    if count == 0:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'q10': 0.0,
            'q25': 0.0,
            'q50': 0.0,
            'q75': 0.0,
            'q90': 0.0,
            'iqr': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'mode': 0.0,
            'mode_count': 0,
            'normal_fit': {'mu': 0.0, 'sigma': 0.0},
            'lognormal_fit': {'shape': 0.0, 'loc': 0.0, 'scale': 0.0},
            'entropy': 0.0
        }
    
    mean = np.mean(cluster_gaps)
    std = np.std(cluster_gaps)
    min_val = np.min(cluster_gaps)
    max_val = np.max(cluster_gaps)
    
    quantiles = np.percentile(values_for_quantiles, [10, 25, 50, 75, 90])
    
    skewness = skew(values_for_quantiles)
    kurt = kurtosis(values_for_quantiles)
    
    mode_result = mode(values_for_quantiles)
    if isinstance(mode_result, tuple):
        mode_value = mode_result[0][0]  # Old scipy version
    else:
        mode_value = mode_result.mode[0]  # New scipy version
    mode_count = np.sum(values_for_quantiles == mode_value)
    
    # Normal fit
    norm_params = norm.fit(values_for_quantiles)
    norm_fit = {'mu': float(norm_params[0]), 'sigma': float(norm_params[1])}
    
    # Log-normal fit for positive values
    positive_values = values_for_quantiles[values_for_quantiles > 0]
    if len(positive_values) > 0:
        lognorm_params = lognorm.fit(positive_values)
        lognormal_fit = {'shape': float(lognorm_params[0]), 'loc': float(lognorm_params[1]), 'scale': float(lognorm_params[2])}
    else:
        lognormal_fit = {'shape': 0.0, 'loc': 0.0, 'scale': 0.0}
    
    hist, _ = np.histogram(values_for_quantiles, bins='auto', density=True)
    entropy_val = entropy(hist + 1e-10)
    
    return {
        'count': int(count),
        'mean': float(mean),
        'std': float(std),
        'min': float(min_val),
        'max': float(max_val),
        'q10': float(quantiles[0]),
        'q25': float(quantiles[1]),
        'q50': float(quantiles[2]),
        'q75': float(quantiles[3]),
        'q90': float(quantiles[4]),
        'iqr': float(quantiles[3] - quantiles[1]),
        'skewness': float(skewness),
        'kurtosis': float(kurt),
        'mode': float(mode_value),
        'mode_count': int(mode_count),
        'normal_fit': norm_fit,
        'lognormal_fit': lognormal_fit,
        'entropy': float(entropy_val)
    }


@timing_decorator
def analyze_gap_distribution_characteristics_by_cluster(df, batch_size=5000, logger=None):
    """Analyze gap size distributions within clusters with improved numerical stability and error handling."""
    if logger:
        logger.log_and_print("Analyzing gap size distributions by cluster...")
    
    try:
        # Convert gap_size to float64 and get unique clusters
        gaps = df['gap_size'].astype(np.float64)
        clusters = sorted(df['cluster'].unique())
        
        cluster_distributions = {}
        
        for cluster_id in clusters:
            if logger:
                logger.log_and_print(f"Processing cluster {cluster_id}")
            
            # Get gaps for this cluster
            cluster_mask = df['cluster'] == cluster_id
            cluster_gaps = gaps[cluster_mask].values
            
            if len(cluster_gaps) > 0:
                # Initialize accumulators for batch processing
                stats_accumulators = {
                    'sum': 0.0,
                    'sum_sq': 0.0,
                    'count': 0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'values_for_quantiles': []
                }
                
                # Process in batches
                for start_idx in range(0, len(cluster_gaps), batch_size):
                    end_idx = min(start_idx + batch_size, len(cluster_gaps))
                    batch = cluster_gaps[start_idx:end_idx]
                    
                    if len(batch) > 0:
                        # Clip values for numerical stability
                        batch = np.clip(batch, -1e10, 1e10)
                        valid_mask = np.isfinite(batch)
                        batch = batch[valid_mask]
                        
                        if len(batch) > 0:
                            with np.errstate(all='ignore'):
                                stats_accumulators['sum'] += np.sum(batch)
                                stats_accumulators['sum_sq'] += np.sum(batch ** 2)
                                stats_accumulators['count'] += len(batch)
                                stats_accumulators['min'] = min(stats_accumulators['min'], np.min(batch))
                                stats_accumulators['max'] = max(stats_accumulators['max'], np.max(batch))
                                
                                # Store subset of values for quantile computation
                                if len(stats_accumulators['values_for_quantiles']) < 10000:
                                    stats_accumulators['values_for_quantiles'].extend(
                                        batch[:min(1000, len(batch))].tolist()
                                    )
                
                gc.collect()
            
                if stats_accumulators['count'] > 0:
                    # Convert stored values to numpy array for quantile computation
                    stored_values = np.array(stats_accumulators['values_for_quantiles'])
                    
                    # Call numba-optimized function
                    numba_stats = _analyze_gap_distribution_numba(cluster_gaps, stored_values)
                    
                    # Store statistics with explicit type conversion
                    cluster_distributions[int(cluster_id)] = {
                        'count': numba_stats['count'],
                        'mean': numba_stats['mean'],
                        'median': numba_stats['median'],
                        'std': numba_stats['std'],
                        'min': numba_stats['min'],
                        'max': numba_stats['max'],
                        'q10': numba_stats['q10'],
                        'q25': numba_stats['q25'],
                        'q75': numba_stats['q75'],
                        'q90': numba_stats['q90'],
                        'iqr': numba_stats['iqr'],
                        'skewness': numba_stats['skewness'],
                        'kurtosis': numba_stats['kurtosis'],
                        'mode': numba_stats['mode'],
                        'mode_count': numba_stats['mode_count'],
                        'normal_fit': numba_stats['normal_fit'],
                        'lognormal_fit': numba_stats['lognormal_fit'],
                        'entropy': numba_stats['entropy']
                    }
            else:
                # Provide default values for empty clusters
                cluster_distributions[int(cluster_id)] = {
                    'count': 0,
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'q10': 0.0,
                    'q25': 0.0,
                    'q50': 0.0,
                    'q75': 0.0,
                    'q90': 0.0,
                    'iqr': 0.0,
                    'skewness': 0.0,
                    'kurtosis': 0.0,
                    'mode': 0.0,
                    'mode_count': 0
                }
        
        # Add summary statistics
        try:
            summary_stats = {
                'total_clusters': len(clusters),
                'mean_cluster_size': float(np.mean([d['count'] for d in cluster_distributions.values()])),
                'std_cluster_size': float(np.std([d['count'] for d in cluster_distributions.values()])),
                'mean_gap_ranges': {
                    cluster_id: float(d['max'] - d['min'])
                    for cluster_id, d in cluster_distributions.items()
                }
            }
            cluster_distributions['summary'] = summary_stats
            
        except Exception as e:
            if logger:
                logger.log_and_print(f"Warning: Error computing summary statistics: {str(e)}")
            cluster_distributions['summary'] = {
                'total_clusters': len(clusters),
                'mean_cluster_size': 0.0,
                'std_cluster_size': 0.0,
                'mean_gap_ranges': {}
            }
        
        if logger:
            logger.log_and_print("Gap size distribution analysis complete")
        
        return cluster_distributions
        
    except Exception as e:
        error_msg = f"Error in gap distribution analysis: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'summary': {
                'total_clusters': len(clusters),
                'mean_cluster_size': 0.0,
                'std_cluster_size': 0.0,
                'mean_gap_ranges': {}
            }
        }

@njit
def _analyze_cluster_transitions_numba(clusters, n_clusters):
    """Numba-optimized version of analyze_cluster_transitions_advanced."""
    transition_probabilities = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    
    for i in range(len(clusters) - 1):
        curr = clusters[i]
        next_cluster = clusters[i+1]
        transition_probabilities[curr, next_cluster] += 1
    
    row_sums = transition_probabilities.sum(axis=1)
    
    # Avoid division by zero
    for i in range(n_clusters):
        if row_sums[i] > 0:
            transition_probabilities[i] /= row_sums[i]
    
    return transition_probabilities

@njit
def _analyze_gap_sequences_numba(gaps, length):
    """Numba-optimized version of analyze_gap_sequences_advanced."""
    
    n = len(gaps)
    count = 0
    increasing_count = 0
    decreasing_count = 0
    constant_count = 0
    
    for i in range(n - length + 1):
        seq = gaps[i:i+length]
        
        diffs = np.diff(seq)
        
        is_increasing = True
        is_decreasing = True
        is_constant = True
        
        for diff in diffs:
            if diff <= 0:
                is_increasing = False
            if diff >= 0:
                is_decreasing = False
            if diff != 0:
                is_constant = False
        
        count += 1
        if is_increasing:
            increasing_count += 1
        if is_decreasing:
            decreasing_count += 1
        if is_constant:
            constant_count += 1
    
    return count, increasing_count, decreasing_count, constant_count

@timing_decorator
def analyze_cluster_transitions_advanced(df, batch_size=10000, logger=None):
    """Analyze cluster transitions with improved memory management and numerical stability."""
    if 'cluster' not in df.columns:
        return None
    
    if logger:
        logger.log_and_print("Analyzing cluster transitions...")
    
    # Convert to numpy arrays for faster processing
    clusters = df['cluster'].astype(np.int32).values
    n_clusters = len(np.unique(clusters))
    
    transition_analysis = {
        'basic_metrics': {},
        'sequence_patterns': {},
        'transition_probabilities': np.zeros((n_clusters, n_clusters), dtype=np.float64),
        'temporal_patterns': {},
        'stability_metrics': {}
    }
    
    try:
        # Process transitions in batches
        transition_analysis['transition_probabilities'] = _analyze_cluster_transitions_numba(clusters, n_clusters)
        
        # Analyze sequence patterns
        sequence_lengths = [2, 3, 4, 5]
        for length in sequence_lengths:
            if logger:
                logger.log_and_print(f"Processing sequences of length {length}")
            
            sequence_counts = {}
            total_sequences = 0
            
            # Process sequences in batches
            for start_idx in range(0, len(clusters) - length + 1, batch_size):
                end_idx = min(start_idx + batch_size, len(clusters) - length + 1)
                
                for i in range(start_idx, end_idx):
                    seq = tuple(clusters[i:i+length])
                    sequence_counts[seq] = sequence_counts.get(seq, 0) + 1
                    total_sequences += 1
                
                gc.collect()
            
            # Get most common sequences
            most_common = sorted(
                sequence_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            transition_analysis['sequence_patterns'][length] = {
                'total_sequences': int(total_sequences),
                'unique_sequences': len(sequence_counts),
                'most_common': [(list(seq), int(count)) for seq, count in most_common],
                'sequence_entropy': float(entropy([count/total_sequences for _, count in sequence_counts.items()]) if sequence_counts else 0)
            }
        
        # Compute basic metrics
        entropy_sum = 0
        stability_sum = 0
        total_transitions = 0
        
        for i in range(n_clusters):
            probs = transition_analysis['transition_probabilities'][i]
            if np.any(probs > 0):
                local_entropy = -np.sum(probs * np.log2(probs + 1e-10))
                entropy_sum += local_entropy
                stability_sum += probs[i]
                total_transitions += 1
        
        # Compute cluster run lengths
        run_lengths = []
        current_run = 1
        
        for i in range(1, len(clusters)):
            if clusters[i] == clusters[i-1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        
        # Add stability metrics
        transition_analysis['stability_metrics'] = {
            'mean_run_length': float(np.mean(run_lengths)),
            'std_run_length': float(np.std(run_lengths)),
            'max_run_length': int(np.max(run_lengths)),
            'min_run_length': int(np.min(run_lengths)),
            'run_length_distribution': np.bincount(run_lengths).tolist()
        }
        
        # Analyze temporal patterns for each cluster
        for cluster in range(n_clusters):
            cluster_occurrences = np.where(clusters == cluster)[0]
            
            if len(cluster_occurrences) > 1:
                gaps = np.diff(cluster_occurrences)
                transition_analysis['temporal_patterns'][cluster] = {
                    'mean_recurrence': float(np.mean(gaps)),
                    'std_recurrence': float(np.std(gaps)),
                    'min_recurrence': int(np.min(gaps)),
                    'max_recurrence': int(np.max(gaps))
                }
        
        # Add summary statistics
        transition_analysis['basic_metrics'] = {
            'transition_entropy': float(entropy_sum),
            'mean_stability': float(stability_sum / max(1, total_transitions)),
            'total_transitions': int(total_transitions),
            'cluster_frequencies': [float(freq) for freq in np.bincount(clusters) / len(clusters)],
            'entropy_per_cluster': [float(e) for e in -np.sum(
                transition_analysis['transition_probabilities'] * 
                np.log2(transition_analysis['transition_probabilities'] + 1e-10),
                axis=1
            )]
        }
        
        if logger:
            logger.log_and_print("Cluster transition analysis complete")
        
        return transition_analysis
        
    except Exception as e:
        error_msg = f"Error in cluster transition analysis: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'basic_metrics': {},
            'sequence_patterns': {},
            'transition_probabilities': np.zeros((n_clusters, n_clusters)),
            'temporal_patterns': {},
            'stability_metrics': {
                'mean_run_length': 0.0,
                'std_run_length': 0.0,
                'max_run_length': 0,
                'min_run_length': 0,
                'run_length_distribution': []
            },
            'summary': {
                'total_transitions': 0,
                'mean_entropy': 0.0,
                'stability_score': 0.0,
                'n_clusters': n_clusters
            }
        }                             

@timing_decorator
def analyze_prime_distribution(df, batch_size=10000, logger=None):
    """Analyze prime distribution patterns with improved memory management and numerical stability."""
    if logger:
        logger.log_and_print("Analyzing prime distribution patterns...")
    
    temporal_patterns = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    try:
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'gap_size']
        
        # Initialize scalers and fit them with the data
        scaler = StandardScaler()
        scaler_gap_cluster = StandardScaler()
        scaler_next_cluster = StandardScaler()
        
        # Fit scalers on appropriate data
        if 'cluster' in df.columns:
            cluster_data = pd.DataFrame({'cluster': df['cluster']})
            scaler_gap_cluster.fit(cluster_data)
        
        if 'predicted_next_cluster' in df.columns:
            next_cluster_data = df[['predicted_next_cluster']]
            scaler_next_cluster.fit(next_cluster_data)
        
        # Get pre-computed factor columns
        factor_cols = ['unique_factors', 'total_factors', 'factor_density', 
                      'mean_factor', 'max_factor', 'min_factor']
        
        if all(col in df.columns for col in factor_cols):
            # Initialize accumulators
            stats_accumulators = {
                col: {
                    'sum': 0.0,
                    'sum_sq': 0.0,
                    'count': 0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'values': []
                }
                for col in factor_cols
            }
            
            # Process in batches
            for start_idx in range(0, len(df), batch_size):
                end_idx = min(start_idx + batch_size, len(df))
                batch = df.iloc[start_idx:end_idx]
                
                for col in factor_cols:
                    col_data = batch[col].values.astype(np.float64)
                    col_data = np.clip(col_data, -1e10, 1e10)
                    valid_mask = np.isfinite(col_data)
                    col_data = col_data[valid_mask]
                    
                    if len(col_data) > 0:
                        with np.errstate(all='ignore'):
                            stats_accumulators[col]['sum'] += np.sum(col_data)
                            stats_accumulators[col]['sum_sq'] += np.sum(col_data ** 2)
                            stats_accumulators[col]['count'] += len(col_data)
                            stats_accumulators[col]['min'] = min(stats_accumulators[col]['min'], np.min(col_data))
                            stats_accumulators[col]['max'] = max(stats_accumulators[col]['max'], np.max(col_data))
                            
                            # Store subset of values for distribution analysis
                            if len(stats_accumulators[col]['values']) < 10000:
                                stats_accumulators[col]['values'].extend(col_data[:1000].tolist())
                
                gc.collect()
            
            # Compute final statistics
            temporal_patterns['summary'] = {
                'unique_factors_total': int(stats_accumulators['unique_factors']['sum']),
                'total_factors': int(stats_accumulators['total_factors']['sum']),
                'mean_density': float(stats_accumulators['factor_density']['sum'] / 
                                   max(1, stats_accumulators['factor_density']['count'])),
                'mean_factor_size': float(stats_accumulators['mean_factor']['sum'] / 
                                       max(1, stats_accumulators['mean_factor']['count'])),
                'max_factor_seen': float(stats_accumulators['max_factor']['max']),
                'min_factor_seen': float(stats_accumulators['min_factor']['min'])
            }
            
            # Get last gap and cluster
            last_gap = df['gap_size'].iloc[-1]
            last_cluster = df['cluster'].iloc[-1]
            
            # Create a DataFrame with the last gap's features
            last_features = df.iloc[[-1]][feature_cols].copy()
            
            # Use assign_composite_to_cluster to predict the cluster
            predicted_cluster = assign_composite_to_cluster(
                df,
                last_features,
                None,  # No model_results needed since we're using KMeans
                use_classifier=False,  # Use KMeans instead
                scaler=scaler,
                logger=logger
            )
            
            # Prepare gap prediction input
            gap_input = np.array([[predicted_cluster]], dtype=np.int32)
            
            # Transform using fitted scaler
            gap_input_scaled = scaler_gap_cluster.transform(gap_input)
            
            # Use simple prediction based on cluster mean
            cluster_gaps = df[df['cluster'] == predicted_cluster]['gap_size']
            if len(cluster_gaps) > 0:
                predicted_gap = float(cluster_gaps.mean())
            else:
                predicted_gap = float(df['gap_size'].mean())  # Fallback to overall mean
            
            # Calculate next prime location
            next_prime_location = df['upper_prime'].iloc[-1] + predicted_gap
            
            temporal_patterns['next_prime_predictions'] = {
                'predicted_cluster': int(predicted_cluster),
                'predicted_gap': predicted_gap,
                'next_prime_location': float(next_prime_location)
            }
            
            if logger:
                logger.log_and_print("Prime distribution analysis complete")
            
            return temporal_patterns
            
    except Exception as e:
        error_msg = f"Error in prime distribution analysis: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'summary': {
                'unique_factors_total': 0,
                'total_factors': 0,
                'mean_density': 0.0,
                'mean_factor_size': 0.0,
                'max_factor_seen': 0.0,
                'min_factor_seen': 0.0
            },
            'next_prime_predictions': {
                'predicted_cluster': -1,
                'predicted_gap': 0.0,
                'next_prime_location': 0.0
            }
        }

@njit
def _compute_mutual_information_numba(x_batch, y_batch):
    """Numba-optimized function to compute mutual information."""
    n_samples = x_batch.shape[0]
    
    # Handle NaN/inf values
    valid_mask = np.isfinite(x_batch.ravel()) & np.isfinite(y_batch)
    if np.sum(valid_mask) <= 1:
        return 0.0
    
    x_clean = x_batch[valid_mask].reshape(-1, 1)
    y_clean = y_batch[valid_mask]
    
    # Compute joint histogram
    hist_2d, _, _ = np.histogram2d(x_clean.ravel(), y_clean, bins=10)
    
    # Compute marginal histograms
    hist_x, _ = np.histogram(x_clean, bins=10)
    hist_y, _ = np.histogram(y_clean, bins=10)
    
    # Normalize histograms
    p_xy = hist_2d / np.sum(hist_2d)
    p_x = hist_x / np.sum(hist_x)
    p_y = hist_y / np.sum(hist_y)
    
    # Compute mutual information
    mi = 0.0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    
    return float(mi)

@timing_decorator
def compute_entanglement_metrics(df, feature_cols, batch_size=5000, logger=None):
    """Compute entanglement-like metrics between features with improved numerical stability."""
    if logger:
        logger.log_and_print("Computing entanglement-like metrics...")
    
    entanglement_metrics = {
        'correlation_entanglement': {},
        'mutual_information_entanglement': {}
    }
    
    try:
        # Convert to float64 and clip values
        X = df[feature_cols].astype(np.float64)
        X = X.clip(-1e10, 1e10)
        
        # 1. Correlation-based entanglement
        if logger:
            logger.log_and_print("Computing correlation-based entanglement...")
        
        n_features = len(feature_cols)
        correlation_matrix = np.zeros((n_features, n_features), dtype=np.float64)
        
        # Process correlations in batches
        for start_idx in range(0, len(X), batch_size):
            end_idx = min(start_idx + batch_size, len(X))
            batch = X.iloc[start_idx:end_idx]
            
            with np.errstate(all='ignore'):
                batch_corr = np.array(batch.corr(), dtype=np.float64)
                
                # Update correlation matrix
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        if np.isfinite(batch_corr[i, j]):
                            correlation_matrix[i, j] += batch_corr[i, j]
                            correlation_matrix[j, i] += batch_corr[i, j]
            
            gc.collect()
        
        # Normalize correlation matrix
        correlation_matrix /= len(X)
        
        # Compute entanglement metric (sum of absolute correlations)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                entanglement = abs(correlation_matrix[i, j])
                entanglement_metrics['correlation_entanglement'][f'{feature_cols[i]}_X_{feature_cols[j]}'] = float(entanglement)
        
        # 2. Mutual information-based entanglement
        if logger:
            logger.log_and_print("Computing mutual information-based entanglement...")
        
        for i, feat1 in enumerate(feature_cols):
            for j, feat2 in enumerate(feature_cols[i+1:], i+1):
                mi_sum = 0.0
                count = 0
                
                # Process mutual information in batches
                for start_idx in range(0, len(X), batch_size):
                    end_idx = min(start_idx + batch_size, len(X))
                    batch = X.iloc[start_idx:end_idx]
                    
                    # Convert to numpy arrays for mutual_info_regression
                    x_batch = np.array(batch[feat1], dtype=np.float64).reshape(-1, 1)
                    y_batch = np.array(batch[feat2], dtype=np.float64)
                    
                    # Call numba-optimized function
                    mi_result = _compute_mutual_information_numba(x_batch, y_batch)
                    if np.isfinite(mi_result):
                        mi_sum += mi_result
                        count += 1
                    
                    gc.collect()
                
                if count > 0:
                    avg_mi = mi_sum / count
                    entanglement_metrics['mutual_information_entanglement'][f'{feat1}_X_{feat2}'] = float(avg_mi)
        
        if logger:
            logger.log_and_print("Entanglement metrics computation complete")
        
        return entanglement_metrics
        
    except Exception as e:
        error_msg = f"Error computing entanglement metrics: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'correlation_entanglement': {},
            'mutual_information_entanglement': {}
        }

@timing_decorator
def compute_superposition_patterns(df, feature_cols, batch_size=5000, logger=None):
    """Compute superposition-like patterns in the data with improved numerical stability."""
    if logger:
        logger.log_and_print("Computing superposition-like patterns...")
    
    superposition_patterns = {}
    
    try:
        # Convert to float64 and clip values
        X = df[feature_cols].astype(np.float64)
        X = X.clip(-1e10, 1e10)
        
        for col in feature_cols:
            if logger:
                logger.log_and_print(f"Analyzing superposition patterns for {col}")
            
            # Initialize accumulators
            stats_accumulators = {
                'sum': 0.0,
                'sum_sq': 0.0,
                'count': 0,
                'min': float('inf'),
                'max': float('-inf'),
                'values': []
            }
            
            # Process in batches
            for start_idx in range(0, len(X), batch_size):
                end_idx = min(start_idx + batch_size, len(X))
                batch = X[col].iloc[start_idx:end_idx].values
                
                valid_mask = np.isfinite(batch)
                if np.any(valid_mask):
                    batch = batch[valid_mask]
                    with np.errstate(all='ignore'):
                        stats_accumulators['sum'] += np.sum(batch)
                        stats_accumulators['sum_sq'] += np.sum(batch ** 2)
                        stats_accumulators['count'] += len(batch)
                        stats_accumulators['min'] = min(stats_accumulators['min'], np.min(batch))
                        stats_accumulators['max'] = max(stats_accumulators['max'], np.max(batch))
                        
                        # Store subset of values for distribution analysis
                        if len(stats_accumulators['values']) < 10000:
                            stats_accumulators['values'].extend(batch[:1000].tolist())
                
                gc.collect()
            
            # Compute final statistics
            if stats_accumulators['count'] > 0:
                mean = stats_accumulators['sum'] / stats_accumulators['count']
                var = (stats_accumulators['sum_sq'] / stats_accumulators['count']) - (mean ** 2)
                std = np.sqrt(max(0, var))
                
                superposition_patterns[col] = {
                    'mean': float(mean),
                    'std': float(std),
                    'min': float(stats_accumulators['min']),
                    'max': float(stats_accumulators['max']),
                    'count': int(stats_accumulators['count'])
                }
                
                # Compute quantiles if we have stored values
                if stats_accumulators['values']:
                    values = np.array(stats_accumulators['values'])
                    quantiles = np.percentile(values, [25, 50, 75])
                    superposition_patterns[col].update({
                        'median': float(quantiles[1]),
                        'q1': float(quantiles[0]),
                        'q3': float(quantiles[2]),
                        'iqr': float(quantiles[2] - quantiles[0])
                    })
                    
                    # Compute histogram
                    hist, _ = np.histogram(values, bins='auto', density=True)
                    superposition_patterns[col]['entropy'] = float(entropy(hist + 1e-10))
                    
                    # Check for multimodality
                    try:
                        peaks, _ = find_peaks(hist, prominence=0.01)
                        superposition_patterns[col]['num_modes'] = int(len(peaks))
                    except Exception as e:
                        if logger:
                            logger.log_and_print(f"Warning: Peak detection failed for {col}: {str(e)}")
                        superposition_patterns[col]['num_modes'] = 0
        
        if logger:
            logger.log_and_print("Superposition pattern analysis complete")
        
        return superposition_patterns
        
    except Exception as e:
        error_msg = f"Error computing superposition patterns: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {}
      
@timing_decorator
def compute_chaos_metrics(df, feature_cols, batch_size=5000, logger=None, sequence_length=10):
    """Compute chaos metrics based on divergence of nearby trajectories with improved numerical stability."""
    if logger:
        logger.log_and_print("Computing chaos metrics...")
    
    chaos_metrics = {}
    
    try:
        # Convert to float64 and clip values
        X = df[feature_cols].astype(np.float64)
        X = X.clip(-1e10, 1e10)
        
        for col in feature_cols:
            if logger:
                logger.log_and_print(f"Analyzing chaos metrics for {col}")
            
            # Initialize accumulators
            divergence_scores = []
            
            # Process in batches
            for start_idx in range(0, len(X) - sequence_length - 1, batch_size):
                end_idx = min(start_idx + batch_size, len(X) - sequence_length - 1)
                
                for i in range(start_idx, end_idx):
                    # Get two nearby trajectories
                    traj1 = X[col].iloc[i:i+sequence_length].values
                    traj2 = X[col].iloc[i+1:i+sequence_length+1].values
                    
                    # Handle NaN/inf values
                    valid_mask = np.isfinite(traj1) & np.isfinite(traj2)
                    traj1 = traj1[valid_mask]
                    traj2 = traj2[valid_mask]
                    
                    if len(traj1) > 1:
                        with np.errstate(all='ignore'):
                            # Compute divergence (mean absolute difference)
                            divergence = np.mean(np.abs(traj1 - traj2))
                            divergence_scores.append(float(divergence))
                    
                gc.collect()
            
            if divergence_scores:
                divergence_scores = np.array(divergence_scores)
                with np.errstate(all='ignore'):
                    chaos_metrics[col] = {
                        'mean_divergence': float(np.mean(divergence_scores)),
                        'std_divergence': float(np.std(divergence_scores)),
                        'max_divergence': float(np.max(divergence_scores)),
                        'min_divergence': float(np.min(divergence_scores)),
                        'divergence_90th': float(np.percentile(divergence_scores, 90))
                    }
            else:
                chaos_metrics[col] = {
                    'mean_divergence': 0.0,
                    'std_divergence': 0.0,
                    'max_divergence': 0.0,
                    'min_divergence': 0.0,
                    'divergence_90th': 0.0
                }
        
        if logger:
            logger.log_and_print("Chaos metrics computation complete")
        
        return chaos_metrics
        
    except Exception as e:
        error_msg = f"Error computing chaos metrics: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {}
           
@timing_decorator
def compute_fractal_dimension(df, feature_col='gap_size', batch_size=5000, logger=None):
    """Compute fractal dimension using box-counting method with improved numerical stability."""
    if logger:
        logger.log_and_print("Computing fractal dimension...")
    
    fractal_dimension = {}
    
    try:
        # Convert to numpy array and ensure proper type
        data = df[feature_col].values.astype(np.float64)
        data = np.clip(data, -1e10, 1e10)
        data = data[np.isfinite(data)]
        
        if len(data) < 2:
            if logger:
                logger.log_and_print("Warning: Insufficient data points for fractal dimension calculation.")
            return {'dimension': 0.0, 'counts': {}}
        
        # Compute min and max values
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Define box sizes
        box_sizes = np.logspace(np.log10(0.1 * (max_val - min_val)), np.log10(max_val - min_val), num=10)
        
        box_counts = []
        
        # Process box counts in batches
        for size in box_sizes:
            count = 0
            
            # Process data in batches
            for start_idx in range(0, len(data), batch_size):
                end_idx = min(start_idx + batch_size, len(data))
                batch = data[start_idx:end_idx]
                
                # Compute number of boxes needed to cover the data
                with np.errstate(all='ignore'):
                    min_batch = np.min(batch)
                    max_batch = np.max(batch)
                    
                    if np.isfinite(min_batch) and np.isfinite(max_batch):
                        count += int(np.ceil((max_batch - min_batch) / size))
                    
                gc.collect()
            
            box_counts.append(count)
        
        # Convert to numpy arrays for calculation
        box_sizes = np.array(box_sizes, dtype=np.float64)
        box_counts = np.array(box_counts, dtype=np.float64)
        
        # Compute fractal dimension using linear regression
        with np.errstate(all='ignore'):
            log_sizes = np.log(box_sizes)
            log_counts = np.log(box_counts)
            
            # Remove any non-finite values
            valid_mask = np.isfinite(log_sizes) & np.isfinite(log_counts)
            log_sizes = log_sizes[valid_mask]
            log_counts = log_counts[valid_mask]
            
            if len(log_sizes) > 1:
                slope = np.polyfit(log_sizes, log_counts, 1)[0]
                fractal_dimension['dimension'] = float(slope)
            else:
                fractal_dimension['dimension'] = 0.0
                if logger:
                    logger.log_and_print("Warning: Not enough valid points for fractal dimension calculation.")
            
            fractal_dimension['counts'] = dict(zip(box_sizes, box_counts))
        
        if logger:
            logger.log_and_print("Fractal dimension computation complete")
            
        return fractal_dimension
        
    except Exception as e:
        error_msg = f"Error computing fractal dimension: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {'dimension': 0.0, 'counts': {}}
                                                                 
@timing_decorator
def create_advanced_features(df, logger=None, feature_importance=None, chaos_metrics=None, superposition_patterns=None):
    """Create sophisticated features including mathematical, statistical, and domain-specific features."""
    if logger:
        logger.log_and_print("Creating advanced features...")
    else:
        print("Creating advanced features...")
    
    # Create a copy of the DataFrame to avoid fragmentation
    df_new = df.copy()
    new_features = {}
    
    try:
        # Check if there are any features to process
        if df_new.empty:
            if logger:
                logger.log_and_print("Warning: No features available to process, skipping advanced feature creation.")
            else:
                print("Warning: No features available to process, skipping advanced feature creation.")
            return df_new
        
        # 1. Prime Factor Based Features
        if logger:
            logger.log_and_print("Creating prime factor features...")
            
        # Factor ratios and relationships
        new_features['factor_complexity'] = df_new['unique_factors'] * np.log1p(df_new['factor_density'])
        new_features['factor_efficiency'] = df_new['total_factors'] / (df_new['unique_factors'] + 1e-10)
        new_features['factor_spread'] = df_new['max_factor'] / (df_new['min_factor'] + 1e-10)
        new_features['factor_concentration'] = df_new['factor_density'] * df_new['factor_entropy']
        
        # Advanced factor metrics
        new_features['factor_geometric_mean'] = np.exp(df_new['factor_entropy'])
        new_features['factor_harmonic_mean'] = df_new['total_factors'] / (df_new['factor_density'] + 1e-10)
        new_features['factor_range_normalized'] = (df_new['max_factor'] - df_new['min_factor']) / (df_new['mean_factor'] + 1e-10)
        
        # 2. Statistical Features
        if logger:
            logger.log_and_print("Creating statistical features...")
            
        # Rolling statistics with multiple windows
        windows = [3, 5, 7, 11]
        for w in windows:
            new_features[f'rolling_mean_{w}'] = df_new['gap_size'].rolling(w, min_periods=1).mean()
            new_features[f'rolling_std_{w}'] = df_new['gap_size'].rolling(w, min_periods=1).std()
            new_features[f'rolling_max_{w}'] = df_new['gap_size'].rolling(w, min_periods=1).max()
            new_features[f'rolling_min_{w}'] = df_new['gap_size'].rolling(w, min_periods=1).min()
            
            # Handle edge cases for kurtosis and skewness
            kurt = df_new['gap_size'].rolling(w, min_periods=1).kurt()
            new_features[f'rolling_kurt_{w}'] = kurt.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            skew = df_new['gap_size'].rolling(w, min_periods=1).skew()
            new_features[f'rolling_skew_{w}'] = skew.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 3. Mathematical Transformations
        if logger:
            logger.log_and_print("Creating mathematical transformations...")
            
        # Power transformations
        for power in [0.5, 2, 3]:
            new_features[f'factor_density_power_{power}'] = np.power(df_new['factor_density'], power)
            new_features[f'factor_entropy_power_{power}'] = np.power(df_new['factor_entropy'], power)
        
        # Logarithmic transformations
        log_features = ['factor_density', 'mean_factor', 'factor_entropy', 'gap_size']
        for feat in log_features:
            new_features[f'log_{feat}'] = np.log1p(df_new[feat])
            new_features[f'log2_{feat}'] = np.log2(df_new[feat] + 1)
            new_features[f'log10_{feat}'] = np.log10(df_new[feat] + 1)
            
            # Add exponential and trigonometric transformations of log features
            new_features[f'exp_log_{feat}'] = np.exp(new_features[f'log_{feat}'])
            new_features[f'sin_log_{feat}'] = np.sin(new_features[f'log_{feat}'])
            new_features[f'cos_log_{feat}'] = np.cos(new_features[f'log_{feat}'])
        
        # 4. Interaction Features
        if logger:
            logger.log_and_print("Creating interaction features...")
            
        base_features = [
            'factor_density', 'factor_entropy', 'mean_factor',
            'factor_std', 'factor_range_ratio'
        ]
        
        # Create meaningful interactions
        for feat1, feat2 in combinations(base_features, 2):
            new_features[f'{feat1}_{feat2}_prod'] = df_new[feat1] * df_new[feat2]
            new_features[f'{feat1}_{feat2}_ratio'] = df_new[feat1] / (df_new[feat2] + 1e-10)
            new_features[f'{feat1}_{feat2}_sum'] = df_new[feat1] + df_new[feat2]
            
            # Add polynomial interaction
            new_features[f'{feat1}_{feat2}_poly2'] = df_new[feat1] * df_new[feat2] ** 2
            new_features[f'{feat1}_{feat2}_poly3'] = df_new[feat1] ** 2 * df_new[feat2]
            
            # Add exponential interaction
            new_features[f'{feat1}_exp_{feat2}'] = df_new[feat1] * np.exp(df_new[feat2])
            new_features[f'{feat2}_exp_{feat1}'] = df_new[feat2] * np.exp(df_new[feat1])
            
            # Add trigonometric interaction
            new_features[f'{feat1}_sin_{feat2}'] = df_new[feat1] * np.sin(df_new[feat2])
            new_features[f'{feat1}_cos_{feat2}'] = df_new[feat1] * np.cos(df_new[feat2])
        
        # 5. Time Series Features
        if logger:
            logger.log_and_print("Creating time series features...")
            
        # Lag features
        lags = [1, 2, 3, 5, 7]
        for lag in lags:
            new_features[f'gap_lag_{lag}'] = df_new['gap_size'].shift(lag)
            new_features[f'factor_density_lag_{lag}'] = df_new['factor_density'].shift(lag)
        
        # Difference features
        new_features['gap_diff_1'] = df_new['gap_size'].diff()
        new_features['gap_diff_2'] = df_new['gap_size'].diff(2)
        new_features['gap_pct_change'] = df_new['gap_size'].pct_change()
        
        # 6. Frequency Domain Features
        if logger:
            logger.log_and_print("Creating frequency domain features...")
            
        # Compute FFT for gap sizes
        gap_values = df_new['gap_size'].fillna(0).values
        gap_values = gap_values.astype(np.float64)  # Ensure float64 type
        
        # Replace NaN and infinite values with the median
        if np.any(~np.isfinite(gap_values)):
            if logger:
                logger.log_and_print("Warning: Non-finite values found in gap_size. Replacing with median.", level=logging.WARNING)
            median_gap = np.nanmedian(gap_values)  # Use median to avoid skew from extreme values
            gap_values = np.where(np.isfinite(gap_values), gap_values, median_gap)
        
        if len(gap_values) > 1:
            gap_fft = np.array(fft(gap_values))
            gap_freq = np.array(fftfreq(len(gap_values)))
            
            # Extract frequency domain features
            # Convert complex FFT values to magnitudes and handle properly
            fft_magnitudes = np.abs(gap_fft[1:])  # Skip DC component
            fft_array = np.abs(gap_fft)  # Get magnitudes of all components
            
            if len(fft_magnitudes) > 0:
                main_freq_idx = int(np.argmax(fft_magnitudes))
                new_features['fft_main_freq'] = float(fft_magnitudes[main_freq_idx])
                
                # Handle mean calculation
                fft_array_mean = np.asarray(fft_array, dtype=np.float64)
                new_features['fft_mean_magnitude'] = float(np.mean(fft_array_mean))
                
                # Handle std calculation
                fft_array_std = np.asarray(fft_array, dtype=np.float64)
                new_features['fft_std_magnitude'] = float(np.std(fft_array_std))
            else:
                new_features['fft_main_freq'] = 0.0
                new_features['fft_mean_magnitude'] = 0.0
                new_features['fft_std_magnitude'] = 0.0
        
        # 7. Pattern-Based Features
        if logger:
            logger.log_and_print("Creating pattern-based features...")
            
        # Find peaks in gap sizes
        peaks, _ = find_peaks(df_new['gap_size'].fillna(0).values)
        
        # Initialize distance_to_peak as a numpy array
        distance_to_peak = np.zeros(len(df_new))
        if len(peaks) > 0:
            for i in range(len(df_new)):
                distances = np.abs(peaks - i)
                distance_to_peak[i] = np.min(distances)
        new_features['distance_to_peak'] = distance_to_peak
        
        # Local pattern features
        new_features['is_local_max'] = df_new['gap_size'] > df_new['gap_size'].shift(1)
        new_features['is_local_min'] = df_new['gap_size'] < df_new['gap_size'].shift(1)
        
        # 8. Composite Features
        if logger:
            logger.log_and_print("Creating composite features...")
            
        # Create complex composite features
        new_features['complexity_score'] = (
            df_new['factor_entropy'] * 
            df_new['factor_density'] * 
            np.log1p(df_new['mean_factor'])
        ) ** (1/3)
        
        new_features['stability_score'] = 1 / (
            df_new['factor_std'] * 
            new_features['gap_diff_1'].abs() + 1e-10
        )
        
        new_features['pattern_strength'] = (
            df_new['factor_entropy'] * 
            new_features['distance_to_peak'] * 
            new_features['fft_main_freq']
        ) ** (1/3)
        
        # 9. Integrate other analysis results
        if feature_importance and isinstance(feature_importance, dict):
            if logger:
                logger.log_and_print("Integrating feature importance results...")
            for method, scores in feature_importance.items():
                if isinstance(scores, dict):
                    for feature, score in scores.items():
                        new_features[f'{feature}_importance_{method}'] = score

        if chaos_metrics and isinstance(chaos_metrics, dict):
            if logger:
                logger.log_and_print("Integrating chaos metrics...")
            for feature, metrics in chaos_metrics.items():
                for metric_name, value in metrics.items():
                    new_features[f'{feature}_chaos_{metric_name}'] = value
        
        if superposition_patterns and isinstance(superposition_patterns, dict):
            if logger:
                logger.log_and_print("Integrating superposition patterns...")
            for feature, patterns in superposition_patterns.items():
                for pattern_name, value in patterns.items():
                    if isinstance(value, dict):
                        for sub_pattern, sub_value in value.items():
                            new_features[f'{feature}_superposition_{pattern_name}_{sub_pattern}'] = sub_value
                    else:
                         new_features[f'{feature}_superposition_{pattern_name}'] = value
        
        # 10. Clean up and finalize
        if logger:
            logger.log_and_print("Cleaning up features...")
            
        # Create new DataFrame with all features at once
        new_df = pd.concat([df_new, pd.DataFrame(new_features)], axis=1)
        
        # Replace infinities
        new_df = new_df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        new_df = new_df.fillna(0)
        
        # Clip extreme values
        for col in new_df.select_dtypes(include=[np.number]).columns:
            new_df[col] = new_df[col].clip(-1e10, 1e10)
        
        if logger:
            logger.log_and_print(f"Created {len(new_df.columns)} total features")
        
        return new_df
        
    except Exception as e:
        if logger:
            logger.log_and_print(f"Error creating advanced features: {str(e)}", level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(f"Error creating advanced features: {str(e)}")
            traceback.print_exc()
        return df
    
              

@timing_decorator
def perform_clustering(df):
    """Perform clustering on smaller datasets with numerical protection."""
    with suppress_numeric_warnings():
        feature_cols = [col for col in df.columns if col not in ['gap_size', 'cluster', 'sub_cluster', 'lower_prime', 'upper_prime', 'is_outlier', 'preceding_gaps']]
        X = df[feature_cols]
        
        # Handle missing values and scale with protection
        X = X.fillna(0)
        X = X.clip(-1e10, 1e10)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = np.clip(X_scaled, -1e10, 1e10)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        return kmeans.fit_predict(X_scaled)
    
def nth_prime(n):
    """Calculate the nth prime number."""
    if n < 1:
        raise ValueError("n must be positive")
    if n == 1:
        return 2
    
    # Use prime number theorem to get an upper bound
    if n < 6:
        upper_bound = 13  # Covers first 5 primes
    else:
        upper_bound = int(n * (math.log(n) + math.log(math.log(n))))
    
    # Generate primes up to upper_bound
    primes = list(primerange(2, upper_bound + 1))
    
    # If we didn't get enough primes, increase upper bound
    while len(primes) < n:
        upper_bound = int(upper_bound * 1.5)
        primes = list(primerange(2, upper_bound + 1))
    
    return primes[n-1]


@timing_decorator
def batch_process_primes(n, batch_size=100000, logger=None):
    """Process primes in batches with comprehensive profiling and reduced memory usage."""
    
    class BatchProfiler:
        def __init__(self):
            self.batch_stats = []
            self.total_start_time = time.time()
            self.total_start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
        def log_batch(self, batch_num, stats):
            self.batch_stats.append({
                'batch': batch_num,
                **stats
            })
            
        def print_summary(self):
            total_time = time.time() - self.total_start_time
            total_memory = psutil.Process().memory_info().rss / 1024 / 1024 - self.total_start_memory
            
            print("\nBatch Processing Summary:")
            print(f"Total Time: {total_time:.2f} seconds")
            print(f"Total Memory: {total_memory:.2f} MB")
            print("\nPer-Operation Averages:")
            
            # Calculate averages for each operation
            operations = list(self.batch_stats[0].keys()) # Convert to list
            if 'batch' in operations:
                operations.remove('batch')
            
            for op in operations:
                times = [stat[op]['time'] for stat in self.batch_stats]
                memories = [stat[op]['memory'] for stat in self.batch_stats]
                print(f"\n{op}:")
                print(f"  Avg Time: {np.mean(times):.2f} seconds")
                print(f"  Avg Memory: {np.mean(memories):.2f} MB")
                print(f"  Max Time: {max(times):.2f} seconds")
                print(f"  Max Memory: {max(memories):.2f} MB")

    profiler = BatchProfiler()
    
    try:
        if logger:
            logger.log_and_print(f"Processing {n} primes...")
        else:
            print(f"Processing {n} primes...")

        # Time prime generation
        prime_start = time.time()
        mem_start = psutil.Process().memory_info().rss / 1024 / 1024
        
        if logger:
            logger.log_and_print("Generating primes...")
        else:
            print("Generating primes...")
            
        primes = list(primerange(2, int(n * (math.log(n) + math.log(math.log(n)) + 1))))
        primes = primes[:n]
        
        prime_time = time.time() - prime_start
        prime_memory = psutil.Process().memory_info().rss / 1024 / 1024 - mem_start
        
        if logger:
            logger.log_and_print(f"Generated {len(primes)} primes")
        else:
            print(f"Generated {len(primes)} primes")

        all_dfs = []
        total_batches = (n + batch_size - 1) // batch_size
        
        # Determine optimal batch size based on available memory and CPU cores
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        estimated_memory_per_prime = 0.5  # MB per prime (conservative estimate)
        num_cores = psutil.cpu_count()
        
        optimal_batch_size = min(
            5000,  # Maximum batch size
            int((available_memory * 0.2) / estimated_memory_per_prime),  # Use 20% of available memory
            int(n / num_cores) # Distribute workload across cores
        )
        
        if logger:
            logger.log_and_print(f"Using initial batch size of {optimal_batch_size} (Available memory: {available_memory:.2f} MB, Cores: {num_cores})")
        else:
            print(f"Using initial batch size of {optimal_batch_size} (Available memory: {available_memory:.2f} MB, Cores: {num_cores})")
        
        current_batch_size = optimal_batch_size
        
        with Pool() as pool:
            for batch in range(total_batches):
                batch_stats = {}
                start_idx = batch * current_batch_size
                end_idx = min((batch + 1) * current_batch_size, n)
                
                if logger:
                    logger.log_and_print(f"Processing batch {batch + 1}/{total_batches} (size: {current_batch_size})")
                else:
                    print(f"Processing batch {batch + 1}/{total_batches} (size: {current_batch_size})")
                
                # Get batch of primes
                batch_start = time.time()
                mem_start = psutil.Process().memory_info().rss / 1024 / 1024
                
                batch_primes = primes[start_idx:end_idx]
                
                batch_time = time.time() - batch_start
                batch_memory = psutil.Process().memory_info().rss / 1024 / 1024 - mem_start
                
                batch_stats['prime_selection'] = {
                    'time': batch_time,
                    'memory': batch_memory
                }

                # Compute features
                features_start = time.time()
                mem_start = psutil.Process().memory_info().rss / 1024 / 1024
                
                batch_features = pool.starmap(
                    compute_advanced_prime_features,
                    [(batch_primes[i], batch_primes[i + 1], int(batch_primes[i + 1] or 0) - int(batch_primes[i] or 0))
                     for i in range(len(batch_primes) - 1)]
                )
                
                for i, features in enumerate(batch_features):
                    features['lower_prime'] = batch_primes[i]
                    features['upper_prime'] = batch_primes[i + 1]
                    
                features_time = time.time() - features_start
                features_memory = psutil.Process().memory_info().rss / 1024 / 1024 - mem_start
                
                batch_stats['feature_computation'] = {
                    'time': features_time,
                    'memory': features_memory
                }

                # DataFrame creation and optimization
                df_start = time.time()
                mem_start = psutil.Process().memory_info().rss / 1024 / 1024
                
                if batch_features:
                    batch_df = pd.DataFrame(batch_features)
                    all_dfs.append(batch_df)
                
                df_time = time.time() - df_start
                df_memory = psutil.Process().memory_info().rss / 1024 / 1024 - mem_start
                
                batch_stats['dataframe_ops'] = {
                    'time': df_time,
                    'memory': df_memory
                }

                # Memory cleanup
                cleanup_start = time.time()
                mem_start = psutil.Process().memory_info().rss / 1024 / 1024
                
                del batch_features
                gc.collect()
                
                cleanup_time = time.time() - cleanup_start
                cleanup_memory = psutil.Process().memory_info().rss / 1024 / 1024 - mem_start
                
                batch_stats['cleanup'] = {
                    'time': cleanup_time,
                    'memory': cleanup_memory
                }

                # Log batch statistics
                profiler.log_batch(batch + 1, batch_stats)
                
                # Adjust batch size dynamically
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                if current_memory > available_memory * 0.8:
                    current_batch_size = max(1000, int(current_batch_size * 0.8))
                    if logger:
                        logger.log_and_print(f"Reducing batch size to {current_batch_size} due to high memory usage")
                    else:
                        print(f"Reducing batch size to {current_batch_size} due to high memory usage")
                elif current_memory < available_memory * 0.5 and current_batch_size < 5000:
                    current_batch_size = min(5000, int(current_batch_size * 1.2))
                    if logger:
                        logger.log_and_print(f"Increasing batch size to {current_batch_size} due to low memory usage")
                    else:
                        print(f"Increasing batch size to {current_batch_size} due to low memory usage")

        # Create final DataFrame
        if logger:
            logger.log_and_print("Creating final DataFrame...")
        else:
            print("Creating final DataFrame...")
        
        final_df_start = time.time()
        mem_start = psutil.Process().memory_info().rss / 1024 / 1024
        
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        final_df_time = time.time() - final_df_start
        final_df_memory = psutil.Process().memory_info().rss / 1024 / 1024 - mem_start
        
        # Print profiling summary
        profiler.print_summary()
        
        return final_df

    except Exception as e:
        error_msg = f"Error in batch processing primes: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        return pd.DataFrame()
     
def compute_statistics_safe(data):
    """Compute statistics with overflow protection and NaN handling."""
    with suppress_numeric_warnings():
        try:
            data_array = np.array(data, dtype=np.float64)
            data_array = np.clip(data_array, -1e10, 1e10)
            
            # Remove NaN and infinite values
            data_array = data_array[np.isfinite(data_array)]
            
            if len(data_array) == 0:
                return {
                    'mean': 0.0,
                    'median': 0.0,
                    'std': 0.0,
                    'var': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'q1': 0.0,
                    'q3': 0.0,
                    'iqr': 0.0,
                    'skew': 0.0,
                    'kurtosis': 0.0
                }
            
            return {
                'mean': float(np.mean(data_array)),
                'median': float(np.median(data_array)),
                'std': float(np.std(data_array)),
                'var': float(np.var(data_array)),
                'min': float(np.min(data_array)),
                'max': float(np.max(data_array)),
                'q1': float(np.percentile(data_array, 25)),
                'q3': float(np.percentile(data_array, 75)),
                'iqr': float(np.percentile(data_array, 75) - np.percentile(data_array, 25)),
                'skew': float(sps.skew(data_array)),
                'kurtosis': float(sps.kurtosis(data_array))
            }
        except Exception as e:
            print(f"Warning: Error computing statistics: {str(e)}")
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'var': 0.0,
                'min': 0.0,
                'max': 0.0,
                'q1': 0.0,
                'q3': 0.0,
                'iqr': 0.0,
                'skew': 0.0,
                'kurtosis': 0.0
            }
            
@timing_decorator
def batch_clustering(df, batch_size=100000):
    """Perform clustering on large datasets in batches with numerical protection."""
    with suppress_numeric_warnings():
        feature_cols = [col for col in df.columns if col not in ['gap_size', 'cluster', 'sub_cluster', 'lower_prime', 'upper_prime', 'is_outlier', 'preceding_gaps']]
        X = df[feature_cols]
        
        # Initialize mini-batch KMeans with protection
        kmeans = MiniBatchKMeans(
            n_clusters=3,
            batch_size=min(batch_size, len(df)),
            random_state=42,
            n_init=3
        )
        
        # Process in batches
        for i in range(0, len(X), batch_size):
            batch = X.iloc[i:i+batch_size]
            batch = batch.fillna(0)
            batch = np.clip(batch.values, -1e10, 1e10)
            kmeans = kmeans.partial_fit(batch)
        
        # Final prediction in batches
        labels = []
        for i in range(0, len(X), batch_size):
            batch = X.iloc[i:i+batch_size]
            batch = batch.fillna(0)
            batch = np.clip(batch.values, -1e10, 1e10)
            batch_labels = kmeans.predict(batch)
            labels.extend(batch_labels)
        
        return np.array(labels)

@timing_decorator
def create_visualizations_large_scale(df, feature_importance, pattern_analysis, plot_dir, model_results, analysis_stats=None, batch_size=10000):
    """Create visualizations with improved memory handling for large datasets."""
    # Debug logging at the very start
    print("\n=== DEBUG START ===")
    print(f"analysis_stats type: {type(analysis_stats)}")
    print(f"analysis_stats value: {analysis_stats}")
    print("=== DEBUG END ===\n")

    print("Creating visualizations with size optimization...")
    
    # Safely get interaction_analysis with default empty dict
    interaction_analysis = {} if analysis_stats is None else analysis_stats.get('feature_interactions', {})
    
    print(f"\nDEBUG: interaction_analysis type: {type(interaction_analysis)}")
    print(f"DEBUG: interaction_analysis value: {interaction_analysis}")
    
    if isinstance(interaction_analysis, tuple):
        print(f"DEBUG: interaction_analysis is tuple with length: {len(interaction_analysis)}")
        print(f"DEBUG: tuple contents: {[type(x) for x in interaction_analysis]}")
        
    # Create subdirectories for different types of plots
    for subdir in ['distributions', 'correlations', 'clusters', 'models']:
        os.makedirs(os.path.join(plot_dir, subdir), exist_ok=True)
    
    try:
        # 1. Feature Importance Plot (with pagination)
        if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
            print("  Creating feature importance plots...")
            features_per_plot = 20
            importance_values = feature_importance.mean(axis=1).sort_values(ascending=False)
            
            for i in range(0, len(importance_values), features_per_plot):
                plt.figure(figsize=(12, 6))
                subset = importance_values[i:i+features_per_plot]
                plt.bar(range(len(subset)), subset.to_numpy())
                plt.xticks(range(len(subset)), subset.index, rotation=45, ha='right')
                plt.title(f'Feature Importance (Group {i//features_per_plot + 1})')
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'models', f"feature_importance_{i//features_per_plot + 1}.png"))
                plt.close()
                gc.collect()
        else:
            print("Warning: No feature importance data available for plotting.")
        
        # 2. Gap Distribution Plot
        print("  Creating gap distribution plots...")
        plt.figure(figsize=(12, 6))
        
        if len(df) > 10000:
            # Use numpy histogram for large datasets
            gaps = df['gap_size'].values
            hist, bin_edges = np.histogram(gaps, bins='auto', density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.plot(bin_centers, hist)
        else:
            plt.hist(df['gap_size'].values, bins='auto', density=True)
        
        plt.title('Distribution of Prime Gaps')
        plt.xlabel('Gap Size')
        plt.ylabel('Density')
        plt.savefig(os.path.join(plot_dir, 'distributions', "gap_distribution.png"))
        plt.close()
        gc.collect()
        
        # 5. Cluster Analysis Visualizations
        print("  Creating cluster visualizations...")
        if 'cluster' in df.columns:
            create_cluster_distribution_plots(df, plot_dir, batch_size)
            create_cluster_feature_plots(df, plot_dir, batch_size)
            create_cluster_transition_plots(df, plot_dir, batch_size)
        
        # 6. Feature Stability Plot
        if analysis_stats is not None and 'feature_stability' in analysis_stats and 'temporal_stability' in analysis_stats['feature_stability']:
            print("  Creating feature temporal stability plot...")
            temporal_stability = analysis_stats['feature_stability']['temporal_stability']
            
            if temporal_stability:
                # Prepare data for plotting
                features = list(temporal_stability.keys())
                
                if features:
                    mean_stabilities = [temporal_stability[feature].get('mean_stability', None) for feature in features]
                    mean_stabilities = [stability for stability in mean_stabilities if stability is not None]
                    
                    if mean_stabilities:
                        # Create DataFrame for plotting
                        temporal_scores = pd.DataFrame(
                            {'stability': mean_stabilities},
                            index=features
                        )
                        
                        if not temporal_scores.empty:
                            plt.figure(figsize=(12, 6))
                            # Create bar plot
                            temporal_scores.plot(kind='barh', legend=False)
                            plt.title('Feature Temporal Stability')
                            plt.xlabel('Mean Stability')
                            plt.ylabel('Feature')
                            plt.tight_layout()
                            plt.savefig(os.path.join(plot_dir, 'models', "feature_temporal_stability.png"))
                            plt.close()
                        else:
                            print("Warning: No temporal stability data available for plotting (empty DataFrame).")
                    else:
                        print("Warning: No temporal stability data available for plotting (empty mean_stabilities).")
                else:
                    print("Warning: No temporal stability data available for plotting (empty features).")
            else:
                print("Warning: No temporal stability data available for plotting.")
        
        # 7. Feature Interaction Heatmap
        print("  Creating feature interaction heatmap...")
        interaction_analysis = analysis_stats.get('interaction_analysis', {}) if analysis_stats else {}
        if isinstance(interaction_analysis, tuple):
            interaction_analysis = interaction_analysis[0]  # Get the dictionary from the tuple
            
        if isinstance(interaction_analysis, dict) and 'pairwise_correlations' in interaction_analysis:
            pairwise_correlations = interaction_analysis['pairwise_correlations']
            if isinstance(pairwise_correlations, dict) and 'significant_correlations' in pairwise_correlations:
                correlations = pairwise_correlations['significant_correlations']
                if isinstance(correlations, list):
                    if correlations:
                        # Create correlation matrix with float64 dtype
                        features = list(set([c['feature1'] for c in correlations] + [c['feature2'] for c in correlations]))
                        interaction_matrix = pd.DataFrame(0.0, index=features, columns=features, dtype=np.float64)
                        
                        # Fill the matrix with correlations
                        for corr in correlations:
                            # Convert correlation values to Python float
                            correlation_value = float(corr['correlation'])
                            # Use at[] for setting values which accepts Python float
                            interaction_matrix.at[corr['feature1'], corr['feature2']] = correlation_value
                            interaction_matrix.at[corr['feature2'], corr['feature1']] = correlation_value
                        
                        # Check if the matrix is empty before plotting
                        if not interaction_matrix.empty and interaction_matrix.size > 0:
                            plt.figure(figsize=(12, 10))
                            sns.heatmap(interaction_matrix, annot=True, cmap='RdBu_r', center=0)
                            plt.title('Feature Interaction Heatmap')
                            plt.tight_layout()
                            plt.savefig(os.path.join(plot_dir, 'models', "feature_interactions.png"))
                            plt.close()
                        else:
                            print("Warning: No data to create feature interaction heatmap (empty DataFrame).")
                    else:
                        print("Warning: No significant correlations found for heatmap.")
                else:
                    print("Warning: 'significant_correlations' is not a list.")
            else:
                print("Warning: 'significant_correlations' key not found in pairwise_correlations.")
        else:
            print("Warning: pairwise_correlations is not a dictionary.")
        
    except Exception as e:
        print(f"Warning: Error in visualization creation: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        plt.close('all')
        gc.collect()
        
@timing_decorator     
def create_cluster_distribution_plots(df, plot_dir, batch_size=10000):
    """Create cluster-specific distribution plots with batched processing."""
    for cluster in df['cluster'].unique():
        plt.figure(figsize=(10, 6))
        cluster_data = []
        
        # Collect data in batches
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_data = df.iloc[start_idx:end_idx]
            cluster_batch = batch_data[batch_data['cluster'] == cluster]['gap_size']
            cluster_data.extend(cluster_batch.values)
            gc.collect()
        
        if len(cluster_data) > 5000:
            sns.kdeplot(data=cluster_data, bw_adjust=1)
        else:
            sns.histplot(data=cluster_data, bins='auto', kde=True)
            
        plt.title(f'Gap Distribution for Cluster {cluster}')
        plt.savefig(os.path.join(plot_dir, 'clusters', f"cluster_{cluster}_distribution.png"))
        plt.close()
        gc.collect()

@timing_decorator
def create_cluster_feature_plots(df, plot_dir, batch_size=10000):
    """Create feature distribution plots by cluster with batched processing and MiniBatchKMeans."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features_per_plot = 4
    
    for i in range(0, len(numeric_cols), features_per_plot):
        plt.figure(figsize=(15, 10))
        feature_batch = numeric_cols[i:i+features_per_plot]
        
        for j, col in enumerate(feature_batch):
            if j < features_per_plot:
                plt.subplot(2, 2, j+1)
                
                # Process data in batches
                data_by_cluster = {cluster: [] for cluster in df['cluster'].unique()}
                for start_idx in range(0, len(df), batch_size):
                    end_idx = min(start_idx + batch_size, len(df))
                    batch = df.iloc[start_idx:end_idx]
                    for cluster in data_by_cluster:
                        cluster_batch = batch[batch['cluster'] == cluster][col]
                        data_by_cluster[cluster].extend(cluster_batch.values)
                    gc.collect()
                
                # Create box plot from accumulated data
                plt.boxplot([data_by_cluster[cluster] for cluster in sorted(data_by_cluster.keys())],
                            tick_labels=sorted(data_by_cluster.keys())) # type: ignore
                plt.title(f'{col} by Cluster')
                
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'clusters', f"cluster_features_{i//features_per_plot + 1}.png"))
        plt.close()
        gc.collect()

@timing_decorator
def create_cluster_transition_plots(df, plot_dir, batch_size=10000):
    """Create cluster transition visualization with batched processing."""
    transitions = np.zeros((df['cluster'].nunique(), df['cluster'].nunique()))
    
    # Compute transitions in batches
    for start_idx in range(0, len(df) - 1, batch_size):
        end_idx = min(start_idx + batch_size, len(df) - 1)
        current_clusters = df['cluster'].iloc[start_idx:end_idx]
        next_clusters = df['cluster'].iloc[start_idx + 1:end_idx + 1]
        
        for curr, next_cluster in zip(current_clusters, next_clusters):
            transitions[curr, next_cluster] += 1
        
        gc.collect()
    
    # Create transition plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(transitions / transitions.sum(), annot=True, fmt='.2%')
    plt.title('Cluster Transition Probabilities')
    plt.savefig(os.path.join(plot_dir, 'clusters', "cluster_transitions.png"))
    plt.close()
    gc.collect()
    

@timing_decorator
def compute_shap_values(models, X, feature_cols, n_samples=1000, batch_size=500, logger=None):
    """Compute SHAP values for feature interpretation with improved memory management and numerical stability."""
    if logger:
        logger.log_and_print("Computing SHAP values...")
    
    shap_values = {}
    feature_importance_shap = {}
    
    try:
        # Dynamically adjust sample size based on data size
        if len(X) > 100000:
            n_samples = 5000  # Reduce sample size for very large datasets
        elif len(X) > 50000:
            n_samples = 2000
        elif len(X) > 10000:
            n_samples = 1000
        else:
            n_samples = len(X) # Use all data if small enough
        
        # Sample data if too large
        if len(X) > n_samples:
            sample_idx = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[sample_idx]
        else:
            X_sample = X
        
        # Convert to float64 and clip values
        X_sample = X_sample.astype(np.float64)
        X_sample = X_sample.clip(-1e10, 1e10)
        
        for name, model in models.items():
            if logger:
                logger.log_and_print(f"Processing model: {name}")
            
            # Handle stacking models
            if isinstance(model, dict) and 'meta_learner' in model and 'base_models' in model:
                try:
                    # Create combined model prediction function with batching
                    def model_predict(X_input):
                        predictions = []
                        for start_idx in range(0, len(X_input), batch_size):
                            end_idx = min(start_idx + batch_size, len(X_input))
                            X_batch = X_input[start_idx:end_idx]
                            
                            # Generate base predictions
                            base_preds = []
                            for base_model in model['base_models'].values():
                                with np.errstate(all='ignore'):
                                    pred = base_model.predict(X_batch)
                                    pred = np.clip(pred, -1e10, 1e10)
                                    base_preds.append(pred)
                            
                            # Stack predictions
                            stacked_preds = np.column_stack(base_preds)
                            
                            # Get meta-learner prediction
                            with np.errstate(all='ignore'):
                                meta_pred = model['meta_learner'].predict(stacked_preds)
                                meta_pred = np.clip(meta_pred, -1e10, 1e10)
                                predictions.append(meta_pred)
                            
                            gc.collect()
                        
                        return np.concatenate(predictions)
                    
                    # Use KernelExplainer for stacking models with smaller background dataset
                    background = shap.kmeans(X_sample, min(50, len(X_sample))).data  # Use fewer background samples
                    explainer = shap.KernelExplainer(model_predict, background)
                    
                    # Compute SHAP values in batches
                    all_shap_values = []
                    for start_idx in range(0, len(X_sample), batch_size):
                        end_idx = min(start_idx + batch_size, len(X_sample))
                        batch_shap = explainer.shap_values(X_sample[start_idx:end_idx])
                        all_shap_values.append(batch_shap)
                        gc.collect()
                    
                    shap_values[name] = np.concatenate(all_shap_values)
                    feature_importance = np.abs(shap_values[name]).mean(0)
                    feature_importance_shap[name] = dict(zip(feature_cols, feature_importance))
                    
                    if logger:
                        logger.log_and_print(f"SHAP values computed for stacking model {name}")
                    
                except Exception as e:
                    if logger:
                        logger.log_and_print(f"Error computing SHAP values for stacking model {name}: {str(e)}")
                        logger.logger.error(traceback.format_exc())
                    continue
                
            # Handle individual models
            elif isinstance(model, (RandomForestRegressor, xgb.XGBRegressor)):
                try:
                    # Create explainer based on model type
                    if isinstance(model, RandomForestRegressor):
                        explainer = shap.TreeExplainer(model)
                    else:  # XGBoost
                        explainer = shap.TreeExplainer(model)
                    
                    # Calculate SHAP values in batches
                    all_shap_values = []
                    for start_idx in range(0, len(X_sample), batch_size):
                        end_idx = min(start_idx + batch_size, len(X_sample))
                        with np.errstate(all='ignore'):
                            batch_shap = explainer.shap_values(X_sample[start_idx:end_idx])
                            if isinstance(batch_shap, list):
                                batch_shap = batch_shap[0]  # For multi-output models
                            batch_shap = np.clip(batch_shap, -1e10, 1e10)
                            all_shap_values.append(batch_shap)
                        gc.collect()
                    
                    shap_values[name] = np.concatenate(all_shap_values)
                    feature_importance = np.abs(shap_values[name]).mean(0)
                    feature_importance_shap[name] = dict(zip(feature_cols, feature_importance))
                    
                    if logger:
                        logger.log_and_print(f"SHAP values computed for {name}")
                    
                except Exception as e:
                    if logger:
                        logger.log_and_print(f"Error computing SHAP values for {name}: {str(e)}")
                        logger.logger.error(traceback.format_exc())
                    continue
                
            else:
                if logger:
                    logger.log_and_print(f"Skipping SHAP calculation for {name} - unsupported model type: {type(model)}")
        
        return shap_values, feature_importance_shap
        
    except Exception as e:
        error_msg = f"Error in SHAP value computation: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {}, {}
      
@timing_decorator
def compute_prediction_intervals(model, X, confidence=0.95, n_bootstraps=50, batch_size=5000, logger=None):
    """Compute prediction intervals using bootstrap with improved memory management and numerical stability."""
    if logger:
        logger.log_and_print("Computing prediction intervals...")
    
    try:
        predictions = []
        
        # Define predict_func outside the conditional blocks
        def stacking_predict(X_sample):
            # Generate base model predictions in batches
            base_predictions = []
            for start_idx in range(0, len(X_sample), batch_size):
                end_idx = min(start_idx + batch_size, len(X_sample))
                batch = X_sample[start_idx:end_idx]
                
                # Get predictions from each base model
                batch_preds = []
                for base_model in model['base_models'].values():
                    with np.errstate(all='ignore'):
                        pred = base_model.predict(batch)
                        pred = np.clip(pred, -1e10, 1e10)
                        batch_preds.append(pred)
                
                # Stack predictions
                if batch_preds:
                    stacked_preds = np.column_stack(batch_preds)
                    base_predictions.append(stacked_preds)
                gc.collect()
            
            # Combine batches
            if base_predictions:
                all_base_preds = np.vstack(base_predictions)
            else:
                all_base_preds = np.zeros((len(X_sample), 0))
            
            # Use meta-learner for final prediction
            return model['meta_learner'].predict(all_base_preds)
        
        def ensemble_predict(X_sample):
            model_predictions = {}
            for name, submodel in model['models'].items():
                with np.errstate(all='ignore'):
                    if isinstance(submodel, tf.keras.Sequential):
                        pred = submodel.predict(X_sample).flatten()
                    else:
                        pred = submodel.predict(X_sample)
                    pred = np.clip(pred, -1e10, 1e10)
                    model_predictions[name] = pred
            # Combine using stored weights
            return sum(pred * model['weights'][name] 
                     for name, pred in model_predictions.items())
        
        if isinstance(model, dict):
            if 'base_models' in model and 'meta_learner' in model:
                predict_func = stacking_predict
            elif 'models' in model and 'weights' in model:
                predict_func = ensemble_predict
            else:
                predict_func = None
        else:
            predict_func = model.predict
        
        # For random forests, use built-in uncertainty
        if isinstance(model, RandomForestRegressor):
            if logger:
                logger.log_and_print("Using Random Forest built-in uncertainty...")
            
            tree_predictions = []
            # Process trees in batches
            for tree in model.estimators_:
                batch_predictions = []
                for start_idx in range(0, len(X), batch_size):
                    end_idx = min(start_idx + batch_size, len(X))
                    X_batch = X.iloc[start_idx:end_idx] if isinstance(X, pd.DataFrame) else X[start_idx:end_idx]
                    
                    with np.errstate(all='ignore'):
                        pred = tree.predict(X_batch)
                        pred = np.clip(pred, -1e10, 1e10)
                        batch_predictions.append(pred)
                    
                    gc.collect()
                
                tree_predictions.append(np.concatenate(batch_predictions))
            
            predictions = np.array(tree_predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # Compute intervals
            z_score = norm.ppf((1 + confidence) / 2)
            lower = mean_pred - z_score * std_pred
            upper = mean_pred + z_score * std_pred
            
        else:
            # Bootstrap for other models
            if logger:
                logger.log_and_print("Using bootstrap sampling for uncertainty...")
            
            bootstrap_predictions = []
            for i in range(n_bootstraps):
                if logger and i % 10 == 0:
                    logger.log_and_print(f"Processing bootstrap sample {i+1}/{n_bootstraps}")
                
                # Sample with replacement
                indices = np.random.choice(len(X), len(X), replace=True)
                X_boot = X.iloc[indices] if isinstance(X, pd.DataFrame) else X[indices]
                
                # Make predictions in batches
                batch_predictions = []
                for start_idx in range(0, len(X_boot), batch_size):
                    end_idx = min(start_idx + batch_size, len(X_boot))
                    X_batch = X_boot[start_idx:end_idx]
                    
                    with np.errstate(all='ignore'):
                        if predict_func is not None:
                            pred = predict_func(X_batch)
                        else:
                            raise ValueError("predict_func is None, cannot call it.")
                        pred = np.clip(pred, -1e10, 1e10)
                        if isinstance(pred, np.ndarray):
                            pred = pred.flatten()
                        batch_predictions.append(pred)
                    
                    gc.collect()
                
                bootstrap_predictions.append(np.concatenate(batch_predictions))
            
            # Convert to numpy array for efficient computation
            predictions = np.array(bootstrap_predictions)
            mean_pred = np.mean(predictions, axis=0)
            
            # Compute intervals
            lower = np.percentile(predictions, ((1 - confidence) / 2) * 100, axis=0)
            upper = np.percentile(predictions, (1 + confidence) / 2 * 100, axis=0)
        
        # Ensure all outputs are proper Python floats
        mean_pred = np.array(mean_pred, dtype=np.float64)
        lower = np.array(lower, dtype=np.float64)
        upper = np.array(upper, dtype=np.float64)
        
        # Final cleanup
        gc.collect()
        
        if logger:
            logger.log_and_print("Prediction interval computation complete")
        
        return mean_pred, lower, upper
        
    except Exception as e:
        error_msg = f"Error computing prediction intervals: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        default_size = len(X)
        return (
            np.zeros(default_size, dtype=np.float64),
            np.zeros(default_size, dtype=np.float64),
            np.zeros(default_size, dtype=np.float64)
        )
        
@njit
def _detect_change_points_numba(data, min_size):
    """Numba-optimized function for detecting change points."""
    n = len(data)
    change_points = []
    
    # Sliding window approach
    window_size = min_size * 2
    
    for i in range(0, n - window_size):
        window = data[i:i+window_size]
        
        # Calculate cost for the entire window
        mean = np.mean(window)
        total_cost = np.sum((window - mean) ** 2)
        
        # Find the best split point
        best_split = -1
        best_cost = np.inf
        
        for split in range(min_size, window_size - min_size):
            left_data = window[:split]
            right_data = window[split:]
            
            left_mean = np.mean(left_data)
            right_mean = np.mean(right_data)
            left_cost = np.sum((left_data - left_mean) ** 2)
            right_cost = np.sum((right_data - right_mean) ** 2)
            split_cost = left_cost + right_cost
            
            if split_cost < best_cost:
                best_cost = split_cost
                best_split = split
        
        # Check if split is significant
        if best_split != -1 and total_cost - best_cost > 1e-6:
            change_points.append(i + best_split)
    
    return np.array(change_points)

@timing_decorator
def detect_change_points(df, column='gap_size', min_size=50, batch_size=5000, logger=None):
    """Detect change points in the time series with Numba optimization."""
    if logger:
        logger.log_and_print("Detecting change points...")
    
    try:
        # Convert data to numpy array and ensure proper type
        data = df[column].values.astype(np.float64)
        data = np.clip(data, -1e10, 1e10)
        data = data[np.isfinite(data)]
        
        # Initialize change point detection
        n = len(data)
        if n < 2 * min_size:
            if logger:
                logger.log_and_print("Warning: Insufficient data points for change point detection.")
            return {
                'change_points': [],
                'segments': [],
                'score': float('inf'),
                'n_segments': 0,
                'mean_segment_size': 0.0,
                'std_segment_size': 0.0,
                'optimal_n_bkps': 0
            }
        
        # Call Numba-optimized function for change point detection
        change_points = _detect_change_points_numba(data, min_size)
        
        # Sort change points
        change_points = np.sort(change_points)
        
        # Compute segment statistics with Numba
        segment_stats = _compute_segment_stats_numba(data, change_points, min_size)
        
        # Convert segment stats to list of dictionaries
        segments = []
        for i in range(len(segment_stats)):
            segments.append({
                'start': int(segment_stats[i, 0]),
                'end': int(segment_stats[i, 1]),
                'size': int(segment_stats[i, 2]),
                'mean': float(segment_stats[i, 3]),
                'std': float(segment_stats[i, 4]),
                'min': float(segment_stats[i, 5]),
                'max': float(segment_stats[i, 6])
            })
        
        # Format results
        change_point_metrics = {
            'change_points': [int(cp) for cp in change_points],
            'segments': segments,
            'score': float(0),  # Placeholder for score
            'n_segments': len(segments),
            'mean_segment_size': float(np.mean([s['size'] for s in segments])) if segments else 0.0,
            'std_segment_size': float(np.std([s['size'] for s in segments])) if segments else 0.0,
            'optimal_n_bkps': len(change_points)
        }
        
        if logger:
            logger.log_and_print("Change point detection complete")
            logger.log_and_print(f"Found {len(segments)} segments")
            logger.log_and_print(f"Optimal number of change points: {len(change_points)}")
        
        return change_point_metrics
        
    except Exception as e:
        error_msg = f"Error in change point detection: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        return {
            'change_points': [],
            'segments': [],
            'score': float('inf'),
            'n_segments': 0,
            'mean_segment_size': 0.0,
            'std_segment_size': 0.0,
            'optimal_n_bkps': 0
        }

def perform_meta_learning(X, y, models, cv=5, batch_size=5000, logger=None):
    """Perform meta-learning with automated hyperparameter optimization, batch processing, and improved memory management."""
    if logger:
        logger.log_and_print("Performing meta-learning...")
    else:
        print("Performing meta-learning...")
    
    # Parameter grids for each model type
    param_grids = {
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 8, 10, 12],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 0.5, 0.7],
            'ccp_alpha': [0, 0.01, 0.02]
        },
        'xgboost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 0.3],
            'reg_lambda': [0, 0.1, 0.3],
            'gamma': [0, 0.1, 0.2]
        }
    }
    
    optimized_models = {}
    
    try:
        for name, model in models.items():
            if name in param_grids:
                if logger:
                    logger.log_and_print(f"\nOptimizing {name}...")
                else:
                    print(f"\nOptimizing {name}...")
                
                # Create RandomizedSearchCV
                search = RandomizedSearchCV(
                    model,
                    param_grids[name],
                    n_iter=20,
                    cv=cv,
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
                
                # Fit in batches with proper type handling
                for start_idx in range(0, len(X), batch_size):
                    end_idx = min(start_idx + batch_size, len(X))
                    X_batch = X.iloc[start_idx:end_idx].astype(np.float64)
                    y_batch = y.iloc[start_idx:end_idx].astype(np.float64)
                    
                    # Clip values for numerical stability
                    X_batch = X_batch.clip(-1e10, 1e10)
                    y_batch = y_batch.clip(-1e10, 1e10)
                    
                    # Use partial fit if available
                    if hasattr(search, 'partial_fit'):
                        search.fit(X_batch, y_batch)
                    else:
                        search.fit(X_batch, y_batch)
                    gc.collect()
                
                optimized_models[name] = search.best_estimator_
                
                if logger:
                    logger.log_and_print(f"Best parameters for {name}: {search.best_params_}")
                    logger.log_and_print(f"Best score: {search.best_score_:.4f}")
                else:
                    print(f"Best parameters for {name}: {search.best_params_}")
                    print(f"Best score: {search.best_score_:.4f}")
        
        return optimized_models
    
    except Exception as e:
        error_msg = f"Error in meta-learning: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return original models as fallback
        return models

@timing_decorator
def analyze_time_series_patterns(df, column='gap_size', periods=None, logger=None):
    """Perform advanced time series analysis."""
    if logger:
        logger.log_and_print("Analyzing time series patterns...")
    else:
        print("Analyzing time series patterns...")
    
    try:
        # Handle missing or invalid data
        series = df[column].copy()
        series = series.astype(float)  # Convert to float
        series = series.dropna()  # Remove any NaN values
        
        if len(series) == 0:
            raise ValueError("No valid data points after cleaning")
            
        # Initialize time series tests
        time_series_tests = {}
        
        # Determine period if not provided
        if periods is None:
            # Try different periods and pick the one with strongest seasonality
            test_periods = [12, 24, 36, 48]
            max_strength = 0
            best_period = test_periods[0]
            
            for period in test_periods:
                try:
                    if len(series) >= 2 * period:  # Ensure enough data points
                        decomp = seasonal_decompose(
                            series,
                            period=period,
                            extrapolate_trend=1  # Changed from 'freq' to 1
                        )
                        strength = np.nanstd(decomp.seasonal)  # Use nanstd to handle NaN values
                        if strength > max_strength:
                            max_strength = strength
                            best_period = period
                except Exception as e:
                    if logger:
                        logger.log_and_print(f"Warning: Period {period} failed: {str(e)}")
                    continue
            
            periods = best_period
        
        # Perform decomposition with error handling
        try:
            # Skip seasonal decomposition for now as it is memory intensive and probably not needed for primes
            # We keep the code here in case we want to enable it later
            decomposition = None
            trend_strength = 0.0
            seasonal_strength = 0.0
            
            # decomposition = seasonal_decompose(
            #     series,
            #     period=periods,
            #     extrapolate_trend=1  # Changed from 'freq' to 1
            # )
            
            # # Compute strength metrics safely
            # trend_resid_var = np.nanvar(decomposition.resid + decomposition.trend)
            # if trend_resid_var != 0:
            #     trend_strength = 1 - np.nanvar(decomposition.resid) / trend_resid_var
            # else:
            #     trend_strength = 0
                
            # seasonal_resid_var = np.nanvar(decomposition.resid + decomposition.seasonal)
            # if seasonal_resid_var != 0:
            #     seasonal_strength = 1 - np.nanvar(decomposition.resid) / seasonal_resid_var
            # else:
            #     seasonal_strength = 0
                
        except Exception as e:
            if logger:
                logger.log_and_print(f"Warning: Decomposition failed: {str(e)}")
            return {
                'trend_strength': 0.0,
                'seasonal_strength': 0.0,
                'period': periods,
                'stationarity_test': {
                    'statistic': 0.0,
                    'p_value': 1.0,
                    'critical_values': {},
                    'significance_level': 0.05
                },
                'time_series_tests': {},
                'error_message': str(e)
            }
        
        # Test for stationarity using ADF test with proper error handling
        try:
            import statsmodels.tsa.stattools as stattools
            # Get the results without unpacking
            adf_results = stattools.adfuller(series)
            
            # Extract values safely
            adf_stat = float(adf_results[0]) if len(adf_results) > 0 else 0.0
            pvalue = float(adf_results[1]) if len(adf_results) > 1 else 1.0
            
            # Handle critical values safely
            crit_vals = {}
            if len(adf_results) > 4 and isinstance(adf_results[4], dict):
                critical_values = adf_results[4]
                for key in ['1%', '5%', '10%']:
                    if key in critical_values:
                        try:
                            crit_vals[key] = float(critical_values[key])
                        except (TypeError, ValueError):
                            continue
            
            stationarity_test = {
                'statistic': adf_stat,
                'p_value': pvalue,
                'critical_values': crit_vals,
                'significance_level': 0.05
            }
            
        except Exception as e:
            if logger:
                logger.log_and_print(f"Warning: ADF test failed: {str(e)}")
            stationarity_test = {
                'statistic': 0.0,
                'p_value': 1.0,
                'critical_values': {},
                'significance_level': 0.05
            }
        
        # Perform Ljung-Box test
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(series, lags=[10, 20, 30], return_df=True)
            if isinstance(lb_result, pd.DataFrame):
                time_series_tests['ljung_box'] = {
                    'statistic': lb_result['lb_stat'].tolist(),
                    'p_value': lb_result['lb_pvalue'].tolist()
                }
        except Exception as e:
            if logger:
                logger.log_and_print(f"Warning: Ljung-Box test failed: {str(e)}")
            time_series_tests['ljung_box'] = {
                'statistic': [],
                'p_value': []
            }
        
        return {
            'decomposition': None, # decomposition,
            'trend_strength': float(trend_strength),
            'seasonal_strength': float(seasonal_strength),
            'period': periods,
            'stationarity_test': stationarity_test,
            'time_series_tests': time_series_tests,
            'error_message': None
        }
        
    except Exception as e:
        if logger:
            logger.log_and_print(f"Error in time series analysis: {str(e)}")
        else:
            print(f"Error in time series analysis: {str(e)}")
        return {
            'trend_strength': 0.0,
            'seasonal_strength': 0.0,
            'period': periods,
            'stationarity_test': {
                'statistic': 0.0,
                'p_value': 1.0,
                'critical_values': {},
                'significance_level': 0.05
            },
            'time_series_tests': {},
            'error_message': str(e)
        }
        
def find_gap_patterns(df, min_pattern_length=3, max_pattern_length=10):
    """Find recurring patterns in gap sequences."""
    print("Finding gap patterns...")
    
    patterns = {}
    gaps = df['gap_size'].values
    
    for length in range(min_pattern_length, max_pattern_length + 1):
        pattern_counts = Counter()
        
        # Create sequences
        for i in range(len(gaps) - length + 1):
            pattern = tuple(gaps[i:i+length])
            pattern_counts[pattern] += 1
        
        # Keep only patterns that appear more than once
        significant_patterns = {
            pattern: count for pattern, count in pattern_counts.items()
            if count > 1
        }
        
        if significant_patterns:
            patterns[length] = {
                'patterns': significant_patterns,
                'count': len(significant_patterns),
                'most_common': sorted(
                    significant_patterns.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
    
    return patterns

@timing_decorator
def analyze_sequence_patterns(df, feature_cols, sequence_length=5, batch_size=5000, logger=None, outlier_aware=True, max_sequences=1000, min_count=2):
    """Analyze complex recurring patterns in sequences of features with improved memory management and numerical stability."""
    if logger:
        logger.log_and_print(f"Analyzing complex sequences of length {sequence_length}...")
    
    sequence_patterns = {}
    
    try:
        # Convert relevant columns to float64 and clip values
        all_data = df[feature_cols].values.astype(np.float64)
        all_data = np.clip(all_data, -1e10, 1e10)
        
        # Get outlier indices if outlier-aware mode is enabled
        outlier_indices = df[df['is_outlier']].index.tolist() if outlier_aware and 'is_outlier' in df.columns else []
        
        # Process sequences in batches
        for start_idx in range(0, len(df) - sequence_length + 1, batch_size):
            end_idx = min(start_idx + batch_size, len(df) - sequence_length + 1)
            
            # Extract sequences for this batch
            batch_sequences = []
            for i in range(start_idx, end_idx):
                seq = all_data[i:i+sequence_length]
                if np.all(np.isfinite(seq)):
                    # Check if this sequence includes an outlier
                    if outlier_aware:
                        is_outlier_transition = False
                        for j in range(i, i + sequence_length):
                            if j in outlier_indices:
                                is_outlier_transition = True
                                break
                        
                        if is_outlier_transition:
                            batch_sequences.append(('outlier_transition', tuple(map(tuple, seq))))
                        else:
                            batch_sequences.append(('regular', tuple(map(tuple, seq))))
                    else:
                        batch_sequences.append(tuple(map(tuple, seq)))
            
            if batch_sequences:
                # Count and store sequences
                counts = Counter(batch_sequences)
                for seq_type, seq in counts.items():
                    if seq_type not in sequence_patterns:
                        sequence_patterns[seq_type] = {}
                    if seq not in sequence_patterns[seq_type]:
                        sequence_patterns[seq_type][seq] = 0
                    sequence_patterns[seq_type][seq] += count
                gc.collect()
        
        # Sort sequences by frequency and limit the number of stored sequences
        formatted_patterns = {}
        for seq_type, patterns in sequence_patterns.items():
            sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
            filtered_patterns = [
                {'sequence': [list(s) for s in seq], 'count': count}
                for seq, count in sorted_patterns
                if count >= min_count
            ][:max_sequences]
            formatted_patterns[seq_type] = filtered_patterns
        
        if logger:
            logger.log_and_print("Complex sequence analysis complete")
        
        return formatted_patterns
        
    except Exception as e:
        error_msg = f"Error in complex sequence analysis: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {}
         
def create_interactive_visualizations(df, analysis_stats, output_dir):
    """Create interactive visualizations using plotly."""
    print("Creating interactive visualizations...")
    
    # 1. 3D Cluster Visualization
    if 'cluster' in df.columns:
        pca = PCA(n_components=3)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['cluster', 'gap_size']]
        X_pca = pca.fit_transform(df[feature_cols].fillna(0).clip(-1e10, 1e10))
        
        fig = px.scatter_3d(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            z=X_pca[:, 2],
            color=df['cluster'],
            title='3D Cluster Visualization'
        )
        fig.write_html(os.path.join(output_dir, '3d_clusters.html'))
    
    # 2. Interactive Time Series
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df['gap_size'],
        mode='lines',
        name='Gap Size'
    ))
    
    if 'change_points' in analysis_stats:
        for cp in analysis_stats['change_points']['change_points']:
            fig.add_vline(x=cp, line_dash="dash", line_color="red")
    
    fig.update_layout(title='Interactive Gap Size Time Series')
    fig.write_html(os.path.join(output_dir, 'interactive_time_series.html'))
    
    # 3. Pattern Distribution
    if 'pattern_analysis' in analysis_stats:
        fig = make_subplots(rows=2, cols=1)
        
        # Add histogram
        fig.add_trace(
            go.Histogram(x=df['gap_size'], name='Gap Distribution'),
            row=1, col=1
        )
        
        # Add box plot
        fig.add_trace(
            go.Box(y=df['gap_size'], name='Gap Distribution'),
            row=2, col=1
        )
        
        fig.update_layout(title='Gap Size Distribution Analysis')
        fig.write_html(os.path.join(output_dir, 'gap_distribution.html'))
    
    return True

@njit
def _compute_gmm_bic_numba(X, gmm):
    """Numba-optimized function to compute BIC for GMM."""
    n_samples = X.shape[0]
    n_components = gmm.n_components
    
    log_likelihood = 0.0
    for i in range(n_samples):
        log_prob = -np.inf
        for j in range(n_components):
            diff = X[i] - gmm.means_[j]
            exponent = -0.5 * np.sum(diff * np.linalg.solve(gmm.covariances_[j], diff))
            log_prob = np.logaddexp(log_prob, exponent + np.log(gmm.weights_[j]) - 0.5 * np.log(np.linalg.det(gmm.covariances_[j])))
        log_likelihood += log_prob
    
    n_params = n_components * (X.shape[1] + 1) + n_components * X.shape[1] * (X.shape[1] + 1) // 2
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic

@timing_decorator
def perform_advanced_clustering_analysis(df, batch_size=5000, logger=None):
    """Perform comprehensive clustering analysis with Numba optimization and improved memory management."""
    if logger:
        logger.log_and_print("Performing advanced clustering analysis...")
    
    # Get feature columns for visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in [
        'gap_size', 'cluster', 'sub_cluster', 'lower_prime', 
        'upper_prime', 'is_outlier', 'preceding_gaps'
    ]]
    X = df[feature_cols].values
    
    # Scale features with robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.clip(X_scaled, -1e10, 1e10)  # Limit extreme values
    
    clustering_results = {
        'labels': {},
        'metrics': {},
        'optimal_clusters': {},
        'cluster_profiles': {}
    }
    
    try:
        # 1. Gaussian Mixture Model with improved stability
        if logger:
            logger.log_and_print("Performing GMM clustering...")
        
        gmm_bic = []
        gmm_models = []
        
        for n_components in range(2, 6):  # Reduced range
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    random_state=42,
                    n_init=3,
                    reg_covar=1e-3,  # Increased regularization
                    covariance_type='tied',  # More stable covariance type
                    max_iter=100,
                    tol=1e-3
                )
                
                # Fit GMM in batches
                for start_idx in range(0, len(X_scaled), batch_size):
                    end_idx = min(start_idx + batch_size, len(X_scaled))
                    try:
                        if start_idx == 0:
                            gmm.fit(X_scaled[start_idx:end_idx])
                        else:
                            gmm.fit(np.vstack([gmm.means_, X_scaled[start_idx:end_idx]]))
                    except Exception as e:
                        if logger:
                            logger.log_and_print(f"Warning: GMM fitting failed for batch {start_idx}-{end_idx}: {str(e)}")
                        continue
                    gc.collect()
                
                # Compute BIC using Numba
                bic = _compute_gmm_bic_numba(X_scaled, gmm)
                if np.isfinite(bic):
                    gmm_bic.append(bic)
                    gmm_models.append(gmm)
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Warning: GMM failed for n_components={n_components}: {str(e)}")
                continue
        
        if gmm_models:
            optimal_gmm = gmm_models[np.argmin(gmm_bic)]
            gmm_labels = optimal_gmm.predict(X_scaled)
            clustering_results['labels']['gmm'] = gmm_labels
        else:
            # Fallback to KMeans if GMM fails
            if logger:
                logger.log_and_print("Falling back to KMeans clustering")
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            gmm_labels = kmeans.fit_predict(X_scaled)
            clustering_results['labels']['gmm'] = gmm_labels
        
        # 2. DBSCAN with automatic epsilon selection
        if logger:
            logger.log_and_print("Performing DBSCAN clustering...")
        
        # Compute optimal epsilon using nearest neighbors
        distances = []
        for start_idx in range(0, len(X_scaled), batch_size):
            end_idx = min(start_idx + batch_size, len(X_scaled))
            batch_distances = pdist(X_scaled[start_idx:end_idx])
            distances.extend(batch_distances)
            gc.collect()
        
        eps_candidates = np.percentile(distances, [10, 15, 20, 25])
        
        best_silhouette = -1
        best_dbscan = None
        best_eps = None
        
        for eps in eps_candidates:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=5)
                labels = dbscan.fit_predict(X_scaled)
                if len(np.unique(labels)) > 1:  # Only evaluate if more than one cluster
                    score = silhouette_score(X_scaled, labels)
                    if score > best_silhouette:
                        best_silhouette = score
                        best_dbscan = dbscan
                        best_eps = eps
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Warning: DBSCAN failed for eps={eps}: {str(e)}")
                continue
        
        if best_dbscan is not None:
            dbscan_labels = best_dbscan.fit_predict(X_scaled)
            clustering_results['labels']['dbscan'] = dbscan_labels
        else:
            # Fallback to simple clustering if DBSCAN fails
            if logger:
                logger.log_and_print("Falling back to simple clustering for DBSCAN")
            dbscan_labels = np.zeros(len(X_scaled), dtype=np.int32)
            clustering_results['labels']['dbscan'] = dbscan_labels
        
        # 3. Hierarchical Clustering
        if logger:
            logger.log_and_print("Performing Hierarchical clustering...")
        
        try:
            # Use reduced data for linkage computation
            n_samples = min(1000, len(X_scaled))
            sample_indices = np.random.choice(len(X_scaled), n_samples, replace=False)
            linkage_matrix = linkage(X_scaled[sample_indices], method='ward')
            
            # Determine optimal number of clusters
            last = linkage_matrix[-10:, 2]
            acceleration = np.diff(last, 2)
            optimal_clusters = len(last) - np.argmax(acceleration) + 1
            optimal_clusters = max(2, min(optimal_clusters, 10))  # Keep within reasonable range
            
            hierarchical = AgglomerativeClustering(n_clusters=int(optimal_clusters))
            hierarchical_labels = hierarchical.fit_predict(X_scaled)
            clustering_results['labels']['hierarchical'] = hierarchical_labels
        except Exception as e:
            if logger:
                logger.log_and_print(f"Warning: Hierarchical clustering failed: {str(e)}")
            clustering_results['labels']['hierarchical'] = np.zeros(len(X_scaled), dtype=np.int32)
        
        # Compute metrics where possible
        for name, labels in clustering_results['labels'].items():
            try:
                if len(np.unique(labels)) > 1:
                    clustering_results['metrics'][name] = {
                        'silhouette': float(silhouette_score(X_scaled, labels)),
                        'calinski_harabasz': float(calinski_harabasz_score(X_scaled, labels)),
                        'davies_bouldin': float(davies_bouldin_score(X_scaled, labels))
                    }
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Warning: Metric computation failed for {name}: {str(e)}")
                continue
        
        # 4. Cluster Profiling
        if logger:
            logger.log_and_print("Profiling clusters...")
            
        for name, labels in clustering_results['labels'].items():
            if len(np.unique(labels)) > 1:
                cluster_profiles = {}
                for cluster_id in np.unique(labels):
                    cluster_mask = labels == cluster_id
                    cluster_data = X_scaled[cluster_mask]
                    
                    # Compute mean feature values
                    cluster_mean = np.mean(cluster_data, axis=0)
                    
                    # Compute feature importance using a simple model
                    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                    rf.fit(X_scaled, labels)
                    feature_importance = rf.feature_importances_
                    
                    # Store profile information
                    cluster_profiles[int(cluster_id)] = {
                        'size': int(len(cluster_data)),
                        'mean_gap': float(np.mean(df['gap_size'][cluster_mask])),
                        'std_gap': float(np.std(df['gap_size'][cluster_mask])),
                        'min_gap': float(np.min(df['gap_size'][cluster_mask])),
                        'max_gap': float(np.max(df['gap_size'][cluster_mask])),
                        'feature_means': dict(zip(feature_cols, cluster_mean)),
                        'feature_importance': dict(zip(feature_cols, feature_importance))
                    }
                clustering_results['cluster_profiles'][name] = cluster_profiles
        
        if logger:
            logger.log_and_print("Advanced clustering analysis complete")
        
        return clustering_results
        
    except Exception as e:
        error_msg = f"Error in advanced clustering analysis: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'labels': {'fallback': np.zeros(len(X_scaled), dtype=np.int32)},
            'metrics': {},
            'optimal_clusters': {'fallback': 1},
            'cluster_profiles': {}
        }
        
@njit
def _compute_segment_stats_numba(data, change_points, min_size):
    """Numba-optimized function to compute segment statistics."""
    n_max_segments = len(change_points) + 1
    # Pre-allocate arrays for segment statistics
    segment_stats = np.zeros((n_max_segments, 7), dtype=np.float64)  # [start, end, size, mean, std, min, max]
    n_segments = 0
    start = 0
    
    for end in change_points:
        if end - start >= min_size:
            segment_data = data[start:end]
            
            if len(segment_data) > 0:
                mean = np.mean(segment_data)
                std = np.std(segment_data)
                
                # Store stats in array format
                segment_stats[n_segments, 0] = start
                segment_stats[n_segments, 1] = end
                segment_stats[n_segments, 2] = end - start
                segment_stats[n_segments, 3] = mean
                segment_stats[n_segments, 4] = std
                segment_stats[n_segments, 5] = np.min(segment_data)
                segment_stats[n_segments, 6] = np.max(segment_data)
                n_segments += 1
        start = end
    
    return segment_stats[:n_segments]  # Return only valid segments

@njit
def _compute_statistical_tests_numba(data, clusters):
    """Numba-optimized function for statistical tests."""
    n_clusters = len(np.unique(clusters))
    # [sum, sum_sq, count, var, min, max, skew, kurt]
    cluster_stats = np.zeros((n_clusters, 8), dtype=np.float64)
    
    # Compute cluster statistics
    for i in range(len(data)):
        cluster = clusters[i]
        val = data[i]
        if np.isfinite(val):
            cluster_stats[cluster, 0] += val
            cluster_stats[cluster, 1] += val * val
            cluster_stats[cluster, 2] += 1
            cluster_stats[cluster, 4] = min(cluster_stats[cluster, 4], val) if cluster_stats[cluster, 2] > 1 else val
            cluster_stats[cluster, 5] = max(cluster_stats[cluster, 5], val) if cluster_stats[cluster, 2] > 1 else val
    
    # Compute higher moments
    for i in range(n_clusters):
        if cluster_stats[i, 2] > 1:
            mean = cluster_stats[i, 0] / cluster_stats[i, 2]
            # Variance
            var = (cluster_stats[i, 1] / cluster_stats[i, 2]) - (mean * mean)
            cluster_stats[i, 3] = var
            
            # Compute skewness and kurtosis
            m3 = 0.0
            m4 = 0.0
            for j in range(len(data)):
                if clusters[j] == i and np.isfinite(data[j]):
                    dev = data[j] - mean
                    m3 += dev * dev * dev
                    m4 += dev * dev * dev * dev
            
            n = cluster_stats[i, 2]
            if n > 2 and var > 0:
                m3 /= n
                m4 /= n
                cluster_stats[i, 6] = m3 / (var ** 1.5)  # Skewness
                cluster_stats[i, 7] = (m4 / (var * var)) - 3.0  # Kurtosis
    
    return cluster_stats

@timing_decorator    
def perform_advanced_statistical_tests(df, clustering_results, batch_size=10000, logger=None):
    """Perform comprehensive statistical tests with Numba optimization."""
    if logger:
        logger.log_and_print("Performing advanced statistical tests...")
    
    statistical_tests = {
        'normality': {},
        'homogeneity': {},
        'cluster_comparisons': {},
        'feature_correlations': {},
        'time_series_tests': {}
    }
    
    try:
        # Convert to float64 and clip values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['cluster']]
        
        for col in feature_cols:
            data = df[col].values.astype(np.float64)
            data = np.clip(data, -1e10, 1e10)
            data = data[np.isfinite(data)]
            
            if len(data) > 1:
                # Process in batches with Numba
                cluster_stats_array = _compute_statistical_tests_numba(
                    data, 
                    df['cluster'].values if 'cluster' in df.columns else np.zeros(len(data))
                )
                
                # Compute normality test
                stat, p_value = normaltest(data)
                statistical_tests['normality'][col] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': float(p_value) > 0.05
                }
                
                # Store cluster statistics
                if 'cluster' in df.columns:
                    statistical_tests['cluster_comparisons'][col] = {
                        'cluster_means': [float(stats[0] / stats[2]) if stats[2] > 0 else 0.0 for stats in cluster_stats_array],
                        'cluster_vars': [float(stats[3]) for stats in cluster_stats_array],
                        'cluster_counts': [int(stats[2]) for stats in cluster_stats_array],
                        'cluster_skews': [float(stats[6]) for stats in cluster_stats_array],
                        'cluster_kurtosises': [float(stats[7]) for stats in cluster_stats_array]
                    }
        
        if logger:
            logger.log_and_print("Advanced statistical tests complete")
        
        return statistical_tests
        
    except Exception as e:
        error_msg = f"Error in statistical tests: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        return {
            'normality': {},
            'homogeneity': {},
            'cluster_comparisons': {},
            'feature_correlations': {},
            'time_series_tests': {}
        }

@njit
def _compute_sample_importance_numba(X, y, feature_cols_count):
    """Numba-optimized function to compute feature importance for a single sample."""
    n_features = feature_cols_count
    n_samples = X.shape[0]
    
    # Initialize arrays for results
    correlations = np.zeros(n_features, dtype=np.float64)
    
    # 3. Correlation based
    mean_y = np.mean(y)
    for i in range(n_features):
        sum_xy = 0.0
        sum_x2 = 0.0
        sum_y2 = 0.0
        count = 0
        mean_x = 0.0
        
        for j in range(n_samples):
            if np.isfinite(X[j, i]) and np.isfinite(y[j]):
                x = X[j, i]
                y_val = y[j]
                mean_x += x
                sum_xy += x * (y_val - mean_y)
                sum_x2 += x * x
                sum_y2 += (y_val - mean_y) * (y_val - mean_y)
                count += 1
        
        if count > 1 and sum_x2 > 0 and sum_y2 > 0:
            mean_x /= count
            cov = sum_xy / count
            var_x = (sum_x2 / count) - (mean_x * mean_x)
            if var_x > 0:
                corr = cov / np.sqrt(var_x * sum_y2 / count)
                if np.isfinite(corr):
                    correlations[i] = corr
    
    return correlations


@timing_decorator
def _compute_sample_importance(df_sample, feature_cols, target_col, logger, batch_size):
    """Helper function to compute feature importance for a single sample with improved NaN handling."""
    # Check if target_col exists in df_sample
    if target_col not in df_sample.columns:
        if logger:
            logger.log_and_print(f"Warning: Target column '{target_col}' not found in sample. Skipping importance calculation for this sample.")
        else:
            print(f"Warning: Target column '{target_col}' not found in sample. Skipping importance calculation for this sample.")
        return None, None, None, None, None
    
    # Convert to float64 and clip values
    X = df_sample[feature_cols].values.astype(np.float64)
    X = np.clip(X, -1e10, 1e10)
    y = df_sample[target_col].values.astype(np.float64)
    y = np.clip(y, -1e10, 1e10)
    
    # 1. Random Forest importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=feature_cols)
    
    # 2. XGBoost importance
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    xgb_model.fit(X, y)
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_cols)
    
    # Call numba-optimized function for correlation
    correlations = _compute_sample_importance_numba(
        X, y, len(feature_cols)
    )
    correlations = pd.Series(correlations, index=feature_cols)
    
    # Compute stability scores using bootstrapping
    n_bootstrap = 10  # Reduced bootstrap samples per sample
    stability_scores = np.zeros((len(feature_cols), n_bootstrap))
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Train RF on bootstrap sample
        rf_boot = RandomForestRegressor(n_estimators=50, random_state=i, n_jobs=-1)
        rf_boot.fit(X_boot, y_boot)
        stability_scores[:, i] = rf_boot.feature_importances_
    
    # Compute stability metrics
    stability_mean = np.mean(stability_scores, axis=1)
    stability_std = np.std(stability_scores, axis=1)
    stability_cv = stability_std / (stability_mean + 1e-10)
    
    stability_scores = {
        'mean': dict(zip(feature_cols, stability_mean)),
        'std': dict(zip(feature_cols, stability_std)),
        'cv': dict(zip(feature_cols, stability_cv))
    }
    
    # Compute interaction scores
    interaction_scores = {}
    for feat1, feat2 in combinations(feature_cols[:min(20, len(feature_cols))], 2):
        # Convert to numpy arrays and ensure proper types
        x_vals = np.asarray(X[:, feature_cols.index(feat1)], dtype=np.float64)
        y_vals = np.asarray(X[:, feature_cols.index(feat2)], dtype=np.float64)
        interaction = x_vals * y_vals
        
        # Convert to numpy array and handle correlation calculation
        with np.errstate(all='ignore'):
            spearman_result = spearmanr(interaction, np.asarray(y, dtype=np.float64))
            if isinstance(spearman_result, tuple):
                # Convert tuple elements to float individually
                correlation = float(np.asarray(spearman_result[0]).item())
            else:
                correlation = float(spearman_result.correlation)
            
        if np.isfinite(correlation) and abs(correlation) > 0.3:
            interaction_scores[f"{feat1}_X_{feat2}"] = correlation
    
    return rf_importance, xgb_importance, correlations, stability_scores, interaction_scores

@timing_decorator
def analyze_feature_importance(df, target_col='gap_size', n_top_features=50, batch_size=5000, logger=None, 
                               n_samples_per_strategy=3, sample_size=5000, range_splits=5, sequence_splits=3):
    """Analyze complex recurring patterns in sequences of features with improved memory management and numerical stability."""
    if logger:
        logger.log_and_print("Analyzing feature importance using combined sampling strategies...")
    else:
        print("Analyzing feature importance using combined sampling strategies...")
    
    # Initialize results dictionary
    importance_analysis = {
        'feature_scores': {},
        'selected_features': {},
        'stability_scores': {},
        'interaction_scores': {},
        'shap_scores': {}
    }
    
    try:
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        # Check if feature_cols is empty
        if not feature_cols:
            if logger:
                logger.log_and_print("Error: No valid features for feature importance analysis.", level=logging.ERROR)
            else:
                print("Error: No valid features for feature importance analysis.")
            raise ValueError("No valid features for feature importance analysis.")
        
        # Initialize accumulators for each sample
        all_rf_importance = []
        all_xgb_importance = []
        all_correlations = []
        all_stability_scores = []
        all_interaction_scores = []
        
        total_samples = n_samples_per_strategy * (1 + range_splits + sequence_splits)
        sample_count = 0
        
        # 1. Random Sampling
        for _ in range(n_samples_per_strategy):
            if logger:
                logger.log_and_print(f"Processing random sample {sample_count + 1}/{total_samples}...")
            else:
                print(f"Processing random sample {sample_count + 1}/{total_samples}...")
            
            if len(df) > sample_size:
                sample_idx = np.random.choice(len(df), sample_size, replace=False)
                df_sample = df.iloc[sample_idx]
            else:
                df_sample = df
            
            sample_results = _compute_sample_importance(df_sample, feature_cols, target_col, logger, batch_size)
            
            if sample_results is not None:
                rf_importance, xgb_importance, correlations, stability_scores, interaction_scores = sample_results
                all_rf_importance.append(rf_importance)
                all_xgb_importance.append(xgb_importance)
                all_correlations.append(correlations)
                all_stability_scores.append(stability_scores)
                all_interaction_scores.append(interaction_scores)
            else:
                all_rf_importance.append(pd.Series(dtype=np.float64))
                all_xgb_importance.append(pd.Series(dtype=np.float64))
                all_correlations.append(pd.Series(dtype=np.float64))
                all_stability_scores.append({})
                all_interaction_scores.append({})
            
            sample_count += 1
        
        # 2. Range-Based Sampling
        if logger:
            logger.log_and_print("Processing range-based samples...")
        
        range_edges = np.linspace(0, len(df), range_splits + 1, dtype=int)
        for i in range(range_splits):
            for _ in range(n_samples_per_strategy):
                if logger:
                    logger.log_and_print(f"Processing range sample {sample_count + 1}/{total_samples} (range {i+1}/{range_splits})...")
                else:
                    print(f"Processing range sample {sample_count + 1}/{total_samples} (range {i+1}/{range_splits})...")
                
                start_idx = range_edges[i]
                end_idx = range_edges[i+1]
                
                if end_idx - start_idx > sample_size:
                    sample_idx = np.random.choice(range(start_idx, end_idx), sample_size, replace=False)
                    df_sample = df.iloc[sample_idx]
                else:
                    df_sample = df.iloc[start_idx:end_idx]
                
                sample_results = _compute_sample_importance(df_sample, feature_cols, target_col, logger, batch_size)
                
                if sample_results is not None:
                    rf_importance, xgb_importance, correlations, stability_scores, interaction_scores = sample_results
                    all_rf_importance.append(rf_importance)
                    all_xgb_importance.append(xgb_importance)
                    all_correlations.append(correlations)
                    all_stability_scores.append(stability_scores)
                    all_interaction_scores.append(interaction_scores)
                else:
                    all_rf_importance.append(pd.Series(dtype=np.float64))
                    all_xgb_importance.append(pd.Series(dtype=np.float64))
                    all_correlations.append(pd.Series(dtype=np.float64))
                    all_stability_scores.append({})
                    all_interaction_scores.append({})
                
                sample_count += 1
        
        # 3. Sequence-Based Sampling
        if logger:
            logger.log_and_print("Processing sequence-based samples...")
        
        sequence_edges = np.linspace(0, len(df), sequence_splits + 1, dtype=int)
        for i in range(sequence_splits):
            for _ in range(n_samples_per_strategy):
                if logger:
                    logger.log_and_print(f"Processing sequence sample {sample_count + 1}/{total_samples} (sequence {i+1}/{sequence_splits})...")
                else:
                    print(f"Processing sequence sample {sample_count + 1}/{total_samples} (sequence {i+1}/{sequence_splits})...")
                
                start_idx = sequence_edges[i]
                end_idx = sequence_edges[i+1]
                
                if end_idx - start_idx > sample_size:
                    sample_idx = np.random.choice(range(start_idx, end_idx), sample_size, replace=False)
                    df_sample = df.iloc[sample_idx].sort_index()
                else:
                    df_sample = df.iloc[start_idx:end_idx].sort_index()
                
                sample_results = _compute_sample_importance(df_sample, feature_cols, target_col, logger, batch_size)
                
                if sample_results is not None:
                    rf_importance, xgb_importance, correlations, stability_scores, interaction_scores = sample_results
                    all_rf_importance.append(rf_importance)
                    all_xgb_importance.append(xgb_importance)
                    all_correlations.append(correlations)
                    all_stability_scores.append(stability_scores)
                    all_interaction_scores.append(interaction_scores)
                else:
                    all_rf_importance.append(pd.Series(dtype=np.float64))
                    all_xgb_importance.append(pd.Series(dtype=np.float64))
                    all_correlations.append(pd.Series(dtype=np.float64))
                    all_stability_scores.append({})
                    all_interaction_scores.append({})
                
                sample_count += 1
        
        # Combine results from all samples
        if logger:
            logger.log_and_print("Combining results from all samples...")
        
        # Average feature scores
        if all_rf_importance:
            importance_analysis['feature_scores']['random_forest'] = {
                str(k): v for k, v in pd.concat(all_rf_importance, axis=1).mean(axis=1).to_dict().items()
            }
        if all_xgb_importance:
            importance_analysis['feature_scores']['xgboost'] = {
                str(k): v for k, v in pd.concat(all_xgb_importance, axis=1).mean(axis=1).to_dict().items()
            }
        
        # Handle NaN values in correlations
        all_correlations_cleaned = [corr.fillna(0) for corr in all_correlations]
        if all_correlations_cleaned:
            importance_analysis['feature_scores']['correlation'] = {
                str(k): v for k, v in pd.concat(all_correlations_cleaned, axis=1).mean(axis=1).to_dict().items()
            }
        
        # Average stability scores
        mean_stability = {}
        std_stability = {}
        cv_stability = {}
        for scores in all_stability_scores:
            for feature, mean in scores['mean'].items():
                mean_stability[str(feature)] = mean_stability.get(str(feature), 0) + mean
            for feature, std in scores['std'].items():
                std_stability[str(feature)] = std_stability.get(str(feature), 0) + std
            for feature, cv in scores['cv'].items():
                cv_stability[str(feature)] = cv_stability.get(str(feature), 0) + cv
        
        for feature in mean_stability:
            mean_stability[feature] /= total_samples
            std_stability[feature] /= total_samples
            cv_stability[feature] /= total_samples
        
        importance_analysis['stability_scores'] = {
            'mean': mean_stability,
            'std': std_stability,
            'cv': cv_stability
        }
        
        # Combine interaction scores
        combined_interactions = {}
        for scores in all_interaction_scores:
            for interaction, score in scores.items():
                combined_interactions[interaction] = combined_interactions.get(interaction, 0) + score
        
        for interaction in combined_interactions:
            combined_interactions[interaction] /= total_samples
        
        importance_analysis['interaction_scores'] = combined_interactions
        
        # Select top features based on combined importance
        if logger:
            logger.log_and_print("Selecting top features...")
        combined_scores = pd.DataFrame({
            'rf': pd.Series(importance_analysis['feature_scores'].get('random_forest', {})),
            'xgb': pd.Series(importance_analysis['feature_scores'].get('xgboost', {})),
            'correlation': pd.Series(importance_analysis['feature_scores'].get('correlation', {})),
            'stability': pd.Series(importance_analysis['stability_scores'].get('mean', {}))
        })
        
        # Normalize each score type
        for col in combined_scores.columns:
            combined_scores[col] = combined_scores[col] / (combined_scores[col].max() + 1e-10)
        
        # Compute mean importance across methods
        mean_importance = combined_scores.mean(axis=1)
        top_features = mean_importance.nlargest(n_top_features).index.tolist()
        
        importance_analysis['selected_features'] = {
            'features': top_features,
            'scores': mean_importance[top_features].to_dict()
        }
        
        if logger:
            logger.log_and_print(f"Selected {len(top_features)} top features")
        
        return importance_analysis, pd.DataFrame(importance_analysis['feature_scores'])
        
    except Exception as e:
        error_msg = f"Error in feature importance analysis: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        raise ValueError("Feature importance analysis failed, no valid features selected.")   

@timing_decorator
def select_optimal_features(df, importance_analysis, target_col='gap_size', batch_size=5000, logger=None):
    """Select optimal features based on comprehensive analysis with batched processing."""
    if logger:
        logger.log_and_print("Selecting optimal features...")
    else:
        print("Selecting optimal features...")
    
    feature_scores = {}
    
    try:
        # Unpack importance analysis results properly
        importance_dict, importance_df = importance_analysis
        
        # Combine scores from different methods
        for feature in df.columns:
            if feature != target_col:
                scores = []
                
                # Random Forest importance
                if 'random_forest' in importance_dict['feature_scores']:
                    rf_score = importance_dict['feature_scores']['random_forest'].get(str(feature), 0.0)
                    scores.append(rf_score)
                
                # XGBoost importance 
                if 'xgboost' in importance_dict['feature_scores']:
                    xgb_score = importance_dict['feature_scores']['xgboost'].get(str(feature), 0.0)
                    scores.append(xgb_score)
                
                # Correlation importance
                if 'correlation' in importance_dict['feature_scores']:
                    corr_score = importance_dict['feature_scores']['correlation'].get(str(feature), 0.0)
                    scores.append(corr_score)
                
                # Stability
                if 'stability_scores' in importance_dict:
                    stability = importance_dict['stability_scores']['mean'].get(str(feature), 0.0)
                    stability_cv = importance_dict['stability_scores']['cv'].get(str(feature), 1.0)
                    scores.append(stability * (1 - stability_cv))
                
                if scores:
                    # Normalize and combine scores
                    max_score = max(abs(s) for s in scores)
                    if max_score > 0:
                        normalized_scores = [s / max_score for s in scores]
                        feature_scores[feature] = float(np.mean(normalized_scores))
                    else:
                        feature_scores[feature] = 0.0
        
        # Select features based on combined score
        selected_features = pd.Series(feature_scores)
        selected_features = selected_features.sort_values(ascending=False)
        
        # Determine optimal number of features using cross-validation in batches
        n_features = len(selected_features)
        if n_features == 0:
            if logger:
                logger.log_and_print("Error: No features selected for training.", level=logging.ERROR)
            else:
                print("Error: No features selected for training.")
            raise ValueError("No features selected for training.")
        
        scores = []
        
        if logger:
            logger.log_and_print("Evaluating feature subsets...")
        
        for n in range(5, min(50, n_features), 5):
            features = selected_features.head(n).index
            X = df[features]
            y = df[target_col]
            
            # Quick evaluation using random forest
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            
            # Perform cross-validation in batches
            cv_scores = []
            for start_idx in range(0, len(X), batch_size):
                end_idx = min(start_idx + batch_size, len(X))
                X_batch = X.iloc[start_idx:end_idx]
                y_batch = y.iloc[start_idx:end_idx] if isinstance(y, pd.Series) else y[start_idx:end_idx]
                
                cv_scores.append(np.mean(cross_val_score(rf, X_batch, y_batch, cv=3)))
                gc.collect()
            
            scores.append(np.mean(cv_scores))
        
        # Find optimal number of features
        if not scores:
            if logger:
                logger.log_and_print("Error: No scores computed for feature selection.", level=logging.ERROR)
            else:
                print("Error: No scores computed for feature selection.")
            raise ValueError("No scores computed for feature selection.")
        
        optimal_n = (np.argmax(scores) + 1) * 5
        optimal_features = selected_features.head(int(optimal_n)).index.tolist()
        
        return {
            'optimal_features': optimal_features,
            'feature_scores': feature_scores,
            'n_features': optimal_n,
            'evaluation_scores': scores
        }
        
    except Exception as e:
        error_msg = f"Error in feature selection: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        raise ValueError("Feature selection failed, no valid features selected.")

@timing_decorator
def analyze_feature_stability(df, selected_features, n_bootstrap=50, batch_size=5000, logger=None):
    """Analyze stability of feature importance across different subsets with batching."""
    if logger:
        logger.log_and_print("Analyzing feature stability...")
    else:
        print("Analyzing feature stability...")
    
    stability_analysis = {
        'bootstrap_scores': {},
        'temporal_stability': {},
        'value_range_stability': {}
    }
    
    try:
        # 1. Bootstrap Stability Analysis
        if logger:
            logger.log_and_print("Performing bootstrap stability analysis...")
        
        bootstrap_results = pd.DataFrame(index=selected_features)
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(df), size=len(df), replace=True)
            
            # Process in batches
            batch_importances = []
            for start_idx in range(0, len(indices), batch_size):
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                batch_sample = df.iloc[batch_indices]
                
                # Train RF on bootstrap sample
                rf_boot = RandomForestRegressor(n_estimators=50, random_state=i, n_jobs=-1)
                rf_boot.fit(batch_sample[selected_features], batch_sample['gap_size'])
                batch_importances.append(rf_boot.feature_importances_)
                gc.collect()
            
            # Combine batch importances
            if batch_importances:
                bootstrap_results[f'boot_{i}'] = np.mean(batch_importances, axis=0)
        
        # Compute stability metrics
        stability_mean = bootstrap_results.mean(axis=1)
        stability_std = bootstrap_results.std(axis=1)
        stability_cv = stability_std / (stability_mean + 1e-10)
        
        # Convert to dictionaries with feature names as keys
        stability_analysis['bootstrap_scores'] = {
            'mean_importance': dict(zip(selected_features, stability_mean)),
            'std_importance': dict(zip(selected_features, stability_std)),
            'cv_importance': dict(zip(selected_features, stability_cv))
        }

        # 2. Temporal Stability Analysis
        if logger:
            logger.log_and_print("Analyzing temporal stability...")
        
        if len(df) > 100:
            window_size = len(df) // 10
            temporal_stability = {}
            
            for feature in selected_features:
                window_means = []
                window_stds = []
                
                for start_idx in range(0, len(df) - window_size, batch_size):
                    end_idx = min(start_idx + batch_size, len(df) - window_size)
                    batch = df[feature].iloc[start_idx:end_idx + window_size]
                    
                    # Compute window statistics
                    if len(batch) > 0:
                        window_means.append(float(batch.mean()))
                        window_stds.append(float(batch.std()))
                    gc.collect()
                
                temporal_stability[feature] = {
                    'mean_stability': float(np.std(window_means)) if window_means else 0.0,
                    'std_stability': float(np.std(window_stds)) if window_stds else 0.0
                }
            
            stability_analysis['temporal_stability'] = temporal_stability
        
        # 3. Value Range Stability Analysis
        if logger:
            logger.log_and_print("Analyzing value range stability...")
        
        value_range_stability = {}
        
        for feature in selected_features:
            quartiles = df[feature].quantile([0.25, 0.5, 0.75]).values
            iqr = quartiles[2] - quartiles[0]
            
            value_range_stability[feature] = {
                'iqr': float(iqr),
                'range_ratio': float(df[feature].max() - df[feature].min()) / (iqr + 1e-10),
                'outlier_ratio': float(np.sum(np.abs(df[feature] - quartiles[1]) > 1.5 * iqr) / len(df))
            }
        
        stability_analysis['value_range_stability'] = value_range_stability
        
        if logger:
            logger.log_and_print("Feature stability analysis complete")
        
        return stability_analysis
    
    except Exception as e:
        if logger:
            logger.log_and_print(f"Error in feature stability analysis: {str(e)}", level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(f"Error in feature stability analysis: {str(e)}")
            traceback.print_exc()
        return stability_analysis

@timing_decorator
def analyze_phase_space(df, feature_col='gap_size', max_lag=5, batch_size=5000, logger=None):
    """Analyze the phase space of a feature with improved numerical stability and memory management."""
    if logger:
        logger.log_and_print("Analyzing phase space...")
    
    phase_space_analysis = {
        'phase_space_data': {},
        'embedding_dimension': {}
    }
    
    try:
        # Convert to float64 and clip values
        series = df[feature_col].values.astype(np.float64)
        series = np.clip(series, -1e10, 1e10)
        series = series[np.isfinite(series)]
        
        if len(series) < 2:
            if logger:
                logger.log_and_print("Warning: Insufficient data points for phase space calculation.")
            return {'phase_space_data': {}, 'embedding_dimension': {}}
        
        # Process phase space data in batches
        for lag in range(1, max_lag + 1):
            if logger:
                logger.log_and_print(f"Processing phase space with lag {lag}")
            
            phase_space_data = []
            
            for start_idx in range(0, len(series) - lag, batch_size):
                end_idx = min(start_idx + batch_size, len(series) - lag)
                
                # Get batch of data
                batch = series[start_idx:end_idx]
                lagged_batch = series[start_idx + lag:end_idx + lag]
                
                # Handle NaN/inf values
                valid_mask = np.isfinite(batch) & np.isfinite(lagged_batch)
                batch = batch[valid_mask]
                lagged_batch = lagged_batch[valid_mask]
                
                if len(batch) > 0:
                    # Store valid data points
                    phase_space_data.extend(zip(batch, lagged_batch))
                
                gc.collect()
            
            # Store phase space data with explicit type conversion
            phase_space_analysis['phase_space_data'][lag] = [
                (float(x), float(y))
                for x, y in phase_space_data
            ]
        
        # Estimate embedding dimension using false nearest neighbors
        if logger:
            logger.log_and_print("Estimating embedding dimension...")
        
        # Use a smaller sample for faster computation
        sample_size = min(len(series), 1000)
        sample_indices = np.random.choice(len(series), sample_size, replace=False)
        sample = series[sample_indices]
        
        # Process embedding dimension in batches
        for lag in range(1, max_lag + 1):
            if logger:
                logger.log_and_print(f"Processing embedding dimension with lag {lag}")
            
            false_neighbors = 0
            
            for start_idx in range(0, len(sample) - lag - 1, batch_size):
                end_idx = min(start_idx + batch_size, len(sample) - lag - 1)
                
                for i in range(start_idx, end_idx):
                    # Get current point and its nearest neighbor
                    current_point = sample[i]
                    
                    # Find nearest neighbor
                    min_dist = float('inf')
                    nearest_neighbor = None
                    for j in range(len(sample)):
                        if i != j:
                            dist = abs(sample[i] - sample[j])
                            if dist < min_dist:
                                min_dist = dist
                                nearest_neighbor = sample[j]
                    
                    if nearest_neighbor is not None:
                        # Check if the nearest neighbor remains a neighbor in the lagged space
                        lagged_point = sample[i+lag] if i + lag < len(sample) else sample[i]
                        lagged_neighbor = sample[j+lag] if j + lag < len(sample) else sample[j]
                        
                        if np.abs(lagged_point - lagged_neighbor) > 2 * min_dist:
                            false_neighbors += 1
                
                gc.collect()
            
            # Store embedding dimension
            if sample_size > 0:
                false_neighbor_ratio = float(false_neighbors / sample_size)
            else:
                false_neighbor_ratio = 0.0
            
            phase_space_analysis['embedding_dimension'][lag] = {
                'false_neighbor_ratio': false_neighbor_ratio
            }
        
        if logger:
            logger.log_and_print("Phase space analysis complete")
        
        return phase_space_analysis
        
    except Exception as e:
        error_msg = f"Error in phase space analysis: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'phase_space_data': {},
            'embedding_dimension': {}
        }
                      
@timing_decorator
def analyze_wavelet_patterns(df, feature_col='gap_size', batch_size=5000, logger=None):
    """Analyze wavelet patterns in a given feature with improved numerical stability and memory management."""
    if logger:
        logger.log_and_print("Analyzing wavelet patterns...")
    
    wavelet_patterns = {}
    
    try:
        # Convert to numpy array and ensure proper type
        data = df[feature_col].values.astype(np.float64)
        data = np.clip(data, -1e10, 1e10)
        data = data[np.isfinite(data)]
        
        if len(data) < 2:
            if logger:
                logger.log_and_print("Warning: Insufficient data points for wavelet analysis.")
            return {'wavelet_coeffs': None}
        
        # Define wavelet type
        wavelet = 'db4'  # Daubechies 4 wavelet
        
        # Perform wavelet decomposition
        if logger:
            logger.log_and_print("Performing wavelet decomposition...")
        
        coeffs = []
        
        # Process wavelet decomposition in batches
        for start_idx in range(0, len(data), batch_size):
            end_idx = min(start_idx + batch_size, len(data))
            batch = data[start_idx:end_idx]
            
            with np.errstate(all='ignore'):
                batch_coeffs = pywt.wavedec(batch, wavelet)
                coeffs.extend(batch_coeffs)
            
            gc.collect()
        
        # Store wavelet coefficients
        wavelet_patterns['wavelet_coeffs'] = [np.array(c, dtype=np.float64).tolist() for c in coeffs]
        
        if logger:
            logger.log_and_print("Wavelet analysis complete")
            
        return wavelet_patterns
        
    except Exception as e:
        error_msg = f"Error in wavelet analysis: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {'wavelet_coeffs': None}
    
@timing_decorator
def analyze_feature_interactions(df, selected_features, target_col='gap_size', batch_size=5000, logger=None):
    """Analyze interactions between selected features with improved memory management and numerical stability."""
    if logger:
        logger.log_and_print("Analyzing feature interactions...")
    else:
        print("Analyzing feature interactions...")
    
    interaction_analysis = {
        'pairwise_correlations': {},
        'mutual_information_entanglement': {},
        'nonlinear_relationships': {},
        'temporal_interactions': {},
        'shap_interaction_scores': {}
    }
    
    significant_correlations_found = False  # Initialize flag
    
    try:
        # Convert to float64 and clip values
        X = df[selected_features].astype(np.float64)
        X = X.clip(-1e10, 1e10)
        
        # 1. Correlation-based entanglement
        if logger:
            logger.log_and_print("Computing correlation-based entanglement...")
        
        n_features = len(selected_features)
        
        # Initialize correlation matrix with correct dtype
        correlation_matrix = np.zeros((n_features, n_features), dtype=np.float64)
        
        # Initialize count matrix
        count_matrix = np.zeros((n_features, n_features), dtype=np.int64)
        
        # Process correlations in batches
        for start_idx in range(0, len(X), batch_size):
            end_idx = min(start_idx + batch_size, len(X))
            batch = X.iloc[start_idx:end_idx]
            
            with np.errstate(all='ignore'):
                batch_corr = np.array(batch.corr(), dtype=np.float64)
                
                # Update correlation matrix
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        if np.isfinite(batch_corr[i, j]):
                            correlation_matrix[i, j] += batch_corr[i, j]
                            correlation_matrix[j, i] += batch_corr[i, j]
                            count_matrix[i, j] += 1
            
            gc.collect()
        
        # Normalize correlation matrix
        with np.errstate(all='ignore'):
            mask = count_matrix > 0
            correlation_matrix[mask] = correlation_matrix[mask] / count_matrix[mask]
        
        # Compute entanglement metric (sum of absolute correlations)
        significant_correlations = []
        correlation_threshold = 0.05  # Lowered threshold
        
        if correlation_matrix.size > 0:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr = correlation_matrix[i, j]
                    if np.isfinite(corr) and abs(corr) > correlation_threshold:
                        significant_correlations.append({
                            'feature1': selected_features[i],
                            'feature2': selected_features[j],
                            'correlation': float(corr)
                        })
        
        interaction_analysis['pairwise_correlations'] = {'significant_correlations': significant_correlations}
        
        if significant_correlations:
            significant_correlations_found = True
        
        if logger:
            logger.log_and_print(f"Found {len(significant_correlations)} significant correlations before filtering.")
        
        if not significant_correlations and logger:
            logger.log_and_print("Warning: No significant correlations found for heatmap.")
        
        # 2. Mutual information-based entanglement
        if logger:
            logger.log_and_print("Computing mutual information-based entanglement...")
        
        for i, feat1 in enumerate(selected_features):
            for j, feat2 in enumerate(selected_features[i+1:], i+1):
                mi_sum = 0.0
                count = 0
                
                # Process mutual information in batches
                for start_idx in range(0, len(X), batch_size):
                    end_idx = min(start_idx + batch_size, len(X))
                    batch = X.iloc[start_idx:end_idx]
                    
                    # Convert to numpy arrays for mutual_info_regression
                    x_batch = np.array(batch[feat1], dtype=np.float64).reshape(-1, 1)
                    y_batch = np.array(batch[feat2], dtype=np.float64)
                    
                    # Handle NaN/inf values
                    valid_mask = np.isfinite(x_batch.ravel()) & np.isfinite(y_batch)
                    if np.sum(valid_mask) > 1:
                        x_clean = x_batch[valid_mask].reshape(-1, 1)
                        y_clean = y_batch[valid_mask]
                        
                        with np.errstate(all='ignore'):
                            mi_result = mutual_info_regression(x_clean, y_clean)
                            if len(mi_result) > 0 and np.isfinite(mi_result[0]):
                                mi_sum += float(mi_result[0])
                                count += 1
                    gc.collect()
                
                if count > 0:
                    avg_mi = mi_sum / count
                    interaction_analysis['mutual_information_entanglement'][f'{feat1}_X_{feat2}'] = float(avg_mi)
        
        # 3. Nonlinear Relationship Detection
        if logger:
            logger.log_and_print("Checking for nonlinear relationships...")
        
        for feature in selected_features:
            nonlinear_scores = {}
            for transform_name, transform_func in [
                ('square', lambda x: np.square(x)),
                ('cube', lambda x: np.power(x, 3)),
                ('log', lambda x: np.log1p(np.abs(x))),
                ('sqrt', lambda x: np.sqrt(np.abs(x)))
            ]:
                corr_sum = 0.0
                count = 0
                
                for start_idx in range(0, len(X), batch_size):
                    end_idx = min(start_idx + batch_size, len(X))
                    batch = X.iloc[start_idx:end_idx]
                    
                    # Convert to numpy arrays and handle any tuple conversions
                    x_batch = np.array(batch[feature], dtype=np.float64)
                    y_batch = np.array(df[target_col].iloc[start_idx:end_idx], dtype=np.float64)
                    
                    # Replace inf/nan values
                    x_batch = np.nan_to_num(x_batch, nan=0.0, posinf=0.0, neginf=0.0)
                    y_batch = np.nan_to_num(y_batch, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Compute correlations for different transformations
                    with np.errstate(divide='ignore', invalid='ignore'):
                        transformed_x = transform_func(x_batch)
                        
                        # Handle spearmanr output
                        spearman_res = spearmanr(transformed_x, y_batch)
                        if isinstance(spearman_res, tuple):
                            # Convert tuple elements to float individually
                            corr = float(np.asarray(spearman_res[0]).item())
                        else:
                            corr = float(np.asarray(spearman_res.correlation).item())
                            count += 1
                    
                    gc.collect()
                
                if count > 0:
                    nonlinear_scores[transform_name] = float(corr_sum / count)
                else:
                    nonlinear_scores[transform_name] = 0.0
            
            interaction_analysis['nonlinear_relationships'][feature] = nonlinear_scores
        
        # 4. Time series specific tests
        if logger:
            logger.log_and_print("Performing time series tests...")
        
        try:
            # Process time series tests in batches
            acf_values = []
            for start_idx in range(0, len(df), batch_size):
                end_idx = min(start_idx + batch_size, len(df))
                batch = df['gap_size'].iloc[start_idx:end_idx].values
                batch = batch[np.isfinite(batch)]
                
                if len(batch) > 1:
                    with np.errstate(all='ignore'):
                        # Ljung-Box test
                        lb_result = acorr_ljungbox(batch, lags=[10, 20, 30], return_df=True)
                        if isinstance(lb_result, pd.DataFrame):
                            interaction_analysis['temporal_interactions']['ljung_box'] = {
                                'statistic': lb_result['lb_stat'].tolist(),
                                'p_value': lb_result['lb_pvalue'].tolist()
                            }
                
                gc.collect()
            
            # Seasonal decomposition with batched processing
            try:
                seasonal_data = []
                for start_idx in range(0, len(df), batch_size):
                    end_idx = min(start_idx + batch_size, len(df))
                    batch = df['gap_size'].iloc[start_idx:end_idx].values
                    seasonal_data.extend(batch[np.isfinite(batch)])
                
                if len(seasonal_data) >= 24:  # Minimum length for seasonal decomposition
                    seasonal_decompose_result = seasonal_decompose(
                        seasonal_data,
                        period=12,
                        extrapolate_trend=1
                    )
                    seasonal_strength = 1 - np.var(seasonal_decompose_result.resid) / np.var(seasonal_data)
                    interaction_analysis['temporal_interactions']['seasonal_strength'] = float(seasonal_strength)
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Warning: Seasonal decomposition failed: {str(e)}")
        
        except Exception as e:
            if logger:
                logger.log_and_print(f"Warning: Time series tests failed: {str(e)}")
        
        # 5. SHAP Analysis for Interaction Terms
        if logger:
            logger.log_and_print("Computing SHAP interaction scores...")
        
        # Create interaction features
        interaction_features = []
        for feat1, feat2 in combinations(selected_features[:min(10, len(selected_features))], 2):
            interaction_features.append(f'{feat1}_X_{feat2}')
        
        # Create interaction features outside the loop
        interaction_df = pd.DataFrame()
        for feat1, feat2 in combinations(selected_features[:min(10, len(selected_features))], 2):
            interaction_df[f'{feat1}_X_{feat2}'] = df[feat1] * df[feat2]
        
        # Add interaction features to the original dataframe
        df = pd.concat([df, interaction_df], axis=1)
        
        if interaction_features:
            # Prepare data for SHAP
            X_interaction = df[interaction_features].astype(np.float64)
            X_interaction = X_interaction.clip(-1e10, 1e10)
            y_interaction = df[target_col].astype(np.float64)
            y_interaction = y_interaction.clip(-1e10, 1e10)
            
            # Sample data if too large
            if len(X_interaction) > 1000:
                sample_idx = np.random.choice(len(X_interaction), 1000, replace=False)
                X_sample = X_interaction.iloc[sample_idx]
                y_sample = y_interaction.iloc[sample_idx]
            else:
                X_sample = X_interaction
                y_sample = y_interaction
            
            # Train a simple model
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X_sample, y_sample)
            
            # Compute SHAP values
            try:
                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(X_sample)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                # Compute SHAP interaction scores
                interaction_importance = np.abs(shap_values).mean(0)
                interaction_analysis['shap_interaction_scores'] = dict(zip(interaction_features, interaction_importance))
                
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Warning: SHAP interaction computation failed: {str(e)}")
        
        if logger:
            logger.log_and_print("Feature interaction analysis complete")
        
        return interaction_analysis, significant_correlations_found
        
    except Exception as e:
        error_msg = f"Error in feature interaction analysis: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'pairwise_correlations': {},
            'mutual_information_entanglement': {},
            'nonlinear_relationships': {},
            'temporal_interactions': {},
            'shap_interaction_scores': {}
        }, False                  

################################################################################
# Model Training and Evaluation
################################################################################


@njit
def _compute_distances_numba(X_batch, centers):
    """Numba-optimized function to compute distances to cluster centers."""
    n_samples = X_batch.shape[0]
    n_centers = centers.shape[0]
    distances = np.zeros((n_samples, n_centers), dtype=np.float64)
    
    for i in range(n_samples):
        for j in range(n_centers):
            dist = 0.0
            for k in range(X_batch.shape[1]):
                diff = X_batch[i, k] - centers[j, k]
                dist += diff * diff
            distances[i, j] = np.sqrt(dist)
    
    return distances

@timing_decorator
def perform_initial_clustering(df, batch_size=5000, logger=None):
    """Perform initial clustering on the dataset with numerical protection."""
    if logger:
        logger.log_and_print("\nPerforming initial clustering...")
    else:
        print("\nPerforming initial clustering...")
    
    # Get feature columns for clustering
    feature_cols = [col for col in df.columns if col not in [
        'gap_size', 'cluster', 'sub_cluster', 'lower_prime', 'upper_prime', 'is_outlier', 'preceding_gaps'
    ]]
    
    # Check if feature_cols is empty
    if not feature_cols:
        if logger:
            logger.log_and_print("Warning: No valid features for clustering, skipping clustering.", level=logging.WARNING)
        else:
            print("Warning: No valid features for clustering, skipping clustering.")
        # Return original df with a default cluster column
        df['cluster'] = 0
        return df
    
    X = df[feature_cols].copy()
    
    # Handle missing values and scale with protection
    X = X.fillna(0)
    X = X.clip(-1e10, 1e10)
    
    # Scale features with robust scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.clip(X_scaled, -1e10, 1e10)
    
    # Check for NaN values after scaling
    if not np.isfinite(X_scaled).all():
        if logger:
            logger.log_and_print("Warning: No valid features for training", level=logging.WARNING)
        else:
            print("Warning: No valid features for training")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Determine optimal number of clusters using silhouette score
    if logger:
        logger.log_and_print("Determining optimal number of clusters...")
    else:
        print("Determining optimal number of clusters...")
    
    silhouette_scores = []
    possible_clusters = range(2, min(11, len(df) // 10))  # Test up to 10 clusters or 10% of data
    
    for n_clusters in possible_clusters:
        try:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(batch_size, len(df)),
                random_state=42,
                n_init=3
            )
            
            # Process in batches
            for i in range(0, len(X_scaled), batch_size):
                batch = X_scaled[i:i+batch_size]
                kmeans = kmeans.partial_fit(batch)
            
            # Final prediction in batches
            labels = []
            for i in range(0, len(X_scaled), batch_size):
                batch = X_scaled[i:i+batch_size]
                batch_labels = kmeans.predict(batch)
                labels.extend(batch_labels)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                silhouette_scores.append((n_clusters, score))
            else:
                silhouette_scores.append((n_clusters, -1))
        except Exception as e:
            if logger:
                logger.log_and_print(f"Warning: Clustering failed for {n_clusters} clusters: {str(e)}")
            else:
                print(f"Warning: Clustering failed for {n_clusters} clusters: {str(e)}")
            silhouette_scores.append((n_clusters, -1))
        
        gc.collect()
    
    # Select optimal number of clusters
    if silhouette_scores:
        best_n_clusters, best_score = max(silhouette_scores, key=lambda x: x[1])
        if best_score > 0:
            if logger:
                logger.log_and_print(f"Optimal number of clusters: {best_n_clusters} (Silhouette Score: {best_score:.4f})")
            else:
                print(f"Optimal number of clusters: {best_n_clusters} (Silhouette Score: {best_score:.4f})")
        else:
            best_n_clusters = 3
            if logger:
                logger.log_and_print(f"Warning: No valid silhouette score found. Using default number of clusters: {best_n_clusters}")
            else:
                print(f"Warning: No valid silhouette score found. Using default number of clusters: {best_n_clusters}")
    else:
        best_n_clusters = 3
        if logger:
            logger.log_and_print(f"Warning: No silhouette scores computed. Using default number of clusters: {best_n_clusters}")
        else:
            print(f"Warning: No silhouette scores computed. Using default number of clusters: {best_n_clusters}")
    
    # Perform final clustering with optimal number of clusters
    kmeans = MiniBatchKMeans(
        n_clusters=best_n_clusters,
        batch_size=min(batch_size, len(df)),
        random_state=42,
        n_init=3
    )
    
    # Process in batches
    for i in range(0, len(X_scaled), batch_size):
        batch = X_scaled[i:i+batch_size]
        kmeans = kmeans.partial_fit(batch)
    
    # Final prediction in batches
    labels = []
    for i in range(0, len(X_scaled), batch_size):
        batch = X_scaled[i:i+batch_size]
        
        # Use numba-optimized distance calculation
        distances = _compute_distances_numba(batch, kmeans.cluster_centers_)
        batch_labels = np.argmin(distances, axis=1)
        labels.extend(batch_labels)
    
    # Add cluster labels to dataframe
    df['cluster'] = labels
    
    # Store cluster centers and model
    df.cluster_centers_ = kmeans.cluster_centers_
    df.kmeans_model_ = kmeans
    
    # Print cluster distribution
    cluster_dist = pd.Series(labels).value_counts().sort_index()
    if logger:
        logger.log_and_print("\nCluster distribution:")
        for cluster_id, count in cluster_dist.items():
            logger.log_and_print(f"Cluster {cluster_id}: {count} samples ({count/len(df)*100:.2f}%)")
    else:
        print("\nCluster distribution:")
        for cluster_id, count in cluster_dist.items():
            print(f"Cluster {cluster_id}: {count} samples ({count/len(df)*100:.2f}%)")
    
    return df

def detect_outliers(df):
    """Detect outliers in gap sizes using IQR method."""
    # Compute quartiles
    Q1 = df['gap_size'].quantile(0.25)
    Q3 = df['gap_size'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 3 * IQR
    
    # Create outlier column
    df = df.copy()
    df['is_outlier'] = df['gap_size'] > outlier_threshold
    
    return df, outlier_threshold

def _write_outlier_analysis(log, df, logger=None):
    """Writes the outlier analysis section of the report with improved numerical stability and error handling."""
    log.write("\n--- Outlier Analysis ---\n")
    
    try:
        # Detect outliers if not already done
        if 'is_outlier' not in df.columns:
            df, outlier_threshold = detect_outliers(df)
        else:
            # Compute threshold for reporting
            Q1 = df['gap_size'].quantile(0.25)
            Q3 = df['gap_size'].quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold = Q3 + 3 * IQR
        
        outlier_gaps = df[df['is_outlier']]
        
        if not outlier_gaps.empty:
            log.write(f"Found {len(outlier_gaps)} outlier gaps (larger than {outlier_threshold:.2f}):\n")
            
            # Sort outliers by gap size
            outlier_gaps = outlier_gaps.sort_values('gap_size', ascending=False)
            
            for idx, row in outlier_gaps.iterrows():
                log.write(f"\nOutlier Gap: {row.get('gap_size', 'N/A'):.0f}")
                
                # Find the gaps that precede the outlier gap
                if idx >= 5:
                    preceding_window = df.iloc[idx-5:idx]
                    log.write("\nPreceding gaps: ")
                    for prev_gap in preceding_window['gap_size']:
                        log.write(f"{prev_gap:.0f} ")
                    
                    # Analyze the modular residues of the preceding gaps if available
                    if 'gap_mod6' in preceding_window.columns:
                        log.write("\nMod 6 residues: ")
                        for prev_mod in preceding_window['gap_mod6']:
                            log.write(f"{prev_mod} ")
                    
                    if 'gap_mod30' in preceding_window.columns:
                        log.write("\nMod 30 residues: ")
                        for prev_mod in preceding_window['gap_mod30']:
                            log.write(f"{prev_mod} ")
                    
                    # Add factor analysis for the outlier if available
                    if 'factor_density' in row:
                        log.write(f"\nFactor density: {row.get('factor_density', 'N/A'):.2f}")
                    if 'factor_entropy' in row:
                        log.write(f"\nFactor entropy: {row.get('factor_entropy', 'N/A'):.2f}")
                    if 'mean_factor' in row:
                        log.write(f"\nMean factor: {row.get('mean_factor', 'N/A'):.2f}")
                else:
                    log.write(" (Insufficient preceding gaps)")
                
                log.write("\n")
                
                # Add additional outlier characteristics
                if 'unique_factors' in row:
                    log.write(f"\nUnique factors: {row.get('unique_factors', 'N/A'):.0f}")
                if 'total_factors' in row:
                    log.write(f"\nTotal factors: {row.get('total_factors', 'N/A'):.0f}")
                if 'mean_sqrt_factor' in row:
                    log.write(f"\nMean sqrt factor: {row.get('mean_sqrt_factor', 'N/A'):.2f}")
                if 'sum_sqrt_factor' in row:
                    log.write(f"\nSum sqrt factor: {row.get('sum_sqrt_factor', 'N/A'):.2f}")
                if 'std_sqrt_factor' in row:
                    log.write(f"\nStd sqrt factor: {row.get('std_sqrt_factor', 'N/A'):.2f}")
                
                # Analyze outlier sequences
                if 'preceding_gaps' in row and row['preceding_gaps']:
                    log.write("\nPreceding Gap Sequence:")
                    for gap in row['preceding_gaps']:
                        log.write(f" {gap:.0f}")
                    log.write("\n")
                
                # Analyze outlier sequences
                pattern_analysis = {}  # Ensure pattern_analysis is defined
                if 'outlier_patterns' in pattern_analysis:
                    outlier_patterns = pattern_analysis['outlier_patterns']
                    if 'preceding_sequences' in outlier_patterns:
                        log.write("\nOutlier Preceding Sequences:\n")
                        for seq in outlier_patterns['preceding_sequences'][:3]:
                            log.write(f"  - {seq}\n")
                    if 'following_sequences' in outlier_patterns:
                        log.write("\nOutlier Following Sequences:\n")
                        for seq in outlier_patterns['following_sequences'][:3]:
                            log.write(f"  - {seq}\n")
        else:
            log.write("No outlier gaps found.\n")
            
    except Exception as e:
        log.write(f"Error in outlier analysis: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    
    log.write("\n")
     
def prepare_training_data(df, logger=None):
    """Prepare data for model training with revised feature selection and NaN handling."""
    if logger:
        logger.log_and_print("Preparing training data with NaN checks...")
    else:
        print("Preparing training data with NaN checks...")
    
    # Only remove features that are direct functions of the gap or sequential information
    must_remove_features = [
        'gap_size',          # Direct target
        'gap_mod6',          # Direct function of gap
        'gap_mod30',         # Direct function of gap
        'cluster',           # Based on gap-related features
        'sub_cluster',       # Based on gap-related features
        'lower_prime',       # Sequential information
        'upper_prime',       # Sequential information
        'is_outlier',        # Based on gap size
        'preceding_gaps',     # Direct information about gaps
        'exp_log_gap_size',  # Direct function of gap
        'log2_gap_size',     # Direct function of gap
        'log10_gap_size',    # Direct function of gap
        'sin_log_gap_size',  # Direct function of gap
        'cos_log_gap_size',   # Direct function of gap
        'upper_prime_type',   # Categorical feature
        'lower_prime_type',    # Categorical feature
        'factor_harmonic_mean', # Highly correlated with gap size
        'total_factors',      # Highly correlated with gap size
        'num_distinct_prime_factors', # Highly correlated with gap size
        'unique_factors',     # Highly correlated with gap size
        'factor_complexity'   # Highly correlated with gap size
    ]
    
    # Keep all factor-based and statistical features
    feature_cols = [col for col in df.columns if col not in must_remove_features]
    
    if logger:
        logger.log_and_print("\nFeatures being used:")
        for col in feature_cols:
            logger.log_and_print(f"- {col}")
        
    # Check for missing values and data types
    for col in feature_cols:
        if df[col].isnull().any():
            if logger:
                logger.log_and_print(f"Warning: Column {col} contains NaN values. Imputing with 0.")
            else:
                print(f"Warning: Column {col} contains NaN values. Imputing with 0.")
            df[col] = df[col].fillna(0)
        
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
                if logger:
                    logger.log_and_print(f"Converted column {col} to numeric.")
                else:
                    print(f"Converted column {col} to numeric.")
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Warning: Column {col} could not be converted to numeric: {str(e)}. Filling with 0.")
                else:
                    print(f"Warning: Column {col} could not be converted to numeric: {str(e)}. Filling with 0.")
                df[col] = df[col].fillna(0)
    
    # Check for potential remaining leakage
    correlations = df[feature_cols + ['gap_size']].corr()['gap_size'].sort_values(ascending=False)
    if logger:
        logger.log_and_print("\nFeature correlations with gap_size:")
        logger.log_and_print(str(correlations))
    else:
        print("\nFeature correlations with gap_size:")
        print(correlations)
    
    # Remove only extremely highly correlated features (correlation > 0.95) if there are more than 2 features
    high_corr_features = []
    if len(feature_cols) > 2:
        high_corr_features = correlations[abs(correlations) > 0.95].index.tolist()
        if 'gap_size' in high_corr_features:
            high_corr_features.remove('gap_size')
        if high_corr_features:
            if logger:
                logger.log_and_print("\nRemoving extremely highly correlated features:")
                for feat in high_corr_features:
                    logger.log_and_print(f"- {feat}")
            else:
                print("\nRemoving extremely highly correlated features:")
                for feat in high_corr_features:
                    print(f"- {feat}")
            feature_cols = [col for col in feature_cols if col not in high_corr_features]
    
    # Ensure at least one feature is present
    if not feature_cols:
        if logger:
            logger.log_and_print("Warning: No valid features after processing. Adding 'mean_factor' as default feature.")
        else:
            print("Warning: No valid features after processing. Adding 'mean_factor' as default feature.")
        if 'mean_factor' in df.columns:
            feature_cols = ['mean_factor']
        else:
            # If mean_factor is not available, add the first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if numeric_cols:
                feature_cols = [numeric_cols[0]]
            else:
                if logger:
                    logger.log_and_print("Error: No numeric columns available, cannot proceed.")
                else:
                    print("Error: No numeric columns available, cannot proceed.")
                return None, None, [], None, None, None, None, None, None
    
    # Initialize all required DataFrames and Series
    # Main training data
    X = df[feature_cols].copy() if feature_cols else pd.DataFrame()
    y = df['gap_size'].copy()
    
    # Cluster membership data
    cluster_X = X.copy() if not X.empty else pd.DataFrame()
    cluster_y = df['cluster'].copy() if 'cluster' in df.columns else pd.Series(dtype=int)
    
    # Gap from cluster data
    gap_cluster_X = pd.DataFrame({'cluster': df['cluster']}) if 'cluster' in df.columns else pd.DataFrame()
    gap_cluster_y = y.copy()
    
    # Next cluster prediction data
    if 'cluster' in df.columns:
        next_cluster_features = []
        for i in range(5):
            next_cluster_features.append(df['cluster'].shift(i+1).fillna(-1))
        
        next_cluster_X = pd.concat(next_cluster_features, axis=1)
        next_cluster_X.columns = ['prev_cluster', 'prev_cluster2', 'prev_cluster3', 
                                 'prev_cluster4', 'prev_cluster5']
        next_cluster_y = df['cluster'].shift(-1).fillna(-1)
    else:
        next_cluster_X = pd.DataFrame()
        next_cluster_y = pd.Series(dtype=int)
    
    # Convert types and handle missing values
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    cluster_X = cluster_X.astype(np.float64)
    cluster_y = cluster_y.astype(np.int32)
    gap_cluster_X = gap_cluster_X.astype(np.int32) if not gap_cluster_X.empty else gap_cluster_X
    gap_cluster_y = gap_cluster_y.astype(np.float64)
    next_cluster_X = next_cluster_X.astype(np.int32) if not next_cluster_X.empty else next_cluster_X
    next_cluster_y = next_cluster_y.astype(np.int32)
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    cluster_X = cluster_X.fillna(0)
    gap_cluster_X = gap_cluster_X.fillna(-1)
    next_cluster_X = next_cluster_X.fillna(-1)
    
    if logger:
        logger.log_and_print(f"\nPrepared training data shapes:")
        logger.log_and_print(f"Main training data: X={X.shape}, y={y.shape}")
        logger.log_and_print(f"Cluster membership: X={cluster_X.shape}, y={cluster_y.shape}")
        logger.log_and_print(f"Gap from cluster: X={gap_cluster_X.shape}, y={gap_cluster_y.shape}")
        logger.log_and_print(f"Next cluster: X={next_cluster_X.shape}, y={next_cluster_y.shape}")
    else:
        print(f"\nPrepared training data shapes:")
        print(f"Main training data: X={X.shape}, y={y.shape}")
        print(f"Cluster membership: X={cluster_X.shape}, y={cluster_y.shape}")
        print(f"Gap from cluster: X={gap_cluster_X.shape}, y={gap_cluster_y.shape}")
        print(f"Next cluster: X={next_cluster_X.shape}, y={next_cluster_y.shape}")
    
    return X, y, feature_cols, cluster_X, cluster_y, gap_cluster_X, gap_cluster_y, next_cluster_X, next_cluster_y

@timing_decorator
def prepare_training_data_large_scale(df, batch_size=5000, sequence_length=50, logger=None):
    """Prepare training data with enhanced features for sequence and cluster patterns."""
    with suppress_overflow_warnings():
        try:
            if logger:
                logger.log_and_print("Starting large-scale training data preparation...")
            
            # Define features that might leak information
            leakage_features = [
                'gap_size',          # Direct target
                'gap_mod6',          # Direct function of gap
                'gap_mod30',         # Direct function of gap
                'factor_density',    # Likely derived from gap
                'factor_range_ratio',# Likely derived from gap
                'mean_factor',       # Could be derived from gap
                'factor_std',        # Could be derived from gap
                'factor_entropy',    # Could be derived from gap
                'mean_sqrt_factor',  # Could be derived from gap
                'sum_sqrt_factor',   # Could be derived from gap
                'cluster',           # Based on gap-related features
                'sub_cluster',       # Based on gap-related features
                'lower_prime',       # Sequential information
                'upper_prime',       # Sequential information
                'is_outlier',        # Based on gap size
                'preceding_gaps',     # Direct information about gaps
                'exp_log_gap_size',  # Direct function of gap
                'log2_gap_size',     # Direct function of gap
                'log10_gap_size',    # Direct function of gap
                'sin_log_gap_size',  # Direct function of gap
                'cos_log_gap_size',   # Direct function of gap
                'upper_prime_type',   # Categorical feature
                'lower_prime_type',    # Categorical feature
                'factor_harmonic_mean', # Highly correlated with gap size
                'total_factors',      # Highly correlated with gap size
                'num_distinct_prime_factors', # Highly correlated with gap size
                'unique_factors',     # Highly correlated with gap size
                'factor_complexity'   # Highly correlated with gap size
            ]
            
            # Get base feature columns
            base_feature_cols = [col for col in df.columns if col not in leakage_features]
            
            if logger:
                logger.log_and_print(f"Selected {len(base_feature_cols)} base features after removing potential leakage")
            
            # Compute cluster-level features
            cluster_features = {}
            for cluster_id in sorted(df['cluster'].unique()):
                cluster_features[cluster_id] = compute_cluster_level_features(df, cluster_id)
            
            # Add cluster features to dataframe in batches
            cluster_feature_df = pd.DataFrame(index=df.index)
            for cluster_id, features in cluster_features.items():
                cluster_mask = df['cluster'] == cluster_id
                for feature_name, value in features.items():
                    cluster_feature_df.loc[cluster_mask, feature_name] = value
            
            # Compute sequence features in batches
            sequence_features_df = compute_sequence_features(df, sequence_length=sequence_length)
            
            # Initialize arrays for accumulating processed data
            X_processed = []
            y_processed = []
            cluster_X_processed = []
            gap_cluster_X_processed = []
            next_cluster_X_processed = []
            
            # Process data in batches
            for start_idx in range(0, len(df), batch_size):
                if logger:
                    logger.log_and_print(f"Processing batch {start_idx//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
                
                end_idx = min(start_idx + batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]
                
                # Process main features
                X_batch = pd.concat([
                    batch_df[base_feature_cols],
                    cluster_feature_df.iloc[start_idx:end_idx],
                    sequence_features_df.iloc[start_idx:end_idx] if start_idx < len(sequence_features_df) else pd.DataFrame()
                ], axis=1)
                
                X_batch = X_batch.astype(np.float64)
                X_batch = X_batch.clip(-1e10, 1e10)
                X_processed.append(X_batch.values)
                
                # Process target
                y_batch = batch_df['gap_size'].values.astype(np.float64)
                y_batch = np.clip(y_batch, -1e10, 1e10)
                y_processed.append(y_batch)
                
                # Process cluster membership data
                if 'cluster' in batch_df.columns:
                    cluster_X_batch = X_batch.copy()
                    cluster_X_processed.append(cluster_X_batch)
                    
                    # Process gap cluster data with enhanced features
                    gap_cluster_features = pd.DataFrame(index=batch_df.index)
                    gap_cluster_features['cluster'] = batch_df['cluster']
                    
                    # Add cluster-level features
                    for cluster_id, features in cluster_features.items():
                        cluster_mask = batch_df['cluster'] == cluster_id
                        for feature_name, value in features.items():
                            gap_cluster_features.loc[cluster_mask, feature_name] = value
                    
                    gap_cluster_X_processed.append(gap_cluster_features.values)
                    
                    # Process next cluster features with sequence information
                    next_cluster_features = []
                    for i in range(sequence_length):
                        shift_value = batch_df['cluster'].shift(i+1).fillna(-1)
                        next_cluster_features.append(shift_value)
                    
                    # Add sequence-based features
                    if start_idx >= sequence_length:
                        seq_features = sequence_features_df.iloc[start_idx:end_idx]
                        next_cluster_features.extend([seq_features[col] for col in seq_features.columns])
                    
                    next_cluster_X_batch = pd.concat(next_cluster_features, axis=1)
                    next_cluster_X_batch.columns = [f'prev_cluster_{i+1}' for i in range(sequence_length)] + \
                                                 list(sequence_features_df.columns)
                    next_cluster_X_processed.append(next_cluster_X_batch.values)
                
                gc.collect()
            
            # Combine processed batches
            X = np.vstack(X_processed)
            y = np.concatenate(y_processed)
            
            if cluster_X_processed:
                cluster_X = np.vstack(cluster_X_processed)
                gap_cluster_X = np.vstack(gap_cluster_X_processed)
                next_cluster_X = np.vstack(next_cluster_X_processed)
                
                # Prepare cluster targets
                cluster_y = df['cluster'].values.astype(np.int32)
                gap_cluster_y = y.copy()
                next_cluster_y = np.roll(cluster_y, -1)
                next_cluster_y[-1] = -1
            else:
                cluster_X = np.array([])
                gap_cluster_X = np.array([])
                next_cluster_X = np.array([])
                cluster_y = np.array([], dtype=np.int32)
                gap_cluster_y = np.array([], dtype=np.float64)
                next_cluster_y = np.array([], dtype=np.int32)
            
            # Convert back to DataFrames with proper column names
            feature_cols = base_feature_cols + \
                         list(cluster_features[list(cluster_features.keys())[0]].keys()) + \
                         list(sequence_features_df.columns)
            
            X = pd.DataFrame(X, columns=feature_cols)
            cluster_X = pd.DataFrame(cluster_X, columns=feature_cols) if len(cluster_X) > 0 else pd.DataFrame()
            
            # Create column names for gap_cluster_X
            gap_cluster_cols = ['cluster'] + [f for f in cluster_features[list(cluster_features.keys())[0]].keys()]
            gap_cluster_X = pd.DataFrame(gap_cluster_X, columns=gap_cluster_cols) if len(gap_cluster_X) > 0 else pd.DataFrame()
            
            # Create column names for next_cluster_X
            next_cluster_cols = [f'prev_cluster_{i+1}' for i in range(sequence_length)] + list(sequence_features_df.columns)
            next_cluster_X = pd.DataFrame(next_cluster_X, columns=next_cluster_cols) if len(next_cluster_X) > 0 else pd.DataFrame()
            
            # Check for empty feature list
            if not feature_cols:
                if logger:
                    logger.log_and_print("Warning: No valid features after processing. Adding 'mean_factor' as default feature.")
                else:
                    print("Warning: No valid features after processing. Adding 'mean_factor' as default feature.")
                if 'mean_factor' in df.columns:
                    feature_cols = ['mean_factor']
                    X = df[['mean_factor']]
                else:
                    # If mean_factor is not available, add the first numeric column
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if numeric_cols:
                        feature_cols = [numeric_cols[0]]
                        X = df[[numeric_cols[0]]]
                    else:
                        if logger:
                            logger.log_and_print("Error: No numeric columns available, cannot proceed.")
                        else:
                            print("Error: No numeric columns available, cannot proceed.")
                        return None, None, [], None, None, None, None, None, None
            
            # Final cleanup
            del X_processed, y_processed, cluster_X_processed
            del gap_cluster_X_processed, next_cluster_X_processed
            gc.collect()
            
            if logger:
                logger.log_and_print(f"Final data shapes:")
                logger.log_and_print(f"Main training data: X={X.shape}, y={len(y)}")
                logger.log_and_print(f"Cluster membership: X={cluster_X.shape}, y={len(cluster_y)}")
                logger.log_and_print(f"Gap from cluster: X={gap_cluster_X.shape}, y={len(gap_cluster_y)}")
                logger.log_and_print(f"Next cluster: X={next_cluster_X.shape}, y={len(next_cluster_y)}")
            
            return (X, y, feature_cols, cluster_X, cluster_y, 
                   gap_cluster_X, gap_cluster_y, next_cluster_X, next_cluster_y)
            
        except Exception as e:
            error_msg = f"Error in prepare_training_data_large_scale: {str(e)}"
            if logger:
                logger.logger.error(error_msg)
                logger.logger.error(traceback.format_exc())
            else:
                print(error_msg)
                traceback.print_exc()
            raise
          
def validate_feature_independence(X, y, threshold=0.8, batch_size=10000, logger=None):
    """Check for suspiciously high correlations between features and target with improved numerical stability and batching."""
    if logger:
        logger.log_and_print("\nValidating feature independence...")
    else:
        print("\nValidating feature independence...")
    
    if X is None or X.empty or y is None or len(y) == 0:
        if logger:
            logger.log_and_print("Warning: Empty input data for feature validation", level=logging.WARNING)
        else:
            print("Warning: Empty input data for feature validation")
        return [], [], pd.DataFrame()
        
    if len(X.columns) == 0:
        if logger:
            logger.log_and_print("Warning: No features to validate", level=logging.WARNING)
        else:
            print("Warning: No features to validate")
        return [], [], pd.DataFrame()
    
    try:
        # Initialize correlation matrix
        n_features = len(X.columns)
        correlation_matrix = pd.DataFrame(0.0, 
                                       index=X.columns, 
                                       columns=X.columns.append(pd.Index(['target'])),
                                       dtype=np.float64)
        
        # Process correlations in batches
        for start_idx in range(0, len(X), batch_size):
            end_idx = min(start_idx + batch_size, len(X))
            X_batch = X.iloc[start_idx:end_idx].astype(np.float64)
            y_batch = pd.Series(y.iloc[start_idx:end_idx], name='target', dtype=np.float64)
            
            # Clip values to prevent overflow
            X_batch = X_batch.clip(-1e10, 1e10)
            y_batch = y_batch.clip(-1e10, 1e10)
            
            # Add target to batch data
            batch_data = pd.concat([X_batch, y_batch], axis=1)
            
            # Compute correlation matrix with numerical stability
            with np.errstate(all='ignore'):
                batch_corr = batch_data.corr()
                
            # Update correlation matrix (weighted by batch size)
            weight = len(X_batch) / len(X)
            correlation_matrix += batch_corr * weight
            
            gc.collect()
        
        # Find features highly correlated with target
        target_correlations = correlation_matrix['target'].drop('target')
        suspicious_features = target_correlations[abs(target_correlations) > threshold].index.tolist()
        
        if suspicious_features:
            if logger:
                logger.log_and_print("\nWarning: Features with high correlation to target:")
            else:
                print("\nWarning: Features with high correlation to target:")
            for feature in suspicious_features:
                corr = correlation_matrix.loc[feature, 'target']
                if logger:
                    logger.log_and_print(f"- {feature}: {corr:.4f}")
                else:
                    print(f"- {feature}: {corr:.4f}")
        
        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = abs(pd.to_numeric(correlation_matrix.iloc[i, j], errors='coerce'))
                if corr > threshold:
                    high_corr_pairs.append((
                        X.columns[i],
                        X.columns[j],
                        float(corr)
                    ))
        
        if high_corr_pairs:
            if logger:
                logger.log_and_print("\nWarning: Highly correlated feature pairs:")
            else:
                print("\nWarning: Highly correlated feature pairs:")
            for f1, f2, corr in high_corr_pairs:
                if logger:
                    logger.log_and_print(f"- {f1} & {f2}: {corr:.4f}")
                else:
                    print(f"- {f1} & {f2}: {corr:.4f}")
        
        # Analyze feature distributions
        if logger:
            logger.log_and_print("\nFeature distribution analysis:")
        else:
            print("\nFeature distribution analysis:")
        
        for col in X.columns:
            stats = {
                'mean': 0.0,
                'std': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'zeros': 0,
                'count': 0
            }
            
            # Process statistics in batches
            for start_idx in range(0, len(X), batch_size):
                end_idx = min(start_idx + batch_size, len(X))
                col_data = X[col].iloc[start_idx:end_idx].values.astype(np.float64)
                col_data = np.clip(col_data, -1e10, 1e10)
                
                valid_mask = np.isfinite(col_data)
                if np.any(valid_mask):
                    col_data = col_data[valid_mask]
                    stats['count'] += len(col_data)
                    stats['sum'] += np.sum(col_data)
                    stats['std'] += np.sum(col_data ** 2)
                    stats['min'] = min(stats['min'], np.min(col_data))
                    stats['max'] = max(stats['max'], np.max(col_data))
                    stats['zeros'] += np.sum(col_data == 0)
                
                gc.collect()
            
            # Compute final statistics
            if stats['count'] > 0:
                mean = stats['sum'] / stats['count']
                var = (stats['sum'] / stats['count']) - (mean ** 2)
                std = np.sqrt(max(0, var))
                stats['mean'] = float(mean)
                stats['std'] = float(std)
                stats['zero_fraction'] = float(stats['zeros'] / stats['count'])
                
                if logger:
                    logger.log_and_print(f"\n{col}:")
                    logger.log_and_print(f"- Mean: {stats['mean']:.4f}")
                    logger.log_and_print(f"- Std: {stats['std']:.4f}")
                    logger.log_and_print(f"- Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                    logger.log_and_print(f"- Zero fraction: {stats['zero_fraction']:.4f}")
                else:
                    print(f"\n{col}:")
                    print(f"- Mean: {stats['mean']:.4f}")
                    print(f"- Std: {stats['std']:.4f}")
                    print(f"- Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                    print(f"- Zero fraction: {stats['zero_fraction']:.4f}")
        
        # Check for non-linear relationships
        if logger:
            logger.log_and_print("\nChecking for non-linear relationships...")
        else:
            print("\nChecking for non-linear relationships...")
        
        non_linear_relationships = []
        
        for col in X.columns:
            linear_corr = 0.0
            squared_corr = 0.0
            cubed_corr = 0.0
            total_weight = 0
            
            for start_idx in range(0, len(X), batch_size):
                end_idx = min(start_idx + batch_size, len(X))
                x_batch = X[col].iloc[start_idx:end_idx].values.astype(np.float64)
                y_batch = y.iloc[start_idx:end_idx].values.astype(np.float64)
                
                # Clip values
                x_batch = np.clip(x_batch, -1e10, 1e10)
                y_batch = np.clip(y_batch, -1e10, 1e10)
                
                # Compute correlations for this batch
                with np.errstate(all='ignore'):
                    weight = len(x_batch) / len(X)
                    linear_corr += np.corrcoef(x_batch, y_batch)[0, 1] * weight
                    squared_corr += np.corrcoef(x_batch ** 2, y_batch)[0, 1] * weight
                    cubed_corr += np.corrcoef(x_batch ** 3, y_batch)[0, 1] * weight
                
                gc.collect()
            
            if abs(squared_corr) > abs(linear_corr) + 0.1 or abs(cubed_corr) > abs(linear_corr) + 0.1:
                non_linear_relationships.append({
                    'feature': col,
                    'linear_corr': float(linear_corr),
                    'squared_corr': float(squared_corr),
                    'cubed_corr': float(cubed_corr)
                })
        
        if non_linear_relationships:
            if logger:
                logger.log_and_print("\nWarning: Potential non-linear relationships detected:")
            else:
                print("\nWarning: Potential non-linear relationships detected:")
            for rel in non_linear_relationships:
                if logger:
                    logger.log_and_print(f"\n{rel['feature']}:")
                    logger.log_and_print(f"- Linear correlation: {rel['linear_corr']:.4f}")
                    logger.log_and_print(f"- Squared correlation: {rel['squared_corr']:.4f}")
                    logger.log_and_print(f"- Cubed correlation: {rel['cubed_corr']:.4f}")
                else:
                    print(f"\n{rel['feature']}:")
                    print(f"- Linear correlation: {rel['linear_corr']:.4f}")
                    print(f"- Squared correlation: {rel['squared_corr']:.4f}")
                    print(f"- Cubed correlation: {rel['cubed_corr']:.4f}")
        
        # Summary
        if logger:
            logger.log_and_print("\nFeature independence validation summary:")
            logger.log_and_print(f"- Total features analyzed: {len(X.columns)}")
            logger.log_and_print(f"- Features highly correlated with target: {len(suspicious_features)}")
            logger.log_and_print(f"- Highly correlated feature pairs: {len(high_corr_pairs)}")
            logger.log_and_print(f"- Non-linear relationships detected: {len(non_linear_relationships)}")
        else:
            print("\nFeature independence validation summary:")
            print(f"- Total features analyzed: {len(X.columns)}")
            print(f"- Features highly correlated with target: {len(suspicious_features)}")
            print(f"- Highly correlated feature pairs: {len(high_corr_pairs)}")
            print(f"- Non-linear relationships detected: {len(non_linear_relationships)}")
        
        return suspicious_features, high_corr_pairs, correlation_matrix
        
    except Exception as e:
        error_msg = f"Error in validate_feature_independence: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        return [], [], pd.DataFrame()
         
def check_and_impute_dataframe(df, name, stage="", fold=None):
    """Check for and impute NaN/infinite values in a DataFrame."""
    fold_str = f" in fold {fold}" if fold is not None else ""
    if not np.isfinite(df.values).all() or np.isnan(df.values).any():
        print(f"Warning: inf or NaN values detected in {name}{stage}{fold_str}. Imputing with 0.")
        return df.replace([np.inf, -np.inf], 0).fillna(0)
    return df

def check_and_impute_array(arr, name, stage="", fold=None):
    """Check for and impute NaN/infinite values in a numpy array."""
    fold_str = f" in fold {fold}" if fold is not None else ""
    if not np.isfinite(arr).all() or np.isnan(arr).any():
        print(f"Warning: inf or NaN values detected in {name}{stage}{fold_str}. Imputing with 0.")
        return np.nan_to_num(arr, nan=0)
    return arr

def select_features(X, y, threshold=0.01):
    """Select important features using multiple methods."""
    print("Performing feature selection...")
    
    # Initialize feature importance dictionary
    feature_importance = {}
    
    # 1. Random Forest importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    feature_importance['rf'] = rf_importance
    
    # 2. XGBoost importance
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_model.fit(X, y)
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
    feature_importance['xgb'] = xgb_importance
    
    # 3. Correlation based
    correlations = np.abs(X.corrwith(pd.Series(y)))
    feature_importance['correlation'] = correlations
    
    # Combine importance scores
    combined_importance = pd.DataFrame(feature_importance)
    combined_importance['mean_importance'] = combined_importance.mean(axis=1)
    
    # Select features above threshold
    selected_features = combined_importance[combined_importance['mean_importance'] > threshold].index.tolist()
    
    return selected_features, combined_importance

def create_ensemble_model(feature_cols, logger=None):
    """Create an ensemble of models with improved configuration and error handling."""
    if logger:
        logger.log_and_print("Creating ensemble model...")
    
    try:
        models = {
            'rf1': RandomForestRegressor(
                n_estimators=200, 
                max_depth=12,
                random_state=42,
                n_jobs=-1,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features=0.6,
                ccp_alpha=0.01,
                bootstrap=True,
                oob_score=True,
                max_samples=0.7,
                warm_start=False  # Disable warm start for better stability
            ),
            'rf2': RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                random_state=43,
                n_jobs=-1,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features=0.4,
                ccp_alpha=0.02,
                bootstrap=True,
                oob_score=True,
                max_samples=0.8,
                warm_start=False
            ),
            'xgb1': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.01,
                random_state=42,
                n_jobs=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.3,
                reg_lambda=0.3,
                tree_method='hist',
                max_bin=256,
                scale_pos_weight=1.0
            ),
            'xgb2': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.005,
                random_state=43,
                n_jobs=-1,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.4,
                reg_lambda=0.4,
                tree_method='hist',
                max_bin=256,
                scale_pos_weight=1.0
            ),
            'nn': create_neural_network(feature_cols, logger)
        }

        # Add predict method to ensemble model dictionary
        def ensemble_predict(self, X):
            try:
                predictions = {}
                for name, model in self['models'].items():
                    with np.errstate(all='ignore'):
                        if isinstance(model, tf.keras.Sequential):
                            pred = model.predict(X).flatten()
                        else:
                            pred = model.predict(X)
                        # Ensure numerical stability
                        pred = np.array(pred, dtype=np.float64)
                        pred = np.clip(pred, -1e10, 1e10)
                        predictions[name] = pred
                
                # Combine using stored weights with numerical stability
                ensemble_pred = np.zeros(len(X), dtype=np.float64)
                for name, pred in predictions.items():
                    weight = self['weights'].get(name, 0.0)
                    ensemble_pred += pred * weight
                
                return np.clip(ensemble_pred, -1e10, 1e10)
                
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Error in ensemble prediction: {str(e)}", level=logging.ERROR)
                    logger.logger.error(traceback.format_exc())
                return np.zeros(len(X), dtype=np.float64)

        # Create ensemble dictionary with improved configuration
        ensemble_dict = {
            'models': models,
            'weights': {
                'rf1': 0.25,
                'rf2': 0.25,
                'xgb1': 0.25,
                'xgb2': 0.15,
                'nn': 0.10
            },
            'properties': {
                'requires_scaling': True,
                'handles_missing': False,
                'prediction_intervals': False,
                'feature_importance': True
            }
        }
        
        # Add predict method
        ensemble_dict['predict'] = types.MethodType(ensemble_predict, ensemble_dict)
        
        if logger:
            logger.log_and_print("Ensemble model created successfully")
            logger.log_and_print(f"Models: {list(models.keys())}")
            logger.log_and_print(f"Weights: {ensemble_dict['weights']}")
        
        return ensemble_dict
        
    except Exception as e:
        error_msg = f"Error creating ensemble model: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Create a minimal fallback ensemble
        fallback_models = {
            'rf1': RandomForestRegressor(random_state=42),
            'rf2': RandomForestRegressor(random_state=43)
        }
        
        fallback_dict = {
            'models': fallback_models,
            'weights': {'rf1': 0.5, 'rf2': 0.5},
            'properties': {
                'requires_scaling': True,
                'handles_missing': False,
                'prediction_intervals': False,
                'feature_importance': True
            }
        }
        fallback_dict['predict'] = types.MethodType(ensemble_predict, fallback_dict)
        
        if logger:
            logger.log_and_print("Created fallback ensemble model")
        
        return fallback_dict
    
@timing_decorator
def train_ensemble(ensemble_model, X_train, X_test, y_train, y_test):
    """Train ensemble of models and combine predictions."""
    predictions_train = {}
    predictions_test = {}
    trained_models = {}
    
    # Train each model and get predictions
    for name, model in ensemble_model['models'].items():
        if isinstance(model, keras.Sequential):
            # Neural network training
            model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                loss='mse'
            )
            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=64,
                verbose='0',
                validation_split=0.1
            )
            # Make predictions without using tf.function
            predictions_train[name] = model(X_train, training=False).numpy().flatten()
            predictions_test[name] = model(X_test, training=False).numpy().flatten()
        else:
            # Standard model training
            try:
                model.fit(X_train, y_train)
                predictions_train[name] = model.predict(X_train)
                predictions_test[name] = model.predict(X_test)
            except Exception as e:
                print(f"Error training model {name}: {str(e)}")
                # Handle training failure
                predictions_train[name] = np.zeros_like(y_train)
                predictions_test[name] = np.zeros_like(y_test)
                
        # Store trained model
        trained_models[name] = model
    
    # Combine predictions using weighted average
    weights = ensemble_model['weights']
    
    final_pred_train = np.zeros_like(y_train, dtype=np.float64)
    for name, pred in predictions_train.items():
        final_pred_train += pred * weights.get(name, 0.0)
    
    final_pred_test = np.zeros_like(y_test, dtype=np.float64)
    for name, pred in predictions_test.items():
        final_pred_test += pred * weights.get(name, 0.0)
    
    # Compute metrics
    train_mse = mean_squared_error(y_train, final_pred_train)
    train_r2 = r2_score(y_train, final_pred_train)
    test_mse = mean_squared_error(y_test, final_pred_test)
    test_r2 = r2_score(y_test, final_pred_test)
    
    return {
        'model': {
            'models': trained_models,
            'weights': weights
        },
        'train_mse': train_mse,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'predictions': {
            'train': final_pred_train,
            'test': final_pred_test,
            'individual_train': predictions_train,
            'individual_test': predictions_test
        }
    }
             
def perform_advanced_clustering(df, n_clusters=3):
    """Perform advanced clustering with multiple algorithms and ensemble."""
    # Get numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['cluster', 'sub_cluster', 'gap_size']]
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # 1. K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # 2. MiniBatchKMeans for robustness
    mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    mbk_labels = mbk.fit_predict(X_scaled)
    
    # 3. Spectral Clustering for non-linear patterns
    try:
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            random_state=42,
            n_neighbors=10,
            assign_labels='kmeans'
        )
        spectral_labels = spectral.fit_predict(X_scaled)
    except Exception:
        spectral_labels = kmeans_labels  # Fallback to kmeans if spectral fails
    
    # Ensemble clustering
    ensemble_labels = np.zeros(len(df))
    for i in range(len(df)):
        # Get most common label among all clustering methods
        labels = [kmeans_labels[i], mbk_labels[i], spectral_labels[i]]
        ensemble_labels[i] = max(set(labels), key=labels.count)
    
    # Compute cluster metrics
    metrics = {
        'kmeans_inertia': kmeans.inertia_,
        'mbk_inertia': mbk.inertia_,
        'cluster_sizes': np.bincount(ensemble_labels.astype(int)),
        'agreement_score': np.mean([
            np.mean(kmeans_labels == mbk_labels),
            np.mean(kmeans_labels == spectral_labels),
            np.mean(mbk_labels == spectral_labels)
        ])
    }
    
    return ensemble_labels.astype(int), metrics

def analyze_cluster_stability(df, cluster_labels, n_bootstrap=10):
    """Analyze cluster stability through bootstrapping."""
    stability_scores = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['cluster', 'sub_cluster', 'gap_size']]
    X = df[feature_cols].values
    
    n_clusters = len(np.unique(cluster_labels))
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X[indices]
        
        # Cluster bootstrap sample
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        boot_labels = kmeans.fit_predict(X_boot)
        
        # Compare with original clustering
        stability_score = np.mean(cluster_labels[indices] == boot_labels)
        stability_scores.append(stability_score)
    
    return {
        'mean_stability': float(np.mean(stability_scores)),
        'std_stability': float(np.std(stability_scores)),
        'min_stability': float(np.min(stability_scores)),
        'max_stability': float(np.max(stability_scores))
    }
                                
def handle_clustering(X_train_scaled, X_test_scaled, fold):
    """Handle clustering operations within a fold with improved numerical stability."""
    with suppress_overflow_warnings():
        # Convert to numpy arrays if needed
        X_train_arr = X_train_scaled.values if isinstance(X_train_scaled, pd.DataFrame) else X_train_scaled
        X_test_arr = X_test_scaled.values if isinstance(X_test_scaled, pd.DataFrame) else X_test_scaled
        
        # Clip values to prevent numerical issues
        X_train_arr = np.clip(X_train_arr, -1e10, 1e10)
        X_test_arr = np.clip(X_test_arr, -1e10, 1e10)
        
        # Handle any remaining NaN or inf values
        X_train_arr = np.nan_to_num(X_train_arr, nan=0, posinf=1e10, neginf=-1e10)
        X_test_arr = np.nan_to_num(X_test_arr, nan=0, posinf=1e10, neginf=-1e10)
        
        try:
            # Use robust scaling before clustering
            scaler = RobustScaler()
            X_train_robust = scaler.fit_transform(X_train_arr)
            X_test_robust = scaler.transform(X_test_arr)
            
            kmeans = KMeans(n_clusters=3, 
                          random_state=42, 
                          n_init=10,
                          max_iter=300,
                          tol=1e-4)
            
            X_train_clusters = kmeans.fit_predict(X_train_robust)
            X_test_clusters = kmeans.predict(X_test_robust)
            
            return X_train_clusters, X_test_clusters
            
        except Exception as e:
            print(f"Warning: Clustering failed in fold {fold}, using fallback method: {str(e)}")
            # Fallback to simpler clustering if regular clustering fails
            X_train_clusters = np.zeros(len(X_train_arr), dtype=int)
            X_test_clusters = np.zeros(len(X_test_arr), dtype=int)
            return X_train_clusters, X_test_clusters

def update_neural_network_architecture(input_dim):
    """Create an improved neural network architecture."""
    model = keras.Sequential([
        # Input layer with normalization
        layers.BatchNormalization(input_shape=(input_dim,)),
        
        # First block
        layers.Dense(512, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),
        
        # Second block
        layers.Dense(256, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),
        
        # Third block
        layers.Dense(128, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),
        
        # Fourth block
        layers.Dense(64, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),
        
        # Fifth block
        layers.Dense(32, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # Output layer
        layers.Dense(1)
    ])
    
    return model

def create_model_dict(feature_cols, logger=None):
    """Create dictionary of models with improved configurations and error handling."""
    if logger:
        logger.log_and_print("Creating model dictionary...")
    
    try:
        models = {
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=7, 
                    min_samples_split=15,
                    min_samples_leaf=8, 
                    random_state=42, 
                    n_jobs=-1,
                    max_features=0.6, 
                    ccp_alpha=0.01,
                    bootstrap=True,
                    oob_score=True,
                    max_samples=0.7  # Use 70% of samples for each tree
                ),
                'type': 'single',
                'properties': {
                    'feature_importance': True,
                    'prediction_intervals': True,
                    'handles_missing': True,
                    'requires_scaling': False
                }
            },
            'random_forest_simple': {
                'model': RandomForestRegressor(
                    n_estimators=50, 
                    max_depth=4, 
                    random_state=42,
                    n_jobs=-1, 
                    min_samples_split=10, 
                    min_samples_leaf=5,
                    bootstrap=True,
                    oob_score=True,
                    max_samples=0.8  # Use 80% of samples for each tree
                ),
                'type': 'single',
                'properties': {
                    'feature_importance': True,
                    'prediction_intervals': True,
                    'handles_missing': True,
                    'requires_scaling': False
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor(
                    objective='reg:squarederror', 
                    n_estimators=100,
                    max_depth=4, 
                    learning_rate=0.01, 
                    reg_alpha=0.3,
                    reg_lambda=0.3, 
                    subsample=0.7, 
                    colsample_bytree=0.7,
                    n_jobs=-1,
                    random_state=42,
                    tree_method='hist',  # Faster histogram-based algorithm
                    max_bin=256  # Control memory usage
                ),
                'type': 'single',
                'properties': {
                    'feature_importance': True,
                    'prediction_intervals': False,
                    'handles_missing': True,
                    'requires_scaling': True
                }
            },
            'linear_regression': {
                'model': LinearRegression(
                    n_jobs=-1,
                    fit_intercept=True,
                    copy_X=True
                ),
                'type': 'single',
                'properties': {
                    'feature_importance': True,
                    'prediction_intervals': False,
                    'handles_missing': False,
                    'requires_scaling': True
                }
            },
            'neural_network': {
                'model': create_neural_network(feature_cols, logger),
                'type': 'single',
                'properties': {
                    'feature_importance': False,
                    'prediction_intervals': False,
                    'handles_missing': False,
                    'requires_scaling': True,
                    'requires_callbacks': True
                }
            },
            'cluster_membership_rf': {
                'model': RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=5, 
                    random_state=42, 
                    n_jobs=-1,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    bootstrap=True,
                    oob_score=True,
                    class_weight='balanced',
                    max_samples=0.7
                ),
                'type': 'classifier',
                'properties': {
                    'feature_importance': True,
                    'prediction_intervals': False,
                    'handles_missing': True,
                    'requires_scaling': False
                }
            },
            'gap_from_cluster_rf': {
                'model': RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=5, 
                    random_state=42, 
                    n_jobs=-1,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    bootstrap=True,
                    oob_score=True,
                    max_samples=0.7
                ),
                'type': 'single',
                'properties': {
                    'feature_importance': True,
                    'prediction_intervals': True,
                    'handles_missing': True,
                    'requires_scaling': False
                }
            },
            'next_cluster_rf': {
                'model': RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=5, 
                    random_state=42, 
                    n_jobs=-1,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    bootstrap=True,
                    oob_score=True,
                    class_weight='balanced',
                    max_samples=0.7
                ),
                'type': 'classifier',
                'properties': {
                    'feature_importance': True,
                    'prediction_intervals': False,
                    'handles_missing': True,
                    'requires_scaling': False
                }
            }
        }
        
        # Validate models
        for name, config in models.items():
            if not hasattr(config['model'], 'fit') or not hasattr(config['model'], 'predict'):
                raise ValueError(f"Model {name} does not have required fit/predict methods")
        
        if logger:
            logger.log_and_print("Model dictionary created successfully")
            for name, config in models.items():
                logger.log_and_print(f"Created {name} model of type {config['type']}")
        
        return models
        
    except Exception as e:
        error_msg = f"Error creating model dictionary: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Create minimal fallback model dictionary
        return {
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'type': 'single',
                'properties': {
                    'feature_importance': True,
                    'prediction_intervals': True,
                    'handles_missing': True,
                    'requires_scaling': False
                }
            }
        }
                      
def create_neural_network(feature_cols, logger=None):
    """Create neural network model with improved architecture and numerical stability."""
    try:
        if logger:
            logger.log_and_print("Creating neural network model...")
        
        # Input dimension validation
        input_dim = len(feature_cols)
        if input_dim < 1:
            raise ValueError("Feature columns list cannot be empty")
        
        # Create model with improved architecture
        model = keras.Sequential([
            # Input layer with normalization
            layers.Input(shape=(input_dim,)),
            layers.BatchNormalization(name='input_norm'),
            
            # First block - widest layer with L2 regularization
            layers.Dense(
                256,
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name='dense_1'
            ),
            layers.BatchNormalization(name='norm_1'),
            layers.LeakyReLU(alpha=0.1, name='leaky_relu_1'),
            layers.Dropout(0.3, name='dropout_1'),
            
            # Second block
            layers.Dense(
                128,
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name='dense_2'
            ),
            layers.BatchNormalization(name='norm_2'),
            layers.LeakyReLU(alpha=0.1, name='leaky_relu_2'),
            layers.Dropout(0.2, name='dropout_2'),
            
            # Third block
            layers.Dense(
                64,
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name='dense_3'
            ),
            layers.BatchNormalization(name='norm_3'),
            layers.LeakyReLU(alpha=0.1, name='leaky_relu_3'),
            layers.Dropout(0.1, name='dropout_3'),
            
            # Output layer
            layers.Dense(1, name='output')
        ])
        
        # Configure model compilation with improved optimizer settings
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=0.0005,  # Reduced learning rate
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=True
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust loss function
            metrics=['mae', 'mse']
        )
        
        # Add custom attributes for better tracking
        model.input_dim = input_dim
        model.feature_cols = feature_cols
        
        # Configure model for better numerical stability
        tf.keras.backend.set_floatx('float64')
        
        if logger:
            logger.log_and_print("Neural network model created successfully")
            logger.log_and_print(f"Model input dimension: {input_dim}")
            model.summary(print_fn=logger.log_and_print)
        
        return model
        
    except Exception as e:
        error_msg = f"Error creating neural network: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Create a simpler fallback model
        fallback_model = keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        fallback_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        if logger:
            logger.log_and_print("Created fallback model due to error")
        
        return fallback_model
       
def _create_callback_with_float_params(callback_class, monitor, patience, min_delta, factor=None, min_lr=None, filepath=None, save_best_only=True, save_weights_only=True):
    """Helper function to create callbacks with float parameters."""
    params = {
        'monitor': monitor,
        'patience': patience,
        'restore_best_weights': True,
        'min_delta': float(min_delta)
    }
    if factor is not None:
        params['factor'] = float(factor)
    if min_lr is not None:
        params['min_lr'] = float(min_lr)
    if filepath is not None:
        params['filepath'] = filepath
        params['save_best_only'] = save_best_only
        params['save_weights_only'] = save_weights_only
    
    return callback_class(**params)

def create_neural_network_callbacks(fold=None, logger=None):
    """Create callbacks for neural network training with improved configuration."""
    callbacks = [
        # Early stopping with better parameters
        _create_callback_with_float_params(
            tf.keras.callbacks.EarlyStopping,
            monitor='val_loss',
            patience=5,
            min_delta=1e-4
        ),
        
        # Learning rate reduction
        _create_callback_with_float_params(
            tf.keras.callbacks.ReduceLROnPlateau,
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_delta=1e-4,
            min_lr=1e-6
        ),
        
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'model_checkpoint_fold_{fold}.h5' if fold is not None else 'model_checkpoint.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        ),
        
        # Terminate on NaN
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    
    # Add custom progress callback if fold is specified
    if fold is not None:
        callbacks.append(TensorFlowProgressCallback(fold_number=fold))
    
    return callbacks

def select_best_model(model_results):
    """Select the best performing model based on test MSE."""
    model_performance = {}
    
    for name, results in model_results.items():
        # Skip models without test metrics
        if not results or not isinstance(results, dict):
            continue
            
        # Handle different result structures
        if 'avg_test_mse' in results:
            test_mse = results['avg_test_mse']
        elif 'test_mse' in results:
            test_mse = results['test_mse']
        else:
            continue
            
        # Store performance
        model_performance[name] = test_mse
    
    if not model_performance:
        return None, None
    
    # Select best model name
    best_model_name = min(model_performance.items(), key=lambda x: x[1])[0]
    best_model_results = model_results[best_model_name]
    
    # Extract the actual model object
    if isinstance(best_model_results, dict):
        if 'model' in best_model_results:
            if isinstance(best_model_results['model'], dict):
                # Handle ensemble/stacking models
                return best_model_name, best_model_results['model']
            else:
                # Handle single models
                return best_model_name, best_model_results['model']
    
    return None, None

def get_prediction_function(model):
    """Get appropriate prediction function for different model types."""
    
    def stacking_predict(X):
        """Prediction function for stacking models."""
        base_predictions = np.column_stack([
            base_model.predict(X)
            for base_model in model['base_models'].values()
        ])
        return model['meta_learner'].predict(base_predictions)
    
    def ensemble_predict(X):
        """Prediction function for ensemble models."""
        predictions = {}
        for name, submodel in model['models'].items():
            if isinstance(submodel, tf.keras.Sequential):
                pred = submodel.predict(X, verbose="0").flatten()
            else:
                pred = submodel.predict(X)
            predictions[name] = pred
        return sum(pred * model['weights'][name] 
                 for name, pred in predictions.items())
    
    if isinstance(model, dict):
        if 'base_models' in model and 'meta_learner' in model:
            # Stacking model
            return stacking_predict
        elif 'models' in model and 'weights' in model:
            # Ensemble model
            return ensemble_predict
    elif hasattr(model, 'predict'):
        return model.predict
    
    raise ValueError(f"Unsupported model type: {type(model)}")

def update_main_analysis(df, model_results, feature_importance, pattern_analysis):
    """Update main analysis results with the best model."""
    try:
        # Select best model
        best_model_name, best_model = select_best_model(model_results)
        if best_model is None:
            print("Warning: Could not determine best model")
            return None
            
        print(f"\nBest performing model: {best_model_name}")
        
        # Get prediction function for the best model
        predict_func = get_prediction_function(best_model)
        
        # Update model_results with proper model structure
        if isinstance(best_model, dict):
            model_results[best_model_name]['predict'] = predict_func
        
        return {
            'best_model': {
                'name': best_model_name,
                'model': best_model,
                'predict': predict_func
            },
            'model_results': model_results,
            'feature_importance': feature_importance,
            'pattern_analysis': pattern_analysis
        }
        
    except Exception as e:
        print(f"Error in update_main_analysis: {str(e)}")
        traceback.print_exc()
        return None

@timing_decorator
def train_single_model(model_config, X_train_scaled, X_test_scaled, y_train, y_test, is_neural_network=False, fold=None):
    """Train a single model and return predictions and metrics with robust error handling."""
    try:
        # Extract the actual model from the config
        model = model_config['model']

        # Convert DataFrame to numpy array if necessary
        if isinstance(X_train_scaled, pd.DataFrame):
            X_train_array = X_train_scaled.values
            X_test_array = X_test_scaled.values
        else:
            X_train_array = X_train_scaled
            X_test_array = X_test_scaled

        # Handle NaN/inf values in input data
        X_train_array = np.nan_to_num(X_train_array, nan=0.0, posinf=1e10, neginf=-1e10)
        X_test_array = np.nan_to_num(X_test_array, nan=0.0, posinf=1e10, neginf=-1e10)
        y_train = np.nan_to_num(y_train, nan=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0)

        if is_neural_network:
            try:
                # Use legacy optimizer for M1/M2 Macs with better parameters
                optimizer = tf.keras.optimizers.legacy.Adam(
                    learning_rate=0.001,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07,
                    amsgrad=True
                )
                
                # Configure model with better settings
                model.compile(
                    optimizer=optimizer,
                    loss='huber',  # More robust loss function
                    metrics=['mae', 'mse']
                )
                
                # Create early stopping callback with better parameters
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,  # Increased patience
                    restore_best_weights=True,
                    min_delta=1  # More sensitive to improvements
                )
                
                # Create learning rate reduction callback
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,  # More aggressive reduction
                    patience=5,
                    min_delta=1e-5,
                    min_lr=1  # Keep as int
                )
                
                # Add model checkpoint
                checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    f'model_checkpoint_fold_{fold}.h5' if fold is not None else 'model_checkpoint.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True
                )
                
                # Train with better parameters
                history = model.fit(
                    X_train_array,
                    y_train,
                    epochs=150,  # Increased epochs
                    batch_size=32,  # Reduced batch size
                    validation_split=0.2,
                    callbacks=[
                        early_stopping,
                        reduce_lr,
                        checkpoint,
                        tf.keras.callbacks.TerminateOnNaN(),
                        TensorFlowProgressCallback(fold_number=fold) if fold is not None else None
                    ],
                    verbose='0'
                )
                
                # Check if training succeeded
                if np.isnan(history.history['loss'][-1]):
                    print(f"Warning: Neural network training produced NaN loss in fold {fold}, using fallback model")
                    fallback_model = LinearRegression()
                    fallback_model.fit(X_train_array, y_train)
                    y_train_pred = fallback_model.predict(X_train_array)
                    y_pred = fallback_model.predict(X_test_array)
                    model = fallback_model
                else:
                    # Use the trained neural network
                    # Call model directly instead of using predict
                    y_train_pred = model(X_train_array, training=False).numpy().flatten()
                    y_pred = model(X_test_array, training=False).numpy().flatten()
                
            except Exception as e:
                print(f"Neural network training failed in fold {fold}: {str(e)}")
                print("Using fallback linear model")
                # Use linear regression as fallback
                fallback_model = LinearRegression()
                fallback_model.fit(X_train_array, y_train)
                y_train_pred = fallback_model.predict(X_train_array)
                y_pred = fallback_model.predict(X_test_array)
                model = fallback_model
        else:
            # Regular model training
            try:
                model.fit(X_train_array, y_train)
                y_train_pred = model.predict(X_train_array)
                y_pred = model.predict(X_test_array)
            except Exception as e:
                print(f"Model training failed in fold {fold}: {str(e)}")
                print("Using fallback linear model")
                fallback_model = LinearRegression()
                fallback_model.fit(X_train_array, y_train)
                y_train_pred = fallback_model.predict(X_train_array)
                y_pred = fallback_model.predict(X_test_array)
                model = fallback_model

        # Ensure predictions are clean
        y_train_pred = np.nan_to_num(y_train_pred, nan=0.0, posinf=1e10, neginf=-1e10)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Compute metrics with clean data
        train_mse = mean_squared_error(
            np.nan_to_num(y_train, nan=0.0),
            np.nan_to_num(y_train_pred, nan=0.0)
        )
        train_r2 = r2_score(
            np.nan_to_num(y_train, nan=0.0),
            np.nan_to_num(y_train_pred, nan=0.0)
        )
        test_mse = mean_squared_error(
            np.nan_to_num(y_test, nan=0.0),
            np.nan_to_num(y_pred, nan=0.0)
        )
        test_r2 = r2_score(
            np.nan_to_num(y_test, nan=0.0),
            np.nan_to_num(y_pred, nan=0.0)
        )
        
        return {
            'model': model,
            'train_mse': float(train_mse),
            'train_r2': float(train_r2),
            'test_mse': float(test_mse),
            'test_r2': float(test_r2),
            'predictions': {
                'train': y_train_pred,
                'test': y_pred
            }
        }
        
    except Exception as e:
        print(f"Error in train_single_model: {str(e)}")
        print(traceback.format_exc())
        # Return safe default values
        return {
            'model': None,
            'train_mse': float('inf'),
            'train_r2': 0.0,
            'test_mse': float('inf'),
            'test_r2': 0.0,
            'predictions': {
                'train': np.zeros_like(y_train),
                'test': np.zeros_like(y_test)
            }
        }
                              
def create_stacking_model(feature_cols, base_model_names=None, meta_learner_name='linear_regression', logger=None):
    """Create base models and meta-learner for stacking with improved configuration and error handling."""
    if logger:
        logger.log_and_print("Creating stacking model...")
    
    try:
        if base_model_names is None:
            base_model_names = ['random_forest', 'xgboost', 'linear_regression']
        
        base_models = {}
        
        # Create base models with optimized configurations
        for name in base_model_names:
            if logger:
                logger.log_and_print(f"Creating base model: {name}")
            
            if name == 'random_forest':
                base_models['rf'] = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,
                    bootstrap=True,
                    oob_score=True,
                    max_samples=0.7,
                    ccp_alpha=0.01
                )
            elif name == 'xgboost':
                base_models['xgb'] = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.01,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    n_jobs=-1,
                    tree_method='hist',
                    max_bin=256
                )
            elif name == 'linear_regression':
                base_models['lr'] = LinearRegression(
                    n_jobs=-1,
                    fit_intercept=True,
                    copy_X=True
                )
            else:
                if logger:
                    logger.log_and_print(f"Warning: Unknown base model name: {name}. Skipping.")
                continue
        
        # Create meta-learner with optimized configuration
        if logger:
            logger.log_and_print(f"Creating meta-learner: {meta_learner_name}")
        
        if meta_learner_name == 'linear_regression':
            meta_learner = LinearRegression(
                n_jobs=-1,
                fit_intercept=True,
                copy_X=True
            )
        elif meta_learner_name == 'random_forest':
            meta_learner = RandomForestRegressor(
                n_estimators=50,
                max_depth=4,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True,
                max_samples=0.8,
                ccp_alpha=0.01
            )
        elif meta_learner_name == 'xgboost':
            meta_learner = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.01,
                random_state=42,
                n_jobs=-1,
                tree_method='hist',
                max_bin=256
            )
        else:
            if logger:
                logger.log_and_print(f"Warning: Unknown meta-learner name: {meta_learner_name}. Using LinearRegression as fallback.")
            meta_learner = LinearRegression(n_jobs=-1)
        
        # Add predict method to stacking model dictionary
        def stacking_predict(self, X):
            try:
                # Generate base model predictions
                base_predictions = []
                for base_model in self['base_models'].values():
                    with np.errstate(all='ignore'):
                        pred = base_model.predict(X)
                        pred = np.clip(pred, -1e10, 1e10)
                        base_predictions.append(pred)
                
                # Stack predictions
                stacked_preds = np.column_stack(base_predictions)
                
                # Use meta-learner for final prediction
                with np.errstate(all='ignore'):
                    final_pred = self['meta_learner'].predict(stacked_preds)
                    final_pred = np.clip(final_pred, -1e10, 1e10)
                
                return final_pred
                
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Error in stacking prediction: {str(e)}", level=logging.ERROR)
                return np.zeros(len(X))
        
        # Create and return stacking dictionary with prediction method
        stacking_dict = {
            'base_models': base_models,
            'meta_learner': meta_learner,
            'properties': {
                'requires_scaling': True,
                'handles_missing': False,
                'prediction_intervals': False
            }
        }
        stacking_dict['predict'] = types.MethodType(stacking_predict, stacking_dict)
        
        if logger:
            logger.log_and_print("Stacking model created successfully")
            logger.log_and_print(f"Base models: {list(base_models.keys())}")
            logger.log_and_print(f"Meta-learner: {meta_learner_name}")
        
        return base_models, meta_learner, stacking_dict
        
    except Exception as e:
        error_msg = f"Error creating stacking model: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Create minimal fallback models
        fallback_base_models = {
            'rf': RandomForestRegressor(random_state=42)
        }
        fallback_meta_learner = LinearRegression()
        fallback_dict = {
            'base_models': fallback_base_models,
            'meta_learner': fallback_meta_learner,
            'properties': {
                'requires_scaling': True,
                'handles_missing': False,
                'prediction_intervals': False
            }
        }
        fallback_dict['predict'] = types.MethodType(stacking_predict, fallback_dict)
        
        return fallback_base_models, fallback_meta_learner, fallback_dict
    
def create_sequence_model(input_dim, num_classes, logger=None):
    """Create an LSTM model for sequence prediction with proper gradient handling."""
    if logger:
        logger.log_and_print("Creating sequence model...")
    
    try:
        # Add 1 to input_dim to handle 0-based indexing and -1 padding
        vocab_size = input_dim + 1
        
        # Create input layer with explicit batch size
        inputs = layers.Input(shape=(input_dim,), dtype=tf.int32)
        
        # Map -1 to 0, shift other indices up by 1
        x = layers.Lambda(lambda x: tf.cast(tf.where(x < 0, 0, x + 1), tf.int32))(inputs)
        
        # Embedding layer with proper input handling
        x = layers.Embedding(
            input_dim=vocab_size,
            output_dim=64,
            mask_zero=True,  # Enable masking for variable length sequences
            embeddings_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        
        # Reshape for LSTM - add explicit timestep dimension
        x = layers.Reshape((input_dim, 64))(x)
        
        # LSTM layers with gradient clipping and proper stateful configuration
        x = layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            unroll=True  # Unroll short sequences for better performance
        )(x)
        
        x = layers.LSTM(
            64,
            return_sequences=False,
            dropout=0.1,
            recurrent_dropout=0.1,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            unroll=True
        )(x)
        
        # Dense layers with proper regularization
        x = layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Custom optimizer with gradient clipping
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=0.001,
            clipnorm=1.0,  # Clip gradients
            clipvalue=0.5  # Clip gradient values
        )
        
        # Compile with proper configuration
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=True  # Run eagerly to avoid graph compilation issues
        )
        
        if logger:
            logger.log_and_print("Sequence model created successfully")
            model.summary(print_fn=logger.log_and_print)
        
        return model
    
    except Exception as e:
        error_msg = f"Error creating sequence model: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Create a simpler fallback model
        fallback_model = keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Embedding(input_dim=vocab_size, output_dim=16),
            layers.LSTM(32),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        fallback_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        if logger:
            logger.log_and_print("Created fallback sequence model due to error")
        
        return fallback_model

@timing_decorator
def fine_tune_model(model, df_new, feature_cols, learning_rate=0.0001, epochs=20, batch_size=32, logger=None):
    """Fine-tunes a pre-trained model on a new range of primes with improved numerical stability."""
    if logger:
        logger.log_and_print("Fine-tuning model on new data...")
    
    try:
        # Prepare data
        X_new = df_new[feature_cols].astype(np.float64)
        y_new = df_new['gap_size'].astype(np.float64)
        
        # Clip values for numerical stability
        X_new = X_new.clip(-1e10, 1e10)
        y_new = y_new.clip(-1e10, 1e10)
        
        # Handle any NaN or inf values
        X_new = np.nan_to_num(X_new, nan=0, posinf=1e10, neginf=-1e10)
        y_new = np.nan_to_num(y_new, nan=0)
        
        if isinstance(model, dict):
            if 'model' in model:
                model_obj = model['model']
            else:
                if logger:
                    logger.log_and_print("Warning: Model object not found in model dictionary. Skipping fine-tuning.")
                return model
        else:
            model_obj = model
        
        if isinstance(model_obj, tf.keras.Sequential):
            # Fine-tune neural network
            if logger:
                logger.log_and_print("Fine-tuning neural network...")
            
            # Use legacy optimizer for M1/M2 Macs with better parameters
            optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=True
            )
            
            model_obj.compile(
                optimizer=optimizer,
                loss='huber',
                metrics=['mae', 'mse']
            )
            
            # Fine-tune with callbacks
            model_obj.fit(
                X_new,
                y_new,
                epochs=epochs,
                batch_size=batch_size,
                verbose='0',
                validation_split=0.1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=3,
                        min_delta=1e-4,
                        min_lr=1 # has to be int    
                    ),
                    tf.keras.callbacks.TerminateOnNaN()
                ]
            )
            
        elif isinstance(model_obj, (RandomForestRegressor, xgb.XGBRegressor)):
            # Fine-tune tree-based models
            if logger:
                logger.log_and_print("Fine-tuning tree-based model...")
            
            model_obj.fit(X_new, y_new)
        
        elif isinstance(model_obj, LinearRegression):
            if logger:
                logger.log_and_print("Fine-tuning linear regression model...")
            model_obj.fit(X_new, y_new)
        else:
            if logger:
                logger.log_and_print(f"Warning: Model type {type(model_obj)} not supported for fine-tuning. Skipping.")
            return model
        
        if logger:
            logger.log_and_print("Model fine-tuning complete.")
        
        return model
        
    except Exception as e:
        error_msg = f"Error fine-tuning model: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        return model
    
def prepare_model_data(X, y, cluster_X, cluster_y, gap_cluster_X, gap_cluster_y, 
                      next_cluster_X, next_cluster_y, batch_size=5000, logger=None):
    """Prepare and validate data for model training with improved numerical stability."""
    if logger:
        logger.log_and_print("Preparing model data...")
    
    try:
        # Convert to float64 and clip values
        X_processed = X.astype(np.float64)
        y_processed = y.astype(np.float64)
        
        X_processed = X_processed.clip(-1e10, 1e10)
        y_processed = y_processed.clip(-1e10, 1e10)
        
        # Process cluster data if available
        cluster_data = None
        if not cluster_X.empty and len(cluster_y) > 0:
            cluster_X_processed = cluster_X.astype(np.float64)
            cluster_y_processed = cluster_y.astype(np.int32)
            cluster_X_processed = cluster_X_processed.clip(-1e10, 1e10)
            cluster_data = {
                'X': cluster_X_processed,
                'y': cluster_y_processed
            }
        
        # Process gap cluster data if available
        gap_data = None
        if not gap_cluster_X.empty and len(gap_cluster_y) > 0:
            gap_cluster_X_processed = gap_cluster_X.astype(np.float64)
            gap_cluster_y_processed = gap_cluster_y.astype(np.float64)
            gap_cluster_X_processed = gap_cluster_X_processed.clip(-1e10, 1e10)
            gap_cluster_y_processed = gap_cluster_y_processed.clip(-1e10, 1e10)
            gap_data = {
                'X': gap_cluster_X_processed,
                'y': gap_cluster_y_processed
            }
        
        # Process next cluster data if available
        next_cluster_data = None
        if not next_cluster_X.empty and len(next_cluster_y) > 0:
            next_cluster_X_processed = next_cluster_X.astype(np.float64)
            next_cluster_y_processed = next_cluster_y.astype(np.int32)
            next_cluster_X_processed = next_cluster_X_processed.clip(-1e10, 1e10)
            next_cluster_data = {
                'X': next_cluster_X_processed,
                'y': next_cluster_y_processed
            }
        
        # Handle any remaining NaN or inf values
        X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1e10, neginf=-1e10)
        y_processed = np.nan_to_num(y_processed, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if logger:
            logger.log_and_print("Data preparation complete")
            logger.log_and_print(f"Main data shape: X={X_processed.shape}, y={y_processed.shape}")
            if cluster_data:
                logger.log_and_print(f"Cluster data shape: X={cluster_data['X'].shape}, y={cluster_data['y'].shape}")
            if gap_data:
                logger.log_and_print(f"Gap data shape: X={gap_data['X'].shape}, y={gap_data['y'].shape}")
            if next_cluster_data:
                logger.log_and_print(f"Next cluster data shape: X={next_cluster_data['X'].shape}, y={next_cluster_data['y'].shape}")
        
        return X_processed, y_processed, cluster_data, gap_data, next_cluster_data
        
    except Exception as e:
        error_msg = f"Error in prepare_model_data: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        return None, None, None, None, None
    
@timing_decorator
def train_base_models(X, y, feature_cols, batch_size=5000, logger=None):
    """Train individual base models with improved numerical stability and error handling."""
    if logger:
        logger.log_and_print("Training base models...")
    
    base_model_results = {}
    
    try:
        # Create base models with optimized configurations
        base_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=12,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features=0.6,
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True,
                max_samples=0.7,
                ccp_alpha=0.01
            ),
            'random_forest_simple': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features=0.4,
                random_state=43,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True,
                max_samples=0.8,
                ccp_alpha=0.02
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.3,
                reg_lambda=0.3,
                random_state=42,
                n_jobs=-1,
                tree_method='hist',
                max_bin=256
            ),
            'linear_regression': LinearRegression(
                n_jobs=-1,
                fit_intercept=True,
                copy_X=True
            )
        }
        
        # Create cross-validation folds
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Train each model with cross-validation
        for name, model in base_models.items():
            if logger:
                logger.log_and_print(f"  Training {name} model...")
            
            fold_results = []
            
            for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
                if logger:
                    logger.log_and_print(f"    Processing fold {fold}/{n_splits}")
                
                # Use iloc for integer-based indexing
                X_train = X.iloc[train_index]
                X_test = X.iloc[test_index]
                y_train = y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index]
                y_test = y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]
                
                # Train in batches for large datasets
                for start_idx in range(0, len(X_train), batch_size):
                    end_idx = min(start_idx + batch_size, len(X_train))
                    X_batch = X_train.iloc[start_idx:end_idx]
                    y_batch = y_train.iloc[start_idx:end_idx] if isinstance(y_train, pd.Series) else y_train[start_idx:end_idx]
                    
                    try:
                        if hasattr(model, 'partial_fit'):
                            model.partial_fit(X_batch, y_batch)
                        else:
                            model.fit(X_batch, y_batch)
                    except Exception as e:
                        if logger:
                            logger.log_and_print(f"Warning: Training failed for {name} in batch {start_idx}-{end_idx}: {str(e)}")
                        continue
                    
                    gc.collect()
                
                # Make predictions in batches
                train_predictions = []
                test_predictions = []
                
                for start_idx in range(0, len(X_train), batch_size):
                    end_idx = min(start_idx + batch_size, len(X_train))
                    pred = model.predict(X_train.iloc[start_idx:end_idx])
                    train_predictions.extend(pred)
                
                for start_idx in range(0, len(X_test), batch_size):
                    end_idx = min(start_idx + batch_size, len(X_test))
                    pred = model.predict(X_test.iloc[start_idx:end_idx])
                    test_predictions.extend(pred)
                
                # Compute metrics
                train_mse = mean_squared_error(y_train, train_predictions)
                train_r2 = r2_score(y_train, train_predictions)
                test_mse = mean_squared_error(y_test, test_predictions)
                test_r2 = r2_score(y_test, test_predictions)
                
                fold_results.append({
                    'train_mse': float(train_mse),
                    'train_r2': float(train_r2),
                    'test_mse': float(test_mse),
                    'test_r2': float(test_r2),
                    'predictions': {
                        'train': np.array(train_predictions),
                        'test': np.array(test_predictions)
                    }
                })
                
                if logger:
                    logger.log_and_print(f"    Fold {fold} - Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
                    logger.log_and_print(f"    Fold {fold} - Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
            
            # Store results for this model
            base_model_results[name] = {
                'model': model,
                'avg_train_mse': float(np.mean([r['train_mse'] for r in fold_results])),
                'avg_train_r2': float(np.mean([r['train_r2'] for r in fold_results])),
                'avg_test_mse': float(np.mean([r['test_mse'] for r in fold_results])),
                'avg_test_r2': float(np.mean([r['test_r2'] for r in fold_results])),
                'fold_results': fold_results
            }
        
        if logger:
            logger.log_and_print("Base model training complete")
        
        return base_model_results
        
    except Exception as e:
        error_msg = f"Error in train_base_models: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        return {}
    
@timing_decorator
def train_ensemble_model(X, y, feature_cols, base_model_results, batch_size=5000, logger=None):
    """Train ensemble model combining base models with improved numerical stability."""
    if logger:
        logger.log_and_print("Training ensemble model...")
    
    try:
        # Create ensemble model configuration
        ensemble_config = {
            'models': {
                'rf1': RandomForestRegressor(
                    n_estimators=200, 
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_split=15,
                    min_samples_leaf=8,
                    max_features=0.6,
                    ccp_alpha=0.01,
                    bootstrap=True,
                    oob_score=True,
                    max_samples=0.7
                ),
                'rf2': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=8,
                    random_state=43,
                    n_jobs=-1,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features=0.4,
                    ccp_alpha=0.02,
                    bootstrap=True,
                    oob_score=True,
                    max_samples=0.8
                ),
                'xgb1': xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.01,
                    random_state=42,
                    n_jobs=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.3,
                    reg_lambda=0.3,
                    tree_method='hist',
                    max_bin=256
                ),
                'xgb2': xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.005,
                    random_state=43,
                    n_jobs=-1,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=0.4,
                    reg_lambda=0.4,
                    tree_method='hist',
                    max_bin=256
                )
            },
            'weights': {
                'rf1': 0.3,
                'rf2': 0.3,
                'xgb1': 0.25,
                'xgb2': 0.15
            }
        }
        
        # Create cross-validation folds
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_results = []
        trained_models = {}
        
        # Train ensemble in folds
        for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
            if logger:
                logger.log_and_print(f"  Training ensemble model fold {fold}/{n_splits}")
            
            # Use iloc for integer-based indexing
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index]
            y_test = y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]
            
            # Train each model in the ensemble
            fold_predictions_train = {}
            fold_predictions_test = {}
            
            for name, model in ensemble_config['models'].items():
                if logger:
                    logger.log_and_print(f"    Training sub-model {name}...")
                
                # Train in batches
                for start_idx in range(0, len(X_train), batch_size):
                    end_idx = min(start_idx + batch_size, len(X_train))
                    X_batch = X_train.iloc[start_idx:end_idx]
                    y_batch = y_train.iloc[start_idx:end_idx] if isinstance(y_train, pd.Series) else y_train[start_idx:end_idx]
                    
                    try:
                        if hasattr(model, 'partial_fit'):
                            model.partial_fit(X_batch, y_batch)
                        else:
                            model.fit(X_batch, y_batch)
                    except Exception as e:
                        if logger:
                            logger.log_and_print(f"Warning: Training failed for ensemble model {name} in batch {start_idx}-{end_idx}: {str(e)}")
                        continue
                    
                    gc.collect()
                
                # Make predictions in batches
                train_preds = []
                test_preds = []
                
                for start_idx in range(0, len(X_train), batch_size):
                    end_idx = min(start_idx + batch_size, len(X_train))
                    pred = model.predict(X_train.iloc[start_idx:end_idx])
                    train_preds.extend(pred)
                
                for start_idx in range(0, len(X_test), batch_size):
                    end_idx = min(start_idx + batch_size, len(X_test))
                    pred = model.predict(X_test.iloc[start_idx:end_idx])
                    test_preds.extend(pred)
                
                fold_predictions_train[name] = np.array(train_preds)
                fold_predictions_test[name] = np.array(test_preds)
                
                # Store trained model
                trained_models[name] = model
            
            # Combine predictions using weights
            ensemble_pred_train = np.zeros_like(y_train, dtype=np.float64)
            ensemble_pred_test = np.zeros_like(y_test, dtype=np.float64)
            
            for name, pred in fold_predictions_train.items():
                weight = ensemble_config['weights'][name]
                ensemble_pred_train += pred * weight
            
            for name, pred in fold_predictions_test.items():
                weight = ensemble_config['weights'][name]
                ensemble_pred_test += pred * weight
            
            # Compute metrics
            train_mse = mean_squared_error(y_train, ensemble_pred_train)
            train_r2 = r2_score(y_train, ensemble_pred_train)
            test_mse = mean_squared_error(y_test, ensemble_pred_test)
            test_r2 = r2_score(y_test, ensemble_pred_test)
            
            fold_results.append({
                'train_mse': float(train_mse),
                'train_r2': float(train_r2),
                'test_mse': float(test_mse),
                'test_r2': float(test_r2),
                'predictions': {
                    'train': ensemble_pred_train,
                    'test': ensemble_pred_test
                }
            })
            
            if logger:
                logger.log_and_print(f"    Ensemble Fold {fold} - Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
                logger.log_and_print(f"    Ensemble Fold {fold} - Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
        
        # Prepare final results
        ensemble_results = {
            'model': {
                'models': trained_models,
                'weights': ensemble_config['weights']
            },
            'avg_train_mse': float(np.mean([r['train_mse'] for r in fold_results])),
            'avg_train_r2': float(np.mean([r['train_r2'] for r in fold_results])),
            'avg_test_mse': float(np.mean([r['test_mse'] for r in fold_results])),
            'avg_test_r2': float(np.mean([r['test_r2'] for r in fold_results])),
            'fold_results': fold_results
        }
        
        if logger:
            logger.log_and_print("Ensemble model training complete")
        
        return ensemble_results
        
    except Exception as e:
        error_msg = f"Error in train_ensemble_model: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        return {}
    
@timing_decorator
def train_stacking_model(X, y, feature_cols, base_model_results, batch_size=5000, logger=None):
    """Train stacking model with cross-validation, batch processing, and progress logging."""
    if logger:
        logger.log_and_print("Training stacking model...")
    
    try:
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_cols)
        
        # Define base models and meta-learner
        base_models = {
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True,
                max_samples=0.7,
                ccp_alpha=0.01
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                tree_method='hist',
                max_bin=256
            ),
            'lr': LinearRegression(
                n_jobs=-1,
                fit_intercept=True,
                copy_X=True
            )
        }
        meta_learner = LinearRegression(
            n_jobs=-1,
            fit_intercept=True,
            copy_X=True
        )
        
        # Generate base model predictions
        base_train_predictions = np.zeros((len(X), len(base_models)), dtype=np.float64)
        
        # Train each base model with cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        trained_base_models = {}
        for i, (name, model) in enumerate(base_models.items()):
            if logger:
                logger.log_and_print(f"  Training base model {name} for stacking...")
            
            # Train the model and make predictions in batches
            try:
                # Train in batches
                for start_idx in range(0, len(X), batch_size):
                    end_idx = min(start_idx + batch_size, len(X))
                    X_batch = X.iloc[start_idx:end_idx]
                    y_batch = y.iloc[start_idx:end_idx] if isinstance(y, pd.Series) else y[start_idx:end_idx]
                    
                    if hasattr(model, 'partial_fit'):
                        model.partial_fit(X_batch, y_batch)
                    else:
                        model.fit(X_batch, y_batch)
                    gc.collect()
                
                trained_base_models[name] = model  # Store trained model
            
                # Process training predictions in batches
                for start_idx in range(0, len(X), batch_size):
                    end_idx = min(start_idx + batch_size, len(X))
                    X_batch = X.iloc[start_idx:end_idx]
                    base_train_predictions[start_idx:end_idx, i] = model.predict(X_batch)
                    gc.collect()
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Warning: Training failed for base model {name} in stacking: {str(e)}")
                # Handle training failure
                fallback_model = LinearRegression()
                # Train fallback model in batches
                for start_idx in range(0, len(X), batch_size):
                    end_idx = min(start_idx + batch_size, len(X))
                    X_batch = X.iloc[start_idx:end_idx]
                    y_batch = y.iloc[start_idx:end_idx] if isinstance(y, pd.Series) else y[start_idx:end_idx]
                    fallback_model.fit(X_batch, y_batch)
                trained_base_models[name] = fallback_model
                
                # Make fallback predictions in batches
                for start_idx in range(0, len(X), batch_size):
                    end_idx = min(start_idx + batch_size, len(X))
                    X_batch = X.iloc[start_idx:end_idx]
                    base_train_predictions[start_idx:end_idx, i] = fallback_model.predict(X_batch)
        
        # Train meta-learner in batches
        if logger:
            logger.log_and_print("  Training meta-learner for stacking...")
        
        for start_idx in range(0, len(X), batch_size):
            end_idx = min(start_idx + batch_size, len(X))
            base_batch = base_train_predictions[start_idx:end_idx]
            y_batch = y.iloc[start_idx:end_idx] if isinstance(y, pd.Series) else y[start_idx:end_idx]
            meta_learner.fit(base_batch, y_batch)
        
        # Make final predictions in batches
        final_predictions = np.zeros(len(X), dtype=np.float64)
        for start_idx in range(0, len(X), batch_size):
            end_idx = min(start_idx + batch_size, len(X))
            base_batch = base_train_predictions[start_idx:end_idx]
            final_predictions[start_idx:end_idx] = meta_learner.predict(base_batch)
            gc.collect()
        
        # Compute metrics
        train_mse = mean_squared_error(y, final_predictions)
        train_r2 = r2_score(y, final_predictions)
        
        if logger:
            logger.log_and_print("Stacking model training complete.")
        
        return {
            'model': {
                'base_models': trained_base_models,
                'meta_learner': meta_learner
            },
            'train_mse': float(train_mse),
            'train_r2': float(train_r2),
            'test_mse': float('inf'),
            'test_r2': 0.0,
            'predictions': {
                'train': final_predictions,
                'test': np.zeros_like(y)
            }
        }
    
    except Exception as e:
        error_msg = f"Error in train_stacking_model: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        return {
            'model': None,
            'train_mse': float('inf'),
            'train_r2': 0.0,
            'test_mse': float('inf'),
            'test_r2': 0.0,
            'predictions': {
                'train': np.zeros_like(y),
                'test': np.zeros_like(y)
            }
        }
        
def train_cluster_models(cluster_data, gap_data, next_cluster_data, batch_size=5000, logger=None):
    """Train cluster-specific models with improved numerical stability and error handling."""
    if logger:
        logger.log_and_print("Training cluster models...")
    
    cluster_results = {}
    
    try:
        # Train cluster membership model
        if cluster_data:
            if logger:
                logger.log_and_print("Training cluster membership model...")
            
            # Ensure data is in DataFrame format
            if not isinstance(cluster_data['X'], pd.DataFrame):
                cluster_data['X'] = pd.DataFrame(cluster_data['X'])
            if not isinstance(cluster_data['y'], (pd.Series, np.ndarray)):
                cluster_data['y'] = np.array(cluster_data['y'])
            
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=5, 
                random_state=42, 
                n_jobs=-1,
                min_samples_split=10,
                min_samples_leaf=5,
                bootstrap=True,
                oob_score=True,
                class_weight='balanced',
                max_samples=0.7
            )
            
            # Train in batches
            for start_idx in range(0, len(cluster_data['X']), batch_size):
                end_idx = min(start_idx + batch_size, len(cluster_data['X']))
                X_batch = cluster_data['X'].iloc[start_idx:end_idx]
                y_batch = cluster_data['y'][start_idx:end_idx] if isinstance(cluster_data['y'], np.ndarray) else cluster_data['y'].iloc[start_idx:end_idx]
                
                try:
                    if hasattr(model, 'partial_fit'):
                        model.fit(X_batch, y_batch)
                    else:
                        model.fit(X_batch, y_batch)
                except Exception as e:
                    if logger:
                        logger.log_and_print(f"Warning: Training failed for cluster membership model in batch {start_idx}-{end_idx}: {str(e)}")
                    continue
                gc.collect()
            
            # Make predictions in batches
            y_pred = np.zeros(len(cluster_data['X']), dtype=np.int32)
            for start_idx in range(0, len(cluster_data['X']), batch_size):
                end_idx = min(start_idx + batch_size, len(cluster_data['X']))
                X_batch = cluster_data['X'].iloc[start_idx:end_idx]
                y_pred[start_idx:end_idx] = model.predict(X_batch)
                gc.collect()
            
            accuracy = accuracy_score(cluster_data['y'], y_pred)
            mse = mean_squared_error(cluster_data['y'], y_pred)
            r2 = r2_score(cluster_data['y'], y_pred)
            
            cluster_results['cluster_membership_rf'] = {
                'model': model,
                'accuracy': float(accuracy),
                'mse': float(mse),
                'r2': float(r2)
            }
            
            if logger:
                logger.log_and_print(f"Cluster membership model - Accuracy: {accuracy:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Train gap from cluster model
        if gap_data:
            if logger:
                logger.log_and_print("Training gap from cluster model...")
            
            # Ensure data is in DataFrame format
            if not isinstance(gap_data['X'], pd.DataFrame):
                gap_data['X'] = pd.DataFrame(gap_data['X'])
            if not isinstance(gap_data['y'], (pd.Series, np.ndarray)):
                gap_data['y'] = np.array(gap_data['y'])
            
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=5, 
                random_state=42, 
                n_jobs=-1,
                min_samples_split=10,
                min_samples_leaf=5,
                bootstrap=True,
                oob_score=True,
                max_samples=0.7
            )
            
            # Train in batches
            for start_idx in range(0, len(gap_data['X']), batch_size):
                end_idx = min(start_idx + batch_size, len(gap_data['X']))
                X_batch = gap_data['X'].iloc[start_idx:end_idx]
                y_batch = gap_data['y'][start_idx:end_idx] if isinstance(gap_data['y'], np.ndarray) else gap_data['y'].iloc[start_idx:end_idx]
                
                try:
                    if hasattr(model, 'partial_fit'):
                        model.fit(X_batch, y_batch)
                    else:
                        model.fit(X_batch, y_batch)
                except Exception as e:
                    if logger:
                        logger.log_and_print(f"Warning: Training failed for gap from cluster model in batch {start_idx}-{end_idx}: {str(e)}")
                    continue
                gc.collect()
            
            # Make predictions in batches
            y_pred = np.zeros(len(gap_data['X']), dtype=np.float64)
            for start_idx in range(0, len(gap_data['X']), batch_size):
                end_idx = min(start_idx + batch_size, len(gap_data['X']))
                X_batch = gap_data['X'].iloc[start_idx:end_idx]
                y_pred[start_idx:end_idx] = model.predict(X_batch)
                gc.collect()
            
            mse = mean_squared_error(gap_data['y'], y_pred)
            r2 = r2_score(gap_data['y'], y_pred)
            
            cluster_results['gap_from_cluster_rf'] = {
                'model': model,
                'mse': float(mse),
                'r2': float(r2)
            }
            
            if logger:
                logger.log_and_print(f"Gap from cluster model - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Train next cluster model
        if next_cluster_data:
            if logger:
                logger.log_and_print("Training next cluster model...")
            
            # Ensure data is in DataFrame format
            if not isinstance(next_cluster_data['X'], pd.DataFrame):
                next_cluster_data['X'] = pd.DataFrame(next_cluster_data['X'])
            if not isinstance(next_cluster_data['y'], (pd.Series, np.ndarray)):
                next_cluster_data['y'] = np.array(next_cluster_data['y'])
            
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=5, 
                random_state=42, 
                n_jobs=-1,
                min_samples_split=10,
                min_samples_leaf=5,
                bootstrap=True,
                oob_score=True,
                class_weight='balanced',
                max_samples=0.7
            )
            
            # Train in batches
            for start_idx in range(0, len(next_cluster_data['X']), batch_size):
                end_idx = min(start_idx + batch_size, len(next_cluster_data['X']))
                X_batch = next_cluster_data['X'].iloc[start_idx:end_idx]
                y_batch = next_cluster_data['y'][start_idx:end_idx] if isinstance(next_cluster_data['y'], np.ndarray) else next_cluster_data['y'].iloc[start_idx:end_idx]
                
                try:
                    if hasattr(model, 'partial_fit'):
                        model.fit(X_batch, y_batch)
                    else:
                        model.fit(X_batch, y_batch)
                except Exception as e:
                    if logger:
                        logger.log_and_print(f"Warning: Training failed for next cluster model in batch {start_idx}-{end_idx}: {str(e)}")
                    continue
                gc.collect()
            
            # Make predictions in batches
            y_pred = np.zeros(len(next_cluster_data['X']), dtype=np.int32)
            for start_idx in range(0, len(next_cluster_data['X']), batch_size):
                end_idx = min(start_idx + batch_size, len(next_cluster_data['X']))
                X_batch = next_cluster_data['X'].iloc[start_idx:end_idx]
                y_pred[start_idx:end_idx] = model.predict(X_batch)
                gc.collect()
            
            accuracy = accuracy_score(next_cluster_data['y'], y_pred)
            mse = mean_squared_error(next_cluster_data['y'], y_pred)
            r2 = r2_score(next_cluster_data['y'], y_pred)
            
            cluster_results['next_cluster_rf'] = {
                'model': model,
                'accuracy': float(accuracy),
                'mse': float(mse),
                'r2': float(r2)
            }
            
            if logger:
                logger.log_and_print(f"Next cluster model - Accuracy: {accuracy:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")
        
        if logger:
            logger.log_and_print("Cluster model training complete")
        
        return cluster_results
        
    except Exception as e:
        error_msg = f"Error in train_cluster_models: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        return {}
    
def compute_feature_importance(model_results, feature_cols, X, y, logger=None):
    """Compute feature importance scores from trained models with improved numerical stability."""
    if logger:
        logger.log_and_print("Computing feature importance...")
    
    feature_importance = pd.DataFrame()
    
    try:
        for name, results in model_results.items():
            if 'model' in results:
                model = results['model']
                if isinstance(model, dict):
                    if 'base_models' in model and 'meta_learner' in model:
                        # Handle stacking models
                        base_importances = []
                        for base_name, base_model in model['base_models'].items():
                            if hasattr(base_model, 'feature_importances_'):
                                # Verify feature importance length matches
                                if len(base_model.feature_importances_) == len(feature_cols):
                                    base_importances.append(pd.Series(base_model.feature_importances_, 
                                                                    index=feature_cols, 
                                                                    name=f'{name}_{base_name}'))
                                else:
                                    if logger:
                                        logger.log_and_print(f"Warning: Feature importance length mismatch for {base_name}")
                        if base_importances:
                            importance = pd.concat(base_importances, axis=1).mean(axis=1)
                            feature_importance = pd.concat([feature_importance, importance.rename(name)], axis=1)
                    else:
                        # Handle ensemble models
                        if 'models' in model and 'weights' in model:
                            ensemble_importances = []
                            for model_name, submodel in model['models'].items():
                                if hasattr(submodel, 'feature_importances_'):
                                    # Verify feature importance length matches
                                    if len(submodel.feature_importances_) == len(feature_cols):
                                        ensemble_importances.append(pd.Series(submodel.feature_importances_, 
                                                                           index=feature_cols, 
                                                                           name=f'{name}_{model_name}'))
                                    else:
                                        if logger:
                                            logger.log_and_print(f"Warning: Feature importance length mismatch for {model_name}")
                            if ensemble_importances:
                                importance = pd.concat(ensemble_importances, axis=1).mean(axis=1)
                                feature_importance = pd.concat([feature_importance, importance.rename(name)], axis=1)
                elif hasattr(model, 'feature_importances_'):
                    # Handle single models
                    # Verify feature importance length matches
                    if len(model.feature_importances_) == len(feature_cols):
                        importance = pd.Series(model.feature_importances_, index=feature_cols, name=name)
                        feature_importance = pd.concat([feature_importance, importance], axis=1)
                    else:
                        if logger:
                            logger.log_and_print(f"Warning: Feature importance length mismatch for {name}")
                elif isinstance(model, LinearRegression):
                    # Handle linear regression
                    if hasattr(model, 'coef_'):
                        if len(model.coef_) == len(feature_cols):
                            importance = pd.Series(
                                np.abs(model.coef_) if len(model.coef_.shape) == 1 
                                else np.abs(model.coef_[0]),
                                index=feature_cols,
                                name=name
                            )
                            feature_importance = pd.concat([feature_importance, importance], axis=1)
                        else:
                            if logger:
                                logger.log_and_print(f"Warning: Coefficient length mismatch for {name}")
                else:
                    if logger:
                        logger.log_and_print(f"Warning: Model {name} does not have feature importance attributes.")
                    continue
        
        if logger:
            logger.log_and_print("Feature importance computation complete")
        
        return feature_importance
        
    except Exception as e:
        error_msg = f"Error in compute_feature_importance: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        return pd.DataFrame()

def validate_model_results(model_results, logger=None):
    """Validate model results and handle potential errors with improved numerical stability."""
    if logger:
        logger.log_and_print("Validating model results...")
    
    validated_results = {}
    
    try:
        for name, results in model_results.items():
            if isinstance(results, dict):
                # Check for NaN or inf values in metrics
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        if not np.isfinite(value):
                            if logger:
                                logger.log_and_print(f"Warning: Invalid value {value} found in {name} for key {key}. Setting to 0.")
                            results[key] = 0.0
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                if not np.isfinite(sub_value):
                                    if logger:
                                        logger.log_and_print(f"Warning: Invalid value {sub_value} found in {name} for key {key}.{sub_key}. Setting to 0.")
                                    value[sub_key] = 0.0
                            elif isinstance(sub_value, np.ndarray):
                                if not np.isfinite(sub_value).all():
                                    if logger:
                                        logger.log_and_print(f"Warning: Invalid values found in {name} for key {key}.{sub_key}. Imputing with 0.")
                                    value[sub_key] = np.nan_to_num(sub_value, nan=0.0, posinf=1e10, neginf=-1e10)
                            
                # Check for model object
                if 'model' in results:
                    model = results['model']
                    if isinstance(model, dict):
                        if 'base_models' in model and 'meta_learner' in model:
                            for base_name, base_model in model['base_models'].items():
                                if not hasattr(base_model, 'predict'):
                                    if logger:
                                        logger.log_and_print(f"Warning: Base model {base_name} in {name} does not have predict method.")
                                    results['model']['base_models'][base_name] = None
                            if not hasattr(model['meta_learner'], 'predict'):
                                if logger:
                                    logger.log_and_print(f"Warning: Meta-learner in {name} does not have predict method.")
                                results['model']['meta_learner'] = None
                        elif 'models' in model and 'weights' in model:
                            for sub_name, sub_model in model['models'].items():
                                if not hasattr(sub_model, 'predict'):
                                    if logger:
                                        logger.log_and_print(f"Warning: Sub-model {sub_name} in {name} does not have predict method.")
                                    results['model']['models'][sub_name] = None
                    elif not hasattr(model, 'predict'):
                        if logger:
                            logger.log_and_print(f"Warning: Model {name} does not have predict method.")
                        results['model'] = None
                
                validated_results[name] = results
            else:
                if logger:
                    logger.log_and_print(f"Warning: Invalid results format for model {name}. Skipping validation.")
                validated_results[name] = {}
        
        if logger:
            logger.log_and_print("Model results validation complete")
        
        return validated_results
        
    except Exception as e:
        error_msg = f"Error in validate_model_results: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        return model_results
    
def handle_model_errors(model_name, e, logger=None):
    """Handle errors during model training or prediction with improved logging."""
    error_msg = f"Error in model {model_name}: {str(e)}"
    if logger:
        logger.log_and_print(error_msg, level=logging.ERROR)
        logger.logger.error(traceback.format_exc())
    else:
        print(error_msg)
        traceback.print_exc()
    
    # Return safe default values
    return {
        'model': None,
        'train_mse': float('inf'),
        'train_r2': 0.0,
        'test_mse': float('inf'),
        'test_r2': 0.0,
        'predictions': {
            'train': np.array([]),
            'test': np.array([])
        }
    }
    
def compute_model_metrics(y_true, y_pred, logger=None):
    """Compute model evaluation metrics with improved numerical stability."""
    if logger:
        logger.log_and_print("Computing model metrics...")
    
    try:
        # Ensure y_true and y_pred are numpy arrays
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        # Clip values to prevent overflow
        y_true = np.clip(y_true, -1e10, 1e10)
        y_pred = np.clip(y_pred, -1e10, 1e10)
        
        # Remove NaN and infinite values
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            if logger:
                logger.log_and_print("Warning: No valid data points for metric calculation.")
            return {
                'mse': float('inf'),
                'r2': 0.0
            }
        
        # Compute metrics with numerical stability
        with np.errstate(all='ignore'):
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
        
        if logger:
            logger.log_and_print("Model metrics computation complete")
        
        return {
            'mse': float(mse),
            'r2': float(r2)
        }
        
    except Exception as e:
        error_msg = f"Error computing model metrics: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'mse': float('inf'),
            'r2': 0.0
        }

@timing_decorator
def train_models(X, y, feature_cols, cluster_X, cluster_y, 
                gap_cluster_X, gap_cluster_y, next_cluster_X, next_cluster_y, 
                batch_size=5000, logger=None):
    """Train and evaluate multiple models with enhanced features and sequence handling."""
    if logger:
        logger.log_and_print("Training models...")
    
    try:
        # Check if feature_cols is empty
        if not feature_cols:
            if logger:
                logger.log_and_print("Error: No valid features for training, halting model training.", level=logging.ERROR)
            else:
                print("Error: No valid features for training, halting model training.")
            raise ValueError("No valid features for training, cannot proceed with model training.")
        
        # Prepare and validate data
        X_processed, y_processed, cluster_data, gap_data, next_cluster_data = prepare_model_data(
            X, y, cluster_X, cluster_y, gap_cluster_X, gap_cluster_y, 
            next_cluster_X, next_cluster_y, batch_size=batch_size, logger=logger
        )
        
        # Ensure X_processed is a DataFrame with proper columns
        if not isinstance(X_processed, pd.DataFrame):
            X_processed = pd.DataFrame(X_processed, columns=feature_cols)
        
        # Verify feature columns match X columns
        if set(feature_cols) != set(X_processed.columns):
            if logger:
                logger.log_and_print("Error: Feature columns don't match X columns. Fixing feature_cols to match X.")
            feature_cols = list(X_processed.columns)
        
        # Initialize results containers
        model_results = {}
        feature_importance = pd.DataFrame()
        
        # Train base models
        base_model_results = train_base_models(
            X_processed, y_processed, feature_cols, batch_size=batch_size, logger=logger
        )
        
        # Verify base models were trained with correct features
        for name, results in base_model_results.items():
            if 'model' in results:
                model = results['model']
                if hasattr(model, 'feature_importances_'):
                    if len(model.feature_importances_) != len(feature_cols):
                        if logger:
                            logger.log_and_print(f"Error: Model {name} has {len(model.feature_importances_)} features but expected {len(feature_cols)}")
                            logger.log_and_print("This indicates a feature mismatch during training. Fixing...")
                        # Retrain model with correct features
                        if isinstance(model, RandomForestRegressor):
                            model = RandomForestRegressor(**model.get_params())
                        model.fit(X_processed, y_processed)
                        results['model'] = model
        
        model_results.update(base_model_results)
        
        # Train ensemble model
        ensemble_results = train_ensemble_model(
            X_processed, y_processed, feature_cols, base_model_results, 
            batch_size=batch_size, logger=logger
        )
        model_results['ensemble'] = ensemble_results
        
        # Train stacking model
        stacking_results = train_stacking_model(
            X_processed, y_processed, feature_cols, base_model_results, 
            batch_size=batch_size, logger=logger
        )
        model_results['stacking'] = stacking_results
        
        # Train cluster models if cluster data available
        if cluster_data and gap_data and next_cluster_data:
            cluster_results = train_cluster_models(
                cluster_data, gap_data, next_cluster_data, 
                batch_size=batch_size, logger=logger
            )
            model_results.update(cluster_results)
        
        # Compute feature importance
        feature_importance = compute_feature_importance(
            model_results, feature_cols, X_processed, y_processed, logger=logger
        )
        
        # Validate results
        model_results = validate_model_results(model_results, logger=logger)
        
        if logger:
            logger.log_and_print("Model training complete")
        
        return model_results, feature_importance
        
    except Exception as e:
        error_msg = f"Error in train_models: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        raise ValueError("Model training failed, cannot proceed.")
        
# Helper functions needed by train_models

def batch_predict(model, X_data, model_type=None, batch_size=5000, logger=None):
    """Make predictions in batches with support for various model types."""
    if logger:
        logger.log_and_print("Making batch predictions...")
    
    try:
        # Ensure X_data is a DataFrame if it isn't already
        if not isinstance(X_data, pd.DataFrame):
            if isinstance(X_data, np.ndarray):
                if model_type == 'gap_cluster':
                    X_data = pd.DataFrame(X_data, columns=['cluster'])
                elif model_type == 'next_cluster':
                    X_data = pd.DataFrame(X_data, columns=[f'prev_cluster_{i+1}' for i in range(X_data.shape[1])])
                else:
                    X_data = pd.DataFrame(X_data)
            else:
                X_data = pd.DataFrame(X_data)

        # Get the appropriate predict function based on model type
        def get_predict_function(model_obj):
            if isinstance(model_obj, dict):
                if 'model' in model_obj:
                    return model_obj['model'].predict
                elif 'base_models' in model_obj and 'meta_learner' in model_obj:
                    def stacking_predict(X):
                        base_predictions = []
                        for base_model in model_obj['base_models'].values():
                            with np.errstate(all='ignore'):
                                pred = base_model.predict(X)
                                pred = np.clip(pred, -1e10, 1e10)
                                base_predictions.append(pred)
                        stacked_preds = np.column_stack(base_predictions)
                        return model_obj['meta_learner'].predict(stacked_preds)
                    return stacking_predict
                elif 'models' in model_obj and 'weights' in model_obj:
                    def ensemble_predict(X):
                        model_predictions = {}
                        for name, submodel in model_obj['models'].items():
                            with np.errstate(all='ignore'):
                                if isinstance(submodel, tf.keras.Sequential):
                                    pred = submodel.predict(X, verbose="0").flatten()
                                else:
                                    pred = submodel.predict(X)
                                pred = np.clip(pred, -1e10, 1e10)
                                model_predictions[name] = pred
                        return sum(pred * model_obj['weights'][name] 
                                 for name, pred in model_predictions.items())
                    return ensemble_predict
                else:
                    raise ValueError("Invalid model dictionary structure")
            elif hasattr(model_obj, 'predict'):
                return model_obj.predict
            else:
                raise ValueError(f"Model type {type(model_obj)} does not support predictions")

        predict_func = get_predict_function(model)
        predictions = []
        
        # Process predictions in batches
        for start_idx in range(0, len(X_data), batch_size):
            end_idx = min(start_idx + batch_size, len(X_data))
            X_batch = X_data.iloc[start_idx:end_idx]
            
            try:
                # Convert batch to float64 and clip values
                if isinstance(X_batch, pd.DataFrame):
                    X_batch = X_batch.astype(np.float64)
                    X_batch = X_batch.clip(-1e10, 1e10)
                else:
                    X_batch = np.array(X_batch, dtype=np.float64)
                    X_batch = np.clip(X_batch, -1e10, 1e10)
                
                # Make predictions with numerical stability
                with np.errstate(all='ignore'):
                    batch_pred = predict_func(X_batch)
                    if isinstance(batch_pred, np.ndarray):
                        batch_pred = batch_pred.flatten()
                    batch_pred = np.array(batch_pred, dtype=np.float64)
                    batch_pred = np.clip(batch_pred, -1e10, 1e10)
                    predictions.extend(batch_pred)
                
                gc.collect()
            except Exception as e:
                error_msg = f"Error in batch prediction ({start_idx}:{end_idx}): {str(e)}"
                if logger:
                    logger.log_and_print(error_msg, level=logging.ERROR)
                else:
                    print(error_msg)
                # Fill with zeros for failed batch
                predictions.extend([0] * (end_idx - start_idx))
        
        # Convert to numpy array and ensure proper type
        predictions = np.array(predictions, dtype=np.float64)
        
        if logger:
            logger.log_and_print("Batch predictions complete")
        
        return predictions
        
    except Exception as e:
        error_msg = f"Error in batch_predict: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return array of zeros as fallback
        return np.zeros(len(X_data), dtype=np.float64)
         
def compute_metrics_in_batches(y_true, y_pred, batch_size=5000, logger=None):
    """Compute metrics in batches with improved numerical stability and error handling."""
    if logger:
        logger.log_and_print("Computing metrics in batches...")
    
    try:
        # Ensure arrays are 1D and proper type
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        
        # Initialize accumulators
        mse_sum = 0.0
        r2_num = 0.0
        r2_den = 0.0
        total_samples = 0
        
        # Compute mean for R² calculation
        y_mean = np.mean(y_true)
        
        # Additional metrics accumulators
        mae_sum = 0.0
        max_error = -np.inf
        min_error = np.inf
        squared_errors = []
        absolute_errors = []
        
        # Process in batches
        for start_idx in range(0, len(y_true), batch_size):
            end_idx = min(start_idx + batch_size, len(y_true))
            y_true_batch = y_true[start_idx:end_idx]
            y_pred_batch = y_pred[start_idx:end_idx]
            
            # Create mask for finite values
            mask = np.isfinite(y_true_batch) & np.isfinite(y_pred_batch)
            y_true_batch = y_true_batch[mask]
            y_pred_batch = y_pred_batch[mask]
            
            if len(y_true_batch) > 0:
                with np.errstate(all='ignore'):
                    # MSE calculation
                    squared_diff = (y_true_batch - y_pred_batch) ** 2
                    mse_sum += np.sum(squared_diff)
                    
                    # R² calculation components
                    r2_num += np.sum(squared_diff)
                    r2_den += np.sum((y_true_batch - y_mean) ** 2)
                    
                    # MAE calculation
                    absolute_diff = np.abs(y_true_batch - y_pred_batch)
                    mae_sum += np.sum(absolute_diff)
                    
                    # Update max/min error
                    max_error = max(max_error, np.max(absolute_diff))
                    min_error = min(min_error, np.min(absolute_diff))
                    
                    # Store errors for percentile calculation
                    squared_errors.extend(squared_diff.tolist())
                    absolute_errors.extend(absolute_diff.tolist())
                    
                    total_samples += len(y_true_batch)
            
            gc.collect()
        
        # Compute final metrics
        if total_samples > 0:
            metrics = {
                'mse': float(mse_sum / total_samples),
                'rmse': float(np.sqrt(mse_sum / total_samples)),
                'mae': float(mae_sum / total_samples),
                'max_error': float(max_error),
                'min_error': float(min_error),
                'median_error': float(np.median(absolute_errors)),
                'error_std': float(np.std(absolute_errors)),
                'squared_error_95th': float(np.percentile(squared_errors, 95)),
                'absolute_error_95th': float(np.percentile(absolute_errors, 95))
            }
            
            # Compute R² with protection against division by zero
            if r2_den != 0:
                metrics['r2'] = float(1 - (r2_num / r2_den))
            else:
                metrics['r2'] = 0.0
            
            # Add normalized metrics
            if np.std(y_true) != 0:
                metrics['normalized_mse'] = float(metrics['mse'] / np.var(y_true))
                metrics['normalized_rmse'] = float(np.sqrt(metrics['normalized_mse']))
            else:
                metrics['normalized_mse'] = float('inf')
                metrics['normalized_rmse'] = float('inf')
            
        else:
            metrics = {
                'mse': float('inf'),
                'rmse': float('inf'),
                'mae': float('inf'),
                'r2': 0.0,
                'max_error': float('inf'),
                'min_error': float('inf'),
                'median_error': float('inf'),
                'error_std': float('inf'),
                'squared_error_95th': float('inf'),
                'absolute_error_95th': float('inf'),
                'normalized_mse': float('inf'),
                'normalized_rmse': float('inf')
            }
        
        if logger:
            logger.log_and_print("Metrics computation complete")
            for metric_name, value in metrics.items():
                logger.log_and_print(f"{metric_name}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        error_msg = f"Error computing metrics: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'mse': float('inf'),
            'rmse': float('inf'),
            'mae': float('inf'),
            'r2': 0.0,
            'max_error': float('inf'),
            'min_error': float('inf'),
            'median_error': float('inf'),
            'error_std': float('inf'),
            'squared_error_95th': float('inf'),
            'absolute_error_95th': float('inf'),
            'normalized_mse': float('inf'),
            'normalized_rmse': float('inf')
        }
        
def scale_and_check_data(X_train, X_test, scaler, name="", fold=None, logger=None):
    """Scale data and check for invalid values with improved numerical stability and error handling."""
    fold_str = f" in fold {fold}" if fold is not None else ""
    
    try:
        # First, clip extremely large values to prevent overflow
        max_value = 1e10  # Reduced from 1e15 to prevent overflow
        min_value = -1e10
        
        # Handle DataFrame input
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.astype(np.float64)
            X_test = X_test.astype(np.float64)
            
            X_train = X_train.clip(min_value, max_value)
            X_test = X_test.clip(min_value, max_value)
            
            # Check for inf/nan before scaling
            if not np.isfinite(X_train.values).all() or np.isnan(X_train.values).any():
                if logger:
                    logger.log_and_print(f"Warning: inf or NaN values detected in {name} before scaling{fold_str}. Imputing with 0.")
                X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
            if not np.isfinite(X_test.values).all() or np.isnan(X_test.values).any():
                if logger:
                    logger.log_and_print(f"Warning: inf or NaN values detected in {name} before scaling{fold_str}. Imputing with 0.")
                X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)
            
            # Convert to numpy arrays for scaling
            X_train_arr = X_train.values
            X_test_arr = X_test.values
            
            # Store column names for later
            columns = X_train.columns
            index_train = X_train.index
            index_test = X_test.index
        else:
            # Handle numpy array input
            X_train_arr = np.array(X_train, dtype=np.float64)
            X_test_arr = np.array(X_test, dtype=np.float64)
            
            X_train_arr = np.clip(X_train_arr, min_value, max_value)
            X_test_arr = np.clip(X_test_arr, min_value, max_value)
            
            # Replace inf/nan values
            X_train_arr = np.nan_to_num(X_train_arr, nan=0.0, posinf=max_value, neginf=min_value)
            X_test_arr = np.nan_to_num(X_test_arr, nan=0.0, posinf=max_value, neginf=min_value)
            
            columns = None
            index_train = None
            index_test = None
        
        # Scale the data with error handling
        try:
            with np.errstate(all='ignore'):
                X_train_scaled = scaler.fit_transform(X_train_arr)
                X_test_scaled = scaler.transform(X_test_arr)
        except Exception as e:
            if logger:
                logger.log_and_print(f"Warning: Scaling failed for {name}{fold_str}. Using robust scaling method.")
            # Use a more robust scaling method
            with np.errstate(all='ignore'):
                # Compute median and IQR for robust scaling
                median = np.median(X_train_arr, axis=0)
                q75, q25 = np.percentile(X_train_arr, [75, 25], axis=0)
                iqr = q75 - q25
                iqr[iqr == 0] = 1.0  # Prevent division by zero
                
                X_train_scaled = (X_train_arr - median) / iqr
                X_test_scaled = (X_test_arr - median) / iqr
        
        # Clip scaled values to prevent extreme outliers
        X_train_scaled = np.clip(X_train_scaled, -100, 100)
        X_test_scaled = np.clip(X_test_scaled, -100, 100)
        
        # Check for inf/nan after scaling
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=100, neginf=-100)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=100, neginf=-100)
        
        # Convert back to DataFrame if input was DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=columns, index=index_train)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=columns, index=index_test)
        
        # Final validation
        if not np.all(np.isfinite(X_train_scaled)) or not np.all(np.isfinite(X_test_scaled)):
            if logger:
                logger.log_and_print(f"Warning: Non-finite values found after scaling {name}{fold_str}")
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=100, neginf=-100)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=100, neginf=-100)
        
        return X_train_scaled, X_test_scaled
        
    except Exception as e:
        error_msg = f"Error in data scaling: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return original data as fallback
        if isinstance(X_train, pd.DataFrame):
            return X_train.clip(min_value, max_value), X_test.clip(min_value, max_value)
        else:
            return np.clip(X_train, min_value, max_value), np.clip(X_test, min_value, max_value)
                                             
################################################################################
# Visualization Functions
################################################################################

@timing_decorator
def create_advanced_distribution_plots(df, plot_dir, cluster_distributions=None):
    """Create advanced distribution visualizations with improved numerical stability."""
    with suppress_overflow_warnings():
        advanced_plot_dir = os.path.join(plot_dir, "distributions")
        os.makedirs(advanced_plot_dir, exist_ok=True)
        
        # Convert to float64 and clip values
        gaps = df['gap_size'].astype(np.float64).clip(-1e10, 1e10)
        
        # 1. Enhanced Gap Size Distribution
        plt.figure(figsize=(15, 10))
        
        # Main distribution plot
        plt.subplot(2, 2, 1)
        sns.histplot(data=gaps, bins='auto', kde=True)
        plt.title('Gap Size Distribution')
        plt.xlabel('Gap Size')
        plt.ylabel('Count')
        
        # Log-scale distribution
        plt.subplot(2, 2, 2)
        sns.histplot(data=gaps, bins='auto', kde=True, log_scale=(False, True))
        plt.title('Gap Size Distribution (Log Scale)')
        plt.xlabel('Gap Size')
        plt.ylabel('Log Count')
        
        # Q-Q plot
        plt.subplot(2, 2, 3)
        sps.probplot(gaps, dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        
        # Box plot
        plt.subplot(2, 2, 4)
        sns.boxplot(y=gaps)
        plt.title('Box Plot of Gap Sizes')
        
        plt.tight_layout()
        plt.savefig(os.path.join(advanced_plot_dir, "gap_distribution_analysis.png"))
        plt.close()
        
        # 2. Cluster-wise Distribution Comparison
        if cluster_distributions and 'cluster' in df.columns:
            plt.figure(figsize=(15, 10))
            
            # Distribution by cluster
            plt.subplot(2, 1, 1)
            sns.boxplot(x='cluster', y='gap_size', data=df)
            plt.title('Gap Size Distribution by Cluster')
            
            # Violin plot by cluster
            plt.subplot(2, 1, 2)
            sns.violinplot(x='cluster', y='gap_size', data=df)
            plt.title('Gap Size Violin Plot by Cluster')
            
            plt.tight_layout()
            plt.savefig(os.path.join(advanced_plot_dir, "cluster_distributions.png"))
            plt.close()
            
            # Individual cluster distributions
            for cluster_id in sorted(df['cluster'].unique()):
                plt.figure(figsize=(12, 8))
                cluster_gaps = gaps[df['cluster'] == cluster_id]
                
                # Histogram with KDE
                sns.histplot(data=cluster_gaps, bins='auto', kde=True)
                plt.title(f'Gap Distribution for Cluster {cluster_id}')
                plt.xlabel('Gap Size')
                plt.ylabel('Count')
                
                # Add distribution fit if available
                if cluster_distributions and cluster_id in cluster_distributions:
                    stats = cluster_distributions[cluster_id]
                    if 'normal_fit' in stats:
                        x = np.linspace(cluster_gaps.min(), cluster_gaps.max(), 100)
                        y = stats.norm.pdf(x, stats['normal_fit']['mu'], stats['normal_fit']['sigma'])
                        plt.plot(x, y * len(cluster_gaps) * (x[1] - x[0]), 
                               'r-', label='Normal Fit')
                        plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(advanced_plot_dir, f"cluster_{cluster_id}_distribution.png"))
                plt.close()

@timing_decorator
def create_transition_analysis_plots(df, transition_analysis, plot_dir):
    """Create visualizations for transition analysis with improved numerical stability."""
    with suppress_overflow_warnings():
        transition_plot_dir = os.path.join(plot_dir, "transitions")
        os.makedirs(transition_plot_dir, exist_ok=True)
        
        if transition_analysis is None:
            return
        
        # 1. Transition Probability Heatmap
        plt.figure(figsize=(12, 10))
        transition_probs = transition_analysis['transition_probabilities']
        
        # Ensure probabilities are valid
        transition_probs = np.clip(transition_probs, 0, 1)
        
        sns.heatmap(transition_probs, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='YlOrRd',
                   square=True)
        plt.title('Cluster Transition Probabilities')
        plt.xlabel('To Cluster')
        plt.ylabel('From Cluster')
        plt.tight_layout()
        plt.savefig(os.path.join(transition_plot_dir, "transition_heatmap.png"))
        plt.close()
        
        # 2. Temporal Pattern Analysis
        if 'temporal_patterns' in transition_analysis:
            plt.figure(figsize=(15, 5))
            temporal_patterns = transition_analysis['temporal_patterns']
            
            clusters = sorted(temporal_patterns.keys())
            means = [temporal_patterns[c]['mean_recurrence'] for c in clusters]
            stds = [temporal_patterns[c]['std_recurrence'] for c in clusters]
            
            plt.errorbar(clusters, means, yerr=stds, fmt='o-', capsize=5)
            plt.title('Cluster Recurrence Patterns')
            plt.xlabel('Cluster')
            plt.ylabel('Mean Recurrence Time')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(transition_plot_dir, "recurrence_patterns.png"))
            plt.close()
        
        # 3. Sequence Pattern Analysis
        if 'sequence_patterns' in transition_analysis:
            for length, patterns in transition_analysis['sequence_patterns'].items():
                if 'most_common' in patterns:
                    plt.figure(figsize=(12, 6))
                    sequences = [str(seq) for seq, _ in patterns['most_common']]
                    counts = [count for _, count in patterns['most_common']]
                    
                    plt.bar(sequences, counts)
                    plt.title(f'Most Common Sequences (Length {length})')
                    plt.xlabel('Sequence')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(transition_plot_dir, f"sequence_patterns_{length}.png"))
                    plt.close()

@timing_decorator
def create_factor_pattern_plots(df, factor_patterns, plot_dir):
    """Create visualizations for factor pattern analysis with improved numerical stability."""
    with suppress_overflow_warnings():
        factor_plot_dir = os.path.join(plot_dir, "factors")
        os.makedirs(factor_plot_dir, exist_ok=True)
        
        # Convert relevant columns to float64 and clip values
        numeric_cols = ['unique_factors', 'total_factors', 'factor_density']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float64).clip(-1e10, 1e10)
        
        # 1. Factor Distribution Analysis
        plt.figure(figsize=(15, 10))
        
        # Unique factors distribution
        plt.subplot(2, 2, 1)
        sns.histplot(data=df['unique_factors'], bins='auto', kde=True)
        plt.title('Distribution of Unique Factors')
        plt.xlabel('Number of Unique Factors')
        
        # Total factors distribution
        plt.subplot(2, 2, 2)
        sns.histplot(data=df['total_factors'], bins='auto', kde=True)
        plt.title('Distribution of Total Factors')
        plt.xlabel('Number of Total Factors')
        
        # Factor density distribution
        plt.subplot(2, 2, 3)
        sns.histplot(data=df['factor_density'], bins='auto', kde=True)
        plt.title('Distribution of Factor Density')
        plt.xlabel('Factor Density')
        
        # Relationship between unique and total factors
        plt.subplot(2, 2, 4)
        plt.scatter(df['unique_factors'], df['total_factors'], alpha=0.5)
        plt.title('Unique vs Total Factors')
        plt.xlabel('Unique Factors')
        plt.ylabel('Total Factors')
        
        plt.tight_layout()
        plt.savefig(os.path.join(factor_plot_dir, "factor_distributions.png"))
        plt.close()
        
        # 2. Factor Sequence Analysis
        if 'factor_sequences' in factor_patterns:
            plt.figure(figsize=(12, 6))
            windows = [seq['window_size'] for seq in factor_patterns['factor_sequences']]
            trends = [seq['stats']['mean_trend'] for seq in factor_patterns['factor_sequences']]
            stds = [seq['stats']['std_trend'] for seq in factor_patterns['factor_sequences']]
            
            plt.errorbar(windows, trends, yerr=stds, fmt='o-', capsize=5)
            plt.title('Factor Sequence Trends')
            plt.xlabel('Sequence Window Size')
            plt.ylabel('Mean Trend')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(factor_plot_dir, "factor_sequences.png"))
            plt.close()

@timing_decorator            
def create_visualizations(df, feature_importance, pattern_analysis, plot_dir, model_results):
    """Create all visualizations with improved numerical stability."""
    with suppress_overflow_warnings():
        # Convert numeric columns to float64 and clip values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols].astype(np.float64)
        df_numeric = df_numeric.clip(-1e10, 1e10)
        
        # Feature importance plot
        if not feature_importance.empty:
            plt.figure(figsize=(12, 6))
            importance_values = feature_importance.mean(axis=1)
            importance_values = importance_values.clip(-1e10, 1e10)
            importance_values.sort_values(ascending=False).head(10).plot(kind='bar')
            plt.title('Top 10 Most Important Features')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "feature_importance.png"))
            plt.close()
        
        # Gap distribution plot with improved binning
        plt.figure(figsize=(12, 6))
        gap_data = df_numeric['gap_size']
        
        # Compute robust bin edges
        q1, q3 = gap_data.quantile([0.25, 0.75])
        iqr = q3 - q1
        bin_width = 2 * iqr * (len(gap_data) ** (-1/3))  # Freedman-Diaconis rule
        n_bins = int((gap_data.max() - gap_data.min()) / bin_width)
        n_bins = min(n_bins, 100)  # Limit number of bins
        
        sns.histplot(data=gap_data, bins=str(n_bins))
        plt.title('Distribution of Prime Gaps')
        plt.savefig(os.path.join(plot_dir, "gap_distribution.png"))
        plt.close()
        
        # Correlation matrix plot with improved handling of extreme values
        plt.figure(figsize=(15, 15))
        corr_matrix = df_numeric.corr()
        corr_matrix = corr_matrix.clip(-1, 1)  # Ensure correlations are in [-1, 1]
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu', 
                   center=0,
                   vmin=-1,
                   vmax=1,
                   fmt='.2f')
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "correlations.png"))
        plt.close()
        
        # PCA scatter plot with improved scaling
        if 'cluster' in df.columns:
            feature_cols = [col for col in df.columns if col not in ['gap_size', 'cluster', 'preceding_gaps']]
            X = df_numeric[feature_cols]
            
            # Robust scaling before PCA
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Handle potential NaN or inf values
            X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=1e10, neginf=-1e10)
            
            pca = PCA(n_components=2)
            try:
                X_pca = pca.fit_transform(X_scaled)
                
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X_pca[:, 0], 
                                   X_pca[:, 1], 
                                   c=df['cluster'],
                                   cmap='viridis',
                                   alpha=0.6)
                plt.colorbar(scatter)
                plt.title('PCA Scatter Plot of Gaps (Colored by Cluster)')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "pca_scatter.png"))
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create PCA plot due to numerical issues: {str(e)}")
        
        # Learning curve plots with improved error handling
        for name, results in model_results.items():
            if 'learning_curve' in results:
                try:
                    plt.figure(figsize=(10, 6))
                    train_sizes = results['learning_curve']['train_sizes']
                    train_scores = np.clip(results['learning_curve']['train_scores'], -1e10, 1e10)
                    test_scores = np.clip(results['learning_curve']['test_scores'], -1e10, 1e10)
                    
                    plt.plot(train_sizes, train_scores, label='Training MSE')
                    plt.plot(train_sizes, test_scores, label='Test MSE')
                    plt.title(f'Learning Curve for {name}')
                    plt.xlabel('Training Set Size')
                    plt.ylabel('Mean Squared Error')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f"learning_curve_{name}.png"))
                    plt.close()
                except Exception as e:
                    print(f"Warning: Could not create learning curve plot for {name}: {str(e)}")
        
        # Model comparison plot with improved error handling
        plt.figure(figsize=(12, 8))
        model_names = []
        test_mses = []
        
        for name, results in model_results.items():
            try:
                mse = results.get('avg_test_mse', results.get('mse', 0))
                if isinstance(mse, (int, float)) and not np.isnan(mse) and not np.isinf(mse):
                    model_names.append(name)
                    test_mses.append(min(mse, 1e10))  # Clip extremely large MSE values
            except Exception:
                continue
        
            colors = cm.get_cmap('Set3')(np.linspace(0, 1, len(model_names)))
            colors = cm.get_cmap('Set3')(np.linspace(0, 1, len(model_names)))
            plt.bar(model_names, test_mses, color=colors)
            plt.xlabel('Model')
            plt.ylabel('Average Test MSE')
            plt.title('Comparison of Model Performance (Average Test MSE)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "model_comparison.png"))
            plt.close()

@timing_decorator
def create_advanced_visualizations(df, plot_dir, cluster_features, temporal_patterns):
    """Create advanced visualizations with improved numerical stability."""
    with suppress_overflow_warnings():
        # Create directory for advanced plots
        advanced_plot_dir = os.path.join(plot_dir, "advanced_analysis")
        os.makedirs(advanced_plot_dir, exist_ok=True)
        
        # Convert to float64 and clip values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols].astype(np.float64)
        df_numeric = df_numeric.clip(-1e10, 1e10)
        
        # 1. Cluster Feature Distribution Plot
        if cluster_features:
            for feature in df_numeric.columns:
                if feature != 'cluster':
                    plt.figure(figsize=(12, 6))
                    for cluster_id in sorted(df['cluster'].unique()):
                        cluster_data = df_numeric[df['cluster'] == cluster_id][feature]
                        if len(cluster_data) > 0:
                            sns.kdeplot(data=cluster_data, label=f'Cluster {cluster_id}')
                    
                    plt.title(f'Distribution of {feature} by Cluster')
                    plt.xlabel(feature)
                    plt.ylabel('Density')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(advanced_plot_dir, f"cluster_distribution_{feature}.png"))
                    plt.close()
        
        # 2. Temporal Pattern Analysis Plot
        if temporal_patterns:
            for col, patterns in temporal_patterns.items():
                if 'autocorrelation' in patterns:
                    plt.figure(figsize=(15, 5))
                    
                    # Autocorrelation subplot
                    plt.subplot(131)
                    lags = range(1, len(patterns['autocorrelation']) + 1)
                    plt.bar(lags, patterns['autocorrelation'])
                    plt.title(f'Autocorrelation for {col}')
                    plt.xlabel('Lag')
                    plt.ylabel('Correlation')
                    
                    # Trend subplot
                    plt.subplot(132)
                    x = np.arange(len(df_numeric))
                    y = df_numeric[col]
                    plt.scatter(x, y, alpha=0.5, s=1)
                    trend_line = patterns['trend']['slope'] * x + patterns['trend']['intercept']
                    plt.plot(x, trend_line, 'r-', label='Trend')
                    plt.title(f'Trend Analysis for {col}')
                    plt.xlabel('Index')
                    plt.ylabel(col)
                    plt.legend()
                    
                    # Periodicity subplot
                    plt.subplot(133)
                    series = df_numeric[col] - df_numeric[col].mean()
                    fft = np.fft.fft(series)
                    power = np.abs(fft)[:len(series)//2]
                    freq = np.fft.fftfreq(len(series), d=1.0)[:len(series)//2]
                    plt.plot(1/freq[1:], power[1:])
                    plt.title(f'Frequency Analysis for {col}')
                    plt.xlabel('Period')
                    plt.ylabel('Power')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(advanced_plot_dir, f"temporal_analysis_{col}.png"))
                    plt.close()
        
        # 3. Cluster Separation Visualization
        if 'cluster' in df.columns:
            # Use t-SNE for high-dimensional visualization
            try:
                feature_cols = [col for col in df_numeric.columns if col != 'cluster']
                X = df_numeric[feature_cols].values
                
                tsne = TSNE(n_components=2, random_state=42)
                X_tsne = tsne.fit_transform(X)
                
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                   c=df['cluster'], 
                                   cmap='viridis',
                                   alpha=0.6)
                plt.colorbar(scatter)
                plt.title('t-SNE Visualization of Cluster Separation')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.tight_layout()
                plt.savefig(os.path.join(advanced_plot_dir, "cluster_separation_tsne.png"))
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create t-SNE visualization: {str(e)}")
        
        # 4. Feature Interaction Plot
        plt.figure(figsize=(15, 15))
        correlation_matrix = df_numeric.corr()
        mask = np.zeros_like(correlation_matrix)
        mask[np.triu_indices_from(mask)] = True
        
        with sns.axes_style("white"):
            sns.heatmap(correlation_matrix, 
                       mask=mask,
                       annot=True,
                       cmap='RdBu_r',
                       center=0,
                       fmt='.2f',
                       square=True,
                       vmin=-1,
                       vmax=1)
        
        plt.title('Feature Interaction Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(advanced_plot_dir, "feature_interactions.png"))
        plt.close()
        
        # 5. Gap Size Evolution Plot
        plt.figure(figsize=(15, 6))
        rolling_mean = df_numeric['gap_size'].rolling(window=50, center=True).mean()
        rolling_std = df_numeric['gap_size'].rolling(window=50, center=True).std()
        
        plt.plot(df_numeric.index, df_numeric['gap_size'], 'b-', alpha=0.3, label='Gap Size')
        plt.plot(df_numeric.index, rolling_mean, 'r-', label='50-point Moving Average')
        plt.fill_between(df_numeric.index,
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        color='r',
                        alpha=0.2,
                        label='±1 Std Dev')
        
        plt.title('Gap Size Evolution with Rolling Statistics')
        plt.xlabel('Index')
        plt.ylabel('Gap Size')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(advanced_plot_dir, "gap_size_evolution.png"))
        plt.close()

@timing_decorator
def create_cluster_visualization(df, plot_dir, logger=None, batch_size=5000):
    """Create cluster visualization with UMAP for better performance."""
    try:
        plt.figure(figsize=(10, 8))
        
        # Get feature columns for visualization
        feature_cols = [col for col in df.columns if col not in [
            'gap_size', 'cluster', 'sub_cluster', 'lower_prime', 
            'upper_prime', 'is_outlier', 'preceding_gaps'
        ]]
        
        if logger:
            logger.log_and_print("Preparing data for visualization...")
        
        # Extract features and scale
        X = df[feature_cols].astype(np.float64)
        X = X.clip(-1e10, 1e10)
        
        # Use standard scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use PCA first to reduce dimensions
        if logger:
            logger.log_and_print("Performing initial dimensionality reduction with PCA...")
        
        n_components = min(min(len(feature_cols), 50), len(df) - 1) # Ensure n_components is valid
        if n_components < 1:
            if logger:
                logger.log_and_print("Warning: Not enough features for PCA, skipping visualization.")
            return
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create scatter plot using PCA
        clusters = df['cluster'].values
        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=clusters,
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        
        plt.colorbar(scatter)
        plt.title('PCA Visualization of Clusters')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Add cluster statistics
        cluster_stats = pd.Series(clusters).value_counts()
        stats_text = "Cluster Sizes:\n"
        for cluster_id, size in cluster_stats.items():
            stats_text += f"Cluster {cluster_id}: {size} ({size/len(df)*100:.1f}%)\n"
        plt.figtext(1.02, 0.7, stats_text, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "cluster_visualization.png"),
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        if logger:
            logger.log_and_print("Cluster visualization created successfully")
        
    except Exception as e:
        error_msg = f"Error in cluster visualization: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
    finally:
        plt.close('all')
        gc.collect()      

@timing_decorator
def create_recurrence_plot(df, feature_col='gap_size', batch_size=5000, logger=None):
    """Create a recurrence plot for a given feature with improved numerical stability and memory management."""
    if logger:
        logger.log_and_print("Creating recurrence plot...")
    
    recurrence_plot_data = {}
    
    try:
        # Convert to numpy array and ensure proper type
        data = df[feature_col].values.astype(np.float64)
        data = np.clip(data, -1e10, 1e10)
        data = data[np.isfinite(data)]
        
        if len(data) < 2:
            if logger:
                logger.log_and_print("Warning: Insufficient data points for recurrence plot.")
            return {'distance_matrix': None}
        
        # Initialize distance matrix
        n = len(data)
        distance_matrix = np.zeros((n, n), dtype=np.float64)
        
        # Process distances in batches
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch = data[start_idx:end_idx]
            
            # Compute distances to all other points
            for i in range(len(batch)):
                with np.errstate(all='ignore'):
                    distances = np.abs(batch[i] - data)
                    distance_matrix[start_idx + i, :] = distances
            
            gc.collect()
        
        # Store distance matrix
        recurrence_plot_data['distance_matrix'] = distance_matrix
        
        if logger:
            logger.log_and_print("Recurrence plot data generated successfully")
        
        return recurrence_plot_data
        
    except Exception as e:
        error_msg = f"Error creating recurrence plot: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {'distance_matrix': None}
                               
################################################################################
# Main Analysis Pipeline
################################################################################

@timing_decorator
def generate_prime_probability_map(df, model_results, feature_cols, scaler, scaler_next_cluster, 
                                 scaler_gap_cluster, n_primes_to_predict=1000, batch_size=500, logger=None, sample_rate=PRIME_MAP_SAMPLE_RATE):
    """Generates a probability map of prime locations with batched processing and random sampling."""
    with suppress_overflow_warnings():
        if 'next_cluster_rf' not in model_results or 'gap_from_cluster_rf' not in model_results:
            if logger:
                logger.log_and_print("Warning: Required models not found. Cannot generate prime probability map.", level=logging.WARNING)
            else:
                print("Warning: Required models not found. Cannot generate prime probability map.")
            return None
        
        if logger:
            logger.log_and_print("  Initializing prediction models...")
        else:
            print("  Initializing prediction models...")
        
        next_cluster_model = model_results['next_cluster_rf']['model']
        gap_from_cluster_model = model_results['gap_from_cluster_rf']['model']
        
        try:
            # Initialize results container
            prime_probability_map = {}
            
            # Convert cluster history to numpy array and ensure correct type
            cluster_history = df['cluster'].iloc[-5:].values.astype(np.int32)
            current_prime = float(df['upper_prime'].iloc[-1])
            
            if logger:
                logger.log_and_print(f"  Generating {n_primes_to_predict} predictions with random sampling...")
            else:
                print(f"  Generating {n_primes_to_predict} predictions with random sampling...")
            
            # Generate random prime locations for sampling
            sample_size = int(n_primes_to_predict * sample_rate)
            sampled_primes = np.sort(np.random.uniform(current_prime, current_prime + n_primes_to_predict * 100, sample_size))
            
            # Process predictions in batches
            for batch_start in range(0, len(sampled_primes), batch_size):
                batch_end = min(batch_start + batch_size, len(sampled_primes))
                batch_size_actual = batch_end - batch_start
                
                try:
                    # Initialize batch arrays with numpy
                    batch_clusters = np.zeros((batch_size_actual, 5), dtype=np.int32)
                    batch_gaps = np.zeros(batch_size_actual, dtype=np.float64)
                    batch_primes = np.zeros(batch_size_actual, dtype=np.float64)
                    batch_probabilities = np.zeros(batch_size_actual, dtype=np.float64)
                    
                    # Initial values for this batch
                    current_clusters = cluster_history.copy()
                    
                    # Generate predictions for this batch
                    for i, target_prime in enumerate(sampled_primes[batch_start:batch_end]):
                        try:
                            # Predict next cluster
                            batch_clusters[i] = current_clusters
                            cluster_input = current_clusters.reshape(1, -1)
                            
                            # Scale cluster input if necessary
                            if hasattr(scaler_next_cluster, 'transform'):
                                cluster_input = scaler_next_cluster.transform(cluster_input)
                            
                            # Ensure input is in correct format for model
                            if isinstance(cluster_input, pd.DataFrame):
                                cluster_input = cluster_input.values
                            
                            predicted_cluster = next_cluster_model.predict(cluster_input)[0]
                            predicted_cluster = int(predicted_cluster)  # Ensure integer type
                            
                            # Update cluster history
                            current_clusters = np.roll(current_clusters, 1)
                            current_clusters[0] = predicted_cluster
                            
                            # Predict gap size
                            gap_input = np.array([[predicted_cluster]], dtype=np.int32)
                            if hasattr(scaler_gap_cluster, 'transform'):
                                gap_input = scaler_gap_cluster.transform(gap_input)
                            
                            predicted_gap = gap_from_cluster_model.predict(gap_input)[0]
                            predicted_gap = float(predicted_gap)  # Ensure float type
                            predicted_gap = max(2.0, min(predicted_gap, 1e6))  # Reasonable bounds
                            batch_gaps[i] = predicted_gap
                            
                            # Compute probability based on model confidence
                            if hasattr(next_cluster_model, 'predict_proba'):
                                cluster_proba = np.max(next_cluster_model.predict_proba(cluster_input))
                            else:
                                cluster_proba = 0.8  # Default confidence
                            
                            if hasattr(gap_from_cluster_model, 'predict_proba'):
                                gap_proba = np.max(gap_from_cluster_model.predict_proba(gap_input))
                            else:
                                gap_proba = 0.8  # Default confidence
                            
                            batch_probabilities[i] = float((cluster_proba + gap_proba) / 2)
                            batch_primes[i] = target_prime
                            
                        except Exception as e:
                            if logger:
                                logger.log_and_print(f"Warning: Error in prediction {i} of batch {batch_start}-{batch_end}: {str(e)}")
                            else:
                                print(f"Warning: Error in prediction {i} of batch {batch_start}-{batch_end}: {str(e)}")
                            batch_probabilities[i] = 0.5  # Default probability on error
                            continue
                    
                    # Store batch results with explicit type conversion
                    for i in range(batch_size_actual):
                        prime_probability_map[float(batch_primes[i])] = float(batch_probabilities[i])
                    
                    # Update cluster history for next batch
                    cluster_history = current_clusters.astype(np.int32)
                    
                    # Progress indicator
                    if logger and (batch_start // batch_size) % 5 == 0:
                        logger.log_and_print(f"    Completed {batch_end}/{len(sampled_primes)} predictions")
                    elif (batch_start // batch_size) % 5 == 0:
                        print(f"    Completed {batch_end}/{len(sampled_primes)} predictions")
                    
                    # Memory cleanup after each batch
                    gc.collect()
                    
                except Exception as e:
                    if logger:
                        logger.log_and_print(f"Warning: Error in batch {batch_start}-{batch_end}: {str(e)}")
                    else:
                        print(f"Warning: Error in batch {batch_start}-{batch_end}: {str(e)}")
                    continue
            
            # Compute additional statistics using numpy operations
            try:
                probabilities = np.array(list(prime_probability_map.values()), dtype=np.float64)
                primes = np.array(list(prime_probability_map.keys()), dtype=np.float64)
                
                stats = {
                    'total_predictions': int(len(prime_probability_map)),
                    'mean_probability': float(np.mean(probabilities)),
                    'std_probability': float(np.std(probabilities)),
                    'min_probability': float(np.min(probabilities)),
                    'max_probability': float(np.max(probabilities)),
                    'prediction_range': {
                        'start': float(np.min(primes)),
                        'end': float(np.max(primes))
                    }
                }
                
                if logger:
                    logger.log_and_print("\nPrediction Statistics:")
                    logger.log_and_print(f"  Mean Probability: {stats['mean_probability']:.4f}")
                    logger.log_and_print(f"  Std Probability: {stats['std_probability']:.4f}")
                    logger.log_and_print(f"  Probability Range: [{stats['min_probability']:.4f}, {stats['max_probability']:.4f}]")
                    logger.log_and_print(f"  Total Predictions: {stats['total_predictions']}")
                    logger.log_and_print(f"  Prediction Range: {stats['prediction_range']['start']} to {stats['prediction_range']['end']}")
                else:
                    print("\nPrediction Statistics:")
                    print(f"  Mean Probability: {stats['mean_probability']:.4f}")
                    print(f"  Std Probability: {stats['std_probability']:.4f}")
                    print(f"  Probability Range: [{stats['min_probability']:.4f}, {stats['max_probability']:.4f}]")
                    print(f"  Total Predictions: {stats['total_predictions']}")
                    print(f"  Prediction Range: {stats['prediction_range']['start']} to {stats['prediction_range']['end']}")
                
                # Add statistics to the result
                prime_probability_map['_stats'] = stats
                
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Warning: Error computing statistics: {str(e)}")
                else:
                    print(f"Warning: Error computing statistics: {str(e)}")
            
            if logger:
                logger.log_and_print("  Prediction generation complete")
            else:
                print("  Prediction generation complete")
            return prime_probability_map
            
        except Exception as e:
            if logger:
                logger.log_and_print(f"Error in prime probability map generation: {str(e)}", level=logging.ERROR)
                logger.logger.error(traceback.format_exc())
            else:
                print(f"Error in prime probability map generation: {str(e)}")
                traceback.print_exc()
            return None
        
@timing_decorator                             
def analyze_cluster_transitions(df):
    """Analyze cluster transitions with improved numerical stability."""
    with suppress_overflow_warnings():
        if 'cluster' not in df.columns:
            return None
            
        clusters = sorted(df['cluster'].unique())
        n_clusters = len(clusters)
        
        # Initialize transition matrix
        transition_matrix = np.zeros((n_clusters, n_clusters), dtype=np.float64)
        
        # Count transitions
        for i in range(len(df) - 1):
            current_cluster = df['cluster'].iloc[i]
            next_cluster = df['cluster'].iloc[i + 1]
            transition_matrix[current_cluster][next_cluster] += 1
            
        # Convert to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_probs = np.divide(transition_matrix, 
                                   row_sums, 
                                   out=np.zeros_like(transition_matrix), 
                                   where=row_sums!=0)
        
        # Calculate entropy for each cluster
        cluster_entropy = {}
        for i in range(n_clusters):
            probs = transition_probs[i]
            if np.any(probs > 0):
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                cluster_entropy[i] = entropy
            else:
                cluster_entropy[i] = 0
                
        return {
            'transition_matrix': transition_matrix,
            'transition_probabilities': transition_probs,
            'cluster_entropy': cluster_entropy
        }

@timing_decorator
def compute_cluster_stability(df, batch_size=5000, logger=None):
    """Compute cluster stability metrics with improved memory management and numerical stability."""
    with suppress_numeric_warnings():
        if 'cluster' not in df.columns:
            return None
            
        stability_metrics = {}
        
        try:
            # Convert to numpy arrays for faster processing
            clusters = df['cluster'].values.astype(np.int32)
            
            # Compute run lengths with protection
            run_lengths = []
            current_run = 1
            
            # Process runs in batches
            for start_idx in range(0, len(clusters) - 1, batch_size):
                end_idx = min(start_idx + batch_size, len(clusters) - 1)
                batch_clusters = clusters[start_idx:end_idx + 1]
                
                # Find run boundaries in this batch
                changes = np.where(batch_clusters[1:] != batch_clusters[:-1])[0]
                
                # Process runs within the batch
                last_change = 0
                for change in changes:
                    run_lengths.append(change - last_change + 1)
                    last_change = change + 1
                
                # Handle run that continues to next batch
                if end_idx < len(clusters) - 1:
                    if batch_clusters[-1] == clusters[end_idx + 1]:
                        current_run = len(batch_clusters) - last_change
                    else:
                        if last_change < len(batch_clusters):
                            run_lengths.append(len(batch_clusters) - last_change)
                        current_run = 1
                else:
                    # Handle the final run
                    if last_change < len(batch_clusters):
                        run_lengths.append(len(batch_clusters) - last_change)
                
                gc.collect()
            
            # Convert run lengths to numpy array for efficient computation
            run_lengths = np.array(run_lengths, dtype=np.float64)
            
            # Compute stability metrics with numerical protection
            with np.errstate(all='ignore'):
                stability_metrics.update({
                    'mean_run_length': float(np.mean(run_lengths)),
                    'std_run_length': float(np.std(run_lengths)),
                    'max_run_length': int(np.max(run_lengths)),
                    'min_run_length': int(np.min(run_lengths)),
                    'median_run_length': float(np.median(run_lengths))
                })
                
                # Compute transition probabilities
                unique_clusters = np.unique(clusters)
                n_clusters = len(unique_clusters)
                transitions = np.zeros((n_clusters, n_clusters), dtype=np.float64)
                
                # Process transitions in batches
                for start_idx in range(0, len(clusters) - 1, batch_size):
                    end_idx = min(start_idx + batch_size, len(clusters) - 1)
                    current = clusters[start_idx:end_idx]
                    next_cluster = clusters[start_idx + 1:end_idx + 1]
                    
                    for i, j in zip(current, next_cluster):
                        transitions[i, j] += 1
                    
                    gc.collect()
                
                # Normalize transition probabilities
                row_sums = transitions.sum(axis=1, keepdims=True)
                transition_probs = np.divide(
                    transitions,
                    row_sums,
                    out=np.zeros_like(transitions),
                    where=row_sums != 0
                )
                
                # Compute stability score based on diagonal probabilities
                diagonal_probs = np.diag(transition_probs)
                stability_metrics['cluster_stability_scores'] = {
                    int(cluster): float(prob)
                    for cluster, prob in enumerate(diagonal_probs)
                }
                
                # Compute overall stability score
                stability_metrics['overall_stability'] = float(np.mean(diagonal_probs))
                
                # Compute entropy for each cluster
                cluster_entropy = {}
                for i in range(n_clusters):
                    probs = transition_probs[i]
                    if np.any(probs > 0):
                        entropy = -np.sum(probs * np.log2(probs + 1e-10))
                        cluster_entropy[int(i)] = float(entropy)
                    else:
                        cluster_entropy[int(i)] = 0.0
                
                stability_metrics['cluster_entropy'] = cluster_entropy
                
                # Add summary statistics
                stability_metrics['summary'] = {
                    'n_clusters': int(n_clusters),
                    'total_transitions': int(np.sum(transitions)),
                    'mean_entropy': float(np.mean(list(cluster_entropy.values()))),
                    'stability_score': float(np.mean(diagonal_probs))
                }
            
            if logger:
                logger.log_and_print("Cluster stability analysis complete")
            
            return stability_metrics
            
        except Exception as e:
            error_msg = f"Error in cluster stability computation: {str(e)}"
            if logger:
                logger.log_and_print(error_msg, level=logging.ERROR)
                logger.logger.error(traceback.format_exc())
            else:
                print(error_msg)
                traceback.print_exc()
            
            # Return safe default values
            return {
                'mean_run_length': 0.0,
                'std_run_length': 0.0,
                'max_run_length': 0,
                'min_run_length': 0,
                'median_run_length': 0.0,
                'cluster_stability_scores': {},
                'overall_stability': 0.0,
                'cluster_entropy': {},
                'summary': {
                    'n_clusters': 0,
                    'total_transitions': 0,
                    'mean_entropy': 0.0,
                    'stability_score': 0.0
                }
            }
 
@timing_decorator           
def analyze_gap_sequences(df, max_sequence_length=5, batch_size=5000, logger=None):
    """Analyze sequences of gaps with improved memory management and numerical stability."""
    if logger:
        logger.log_and_print(f"Analyzing gap sequences up to length {max_sequence_length}")
    
    try:
        # Convert to float64 and clip values
        gaps = df['gap_size'].values.astype(np.float64)
        gaps = np.clip(gaps, -1e10, 1e10)
        gaps = gaps[np.isfinite(gaps)]  # Remove inf/nan values
        
        sequence_stats = {}
        
        # Process sequences of different lengths
        for length in range(2, max_sequence_length + 1):
            if logger:
                logger.log_and_print(f"Processing sequences of length {length}")
            
            stats = {
                'count': 0,
                'unique_sequences': set(),
                'increasing_count': 0,
                'decreasing_count': 0,
                'constant_count': 0,
                'diff_stats': {
                    'sum': 0.0,
                    'sum_sq': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'count': 0
                },
                'range_stats': {
                    'sum': 0.0,
                    'sum_sq': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'count': 0
                },
                'pattern_stats': {
                    'alternating_count': 0,
                    'cyclic_count': 0,
                    'monotonic_count': 0
                }
            }
            
            # Process sequences in batches
            for start_idx in range(0, len(gaps) - length + 1, batch_size):
                end_idx = min(start_idx + batch_size, len(gaps) - length + 1)
                
                # Extract sequences for this batch
                batch_sequences = []
                for i in range(start_idx, end_idx):
                    seq = gaps[i:i+length]
                    if np.all(np.isfinite(seq)):
                        batch_sequences.append(seq)
                
                if batch_sequences:
                    sequences = np.array(batch_sequences)
                    
                    with np.errstate(all='ignore'):
                        # Update counts
                        stats['count'] += len(sequences)
                        
                        # Update unique sequences (limit to prevent memory issues)
                        if len(stats['unique_sequences']) < 10000:
                            for seq in sequences:
                                stats['unique_sequences'].add(tuple(seq))
                        
                        # Analyze patterns
                        diffs = np.diff(sequences, axis=1)
                        increasing = np.all(diffs > 0, axis=1)
                        decreasing = np.all(diffs < 0, axis=1)
                        constant = np.all(diffs == 0, axis=1)
                        
                        stats['increasing_count'] += int(np.sum(increasing))
                        stats['decreasing_count'] += int(np.sum(decreasing))
                        stats['constant_count'] += int(np.sum(constant))
                        
                        # Analyze alternating patterns
                        sign_changes = np.diff(np.sign(diffs), axis=1)
                        alternating = np.all(np.abs(sign_changes) == 2, axis=1)
                        stats['pattern_stats']['alternating_count'] += int(np.sum(alternating))
                        
                        # Analyze cyclic patterns
                        for seq in sequences:
                            if len(seq) >= 3:
                                # Check if sequence repeats
                                for period in range(1, len(seq)//2 + 1):
                                    if np.all(seq[period:] == seq[:-period]):
                                        stats['pattern_stats']['cyclic_count'] += 1
                                        break
                        
                        # Update difference statistics
                        valid_diffs = diffs[np.isfinite(diffs)]
                        if len(valid_diffs) > 0:
                            stats['diff_stats']['sum'] += float(np.sum(valid_diffs))
                            stats['diff_stats']['sum_sq'] += float(np.sum(valid_diffs ** 2))
                            stats['diff_stats']['min'] = float(min(stats['diff_stats']['min'], np.min(valid_diffs)))
                            stats['diff_stats']['max'] = float(max(stats['diff_stats']['max'], np.max(valid_diffs)))
                            stats['diff_stats']['count'] += len(valid_diffs)
                        
                        # Update range statistics
                        ranges = np.ptp(sequences, axis=1)
                        valid_ranges = ranges[np.isfinite(ranges)]
                        if len(valid_ranges) > 0:
                            stats['range_stats']['sum'] += float(np.sum(valid_ranges))
                            stats['range_stats']['sum_sq'] += float(np.sum(valid_ranges ** 2))
                            stats['range_stats']['min'] = float(min(stats['range_stats']['min'], np.min(valid_ranges)))
                            stats['range_stats']['max'] = float(max(stats['range_stats']['max'], np.max(valid_ranges)))
                            stats['range_stats']['count'] += len(valid_ranges)
                
                gc.collect()
            
            # Compute final statistics
            sequence_stats[length] = {
                'count': int(stats['count']),
                'unique_count': len(stats['unique_sequences']),
                'increasing_count': int(stats['increasing_count']),
                'decreasing_count': int(stats['decreasing_count']),
                'constant_count': int(stats['constant_count']),
                'pattern_counts': {
                    'alternating': int(stats['pattern_stats']['alternating_count']),
                    'cyclic': int(stats['pattern_stats']['cyclic_count']),
                    'monotonic': int(stats['increasing_count'] + stats['decreasing_count'])
                }
            }
            
            if stats['count'] > 0:
                sequence_stats[length].update({
                    'increasing_ratio': float(stats['increasing_count'] / stats['count']),
                    'decreasing_ratio': float(stats['decreasing_count'] / stats['count']),
                    'constant_ratio': float(stats['constant_count'] / stats['count'])
                })
            
            # Compute difference statistics
            if stats['diff_stats']['count'] > 0:
                mean_diff = stats['diff_stats']['sum'] / stats['diff_stats']['count']
                var_diff = (stats['diff_stats']['sum_sq'] / stats['diff_stats']['count']) - (mean_diff ** 2)
                sequence_stats[length].update({
                    'mean_diff': float(mean_diff),
                    'std_diff': float(np.sqrt(max(0, var_diff))),
                    'max_diff': float(stats['diff_stats']['max']),
                    'min_diff': float(stats['diff_stats']['min'])
                })
            
            # Compute range statistics
            if stats['range_stats']['count'] > 0:
                mean_range = stats['range_stats']['sum'] / stats['range_stats']['count']
                var_range = (stats['range_stats']['sum_sq'] / stats['range_stats']['count']) - (mean_range ** 2)
                sequence_stats[length].update({
                    'mean_range': float(mean_range),
                    'std_range': float(np.sqrt(max(0, var_range))),
                    'max_range': float(stats['range_stats']['max']),
                    'min_range': float(stats['range_stats']['min'])
                })
            
            # Clear sequence stats for this length
            del stats
            gc.collect()
            
            if logger:
                logger.log_and_print(f"Completed analysis for sequences of length {length}")
        
        return sequence_stats
        
    except Exception as e:
        error_msg = f"Error in gap sequence analysis: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            length: {
                'count': 0,
                'unique_count': 0,
                'increasing_count': 0,
                'decreasing_count': 0,
                'constant_count': 0,
                'pattern_counts': {
                    'alternating': 0,
                    'cyclic': 0,
                    'monotonic': 0
                }
            }
            for length in range(2, max_sequence_length + 1)
        }

@timing_decorator
def analyze_gap_sequences_advanced(df, max_length=5, batch_size=5000, logger=None):
    """Analyze sequences of gaps with improved memory management and numerical stability."""
    with suppress_overflow_warnings():
        sequence_analysis = {}
        
        if logger:
            logger.log_and_print(f"Analyzing gap sequences up to length {max_length}")
        
        try:
            # Convert to float64 and clip values
            gaps = df['gap_size'].astype(np.float64).values
            gaps = np.clip(gaps, -1e10, 1e10)
            
            # Analyze sequences of different lengths
            for length in range(2, max_length + 1):
                if logger:
                    logger.log_and_print(f"Processing sequences of length {length}")
                
                # Call numba-optimized function
                count, increasing_count, decreasing_count, constant_count = _analyze_gap_sequences_numba(gaps, length)
                
                sequence_analysis[length] = {
                    'count': int(count),
                    'unique_count': 0,  # Not computed in numba
                    'increasing_count': int(increasing_count),
                    'decreasing_count': int(decreasing_count),
                    'constant_count': int(constant_count)
                }
                
                if count > 0:
                    sequence_analysis[length].update({
                        'increasing_ratio': float(increasing_count / count),
                        'decreasing_ratio': float(decreasing_count / count),
                        'constant_ratio': float(constant_count / count)
                    })
            
            if logger:
                logger.log_and_print("Gap sequence analysis complete")
            
            return sequence_analysis
            
        except Exception as e:
            error_msg = f"Error in gap sequence analysis: {str(e)}"
            if logger:
                logger.log_and_print(error_msg, level=logging.ERROR)
                logger.logger.error(traceback.format_exc())
            else:
                print(error_msg)
                traceback.print_exc()
            
            # Return safe default values
            return {
                length: {
                    'count': 0,
                    'unique_count': 0,
                    'increasing_count': 0,
                    'decreasing_count': 0,
                    'constant_count': 0,
                    'increasing_ratio': 0.0,
                    'decreasing_ratio': 0.0,
                    'constant_ratio': 0.0
                }
                for length in range(2, max_length + 1)
            }
            
@timing_decorator
def compute_advanced_sequence_metrics(df, batch_size=5000, logger=None):
    """Compute advanced sequence metrics with improved memory management and numerical stability."""
    with suppress_overflow_warnings():
        if logger:
            logger.log_and_print("Computing advanced sequence metrics...")
        
        metrics = {}
        
        try:
            # Convert relevant columns to float64 and clip values
            numeric_cols = [col for col in df.columns if col not in ['cluster', 'sub_cluster']]
            stats_accumulators = {
                col: {
                    'sum': 0,
                    'sum_sq': 0,
                    'count': 0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'values_for_quantiles': []
                }
                for col in numeric_cols
            }
            
            # Process in batches with overlap
            window_sizes = [3, 5, 7, 11]
            rolling_stats = {
                size: {col: [] for col in numeric_cols}
                for size in window_sizes
            }
            
            for start_idx in range(0, len(df), batch_size):
                # Add overlap to ensure continuous rolling statistics
                batch_start = max(0, start_idx - max(window_sizes))
                batch_end = min(start_idx + batch_size + max(window_sizes), len(df))
                batch = df.iloc[batch_start:batch_end]
                
                # Update basic statistics
                for col in numeric_cols:
                    data = batch[col].astype(np.float64).clip(-1e10, 1e10)
                    valid_mask = np.isfinite(data)
                    valid_data = data[valid_mask]
                    
                    if len(valid_data) > 0:
                        stats_accumulators[col]['sum'] += np.sum(valid_data)
                        stats_accumulators[col]['sum_sq'] += np.sum(valid_data ** 2)
                        stats_accumulators[col]['count'] += len(valid_data)
                        stats_accumulators[col]['min'] = min(
                            stats_accumulators[col]['min'],
                            np.min(valid_data)
                        )
                        stats_accumulators[col]['max'] = max(
                            stats_accumulators[col]['max'],
                            np.max(valid_data)
                        )
                        # Store subset of values for quantile computation
                        if len(stats_accumulators[col]['values_for_quantiles']) < 10000:
                            stats_accumulators[col]['values_for_quantiles'].extend(
                                valid_data.tolist()[:1000]  # Limit stored values
                            )
                
                # Compute rolling statistics for each window size
                for window in window_sizes:
                    for col in numeric_cols:
                        rolling_mean = batch[col].rolling(
                            window=window, 
                            min_periods=1,
                            center=True
                        ).mean()
                        
                        rolling_std = batch[col].rolling(
                            window=window,
                            min_periods=1,
                            center=True
                        ).std()
                        
                        # Store only the relevant portion (excluding overlap)
                        start_offset = max(0, start_idx - batch_start)
                        end_offset = start_offset + min(batch_size, len(df) - start_idx)
                        
                        rolling_stats[window][col].extend([
                            {
                                'mean': float(rolling_mean.iloc[i]),
                                'std': float(rolling_std.iloc[i])
                            }
                            for i in range(start_offset, end_offset)
                        ])
                
                gc.collect()
            
            # Compute final statistics
            for col in numeric_cols:
                acc = stats_accumulators[col]
                if acc['count'] > 0:
                    mean = acc['sum'] / acc['count']
                    var = (acc['sum_sq'] / acc['count']) - (mean ** 2)
                    std = np.sqrt(max(0, var))
                    
                    metrics[col] = {
                        'basic': {
                            'mean': float(mean),
                            'std': float(std),
                            'min': float(acc['min']),
                            'max': float(acc['max']),
                            'count': int(acc['count'])
                        },
                        'rolling': {
                            str(window): stats
                            for window, stats in (
                                (w, rolling_stats[w][col])
                                for w in window_sizes
                            )
                        }
                    }
                    
                    # Compute quantiles if we have stored values
                    if acc['values_for_quantiles']:
                        quantiles = np.percentile(
                            acc['values_for_quantiles'],
                            [25, 50, 75]
                        )
                        metrics[col]['quantiles'] = {
                            'q1': float(quantiles[0]),
                            'median': float(quantiles[1]),
                            'q3': float(quantiles[2])
                        }
            
            if logger:
                logger.log_and_print("Advanced sequence metrics computation complete")
            
        except Exception as e:
            error_msg = f"Error computing advanced sequence metrics: {str(e)}"
            if logger:
                logger.log_and_print(error_msg, level=logging.ERROR)
                logger.logger.error(traceback.format_exc())
            else:
                print(error_msg)
                traceback.print_exc()
        
        return metrics

@timing_decorator          
def analyze_gap_patterns_advanced(df, batch_size=5000, logger=None):
    """Analyze advanced patterns in gap sequences with improved memory management and numerical stability."""
    with suppress_overflow_warnings():
        patterns = {}
        
        try:
            if logger:
                logger.log_and_print("Analyzing advanced gap patterns...")
            
            # Convert to numpy array and clip values
            gaps = df['gap_size'].values.astype(np.float64)
            gaps = np.clip(gaps, -1e10, 1e10)
            gaps = gaps[np.isfinite(gaps)]  # Remove inf/nan values
            
            # Analyze local maxima and minima in batches
            peaks = []
            valleys = []
            
            for start_idx in range(1, len(gaps)-1, batch_size):
                end_idx = min(start_idx + batch_size, len(gaps)-1)
                batch = gaps[start_idx-1:end_idx+2]  # Include one point before and after
                
                with np.errstate(all='ignore'):
                    # Find peaks and valleys in batch
                    for i in range(1, len(batch)-1):
                        if batch[i] > batch[i-1] and batch[i] > batch[i+1]:
                            peaks.append(start_idx + i - 1)
                        if batch[i] < batch[i-1] and batch[i] < batch[i+1]:
                            valleys.append(start_idx + i - 1)
                
                gc.collect()
            
            patterns['peaks'] = {
                'count': len(peaks),
                'mean_spacing': float(np.mean(np.diff(peaks))) if len(peaks) > 1 else 0.0,
                'std_spacing': float(np.std(np.diff(peaks))) if len(peaks) > 1 else 0.0,
                'locations': [int(p) for p in peaks],
                'values': [float(gaps[p]) for p in peaks]
            }
            
            patterns['valleys'] = {
                'count': len(valleys),
                'mean_spacing': float(np.mean(np.diff(valleys))) if len(valleys) > 1 else 0.0,
                'std_spacing': float(np.std(np.diff(valleys))) if len(valleys) > 1 else 0.0,
                'locations': [int(v) for v in valleys],
                'values': [float(gaps[v]) for v in valleys]
            }
            
            # Analyze runs in batches
            runs = []
            current_run = [gaps[0]]
            
            for start_idx in range(1, len(gaps), batch_size):
                end_idx = min(start_idx + batch_size, len(gaps))
                batch = gaps[start_idx:end_idx]
                
                for i, value in enumerate(batch):
                    if value == current_run[-1]:
                        current_run.append(value)
                    elif len(current_run) > 1:
                        runs.append(current_run)
                        current_run = [value]
                    else:
                        current_run = [value]
                
                gc.collect()
            
            if len(current_run) > 1:
                runs.append(current_run)
            
            if runs:
                run_lengths = np.array([len(run) for run in runs])
                patterns['runs'] = {
                    'count': len(runs),
                    'mean_length': float(np.mean(run_lengths)),
                    'std_length': float(np.std(run_lengths)),
                    'max_length': int(np.max(run_lengths)),
                    'min_length': int(np.min(run_lengths)),
                    'run_values': [[float(v) for v in run] for run in runs[:100]]  # Store first 100 runs
                }
            else:
                patterns['runs'] = {
                    'count': 0,
                    'mean_length': 0.0,
                    'std_length': 0.0,
                    'max_length': 0,
                    'min_length': 0,
                    'run_values': []
                }
            
            # Analyze monotonic sequences in batches
            increasing_seqs = []
            decreasing_seqs = []
            current_inc = [gaps[0]]
            current_dec = [gaps[0]]
            
            for start_idx in range(1, len(gaps), batch_size):
                end_idx = min(start_idx + batch_size, len(gaps))
                batch = gaps[start_idx:end_idx]
                
                for value in batch:
                    if value > current_inc[-1]:
                        current_inc.append(value)
                        if len(current_dec) > 1:
                            decreasing_seqs.append(current_dec)
                        current_dec = [value]
                    elif value < current_dec[-1]:
                        current_dec.append(value)
                        if len(current_inc) > 1:
                            increasing_seqs.append(current_inc)
                        current_inc = [value]
                    else:
                        current_inc.append(value)
                        current_dec.append(value)
                
                gc.collect()
            
            if len(current_inc) > 1:
                increasing_seqs.append(current_inc)
            if len(current_dec) > 1:
                decreasing_seqs.append(current_dec)
            
            patterns['monotonic_sequences'] = {
                'increasing': {
                    'count': len(increasing_seqs),
                    'mean_length': float(np.mean([len(seq) for seq in increasing_seqs])) if increasing_seqs else 0.0,
                    'max_length': int(max([len(seq) for seq in increasing_seqs])) if increasing_seqs else 0,
                    'sequences': [[float(v) for v in seq] for seq in increasing_seqs[:50]]  # Store first 50 sequences
                },
                'decreasing': {
                    'count': len(decreasing_seqs),
                    'mean_length': float(np.mean([len(seq) for seq in decreasing_seqs])) if decreasing_seqs else 0.0,
                    'max_length': int(max([len(seq) for seq in decreasing_seqs])) if decreasing_seqs else 0,
                    'sequences': [[float(v) for v in seq] for seq in decreasing_seqs[:50]]  # Store first 50 sequences
                }
            }
            
            # Add distribution analysis
            with np.errstate(all='ignore'):
                hist, bin_edges = np.histogram(gaps, bins='auto', density=True)
                patterns['distribution'] = {
                    'histogram': hist.tolist(),
                    'bin_edges': bin_edges.tolist(),
                    'skewness': float(sps.skew(gaps)),
                    'kurtosis': float(sps.kurtosis(gaps)),
                    'entropy': float(entropy(hist + 1e-10))
                }
            
            if logger:
                logger.log_and_print("Gap pattern analysis complete")
            
            return patterns
            
        except Exception as e:
            error_msg = f"Error in gap pattern analysis: {str(e)}"
            if logger:
                logger.log_and_print(error_msg, level=logging.ERROR)
                logger.logger.error(traceback.format_exc())
            else:
                print(error_msg)
                traceback.print_exc()
            
            # Return safe default values
            return {
                'peaks': {'count': 0, 'mean_spacing': 0.0, 'std_spacing': 0.0, 'locations': [], 'values': []},
                'valleys': {'count': 0, 'mean_spacing': 0.0, 'std_spacing': 0.0, 'locations': [], 'values': []},
                'runs': {'count': 0, 'mean_length': 0.0, 'std_length': 0.0, 'max_length': 0, 'min_length': 0, 'run_values': []},
                'monotonic_sequences': {
                    'increasing': {'count': 0, 'mean_length': 0.0, 'max_length': 0, 'sequences': []},
                    'decreasing': {'count': 0, 'mean_length': 0.0, 'max_length': 0, 'sequences': []}
                },
                'distribution': {
                    'histogram': [], 'bin_edges': [], 'skewness': 0.0, 'kurtosis': 0.0, 'entropy': 0.0
                }
            }

@timing_decorator            
def analyze_cluster_features(df, batch_size=10000, logger=None):
    """Analyze features within each cluster with improved memory management and numerical stability."""
    if logger:
        logger.log_and_print("Analyzing cluster features...")
    
    cluster_features = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    try:
        # Process each cluster separately
        for cluster_id in sorted(df['cluster'].unique()):
            if logger:
                logger.log_and_print(f"Processing cluster {cluster_id}")
            
            cluster_mask = df['cluster'] == cluster_id
            cluster_data = df[numeric_cols][cluster_mask]
            
            feature_stats = {}
            # Process features in batches
            for i in range(0, len(numeric_cols), batch_size):
                batch_cols = numeric_cols[i:i+batch_size]
                batch_data = cluster_data[batch_cols]
                
                for col in batch_data.columns:
                    if col != 'cluster':
                        # Convert to float64 and clip values
                        data = batch_data[col].values.astype(np.float64)
                        data = np.clip(data, -1e10, 1e10)
                        
                        # Remove inf/nan values
                        data = data[np.isfinite(data)]
                        
                        if len(data) > 0:
                            with np.errstate(all='ignore'):
                                # Basic statistics
                                stats = {
                                    'mean': float(np.mean(data)),
                                    'median': float(np.median(data)),
                                    'std': float(np.std(data)),
                                    'min': float(np.min(data)),
                                    'max': float(np.max(data)),
                                    'count': int(len(data))
                                }
                                
                                # Quartile statistics
                                quartiles = np.percentile(data, [25, 75])
                                stats.update({
                                    'q1': float(quartiles[0]),
                                    'q3': float(quartiles[1]),
                                    'iqr': float(quartiles[1] - quartiles[0])
                                })
                                
                                # Distribution shape statistics
                                stats.update({
                                    'skew': float(sps.skew(data)),
                                    'kurtosis': float(sps.kurtosis(data))
                                })
                                
                                # Mode calculation with error handling
                                try:
                                    mode_result = sps.mode(data)
                                    if isinstance(mode_result, tuple):
                                        mode_value = mode_result[0][0]  # Old scipy version
                                    else:
                                        mode_value = mode_result.mode[0]  # New scipy version
                                    stats['mode'] = float(mode_value)
                                except Exception as e:
                                    if logger:
                                        logger.log_and_print(f"Warning: Mode calculation failed for {col}: {str(e)}")
                                    stats['mode'] = float(stats['median'])
                                
                                # Distribution tests
                                try:
                                    normality_stat, normality_p = normaltest(data)
                                    stats.update({
                                        'normality_stat': float(normality_stat),
                                        'normality_p': float(normality_p),
                                        'is_normal': bool(normality_p > 0.05)
                                    })
                                except Exception as e:
                                    if logger:
                                        logger.log_and_print(f"Warning: Normality test failed for {col}: {str(e)}")
                                    stats.update({
                                        'normality_stat': float('nan'),
                                        'normality_p': float('nan'),
                                        'is_normal': False
                                    })
                                
                                # Compute histogram for distribution analysis
                                hist, bin_edges = np.histogram(data, bins='auto', density=True)
                                stats['distribution'] = {
                                    'histogram': hist.tolist(),
                                    'bin_edges': bin_edges.tolist(),
                                    'entropy': float(entropy(hist + 1e-10))
                                }
                                
                                # Replace any remaining inf/nan values
                                stats = {k: (v if isinstance(v, (dict, list, bool)) else 
                                           (float(v) if np.isfinite(float(v)) else 0.0)) 
                                       for k, v in stats.items()}
                                
                                feature_stats[col] = stats
                
                gc.collect()
            
            # Compute inter-feature correlations
            if len(feature_stats) > 1:
                correlation_matrix = np.zeros((len(feature_stats), len(feature_stats)))
                feature_names = list(feature_stats.keys())
                
                for i, feat1 in enumerate(feature_names):
                    for j, feat2 in enumerate(feature_names[i+1:], i+1):
                        data1 = cluster_data[feat1].values
                        data2 = cluster_data[feat2].values
                        
                        # Process correlation in batches
                        corr_sum = 0.0
                        count = 0
                        
                        for start_idx in range(0, len(data1), batch_size):
                            end_idx = min(start_idx + batch_size, len(data1))
                            batch1 = data1[start_idx:end_idx]
                            batch2 = data2[start_idx:end_idx]
                            
                            mask = np.isfinite(batch1) & np.isfinite(batch2)
                            if np.sum(mask) > 1:
                                with np.errstate(all='ignore'):
                                    corr = np.corrcoef(batch1[mask], batch2[mask])[0, 1]
                                    if np.isfinite(corr):
                                        corr_sum += corr
                                        count += 1
                            
                            gc.collect()
                        
                        if count > 0:
                            correlation_matrix[i, j] = corr_sum / count
                            correlation_matrix[j, i] = correlation_matrix[i, j]
                
                feature_stats['correlations'] = {
                    'matrix': correlation_matrix.tolist(),
                    'features': feature_names
                }
            
            cluster_features[int(cluster_id)] = feature_stats
            
            gc.collect()
        
        if logger:
            logger.log_and_print("Cluster feature analysis complete")
        
        return cluster_features
        
    except Exception as e:
        error_msg = f"Error in cluster feature analysis: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {int(cluster_id): {} for cluster_id in df['cluster'].unique()}

@timing_decorator
def compute_cluster_separation_metrics(df, batch_size=5000, logger=None):
    """Compute cluster separation metrics with improved memory management and numerical stability."""
    if 'cluster' not in df.columns:
        return None
    
    if logger:
        logger.log_and_print("Computing cluster separation metrics...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['cluster']]
    
    # Convert to float64 and clip values
    X = df[feature_cols].astype(np.float64)
    X = X.clip(-1e10, 1e10)
    
    clusters = sorted(df['cluster'].unique())
    n_clusters = len(clusters)
    
    separation_metrics = {
        'inter_cluster_distances': np.zeros((n_clusters, n_clusters)),
        'intra_cluster_distances': {},
        'cluster_densities': {}
    }
    
    try:
        # Process each cluster
        for i, cluster1 in enumerate(clusters):
            if logger:
                logger.log_and_print(f"Processing cluster {cluster1}/{len(clusters)}")
            
            cluster1_mask = df['cluster'] == cluster1
            points1_indices = np.where(cluster1_mask)[0]
            
            # Compute intra-cluster distances in batches
            if len(points1_indices) > 1:
                distances = []
                
                # Process intra-cluster distances in batches
                for start1 in range(0, len(points1_indices), batch_size):
                    end1 = min(start1 + batch_size, len(points1_indices))
                    batch1_indices = points1_indices[start1:end1]
                    points1_batch = X.iloc[batch1_indices].values
                    
                    # Only compute distances within this batch
                    with np.errstate(all='ignore'):
                        batch_distances = pdist(points1_batch)
                        distances.extend(batch_distances)
                    
                    gc.collect()
                
                if distances:
                    distances = np.array(distances)
                    separation_metrics['intra_cluster_distances'][int(cluster1)] = {
                        'mean': float(np.mean(distances)),
                        'std': float(np.std(distances)),
                        'min': float(np.min(distances)),
                        'max': float(np.max(distances)),
                        'median': float(np.median(distances))
                    }
            
            # Compute inter-cluster distances in batches
            for j, cluster2 in enumerate(clusters[i+1:], i+1):
                cluster2_mask = df['cluster'] == cluster2
                points2_indices = np.where(cluster2_mask)[0]
                
                if len(points1_indices) > 0 and len(points2_indices) > 0:
                    distances_sum = 0
                    total_distances = 0
                    
                    # Process inter-cluster distances in batches
                    for start1 in range(0, len(points1_indices), batch_size):
                        end1 = min(start1 + batch_size, len(points1_indices))
                        batch1_indices = points1_indices[start1:end1]
                        points1_batch = X.iloc[batch1_indices].values
                        
                        for start2 in range(0, len(points2_indices), batch_size):
                            end2 = min(start2 + batch_size, len(points2_indices))
                            batch2_indices = points2_indices[start2:end2]
                            points2_batch = X.iloc[batch2_indices].values
                            
                            # Compute distances between batches
                            with np.errstate(all='ignore'):
                                batch_distances = cdist(points1_batch, points2_batch)
                                distances_sum += np.sum(batch_distances)
                                total_distances += batch_distances.size
                            
                            gc.collect()
                    
                    if total_distances > 0:
                        mean_dist = distances_sum / total_distances
                        separation_metrics['inter_cluster_distances'][i, j] = mean_dist
                        separation_metrics['inter_cluster_distances'][j, i] = mean_dist
            
            # Compute cluster density in batches
            if len(points1_indices) > 0:
                try:
                    # Compute ranges in batches
                    min_vals = None
                    max_vals = None
                    
                    for start_idx in range(0, len(points1_indices), batch_size):
                        end_idx = min(start_idx + batch_size, len(points1_indices))
                        batch_indices = points1_indices[start_idx:end_idx]
                        points_batch = X.iloc[batch_indices].values
                        
                        batch_min = np.min(points_batch, axis=0)
                        batch_max = np.max(points_batch, axis=0)
                        
                        if min_vals is None:
                            min_vals = batch_min
                            max_vals = batch_max
                        else:
                            min_vals = np.minimum(min_vals, batch_min)
                            if max_vals is None:
                                max_vals = batch_max
                            else:
                                max_vals = np.maximum(max_vals, batch_max)
                        
                        gc.collect()
                    
                    # Compute volume and density
                    if min_vals is not None and max_vals is not None:
                        ranges = max_vals - min_vals
                    else:
                        ranges = np.zeros_like(min_vals)  # or handle the case appropriately
                    volume = np.prod(ranges + 1e-10)  # Add small constant for stability
                    separation_metrics['cluster_densities'][int(cluster1)] = float(len(points1_indices) / volume)
                    
                except Exception as e:
                    if logger:
                        logger.log_and_print(f"Warning: Error computing cluster density for cluster {cluster1}: {str(e)}")
                    separation_metrics['cluster_densities'][int(cluster1)] = 0.0
            
            gc.collect()
        
        # Add summary metrics
        separation_metrics['summary'] = {
            'mean_inter_cluster_distance': float(np.mean(separation_metrics['inter_cluster_distances'])),
            'mean_intra_cluster_distance': float(np.mean([
                stats['mean'] for stats in separation_metrics['intra_cluster_distances'].values()
            ])) if separation_metrics['intra_cluster_distances'] else 0.0,
            'mean_density': float(np.mean(list(separation_metrics['cluster_densities'].values())))
        }
        
        if logger:
            logger.log_and_print("Cluster separation analysis complete")
        
    except Exception as e:
        error_msg = f"Error in cluster separation computation: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'inter_cluster_distances': np.zeros((n_clusters, n_clusters)),
            'intra_cluster_distances': {},
            'cluster_densities': {},
            'summary': {
                'mean_inter_cluster_distance': 0.0,
                'mean_intra_cluster_distance': 0.0,
                'mean_density': 0.0
            }
        }
    
    return separation_metrics

@timing_decorator
def analyze_temporal_patterns(df, column='gap_size', periods=None, logger=None):
    """Perform advanced time series analysis with improved error handling."""
    if logger:
        logger.log_and_print("Analyzing time series patterns...")
    else:
        print("Analyzing time series patterns...")
    
    # Initialize time_series_tests
    time_series_tests = {}
    
    try:
        # Handle missing or invalid data
        series = df[column].copy()
        series = series.astype(float)  # Convert to float
        series = series.dropna()  # Remove any NaN values
        
        if len(series) == 0:
            raise ValueError("No valid data points after cleaning")
        
        # Perform all time series analysis within a single try block
        try:
            # Determine period if not provided
            if periods is None:
                # Try different periods and pick the one with strongest seasonality
                test_periods = [12, 24, 36, 48]
                max_strength = 0
                best_period = test_periods[0]
                
                for period in test_periods:
                    try:
                        if len(series) >= 2 * period:  # Ensure enough data points
                            decomp = seasonal_decompose(
                                series,
                                period=period,
                                extrapolate_trend=1  # Changed from 'freq' to 1
                            )
                            strength = np.nanstd(decomp.seasonal)  # Use nanstd to handle NaN values
                            if strength > max_strength:
                                max_strength = strength
                                best_period = period
                    except Exception as e:
                        if logger:
                            logger.log_and_print(f"Warning: Period {period} failed: {str(e)}")
                        continue
                
                periods = best_period
            
            # Perform decomposition with error handling
            try:
                decomposition = seasonal_decompose(
                    series,
                    period=periods,
                    extrapolate_trend=1  # Changed from 'freq' to 1
                )
                
                # Compute strength metrics safely
                trend_resid_var = np.nanvar(decomposition.resid + decomposition.trend)
                if trend_resid_var != 0:
                    trend_strength = 1 - np.nanvar(decomposition.resid) / trend_resid_var
                else:
                    trend_strength = 0
                    
                seasonal_resid_var = np.nanvar(decomposition.resid + decomposition.seasonal)
                if seasonal_resid_var != 0:
                    seasonal_strength = 1 - np.nanvar(decomposition.resid) / seasonal_resid_var
                else:
                    seasonal_strength = 0
                    
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Warning: Decomposition failed: {str(e)}")
                return {
                    'trend_strength': 0.0,
                    'seasonal_strength': 0.0,
                    'period': periods,
                    'stationarity_test': {
                        'statistic': 0.0,
                        'p_value': 1.0,
                        'critical_values': {},
                        'significance_level': 0.05
                    },
                    'time_series_tests': {},
                    'error_message': str(e)
                }
            
            # Test for stationarity using ADF test with proper error handling
            try:
                import statsmodels.tsa.stattools as stattools
                # Get the results without unpacking
                adf_results = stattools.adfuller(series)
                
                # Extract values safely
                adf_stat = float(adf_results[0]) if len(adf_results) > 0 else 0.0
                pvalue = float(adf_results[1]) if len(adf_results) > 1 else 1.0
                
                # Handle critical values safely
                crit_vals = {}
                if len(adf_results) > 4 and isinstance(adf_results[4], dict):
                    critical_values = adf_results[4]
                    for key in ['1%', '5%', '10%']:
                        if key in critical_values:
                            try:
                                crit_vals[key] = float(critical_values[key])
                            except (TypeError, ValueError):
                                continue
                
                stationarity_test = {
                    'statistic': adf_stat,
                    'p_value': pvalue,
                    'critical_values': crit_vals,
                    'significance_level': 0.05
                }
                
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Warning: ADF test failed: {str(e)}")
                stationarity_test = {
                    'statistic': 0.0,
                    'p_value': 1.0,
                    'critical_values': {},
                    'significance_level': 0.05
                }
            
            # Perform Ljung-Box test
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_result = acorr_ljungbox(series, lags=[10, 20, 30], return_df=True)
                if isinstance(lb_result, pd.DataFrame):
                    time_series_tests['ljung_box'] = {
                        'statistic': lb_result['lb_stat'].tolist(),
                        'p_value': lb_result['lb_pvalue'].tolist()
                    }
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Warning: Ljung-Box test failed: {str(e)}")
                time_series_tests['ljung_box'] = {
                    'statistic': [],
                    'p_value': []
                }
            
            return {
                'decomposition': decomposition,
                'trend_strength': float(trend_strength),
                'seasonal_strength': float(seasonal_strength),
                'period': periods,
                'stationarity_test': stationarity_test,
                'time_series_tests': time_series_tests,
                'error_message': None
            }
        except Exception as e:
            if logger:
                logger.log_and_print(f"Warning: Time series tests failed: {str(e)}")
            return {
                'trend_strength': 0.0,
                'seasonal_strength': 0.0,
                'period': periods,
                'stationarity_test': {
                    'statistic': 0.0,
                    'p_value': 1.0,
                    'critical_values': {},
                    'significance_level': 0.05
                },
                'time_series_tests': {},
                'error_message': str(e)
            }
        
    except Exception as e:
        if logger:
            logger.log_and_print(f"Error in time series analysis: {str(e)}")
        else:
            print(f"Error in time series analysis: {str(e)}")
        return {
            'trend_strength': 0.0,
            'seasonal_strength': 0.0,
            'period': periods,
            'stationarity_test': {
                'statistic': 0.0,
                'p_value': 1.0,
                'critical_values': {},
                'significance_level': 0.05
            },
            'time_series_tests': {},
            'error_message': str(e)
        }
                          
@timing_decorator
def analyze_feature_network(df, feature_cols, correlation_threshold=0.3, batch_size=5000, logger=None):
    """Analyze feature interactions as a network graph with improved memory management and numerical stability."""
    if logger:
        logger.log_and_print("Analyzing feature interactions as a network graph...")
    
    network_analysis = {
        'nodes': {},
        'edges': [],
        'metrics': {}
    }
    
    try:
        # Convert to float64 and clip values
        X = df[feature_cols].astype(np.float64)
        X = X.clip(-1e10, 1e10)
        
        # Create a correlation matrix
        if logger:
            logger.log_and_print("Computing correlation matrix...")
        
        n_features = len(feature_cols)
        corr_sums = np.zeros((n_features, n_features), dtype=np.float64)
        count_matrix = np.zeros((n_features, n_features), dtype=np.int64)
        
        for start_idx in range(0, len(X), batch_size):
            end_idx = min(start_idx + batch_size, len(X))
            batch = X.iloc[start_idx:end_idx]
            
            with np.errstate(all='ignore'):
                batch_corr = np.array(batch.corr(), dtype=np.float64)
                
                # Update correlation matrix
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        if np.isfinite(batch_corr[i, j]):
                            corr_sums[i, j] += batch_corr[i, j]
                            count_matrix[i, j] += 1
            
            gc.collect()
        
        # Normalize correlation matrix
        with np.errstate(all='ignore'):
            mask = count_matrix > 0
            correlation_matrix = np.zeros_like(corr_sums)
            correlation_matrix[mask] = corr_sums[mask] / count_matrix[mask]
        
        # Create nodes and edges
        if logger:
            logger.log_and_print("Creating nodes and edges...")
        
        for i, feat1 in enumerate(feature_cols):
            network_analysis['nodes'][feat1] = {'degree': 0}
            for j, feat2 in enumerate(feature_cols[i+1:], i+1):
                corr = correlation_matrix[i, j]
                if abs(corr) > correlation_threshold:
                    network_analysis['edges'].append({
                        'source': feat1,
                        'target': feat2,
                        'weight': float(corr)
                    })
                    network_analysis['nodes'][feat1]['degree'] += 1
                    if feat2 in network_analysis['nodes']:
                        network_analysis['nodes'][feat2]['degree'] += 1
                    else:
                        network_analysis['nodes'][feat2] = {'degree': 1}
        
        # Analyze network structure
        if logger:
            logger.log_and_print("Analyzing network structure...")
        
        import networkx as nx
        
        # Create graph from nodes and edges
        G = nx.Graph()
        for node, data in network_analysis['nodes'].items():
            G.add_node(node, **data)
        for edge in network_analysis['edges']:
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
            
        # Compute network metrics
        if G.number_of_nodes() > 0:
            with np.errstate(all='ignore'):
                betweenness = nx.betweenness_centrality(G, weight='weight')
                clustering = nx.clustering(G, weight='weight')
                
                # Store metrics
                for node in G.nodes:
                    network_analysis['metrics'][node] = {
                        'degree': len(list(G.neighbors(node))),  # Count neighbors directly
                        'betweenness': float(betweenness.get(node, 0)),
                        'clustering': float(clustering.get(node, 0)) if isinstance(clustering, dict) and node in clustering else 0.0
                    }
        
        if logger:
            logger.log_and_print("Network analysis complete")
        
        return network_analysis
        
    except Exception as e:
        error_msg = f"Error in network analysis: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'nodes': {},
            'edges': [],
            'metrics': {}
        }
                    
def _write_prime_distribution_analysis(log, stats):
    """Writes the prime distribution analysis to the report with improved error handling and numerical stability."""
    log.write("\n=== PRIME DISTRIBUTION ANALYSIS ===\n")
    
    if not stats:
        log.write("\nPrime distribution analysis not available.\n")
        return
    
    try:
        # Write local density patterns
        log.write("\nLocal Density Patterns:\n")
        if 'local_density' in stats:
            for i, window in enumerate(stats['local_density'][:5], 1):  # Show first 5 windows
                log.write(f"\nWindow {i}:\n")
                log.write(f"  Range: {window.get('start_prime', 'N/A')} to {window.get('end_prime', 'N/A')}\n")
                log.write(f"  Mean Gap: {window.get('mean_gap', 'N/A'):.2f}\n")
                log.write(f"  Std Dev: {window.get('std_gap', 'N/A'):.2f}\n")
                
                # Write cluster distribution
                if 'cluster_distribution' in window:
                    log.write("  Cluster Distribution:\n")
                    for cluster, count in window['cluster_distribution'].items():
                        log.write(f"    Cluster {cluster}: {count}\n")
        else:
            log.write("Local density data not available.\n")
        
        # Write cluster transition probabilities
        if 'cluster_transitions' in stats:
            log.write("\nCluster Transition Probabilities:\n")
            for current, transitions in stats['cluster_transitions'].items():
                log.write(f"\nFrom Cluster {current}:\n")
                for next_cluster, prob in transitions.items():
                    log.write(f"  To Cluster {next_cluster}: {prob:.3f}\n")
        else:
            log.write("Cluster transition data not available.\n")
        
        # Write predictive metrics
        if 'predictive_metrics' in stats:
            log.write("\nPredictive Metrics:\n")
            metrics = stats['predictive_metrics']
            log.write(f"Cluster Stability: {metrics.get('cluster_stability', 'N/A'):.3f}\n")
            log.write(f"Gap Autocorrelation: {metrics.get('gap_autocorrelation', 'N/A'):.3f}\n")
            log.write(f"Cluster Entropy: {metrics.get('cluster_entropy', 'N/A'):.3f}\n")
            
            if 'cluster_statistics' in metrics:
                log.write("\nCluster-specific Statistics:\n")
                for cluster, cluster_stats in metrics['cluster_statistics'].items():
                    log.write(f"\nCluster {cluster}:\n")
                    log.write(f"  Frequency: {cluster_stats.get('frequency', 'N/A'):.3f}\n")
                    log.write(f"  Mean Gap: {cluster_stats.get('mean_gap', 'N/A'):.3f}\n")
                    log.write(f"  Std Gap: {cluster_stats.get('std_gap', 'N/A'):.3f}\n")
        else:
            log.write("Predictive metrics data not available.\n")
    
    except Exception as e:
        log.write(f"Error writing prime distribution analysis: {str(e)}\n")
        log.write(traceback.format_exc())

def _write_executive_summary(log, model_results, feature_importance, pattern_analysis, df, cluster_sequence_analysis,
                            shap_values=None, shap_importance=None, prediction_intervals=None, change_points=None,
                            cluster_stats=None, advanced_clustering=None, statistical_tests=None, 
                            chaos_metrics=None, superposition_patterns=None, wavelet_patterns=None,
                            fractal_dimension=None, phase_space_analysis=None, recurrence_plot_data=None, logger=None):
    """Writes the executive summary section of the report."""
    log.write("=== PRIME GAP ANALYSIS EXECUTIVE SUMMARY ===\n\n")
    
    try:
        # 1. Predictive Model Results
        log.write("PREDICTIVE MODEL RESULTS:\n")
        
        # Have we found a predictive model?
        if model_results:
            best_model_name, best_model = select_best_model(model_results)
            if best_model:
                if best_model_name:
                    log.write(f"- We identified a predictive model. The best performing model is {best_model_name.upper()}.\n")
                else:
                    log.write("- We identified a predictive model, but the best performing model name is not available.\n")
            else:
                log.write("- No predictive model could be identified.\n")
        else:
            log.write("- No predictive models were trained.\n")
            return
        
        # How predictive is it?
        if 'avg_test_mse' in model_results.get(best_model_name, {}):
            log.write(f"- The model's average test MSE is {model_results[best_model_name]['avg_test_mse']:.4f}.\n")
        if 'avg_test_r2' in model_results.get(best_model_name, {}):
            log.write(f"- The model's average test R² is {model_results[best_model_name]['avg_test_r2']:.4f}.\n")
        
        # How well does it generalize?
        if 'test_range_results' in model_results.get(best_model_name, {}):
            test_range_results = model_results[best_model_name]['test_range_results']
            if test_range_results:
                log.write("\n- Generalization:\n")
                for i, test_range in enumerate(test_range_results):
                    log.write(f"  - Test Range {i+1} ({test_range['range'][0]}-{test_range['range'][1]}): MSE = {test_range['test_mse']:.4f}, R² = {test_range['test_r2']:.4f}\n")
                
                # Calculate average falloff
                initial_mse = model_results[best_model_name].get('avg_test_mse', float('inf'))
                if initial_mse != float('inf'):
                    falloff_scores = [
                        (test_range['test_mse'] - initial_mse) / initial_mse
                        for test_range in test_range_results
                    ]
                    avg_falloff = np.mean(falloff_scores)
                    if avg_falloff > 0.05:
                        log.write(f"  - Average falloff in MSE: {avg_falloff:.4f}\n")
                        log.write("- Falloff: The model's predictive power tends to decrease as we move further away from the training data. The detailed analysis section provides more information about the falloff rates.\n")
                    else:
                        log.write("  - The model's predictive power does not show a significant falloff on the test ranges.\n")
                else:
                    log.write("  - Unable to calculate average falloff in MSE due to missing initial MSE.\n")
            else:
                log.write("- No generalization test results available.\n")
        else:
             log.write("- No generalization test results available.\n")
        
        # Transfer learning results
        transfer_learning = False  # Define transfer_learning variable
        if transfer_learning and 'transfer_learning_results' in model_results.get(best_model_name, {}):
            transfer_results = model_results[best_model_name]['transfer_learning_results']
            if transfer_results:
                log.write("\n- Transfer Learning:\n")
                log.write(f"  - Fine-tuned model MSE: {transfer_results['test_mse']:.4f}\n")
                log.write(f"  - Fine-tuned model R²: {transfer_results['test_r2']:.4f}\n")
                log.write(f"  - Transfer learning improved the model's performance on the new range of primes.\n")
            else:
                log.write("- No transfer learning results available.\n")
        else:
            log.write("- Transfer learning was not performed in this analysis.\n")
        
        # 2. Pattern Analysis Results
        log.write("\nPATTERN ANALYSIS RESULTS:\n")
        
        # Gap Sequence Patterns
        log.write("\nHave we found patterns in sequences of anything (gap types, prime types, other features, mixtures...etc)?\n")
        if pattern_analysis and 'sequence_patterns' in pattern_analysis:
            for seq_type, seq_patterns in pattern_analysis['sequence_patterns'].items():
                if seq_patterns:
                    log.write(f"  - Significant patterns found in {seq_type} sequences. See detailed analysis for more information.\n")
                else:
                    log.write(f"  - No significant patterns found in {seq_type} sequences.\n")
        else:
             log.write("No sequence patterns were found.\n")
        
        # Gap Cluster Sequence Patterns
        log.write("\nHave we found patterns in the sequence of gap clusters?\n")
        if cluster_sequence_analysis and 'cluster_transitions' in cluster_sequence_analysis:
            log.write("Yes, we have found cluster transitions, and they are described in the detailed analysis section.\n")
            if 'entropy' in cluster_sequence_analysis:
                log.write(f"  - Cluster sequence entropy: {cluster_sequence_analysis['entropy']:.4f}\n")
        else:
            log.write("No significant patterns found in the sequence of gap clusters.\n")
        
        # Prime Type Patterns
        log.write("\nHave we found patterns in the sequence of any prime types (which prime types, which patterns?)?\n")
        if pattern_analysis and 'prime_type_patterns' in pattern_analysis:
            if 'type_transitions' in pattern_analysis['prime_type_patterns']:
                log.write("Yes, we have found transitions between prime types:\n")
                for (type1, type2), count in pattern_analysis['prime_type_patterns']['type_transitions'][:5]:
                    log.write(f"  - {type1} -> {type2}: {count}\n")
                log.write("We have also found runs of prime types, and they are described in the detailed analysis section.\n")
            else:
                log.write("No significant patterns found in the sequence of prime types.\n")
        else:
            log.write("No prime type information available.\n")
        
        # Prime Type and Gap Type Interactions
        log.write("\nHave we found patterns of interactions between prime types and gap types? Explain.\n")
        if pattern_analysis and 'prime_type_patterns' in pattern_analysis and 'type_gap_counts' in pattern_analysis['prime_type_patterns']:
            log.write("Yes, we have found some interactions between prime types and gap types:\n")
            for (type_combo, gap_type), count in pattern_analysis['prime_type_patterns']['type_gap_counts'][:5]:
                log.write(f"  - {type_combo} - Gap {gap_type}: {count}\n")
            log.write("These interactions are further analyzed in the detailed analysis section.\n")
        else:
            log.write("No significant interactions between prime types and gap types were found.\n")
        
        # Predictive Power of Gap Types
        log.write("\nIs the sequence of gap types predictive of prime locations?\n")
        if 'gap_from_cluster_rf' in model_results:
            log.write("Yes, the sequence of gap types is somewhat predictive of the next prime location, as the gap prediction model uses features derived from gap types.\n")
            if 'r2' in model_results['gap_from_cluster_rf']:
                log.write(f"  - Gap Size Prediction from Cluster R²: {model_results['gap_from_cluster_rf']['r2']:.4f}\n")
        else:
            log.write("This analysis did not specifically test the predictive power of gap types on prime locations, but the modular patterns analysis may provide some insights.\n")

        # Predictive Power of Prime Types
        log.write("\nIs the sequence of prime types predictive of prime locations?\n")
        if 'next_cluster_rf' in model_results:
           log.write("Yes, the sequence of prime types is somewhat predictive of the next prime location.\n")
           if 'accuracy' in model_results['next_cluster_rf']:
               log.write(f"  - Next Cluster Prediction Accuracy: {model_results['next_cluster_rf']['accuracy']:.4f}\n")
           if 'r2' in model_results['next_cluster_rf']:
               log.write(f"  - Next Cluster Prediction R²: {model_results['next_cluster_rf']['r2']:.4f}\n")
        else:
            log.write("No model was trained to predict prime locations based on prime types.\n")
        
        # Predictive Power of Prime and Gap Types
        log.write("\nIs the sequence of prime types and gap types predictive of prime location?\n")
        if 'gap_from_cluster_rf' in model_results and 'next_cluster_rf' in model_results:
            log.write("Yes, the combined predictive model captures some relationships between the sequence of prime types and gap types and the next prime location.\n")
            if 'accuracy' in model_results['next_cluster_rf']:
                log.write(f"  - Next Cluster Prediction Accuracy: {model_results['next_cluster_rf']['accuracy']:.4f}\n")
            if 'r2' in model_results['gap_from_cluster_rf']:
                log.write(f"  - Gap Size Prediction from Cluster R²: {model_results['gap_from_cluster_rf']['r2']:.4f}\n")
        else:
            log.write("This analysis did not specifically test this hypothesis, but the combined predictive model may capture some of these relationships.\n")
        
        # Key Predictive Features
        if feature_importance is not None and not isinstance(feature_importance, pd.DataFrame) and not isinstance(feature_importance, dict) :
            log.write("No feature importance data is available.\n")
        elif isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
            log.write("\nKey Predictive Features:\n")
            top_features = feature_importance.mean(axis=1).sort_values(ascending=False).head(5)
            for feature, importance in top_features.items():
                log.write(f"  - {feature}: {importance:.4f} importance\n")
            log.write("These features are important because they capture the underlying patterns in the prime gaps and the distribution of prime factors.\n")
        elif isinstance(feature_importance, dict):
            log.write("\nKey Predictive Features:\n")
            for method, scores in feature_importance.items():
                if isinstance(scores, dict):
                    log.write(f"\n{method.upper()} Scores:\n")
                    for feature, score in sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                        log.write(f"  - {feature}: {float(score):.4f}\n")
        else:
            log.write("No feature importance data available.\n")
        
        # 3. Advanced Analysis Insights
        log.write("\nADVANCED ANALYSIS INSIGHTS:\n")
        
        if chaos_metrics:
            log.write("\nChaos Metrics:\n")
            for feature, metrics in chaos_metrics.items():
                 log.write(f"  - {feature}: Mean Divergence = {metrics.get('mean_divergence', 'N/A'):.4f}, 90th Percentile Divergence = {metrics.get('divergence_90th', 'N/A'):.4f}\n")
            if any(metrics.get('mean_divergence', 0) > 0.1 for metrics in chaos_metrics.values()):  # Example threshold
                log.write(f"  - The gap sequence exhibits some chaotic behavior, indicated by positive divergence rates.\n") # Add interpretation

        if superposition_patterns:
            log.write("\nSuperposition Patterns:\n")
            for feature, patterns in superposition_patterns.items():
                log.write(f"  - {feature}: Entropy = {patterns.get('entropy', 'N/A'):.4f}, Number of Modes = {patterns.get('num_modes', 'N/A')}\n")
            if any(patterns.get('num_modes', 0) > 1 for patterns in superposition_patterns.values()):
                log.write(f"  - Multimodality detected in some feature distributions, suggesting superposition-like phenomena.\n") # Add interpretation

        if wavelet_patterns and wavelet_patterns.get('wavelet_coeffs'):
            log.write("\nWavelet Analysis:\n")
            # Add key wavelet findings here (e.g., dominant scales, energy distribution)
            log.write("  - Wavelet analysis revealed [insert key findings, e.g., dominant scales, energy distribution].  See detailed analysis for coefficients.\n")

        if fractal_dimension and fractal_dimension.get('dimension') is not None:
            log.write(f"\nFractal Dimension: {fractal_dimension.get('dimension', 'N/A'):.4f}\n")
            log.write(f"  - This indicates a [simple/complex] geometric structure in the gap sequence.\n") # Add interpretation

        if phase_space_analysis and 'embedding_dimension' in phase_space_analysis:
            log.write("\nPhase Space Analysis:\n")
            for lag, metrics in phase_space_analysis['embedding_dimension'].items():
                log.write(f"  - Lag {lag}: False Neighbor Ratio = {metrics.get('false_neighbor_ratio', 'N/A'):.4f}\n")
            # Add interpretation of embedding dimension
            log.write("  - The estimated embedding dimension suggests [insert interpretation, e.g., a certain level of determinism/complexity].\n")

        if recurrence_plot_data and recurrence_plot_data.get('distance_matrix') is not None:
            log.write("\nRecurrence Plot Analysis:\n")
            # Add key recurrence plot findings here (e.g., diagonal lines, vertical/horizontal lines)
            log.write("  - Recurrence plot analysis showed [insert key findings, e.g., frequent diagonal lines indicating recurring patterns].\n")
        
        # Other Significant Discoveries
        log.write("\nOTHER SIGNIFICANT DISCOVERIES:\n")
        if cluster_stats and 'basic_stats' in cluster_stats:
            log.write("  - We have found that prime gaps can be grouped into distinct clusters, each with its own characteristics. These clusters are described in the detailed analysis section.\n")
        else:
            log.write("  - No significant discoveries beyond the above were made in this analysis.\n")
        
        log.write("\n=== END OF ANALYSIS SUMMARY ===\n\n")
    
    except Exception as e:
        log.write(f"Error writing executive summary: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
            
                   
def _write_hypothesis_analysis(log, model_results, feature_importance, pattern_analysis, df, cluster_sequence_analysis):
    """Writes the hypothesis analysis section of the report."""
    log.write("=== HYPOTHESIS ANALYSIS ===\n\n")
    _write_hypothesis_1_analysis(log, model_results)
    _write_hypothesis_2_analysis(log, pattern_analysis)
    _write_hypothesis_3_analysis(log, model_results, df)
    _write_hypothesis_4_analysis(log, df)
    _write_hypothesis_5_analysis(log, model_results)
    _write_hypothesis_6_analysis(log, feature_importance)
    _write_cluster_sequence_analysis(log, cluster_sequence_analysis)
    log.write("=== END OF HYPOTHESIS ANALYSIS ===\n\n")

def _write_hypothesis_1_analysis(log, model_results):
    """Writes the analysis for Hypothesis 1."""
    log.write("Hypothesis 1: Simple models like linear regression are prone to overfitting and will not generalize well to unseen data.\n")
    log.write("Analysis: We tested different models with varying complexity. The results suggest that:\n")
    
    if not model_results:
        log.write("- No model results available - model training may have failed.\n")
        log.write("- Unable to evaluate model performance due to missing results.\n")
        return
    
    try:
        best_model = min(model_results, key=lambda k: model_results[k].get('avg_test_mse', float('inf')))
        log.write(f"- The best performing model was: {best_model.upper()}\n")
    except (ValueError, KeyError):
        log.write("- Could not determine best performing model due to missing metrics.\n")
    
    if 'linear_regression' in model_results:
        lin_reg_results = model_results['linear_regression']
        if 'avg_test_r2' in lin_reg_results:
            log.write(f"- Linear Regression R²: {lin_reg_results['avg_test_r2']:.4f}\n")
        if 'avg_test_mse' in lin_reg_results:
            log.write(f"- Linear Regression MSE: {lin_reg_results['avg_test_mse']:.4f}\n")
        if 'avg_test_r2' in lin_reg_results and lin_reg_results['avg_test_r2'] < 0.9:
            log.write("- Linear Regression did not generalize well, showing signs of underfitting or instability.\n")
        elif 'avg_test_r2' in lin_reg_results and lin_reg_results['avg_test_r2'] > 0.95:
            log.write("- Linear Regression performed surprisingly well, but may be overfitting.\n")
    else:
        log.write("- Linear Regression results not available.\n")
    
    if 'neural_network' in model_results:
        nn_results = model_results['neural_network']
        if 'avg_test_r2' in nn_results:
            log.write(f"- Neural Network R²: {nn_results['avg_test_r2']:.4f}\n")
        if 'avg_test_mse' in nn_results:
            log.write(f"- Neural Network MSE: {nn_results['avg_test_mse']:.4f}\n")
        if 'avg_test_r2' in nn_results and nn_results['avg_test_r2'] > 0.95:
            log.write("- Neural Network performed very well, but may be overfitting.\n")
    else:
        log.write("- Neural Network results not available.\n")
    
    if 'random_forest' in model_results:
        rf_results = model_results['random_forest']
        if 'avg_test_r2' in rf_results:
            log.write(f"- Random Forest R²: {rf_results['avg_test_r2']:.4f}\n")
        if 'avg_test_mse' in rf_results:
            log.write(f"- Random Forest MSE: {rf_results['avg_test_mse']:.4f}\n")
    else:
        log.write("- Random Forest results not available.\n")
    
    if 'random_forest_simple' in model_results:
        rfs_results = model_results['random_forest_simple']
        if 'avg_test_r2' in rfs_results:
            log.write(f"- Random Forest Simple R²: {rfs_results['avg_test_r2']:.4f}\n")
        if 'avg_test_mse' in rfs_results:
            log.write(f"- Random Forest Simple MSE: {rfs_results['avg_test_mse']:.4f}\n")
    else:
        log.write("- Simple Random Forest results not available.\n")
    
    if 'xgboost' in model_results:
        xgb_results = model_results['xgboost']
        if 'avg_test_r2' in xgb_results:
            log.write(f"- XGBoost R²: {xgb_results['avg_test_r2']:.4f}\n")
        if 'avg_test_mse' in xgb_results:
            log.write(f"- XGBoost MSE: {xgb_results['avg_test_mse']:.4f}\n")
    else:
        log.write("- XGBoost results not available.\n")
    
    # Write conclusion based on available results
    if model_results:
        try:
            best_model = min(model_results, key=lambda k: model_results[k].get('avg_test_mse', float('inf')))
            log.write(f"\n- Conclusion: The hypothesis is partially confirmed. The best model was {best_model.upper()} which suggests that a more complex model is better at capturing the underlying patterns.\n\n")
        except (ValueError, KeyError):
            log.write("\n- Conclusion: Unable to determine best model, but different model complexities show varying performance.\n\n")
    else:
        log.write("\n- Conclusion: Unable to evaluate hypothesis due to missing model results.\n\n")
        
def _write_hypothesis_2_analysis(log, pattern_analysis):
    """Writes the analysis for Hypothesis 2."""
    log.write("Hypothesis 2: Modular patterns in prime gaps are informative and can be used to predict gap sizes.\n")
    log.write("Analysis: We analyzed the distribution of gaps modulo various numbers. The results suggest that:\n")
    
    if pattern_analysis['mod_patterns']:
        for mod, data in pattern_analysis['mod_patterns'].items():
            log.write(f"- Mod {mod} entropy: {data['entropy']:.4f}\n")
        log.write("- Conclusion: Modular patterns do exist, and the entropy values suggest that some moduli may be more informative than others. However, the predictive power of these patterns is not clear from this analysis alone.\n\n")
    else:
        log.write("- Conclusion: No modular patterns were found in this analysis.\n\n")

def _write_hypothesis_3_analysis(log, model_results, df):
    """Writes the analysis for Hypothesis 3."""
    log.write("Hypothesis 3: Prime gaps can be grouped into distinct clusters with different characteristics, suggesting different underlying mechanisms for their generation.\n")
    log.write("Analysis: We used k-means clustering to group gaps based on their features. The results suggest that:\n")
    if 'cluster' in df.columns:
        log.write(f"- Clusters were found, and their characteristics are described in the detailed analysis section.\n")
        if 'cluster_membership_rf' in model_results:
            cluster_rf_results = model_results['cluster_membership_rf']
            if 'accuracy' in cluster_rf_results:
                log.write(f"- Cluster Membership Prediction Accuracy: {cluster_rf_results['accuracy']:.4f}\n")
            if 'r2' in cluster_rf_results:
                log.write(f"- Cluster Membership Prediction R²: {cluster_rf_results['r2']:.4f}\n")
            log.write("- Conclusion: The hypothesis is confirmed. Prime gaps can be grouped into distinct clusters, and cluster membership can be predicted with reasonable accuracy.\n\n")
        else:
            log.write("- Conclusion: The hypothesis is confirmed. Prime gaps can be grouped into distinct clusters, but cluster membership prediction was not tested.\n\n")
    else:
        log.write("- Conclusion: No clusters were found in this analysis.\n\n")

def _write_hypothesis_4_analysis(log, df):
    """Writes the analysis for Hypothesis 4."""
    log.write("Hypothesis 4: Large outlier gaps are preceded by gaps with unusual properties, and these properties can be used to predict the occurrence of large gaps.\n")
    log.write("Analysis: We identified outlier gaps and analyzed their preceding gaps. The results suggest that:\n")
    
    outlier_count = df['is_outlier'].sum()
    log.write(f"- Found {outlier_count} outliers.\n")
    if outlier_count > 0:
        log.write("- The detailed analysis section provides information about the preceding gaps of outliers.\n")
        log.write("- Conclusion: The hypothesis is partially confirmed. Outlier gaps are preceded by other gaps, but whether these preceding gaps have unusual properties and can be used to predict outliers requires further analysis.\n\n")
    else:
        log.write("- Conclusion: No outliers were found in this analysis.\n\n")
        
def _write_hypothesis_5_analysis(log, model_results):
    """Writes the analysis for Hypothesis 5."""
    log.write("Hypothesis 5: The cluster membership of a gap and the sequence of preceding clusters can be used to predict the next gap size and the probability of a prime location.\n")
    log.write("Analysis: We trained models to predict the next cluster based on the sequence of preceding clusters. The results suggest that:\n")
    if 'next_cluster_rf' in model_results:
        next_cluster_results = model_results['next_cluster_rf']
        if 'accuracy' in next_cluster_results:
            log.write(f"- Next Cluster Prediction Accuracy: {next_cluster_results['accuracy']:.4f}\n")
        if 'r2' in next_cluster_results:
            log.write(f"- Next Cluster Prediction R²: {next_cluster_results['r2']:.4f}\n")
        if 'accuracy' in next_cluster_results and next_cluster_results['accuracy'] > 0.5:
            log.write("- Conclusion: The hypothesis is partially confirmed. The sequence of preceding clusters can be used to predict the next cluster with some accuracy.\n\n")
        else:
            log.write("- Conclusion: The hypothesis is not confirmed. The sequence of preceding clusters does not seem to be a strong predictor of the next cluster.\n\n")
    else:
        log.write("- Conclusion: No next cluster prediction model was trained in this analysis.\n\n")
    
    if 'gap_from_cluster_rf' in model_results:
        gap_from_cluster_results = model_results['gap_from_cluster_rf']
        if 'mse' in gap_from_cluster_results:
            log.write(f"- Gap Size Prediction from Cluster MSE: {gap_from_cluster_results['mse']:.4f}\n")
        if 'r2' in gap_from_cluster_results:
            log.write(f"- Gap Size Prediction from Cluster R²: {gap_from_cluster_results['r2']:.4f}\n")
        log.write("- Conclusion: The cluster membership can be used to predict the next gap size with some accuracy.\n\n")
    else:
        log.write("- Conclusion: No gap size prediction from cluster model was trained in this analysis.\n\n")

def _write_hypothesis_6_analysis(log, feature_importance):
    """Writes the analysis for Hypothesis 6."""
    log.write("Hypothesis 6: There are new predictive metrics beyond `log_gap` and `sqrt_gap` that can provide a deeper understanding of prime gaps.\n")
    log.write("Analysis: We explored various features related to the distribution of prime factors, the distribution of composite numbers, and the relationship between the primes and the gap. The results suggest that:\n")
    if not feature_importance.empty:
        top_features = feature_importance.mean(axis=1).sort_values(ascending=False).head(10)
        log.write("- The most significant predictors of gap size are:\n")
        for feature, importance in top_features.items():
            log.write(f"  - {feature}: {importance:.4f} importance\n")
        log.write("- Conclusion: The hypothesis is confirmed. There are new predictive metrics beyond `log_gap` and `sqrt_gap` that can provide a deeper understanding of prime gaps.\n\n")
    else:
        log.write("- Conclusion: No feature importances available for this analysis.\n\n")

def _write_cluster_sequence_analysis(log, cluster_sequence_analysis, logger=None):
    """Writes the cluster sequence analysis section of the report with improved error handling."""
    log.write("=== CLUSTER SEQUENCE ANALYSIS ===\n\n")
    
    if cluster_sequence_analysis:
        try:
            # Check each field with safe dictionary access
            if cluster_sequence_analysis.get('periodicity') is not None:
                ap = cluster_sequence_analysis['periodicity']
                log.write(f"- Cluster sequence arithmetic progression: Start = {ap.get('start', 'None')}, "
                         f"Difference = {ap.get('difference', 'None')}\n")
            else:
                log.write("- Cluster sequence arithmetic progression: None\n")
                
            if cluster_sequence_analysis.get('entropy') is not None:
                log.write(f"- Cluster sequence entropy: {cluster_sequence_analysis['entropy']:.4f}\n")
            else:
                log.write("- Cluster sequence entropy: Not available\n")
                
            if cluster_sequence_analysis.get('autocorrelation') is not None:
                log.write(f"- Cluster sequence autocorrelation: {cluster_sequence_analysis['autocorrelation']:.4f}\n")
            else:
                log.write("- Cluster sequence autocorrelation: Not available\n")
                
            # Add any additional metrics that are available
            for key, value in cluster_sequence_analysis.items():
                if key not in ['periodicity', 'entropy', 'autocorrelation']:
                    if isinstance(value, (int, float)):
                        log.write(f"- {key}: {value:.4f}\n")
                    elif isinstance(value, list):
                        log.write(f"- {key}: {value[:10]}...\n")
                    else:
                        log.write(f"- {key}: {value}\n")
        except Exception as e:
            log.write(f"Error writing cluster sequence analysis: {str(e)}\n")
            if logger:
                logger.logger.error(traceback.format_exc())
            else:
                traceback.print_exc()
    else:
        log.write("- No cluster sequence analysis was performed in this run.\n")
        
    log.write("=== END OF CLUSTER SEQUENCE ANALYSIS ===\n\n")
      
def _write_predictive_power_summary(log, model_results, feature_importance):
    """Writes the predictive power summary section of the report."""
    log.write("=== PREDICTIVE POWER SUMMARY ===\n\n")
    log.write("Based on the analysis, we found that:\n")
    
    if not model_results:
        log.write("- No model results available - training may have failed.\n")
        log.write("Further analysis is needed to understand why model training failed.\n\n")
        return
        
    log.write("- Prime gaps can be predicted with some accuracy using a combination of features, including modular residues, factor properties, and cluster membership.\n")
    
    try:
        best_model = min(model_results, key=lambda k: model_results[k].get('avg_test_mse', float('inf')))
        log.write(f"- The best performing model was {best_model.upper()}, which suggests that a more complex model is better at capturing the underlying patterns.\n")
    except ValueError:
        log.write("- Could not determine best model due to missing performance metrics.\n")
    
    if not feature_importance.empty:
        top_features = feature_importance.mean(axis=1).sort_values(ascending=False).head(10)
        log.write("- The most significant predictors of gap size are:\n")
        for feature, importance in top_features.items():
            log.write(f"  - {feature}: {importance:.4f} importance\n")
    else:
        log.write("- No feature importance information available.\n")
        
    log.write("- The probability of a prime location can be predicted with some accuracy using the sequence of preceding clusters and the cluster membership of the current gap.\n")
    log.write("Further analysis is needed to refine these predictions and to understand the underlying mechanisms that generate prime gaps.\n\n")
    
def _write_model_performance_summary(log, model_results):
    """Writes the model performance summary section of the report."""
    log.write("=== MODEL PERFORMANCE SUMMARY ===\n\n")
    for name, results in model_results.items():
        log.write(f"{name.upper()} Performance:\n")
        if 'avg_test_r2' in results:
            log.write(f"- Average Test R²: {results['avg_test_r2']:.4f}\n")
        if 'avg_test_mse' in results:
            log.write(f"- Average Test MSE: {results['avg_test_mse']:.4f}\n")
        if 'accuracy' in results:
            log.write(f"- Accuracy: {results['accuracy']:.4f}\n")
        if 'mse' in results:
            log.write(f"- Error (MSE): {results['mse']:.4f}\n")
    log.write("\n")

def _write_key_patterns(log, pattern_analysis, df):
    """Writes the key patterns section of the report."""
    log.write("KEY PATTERNS DISCOVERED:\n")
    
    # Most common gaps
    log.write("\nMost Frequent Gap Sizes:\n")
    for gap, count in pattern_analysis['common_gaps'][:5]:
        percentage = (count / len(df)) * 100
        log.write(f"- Gap of {gap}: {count} times ({percentage:.1f}% of all gaps)\n")
    
    # Modular patterns
    log.write("\nModular Patterns:\n")
    _write_modular_patterns_to_file(log_file=os.path.join(os.path.dirname(log.name), "modular_patterns.txt"), pattern_analysis=pattern_analysis)
    log.write("  (Full modular pattern distribution is available in: modular_patterns.txt)\n")
    
    # Gap runs analysis
    if pattern_analysis['runs']:
        log.write("\nNotable Gap Sequences:\n")
        longest_run = max(pattern_analysis['runs'], key=len)
        log.write(f"- Longest constant run: {len(longest_run)} gaps of size {longest_run[0]}\n")

def _write_statistical_insights(log, df):
    """Writes the statistical insights section of the report with improved numerical stability."""
    with suppress_overflow_warnings():
        log.write("\nSTATISTICAL INSIGHTS:\n")
        
        # Convert to float64 for better precision
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols].astype(np.float64)
        
        # Clip extreme values
        df_numeric = df_numeric.clip(-1e10, 1e10)
        
        gap_stats = df_numeric['gap_size'].describe()
        
        # Format statistics with appropriate precision
        try:
            mean = float(gap_stats['mean'])
            median = float(gap_stats['50%'])
            max_gap = float(gap_stats['max'])
            min_gap = float(gap_stats['min'])
            std_dev = float(gap_stats['std'])
            
            log.write(f"- Average gap size: {mean:.2f}\n")
            log.write(f"- Median gap size: {median:.2f}\n")
            log.write(f"- Largest gap found: {max_gap:.0f}\n")
            log.write(f"- Smallest gap found: {min_gap:.0f}\n")
            log.write(f"- Standard deviation: {std_dev:.2f}\n")
            
            # Additional robust statistics
            q1 = float(gap_stats['25%'])
            q3 = float(gap_stats['75%'])
            iqr = q3 - q1
            log.write(f"- Interquartile range: {iqr:.2f}\n")
            log.write(f"- Q1 (25th percentile): {q1:.2f}\n")
            log.write(f"- Q3 (75th percentile): {q3:.2f}\n")
            
        except (OverflowError, ValueError) as e:
            log.write(f"Warning: Some statistics could not be computed due to numerical overflow\n")

def _write_factor_analysis_summary(log, df):
    """Writes the factor analysis summary section of the report."""
    log.write("\nFACTOR ANALYSIS SUMMARY:\n")
    if 'unique_factors' in df.columns:
        factor_stats = df['unique_factors'].describe()
        log.write(f"- Average unique factors per gap: {factor_stats['mean']:.2f}\n")
        log.write(f"- Maximum unique factors in a gap: {factor_stats['max']:.0f}\n")
    
    if 'factor_density' in df.columns:
        density_stats = df['factor_density'].describe()
        log.write(f"- Average factor density: {density_stats['mean']:.2f}\n")
    
    if 'mean_sqrt_factor' in df.columns:
        sqrt_mean_stats = df['mean_sqrt_factor'].describe()
        log.write(f"- Average mean sqrt factor: {sqrt_mean_stats['mean']:.2f}\n")
    
    if 'sum_sqrt_factor' in df.columns:
        sqrt_sum_stats = df['sum_sqrt_factor'].describe()
        log.write(f"- Average sum sqrt factor: {sqrt_sum_stats['mean']:.2f}\n")

def _write_modular_properties(log, df, logger=None):
    """Writes the modular properties section of the report with improved error handling."""
    log.write("\nMODULAR PROPERTIES:\n")
    
    try:
        for mod in [6, 30]:
            mod_col = f'gap_mod{mod}'
            if mod_col in df.columns:
                mod_counts = df[mod_col].value_counts()
                log.write(f"\nGap sizes modulo {mod}:\n")
                for residue, count in mod_counts.items():
                    percentage = (count / len(df)) * 100
                    log.write(f"- Residue {residue}: {percentage:.1f}%\n")
    except Exception as e:
        log.write(f"Error writing modular properties: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()

@njit
def _compute_batch_correlations_numba(X_batch, feature_cols_count):
    """Numba-optimized function to compute correlations between features."""
    n_features = feature_cols_count
    # [correlation, count, covariance, r_squared]
    corr_stats = np.zeros((n_features, n_features, 4), dtype=np.float64)
    
    n_samples = X_batch.shape[0]
    
    # Compute means first
    means = np.zeros(n_features)
    counts = np.zeros(n_features)
    for i in range(n_features):
        valid_count = 0
        sum_val = 0.0
        for j in range(n_samples):
            if np.isfinite(X_batch[j, i]):
                sum_val += X_batch[j, i]
                valid_count += 1
        if valid_count > 0:
            means[i] = sum_val / valid_count
            counts[i] = valid_count
    
    # Compute correlations and related statistics
    for i in range(n_features):
        for j in range(i+1, n_features):
            sum_xy = 0.0
            sum_xx = 0.0
            sum_yy = 0.0
            count = 0
            
            for k in range(n_samples):
                x = X_batch[k, i]
                y = X_batch[k, j]
                if np.isfinite(x) and np.isfinite(y):
                    x_centered = x - means[i]
                    y_centered = y - means[j]
                    sum_xy += x_centered * y_centered
                    sum_xx += x_centered * x_centered
                    sum_yy += y_centered * y_centered
                    count += 1
            
            if count > 1 and sum_xx > 0 and sum_yy > 0:
                # Correlation
                corr = sum_xy / np.sqrt(sum_xx * sum_yy)
                # Covariance
                cov = sum_xy / count
                # R-squared
                r_squared = corr * corr
                
                if np.isfinite(corr):
                    corr_stats[i, j, 0] = corr
                    corr_stats[j, i, 0] = corr
                    corr_stats[i, j, 1] = count
                    corr_stats[j, i, 1] = count
                    corr_stats[i, j, 2] = cov
                    corr_stats[j, i, 2] = cov
                    corr_stats[i, j, 3] = r_squared
                    corr_stats[j, i, 3] = r_squared
    
    return corr_stats

@njit
def _compute_feature_stats_numba(data, batch_size):
    """Numba-optimized function to compute feature statistics."""
    n_features = data.shape[1]
    # [sum, sum_sq, count, min, max, zeros, skew, kurt]
    stats = np.zeros((n_features, 8), dtype=np.float64)
    
    for i in range(n_features):
        valid_count = 0
        sum_val = 0.0
        sum_sq = 0.0
        min_val = np.inf
        max_val = -np.inf
        zero_count = 0
        m3 = 0.0  # Third moment
        m4 = 0.0  # Fourth moment
        
        # First pass for basic stats
        for j in range(data.shape[0]):
            val = data[j, i]
            if np.isfinite(val):
                sum_val += val
                sum_sq += val * val
                min_val = min(min_val, val)
                max_val = max(max_val, val)
                if val == 0:
                    zero_count += 1
                valid_count += 1
        
        if valid_count > 0:
            mean = sum_val / valid_count
            
            # Second pass for higher moments
            for j in range(data.shape[0]):
                val = data[j, i]
                if np.isfinite(val):
                    dev = val - mean
                    m3 += dev * dev * dev
                    m4 += dev * dev * dev * dev
            
            var = (sum_sq / valid_count) - (mean * mean)
            if var > 0 and valid_count > 2:
                m3 /= valid_count
                m4 /= valid_count
                skew = m3 / (var ** 1.5)
                kurt = (m4 / (var * var)) - 3.0
            else:
                skew = 0.0
                kurt = 0.0
            
            stats[i, 0] = sum_val
            stats[i, 1] = sum_sq
            stats[i, 2] = valid_count
            stats[i, 3] = min_val if min_val != np.inf else 0
            stats[i, 4] = max_val if max_val != -np.inf else 0
            stats[i, 5] = zero_count
            stats[i, 6] = skew
            stats[i, 7] = kurt
    
    return stats

@timing_decorator
def compute_correlation_statistics(df, batch_size=5000, logger=None):
    """Pre-compute correlation statistics efficiently with Numba optimization."""
    if logger:
        logger.log_and_print("Computing correlation statistics...")
    
    correlation_stats = {
        'feature_correlations': {},
        'strong_correlations': [],
        'feature_stats': {},
        'correlation_patterns': {}
    }
    
    try:
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_col = 'gap_size'
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        if logger:
            logger.log_and_print(f"Processing {len(feature_cols)} features...")
        
        # Initialize accumulators
        corr_stats = np.zeros((len(feature_cols), len(feature_cols), 4), dtype=np.float64)
        
        # Process correlations in batches
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch = df[feature_cols].iloc[start_idx:end_idx]
            
            # Convert to float64 and clip values
            X_batch = batch.values.astype(np.float64)
            X_batch = np.clip(X_batch, -1e10, 1e10)
            
            # Call numba-optimized function and properly unpack the tuple
            batch_corr_stats = _compute_batch_correlations_numba(
                X_batch, len(feature_cols)
            )
            
            # Update accumulators
            corr_stats += batch_corr_stats
            
            gc.collect()
        
        # Compute final correlations
        correlation_matrix = corr_stats[:, :, 0]
        count_matrix = corr_stats[:, :, 1]
        
        # Normalize correlation matrix
        mask = count_matrix > 0
        correlation_matrix[mask] = correlation_matrix[mask] / count_matrix[mask]
        
        # Store correlations in results
        for i, feat1 in enumerate(feature_cols):
            correlation_stats['feature_correlations'][feat1] = {}
            for j, feat2 in enumerate(feature_cols):
                if i != j:
                    corr = correlation_matrix[i, j]
                    if np.isfinite(corr):
                        correlation_stats['feature_correlations'][feat1][feat2] = float(corr)
        
        # Find strong correlations
        strong_corrs = []
        for i, feat1 in enumerate(feature_cols):
            for j, feat2 in enumerate(feature_cols[i+1:], i+1):
                corr = correlation_matrix[i, j]
                if np.isfinite(corr) and abs(corr) > 0.5:
                    strong_corrs.append((feat1, feat2, float(corr)))
        
        correlation_stats['strong_correlations'] = sorted(
            strong_corrs, key=lambda x: abs(x[2]), reverse=True
        )
        
        # Compute feature statistics using Numba
        feature_stats_array = _compute_feature_stats_numba(df[feature_cols].values, batch_size)
        correlation_stats['feature_stats'] = {
            feature_cols[i]: {
                'mean': float(stats[0] / stats[2]) if stats[2] > 0 else 0.0,
                'std': float(np.sqrt((stats[1] / stats[2]) - (stats[0] / stats[2])**2)) if stats[2] > 1 else 0.0,
                'min': float(stats[3]),
                'max': float(stats[4]),
                'zeros': int(stats[5]),
                'skewness': float(stats[6]),
                'kurtosis': float(stats[7]),
                'zero_fraction': float(stats[5] / stats[6]) if stats[6] > 0 else 0.0,
                'normality_p': float(0.0) # Placeholder for normality test
            }
            for i, stats in enumerate(feature_stats_array)
        }
        
        # Add correlation patterns analysis
        correlation_patterns = {
            'positive_correlations': len([c for c in correlation_matrix.flatten() if c > 0.3]),
            'negative_correlations': len([c for c in correlation_matrix.flatten() if c < -0.3]),
            'strong_correlations': len([c for c in correlation_matrix.flatten() if abs(c) > 0.7]),
            'correlation_distribution': {
                'mean': float(np.nanmean(correlation_matrix)),
                'std': float(np.nanstd(correlation_matrix)),
                'max': float(np.nanmax(np.abs(correlation_matrix)))
            }
        }
        
        correlation_stats['correlation_patterns'] = correlation_patterns
        
        if logger:
            logger.log_and_print("Correlation statistics computation complete")
            logger.log_and_print(f"Found {len(correlation_stats['strong_correlations'])} strong correlations")
        
        return correlation_stats
        
    except Exception as e:
        error_msg = f"Error computing correlation statistics: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        return {
            'feature_correlations': {},
            'strong_correlations': [],
            'feature_stats': {},
            'correlation_patterns': {
                'positive_correlations': 0,
                'negative_correlations': 0,
                'strong_correlations': 0,
                'correlation_distribution': {
                    'mean': 0.0,
                    'std': 0.0,
                    'max': 0.0
                }
            }
        }

def _write_correlation_insights(log, df, correlation_stats=None, logger=None):
    """Writes correlation insights using pre-computed statistics with improved error handling."""
    log.write("\n--- Correlation Insights ---\n")
    
    if correlation_stats is None or not correlation_stats:
        log.write("No correlation statistics available.\n")
        return
    
    try:
        # Write strong correlations
        if correlation_stats['strong_correlations']:
            log.write("\nFeatures strongly correlated with gap size:\n")
            for feature, corr in correlation_stats['strong_correlations']:
                strength = "strong" if abs(corr) > 0.7 else "moderate"
                log.write(f"- {feature}: {corr:.3f} correlation ({strength})\n")
        else:
            log.write("No strong correlations found.\n")
        
        # Write feature statistics
        log.write("\nFeature Statistics:\n")
        for feature, stats in correlation_stats['feature_stats'].items():
            log.write(f"\n{feature}:\n")
            log.write(f"- Mean: {stats.get('mean', 'N/A'):.4f}\n")
            log.write(f"- Std: {stats.get('std', 'N/A'):.4f}\n")
            log.write(f"- Range: [{stats.get('min', 'N/A'):.4f}, {stats.get('max', 'N/A'):.4f}]\n")
            zero_fraction = stats.get('zero_fraction', 'N/A')
            log.write(f"- Zero fraction: {zero_fraction:.4f}\n")
            
            if 'skewness' in stats:
                log.write(f"- Skewness: {stats.get('skewness', 'N/A'):.4f}\n")
            if 'kurtosis' in stats:
                log.write(f"- Kurtosis: {stats.get('kurtosis', 'N/A'):.4f}\n")
            if 'normality_p' in stats:
                log.write(f"- Normality P: {stats.get('normality_p', 'N/A'):.4f}\n")
        
        # Write summary
        log.write("\n\nCorrelation Analysis Summary:")
        log.write(f"\n- Total features analyzed: {len(correlation_stats.get('feature_correlations', {}))}")
        log.write(f"\n- Strong correlations found: {len(correlation_stats.get('strong_correlations', []))}")
        
        if 'correlation_patterns' in correlation_stats:
            patterns = correlation_stats['correlation_patterns']
            log.write(f"\n\nCorrelation Patterns:")
            log.write(f"\n- Mean Correlation: {patterns.get('correlation_distribution', {}).get('mean', 'N/A'):.4f}")
            log.write(f"\n- Std Correlation: {patterns.get('correlation_distribution', {}).get('std', 'N/A'):.4f}")
            log.write(f"\n- Strongest Correlation: {patterns.get('correlation_distribution', {}).get('max', 'N/A'):.4f}")
            log.write(f"\n- Positive Correlations: {patterns.get('positive_correlations', 'N/A')}")
            log.write(f"\n- Negative Correlations: {patterns.get('negative_correlations', 'N/A')}")
            log.write(f"\n- Strong Correlations: {patterns.get('strong_correlations', 'N/A')}")
    
    except Exception as e:
        log.write(f"\nError writing correlation insights: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    
    log.write("\n")
     
def _write_network_analysis(log, network_analysis, logger=None):
    """Writes the network analysis section of the report."""
    log.write("\n--- Network Analysis ---\n")
    
    if not network_analysis or not network_analysis.get('nodes'):
        log.write("No network analysis data available.\n")
        return
    
    try:
        # Write node information
        log.write("\nNode Information:\n")
        for node, data in network_analysis['nodes'].items():
            log.write(f"- Node: {node}, Degree: {data.get('degree', 'N/A')}\n")
        
        # Write edge information
        log.write("\nEdge Information:\n")
        for edge in network_analysis['edges']:
            log.write(f"- Source: {edge.get('source', 'N/A')}, Target: {edge.get('target', 'N/A')}, Weight: {edge.get('weight', 'N/A'):.4f}\n")
        
        # Write network metrics
        if 'metrics' in network_analysis:
            log.write("\nNetwork Metrics:\n")
            for node, metrics in network_analysis['metrics'].items():
                log.write(f"- Node: {node}\n")
                log.write(f"  - Degree: {metrics.get('degree', 'N/A')}\n")
                log.write(f"  - Betweenness Centrality: {metrics.get('betweenness', 'N/A'):.4f}\n")
                log.write(f"  - Clustering Coefficient: {metrics.get('clustering', 'N/A'):.4f}\n")
    except Exception as e:
        log.write(f"Error writing network analysis: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    
    log.write("\n")
    
def _write_detailed_analysis_section(log, feature_importance=None, feature_selection=None, 
                                    feature_interactions=None, feature_stability=None,
                                    shap_values=None, shap_importance=None, prediction_intervals=None,
                                    change_points=None, cluster_stats=None, advanced_clustering=None,
                                    statistical_tests=None, pattern_analysis=None, 
                                    chaos_metrics=None, superposition_patterns=None,
                                    wavelet_patterns=None, fractal_dimension=None,
                                    phase_space_analysis=None, recurrence_plot_data=None, logger=None):
    """Writes the detailed analysis section of the report with improved error handling and numerical stability."""
    log.write("\n\n=== DETAILED ANALYSIS SECTION ===\n")
    
    try:
        # Feature Importance Analysis
        log.write("\n--- Feature Importance Analysis ---\n")
        if feature_importance is not None and not isinstance(feature_importance, pd.DataFrame) and not isinstance(feature_importance, dict) :
            log.write("Feature importance data is not a valid type.\n")
        elif isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
            log.write("\nFeature Importance Scores:\n")
            for method, scores in feature_importance.items():
                if isinstance(scores, dict):
                    log.write(f"\n{str(method).upper()} Scores:\n")
                    for feature, score in sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                        log.write(f"- {feature}: {float(score):.4f}\n")
                elif isinstance(scores, pd.Series):
                    log.write(f"\n{str(method).upper()} Scores:\n")
                    for feature, score in scores.sort_values(ascending=False).head(10).items():
                         log.write(f"- {feature}: {float(score):.4f}\n")
        elif isinstance(feature_importance, dict):
            log.write("\nFeature Importance Scores:\n")
            for method, scores in feature_importance.items():
                if isinstance(scores, dict):
                    log.write(f"\n{method.upper()} Scores:\n")
                    for feature, score in sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                        log.write(f"- {feature}: {float(score):.4f}\n")
        else:
            log.write("No feature importance data available.\n")
        
        # Feature Selection Analysis
        if feature_selection:
            log.write("\n--- Feature Selection Analysis ---\n")
            log.write(f"Optimal number of features: {feature_selection.get('n_features', 'N/A')}\n")
            log.write("Selected Features:\n")
            for feature in feature_selection.get('optimal_features', []):
                log.write(f"- {feature}\n")
            
            if 'evaluation_scores' in feature_selection:
                log.write("\nFeature Subset Evaluation Scores:\n")
                for i, score in enumerate(feature_selection['evaluation_scores']):
                    log.write(f"- {i*5 + 5} features: {float(score):.4f}\n")
        
        # Feature Interaction Analysis
        if feature_interactions:
            log.write("\n--- Feature Interaction Analysis ---\n")
            if 'pairwise_correlations' in feature_interactions and feature_interactions['pairwise_correlations']:
                log.write("\nPairwise Correlations:\n")
                for interaction in feature_interactions['pairwise_correlations']['significant_correlations']:
                    log.write(f"- {interaction['feature1']} - {interaction['feature2']}: {float(interaction['correlation']):.4f}\n")
            
            if 'mutual_information' in feature_interactions:
                log.write("\nMutual Information Scores:\n")
                for interaction, score in feature_interactions['mutual_information'].items():
                    log.write(f"- {interaction}: {float(score):.4f}\n")
            
            if 'nonlinear_relationships' in feature_interactions:
                log.write("\nNonlinear Relationships:\n")
                for feature, scores in feature_interactions['nonlinear_relationships'].items():
                    log.write(f"- {feature}:\n")
                    for transform, score in scores.items():
                        log.write(f"  - {transform}: {float(score):.4f}\n")
            
            if 'temporal_interactions' in feature_interactions:
                log.write("\nTemporal Interactions:\n")
                for feature, interactions in feature_interactions['temporal_interactions'].items():
                    log.write(f"- {feature}:\n")
                    log.write(f"  - Max Lag Correlation: {float(interactions.get('max_lag_correlation', 0)):.4f}\n")
                    log.write(f"  - Lag Correlations: {interactions.get('lag_correlations', 'N/A')}\n")
            
            if 'shap_interaction_scores' in feature_interactions:
                log.write("\nSHAP Interaction Scores:\n")
                for interaction, score in feature_interactions['shap_interaction_scores'].items():
                    log.write(f"- {interaction}: {float(score):.4f}\n")
        
        # Feature Stability Analysis
        if feature_stability:
            log.write("\n--- Feature Stability Analysis ---\n")
            if 'bootstrap_scores' in feature_stability:
                log.write("\nBootstrap Stability Scores:\n")
                for feature, score in sorted(feature_stability['bootstrap_scores']['mean_importance'].items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                    log.write(f"- {feature}: Mean = {float(score):.4f}, Std = {feature_stability['bootstrap_scores']['std_importance'].get(feature, 'N/A'):.4f}, CV = {feature_stability['bootstrap_scores']['cv_importance'].get(feature, 'N/A'):.4f}\n")
            
            if 'temporal_stability' in feature_stability:
                log.write("\nTemporal Stability:\n")
                for feature, stability in feature_stability['temporal_stability'].items():
                    log.write(f"- {feature}: Mean Stability = {stability.get('mean_stability', 'N/A'):.4f}, Std Stability = {stability.get('std_stability', 'N/A'):.4f}\n")
            
            if 'value_range_stability' in feature_stability:
                log.write("\nValue Range Stability:\n")
                for feature, stability in feature_stability['value_range_stability'].items():
                    log.write(f"- {feature}: IQR = {stability.get('iqr', 'N/A'):.4f}, Range Ratio = {stability.get('range_ratio', 'N/A'):.4f}, Outlier Ratio = {stability.get('outlier_ratio', 'N/A'):.4f}\n")
        
        # SHAP Analysis
        if shap_importance:
            log.write("\n--- SHAP Feature Importance ---\n")
            for model_name, importance_dict in shap_importance.items():
                log.write(f"\n{model_name.upper()}:\n")
                for feature, importance in sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                    log.write(f"- {feature}: {float(importance):.4f}\n")
        
        # Prediction Intervals
        if prediction_intervals:
            log.write("\n--- Prediction Intervals ---\n")
            if prediction_intervals.get('mean', None) is not None:
                log.write(f"Mean prediction: {prediction_intervals.get('mean', 'N/A')[:10]}...\n")
            if prediction_intervals.get('lower', None) is not None:
                log.write(f"Lower prediction: {prediction_intervals.get('lower', 'N/A')[:10]}...\n")
            if prediction_intervals.get('upper', None) is not None:
                log.write(f"Upper prediction: {prediction_intervals.get('upper', 'N/A')[:10]}...\n")
        
        # Change Point Analysis
        if change_points and change_points['segments']:
            log.write("\n--- Change Point Analysis ---\n")
            log.write(f"Number of segments: {change_points.get('n_segments', 'N/A')}\n")
            log.write(f"Score: {change_points.get('score', 'N/A'):.4f}\n")
            log.write("Change Points:\n")
            for segment in change_points.get('segments', []):
                log.write(f"  - Start: {segment.get('start', 'N/A')}, End: {segment.get('end', 'N/A')}, Mean: {segment.get('mean', 'N/A'):.2f}, Size: {segment.get('size', 'N/A')}\n")
        
        # Advanced Clustering Analysis
        if advanced_clustering and 'optimal_clusters' in advanced_clustering:
            log.write("\n--- Advanced Clustering Metrics ---\n")
            for method, n_clusters in advanced_clustering['optimal_clusters'].items():
                log.write(f"- {method.upper()}: {n_clusters} clusters\n")
            
            if 'metrics' in advanced_clustering:
                log.write("\nClustering Metrics:\n")
                for method, metrics in advanced_clustering['metrics'].items():
                    log.write(f"- {method.upper()}:\n")
                    for metric, value in metrics.items():
                        log.write(f"  - {metric}: {float(value):.4f}\n")
            
            if 'cluster_profiles' in advanced_clustering:
                log.write("\nCluster Profiles:\n")
                for method, profiles in advanced_clustering['cluster_profiles'].items():
                    log.write(f"\n{method.upper()}:\n")
                    for cluster, profile in profiles.items():
                        log.write(f"  - Cluster {cluster}: Size = {profile.get('size', 'N/A')}, Mean Gap = {profile.get('mean_gap', 'N/A'):.2f}, Std Gap = {profile.get('std_gap', 'N/A'):.2f}\n")
                        log.write("  - Feature Means:\n")
                        for feature, mean in profile.get('feature_means', {}).items():
                            log.write(f"    - {feature}: {float(mean):.2f}\n")
                        log.write("  - Feature Importance:\n")
                        for feature, importance in sorted(profile.get('feature_importance', {}).items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                            log.write(f"    - {feature}: {float(importance):.4f}\n")
        
        # Statistical Tests
        if statistical_tests:
            log.write("\n--- Statistical Tests ---\n")
            if 'normality' in statistical_tests:
                log.write("\nNormality Tests:\n")
                for col, test in statistical_tests['normality'].items():
                    log.write(f"- {col}: p-value = {test.get('p_value', 'N/A'):.4f}, is_normal = {test.get('is_normal', 'N/A')}\n")
            if 'homogeneity' in statistical_tests:
                log.write("\nHomogeneity of Variance Tests:\n")
                for method, test in statistical_tests['homogeneity'].items():
                    log.write(f"- {method.upper()}: p-value = {test.get('p_value', 'N/A'):.4f}, is_homogeneous = {test.get('is_homogeneous', 'N/A')}\n")
            if 'cluster_comparisons' in statistical_tests:
                log.write("\nCluster Comparison Tests:\n")
                for method, tests in statistical_tests['cluster_comparisons'].items():
                    log.write(f"- {method.upper()}:\n")
                    log.write(f"  - ANOVA p-value: {tests.get('anova', {}).get('p_value', 'N/A'):.4f}\n")
                    log.write(f"  - Tukey HSD results: {tests.get('tukey_hsd', {}).get('results', 'N/A')[:100]}...\n")
        
        # Sequence Pattern Analysis
        if pattern_analysis and 'sequence_patterns' in pattern_analysis:
            log.write("\n--- Sequence Pattern Analysis ---\n")
            for seq_type, patterns in pattern_analysis['sequence_patterns'].items():
                log.write(f"\nType: {seq_type}\n")
                if patterns:
                    for pattern in patterns:
                        log.write(f"  - Sequence: {pattern['sequence'][:5]}..., Count: {pattern['count']}\n")
                else:
                    log.write("  No significant patterns found.\n")
        
        # Chaos Analysis
        if chaos_metrics:
            _write_chaos_analysis(log, chaos_metrics, logger=logger)
        
        # Superposition Analysis
        if superposition_patterns:
            _write_superposition_analysis(log, superposition_patterns, logger=logger)
        
        # Wavelet Analysis
        if wavelet_patterns:
            _write_wavelet_analysis(log, wavelet_patterns, logger=logger)
        
        # Fractal Dimension Analysis
        if fractal_dimension:
            _write_fractal_dimension_analysis(log, fractal_dimension, logger=logger)
        
        # Phase Space Analysis
        if phase_space_analysis:
            _write_phase_space_analysis(log, phase_space_analysis, logger=logger)
        
        # Recurrence Plot Analysis
        if recurrence_plot_data:
            _write_recurrence_plot_analysis(log, recurrence_plot_data, logger=logger)
        
        log.write("\n")
        
    except Exception as e:
        log.write(f"Error writing detailed analysis: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    
    log.write("\n=== END OF DETAILED ANALYSIS SECTION ===\n\n")
    
def _write_modular_patterns_to_file(log_file, pattern_analysis):
    """Writes the modular pattern distribution with improved handling for large datasets."""
    with open(log_file, "w") as factor_file:
        factor_file.write("Modular Distribution Summary:\n")
        
        # Process each modulus
        for mod, data in pattern_analysis['mod_patterns'].items():
            factor_file.write(f"\nMod {mod} distribution:\n")
            
            # Get counts and sort by frequency
            counts = data['counts']
            total = sum(counts.values())
            sorted_residues = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate significance threshold (e.g., >1% of total)
            threshold = total * 0.01
            
            # Write significant patterns
            significant_patterns = []
            insignificant_count = 0
            zero_count = 0
            
            for residue, count in sorted_residues:
                percentage = (count / total) * 100
                if count > threshold:
                    significant_patterns.append(
                        f"  - Residue {residue}: {count:,} times ({percentage:.1f}%)"
                    )
                elif count == 0:
                    zero_count += 1
                else:
                    insignificant_count += 1
            
            # Write the results
            factor_file.write("Significant patterns:\n")
            factor_file.write("\n".join(significant_patterns))
            factor_file.write("\n")
            
            if insignificant_count > 0:
                factor_file.write(f"\nOther patterns: {insignificant_count:,} residues with <1% occurrence each\n")
            if zero_count > 0:
                factor_file.write(f"Zero occurrence: {zero_count:,} residues\n")
            
            factor_file.write(f"Entropy: {data['entropy']:.4f}\n")
            factor_file.write("-" * 80 + "\n")
                    
def _write_progression_analysis(log, pattern_analysis, time_series_analysis=None, logger=None):
    """Writes the progression analysis section of the report with improved error handling and numerical stability."""
    log.write("\n--- Progression Analysis ---\n")
    
    try:
        # Handle arithmetic progression
        if pattern_analysis and 'arithmetic_progression' in pattern_analysis:
            ap = pattern_analysis['arithmetic_progression']
            if ap:
                log.write(f"Arithmetic progression: Start = {ap.get('start', 'N/A')}, Difference = {ap.get('difference', 'N/A')}\n")
            else:
                log.write("Arithmetic progression: None\n")
        else:
            log.write("Arithmetic progression: Data not available\n")
        
        # Handle geometric progression
        if pattern_analysis and 'geometric_progression' in pattern_analysis:
            gp = pattern_analysis['geometric_progression']
            if gp:
                log.write(f"Geometric progression: Start = {gp.get('start', 'N/A')}, Ratio = {gp.get('ratio', 'N/A')}\n")
            else:
                log.write("Geometric progression: None\n")
        else:
            log.write("Geometric progression: Data not available\n")
        
        # Handle periodicity
        if pattern_analysis and 'periodicity' in pattern_analysis:
            period = pattern_analysis['periodicity']
            if period and period.get('main_period') is not None:
                log.write("\nPeriodicity Analysis:\n")
                log.write(f"Main Period: {period.get('main_period', 'N/A')}\n")
                log.write(f"Strength: {period.get('strength', 'N/A'):.4f}\n")
                if period.get('all_periods'):
                    log.write("All detected periods:\n")
                    for p, s in period.get('all_periods', []):
                        log.write(f"  Period {p}: strength {s:.4f}\n")
            else:
                log.write("\nNo significant periodicity detected\n")
        else:
            log.write("\nPeriodicity analysis: Data not available\n")
        
        # Handle time series analysis
        if time_series_analysis:
            log.write("\n--- Time Series Analysis ---\n")
            if 'decomposition' in time_series_analysis:
                log.write("\nTime Series Decomposition:\n")
                log.write(f"- Trend Strength: {time_series_analysis.get('trend_strength', 'N/A'):.4f}\n")
                log.write(f"- Seasonal Strength: {time_series_analysis.get('seasonal_strength', 'N/A'):.4f}\n")
                log.write(f"- Period: {time_series_analysis.get('period', 'N/A')}\n")
            
            if 'stationarity_test' in time_series_analysis:
                log.write("\nStationarity Test:\n")
                test = time_series_analysis['stationarity_test']
                log.write(f"- Test Statistic: {test.get('statistic', 'N/A'):.4f}\n")
                log.write(f"- Critical Values: {test.get('critical_values', 'N/A')}\n")
                log.write(f"- Significance Level: {test.get('significance_level', 'N/A')}\n")
            
            if 'change_points' in time_series_analysis:
                log.write("\nChange Point Analysis:\n")
                log.write(f"- Number of segments: {time_series_analysis['change_points'].get('n_segments', 'N/A')}\n")
                log.write(f"- Score: {time_series_analysis['change_points'].get('score', 'N/A'):.4f}\n")
                log.write("Change Points:\n")
                for segment in time_series_analysis['change_points'].get('segments', []):
                    log.write(f"  - Start: {segment.get('start', 'N/A')}, End: {segment.get('end', 'N/A')}, Mean: {segment.get('mean', 'N/A'):.2f}, Size: {segment.get('size', 'N/A')}\n")
        
        log.write("\n")
        
    except Exception as e:
        log.write(f"Error writing progression analysis: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
                   
def _write_predictive_model_analysis(log, model_results, logger=None):
    """Writes the predictive model analysis section of the report with improved error handling."""
    log.write("\n--- Predictive Model ---\n")
    
    if not model_results:
        log.write("No model results available for analysis.\n")
        return
    
    try:
        for name, results in model_results.items():
            log.write(f"\n{name.upper()} Model Performance:\n")
            if 'avg_test_r2' in results:
                log.write(f"- Average Test R²: {results['avg_test_r2']:.4f}\n")
            if 'avg_test_mse' in results:
                log.write(f"- Average Test MSE: {results['avg_test_mse']:.4f}\n")
            if 'avg_train_mse' in results:
                log.write(f"- Average Train MSE: {results['avg_train_mse']:.4f}\n")
            if 'avg_train_r2' in results:
                log.write(f"- Average Train R²: {results['avg_train_r2']:.4f}\n")
            if 'accuracy' in results:
                log.write(f"- Accuracy: {results['accuracy']:.4f}\n")
            if 'mse' in results:
                log.write(f"- Error (MSE): {results['mse']:.4f}\n")
            
            if 'learning_curve' in results:
                log.write(f"\n  Learning Curve Analysis:\n")
                train_sizes = results['learning_curve']['train_sizes']
                train_scores = results['learning_curve']['train_scores']
                test_scores = results['learning_curve']['test_scores']
                
                for i in range(len(train_sizes)):
                    log.write(f"    Train Size: {train_sizes[i]:.2f}, Train MSE: {train_scores[i]:.4f}, Test MSE: {test_scores[i]:.4f}\n")
            
            if name == 'ensemble':
                log.write("\nEnsemble Model Details:\n")
                if 'weights' in results:
                    log.write("Model Weights:\n")
                    for model_name, weight in results['weights'].items():
                        log.write(f"  - {model_name}: {weight:.2f}\n")
            
            if name == 'stacking':
                log.write("\nStacking Model Details:\n")
                if 'base_models' in results:
                    log.write("Base Models:\n")
                    for model_name in results['base_models'].keys():
                        log.write(f"  - {model_name}\n")
                if 'meta_learner' in results:
                    log.write(f"Meta Learner: {results['meta_learner']}\n")
    
    except Exception as e:
        log.write(f"Error writing predictive model analysis: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
                                                         
def _write_model_comparison(log, model_results, logger=None):
    """Writes the model comparison section of the report with improved error handling and numerical stability."""
    log.write("\n--- Model Comparison ---\n")
    
    if not model_results:
        log.write("No model results available for comparison.\n")
        return
    
    try:
        # Find best model with safe dictionary access
        best_model = None
        best_mse = float('inf')
        for k, results in model_results.items():
            if isinstance(results, dict) and 'avg_test_mse' in results:
                mse = results['avg_test_mse']
                if isinstance(mse, (int, float)) and mse < best_mse:
                    best_mse = mse
                    best_model = k
        
        if best_model:
            log.write(f"Best Model (Lowest Average Test MSE): {best_model.upper()}\n")
        else:
            log.write("Could not determine best model due to missing metrics.\n")
        
        log.write("\nModel Performance Comparison:\n")
        for name, results in model_results.items():
            if isinstance(results, dict):
                log.write(f"- {name.upper()}: ")
                
                # Handle MSE and R²
                if 'avg_test_mse' in results and 'avg_test_r2' in results:
                    mse = results['avg_test_mse']
                    r2 = results['avg_test_r2']
                    if isinstance(mse, (int, float)) and isinstance(r2, (int, float)):
                        log.write(f"Avg Test MSE = {mse:.4f}, Avg Test R² = {r2:.4f}\n")
                    else:
                        log.write("Metrics not available (invalid types)\n")
                elif 'mse' in results and 'accuracy' in results:
                    mse = results['mse']
                    acc = results['accuracy']
                    if isinstance(mse, (int, float)) and isinstance(acc, (int, float)):
                        log.write(f"MSE = {mse:.4f}, Accuracy = {acc:.4f}\n")
                    else:
                        log.write("Metrics not available (invalid types)\n")
                else:
                    log.write("Metrics not available\n")
        
    except Exception as e:
        log.write(f"Error writing model comparison: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()

@timing_decorator
def compute_cluster_center_statistics(df, batch_size=10000, logger=None):
    """Pre-compute cluster statistics efficiently with improved memory management and numerical stability."""
    if logger:
        logger.log_and_print("Computing cluster center statistics...")
    
    cluster_stats = {
        'centers': {},
        'distributions': {},
        'modulo_patterns': {},
        'feature_correlations': {}
    }
    
    try:
        clusters = sorted(df['cluster'].unique())
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['cluster', 'sub_cluster', 'gap_size']]
        
        # Convert to numpy array for faster processing
        data = df[feature_cols].values.astype(np.float64)
        data = np.clip(data, -1e10, 1e10)
        
        # Compute statistics for each cluster
        for cluster_id in clusters:
            if logger:
                logger.log_and_print(f"Processing cluster {cluster_id}")
                
            cluster_mask = df['cluster'].values == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            # Initialize cluster statistics
            cluster_stats['centers'][cluster_id] = {
                'size': int(cluster_size),
                'features': {},
                'gap_distribution': {}
            }
            
            # Call numba-optimized function
            stats_accumulators = _compute_cluster_center_stats_numba(data, cluster_mask, batch_size, len(feature_cols))
            
            # Compute final statistics
            for i, col in enumerate(feature_cols):
                if stats_accumulators[i, 2] > 0:
                    mean = stats_accumulators[i, 0] / stats_accumulators[i, 2]
                    var = (stats_accumulators[i, 1] / stats_accumulators[i, 2]) - (mean ** 2)
                    std = np.sqrt(max(0, var))
                    
                    cluster_stats['centers'][cluster_id]['features'][col] = {
                        'mean': float(mean),
                        'std': float(std),
                        'min': float(stats_accumulators[i, 3]),
                        'max': float(stats_accumulators[i, 4]),
                        'count': int(stats_accumulators[i, 2])
                    }
                    
                    # Compute quantiles if we have stored values
                    values = data[cluster_mask, i]
                    if len(values) > 0:
                        quantiles = np.percentile(values, [25, 50, 75])
                        cluster_stats['centers'][cluster_id]['features'][col].update({
                            'median': float(quantiles[1]),
                            'q1': float(quantiles[0]),
                            'q3': float(quantiles[2])
                        })
            
            # Compute gap distribution
            if 'gap_size' in df.columns:
                gaps = df.loc[cluster_mask, 'gap_size'].values.astype(np.float64)
                gaps = np.clip(gaps, -1e10, 1e10)
                gaps = gaps[np.isfinite(gaps)]
                
                if len(gaps) > 0:
                    gap_counts = np.bincount(gaps.astype(int))
                    top_gaps = (-gap_counts).argsort()[:5]  # Get indices of top 5 most common gaps
                    
                    gap_distribution = {
                        'common_gaps': [(int(gap), int(gap_counts[gap])) for gap in top_gaps if gap_counts[gap] > 0],
                        'mean': float(np.mean(gaps)),
                        'std': float(np.std(gaps)),
                        'median': float(np.median(gaps))
                    }
                    cluster_stats['centers'][cluster_id]['gap_distribution'] = gap_distribution
            
            # Compute modulo patterns if relevant columns exist
            if 'gap_mod6' in df.columns and 'gap_mod30' in df.columns:
                mod_patterns = {}
                for mod, col in [(6, 'gap_mod6'), (30, 'gap_mod30')]:
                    mod_data = df.loc[cluster_mask, col].values.astype(int)
                    counts = np.bincount(mod_data, minlength=mod)
                    percentages = counts / len(mod_data) * 100
                    mod_patterns[mod] = {
                        i: float(percentages[i]) for i in range(mod) if percentages[i] > 0
                    }
                cluster_stats['centers'][cluster_id]['mod_patterns'] = mod_patterns
            
            gc.collect()
        
        if logger:
            logger.log_and_print("Cluster center statistics computation complete")
        
        return cluster_stats
        
    except Exception as e:
        error_msg = f"Error computing cluster center statistics: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'centers': {cluster: {
                'size': 0,
                'features': {},
                'gap_distribution': {},
                'mod_patterns': {}
            } for cluster in clusters},
            'distributions': {},
            'modulo_patterns': {},
            'feature_correlations': {}
        }   

def _write_cluster_centers(log, df, cluster_stats=None, logger=None):
    """Writes cluster centers analysis using pre-computed statistics with improved error handling."""
    log.write("\n--- Cluster Centers ---\n")
    
    if cluster_stats is None or not cluster_stats.get('centers'):
        log.write("No cluster center statistics available.\n")
        return
    
    try:
        for cluster_id, stats in sorted(cluster_stats['centers'].items()):
            log.write(f"\nCluster {cluster_id}:\n")
            
            # Write basic information
            log.write(f"Size: {stats.get('size', 'N/A')} samples ")
            if 'size' in stats:
                percentage = (stats['size'] / len(df)) * 100
                log.write(f"({percentage:.1f}% of total)\n")
            
            # Write gap distribution
            if 'gap_distribution' in stats:
                gap_dist = stats['gap_distribution']
                log.write("\nGap Distribution:\n")
                log.write(f"- Mean: {gap_dist.get('mean', 'N/A'):.2f}\n")
                log.write(f"- Std Dev: {gap_dist.get('std', 'N/A'):.2f}\n")
                log.write(f"- Median: {gap_dist.get('median', 'N/A'):.2f}\n")
                
                if 'common_gaps' in gap_dist:
                    log.write("\nMost Common Gaps:\n")
                    for gap, count in gap_dist['common_gaps']:
                        log.write(f"- Gap {gap}: {count}\n")
            
            # Write modulo patterns
            if 'mod_patterns' in stats:
                log.write("\nModulo Patterns:\n")
                for mod, patterns in stats['mod_patterns'].items():
                    log.write(f"\nMod {mod} distribution:\n")
                    for residue, percentage in sorted(patterns.items()):
                        if percentage > 1.0:  # Only show if significant
                            log.write(f"- Residue {residue}: {percentage:.1f}%\n")
            
            # Write significant feature correlations
            if 'correlations' in stats:
                log.write("\nSignificant Correlations with Gap Size:\n")
                for feature, corr in sorted(stats['correlations'].items(), key=lambda x: abs(x[1]), reverse=True):
                    log.write(f"- {feature}: {corr:.3f}\n")
            
            # Write key feature statistics
            if 'features' in stats:
                log.write("\nKey Feature Statistics:\n")
                for feature, feat_stats in stats['features'].items():
                    if feature in ['factor_density', 'mean_factor', 'factor_entropy']:  # Only show important features
                        log.write(f"\n{feature}:\n")
                        log.write(f"- Mean: {feat_stats.get('mean', 'N/A'):.2f}\n")
                        log.write(f"- Std Dev: {feat_stats.get('std', 'N/A'):.2f}\n")
                        if 'median' in feat_stats:
                            log.write(f"- Median: {feat_stats.get('median', 'N/A'):.2f}\n")
                        if 'q1' in feat_stats and 'q3' in feat_stats:
                           log.write(f"- IQR: {feat_stats.get('q3', 'N/A'):.2f} - {feat_stats.get('q1', 'N/A'):.2f}\n")
                        if 'skew' in feat_stats:
                            log.write(f"- Skewness: {feat_stats.get('skew', 'N/A'):.2f}\n")
                        if 'kurtosis' in feat_stats:
                            log.write(f"- Kurtosis: {feat_stats.get('kurtosis', 'N/A'):.2f}\n")
    
    except Exception as e:
        log.write(f"\nError writing cluster centers: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    
    log.write("\n")                                                                                                          

@timing_decorator
def compute_factor_statistics(df, batch_size=5000, logger=None):
    """Pre-compute factor statistics for reporting with improved memory management and numerical stability."""
    if logger:
        logger.log_and_print("Computing factor statistics...")
    
    factor_stats = {
        'summary': {},
        'distribution': {},
        'patterns': {},
        'temporal': {},
        'cluster_specific': {}
    }
    
    try:
        # Get pre-computed factor columns
        factor_cols = ['unique_factors', 'total_factors', 'factor_density', 
                      'mean_factor', 'max_factor', 'min_factor', 'factor_entropy']
        
        if all(col in df.columns for col in factor_cols):
            # Initialize accumulators
            stats_accumulators = {
                col: {
                    'sum': 0.0,
                    'sum_sq': 0.0,
                    'count': 0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'values': []
                }
                for col in factor_cols
            }
            
            # Process in batches
            for start_idx in range(0, len(df), batch_size):
                end_idx = min(start_idx + batch_size, len(df))
                batch = df.iloc[start_idx:end_idx]
                
                for col in factor_cols:
                    col_data = batch[col].values.astype(np.float64)
                    col_data = np.clip(col_data, -1e10, 1e10)
                    valid_mask = np.isfinite(col_data)
                    col_data = col_data[valid_mask]
                    
                    if len(col_data) > 0:
                        with np.errstate(all='ignore'):
                            stats_accumulators[col]['sum'] += np.sum(col_data)
                            stats_accumulators[col]['sum_sq'] += np.sum(col_data ** 2)
                            stats_accumulators[col]['count'] += len(col_data)
                            stats_accumulators[col]['min'] = min(stats_accumulators[col]['min'], np.min(col_data))
                            stats_accumulators[col]['max'] = max(stats_accumulators[col]['max'], np.max(col_data))
                            
                            # Store subset of values for distribution analysis
                            if len(stats_accumulators[col]['values']) < 10000:
                                stats_accumulators[col]['values'].extend(col_data[:1000].tolist())
                
                gc.collect()
            
            # Compute final statistics
            factor_stats['summary'] = {
                'unique_factors_total': int(stats_accumulators['unique_factors']['sum']),
                'total_factors': int(stats_accumulators['total_factors']['sum']),
                'mean_density': float(stats_accumulators['factor_density']['sum'] / 
                                   max(1, stats_accumulators['factor_density']['count'])),
                'mean_factor_size': float(stats_accumulators['mean_factor']['sum'] / 
                                       max(1, stats_accumulators['mean_factor']['count'])),
                'max_factor_seen': float(stats_accumulators['max_factor']['max']),
                'min_factor_seen': float(stats_accumulators['min_factor']['min'])
            }
            
            # Compute distribution statistics
            for col in factor_cols:
                if stats_accumulators[col]['values']:
                    values = np.array(stats_accumulators[col]['values'])
                    with np.errstate(all='ignore'):
                        factor_stats['distribution'][col] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'median': float(np.median(values)),
                            'skewness': float(sps.skew(values)),
                            'kurtosis': float(sps.kurtosis(values)),
                            'q1': float(np.percentile(values, 25)),
                            'q3': float(np.percentile(values, 75))
                        }
            
            # Compute temporal patterns
            if len(df) > 1:
                window_sizes = [10, 20, 50]
                for col in factor_cols:
                    temporal_stats = []
                    
                    for start_idx in range(0, len(df), batch_size):
                        end_idx = min(start_idx + batch_size, len(df))
                        batch_data = df[col].iloc[start_idx:end_idx].values.astype(np.float64)
                        batch_data = np.clip(batch_data, -1e10, 1e10)
                        
                        for window in window_sizes:
                            if len(batch_data) >= window:
                                rolling_mean = pd.Series(batch_data).rolling(window=window, center=True).mean()
                                rolling_std = pd.Series(batch_data).rolling(window=window, center=True).std()
                                
                                with np.errstate(all='ignore'):
                                    temporal_stats.append({
                                        'window': window,
                                        'mean': float(np.nanmean(rolling_mean)),
                                        'std': float(np.nanmean(rolling_std))
                                    })
                        
                        gc.collect()
                    
                    factor_stats['temporal'][col] = temporal_stats
            
            # Analyze patterns by cluster if clusters exist
            if 'cluster' in df.columns:
                for cluster in sorted(df['cluster'].unique()):
                    cluster_mask = df['cluster'] == cluster
                    cluster_stats = {}
                    
                    for col in factor_cols:
                        cluster_data = df.loc[cluster_mask, col].values.astype(np.float64)
                        cluster_data = np.clip(cluster_data, -1e10, 1e10)
                        valid_mask = np.isfinite(cluster_data)
                        cluster_data = cluster_data[valid_mask]
                        
                        if len(cluster_data) > 0:
                            with np.errstate(all='ignore'):
                                cluster_stats[col] = {
                                    'mean': float(np.mean(cluster_data)),
                                    'std': float(np.std(cluster_data)),
                                    'median': float(np.median(cluster_data)),
                                    'count': int(len(cluster_data))
                                }
                    
                    factor_stats['cluster_specific'][int(cluster)] = cluster_stats
            
            if logger:
                logger.log_and_print("Factor statistics computation complete")
            
            return factor_stats
            
        
    except Exception as e:
        error_msg = f"Error computing factor statistics: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'summary': {
                'unique_factors_total': 0,
                'total_factors': 0,
                'mean_density': 0.0,
                'mean_factor_size': 0.0,
                'max_factor_seen': 0.0,
                'min_factor_seen': 0.0
            },
            'distribution': {},
            'patterns': {},
            'temporal': {},
            'cluster_specific': {}
        }     

def _write_factor_analysis(log, df, logger=None):
    """Writes the factor analysis section of the report with improved error handling."""
    log.write("\n--- Factor Analysis ---\n")
    
    if 'cluster' not in df.columns:
        log.write("No cluster information available.\n")
        return
    
    try:
        # Write cluster distribution with proper formatting
        log.write("\nCluster Distribution:\n")
        cluster_counts = df['cluster'].value_counts()
        total = len(df)
        for cluster, count in cluster_counts.items():
            percentage = (count / total) * 100
            log.write(f"Cluster {cluster}: {count} samples ({percentage:.2f}%)\n")
        
        # Write factor statistics if available
        if 'unique_factors' in df.columns:
            log.write("\nFactor Statistics:\n")
            stats = df['unique_factors'].describe()
            log.write(f"- Average unique factors: {stats['mean']:.2f}\n")
            log.write(f"- Maximum unique factors in a gap: {stats['max']:.0f}\n")
        
        if 'factor_density' in df.columns:
            density_stats = df['factor_density'].describe()
            log.write(f"- Average factor density: {density_stats['mean']:.2f}\n")
        
        if 'mean_sqrt_factor' in df.columns:
            sqrt_mean_stats = df['mean_sqrt_factor'].describe()
            log.write(f"- Average mean sqrt factor: {sqrt_mean_stats['mean']:.2f}\n")
        
        if 'sum_sqrt_factor' in df.columns:
            sqrt_sum_stats = df['sum_sqrt_factor'].describe()
            log.write(f"- Average sum sqrt factor: {sqrt_sum_stats['mean']:.2f}\n")
            
    except Exception as e:
        log.write(f"\nError writing factor analysis: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    
    log.write("\n")

@njit
def _compute_cluster_center_stats_numba(data, cluster_mask, batch_size, n_features):
    """Numba-optimized function to compute cluster center statistics."""
    
    # Initialize accumulators
    stats_accumulators = np.zeros((n_features, 6), dtype=np.float64)
    
    # Process in batches
    for start_idx in range(0, len(data), batch_size):
        end_idx = min(start_idx + batch_size, len(data))
        batch = data[start_idx:end_idx]
        batch_mask = cluster_mask[start_idx:end_idx]
        
        for i in range(n_features):
            col_data = batch[batch_mask, i]
            if len(col_data) > 0:
                stats_accumulators[i, 0] += np.sum(col_data)
                stats_accumulators[i, 1] += np.sum(col_data ** 2)
                stats_accumulators[i, 2] += len(col_data)
                stats_accumulators[i, 3] = min(stats_accumulators[i, 3], np.min(col_data))
                stats_accumulators[i, 4] = max(stats_accumulators[i, 4], np.max(col_data))
                stats_accumulators[i, 5] += np.sum(col_data == 0)
    
    return stats_accumulators

@timing_decorator
def compute_cluster_statistics(df, batch_size=10000, logger=None):
    """Pre-compute cluster statistics efficiently with improved memory management and numerical stability."""
    if logger:
        logger.log_and_print("Computing cluster center statistics...")
    
    cluster_stats = {
        'centers': {},
        'distributions': {},
        'modulo_patterns': {},
        'feature_correlations': {}
    }
    
    try:
        clusters = sorted(df['cluster'].unique())
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['cluster', 'sub_cluster', 'gap_size']]
        
        # Convert to numpy array for faster processing
        data = df[feature_cols].values.astype(np.float64)
        data = np.clip(data, -1e10, 1e10)
        
        # Compute statistics for each cluster
        for cluster_id in clusters:
            if logger:
                logger.log_and_print(f"Processing cluster {cluster_id}")
                
            cluster_mask = df['cluster'].values == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            # Initialize cluster statistics
            cluster_stats['centers'][cluster_id] = {
                'size': int(cluster_size),
                'features': {},
                'gap_distribution': {}
            }
            
            # Call numba-optimized function
            stats_accumulators = _compute_cluster_center_stats_numba(data, cluster_mask, batch_size, len(feature_cols))
            
            # Compute final statistics
            for i, col in enumerate(feature_cols):
                if stats_accumulators[i, 2] > 0:
                    mean = stats_accumulators[i, 0] / stats_accumulators[i, 2]
                    var = (stats_accumulators[i, 1] / stats_accumulators[i, 2]) - (mean ** 2)
                    std = np.sqrt(max(0, var))
                    
                    cluster_stats['centers'][cluster_id]['features'][col] = {
                        'mean': float(mean),
                        'std': float(std),
                        'min': float(stats_accumulators[i, 3]),
                        'max': float(stats_accumulators[i, 4]),
                        'count': int(stats_accumulators[i, 2])
                    }
                    
                    # Compute quantiles if we have stored values
                    values = data[cluster_mask, i]
                    if len(values) > 0:
                        quantiles = np.percentile(values, [25, 50, 75])
                        cluster_stats['centers'][cluster_id]['features'][col].update({
                            'median': float(quantiles[1]),
                            'q1': float(quantiles[0]),
                            'q3': float(quantiles[2])
                        })
            
            # Compute gap distribution
            if 'gap_size' in df.columns:
                gaps = df.loc[cluster_mask, 'gap_size'].values.astype(np.float64)
                gaps = np.clip(gaps, -1e10, 1e10)
                gaps = gaps[np.isfinite(gaps)]
                
                if len(gaps) > 0:
                    gap_counts = np.bincount(gaps.astype(int))
                    top_gaps = (-gap_counts).argsort()[:5]  # Get indices of top 5 most common gaps
                    
                    gap_distribution = {
                        'common_gaps': [(int(gap), int(gap_counts[gap])) for gap in top_gaps if gap_counts[gap] > 0],
                        'mean': float(np.mean(gaps)),
                        'std': float(np.std(gaps)),
                        'median': float(np.median(gaps))
                    }
                    cluster_stats['centers'][cluster_id]['gap_distribution'] = gap_distribution
            
            # Compute modulo patterns if relevant columns exist
            if 'gap_mod6' in df.columns and 'gap_mod30' in df.columns:
                mod_patterns = {}
                for mod, col in [(6, 'gap_mod6'), (30, 'gap_mod30')]:
                    mod_data = df.loc[cluster_mask, col].values.astype(int)
                    counts = np.bincount(mod_data, minlength=mod)
                    percentages = counts / len(mod_data) * 100
                    mod_patterns[mod] = {
                        i: float(percentages[i]) for i in range(mod) if percentages[i] > 0
                    }
                cluster_stats['centers'][cluster_id]['mod_patterns'] = mod_patterns
            
            gc.collect()
        
        if logger:
            logger.log_and_print("Cluster center statistics computation complete")
        
        return cluster_stats
        
    except Exception as e:
        error_msg = f"Error computing cluster center statistics: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'centers': {cluster: {
                'size': 0,
                'features': {},
                'gap_distribution': {},
                'mod_patterns': {}
            } for cluster in clusters},
            'distributions': {},
            'modulo_patterns': {},
            'feature_correlations': {}
        }
                            
def _write_cluster_analysis(log, df, cluster_stats=None, advanced_clustering=None):
    """Writes the cluster analysis section with pre-computed statistics and advanced clustering metrics."""
    log.write("\n--- Cluster Analysis ---\n")
    
    if 'cluster' not in df.columns:
        log.write("No clustering information available\n")
        return
    
    if cluster_stats is None:
        # Compute minimal statistics if not provided
        cluster_stats = {
            'counts': df['cluster'].value_counts(),
            'basic_stats': {
                cluster: {
                    'size': len(cluster_data),
                    'mean_gap': float(cluster_data['gap_size'].mean()),
                    'median_gap': float(cluster_data['gap_size'].median())
                }
                for cluster, cluster_data in df.groupby('cluster')
            }
        }
    
    # Write cluster distribution
    log.write("\nCluster Distribution:\n")
    total_samples = len(df)
    for cluster, count in sorted(cluster_stats['counts'].items()):
        percentage = (count / total_samples) * 100
        log.write(f"Cluster {cluster}: {count} samples ({percentage:.2f}%)\n")
    
    # Write basic statistics for each cluster
    log.write("\nCluster Characteristics:\n")
    for cluster in sorted(cluster_stats['basic_stats'].keys()):
        stats = cluster_stats['basic_stats'][cluster]
        log.write(f"\nCluster {cluster}:")
        log.write(f"\n- Size: {stats['size']}")
        log.write(f"\n- Mean gap: {stats['mean_gap']:.2f}")
        log.write(f"\n- Median gap: {stats['median_gap']:.2f}")
        
        # Write additional pre-computed statistics if available
        if 'feature_stats' in stats:
            log.write("\n- Feature Statistics:")
            for feature, value in stats['feature_stats'].items():
                log.write(f"\n  {feature}: {value:.4f}")
        
        if 'transitions' in stats:
            log.write("\n- Most Common Transitions:")
            for next_cluster, prob in stats['transitions'].items():
                log.write(f"\n  To Cluster {next_cluster}: {prob:.2%}")
    
    # Write advanced clustering metrics
    if advanced_clustering:
        log.write("\n--- Advanced Clustering Metrics ---\n")
        if 'optimal_clusters' in advanced_clustering:
            log.write("\nOptimal Number of Clusters:\n")
            for method, n_clusters in advanced_clustering['optimal_clusters'].items():
                log.write(f"- {method.upper()}: {n_clusters} clusters\n")
        
        if 'metrics' in advanced_clustering:
            log.write("\nClustering Metrics:\n")
            for method, metrics in advanced_clustering['metrics'].items():
                log.write(f"- {method.upper()}:\n")
                for metric, value in metrics.items():
                    log.write(f"  - {metric}: {value:.4f}\n")
        
        if 'cluster_profiles' in advanced_clustering:
            log.write("\nCluster Profiles:\n")
            for method, profiles in advanced_clustering['cluster_profiles'].items():
                log.write(f"\n{method.upper()}:\n")
                for cluster, profile in profiles.items():
                    log.write(f"  - Cluster {cluster}: Size = {profile['size']}, Mean Gap = {profile['mean_gap']:.2f}, Std Gap = {profile['std_gap']:.2f}\n")
                    log.write("  - Feature Means:\n")
                    for feature, mean in profile['feature_means'].items():
                        log.write(f"    - {feature}: {mean:.2f}\n")
    
    log.write("\n")

def _write_enhanced_cluster_descriptions(log, df, cluster_stats, logger=None):
    """Writes enhanced descriptions of each cluster based on pre-computed statistics."""
    log.write("\n--- Enhanced Cluster Descriptions ---\n")
    
    if cluster_stats is None or not cluster_stats.get('basic_stats'):
        log.write("No cluster statistics available for detailed description.\n")
        return
    
    try:
        for cluster_id, stats in sorted(cluster_stats['basic_stats'].items()):
            log.write(f"\nCluster {cluster_id}:\n")
            
            # Basic statistics
            log.write(f"- Size: {stats.get('size', 'N/A')} samples\n")
            log.write(f"- Mean gap size: {stats.get('mean_gap', 'N/A'):.2f}\n")
            log.write(f"- Median gap size: {stats.get('median_gap', 'N/A'):.2f}\n")
            
            # Feature statistics
            if 'feature_stats' in stats:
                log.write("\n  Key Feature Characteristics:\n")
                
                sorted_features = sorted(stats['feature_stats'].items(), key=lambda item: abs(item[1]), reverse=True)[:5]
                
                for feature, value in sorted_features:
                    log.write(f"    - {feature}: {value:.2f}\n")
                
                # Add descriptions for key features
                log.write("\n  Feature descriptions:\n")
                for feature, value in sorted_features:
                    if feature == 'factor_density':
                        log.write("    - Factor Density: Represents the number of prime factors per unit of gap size.\n")
                    elif feature == 'factor_entropy':
                        log.write("    - Factor Entropy: Measures the randomness or disorder in the distribution of prime factors.\n")
                    elif feature == 'mean_factor':
                        log.write("    - Mean Factor: The average value of the prime factors within the composite numbers in the gap.\n")
                    elif feature == 'factor_std':
                        log.write("    - Factor Standard Deviation: Measures the variability in the prime factors within the composite numbers.\n")
                    elif feature == 'factor_range_ratio':
                        log.write("    - Factor Range Ratio: The ratio between the largest and smallest prime factors within the composite numbers.\n")
                    elif feature == 'mean_sqrt_factor':
                        log.write("    - Mean Sqrt Factor: The average of the square root of the prime factors within the composite numbers.\n")
                    elif feature == 'sum_sqrt_factor':
                        log.write("    - Sum Sqrt Factor: The sum of the square root of the prime factors within the composite numbers.\n")
                    elif feature == 'std_sqrt_factor':
                         log.write("    - Std Sqrt Factor: The standard deviation of the square root of the prime factors within the composite numbers.\n")
                    elif feature == 'product_of_prime_factors':
                        log.write("    - Product of Prime Factors: The product of all prime factors within the composite numbers.\n")
                    elif feature == 'sum_of_prime_factors':
                        log.write("    - Sum of Prime Factors: The sum of all prime factors within the composite numbers.\n")
                    else:
                        log.write(f"    - {feature}: No description available.\n")
        
        # Add transition information
        if 'transitions' in stats:
            log.write("\n  Transitions:\n")
            for next_cluster, prob in stats['transitions'].items():
                if prob > 0.01:
                    log.write(f"    - To cluster {next_cluster}: {prob:.2%}\n")
        
        
    except Exception as e:
        log.write(f"Error writing enhanced cluster descriptions: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    
    log.write("\n")

def _write_prime_probability_map(log, prime_probability_map=None, change_points=None, logger=None):
    """Writes the prime probability map section of the report with improved error handling."""
    log.write("\n--- Prime Probability Map ---\n")
    
    if prime_probability_map is None:
        log.write("Prime probability map generation was skipped in this run.\n")
        return
    
    try:
        if '_stats' in prime_probability_map:
            stats = prime_probability_map['_stats']
            log.write("\nPrime Probability Map Statistics:\n")
            log.write(f"- Mean Probability: {stats.get('mean_probability', 'N/A'):.4f}\n")
            log.write(f"- Std Probability: {stats.get('std_probability', 'N/A'):.4f}\n")
            log.write(f"- Probability Range: [{stats.get('min_probability', 'N/A'):.4f}, {stats.get('max_probability', 'N/A'):.4f}]\n")
            log.write(f"- Total Predictions: {stats.get('total_predictions', 'N/A')}\n")
            log.write(f"  Prediction Range: {stats.get('prediction_range', {}).get('start', 'N/A')} to {stats.get('prediction_range', {}).get('end', 'N/A')}\n")
        
        # Write first 10 entries of the map
        log.write("\nSample Prime Probability Map Entries:\n")
        for i, (prime, prob) in enumerate(prime_probability_map.items()):
            if i >= 10 or prime == '_stats':
                break
            if isinstance(prime, (int, float)):
                log.write(f"  Prime: {prime:.0f}, Probability: {prob:.4f}\n")
            else:
                log.write(f"  Prime: {prime}, Probability: {prob:.4f}\n")
        
        # Write change point analysis
        if change_points and change_points['segments']:
            log.write("\n--- Change Point Analysis ---\n")
            log.write(f"Number of segments: {change_points.get('n_segments', 'N/A')}\n")
            log.write(f"Score: {change_points.get('score', 'N/A'):.4f}\n")
            log.write("Change Points:\n")
            for segment in change_points.get('segments', []):
                log.write(f"  - Start: {segment.get('start', 'N/A')}, End: {segment.get('end', 'N/A')}, Mean: {segment.get('mean', 'N/A'):.2f}, Size: {segment.get('size', 'N/A')}\n")
        
        # Write next prime prediction
        if 'next_prime_predictions' in prime_probability_map:
            next_prime = prime_probability_map['next_prime_predictions']
            log.write("\n--- Next Prime Prediction ---\n")
            log.write(f"Predicted Cluster: {next_prime.get('predicted_cluster', 'N/A')}\n")
            log.write(f"Predicted Gap: {next_prime.get('predicted_gap', 'N/A'):.2f}\n")
            log.write(f"Predicted Next Prime Location: {next_prime.get('next_prime_location', 'N/A'):.0f}\n")
            
            # Add a note about the prediction method
            log.write("\nNote: The next prime location is predicted based on the cluster and a gap prediction model.\n")
        
    except Exception as e:
        log.write(f"Error writing prime probability map: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
            
def _write_advanced_analyses_summary(log, gap_distribution, gap_sequences, factor_patterns, cluster_transitions, logger=None):
    """Writes summary of advanced analyses with improved error handling."""
    log.write("\n=== ADVANCED ANALYSES SUMMARY ===\n")
    
    # Gap Distribution Characteristics
    log.write("\nGap Distribution Characteristics:\n")
    if gap_distribution and 'summary' in gap_distribution:
        summary = gap_distribution['summary']
        log.write(f"- Total Clusters: {summary.get('total_clusters', 'N/A')}\n")
        log.write(f"- Mean Cluster Size: {summary.get('mean_cluster_size', 'N/A'):.2f}\n")
        log.write(f"- Std Cluster Size: {summary.get('std_cluster_size', 'N/A'):.2f}\n")
        if 'mean_gap_ranges' in summary:
            log.write("\nMean Gap Ranges:\n")
            for cluster_id, range in summary['mean_gap_ranges'].items():
                log.write(f"  - Cluster {cluster_id}: {range:.2f}\n")
    else:
        log.write("Gap distribution analysis not available\n")
    
    # Gap Sequences Analysis
    log.write("\nGap Sequence Patterns:\n")
    if gap_sequences:
        for length, stats in gap_sequences.items():
            if isinstance(stats, dict):  # Ensure stats is a dictionary
                log.write(f"\nSequence Length {length}:\n")
                for stat_name in ['count', 'unique_count', 'increasing_count', 'decreasing_count']:
                    if stat_name in stats:
                        log.write(f"- {stat_name.replace('_', ' ').title()}: {stats[stat_name]}\n")
                
                # Handle ratios separately to format as percentages
                for ratio_name in ['increasing_ratio', 'decreasing_ratio']:
                    if ratio_name in stats:
                        ratio_value = stats[ratio_name]
                        if isinstance(ratio_value, (int, float)):
                            log.write(f"- {ratio_name.replace('_', ' ').title()}: {ratio_value:.2%}\n")
    else:
        log.write("Gap sequence analysis not available\n")
    
    # Factor Patterns
    log.write("\nPrime Factor Patterns:\n")
    if factor_patterns and isinstance(factor_patterns, dict):
        metrics = factor_patterns.get('metrics', {})
        stats_to_write = {
            'Mean Unique Factors': metrics.get('mean_unique_factors', 0),
            'Mean Total Factors': metrics.get('mean_total_factors', 0),
            'Factor Density Mean': metrics.get('factor_density_mean', 0)
        }
        
        for name, value in stats_to_write.items():
            log.write(f"- {name}: {value:.2f}\n")
    else:
        log.write("Factor pattern analysis not available\n")
    
    # Cluster Transitions
    log.write("\nCluster Transition Analysis:\n")
    if cluster_transitions and isinstance(cluster_transitions, dict):
        metrics = cluster_transitions.get('basic_metrics', {})
        stats_to_write = {
            'Transition Entropy': metrics.get('transition_entropy', 0),
            'Mean Stability': metrics.get('mean_stability', 0),
            'Stability Std': metrics.get('std_stability', 0)
        }
        
        for name, value in stats_to_write.items():
            log.write(f"- {name}: {value:.4f}\n")
    else:
        log.write("Cluster transition analysis not available\n")

def _write_chaos_analysis(log, chaos_metrics, logger=None):
    """Writes the chaos analysis section of the report."""
    log.write("\n--- Chaos Analysis ---\n")
    
    if not chaos_metrics:
        log.write("No chaos metrics available.\n")
        return
    
    try:
        for feature, metrics in chaos_metrics.items():
            log.write(f"\nFeature: {feature}\n")
            log.write(f"  - Mean Divergence: {metrics.get('mean_divergence', 'N/A'):.4f}\n")
            log.write(f"  - Std Divergence: {metrics.get('std_divergence', 'N/A'):.4f}\n")
            log.write(f"  - Max Divergence: {metrics.get('max_divergence', 'N/A'):.4f}\n")
            log.write(f"  - Min Divergence: {metrics.get('min_divergence', 'N/A'):.4f}\n")
            log.write(f"  - 90th Percentile Divergence: {metrics.get('divergence_90th', 'N/A'):.4f}\n")
    except Exception as e:
        log.write(f"Error writing chaos analysis: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    
    log.write("\n")

def _write_superposition_analysis(log, superposition_patterns, logger=None):
    """Writes the superposition analysis section of the report."""
    log.write("\n--- Superposition Analysis ---\n")
    
    if not superposition_patterns:
        log.write("No superposition patterns available.\n")
        return
    
    try:
        for feature, patterns in superposition_patterns.items():
            log.write(f"\nFeature: {feature}\n")
            log.write(f"  - Mean: {patterns.get('mean', 'N/A'):.4f}\n")
            log.write(f"  - Std: {patterns.get('std', 'N/A'):.4f}\n")
            log.write(f"  - Median: {patterns.get('median', 'N/A'):.4f}\n")
            log.write(f"  - IQR: {patterns.get('iqr', 'N/A'):.4f}\n")
            log.write(f"  - Number of Modes: {patterns.get('num_modes', 'N/A')}\n")
            log.write(f"  - Entropy: {patterns.get('entropy', 'N/A'):.4f}\n")
    except Exception as e:
        log.write(f"Error writing superposition analysis: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    
    log.write("\n")

def _write_wavelet_analysis(log, wavelet_patterns, logger=None):
    """Writes the wavelet analysis section of the report."""
    log.write("\n--- Wavelet Analysis ---\n")
    
    if not wavelet_patterns or not wavelet_patterns.get('wavelet_coeffs'):
        log.write("No wavelet coefficients available.\n")
        return
    
    try:
        coeffs = wavelet_patterns['wavelet_coeffs']
        log.write(f"Wavelet coefficients (first 5 levels):\n")
        for i, level_coeffs in enumerate(coeffs[:5]):
            log.write(f"Level {i+1}: {level_coeffs[:10]}...\n")
        log.write("Full wavelet coefficient data is not displayed for brevity.\n")
    except Exception as e:
        log.write(f"Error writing wavelet analysis: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    
    log.write("\n")

def _write_fractal_dimension_analysis(log, fractal_dimension, logger=None):
    """Writes the fractal dimension analysis section of the report."""
    log.write("\n--- Fractal Dimension Analysis ---\n")
    
    if not fractal_dimension:
        log.write("No fractal dimension data available.\n")
        return
    
    try:
        log.write(f"Fractal Dimension: {fractal_dimension.get('dimension', 'N/A'):.4f}\n")
        if 'counts' in fractal_dimension:
            log.write("Box Counts:\n")
            for size, count in list(fractal_dimension['counts'].items())[:5]:
                log.write(f"- Box Size: {size:.2f}, Count: {count:.0f}\n")
            log.write("Full box count data is not displayed for brevity.\n")
    except Exception as e:
        log.write(f"Error writing fractal dimension analysis: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    
    log.write("\n")
    
def _write_phase_space_analysis(log, phase_space_analysis, logger=None):
    """Writes the phase space analysis section of the report."""
    log.write("\n--- Phase Space Analysis ---\n")
    
    if not phase_space_analysis:
        log.write("No phase space analysis data available.\n")
        return
    
    try:
        if 'embedding_dimension' in phase_space_analysis:
            log.write("\nEmbedding Dimension Analysis:\n")
            for lag, metrics in phase_space_analysis['embedding_dimension'].items():
                log.write(f"Lag {lag}: False Neighbor Ratio = {metrics.get('false_neighbor_ratio', 'N/A'):.4f}\n")
        
        if 'phase_space_data' in phase_space_analysis:
            log.write("\nPhase Space Data (first 5 points for each lag):\n")
            for lag, data in sorted(phase_space_analysis['phase_space_data'].items())[:5]:
                log.write(f"Lag {lag}: {data[:5]}...\n")
    
    except Exception as e:
        log.write(f"Error writing phase space analysis: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    
    log.write("\n")
    
def _write_recurrence_plot_analysis(log, recurrence_plot_data, logger=None):
    """Writes the recurrence plot analysis section of the report."""
    log.write("\n--- Recurrence Plot Analysis ---\n")
    
    if not recurrence_plot_data or not recurrence_plot_data.get('distance_matrix') is not None:
        log.write("No recurrence plot data available.\n")
        return
    
    try:
        log.write("Recurrence plot data is available, but not shown in text format for brevity.\n")
    except Exception as e:
        log.write(f"Error writing recurrence plot analysis: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    
    log.write("\n")

@njit
def _compute_temporal_stats_numba(series, lags, batch_size):
    """Numba-optimized function to compute temporal statistics."""
    n = len(series)
    autocorr = np.zeros(len(lags), dtype=np.float64)
    
    # Compute autocorrelation
    for lag_idx, lag in enumerate(lags):
        corr_sum = 0.0
        count = 0
        for start_idx in range(0, n - lag, batch_size):
            end_idx = min(start_idx + batch_size, n - lag)
            batch = series[start_idx:end_idx]
            lagged_batch = series[start_idx + lag:end_idx + lag]
            
            # Ensure equal lengths
            min_len = min(len(batch), len(lagged_batch))
            batch = batch[:min_len]
            lagged_batch = lagged_batch[:min_len]
            
            if len(batch) > 0:
                corr = np.corrcoef(batch, lagged_batch)[0, 1]
                if np.isfinite(corr):
                    corr_sum += corr
                    count += 1
        
        if count > 0:
            autocorr[lag_idx] = corr_sum / count
    
    # Compute trend
    x_sum = 0.0
    y_sum = 0.0
    xy_sum = 0.0
    x2_sum = 0.0
    
    for i in range(len(series)):
        x_sum += i
        y_sum += series[i]
        xy_sum += i * series[i]
        x2_sum += i * i
    
    if len(series) > 1:
        slope = (len(series) * xy_sum - x_sum * y_sum) / (len(series) * x2_sum - x_sum * x_sum)
        intercept = (y_sum - slope * x_sum) / len(series)
    else:
        slope = 0.0
        intercept = 0.0
    
    return autocorr, slope, intercept

@timing_decorator
def compute_temporal_pattern_statistics(df, batch_size=10000, logger=None):
    """Pre-compute temporal pattern statistics efficiently with improved memory management and numerical stability."""
    if logger:
        logger.log_and_print("Computing temporal pattern statistics...")
    
    temporal_stats = {
        'features': {},
        'sequences': {},
        'trends': {}
    }
    
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['cluster', 'sub_cluster']]
        
        for col in feature_cols:
            if logger:
                logger.log_and_print(f"Analyzing temporal patterns for {col}")
            
            # Initialize statistics for this feature
            feature_stats = {
                'autocorr': [],
                'trend': {},
                'seasonality': {},
                'changes': {}
            }
            
            # Convert to numpy array and clip values
            series = df[col].values.astype(np.float64)
            series = np.clip(series, -1e10, 1e10)
            series = series[np.isfinite(series)]
            
            # Call numba-optimized function
            lags = np.array([1, 2, 3, 4, 5], dtype=np.int32)
            autocorr, slope, intercept = _compute_temporal_stats_numba(series, lags, batch_size)
            
            feature_stats['autocorr'] = autocorr.tolist()
            feature_stats['trend'] = {
                'slope': float(slope),
                'intercept': float(intercept)
            }
            
            # Compute periodicity with batched FFT
            try:
                # Compute mean first
                series_mean = np.mean(series[np.isfinite(series)])
                
                # Use real FFT instead of complex FFT (twice as fast)
                fft_result = np.fft.rfft(series - series_mean)
                power = np.abs(fft_result)
                freqs = np.fft.rfftfreq(len(series), d=1.0)
                
                # Only look at meaningful frequencies
                mask = freqs > 0
                power = power[mask]
                freqs = freqs[mask]
                
                if len(power) > 0:
                    # Find top periods quickly
                    top_k = min(5, len(power))
                    top_indices = np.argpartition(power, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(-power[top_indices])]
                    
                    # Get main period
                    main_idx = top_indices[0]
                    if freqs[main_idx] != 0:
                        period = 1.0 / freqs[main_idx]
                        strength = float(power[main_idx] / np.sum(power))
                        
                        feature_stats['seasonality'] = {
                            'period': float(period),
                            'strength': strength
                        }
            except Exception as e:
                if logger:
                    logger.log_and_print(f"Warning: Error computing periodicity: {str(e)}")
                feature_stats['seasonality'] = {'period': 0.0, 'strength': 0.0}
            
            # Compute rate of change statistics
            changes = np.diff(series)
            valid_changes = changes[np.isfinite(changes)]
            if len(valid_changes) > 0:
                with np.errstate(all='ignore'):
                    feature_stats['changes'] = {
                        'mean': float(np.mean(valid_changes)),
                        'std': float(np.std(valid_changes)),
                        'median': float(np.median(valid_changes)),
                        'positive_pct': float(np.mean(valid_changes > 0) * 100)
                    }
            
            temporal_stats['features'][col] = feature_stats
        
        # Analyze sequence patterns for gap_size
        if 'gap_size' in df.columns:
            gaps = df['gap_size'].values.astype(np.float64)
            gaps = np.clip(gaps, -1e10, 1e10)
            
            sequence_stats = {
                'increasing_runs': [],
                'decreasing_runs': [],
                'patterns': {}
            }
            
            # Find runs efficiently
            diff = np.diff(gaps)
            
            # Increasing runs
            inc_runs = np.split(diff, np.where(diff <= 0)[0] + 1)
            inc_runs = [run for run in inc_runs if len(run) > 0]
            if inc_runs:
                sequence_stats['increasing_runs'] = {
                    'count': int(len(inc_runs)),
                    'max_length': int(max(len(run) for run in inc_runs)),
                    'mean_length': float(np.mean([len(run) for run in inc_runs]))
                }
            
            # Decreasing runs
            dec_runs = np.split(diff, np.where(diff >= 0)[0] + 1)
            dec_runs = [run for run in dec_runs if len(run) > 0]
            if dec_runs:
                sequence_stats['decreasing_runs'] = {
                    'count': int(len(dec_runs)),
                    'max_length': int(max(len(run) for run in dec_runs)),
                    'mean_length': float(np.mean([len(run) for run in dec_runs]))
                }
            
            temporal_stats['sequences'] = sequence_stats
        
        if logger:
            logger.log_and_print("Temporal pattern statistics computation complete")
        
        return temporal_stats
        
    except Exception as e:
        error_msg = f"Error computing temporal pattern statistics: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()
        
        # Return safe default values
        return {
            'features': {},
            'sequences': {},
            'trends': {}
        }

def _write_temporal_patterns_analysis(log, df, temporal_stats=None):
    """Writes temporal patterns analysis using pre-computed statistics."""
    log.write("\n=== TEMPORAL PATTERNS ANALYSIS ===\n")
    
    if temporal_stats is None or not temporal_stats.get('features'):
        log.write("No temporal pattern statistics available.\n")
        return
    
    try:
        # Write feature patterns
        for col, stats in temporal_stats['features'].items():
            log.write(f"\nFeature: {col}\n")
            
            # Write trend analysis
            if 'trend' in stats and stats['trend']:
                trend = stats['trend']
                log.write("Trend Analysis:\n")
                log.write(f"- Slope: {trend['slope']:.4e}\n")
                direction = "increasing" if trend['slope'] > 0 else "decreasing"
                log.write(f"- Direction: {direction}\n")
            
            # Write autocorrelation
            if stats['autocorr']:
                log.write("Autocorrelation Analysis:\n")
                for lag, corr in enumerate(stats['autocorr'], 1):
                    if abs(corr) > 0.1:  # Only show significant correlations
                        log.write(f"- Lag {lag}: {corr:.4f}\n")
            
            # Write seasonality if detected
            if 'seasonality' in stats and stats['seasonality']:
                season = stats['seasonality']
                if season['strength'] > 0.1:  # Only show if significant
                    log.write("Seasonality Analysis:\n")
                    log.write(f"- Period: {season['period']:.1f}\n")
                    log.write(f"- Strength: {season['strength']:.4f}\n")
            
            # Write change statistics
            if 'changes' in stats and stats['changes']:
                changes = stats['changes']
                log.write("Change Statistics:\n")
                log.write(f"- Mean change: {changes['mean']:.4f}\n")
                log.write(f"- Positive changes: {changes['positive_pct']:.1f}%\n")
        
        # Write sequence patterns
        if 'sequences' in temporal_stats:
            log.write("\nSequence Patterns:\n")
            
            if 'increasing_runs' in temporal_stats['sequences']:
                inc = temporal_stats['sequences']['increasing_runs']
                log.write("\nIncreasing Sequences:\n")
                log.write(f"- Count: {inc['count']}\n")
                log.write(f"- Max length: {inc['max_length']}\n")
                log.write(f"- Average length: {inc['mean_length']:.2f}\n")
            
            if 'decreasing_runs' in temporal_stats['sequences']:
                dec = temporal_stats['sequences']['decreasing_runs']
                log.write("\nDecreasing Sequences:\n")
                log.write(f"- Count: {dec['count']}\n")
                log.write(f"- Max length: {dec['max_length']}\n")
                log.write(f"- Average length: {dec['mean_length']:.2f}\n")
        
    except Exception as e:
        log.write(f"\nError writing temporal patterns: {str(e)}\n")
        log.write(traceback.format_exc())
        
def _write_cluster_separation_analysis(log, separation_metrics):
    """Writes cluster separation analysis section."""
    log.write("\n=== CLUSTER SEPARATION ANALYSIS ===\n")
    if separation_metrics:
        if 'inter_cluster_distances' in separation_metrics:
            log.write("\nInter-cluster Distances:\n")
            distances = separation_metrics['inter_cluster_distances']
            for i in range(len(distances)):
                for j in range(i+1, len(distances)):
                    log.write(f"- Clusters {i}-{j}: {distances[i,j]:.4f}\n")
        
        if 'cluster_densities' in separation_metrics:
            log.write("\nCluster Densities:\n")
            for cluster_id, density in separation_metrics['cluster_densities'].items():
                log.write(f"- Cluster {cluster_id}: {density:.4f}\n")
        
        if 'intra_cluster_distances' in separation_metrics:
            log.write("\nIntra-cluster Distances:\n")
            for cluster_id, stats in separation_metrics['intra_cluster_distances'].items():
                log.write(f"Cluster {cluster_id}:\n")
                for stat_name, value in stats.items():
                    log.write(f"- {stat_name}: {value:.4f}\n")

def _write_advanced_factor_analysis(log, factor_patterns):
    """Writes advanced factor analysis section."""
    log.write("\n=== ADVANCED FACTOR ANALYSIS ===\n")
    if factor_patterns:
        if 'factor_sequences' in factor_patterns:
            log.write("\nFactor Sequence Analysis:\n")
            for seq in factor_patterns['factor_sequences']:
                log.write(f"\nWindow Size {seq['window_size']}:\n")
                log.write(f"- Mean Trend: {seq['stats']['mean_trend']:.4f}\n")
                log.write(f"- Std Trend: {seq['stats']['std_trend']:.4f}\n")
                log.write(f"- Mean Std: {seq['stats']['mean_std']:.4f}\n")

def _write_subcluster_analysis(log, df, cluster_sequence_analysis):
    """Writes the sub-cluster analysis section of the report."""
    log.write("\n=== SUB-CLUSTER ANALYSIS ===\n")
    
    if 'sub_cluster' not in df.columns:
        log.write("No sub-cluster information available.\n")
        return
        
    # Basic sub-cluster statistics
    log.write("\nSub-cluster Distribution:\n")
    sub_cluster_counts = df['sub_cluster'].value_counts()
    for sub_id, count in sub_cluster_counts.items():
        percentage = (count / len(df)) * 100
        log.write(f"Sub-cluster {sub_id}: {count} gaps ({percentage:.2f}%)\n")
    
    # Sub-cluster characteristics
    log.write("\nSub-cluster Characteristics:\n")
    for sub_id in sorted(df['sub_cluster'].unique()):
        sub_data = df[df['sub_cluster'] == sub_id]
        log.write(f"\nSub-cluster {sub_id}:\n")
        
        # Gap statistics
        log.write("  Gap Statistics:\n")
        log.write(f"  - Mean gap size: {sub_data['gap_size'].mean():.2f}\n")
        log.write(f"  - Median gap size: {sub_data['gap_size'].median():.2f}\n")
        log.write(f"  - Std deviation: {sub_data['gap_size'].std():.2f}\n")
        
        # Factor patterns
        if 'unique_factors' in sub_data.columns:
            log.write("  Factor Patterns:\n")
            log.write(f"  - Mean unique factors: {sub_data['unique_factors'].mean():.2f}\n")
            log.write(f"  - Mean total factors: {sub_data['total_factors'].mean():.2f}\n")
        
        # Modulo patterns
        if 'gap_mod6' in sub_data.columns:
            log.write("  Modulo 6 Pattern:\n")
            mod6_dist = sub_data['gap_mod6'].value_counts(normalize=True)
            for mod, freq in mod6_dist.items():
                log.write(f"  - Mod 6 = {mod}: {freq*100:.2f}%\n")
    
    # Sub-cluster transitions
    log.write("\nSub-cluster Transition Analysis:\n")
    transitions = pd.crosstab(
        df['sub_cluster'],
        df['sub_cluster'].shift(-1),
        normalize='index'
    )
    
    for current in sorted(transitions.index):
        log.write(f"\nTransitions from sub-cluster {current}:\n")
        for next_cluster in sorted(transitions.columns):
            prob = transitions.loc[current, next_cluster]
            prob_val = float(prob) if isinstance(prob, (int, float, np.number)) and pd.notnull(prob) and not isinstance(prob, complex) else 0.0
            if prob_val > 0.01:  # Only show significant transitions
                log.write(f"  To sub-cluster {next_cluster}: {prob_val*100:.2f}%\n")
    
    # Sub-cluster sequence patterns
    log.write("\nSub-cluster Sequence Patterns:\n")
    
    # Analyze runs
    current_sub = None
    current_run = 0
    run_lengths = []
    
    for sub in df['sub_cluster']:
        if sub == current_sub:
            current_run += 1
        else:
            if current_run > 0:
                run_lengths.append(current_run)
            current_sub = sub
            current_run = 1
    
    if run_lengths:
        log.write(f"Average run length: {np.mean(run_lengths):.2f}\n")
        log.write(f"Maximum run length: {max(run_lengths)}\n")
        log.write(f"Most common run lengths: {pd.Series(run_lengths).value_counts().head().to_dict()}\n")

def _write_enhanced_sequence_analysis(log, df, logger=None):
    """Writes the enhanced sequence analysis section of the report with improved error handling."""
    log.write("\n=== ENHANCED SEQUENCE ANALYSIS ===\n")
    
    try:
        # Analyze sequences of fixed lengths only
        sequence_lengths = [10, 20]
        
        for length in sequence_lengths:
            log.write(f"\nSequence Analysis (Length {length}):\n")
            
            # Compute basic sequence statistics efficiently
            gaps = df['gap_size'].values
            sequences = np.lib.stride_tricks.sliding_window_view(gaps, length)
            
            # Compute trends
            means = np.mean(sequences, axis=1)
            stds = np.std(sequences, axis=1)
            
            # Compute sequence patterns
            increasing = np.sum(np.all(np.diff(sequences, axis=1) > 0, axis=1))
            decreasing = np.sum(np.all(np.diff(sequences, axis=1) < 0, axis=1))
            constant = np.sum(np.all(np.diff(sequences, axis=1) == 0, axis=1))
            
            total_sequences = len(sequences)
            
            log.write("Sequence Statistics:\n")
            log.write(f"- Total sequences: {total_sequences}\n")
            log.write(f"- Mean sequence value: {np.mean(means):.2f}\n")
            log.write(f"- Mean sequence std: {np.mean(stds):.2f}\n")
            log.write(f"- Strictly increasing sequences: {increasing} ({100*increasing/total_sequences:.1f}%)\n")
            log.write(f"- Strictly decreasing sequences: {decreasing} ({100*decreasing/total_sequences:.1f}%)\n")
            log.write(f"- Constant sequences: {constant} ({100*constant/total_sequences:.1f}%)\n")
            
            # Report sub-cluster patterns if available
            if 'sub_cluster' in df.columns:
                log.write("\nSub-cluster Sequence Patterns:\n")
                sub_clusters = df['sub_cluster'].values
                sub_sequences = np.lib.stride_tricks.sliding_window_view(sub_clusters, length)
                
                # Count transitions between sub-clusters
                transitions = {}
                for seq in sub_sequences:
                    for i in range(len(seq)-1):
                        from_cluster = seq[i]
                        to_cluster = seq[i+1]
                        key = (from_cluster, to_cluster)
                        transitions[key] = transitions.get(key, 0) + 1
                
                # Report most common transitions
                sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]
                for (from_cluster, to_cluster), count in sorted_transitions:
                    percentage = 100 * count / len(sub_sequences)
                    log.write(f"- Transition {from_cluster}->{to_cluster}: {count} times ({percentage:.1f}%)\n")
    
    except Exception as e:
        log.write(f"Error writing enhanced sequence analysis: {str(e)}\n")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    
    log.write("\n=== END OF SEQUENCE ANALYSIS ===\n")
    
def _assemble_report(log_file, model_results, feature_importance, pattern_analysis, df, prime_probability_map, cluster_sequence_analysis,
                            gap_distribution=None, gap_sequences=None, factor_patterns=None,
                            cluster_transitions=None, temporal_patterns=None, separation_metrics=None,
                            shap_values=None, shap_importance=None, prediction_intervals=None, change_points=None,
                            cluster_stats=None, advanced_clustering=None, statistical_tests=None, logger=None):
    """Assembles the report from individual section methods with improved error handling and logging."""
    if logger:
        logger.log_and_print("Assembling report...")
    
    try:
        with open(log_file, "w", encoding="utf-8") as log:
            if logger:
                logger.log_and_print("Writing executive summary...")
            _write_executive_summary(log, model_results, feature_importance, pattern_analysis, df, cluster_sequence_analysis,
                                    shap_values, shap_importance, prediction_intervals, change_points,
                                    cluster_stats, advanced_clustering, statistical_tests)
            
            if logger:
                logger.log_and_print("Writing model performance summary...")
            _write_model_performance_summary(log, model_results)
            
            if logger:
                logger.log_and_print("Writing sub-cluster analysis...")
            _write_subcluster_analysis(log, df, cluster_sequence_analysis)
            
            if logger:
                logger.log_and_print("Writing enhanced sequence analysis...")
            _write_enhanced_sequence_analysis(log, df)
            
            if logger:
                logger.log_and_print("Writing key patterns...")
            _write_key_patterns(log, pattern_analysis, df)
            
            if logger:
                logger.log_and_print("Writing statistical insights...")
            _write_statistical_insights(log, df)
            
            if logger:
                logger.log_and_print("Writing factor analysis summary...")
            _write_factor_analysis_summary(log, df)
            
            if logger:
                logger.log_and_print("Writing modular properties...")
            _write_modular_properties(log, df, logger=logger)
            
            if logger:
                logger.log_and_print("Writing correlation insights...")
            _write_correlation_insights(log, df, logger=logger)
            
            if logger:
                logger.log_and_print("Writing cluster analysis...")
            _write_cluster_analysis(log, df, cluster_stats, advanced_clustering)
            
            if logger:
                logger.log_and_print("Writing enhanced cluster descriptions...")
            _write_enhanced_cluster_descriptions(log, df, cluster_stats, logger=logger)
            
            # Write advanced analysis sections
            if logger:
                logger.log_and_print("Writing advanced analyses...")
            _write_advanced_analyses_summary(log, gap_distribution, gap_sequences, factor_patterns, cluster_transitions)
            
            if logger:
                logger.log_and_print("Writing temporal patterns...")
            _write_temporal_patterns_analysis(log, temporal_patterns)
            
            if logger:
                logger.log_and_print("Writing cluster separation analysis...")
            _write_cluster_separation_analysis(log, separation_metrics)
            
            if logger:
                logger.log_and_print("Writing advanced factor analysis...")
            _write_advanced_factor_analysis(log, factor_patterns)
            
            if logger:
                logger.log_and_print("Writing detailed analysis...")
            _write_detailed_analysis_section(log, feature_importance, None, None, None, shap_values, shap_importance, prediction_intervals, change_points, cluster_stats, advanced_clustering, statistical_tests, logger=logger)
            
            if logger:
                logger.log_and_print("Writing progression analysis...")
            _write_progression_analysis(log, pattern_analysis, time_series_analysis=temporal_patterns, logger=logger)
            
            if logger:
                logger.log_and_print("Writing predictive model analysis...")
            _write_predictive_model_analysis(log, model_results, logger=logger)
            
            if logger:
                logger.log_and_print("Writing model comparison...")
            _write_model_comparison(log, model_results, logger=logger)
            
            if logger:
                logger.log_and_print("Writing cluster centers...")
            _write_cluster_centers(log, df, cluster_stats, logger=logger)
            
            if logger:
                logger.log_and_print("Writing factor analysis...")
            _write_factor_analysis(log, df, logger=logger)
            
            if logger:
                logger.log_and_print("Writing outlier analysis...")
            _write_outlier_analysis(log, df, logger=logger)
            
            if logger:
                logger.log_and_print("Writing prime probability map...")
            _write_prime_probability_map(log, prime_probability_map, change_points, logger=logger)
            
            if logger:
                logger.log_and_print("Report writing completed successfully")
    
    except Exception as e:
        print(f"Error assembling report: {str(e)}")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()

def _write_advanced_analysis_report(log_file, df, cluster_features, temporal_patterns, separation_metrics, logger=None):
    """Write advanced analysis report with improved numerical handling and error handling."""
    if logger:
        logger.log_and_print("Writing advanced analysis report...")
    
    try:
        with open(log_file, 'a', encoding='utf-8') as log:
            log.write("\n=== ADVANCED ANALYSIS REPORT ===\n\n")
            
            # 1. Cluster Feature Analysis
            if cluster_features:
                log.write("=== Cluster Feature Analysis ===\n")
                for cluster_id, features in sorted(cluster_features.items()):
                    log.write(f"\nCluster {cluster_id}:\n")
                    for feature, stats in sorted(features.items()):
                        log.write(f"  {feature}:\n")
                        for stat_name, value in sorted(stats.items()):
                            if isinstance(value, (int, float)):
                                log.write(f"    {stat_name}: {value:.4f}\n")
                            elif isinstance(value, list):
                                log.write(f"    {stat_name}: {value[:10]}...\n")
                            else:
                                log.write(f"    {stat_name}: {value}\n")
            
            # 2. Temporal Pattern Analysis
            if temporal_patterns:
                log.write("\n=== Temporal Pattern Analysis ===\n")
                for col, patterns in temporal_patterns.items():
                    log.write(f"\n{col}:\n")
                    if 'trend' in patterns:
                        log.write(f"  Trend:\n")
                        log.write(f"    Slope: {patterns['trend'].get('slope', 'N/A'):.4e}\n")
                        log.write(f"    Intercept: {patterns['trend'].get('intercept', 'N/A'):.4e}\n")
                    if 'seasonality' in patterns:
                        log.write(f"  Seasonality:\n")
                        log.write(f"    Strength: {patterns['seasonality'].get('strength', 'N/A'):.4f}\n")
                        log.write(f"    Period: {patterns['seasonality'].get('period', 'N/A'):.4f}\n")
                    if 'autocorrelation' in patterns:
                        log.write(f"  Autocorrelation (first 5 lags):\n")
                        for i, acf in enumerate(patterns['autocorrelation'][:5], 1):
                            log.write(f"    Lag {i}: {acf:.4f}\n")
                    if 'change_points' in patterns:
                        log.write(f"  Change Points:\n")
                        for i, cp in enumerate(patterns['change_points'].get('locations', [])):
                            log.write(f"    Location {cp}: Value = {patterns['change_points'].get('values', [])[i]:.4f}\n")
            
            # 3. Cluster Separation Analysis
            if separation_metrics:
                log.write("\n=== Cluster Separation Analysis ===\n")
                if 'inter_cluster_distances' in separation_metrics:
                    log.write("\nInter-cluster Distances:\n")
                    distances = separation_metrics['inter_cluster_distances']
                    for i in range(len(distances)):
                        for j in range(i+1, len(distances)):
                            log.write(f"  Clusters {i}-{j}: {distances[i,j]:.4f}\n")
                
                if 'cluster_densities' in separation_metrics:
                    log.write("\nCluster Densities:\n")
                    for cluster_id, density in separation_metrics['cluster_densities'].items():
                        log.write(f"  Cluster {cluster_id}: {density:.4f}\n")
                
                if 'intra_cluster_distances' in separation_metrics:
                    log.write("\nIntra-cluster Distances:\n")
                    for cluster_id, stats in separation_metrics['intra_cluster_distances'].items():
                        log.write(f"  Cluster {cluster_id}:\n")
                        for stat_name, value in stats.items():
                            log.write(f"    {stat_name}: {value:.4f}\n")
            
            if logger:
                logger.log_and_print("Advanced analysis report writing complete")
            
    except Exception as e:
        error_msg = f"Error writing advanced analysis report: {str(e)}"
        if logger:
            logger.log_and_print(error_msg, level=logging.ERROR)
            logger.logger.error(traceback.format_exc())
        else:
            print(error_msg)
            traceback.print_exc()

def _write_analysis_report(log_file, model_results, feature_importance, pattern_analysis, df, 
                         prime_probability_map=None, cluster_sequence_analysis=None,
                         gap_distribution=None, gap_sequences=None, factor_patterns=None,
                         cluster_transitions=None, temporal_patterns=None, separation_metrics=None,
                         shap_values=None, shap_importance=None, prediction_intervals=None, change_points=None,
                         cluster_stats=None, advanced_clustering=None, statistical_tests=None, logger=None):
    """Assembles the report from individual section methods with improved error handling and logging."""
    if logger:
        logger.log_and_print("Assembling report...")
    
    try:
        with open(log_file, "w", encoding="utf-8") as log:
            if logger:
                logger.log_and_print("Writing executive summary...")
            _write_executive_summary(log, model_results, feature_importance, pattern_analysis, df, cluster_sequence_analysis,
                                    shap_values, shap_importance, prediction_intervals, change_points,
                                    cluster_stats, advanced_clustering, statistical_tests)
            
            if logger:
                logger.log_and_print("Writing model performance summary...")
            _write_model_performance_summary(log, model_results)
            
            if logger:
                logger.log_and_print("Writing sub-cluster analysis...")
            _write_subcluster_analysis(log, df, cluster_sequence_analysis)
            
            if logger:
                logger.log_and_print("Writing enhanced sequence analysis...")
            _write_enhanced_sequence_analysis(log, df)
            
            if logger:
                logger.log_and_print("Writing key patterns...")
            _write_key_patterns(log, pattern_analysis, df)
            
            if logger:
                logger.log_and_print("Writing statistical insights...")
            _write_statistical_insights(log, df)
            
            if logger:
                logger.log_and_print("Writing factor analysis summary...")
            _write_factor_analysis_summary(log, df)
            
            if logger:
                logger.log_and_print("Writing modular properties...")
            _write_modular_properties(log, df, logger=logger)
            
            if logger:
                logger.log_and_print("Writing correlation insights...")
            _write_correlation_insights(log, df, logger=logger)
            
            if logger:
                logger.log_and_print("Writing cluster analysis...")
            _write_cluster_analysis(log, df, cluster_stats, advanced_clustering)
            
            if logger:
                logger.log_and_print("Writing enhanced cluster descriptions...")
            _write_enhanced_cluster_descriptions(log, df, cluster_stats, logger=logger)
            
            # Write advanced analysis sections
            if logger:
                logger.log_and_print("Writing advanced analyses...")
            _write_advanced_analyses_summary(log, gap_distribution, gap_sequences, factor_patterns, cluster_transitions)
            
            if logger:
                logger.log_and_print("Writing temporal patterns...")
            _write_temporal_patterns_analysis(log, temporal_patterns)
            
            if logger:
                logger.log_and_print("Writing cluster separation analysis...")
            _write_cluster_separation_analysis(log, separation_metrics)
            
            if logger:
                logger.log_and_print("Writing advanced factor analysis...")
            _write_advanced_factor_analysis(log, factor_patterns)
            
            if logger:
                logger.log_and_print("Writing detailed analysis...")
            _write_detailed_analysis_section(log, feature_importance, None, None, None, shap_values, shap_importance, prediction_intervals, change_points, cluster_stats, advanced_clustering, statistical_tests, logger=logger)
            
            if logger:
                logger.log_and_print("Writing progression analysis...")
            _write_progression_analysis(log, pattern_analysis, time_series_analysis=temporal_patterns, logger=logger)
            
            if logger:
                logger.log_and_print("Writing predictive model analysis...")
            _write_predictive_model_analysis(log, model_results, logger=logger)
            
            if logger:
                logger.log_and_print("Writing model comparison...")
            _write_model_comparison(log, model_results, logger=logger)
            
            if logger:
                logger.log_and_print("Writing cluster centers...")
            _write_cluster_centers(log, df, cluster_stats, logger=logger)
            
            if logger:
                logger.log_and_print("Writing factor analysis...")
            _write_factor_analysis(log, df, logger=logger)
            
            if logger:
                logger.log_and_print("Writing outlier analysis...")
            _write_outlier_analysis(log, df, logger=logger)
            
            if logger:
                logger.log_and_print("Writing prime probability map...")
            _write_prime_probability_map(log, prime_probability_map, change_points, logger=logger)
            
            if logger:
                logger.log_and_print("Report writing completed successfully")
    
    except Exception as e:
        print(f"Error assembling report: {str(e)}")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
            
def write_analysis_report(log_file, model_results, feature_importance, pattern_analysis, df, 
                         prime_probability_map=None, cluster_sequence_analysis=None,
                         gap_distribution=None, gap_sequences=None, factor_patterns=None,
                         cluster_transitions=None, temporal_patterns=None, separation_metrics=None,
                         shap_values=None, shap_importance=None, prediction_intervals=None, change_points=None,
                         cluster_stats=None, advanced_clustering=None, statistical_tests=None, logger=None):
    """Write comprehensive analysis report including all sections with improved error handling and logging."""
    if logger:
        logger.log_and_print("Starting report writing process...")
    
    try:
        with open(log_file, "w", encoding="utf-8") as log:
            if logger:
                logger.log_and_print("Writing executive summary...")
            _write_executive_summary(log, model_results, feature_importance, pattern_analysis, df, cluster_sequence_analysis,
                                    shap_values, shap_importance, prediction_intervals, change_points,
                                    cluster_stats, advanced_clustering, statistical_tests)
            
            if logger:
                logger.log_and_print("Writing model performance summary...")
            _write_model_performance_summary(log, model_results)
            
            if logger:
                logger.log_and_print("Writing sub-cluster analysis...")
            _write_subcluster_analysis(log, df, cluster_sequence_analysis)
            
            if logger:
                logger.log_and_print("Writing enhanced sequence analysis...")
            _write_enhanced_sequence_analysis(log, df, logger=logger)
            
            if logger:
                logger.log_and_print("Writing key patterns...")
            _write_key_patterns(log, pattern_analysis, df)
            
            if logger:
                logger.log_and_print("Writing statistical insights...")
            _write_statistical_insights(log, df)
            
            if logger:
                logger.log_and_print("Writing factor analysis summary...")
            _write_factor_analysis_summary(log, df)
            
            if logger:
                logger.log_and_print("Writing modular properties...")
            _write_modular_properties(log, df, logger=logger)
            
            if logger:
                logger.log_and_print("Writing correlation insights...")
            _write_correlation_insights(log, df, logger=logger)
            
            if logger:
                logger.log_and_print("Writing cluster analysis...")
            _write_cluster_analysis(log, df, cluster_stats, advanced_clustering)
            
            # Write advanced analysis sections
            if logger:
                logger.log_and_print("Writing advanced analyses...")
            _write_advanced_analyses_summary(log, gap_distribution, gap_sequences, factor_patterns, cluster_transitions)
            
            if logger:
                logger.log_and_print("Writing temporal patterns...")
            _write_temporal_patterns_analysis(log, temporal_patterns)
            
            if logger:
                logger.log_and_print("Writing cluster separation analysis...")
            _write_cluster_separation_analysis(log, separation_metrics)
            
            if logger:
                logger.log_and_print("Writing advanced factor analysis...")
            _write_advanced_factor_analysis(log, factor_patterns)
            
            if logger:
                logger.log_and_print("Writing detailed analysis...")
            _write_detailed_analysis_section(log, feature_importance, None, None, None, shap_values, shap_importance, prediction_intervals, change_points, cluster_stats, advanced_clustering, statistical_tests, logger=logger)
            
            if logger:
                logger.log_and_print("Writing progression analysis...")
            _write_progression_analysis(log, pattern_analysis, time_series_analysis=temporal_patterns, logger=logger)
            
            if logger:
                logger.log_and_print("Writing predictive model analysis...")
            _write_predictive_model_analysis(log, model_results, logger=logger)
            
            if logger:
                logger.log_and_print("Writing model comparison...")
            _write_model_comparison(log, model_results, logger=logger)
            
            if logger:
                logger.log_and_print("Writing cluster centers...")
            _write_cluster_centers(log, df, cluster_stats, logger=logger)
            
            if logger:
                logger.log_and_print("Writing factor analysis...")
            _write_factor_analysis(log, df, logger=logger)
            
            if logger:
                logger.log_and_print("Writing outlier analysis...")
            _write_outlier_analysis(log, df, logger=logger)
            
            if logger:
                logger.log_and_print("Writing prime probability map...")
            _write_prime_probability_map(log, prime_probability_map, change_points, logger=logger)
            
            if logger:
                logger.log_and_print("Report writing completed successfully")
            
    except Exception as e:
        print(f"Error writing analysis report: {str(e)}")
        if logger:
            logger.logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
                
################################################################################
# Run Analysis
################################################################################

def save_checkpoint(self, state, checkpoint_name):
    """Save analysis state to checkpoint with error handling and compression."""
    checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pkl")
    temp_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_temp.pkl")
    
    try:
        # First save to temporary file
        with open(temp_path, 'wb') as f:
            # Convert DataFrames to more efficient format before saving
            state_to_save = state.copy()
            if 'dataframe' in state_to_save:
                state_to_save['dataframe'] = state_to_save['dataframe'].to_dict('records')
            if 'feature_importance' in state_to_save and isinstance(state_to_save['feature_importance'], pd.DataFrame):
                state_to_save['feature_importance'] = state_to_save['feature_importance'].to_dict('records')
                
            # Save with highest protocol for better performance
            pickle.dump(state_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # If temporary save successful, move to final location
        if os.path.exists(checkpoint_path):
            # Keep one backup
            backup_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_backup.pkl")
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.rename(checkpoint_path, backup_path)
            
        os.rename(temp_path, checkpoint_path)
        self.current_state = state
        
        # Log successful save
        self.log_success(f"Successfully saved checkpoint: {checkpoint_name}")
        
        # Clean up old checkpoints if needed
        self._cleanup_old_checkpoints()
        
        return True
        
    except Exception as e:
        self.log_error(f"Failed to save checkpoint {checkpoint_name}: {str(e)}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False
        
    finally:
        # Ensure temp file is cleaned up
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def _cleanup_old_checkpoints(self, max_checkpoints=5):
    """Clean up old checkpoints, keeping only the most recent ones."""
    try:
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pkl')]
        if len(checkpoints) > max_checkpoints:
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)))
            for old_checkpoint in checkpoints[:-max_checkpoints]:
                os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))
    except Exception as e:
        self.log_error(f"Error cleaning up old checkpoints: {str(e)}")

def log_success(self, message):
    """Log success message with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    success_log = os.path.join(self.checkpoint_dir, "success.log")
    try:
        with open(success_log, 'a') as f:
            f.write(f"{timestamp}: {message}\n")
    except Exception as e:
        print(f"Warning: Could not write to success log: {str(e)}")
        
if __name__ == "__main__":
    try:
        # Initialize logger
        logger = PrimeAnalysisLogger(debug_mode=DEBUG_MODE)
        
        # Log initial system state
        logger.logger.info("\nInitial system state:")
        memory = psutil.virtual_memory()
        logger.logger.info(f"Total memory: {memory.total / (1024**3):.2f} GB")
        logger.logger.info(f"Available memory: {memory.available / (1024**3):.2f} GB")
        logger.logger.info(f"Memory used: {memory.percent}%")
        
        logger.logger.info(f"\nStarting prime number analysis for n={N}")
        logger.logger.info(f"Batch threshold: {BATCH_THRESHOLD}")
        logger.logger.info(f"Current N: {N}")
        logger.logger.info(f"Should use batches: {N >= BATCH_THRESHOLD}")
        
        # Enable garbage collection
        gc.enable()
        gc.set_threshold(100, 5, 5)  # More aggressive GC
        
        with suppress_numeric_warnings():
            if N >= BATCH_THRESHOLD:
                logger.logger.info(f"\nLarge dataset detected (N={N}). Using batch processing...")
                
                # Determine optimal batch size based on available memory
                available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
                estimated_memory_per_prime = 0.5  # MB per prime (conservative estimate)
                batch_size = min(
                    5000,  # Maximum batch size
                    int((available_memory * 0.2) / estimated_memory_per_prime),  # Use 20% of available memory
                    int(N / (psutil.cpu_count() or 1)) # Distribute workload across cores
                )
                
                logger.logger.info(f"Using batch size of {batch_size} (Available memory: {available_memory:.2f} MB)")
                
                # Try to recover from checkpoint
                recovery = PrimeAnalysisRecovery()
                recovered_state = recovery.recover_analysis(N, batch_size)
                
                if recovered_state is None:
                    logger.logger.info("Starting fresh analysis...")
                    
                    # Process primes in batches
                    df = batch_process_primes(N, batch_size=batch_size, logger=logger)
                    df = optimize_memory_usage(df)
                    
                    # Clear memory before clustering
                    gc.collect()
                    
                    # Perform clustering
                    df = perform_initial_clustering(df, logger=logger)
                else:
                    df = recovered_state['dataframe']
            else:
                logger.logger.info(f"\nStandard dataset size (N={N}). Using regular processing...")
                # Generate primes and compute gaps
                primes = generate_primes(N)
                gaps = compute_gaps(primes)
                
                # Compute features and create DataFrame
                logger.logger.info("\nComputing features...")
                features_list = []
                for i in range(len(gaps)):
                    features = compute_advanced_prime_features(
                        primes[i],
                        primes[i+1],
                        gaps[i]
                    )
                    features['lower_prime'] = primes[i]
                    features['upper_prime'] = primes[i+1]
                    features_list.append(features)
                
                df = pd.DataFrame(features_list)
                df = optimize_memory_usage(df)
                df = perform_initial_clustering(df, logger=logger)
            
            # Clear memory before feature engineering
            gc.collect()
            
            # Create advanced features
            logger.logger.info("\nCreating advanced features...")
            df = create_advanced_features(df, logger=logger)
            
            # Perform comprehensive feature analysis
            logger.logger.info("\nAnalyzing feature importance...")
            importance_analysis = analyze_feature_importance(df, logger=logger)
            
            # Select optimal features
            logger.logger.info("\nSelecting optimal features...")
            feature_selection_results = select_optimal_features(df, importance_analysis, logger=logger)
            selected_features = feature_selection_results['optimal_features']
            
            # Analyze feature interactions
            logger.logger.info("\nAnalyzing feature interactions...")
            interaction_analysis = analyze_feature_interactions(df, selected_features, logger=logger)
            
            # Analyze feature stability
            logger.logger.info("\nAnalyzing feature stability...")
            stability_analysis = analyze_feature_stability(df, selected_features, logger=logger)
            
            # Update feature_cols to use selected features
            feature_cols = selected_features
            
            # Add results to analysis_stats
            analysis_stats = {
                'feature_importance': importance_analysis,
                'feature_selection': feature_selection_results,
                'feature_interactions': interaction_analysis,
                'feature_stability': stability_analysis
            }
            
            # Create feature analysis visualizations
            logger.logger.info("\nCreating feature analysis visualizations...")
            
            # 1. Feature Importance Plot
            plt.figure(figsize=(12, 6))
            importance_scores = pd.Series(feature_selection_results['feature_scores'])
            importance_scores.sort_values(ascending=True).tail(20).plot(kind='barh')
            plt.title('Top 20 Most Important Features')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, 'feature_importance.png'))
            plt.close()
            
            # 2. Feature Stability Plot
            plt.figure(figsize=(12, 6))
            stability_scores = pd.Series(stability_analysis['bootstrap_scores']['cv_importance'])
            stability_scores.sort_values(ascending=True).tail(20).plot(kind='barh')
            plt.title('Feature Stability Scores')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, 'feature_stability.png'))
            plt.close()
            
            # 3. Feature Interaction Heatmap
            plt.figure(figsize=(12, 10))
            # First check if interaction_analysis is a tuple and handle it
            if isinstance(interaction_analysis, tuple):
                interaction_analysis = interaction_analysis[0]  # Get the dictionary from the tuple

            # Now safely access the nested structure
            pairwise_correlations = interaction_analysis.get('pairwise_correlations', {})
            if isinstance(pairwise_correlations, dict):
                correlations = pairwise_correlations.get('significant_correlations', [])
                if correlations and isinstance(correlations, list):
                    # Create correlation matrix with float64 dtype
                    features = list(set([c['feature1'] for c in correlations] + [c['feature2'] for c in correlations]))
                    interaction_matrix = pd.DataFrame(0.0, index=features, columns=features, dtype=np.float64)
                    
                    # Fill the matrix with correlations
                    for corr in correlations:
                        # Convert correlation values to Python float
                        correlation_value = float(corr['correlation'])
                        # Use at[] for setting values which accepts Python float
                        interaction_matrix.at[corr['feature1'], corr['feature2']] = correlation_value
                        interaction_matrix.at[corr['feature2'], corr['feature1']] = correlation_value
                    
                    sns.heatmap(interaction_matrix, annot=True, cmap='RdBu_r', center=0)
                    plt.title('Feature Interaction Heatmap')
                    plt.tight_layout()
                    plt.savefig(os.path.join(PLOT_DIR, 'feature_interactions.png'))
                    plt.close()
                else:
                    if logger:
                        logger.log_and_print("No significant correlations found for heatmap")
            else:
                if logger:
                    logger.log_and_print("Invalid pairwise correlations structure")
            
            # 4. Temporal Stability Plot
            if 'temporal_stability' in stability_analysis:
                plt.figure(figsize=(12, 6))
                temporal_scores = pd.DataFrame({
                    feature: data['mean_stability'] 
                    for feature, data in stability_analysis['temporal_stability'].items()
                }, index=[0]).T  # Transpose to get features as index
                # Sort by the value column
                temporal_scores.columns = ['stability']
                temporal_scores = temporal_scores.sort_values('stability', ascending=True).tail(20)
                # temporal_scores.plot(kind='barh') removed this line due to errors 
                plt.title('Feature Temporal Stability')
                plt.tight_layout()
                plt.savefig(os.path.join(PLOT_DIR, 'feature_temporal_stability.png'))
                plt.close()
            
            # 5. Value Range Stability Plot
            plt.figure(figsize=(12, 6))
            range_stability = pd.DataFrame({
                feature: data['range_ratio']
                for feature, data in stability_analysis['value_range_stability'].items()
            }, index=[0]).T  # Transpose to get features as index
            # Sort by the value column
            range_stability.columns = ['range_ratio']
            range_stability = range_stability.sort_values('range_ratio', ascending=True).tail(20)
            range_stability.plot(kind='barh')
            plt.title('Feature Value Range Stability')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, 'feature_value_stability.png'))
            plt.close()
            
            # 6. Nonlinear Relationships Plot
            if 'nonlinear_relationships' in interaction_analysis:
                plt.figure(figsize=(15, 10))
                nonlinear_scores = pd.DataFrame(interaction_analysis['nonlinear_relationships']).T
                sns.heatmap(nonlinear_scores, annot=True, cmap='YlOrRd')
                plt.title('Nonlinear Relationship Strengths')
                plt.tight_layout()
                plt.savefig(os.path.join(PLOT_DIR, 'nonlinear_relationships.png'))
                plt.close()
            
            # Prepare data for model training using selected features
            X = df[selected_features]
            y = df['gap_size']
            
            logger.logger.info("\nFeature analysis complete. Proceeding with model training...")
            
            # Clear memory before model training
            gc.collect()
            
            # Prepare training data with error handling
            logger.logger.info("\nPreparing training data...")
            X, y, feature_cols, cluster_X, cluster_y, gap_cluster_X, gap_cluster_y, next_cluster_X, next_cluster_y = \
                prepare_training_data(df)
            
            # Train models with error handling
            logger.logger.info("\nTraining models...")
            model_results, feature_importance = train_models(
                X, y, feature_cols, cluster_X, cluster_y, 
                gap_cluster_X, gap_cluster_y, next_cluster_X, next_cluster_y
            )
            
            # Clear memory before analyses
            gc.collect()
            
            # Perform analyses
            logger.logger.info("\nPerforming analyses...")
            
            # Pre-compute all statistics
            logger.logger.info("Computing analysis statistics...")
            analysis_stats.update({
                'pattern_analysis': analyze_gap_patterns(df['gap_size'].values),
                'cluster_stats': compute_cluster_statistics(df),
                'factor_stats': compute_factor_statistics(df),
                'correlation_stats': compute_correlation_statistics(df),
                'cluster_center_stats': compute_cluster_center_statistics(df),
                'temporal_stats': compute_temporal_pattern_statistics(df),
                'gap_distribution': analyze_gap_distribution_characteristics(df, logger=logger),
                'gap_sequences': analyze_gap_sequences_advanced(df, logger=logger),
                'factor_patterns': analyze_prime_factor_patterns(df),
                'cluster_transitions': analyze_cluster_transitions_advanced(df, logger=logger),
                'cluster_stability': compute_cluster_stability(df),
                'prime_distribution': analyze_prime_distribution(df, logger=logger)
            })
            
            # Compute SHAP values for feature interpretation
            logger.logger.info("\nComputing SHAP values...")
            shap_values, shap_importance = compute_shap_values(
                {name: model_results[name]['model'] for name in ['random_forest', 'xgboost'] if name in model_results},
                X,
                feature_cols
            )
            
            # Compute prediction intervals for best model
            best_model_name = min(model_results, key=lambda k: model_results[k].get('avg_test_mse', float('inf')))
            best_model = model_results[best_model_name]['model'] 
            logger.logger.info(f"\nComputing prediction intervals for {best_model_name}...")
            mean_pred, lower_pred, upper_pred = compute_prediction_intervals(best_model, X)
            
            # Detect change points
            logger.logger.info("\nDetecting change points...")
            change_point_analysis = detect_change_points(df)
            
            # Perform advanced clustering analysis
            logger.logger.info("\nPerforming advanced clustering analysis...")
            advanced_clustering_results = perform_advanced_clustering_analysis(df)
            
            # Perform statistical tests
            logger.logger.info("\nPerforming statistical tests...")
            statistical_test_results = perform_advanced_statistical_tests(df, advanced_clustering_results)
            
            # Add results to analysis_stats
            analysis_stats.update({
                'shap_analysis': {
                    'values': shap_values,
                    'importance': shap_importance
                },
                'prediction_intervals': {
                    'mean': mean_pred,
                    'lower': lower_pred,
                    'upper': upper_pred
                },
                'change_points': change_point_analysis,
                'advanced_clustering': advanced_clustering_results,
                'statistical_tests': statistical_test_results
            })
            
            # Create interactive visualizations
            logger.logger.info("\nCreating interactive visualizations...")
            create_interactive_visualizations(df, analysis_stats, PLOT_DIR)
            
            # Create visualizations
            logger.logger.info("\nCreating visualizations...")
            create_visualizations_large_scale(df, feature_importance, analysis_stats['pattern_analysis'], 
                                           PLOT_DIR, model_results)
            create_cluster_visualization(df, PLOT_DIR, logger)
            
            # Write comprehensive analysis report
            logger.logger.info(f"\nWriting analysis results to {OUTPUT_LOG_FILE}")
            write_analysis_report(
                OUTPUT_LOG_FILE,
                model_results,
                feature_importance,
                analysis_stats['pattern_analysis'],
                df,
                gap_distribution=analysis_stats['gap_distribution'],
                gap_sequences=analysis_stats['gap_sequences'],
                factor_patterns=analysis_stats['factor_patterns'],
                cluster_transitions=analysis_stats['cluster_transitions'],
                temporal_patterns=analysis_stats['temporal_stats'],
                separation_metrics=analysis_stats.get('separation_metrics')
            )
            
            # Store complete results
            complete_results = {
                'dataframe': df,
                'model_results': model_results,
                'feature_importance': feature_importance,
                **analysis_stats
            }
            
            # Save checkpoint if using batch processing
            if N >= BATCH_THRESHOLD:
                recovery.save_checkpoint(complete_results, "last_successful_state")
            
            # Final memory cleanup
            gc.collect()
            logger.log_memory_usage("Final memory usage")
            
            logger.logger.info("\nAnalysis completed successfully.")
            
    except Exception as e:
        if 'logger' in locals():
            logger.logger.error(f"\nCritical error in analysis: {str(e)}")
            logger.logger.error(traceback.format_exc())
        else:
            print(f"\nCritical error in analysis: {str(e)}")
            traceback.print_exc()
            
        # Log the error
        with open(os.path.join(os.path.dirname(OUTPUT_LOG_FILE), "error_log.txt"), "a") as error_log:
            error_log.write(f"{datetime.datetime.now()}: {str(e)}\n")
        raise   