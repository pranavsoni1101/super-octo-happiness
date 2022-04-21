# os mdule provides functions to interact with the os
import os
import scipy
import numpy as np
import pandas as pd
# Seaborn is used to visualise data & and it is based on matplotlib
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from functools import reduce

# pywt is Wavelet transformation software for python
from pywt import wavedec

from scipy import signal
from scipy.io import loadmat
from scipy.stats import entropy
from scipy.fft import fft, ifft

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import KFold,cross_validate

from tensorflow import keras as K
# from tensorflow.keras.layers import Conv1D,Conv2D,Add
# from tensorflow.keras.layers import MaxPool1D, MaxPooling2D
# from tensorflow.keras.models import Sequential, Model, load_model
# from tensorflow.keras.layers import Dense, Activation, Flatten, concatenate, Input, Dropout, LSTM, Bidirectional,BatchNormalization,PReLU,ReLU,Reshape

# Importing and reading oasis dataset
longitudnal_data = pd.read_csv("../oasis_dataset/oasis_longitudnal.csv") 
crossSectional_data = pd.read_csv("../oasis_dataset/oasis_cross-sectional.csv") 