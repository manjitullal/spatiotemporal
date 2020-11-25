import pandas as pd
import pytz
import datetime
import numpy as np
import math
import torch
import h5py
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as MSE
from torch.utils.data import TensorDataset, DataLoader
from HDF5 import HDF5Dataset
data_dir2 = 'C:\\Users\\Manjit\\Downloads\\THP\\Transformer-Hawkes-Process-master4-wc\\data\\'

class Earthquakes:
    """
    data_dir must end with '/'
    features are other features to use beyond category, lat-lon and time, must be continuous
    """
    # features: depth, mag
    def __init__(self, features = None, data_dir = "C:\\Users\\Manjit\\Downloads\\earthquakes_data\\"):
        self.data_dir = data_dir
        try:
                self.data = pd.read_csv(data_dir + "consolidated_data.csv")
        except FileNotFoundError:
                print(f"Exception: {data_dir} does not contain the required data files.")
                return

        if features is None:
            self.features = [] # default features
        else:
            self.features = features

        self.data = self.data[self.data["mag"] >= 4.0]
        self.st, self.cat, self.oth = None, None, None
        self.st_scaler, self.oth_scaler, self.num_class = None, None, None
        self.train, self.val, self.test = None, None, None

        self.preprocess()
        self.extract_features()
        self.dataset()
    
    def preprocess(self):
        print("Preprocessing...")
        try:
            self.data = pd.read_pickle('Earthquakes.pkl')
            print("Finished.")
            return
        except FileNotFoundError:
            print('Preprocessed Earthquakes.pkl not found locally, processing it for the first time...')
            
            self.data['timestamp'] = self.data.apply(lambda row: datetime.datetime.strptime(row['time'], "%Y-%m-%dT%H:%M:%S.%fZ"), axis=1)
            k = self.data["type"].unique()
            v = [i for i in range(len(k))]
            dict1 = dict(zip(k,v))
            self.data['category'] = self.data.apply(lambda row: dict1[row['type']], axis=1)

            self.data = self.data.rename(columns={'latitude': "lat", "longitude": 'lon'})
            
            self.data = self.data.fillna(self.data.select_dtypes(exclude=['object']).mean())
            self.data = self.data.sort_values(by=['timestamp'])
            self.data.to_pickle('Earthquakes.pkl')
            print('Earthquakes.pkl stored locally')
            print("Finished.")
        
    def extract_features(self):
        print("Extracting features ...")
        self.cat = self.data[['category',]].apply(lambda x: x.values.tolist())
        self.st  = self.data[['lat', 'lon', 'timestamp']].apply(lambda x: x.values.tolist())
        self.oth = self.data[self.features].apply(lambda x: x.values.tolist())

        print(f"There are {len(self.st)} records")
        self.lens = self.st.apply(len)
        print("Finished.")
      
    def dataset(self, lookback=10, lookahead=1, split=None, chunk = 20):
        # chunk: # of groups in a hdf5 file
        print("Loading dataset ...")
        
        # Seperate category and continuous data
        self.st_data  = np.array(self.st)
        self.oth_data = np.array(self.oth)
        self.cat_data = np.array(self.cat)

        # timestamp -> delta_t (in seconds)
        # st_data[:, -1][1:] = np.diff(st_data[:, -1]) // 3600
        self.st_data[:, -1][1:] = np.diff(self.st_data[:, -1])  * 1e-9 # seconds
        self.st_data[:, -1][0]  = 0

        # Default split is train:val:test = 8:1:1
        if split is None:
            split = [8, 1, 1]
        split = split / np.sum(split)

        # Min-max scale  spatiotemporal data
        self.st_scaler = MinMaxScaler()
        self.st_scaler.fit(self.st_data)
        self.st_data = self.st_scaler.transform(self.st_data)
        
        # Min-max other features
        if self.oth_data.shape[0] != 0:
            self.oth_scaler = MinMaxScaler()
            self.oth_scaler.fit(self.oth_data)
            oth_data = self.oth_scaler.transform(self.oth_data)

        # create a dictionary
        self.data_dictionary = []
        for i in range(self.data.shape[0]):
            temp_dict = {}
            temp_dict['idx_event'] = i # 'idx_event'
            temp_dict['latitude'] = self.st_data[i][0] # 'latitude'
            temp_dict['longitude'] = self.st_data[i][1] # 'longitude'
            temp_dict['time_since_start'] = self.st_data[i][2]  # 'time_since_start'
            temp_dict['type_event'] = 0 # 'type_event' changing event type for all as 0
            temp_dict['time_since_last_event'] = self.st_data[i][2] # 'time_since_last_event'
            self.data_dictionary.append(temp_dict)

        length = len(self.data_dictionary) - lookback - lookahead
        self.num_class = np.max(self.cat_data) + 1

        lookback = 10
        self.inputs = []
        # labels   = []
        for i in range(len(self.data_dictionary) - lookback - 1):
            self.inputs.append(self.data_dictionary[i:i + lookback])

        test_portion = int(0.05 * len(self.inputs))
        val_portion = int(0.1 * len(self.inputs))
        train_x = self.inputs[:-val_portion]
        val_x = self.inputs[-val_portion:]
        test_x = val_x[-test_portion:]
        val_x = val_x[:-test_portion]
        with open(data_dir2 + 'train_eq.pkl', 'wb') as f:
            pickle.dump(train_x, f)

        with open(data_dir2 + 'val_eq.pkl', 'wb') as f:
            pickle.dump(val_x, f)

        with open(data_dir2 + 'test_eq.pkl', 'wb') as f:
            pickle.dump(test_x, f)

        print("Finished.")
