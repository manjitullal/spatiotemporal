import pandas as pd
import pytz
from datetime import datetime
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
import pickle

data_dir2 = 'C:\\Users\\Manjit\\Downloads\\THP\\Transformer-Hawkes-Process-master4-wc\\data\\'

class NycTaxi:
    """
    data_dir must end with '/'
    features are other features to use beyond category, lat-lon and time, must be continuous
    """

    """
    There are 12 files each with approx 14m records 
    We load the first file
    """

    def __init__(self, features=None, data_dir = 'C:\\Users\\Manjit\\Downloads\\nyc_data\\', chunk_count = 3):

        self.raw_dataiter = pd.read_csv(data_dir + 'trip_data_1.csv', chunksize=100000)

        # load only the mentioned chunks, we do this since the dataset is huge
        chunks = []
        for i, data in enumerate(self.raw_dataiter):
            if i <= chunk_count:
                chunks.append(data)
            else:
                break
        self.trips_data = pd.concat(chunks)

        if features is None:
            # default features
            self.features = ['medallion', 'hack_license', 'vendor_id', 'rate_code',
                             'store_and_fwd_flag', 'pickup_datetime', 'dropoff_datetime',
                             'passenger_count', 'trip_time_in_secs', 'trip_distance',
                             'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                             'dropoff_latitude']
        else:
            self.features = features

        self.st =  None
        self.st_scaler = None
        self.train, self.val, self.test = None, None, None
        self.final_features = None

        self.preprocess_trips()
        self.extract_features()
        self.top_taxi_dataset()

    def preprocess_trips(self):
        print("Preprocessing trips data")
        ids = list(self.trips_data['medallion'].unique()) # Unique medallion, used as an identifier

        try:
            self.trips_data = pd.read_pickle('C:\\Users\\Manjit\\Downloads\\THP\\Transformer-Hawkes-Process-master4-wc\\data\\trips_data.pkl')
            print("Finished.")
            return
        except FileNotFoundError:
            print('Preprocessed trips_data.pkl not found locally, processing it for the first time...')

            self.trips_data['year'] = [d[0:4] for d in self.trips_data['pickup_datetime']]
            self.trips_data['month'] = [d[5:7] for d in self.trips_data['pickup_datetime']]
            self.trips_data['pdate'] = [d[8:10] for d in self.trips_data['pickup_datetime']]
            self.trips_data['phour'] = [d[11:13] for d in self.trips_data['pickup_datetime']]
            self.trips_data['pminute'] = [d[14:16] for d in self.trips_data['pickup_datetime']]
            self.trips_data['ddate'] = [d[8:10] for d in self.trips_data['dropoff_datetime']]
            self.trips_data['dhour'] = [d[11:13] for d in self.trips_data['dropoff_datetime']]
            self.trips_data['dminute'] = [d[14:16] for d in self.trips_data['dropoff_datetime']]

            self.trips_data['year'] = self.trips_data['year'].astype(int)
            self.trips_data['month'] = self.trips_data['month'].astype(int)
            self.trips_data['pdate'] = self.trips_data['pdate'].astype(int) # pickup starts with p
            self.trips_data['phour'] = self.trips_data['phour'].astype(int)
            self.trips_data['pminute'] = self.trips_data['pminute'].astype(int)
            self.trips_data['ddate'] = self.trips_data['ddate'].astype(int) # dropoff starts with d
            self.trips_data['dhour'] = self.trips_data['dhour'].astype(int)
            self.trips_data['dminute'] = self.trips_data['dminute'].astype(int)

            self.trips_data['pickup_longitude'] = self.trips_data['pickup_longitude'].astype(float)
            self.trips_data['pickup_latitude'] = self.trips_data['pickup_latitude'].astype(float)
            self.trips_data['dropoff_longitude'] = self.trips_data['dropoff_longitude'].astype(float)
            self.trips_data['dropoff_latitude'] = self.trips_data['dropoff_latitude'].astype(float)

            self.trips_data['timestamp'] = self.trips_data.apply(
                lambda row: datetime.timestamp(
                    datetime(row.year, row.month, row.pdate, row.phour, 30,
                             tzinfo=pytz.timezone('UTC'))), axis=1)


            self.trips_data = self.trips_data.sort_values(by=['month', 'pdate', 'phour', 'pminute'])

            # there are few trips with no spatial information, exclude those records
            self.trips_data.drop(self.trips_data[self.trips_data.pickup_longitude == 0].index, inplace=True)
            self.trips_data.to_csv('C:\\Users\\Manjit\\Desktop\\Research Project\\zihao\\trips_data.csv' )
            self.trips_data.to_pickle('C:\\Users\\Manjit\\Downloads\\THP\\Transformer-Hawkes-Process-master4-wc\\data\\trips_data.pkl')
            print('trips_data.pkl stored locally')
            print("Finished.")


    def extract_features(self):
        print("Extracting features ...")
        self.final_features = ['pickup_latitude', 'pickup_longitude', 'timestamp']
        grouped = self.trips_data.groupby('medallion')
        self.st  = grouped[self.final_features].apply(lambda x: x.values.tolist())
        print(f"There are {len(self.st)} medallions")
        self.lens = self.st.apply(len)
        print(f"Mean number of taxi trip records: {np.mean(self.lens)}")
        print("Finished.")


    def top_taxi_dataset(self, lookback=10, lookahead=1, split=None):
        top_taxi = self.lens.idxmax()
        self.single_taxi_dataset(top_taxi, lookback, lookahead, split)


    def single_taxi_dataset(self, user, lookback=10, lookahead=1, split=None):
        print("Loading dataset ...")
        
        # Seperate category and continuous data
        st_data  = np.array(self.st[user]) 

        # Default split is train:val:test = 8:1:1
        if split is None:
            split = [8, 1, 1]
        split = split / np.sum(split)

        # Min-max scale  spatiotemporal data
        self.st_scaler = MinMaxScaler()
        self.st_scaler.fit(st_data)
        self.st_data = self.st_scaler.transform(st_data)

        length = len(st_data) - lookback - lookahead

        self.data_dictionary = []
        for i, event in enumerate(self.st_data):
            temp_dict = {}
            temp_dict['idx_event'] = i
            temp_dict['latitude'] = event[0]
            temp_dict['longitude'] = event[1]
            temp_dict['time_since_start'] = event[2]
            temp_dict['type_event'] = 0
            temp_dict['time_since_last_event'] = event[2] # time gap
            self.data_dictionary.append(temp_dict)

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
        with open(data_dir2 + 'train_ny.pkl', 'wb') as f:
            pickle.dump(train_x, f)

        with open(data_dir2 + 'val_ny.pkl', 'wb') as f:
            pickle.dump(val_x, f)

        with open(data_dir2 + 'test_ny.pkl', 'wb') as f:
            pickle.dump(test_x, f)


        # Breaking sequence to training data: [1-1303] -> [1 ~ lookback][lookback+1],
        # [2 ~ lookback+1][lookback+2]...


        '''self.train = TensorDataset(torch.Tensor(st_input[:train_size]),
                                   torch.Tensor(st_label[:train_size]))

        self.val   = TensorDataset(torch.Tensor(st_input[train_size:-test_size]), 
                                   torch.Tensor(st_label[train_size:-test_size]))

        self.test  = TensorDataset(torch.Tensor(st_input[-test_size:]), 
                                   torch.Tensor(st_label[-test_size:]))'''

        print("Finished.")
