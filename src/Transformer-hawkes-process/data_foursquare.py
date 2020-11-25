import datetime
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data_dir = 'C:\\Users\\Manjit\\Desktop\\Research Project\\Foursquare Checkin DataSet\\'
data_dir2 = 'C:\\Users\\Manjit\\Downloads\\THP\\Transformer-Hawkes-Process-master4-wc\\data\\'

class foursquare:
    def __init__(self):
        self.venue_data = pd.read_csv(data_dir + '/Venue_.txt')
        self.ids = set(self.venue_data['CategoryID'])  # Unique category ids
        self.ids.remove('no category information')
        string_to_int_id = dict(zip(self.ids, range(1, len(self.ids) + 1)))
        string_to_int_id['no category information'] = 0

        # Convert string category ID to int category ID
        self.venue_data['CategoryID_'] = self.venue_data.apply(lambda row: string_to_int_id[row.CategoryID], axis=1)
        self.venue_data = self.venue_data.set_index('VenueID')
        self.preprocess()
        self.generateevents()
        self.eventsprocess()
        self.addfields()
        self.extractfields()
        self.scaling_data()
        self.format_data()
        self.createsequence()
        self.create_test_val_trainset()
        print("Completed")

    def preprocess(self):

        '''
        self.checkin_data = pd.read_csv(data_dir + '/Checkin.txt')
        # Add day of week(0-based), spatial info, venue category for each checkin event
        self.checkin_data['dayofweek'] = self.checkin_data.apply(lambda row:
                                                       datetime.datetime(row.year, row.month, row.date).weekday(), axis=1)
        self.checkin_data['category']  = self.checkin_data.apply(lambda row: self.venue_data.loc[row.venueID, 'CategoryID_'], axis=1)
        self.checkin_data['lat']       = self.checkin_data.apply(lambda row: self.venue_data.loc[row.venueID, 'Latitude'], axis=1)
        self.checkin_data['lon']       = self.checkin_data.apply(lambda row: self.venue_data.loc[row.venueID, 'Longitude'], axis=1)
        checkin_data              = self.checkin_data.sort_values(by=['month', 'date', 'hour'])
        checkin_data.to_pickle(data_dir + '/Checkin.pkl')
        '''

        self.checkin_data = pd.read_pickle(data_dir + 'Checkin.pkl')  # Saved processed data

    def generateevents(self):
        # Extracting features and group events by user
        self.features = ['month', 'date', 'hour', 'dayofweek', 'lat', 'lon']
        self.events = self.checkin_data.groupby('userID')[self.features].apply(lambda x: x.values.tolist())
        print("There are {} users".format(len(self.events)))
        self.lens = self.events.apply(len)
        print("Mean number of user checkin records:", np.mean(self.lens))

    def eventsprocess(self):
        # Calculating inter-event time for each user
        for user_events in self.events:
            first_event = True
            for event in user_events:
                if first_event:  # Skip each user's first event
                    first_event = False
                    event.append(0.)  # Treat its delta_t as zero; delta_t is considered as the last feature
                else:
                    time_delta = datetime.datetime(2012, int(event[0]), int(event[1]), int(event[2])) - \
                                 datetime.datetime(2012, int(last_event[0]), int(last_event[1]), int(last_event[2]))
                    event.append(time_delta.days * 24 + time_delta.seconds // 3600)
                last_event = event

    def addfields(self):
        # add the time from start for each events
        self.adjusted_events = self.events.copy(deep=True)

        for user_events in self.adjusted_events:
            first_event = True
            for event in user_events:
                if first_event:
                    first_event = False
                    event.append(0)
                    time_from_start = 0
                else:
                    time_from_start = time_from_start + int(event[6])
                    event.append(time_from_start)

    def extractfields(self):

        self.adjusted_events_st = []

        for user_events in self.adjusted_events:
            temp_events = []
            for event in user_events:
                event = event[4:7]
                temp_events.append(event)
            self.adjusted_events_st.append(temp_events)
        self.adjusted_events_st = pd.Series(self.adjusted_events_st)
        self.adjusted_events_st.index = self.events.index

    def scaling_data(self):
        self.data_orig = np.array(self.adjusted_events_st[self.lens.idxmax()])

        # Min-max scale the input and label data
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data_orig)
        #do not mark it as mark it as self and then see the data not change
        self.data_orig = self.scaler.transform(self.data_orig)

    def format_data(self):
        self.data_dictionary = []
        for i, event in enumerate(self.data_orig):
            temp_dict = {}
            temp_dict['idx_event'] = i
            temp_dict['latitude'] = event[0]
            temp_dict['longitude'] = event[1]
            temp_dict['time_since_start'] = event[2]
            temp_dict['type_event'] = 0
            temp_dict['time_since_last_event'] = event[2] # time gap
            self.data_dictionary.append(temp_dict)

    def createsequence(self):
        lookback = 10
        self.inputs = []
        # labels   = []
        for i in range(len(self.data_dictionary) - lookback - 1):
            self.inputs.append(self.data_dictionary[i:i + lookback])

    def create_test_val_trainset(self):
        test_portion = int(0.05 * len(self.inputs))
        val_portion = int(0.1 * len(self.inputs))
        train_x = self.inputs[:-val_portion]
        val_x = self.inputs[-val_portion:]
        test_x = val_x[-test_portion:]
        val_x = val_x[:-test_portion]
        with open(data_dir2 + 'train_fs.pkl', 'wb') as f:
            pickle.dump(train_x, f)

        with open(data_dir2 + 'val_fs.pkl', 'wb') as f:
            pickle.dump(val_x, f)

        with open(data_dir2 + 'test_fs.pkl', 'wb') as f:
            pickle.dump(test_x, f)
