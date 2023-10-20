from datetime import datetime as dt
import numpy as np
import pandas as pd

def load_obs(observations_path):
    data = pd.read_csv(observations_path)
# create timedelta objects from report 'Start Time' and 'End Time' to help us compare times of reports
    # set dt_format to the format
    dt_format = '%m/%d/%Y %H:%M'
    # create new column in our data table called 'Start timedelta'
    # data['Temp Start Timedelta'] = [dt.strptime(startDate, dt_format) for startDate in data['Start Time']]
    # # replacing NaN values by 'Start Time' values
    # # data['End Time'] = np.where(pd.isna(data['End Time']), data['Start Time'], data['End Time'])
    # # create new column in our data table called 'End timedelta'
    # # data['Temp End Timedelta'] = [dt.strptime(endDate, dt_format) for endDate in data['End Time']]
    # data['Start Timedelta'] = np.where((data['Temp Start Timedelta'] <= data['Temp End Timedelta']),
    #                                    data['Temp Start Timedelta'],
    #                                    data['Temp End Timedelta'])
    # # data['End Timedelta'] = np.where((data['Temp Start Timedelta'] >= data['Temp End Timedelta']),
    # #                                  data['Temp Start Timedelta'],
    # #                                  data['Temp End Timedelta'])

    data['Start Timedelta'] = [dt.strptime(startDate, dt_format) for startDate in data['Start Time']]

    return data