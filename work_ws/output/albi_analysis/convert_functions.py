import datetime as dt
import pandas as pd
import time

def convert_string_to_time(df,col_name,format_):
    '''convert a string into time in format format_, if dataframe returns a dataframe with time column converted otherwise the object converted '''
    time_=[]
    time_stamp=[]
    if isinstance(df, pd):
        for string in df.col_name:
            if not np.isnan(string):
                t=time.mktime(dt.datetime.strptime(string,format_).timetuple()))#format should be consistent"%d/%m/%Y"
                t_=time.mktime(dt.datetime.strptime(string,format_))
                time_stamp.append(t)
                time_.append(t_)
            else:
                time_.append(np.isnan())
                time_stamp.append(np.isnan())
                continue
        df_.col_name=time_
        df_['time_stamp']=time_stamp
        del time_
        del time_stamp
    else:
        for string in df:
            if not np.isnan(string):
                t=time.mktime(dt.datetime.strptime(string,format_).timetuple()))#format should be consistent"%d/%m/%Y"
                t_=time.mktime(dt.datetime.strptime(string,format_))
                time_stamp.append(t)
                time_.append(t_)
            else:
                time_.append(np.isnan())
                time_stamp.append(np.isnan())
                continue
        df=time_
        del time_


    return df,time_stamp


