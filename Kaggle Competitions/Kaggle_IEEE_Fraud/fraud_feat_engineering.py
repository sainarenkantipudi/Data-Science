import pandas as pd
import numpy as np
import datetime


# ---------------------Methods for Adding Datetime Features ---------------------


def addDatetimeFeats(df):
    """Method to convert timedelta's of TransactionDT to datetime object and add datetime-like columns to df:
    - month
    - week
    - yearday
    - hour
    - weekday
    - day

    Input: train/test transactions dataframe

    Output: New cols are added to df and a list of the names of all new cols is returned.
    """
    START_DATE=datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

    #Convert Timedeltas to datetime objects
    df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

    #Add New features
    df['month'] = df['TransactionDT'].dt.month
    df['week'] = df['TransactionDT'].dt.weekofyear
    df['yearday'] = (df['TransactionDT'].dt.year-2017)*365 + df['TransactionDT'].dt.dayofyear
    df['hour'] = df['TransactionDT'].dt.hour
    df['weekday'] = df['TransactionDT'].dt.dayofweek
    df['day'] = df['TransactionDT'].dt.day

    df.drop('TransactionDT',axis=1,inplace=True)

    print("Datetime-like features added to dataframe:\n")
    print(['month','week','yearday','hour','weekday','day'])
    return ['month','week','yearday','hour','weekday','day']


# ---------------------Methods for Adding Interaction Features ---------------------

def addCardAddressInteractionFeats(df):
    """Method to add interaction features by ADDING the values of card_ and addr_ columns.

     Input:
      - df: train/test transactions dataframe
     Output:
     New cols are added to df and a list of the names of all new cols is returned.
    """
    new_feats=[]
    try:
        df['card12'] = df['card1'].astype(str)+'_'+df['card2'].astype(str)
        new_feats.append('card12')
    except:
        pass

    try:
        df['card1235'] = df['card12'].astype(str)+'_'+df['card3'].astype(str)+'_'+df['card5'].astype(str)
        new_feats.append('card1235')
    except:
        pass

    try:
        df['card1235_addr12'] = df['card1235'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['addr2'].astype(str)
        new_feats.append('card1235_addr12')
    except:
        pass

    print("Interaction features added to dataframe:\n")
    print(new_feats)

    return new_feats



def addDatetimeInteractionFeats(df,cols,period_cols):
    """Method to add interaction features by ADDING the values of cols and period_cols
       and computing value frequencies

       Input:
        - df: train/test transactions dataframe
        - cols: list of columns
        - period_cols: list of datetime-like cols

       Output:
       New cols are added to df and a list of the names of all new cols is returned.
       """

    new_feats = []
    for col in cols:
        for period in period_cols:

            try:
                #create temporary column by concatenating str(values) of col and period
                df['temp_col'] = df[col].astype(str) + '_' + (df[period]).astype(str)

                #add new column of value frequencies of temp_col
                freq_dict = df['temp_col'].value_counts().to_dict()
                df[f'{col}_{period}'] = df['temp_col'].map(freq_dict)

                #collect new_feats
                new_feats.append(f'{col}_{period}')
            except:
                pass

    #delete temp_col
    df.drop('temp_col',axis=1,inplace=True)

    print("Interaction features added to dataframe: {}\n".format(len(new_feats)))
    print(new_feats)
    return new_feats


# ---------------------Methods for Adding Aggregated Features ---------------------

def addAggTransAmtFeats(df,cols):
    """Method to add aggregated features by grouping-by col in cols and computing the mean & STD of 'TransactionAmt'

     Input:
      - df: train/test transactions dataframe
      - cols: list of columns

     Output:
     New cols are added to df and a list of the names of all new cols is returned.
     """


    new_feats = []
    for col in cols:
        try:
            #compute aggregated stats of 'TransactionAmt' with col
            temp_df = df[[col,'TransactionAmt']].groupby(col)['TransactionAmt'].agg(['mean','std'])
            temp_df['mean'] = temp_df['mean'].fillna(temp_df['mean'].mean()).replace(np.inf,999)
            temp_df['std'] = temp_df['std'].fillna(temp_df['std'].mean()).replace(np.inf,999)

            #prepare dicts of (col,mean/std) value pairs
            mean_dict = temp_df['mean'].to_dict()
            std_dict = temp_df['std'].to_dict()


            #add two new columns to df
            df[f'TransAmt_{col}_mean']=df[col].map(mean_dict)
            df[f'TransAmt_{col}_std']=df[col].map(std_dict)

            #collect new_feats
            new_feats.extend([f'TransAmt_{col}_mean',f'TransAmt_{col}_std'])

        except:
            pass

    print("Aggregated TransactionAmt features added to dataframe: {}\n".format(len(new_feats)))
    print(new_feats)
    return new_feats




# ---------------------Methods for Indicator Features ---------------------


def addFrequencyFeats(df,cols):
    """Method to add indicator features by computing the value frequencies of cols'.

    Input:
     - df: train/test transactions dataframe
     - cols: list of columns

    Output: New cols are added to df and a list of the names of all new cols is returned
    """
    new_feats = []
    for col in cols:

        try:
            #compute frequency of each value in col
            frequency_dict = df[col].value_counts(dropna=False).to_dict()

            #add a new col to df
            df[f'{col}_freq'] = df[col].map(frequency_dict)

            #collect new feats
            new_feats.append(f'{col}_freq')

        except:
            pass

    print("Frequency features added to dataframe: {}\n".format(len(new_feats)))
    print( new_feats)
    return new_feats