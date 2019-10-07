############ fraud data pre-processing module
############ API for pre proc methods

################# Dependencies

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import decomposition
from sklearn.preprocessing import LabelEncoder



##### Feature selection methods

class Variable:
    """Class to describe a variable"""

    def __init__(self,data_series,col_name,label=None):
        """Method to intiate object.
        Inputs:
            data_series: pandas series Variable
            col_name: string with column's col_name
            label: data series of prediction
        """
        self.name = col_name
        self.top_10 = data_series.value_counts().iloc[:10]
        self.num_cats = len(data_series.unique())
        self.column_type = self.getColumnsType(col_name)
        self.type = str(data_series.dtype)
        self.num_nans = np.sum(data_series.isnull())
        self.nans_rate = self.num_nans/len(data_series)
        self.fraud_nans = np.sum(data_series[label==1].isnull())
        self.fraud_nans_rate = self.fraud_nans/np.sum(label)
        if str(data_series.dtype) == 'object':
            try:
                self.corr = self.correlation_ratio(data_series.values,
                                          label.values)
                self.p_val = 'NA'
            except:
                self.corr = 'Invalid variable'
                self.p_val = 'NA'
                pass
        else:
            self.average = data_series.dropna().values.mean()
            self.std = data_series.dropna().values.std()
            # self.max = data_series.dropna().values.max()
            # self.min = data_series.dropna().values.min()
            #self.median = data_series.dropna().values.median()

            try:
                pears_out = pearsonr(data_series, label)
                self.corr = pears_out[0]
                self.p_val = pears_out[1]
            except:
                self.corr = 'Invalid variable'
                self.p_val = 'NA'
                pass


    @staticmethod
    def getColumnsType(col_name):
        d={'M':'Match','V':'Vesta',
            'D':'Timedelta','C':'Counting',
            'c':'Card','a':'address','d':'distance','i':'ID'}
        initial=col_name[0]
        try:
            col_type=d[initial]
        except:
            col_type=col_name
        return col_type


    @staticmethod
    def correlation_ratio(categories, measurements):
        """Method to calculate the correlation ratio.

        Ths code is by:
        https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
        """
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat)+1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0,cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
        numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
        denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = np.sqrt(numerator/denominator)
        return eta


    # @staticmethod
    # def getColumnsType(col_name):
    #     """Determine whether categorigal or continuous"""
    #     if num_cats<20: categr
    #         return


class TableDescriptor:
    """Class to automatically describe every variable in a df"""

    def __init__(self,df,df_name,label_name=None):
        """Method to intiate object.
        Inputs:
            df: pandas dataframe
            df_name: string with df's col_name
            label_name: series name
        """
        if label_name == None:
            self.label = None
        else:
            self.label = df[label_name]
        self.variables = [Variable(df[col],col,
                                    label=self.label) for col in tqdm(df.columns)]


def getCompletedVars(td,nans_rate_cut_off=0.01,print_output=False):

    """Method to get low nans-rate getCorrelatedFeatures
    Inputs:
        td: table descriptor object
        nans_rate_cut_off: cut-off value

    Outputs:
        list of indices of td.variables,list of types of td.variables

    """

    features=[]
    for var in td.variables:
        try:
            if var.nans_rate < nans_rate_cut_off:
                if print_output:
                    print('Var ' + str(counter) + ': ',name,
                        var.type,var.nans_rate)
                features.append(var)

        except:
            print('Invalid corr value')
            pass
    print('Selected features: {0}/{1}'.format(len(features),len(td.variables)))
    return features


def getCorrelatedFeatures(variables,corr_cut_off=0.01,p_val_cut_off=None,
                            print_output=False):
    """Method to get correlated getCorrelatedFeatures
    Inputs:
        variables: list of variables
        corr_cut_off: correlation value
        p_val_cut_off: p-value cut off

    Outputs:
        list of indices of td.variables,list of types of td.variables

    """


    features=[]
    for var in variables:
        try:
            if var.corr > corr_cut_off:
                if print_output:
                    print('Var ' + str(counter) + ': ',name,
                        var.type,var.corr,var.p_val)
                features.append(var)

        except:
            print('Invalid corr value')
            pass
    print('Selected features: {0}/{1}'.format(len(features),len(variables)))
    return features







def getStratifiedTrainTestSplit(X,y,frac=0.2,n_splits=2,
                                random_state=0):

    """Method for stratified split.
    Inputs:
        X: input data in numpy n_array
        y: label numpy array
        frac=0.2: fraction size of test data
        n_splits=2: number of cross val splits
        random_state=0:randomizer seed
    Output:
        X_train, X_test, y_train, y_test: numpy arrays
    """

    sss = StratifiedShuffleSplit(n_splits=n_splits,
                                test_size=frac,
                                random_state=random_state)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test


class PCATransformer:
    """Utitlity Class for PCA."""

    def __init__(self,X,n_components=0.99):
        """Method to initiate object.
        Inputs:
            X: training data of shape (n,p)
            n_components: Number of components (l) to keep (see sklearn's PCA doc)
        """
        self.pca = decomposition.PCA(n_components).fit(X)
        self.weight_matrix = self.pca.components_ # np.array of shape: (l,p)

    def transform(self,X):
        """Method to transform a n-by-p matrix X by right-multiplication with weight_matrix.T
        Input:
           X: np.array of shape (n,p)
        Output: np.array of shape (n,l)
        """

        return self.pca.transform(X)

    #reconstruction error per data point
    def rec_error(self,X):
        """Method to compute the reconstruction error for each data vector of X
         Input:
            X: np.array of shape (n,p)
         Output: np.array of shape (n,)
        """
        reconstructed_points = np.mean(X,axis=0) + np.dot(self.transform(X),self.weight_matrix)
        return np.sum(((X-reconstructed_points)**2),axis=1)

    def tot_rec_error(self,X):
        """Method to compute the total reconstruction error of X
         Input:
            X: np.array of shape (n,p)
         Output: a float
        """
        return np.sum(self.rec_error(X))

#----------------------- Methods for Numerical and Categorical Data ----------------------


def numerical_categorical_split(variables,min_categories=300):
    """Method to split variables into a list of numerical vars and a list of categorical vars."""
    numerical_vars = [ var for var in variables \
                      if var.type != 'object' and var.num_cats > min_categories]
    categorical_vars = [ var for var in variables \
                        if var.type == 'object' or (var.type != 'object' and var.num_cats <= min_categories)]

    print('No of numerical features: {}'.format(len(numerical_vars)))
    print('No of categorical features: {}'.format(len(categorical_vars)))
    return numerical_vars,categorical_vars


def fill_nans(df,variables,feat_type='numerical'):
    """Method fill in NaNs of all vars in variables
    Input:
        - df: dataframe
        - variables: a list of Variable objects
        - feat_type: a string indicator of variables type, {'numerical','categorical'}
    Output:
        df is modified in-place as NaN values are filled in with:
        - column's mean if numerical (excluding NaNs)
        - column's most frequent value if categorical (excluding NaNs)
        Returns an alert.
    """
    for var in variables:
        if feat_type=='numerical':
            df[var.name] =df[var.name].fillna(var.average)

        if feat_type=='categorical':
            most_frequent = df[var.name].value_counts().idxmax()
            df[var.name] =df[var.name].fillna(most_frequent)

    if feat_type=='numerical':
        return "NaNs have been filled in with column's mean value."
    if feat_type=='categorical':
        return "NaNs have been filled in with column's most frequent value."




def to_categorical(df,cat_cols,how='dummies'):

    """Method to convert non-numerical columns to categorical.
    Inputs:
        df: dataframe
        features: list of feat/column names
        feat_types: type of data of each corresponding feat/column
    Output:
        df with non-numerical converted to categorical
    """
    if how == 'dummies':
        df = pd.get_dummies(data=df,columns=cat_cols)
    if how == 'label_enc':
        for col in cat_cols:
            lbl = LabelEncoder()
            lbl.fit(list(df[col].values))
            df[col] = lbl.transform(list(df[col].values))

    return df









#--------------------- Import Data Methods -------------------------------------------

def import_data(path,nrows=None,reduce_mem=True):
    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(path,nrows=nrows)

    if reduce_mem:
        df = reduce_mem_usage(df)

    return df

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def print_null_cols(df):
    """Method to check if there are null values in df."""

    i=0
    for col in df.columns:
        if np.sum(df[col].isnull())!=0:
            print(col)
            i+=1
    if i==0:
        return "No null columns."
