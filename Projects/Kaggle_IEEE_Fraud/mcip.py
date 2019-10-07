import numpy as np
from IPython.display import clear_output
import itertools as it
import pandas as pd
from tqdm import tqdm_notebook
from sklearn.neighbors import BallTree
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import multiprocessing


def replaceMissValMCIP(dfTrain,dfTest,cols,categorical,continuous,frac=0.01,radius=100,tol_val=0.999):
    
    
    dfTrainSubset = dfTrain[cols].sample(frac=frac,replace=True).reset_index(drop=True).copy()
    dfTestSubset = dfTest[cols].copy()
    
    while dfTrainSubset.isnull().values.all(axis=0).any():
        dfTrainSubset = dfTrain[cols].sample(frac=frac,replace=True).reset_index(drop=True).copy()
        
    #### mcip imputation
    numCores = 15#multiprocessing.cpu_count()
    
    ### create oldIndex
    dfTrainSubset.reset_index(inplace=True)
    dfTestSubset.reset_index(inplace=True)

    dfTrainSubset = dfTrainSubset.rename(index=int,columns={'index':'oldIndex'})
    dfTestSubset = dfTestSubset.rename(index=int,columns={'index':'oldIndex'})

    X = Parallel(n_jobs=numCores)(delayed(pipelineOfVariation)(caseInd, dfTrain=dfTrainSubset, dfTest=dfTestSubset,printOutput=False, \
                                                           tolerance_Value=tol_val,categorical=categorical,continuous=continuous, \
                                                         radius=radius, alpha=0.10, variations=False)
                                                  for caseInd in tqdm_notebook(range(len(dfTest)),leave=False))
    
    #X = [pipelineOfVariation(caseInd, dfTrain=dfTrainSubset, dfTest=dfTestSubset,printOutput=False, \
     #                                                      tolerance_Value=0.999,categorical=categorical,continuous=continuous, \
      #                                                     radius=200, alpha=0.20, variations=False)
       #                                            for caseInd in tqdm_notebook(range(len(dfTest)))]
    
    x = np.empty((0,dfTestSubset.shape[1]))
    
    for i in X:
        x = np.vstack((x,i))
    
    x = pd.DataFrame(data=x, columns=dfTestSubset.columns)
    
    dfTest[cols] = x[cols]
    
    return dfTest

def pipelineOfVariation(caseInd, dfTrain, dfTest, printOutput, tolerance_Value, categorical, continuous, radius, alpha, variations):
    """
    Method that perfroms a pipeline of imputing the data and returns imputed data
    caseInd: index of test case
    dfTrain: training set dataframe
    dfTest:test set dataframe
    printOutput: if for one case maybe want o print the ouput of number if similar cases
    tolerance_Value: represents the tolerance value
    categorical: list of categorical variables
    continuous: list of continuous variables
    radius: scalar that defines the radius of the hypersphere of the BallTree algorithm used to find similar cases
    alpha: the confidence interval value
    variations: boolean that represents whether to generate C.I.s for imputations
    
    Retruns:
    the imputed test case, in the case of missing continuous variables it returns the possible imputations of these continuous variables
    """
    
    nullMatrix = dfTest.isnull().values
    row = nullMatrix[caseInd,:]
    
    combs = getCombinations(row,dfTrain,tolerance_Value=tolerance_Value)
    
    if sum(row)>0: #if there are missing values
        dfAllNNs, _ , _ = getNNs(dfTrain, dfTest, combs, row, radius=radius, printOutput=printOutput, caseInd=caseInd)
    else:
        dfAllNNs = None

    x = getDatasetOfVariations(dfAllNNs, dfTest, row,caseInd=caseInd, categorical=categorical, continuous=continuous, alpha=alpha, variations=variations)
    
    if printOutput==False:
        clear_output()
    
    return x
    


def queryNN(X_train, X_test, radius, leaf_size):
    """
    Method that identifies from a dataset the NN most similar cases (Nearest neighbors).
    X_train: dataset to find neighbours
    X_test: dataset to find neighbors for
    BallTree_leaf_size: leaf size of kd tree
    radius: radius in high dimensional space to search for NNs
    
    Returns:
    counts: count of NNs for each datapoint
    indices: indices of NNs from dataset X_train
    """
    print(X_train.shape,X_test[0].shape)
    
    tree = BallTree(X_train, leaf_size=leaf_size) 
    counts = tree.query_radius(X_test, r=radius, count_only=True)
    indices = tree.query_radius(X_test, r=radius)
    print(counts,indices)
    return counts, indices

def train_test_split(df, testSetSize, extTestSetSize,external_validation, as_dataframe):
    """
    Method that splits a dataset into random training and test set or random training, test and external test set
    df: datafram that constins the data
    testSetSize: defines the size of the test size
    extTestSetSize: defines the size of the external test size
    external_validation: boolean if true generates an external test set
    as_dataframe: boolean if true returns training and test sets as pandas dataframes
    
    Returns:
    The training and test set or the training, test and external test set.
    """
    
    test_ind = np.random.choice(len(df), size= int(np.round(len(df)*testSetSize)),replace=False)
    
    df.reset_index(inplace=True)
    df = df.rename(index=int,columns={'index':'oldIndex'})
    train_ind = [i for i in range(len(df)) if i not in test_ind]
    
    if external_validation:
        extTest_ind = np.random.choice(len(train_ind), size= int(np.round(len(test_ind)*(1+extTestSetSize-testSetSize))),replace=False)
        extTest_ind = [train_ind[i] for i in extTest_ind]
        train_ind = [i for i in train_ind if i not in extTest_ind]
        
        cols = [i for i in df.columns]
        if as_dataframe:
            X_train = df.loc[train_ind,cols]
            X_train.reset_index(inplace=True,drop=True)
            X_test = df.loc[test_ind,cols]
            X_test.reset_index(inplace=True,drop=True)
            X_extTest = df.loc[extTest_ind,cols]
            X_extTest.reset_index(inplace=True,drop=True)
        else:
            X_train = df.loc[train_ind,cols].values
            X_test = df.loc[test_ind,cols].values
            X_extTest = df.loc[extTest_ind,cols].values
            
        return X_train, X_test, X_extTest
    else:
        cols = [i for i in df.columns]
        if as_dataframe:
            X_train = df.loc[train_ind,cols]
            X_train.reset_index(inplace=True,drop=True)
            X_test = df.loc[test_ind,cols]
            X_test.reset_index(inplace=True,drop=True)
        else:
            X_train = df.loc[train_ind,cols].values
            X_test = df.loc[test_ind,cols].values
            
        return X_train, X_test

def getVariablesCI(X,alpha):
    """
    Method that computes the mean of the NN and then uses that mean to define an interval based on a normal distr.
    Returns that interval.
    X: data
    alpha: value of C.I.
    """

    confs = []
    X = X.T

    for i in X:
        
        mean, sigma,conf_int = confidenceInterval(X= i[~np.isnan(i)],alpha=alpha)
        #mean, sigma = np.mean(X[indices,i]), np.std(X[indices,i])
        #conf_int = stats.norm.interval(alpha, loc=mean, scale=sigma)
        confs.append(conf_int)
        
    return confs

def getVariablesLI(X,alpha):
    """
    Method that can be used to define a "manual" interval of a variable based on the initial value of the datapoint
    X: data
    alpha: interval range
    
    Example:
    [1,2,3] if alpha  = 0.1 
    [0.9,1.1] is the interval of the first variable
    [1.9,2.1]
    [2.9,3.1]
    """
    
    confs = []

    for i in X.shape[0]:
        conf_int = np.array([X[i]-X[i]*alpha,X[:,i]+X[i]*alpha]) # +- percentage of variable value
        confs.append(conf_int)
        
    return confs

def confidenceInterval(X,alpha):
    """
    Method: that compute sthe C.I. of a normal distribution.
    
    X:data
    
    Returns:
    mean: mean of distribution
    sigma: standard deviation
    conf_int: the confidence interval
    """
    
    mean, sigma = np.mean(X), np.std(X)
    conf_int = stats.norm.interval(alpha, loc=mean, scale=sigma)
    
    return mean, sigma, conf_int

# function to create all combinations of variables intervals
def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def spaceSteps(step_size,confs):
    """
    Method: to compute all intervals in a linear steps.
    
    step_size: is the the linear step size for each interval.
    confs: is the set of confidence intervals to be used.
    
    Returns:
    intervals: vectors of the intervals for each variable
    """
    
    for i in range(0,len(confs)):
        conf_int = confs[i]
        if i==0:
            intervals = np.linspace(conf_int[0],conf_int[1],step_size)
        else:
            interval = np.linspace(conf_int[0],conf_int[1],step_size)
            intervals = np.column_stack((intervals,interval))
    
    return intervals

def getNNs(dfTrain, dfTest, combs, row, radius, printOutput, caseInd):
    
    """
    Method that identifies the similar cases of a test case and returns them.
    dfTrain: training dataframe
    dfTest: testing dataframe
    combs: combinations of complete variables
    row: boolean vector representing the missing variables
    radius: radius of the hypersphere used by the BallTree algorithm
    printOutput: boolean if True used to printhe indices and similar cases
    caseInd: integer index of the test case
    
    Returns:
    dfAllNNs: dataframe of Nearest Neighbors
    allCounts: count of similar cases
    allIndices: indices of all simialr cases
    """
    
    #######################################################################
    #create dataframe to get similar cases with replacement
    df = dfTrain.copy()
    df.set_index('oldIndex',inplace=True)

    #######################################################################
    allCounts = []
    allIndices = []
    dfAllNNs = pd.DataFrame()
    
    for i in combs:
        ##### get dataframe of training
        X_train = dfTrain[['oldIndex'] + i].dropna(axis=0)
        cols = [k for k in X_train.columns if k not in list(dfTrain.columns[row])]
        X_train = X_train[cols].values

        if X_train.shape[1]>1: #case in which too many missing values
            ##### get dataframe of testing
            X_test = dfTest.loc[caseInd,cols].values
            
            ##### normalize
            scaler = MinMaxScaler()
            scaler.fit(X_train[:,1:])
            X_train[:,1:] = scaler.transform(X_train[:,1:])
            
            print(X_train)

            X_test[1:] = scaler.transform(X_test[1:].reshape(1,-1))
            ##### get NNs
            counts, indices = queryNN(X_train[:,1:],[X_test[1:]],radius=radius*len(i)*0.1,leaf_size=10)

            allCounts.append(counts)
            allIndices.append(indices)

            ##### save NNs
            dfTemp = df.loc[np.asarray(X_train[list(indices[0]),0], dtype=int),:]
            dfTemp.reset_index(inplace=True) #reset index because index of df is oldIndex set up on top
            dfAllNNs = dfAllNNs.append(dfTemp,ignore_index=True)
        
    if printOutput:
        print('Number of NNs: ' , np.sum(allCounts))
        print('Indices: ' ,allIndices)
        
    return dfAllNNs, allCounts, allIndices

def booleanRow(columns, cols):
    """
    Method that gets two string lists and checks if cols elements are in columns, returns boolean vector that represents the element
    of columns in which the elemetn of cols is.
    
    columns: list of string
    cols: lis of strings
    
    Returns:
    Boolean list
    """
    
    boolRow = []
    for i in columns:
        if i in cols:
            boolRow.append(True)
        else:
            boolRow.append(False)

    return boolRow

def getDatasetOfVariations(dfAllNNs,dfTest, row, caseInd, categorical, continuous, alpha, variations):
    
    """
    Method that generates the dataset fo similar cases, called variations for the case of continuous variables as
    all possible cases that fall insed the range of similar cases values are returned.
    dfAllNNs: dataframe of all similar cases
    dfTest: test set dataframe
    row: boolean vector representing the variables with missing data as True
    caseInd: index of test case
    categorical: list of categorical variables
    continuous: list of continuous variables
    alpha: the confidence interval value
    
    Returns:
    A numpy array of the imputed test case
    """

    #######################################################################
    
    x = dfTest.loc[caseInd].values
    
    if sum(row)>0: #if there are missing values
        
        boolCategorical = booleanRow(dfAllNNs.columns,categorical)
        boolContinuous = booleanRow(dfAllNNs.columns,continuous)

        catColumns = np.logical_and(boolCategorical,row) #oldIndex not present in dfAllNNs
        contColumns = np.logical_and(boolContinuous,row)

        if (np.sum(catColumns)>0): 
            cols = dfAllNNs.columns[catColumns]
            freqValues = [dfAllNNs[i].value_counts().index[0] for i in cols]
            ######## impute categorical values
            ind = np.array(catColumns)
            x[ind] = freqValues
        if(np.sum(contColumns)>0):
            cols = dfAllNNs.columns[contColumns]
            confs = getVariablesCI(dfAllNNs[cols].values,alpha=alpha)

            if variations==True:
                ########### Define intervals in linear steps
                intervals = spaceSteps(step_size=10, confs=confs)
                ########### All combinations of variables intervals
                intsCombs = cartesian(intervals.T)

                ind = np.array(contColumns)

                if(intsCombs.shape[0] > 1):
                    x = np.tile(x, (intsCombs.shape[0],1))
                    x[:,ind] = intsCombs
                else:
                    x = np.tile(x, (intsCombs.shape[1],1))
                    x[:,ind] = intsCombs.T
            else:
                confs_mean = [np.mean(i) for i in confs]
                ind = np.array(contColumns)
                x[ind] = confs_mean

    return x

def getCombinations(row, df, tolerance_Value):
    """
    Method: computes all the combinations of the remaining complete variables of the test set, it uses a tolerance value in each iteration
    that represents the percentage of variables included to compute all of the possibel combinations.
    
    row: boolean row representing were there is a mising value in the variables
    df: training dataset
    tolerance_Value: tolerance value represents the tolerance to missing data
    
    Returns:
    All the combinations of possible variables based on the tolerance
    """
    
    cols = ['MRN_D','G_5yearscore','oldIndex'] + list(df.columns[row])
    arrays = [i for i in df.columns if i not in cols]
    combs = []
    for i in range(int(np.round(len(arrays)*tolerance_Value)),len(arrays)+1):
        combs = combs + [list(x) for x in it.combinations(arrays, i)]
    
    return combs