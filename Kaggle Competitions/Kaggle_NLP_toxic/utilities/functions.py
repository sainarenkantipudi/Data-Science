import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from .constants import CLASSES
import pickle
from sklearn.metrics import log_loss, roc_auc_score

#---------------------- Pickle --------------------------------

def dump_objects(path, X):

    with open(path, 'wb') as f:
        pickle.dump(X, f)

def load_objects(path):

    with open(path, 'rb') as f:
        return pickle.load(f)

#----------------------- TF-IDF -------------------------------

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]

    return pd.DataFrame(data=top_feats, columns=['feature', 'tfidf'])

def top_feats_in_doc(X, features, row_id, top_n=25):
    ''' Get top n tfidf values in a specific document (matrix row) 
        and return them with their corresponding feature names.'''
        
    # Remove single-dimensional entries from the shape of an array, e.g. (1,n)->(n,) 
    row = np.squeeze(X[row_id].toarray())
    
    return top_tfidf_feats(row, features, top_n=25)
    
def top_mean_feats(X, features, ids, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in ids. '''
    
    # Select docs
    D = X[ids].toarray()

    # Set the tfidf score of words with less than min_tfidf to zero.
    D[D < min_tfidf] = 0
    
    # Compute mean tfidf score of each document
    tfidf_means = np.mean(D, axis=0)
    
    
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(df_train, X, features, min_tfidf=0.1, top_n=10):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
     
    dfs = []
    for col in CLASSES:
        ids = df_train.index[df_train[col] == 1]
        feats_df = top_mean_feats(X, features, ids, min_tfidf, top_n)
        feats_df.name = col
        dfs.append(feats_df)
        
    return dfs


def plot_top_ngrams_by_label(top_ngrams_by_label, top_n=10, vec_name='TF-IDF', ngram_type='unigrams'):
    ''' Plot barplots for the top ngrams, based on their mean tfidif value, for each class label'''

    fig, axs = plt.subplots(3,2, figsize=(20,20))
    my_pal = sns.color_palette(n_colors=10)
    plt.suptitle(f"Top {vec_name} {ngram_type} per class label",fontsize=20)
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        df_label = top_ngrams_by_label[i]
        sns.barplot(df_label.feature.iloc[:top_n],
                    df_label.tfidf.iloc[:top_n],
                    color=my_pal[i],
                    ax=ax)
        ax.set_title(f"Class Label : {df_label.name.upper()}", fontsize=18)
        ax.xaxis.label.set_visible(False)
        ax.set_ylabel('TF-IDF score', fontsize=15)
        ax.tick_params(axis='x', which='major', labelsize=15)


#------------------------- sklearn/scorers ----------------------------

def mean_log_loss(y, y_pred):
    """Computes the log_loss for each class label and returns the mean loss"""
    losses = []
    for i in range(len(y_pred)):
        losses.append(log_loss(y[:,i],y_pred[i][:,1]))
    
    return np.mean(losses)

def mean_roc_auc(y, y_pred):
    """Computes the roc_auc_score for each class label and returns the mean score"""
    scores = []
    for i in range(len(y_pred)):
        scores.append(roc_auc_score(y[:,i], y_pred[i][:,1]))
    
    return np.mean(scores)


#-------------------------- Manual GridSearchCV utils --------------------------------

def find_best_params(df, param_names):
    # argmax for loss values because we need to find the least negative value
    id_min_loss = df.loss.values.argmax()
    id_max_roc_auc = df.roc_auc.values.argmax()
    print('Best Min Loss Parametes')
    for name in param_names:
        print("{} = {}".format(name, df.loc[id_min_loss, name]))
   
    print('Best Max ROC_AUC Parametes')
    for name in param_names:
        print("{} = {}".format(name, df.loc[id_max_roc_auc, name]))
   

def find_best_scores(df):
    df = df.loc[:,['loss', 'roc_auc']]
    # argmax for loss values because we need to find the least negative value
    min_loss, roc_auc = df.iloc[df.loss.values.argmax()].values
    loss, max_roc_auc = df.iloc[df.roc_auc.values.argmax()].values
    print("Min LogLoss: {:2f} (roc-auc: {:2f})".format(-min_loss, roc_auc))
    print("Max ROC-AUC: {:2f} (logloss: {:2f})".format(max_roc_auc, -loss))
