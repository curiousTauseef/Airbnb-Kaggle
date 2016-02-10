# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:24:59 2015

@author: Piotrek
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import datetime

import pandas as pd
import numpy as np
import glob
import os
import warnings
import operator

N_guesses = 5
N_trees = 100

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

def getSingleScore(row):
    return row.tolist().index(row['label'])

def getReverseOrder(row):
    return np.argsort(row)    
    
def scorePreds(Preds, labels, output, NumFeats):
    labels = labels.reset_index(drop = True)
    Preds = Preds.apply(getReverseOrder, axis = 1)
    Together = Preds[Preds.columns[::-1][:N_guesses]]
    Together['label'] = labels
    Together['score'] = Together.apply(getSingleScore, axis = 1)
    Result = 0
    DivideBy = 0
    for index, row in Together.iterrows():
#        print 'Score:', row['score']
        DivideBy += 1
        if(row['score'] < N_guesses):
#            print 'Score:', row['score']
            Result += np.true_divide(1, np.log2(row['score'] + 2))
#            print 'Adding:', np.true_divide(1, np.log2(row['score'] + 2))
    return np.true_divide(Result, DivideBy)

def shuffle(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df
    
def writeGuesses(Preds, y_test, X_test):
    # Only doing one guess so the score is just a proportion of correct answers
    Guesses = X_test.copy()
    Guesses['True'] = y_test
    Guesses['Guess'] = Preds
    Guesses['Score'] = (y_test == Preds).astype(int)
    Guesses.to_csv('Guesses.csv', index = False)
    print 'Guesses written to CSV'
def soften(row):
    T = 1
    return np.true_divide(np.exp(np.true_divide(row, T)), np.sum(np.exp(np.true_divide(row, T))))

def getSsRFC(IntTest, IntLabels, Countries, mode, NAME, ss):
    NumFeats = len(IntTest.columns) - 2
    print 'Number of features in the second layer model:', NumFeats
######## This is just internal testing
    X_train, X_test, y_train, y_test = train_test_split(IntTest, IntLabels, test_size=0.2)
#         
    RFC = xgb.XGBClassifier(subsample = ss, n_estimators = N_trees, objective='multi:softprob')
    RFC.fit(X_train.drop(['id'], axis = 1).as_matrix(), y_train.as_matrix())
    create_feature_map(X_train.columns.drop(['id']))
    importance = RFC.booster().get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True) 
    pd.DataFrame(importance).to_csv(NAME + 'FeatImp.csv', index = False)    
    Preds = pd.DataFrame(RFC.predict_proba(X_test.drop(['id'], axis = 1)))
    pd.DataFrame(Preds).to_csv(NAME + 'IntProbs.csv', index = False)
    print 'Internal test result:', scorePreds(Preds, y_test, True, NumFeats)
    
######## This is training model for submission
    RFC.fit(IntTest.drop(['id'], axis = 1).as_matrix(), IntLabels.as_matrix())
    
    return RFC
    
def getCountryID(row):
    # Niezly hak!
    if(list(row).index(1) == 0):
        row[0] = 2
    return list(row).index(1) - 1
    
def getData(folder, IntTest, ExtTest, TakeID):
    os.chdir('/Users/piotrgrudzien/Desktop/Airbnb/' + folder)
    cutIn = 12
    ##### Read all Int files
    for file in glob.glob('IntTestProbs*'):
        cutOff = len(file)-4
        FileRead = pd.read_csv(file)
        if(TakeID):
            IntTest['id'] = FileRead['id']
        IntTest[file[cutIn:cutOff] + folder] = FileRead[FileRead.columns[2]]
    ##### Read all Ext
    for file in glob.glob('ExtTestProbs*'):
        cutOff = len(file)-4
        FileRead = pd.read_csv(file)
        if(TakeID):
            ExtTest['id'] = FileRead['id']
            TakeID = False
        ExtTest[file[cutIn:cutOff] + folder] = FileRead[FileRead.columns[2]]
    return IntTest, ExtTest, TakeID

def blenderXGMain(FOLDERS, ss, N_trees):
    print 'Starting', datetime.datetime.now().time()
    warnings.filterwarnings("ignore")
    N_guesses = 5
#    N_trees = 100
    ExtTest = pd.DataFrame()
    IntTest = pd.DataFrame()
    ExtResult = pd.DataFrame(columns = ('id', 'country'))
    NAME = 'ss' + str(ss) + 'Trees' + str(N_trees)
    
    Labels = ['id', 'TargetIsAU', 'TargetIsCA', 'TargetIsDE', 'TargetIsES', 'TargetIsFR', 'TargetIsGB', 'TargetIsIT', 'TargetIsNDF', 'TargetIsNL', 'TargetIsPT', 'TargetIsUS', 'TargetIsother']
    
    Countries = ['AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NDF', 'NL', 'PT', 'US', 'other']
    
    # Deal with the files being read in 'alphabetical' order
    Twisted_names = ['AU', 'CA', 'US', 'other', 'DE', 'ES', 'FR', 'GB', 'IT', 'NDF', 'NL', 'PT']
    
    TakeID = True
    
    for folder in FOLDERS:
        print 'Folder:', folder
        IntTest, ExtTest, TakeID = getData(folder, IntTest, ExtTest, TakeID)
        NAME = NAME + folder 

    #############
    
    IntLabels = pd.read_csv('IntTestLabels.csv')[Labels]
    print('Targets read from CSV')
    
    IntLabels['country'] = IntLabels.apply(getCountryID, axis = 1)
    
    RFC= getSsRFC(IntTest, IntLabels[['country']], Countries, 'Int', NAME, ss)
    print 'Second layer RFC trained'
    
    del IntTest
    
    Probs = RFC.predict_proba(ExtTest.drop('id', axis = 1))
    
    pd.DataFrame(Probs).to_csv(NAME + 'ExtProbs.csv', index = False)
    print 'Probs written to CSV'
    
    id_test = ExtTest['id']
    
    le = LabelEncoder()
    ids = []
    cts = []
    le.fit(Twisted_names)
    for i in range(len(Probs)):
        df = pd.DataFrame(columns = range(0, 12))
        df.loc[i, :] = Probs[i, :]
        Trans1 = df.apply(getReverseOrder, axis = 1)
        Trans2 = Trans1[Trans1.columns[::-1][:N_guesses]]
        idx = id_test[i]
        ids += [idx] * N_guesses
        cts += le.inverse_transform(Trans2).tolist()[0]
    
    ExtResult = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
    
    ExtResult.to_csv(NAME + '.csv', index = False)
    print 'External result written to CSV!'
    print datetime.datetime.now().time()