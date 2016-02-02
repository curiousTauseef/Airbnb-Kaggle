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

N_guesses = 5
N_trees = 100


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
    if(output):
        Together.to_csv('Together' + str(NumFeats) + 'Feats.csv', index = False)
        print 'Together written to CSV'
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

def getSsRFC(IntTest, IntLabels, Countries, mode, NAME):
    NumFeats = len(IntTest.columns) - 2
    print 'Number of features in the second layer model:', NumFeats
######## This is just internal testing
#    X_train, X_test, y_train, y_test = train_test_split(IntTest, IntLabels, test_size=0.2)
    X_train = IntTest.loc[IntTest['IsTest'] == 0, :]
    y_train = IntLabels.loc[IntLabels['IsTest'] == 0, :]
    X_test = IntTest.loc[IntTest['IsTest'] == 1, :]
    y_test = IntLabels.loc[IntLabels['IsTest'] == 1, :]
#    K = 5
#    CV_Res = []
#    for i in range(0, K):
##        RFC = RandomForestClassifier(n_estimators = N_trees)
#        RFC = xgb.XGBClassifier(n_estimators = N_trees, objective='multi:softprob')
#        print 'Fitting the model...', datetime.datetime.now().time()
#        RFC.fit(X_train[0:i * len(X_train) / K].append(X_train[(i + 1) * len(X_train) / K:len(X_train)]).drop('id', axis = 1), y_train[0:i * len(X_train) / K].append(y_train[(i + 1) * len(X_train) / K:len(X_train)]))    
#        Preds = pd.DataFrame(RFC.predict_proba(X_train[i * len(X_train) / K:(i + 1) * len(X_train) / K].drop('id', axis = 1)))
#        #Preds['id'] = X_train.loc[i * len(X_train) / K:(i + 1) * len(X_train) / K, 'id']
#        #Now just implement this function which gets scores from Preds!!!
#        Current_Result = scorePreds(Preds, y_train[i * len(X_train) / K:(i + 1) * len(X_train) / K], False, NumFeats)
#        print 'Current result:', Current_Result
#        CV_Res.append(Current_Result)
#    print 'CV results:', CV_Res           
    RFC = xgb.XGBClassifier(n_estimators = N_trees, objective='multi:softprob')
    RFC.fit(X_train.drop(['id', 'IsTest'], axis = 1).as_matrix(), y_train.drop('IsTest', axis = 1).as_matrix())
    Preds = pd.DataFrame(RFC.predict_proba(X_test.drop(['id', 'IsTest'], axis = 1)))
    pd.DataFrame(Preds).to_csv(NAME + 'IntProbs.csv', index = False)
    print 'Internal test result:', scorePreds(Preds, y_test.drop('IsTest', axis = 1), True, NumFeats)
    
    
####### Write guesses to CSV for analysis
#    writeGuesses(RFC.predict(X_test.drop('id', axis = 1)), y_test, X_test.drop('id', axis = 1))
    
######## This is training model for submission
    RFC.fit(IntTest.drop(['id', 'IsTest'], axis = 1).as_matrix(), IntLabels.drop('IsTest', axis = 1).as_matrix())
    
    return RFC
    
def getCountryID(row):
    # Niezly hak!
    if(list(row).index(1) == 0):
        row[0] = 2
    return list(row).index(1) - 1

def blenderXGMain(FOLDER1, FOLDER2):

    print 'Starting', datetime.datetime.now().time()
    
    
    N_guesses = 5
    N_trees = 100
    THRESH_GLOBAL = 0
    ExtTest = pd.DataFrame()
    IntTest = pd.DataFrame()
    ExtResult = pd.DataFrame(columns = ('id', 'country'))
    
    Labels = ['id', 'TargetIsAU', 'TargetIsCA', 'TargetIsDE', 'TargetIsES', 'TargetIsFR', 'TargetIsGB', 'TargetIsIT', 'TargetIsNDF', 'TargetIsNL', 'TargetIsPT', 'TargetIsUS', 'TargetIsother']
    
    Countries = ['AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NDF', 'NL', 'PT', 'US', 'other']
    
    # Deal with the files being read in 'alphabetical' order
    Twisted_names = ['AU', 'CA', 'US', 'other', 'DE', 'ES', 'FR', 'GB', 'IT', 'NDF', 'NL', 'PT']
    
    #for i in range(0, 13):
    #    ExtTest[Labels[i]] = pd.read_csv('ExtTestProbs' + str(i) + '.csv')[Labels[i]]
    #    IntTest[Labels[i]] = pd.read_csv('IntTestProbs' + str(i) + '.csv')[Labels[i]]
    #    print 'Read files', str(i)
        
    IntNoID = True
    ExtNoID = True
    ############# Add features from a FOLDER1
#    FOLDER1 = 'Times'
    os.chdir('/Users/piotrgrudzien/Desktop/Airbnb/' + FOLDER1)
    RandomSeed1 = np.random.rand(len(glob.glob('IntTestProbs*')), 1)
    index = 0
    thresh = THRESH_GLOBAL
    counter = 0
    for file in glob.glob('IntTestProbs*'):
        if(RandomSeed1[index] > thresh):
            counter += 1
            cutIn = 12
            cutOff = len(file)-4
            FileRead = pd.read_csv(file)
            if(IntNoID):
                IntTest['id'] = FileRead['id']
                IntTest['IsTest'] = FileRead['IsTest']
                IntNoID = False
            IntTest[file[cutIn:cutOff]] = FileRead[FileRead.columns[2]]
    #        print 'Reading file', file 
        index += 1
    print 'Features from folder', FOLDER1, ':', counter
    NumFeats1 = counter
    NAME = FOLDER1 + str(NumFeats1) + 'Trees' + str(N_trees)
    ############# Add features from FOLDER2
#    FOLDER2 = 'FullStats'
    if(FOLDER2 is not None):
        os.chdir('/Users/piotrgrudzien/Desktop/Airbnb/' + FOLDER2)
        RandomSeed2 = np.random.rand(len(glob.glob('IntTestProbs*')), 1)
        index = 0
        thresh = THRESH_GLOBAL
        counter = 0
        for file in glob.glob('IntTestProbs*'):
            if(RandomSeed2[index] > thresh):
                counter += 1
                cutIn = 12
                cutOff = len(file)-4
                FileRead = pd.read_csv(file)
                IntTest[file[cutIn:cutOff] + 'second'] = FileRead[FileRead.columns[2]]
        #        print 'Reading file', file 
            index += 1
        print 'Features after adding from folder', FOLDER2, ':', counter
        NumFeats2 = counter
        NAME = FOLDER1 + str(NumFeats1) + FOLDER2 + str(NumFeats2) + 'Trees' + str(N_trees)
    #############
    
    IntLabels = pd.read_csv('IntTestLabels.csv')[Labels + ['IsTest']]
    print('Targets read from CSV')
    
    IntLabels['country'] = IntLabels.apply(getCountryID, axis = 1)
    
    RFC= getSsRFC(IntTest, IntLabels[['country', 'IsTest']], Countries, 'Int', NAME)
    print 'Second layer RFC trained'
    
    del IntTest
    
    ############### Add Ext features from FOLDER1
    os.chdir('/Users/piotrgrudzien/Desktop/Airbnb/' + FOLDER1)
    index = 0
    counter = 0
    thresh = THRESH_GLOBAL
    print 'Reading Ext files...'
    for file in glob.glob('ExtTestProbs*'):
        if(RandomSeed1[index] > thresh):
            counter += 1
            cutIn = 12
            cutOff = len(file)-4
            FileRead = pd.read_csv(file)
            if(ExtNoID):
                ExtTest['id'] = FileRead['id']
                ExtNoID = False
            ExtTest[file[cutIn:cutOff]] = FileRead[FileRead.columns[2]]
        #    print 'Reading file', file
        index += 1
    print 'Added', str(counter), 'Ext features from', FOLDER1
    ############# Add Ext features from FOLDER2
    if(FOLDER2 is not None):
        os.chdir('/Users/piotrgrudzien/Desktop/Airbnb/' + FOLDER2)
        index = 0
        thresh = THRESH_GLOBAL
        counter = 0
        for file in glob.glob('ExtTestProbs*'):
            if(RandomSeed2[index] > thresh):
                counter += 1
                cutIn = 12
                cutOff = len(file)-4
                FileRead = pd.read_csv(file)
                ExtTest[file[cutIn:cutOff] + 'second'] = FileRead[FileRead.columns[2]]
        #        print 'Reading file', file 
            index += 1
        print 'Added', str(counter), 'Ext features from', FOLDER2
    #############
    
    print 'Finished reading Ext files...'
    
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