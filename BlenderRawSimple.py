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

import BuildFeats as bf


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

def getSsRFC(IntTest, IntLabels, Countries, mode, NAME, lr):
    NumFeats = len(IntTest.columns) - 2
    print 'Number of features:', NumFeats
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
    RFC = xgb.XGBClassifier(n_estimators = N_trees, objective='multi:softprob', learning_rate = lr)
    RFC.fit(X_train.drop(['id', 'IsTest'], axis = 1).as_matrix(), y_train.drop('IsTest', axis = 1).as_matrix())
    create_feature_map(X_train.columns.drop(['id', 'IsTest']))
    importance = RFC.booster().get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True) 
    pd.DataFrame(importance).to_csv(NAME + 'FeatImp.csv', index = False)
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
    
def getData(folder, IntTest, ExtTest, TakeID):
    os.chdir('/Users/piotrgrudzien/Desktop/Airbnb/' + folder)
    cutIn = 12
    ##### Read all Int files
    for file in glob.glob('IntTestProbs*'):
        cutOff = len(file)-4
        FileRead = pd.read_csv(file)
        if(TakeID):
            IntTest['id'] = FileRead['id']
            IntTest['IsTest'] = FileRead['IsTest']
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

def blenderXGMain(FOLDER, fraction, namepart):

    RAW = 'RawResults/'
    NAME = RAW + namepart
    print 'Starting', datetime.datetime.now().time()
    warnings.filterwarnings("ignore")
    N_guesses = 5
    N_trees = 100
    ExtTest = pd.DataFrame()
    IntTest = pd.DataFrame()
    ExtResult = pd.DataFrame(columns = ('id', 'country'))
    
    Labels = ['TargetIsAU', 'TargetIsCA', 'TargetIsDE', 'TargetIsES', 'TargetIsFR', 'TargetIsGB', 'TargetIsIT', 'TargetIsNDF', 'TargetIsNL', 'TargetIsPT', 'TargetIsUS', 'TargetIsother']
    
    Countries = ['AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NDF', 'NL', 'PT', 'US', 'other']
    
    # Deal with the files being read in 'alphabetical' order
    Twisted_names = ['AU', 'CA', 'US', 'other', 'DE', 'ES', 'FR', 'GB', 'IT', 'NDF', 'NL', 'PT']
    
    os.chdir('/Users/piotrgrudzien/Desktop/Airbnb')
    Train = pd.read_csv('train_users_2.csv').drop('date_first_booking', axis=1)
    Train = Train.sample(frac = fraction).reset_index(drop = True)
    Test = pd.read_csv('test_users.csv').drop('date_first_booking', axis=1)
    for f in FOLDER:
        # remove duplicates
#        cfe = list(set(cfe))
        if(FOLDER.index(f) == 0):
            Train_Feats = bf.getFeats(Train, 'Train', f)
            Test_Feats = bf.getFeats(Test, 'Test', f)
        else:
            Train_Feats = pd.merge(Train_Feats, bf.getFeats(Train, 'Train', f), how='left', left_on = 'id', right_on = 'id')
            Test_Feats = pd.merge(Test_Feats, bf.getFeats(Test, 'Test', f), how='left', left_on = 'id', right_on = 'id')

        print 'Added folder', f, 'features:', len(Test_Feats.columns) - 1
        
        # Drop all the 'target' feats
    IntLabels = pd.DataFrame()
    IntLabels['id'] = Train_Feats['id']
    for col in Train_Feats.columns:
        if('Target' in col):
            IntLabels[col] = Train_Feats[col]
            Train_Feats.drop(col, axis = 1, inplace = True)
        
    # Add a column of zeros to Test if it appears in Train
    for col in Train_Feats.columns:
        if(col not in Test_Feats.columns):
            Test_Feats[col] = 0
            
    # Remove all columns from Test that don't appear in Train
    for col in Test_Feats.columns:
        if(col not in Train_Feats.columns):
            Test_Feats.drop(col, axis = 1, inplace = True)
            
    # Remove all id columns
    for col in Train_Feats.columns:
        if(('id' in col) & (col != 'id')):
            if(col in Train_Feats.columns):
                Train_Feats.drop(col, axis = 1, inplace = True)
    for col in Test_Feats.columns:
        if(('id' in col) & (col != 'id')):
            if(col in Test_Feats.columns):
                Test_Feats.drop(col, axis = 1, inplace = True)
       
    # And add the true label 
    Train_Feats['Target'] = Train['country_destination']
    
    Train_Feats = Train_Feats.sort(axis = 1)
    Test_Feats = Test_Feats.sort(axis = 1)
    Train_Feats.fillna(-1, inplace = True)
    Test_Feats.fillna(-1, inplace = True)
    
    # Array of all possible labels
    Labels = ['AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NDF', 'NL', 'PT', 'US', 'other']
    l = 'TargetIs'
    LabelIndex = [l + x for x in Labels]
    print 'Total number of labels:', len(LabelIndex)
    Remove = LabelIndex + ['TargetIsNull']
    
    FirstActiveYear = 'FirstActiveYear_x'
    FirstActiveMonth = 'FirstActiveMonth_x'
    Train_Feats['IsTest'] = ((Train_Feats[FirstActiveYear].isin([2014])) & (Train_Feats[FirstActiveMonth] > 3)).astype(int)
    


    #############
    IntLabels['IsTest'] = Train_Feats['IsTest']
    IntLabels['country'] = IntLabels.apply(getCountryID, axis = 1)

    
    IntTest = Train_Feats.drop('Target', axis = 1)
    ExtTest = Test_Feats
    
    print 'Number of features in Train:', len(IntTest.columns) - 2
    print 'Number of features in Test:', len(ExtTest.columns) - 1 
    
    RFC = getSsRFC(IntTest, IntLabels[['country', 'IsTest']], Countries, 'Int', NAME)    
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