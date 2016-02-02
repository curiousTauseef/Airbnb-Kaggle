# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:44:34 2015

@author: Piotrek
"""
import numpy as np
import pandas as pd
import BuildFeats as bf
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import datetime
import random

def enrich(Feats, Targets, index):
    TargetSize = len(Feats.loc[Targets[index] == 1, :])
    RandomSize = len(Feats.loc[Targets[index] == 0, :])
# Reduced to 0.7 because of memory errors, will get better when you get rid of rubbish columns
    EnrichSize = [19, 19, 19, 19, 12, 19, 19, 0.7, 19, 99, 1, 9]
    Replace = int(TargetSize * EnrichSize[LabelIndex.index(index)]) > RandomSize
    ThisOnly = random.randint(0, 10000)
    
    EnrichedFeats = Feats.loc[Targets[index] == 1, :]
    EnrichedFeats = EnrichedFeats.append(Feats.loc[Targets[index] == 0, :].sample(int(TargetSize * EnrichSize[LabelIndex.index(index)]), random_state = ThisOnly, replace = Replace))
    
    EnrichedTargets = Targets.loc[Targets[index] == 1, :]
    EnrichedTargets = EnrichedTargets.append(Targets.loc[Targets[index] == 0, :].sample(int(TargetSize * EnrichSize[LabelIndex.index(index)]), random_state = ThisOnly, replace = Replace))

    print 'Label', index, ': training size reduced from', len(Feats), 'to', len(EnrichedFeats)
    return EnrichedFeats, EnrichedTargets
    
def enrichPairwise(Feats, Targets, index1, index2):
    print 'Index1 true size:', len(Feats.loc[Targets[index1] == 1, :])
    print 'Index2 true size:', len(Feats.loc[Targets[index2] == 1, :])
    EnrichedFeats = Feats.loc[Targets[index1] == 1, :]
    OutSize = len(EnrichedFeats)
    if(OutSize > 87000):
        OutSize = int(0.7 * OutSize)
    Replace = OutSize > len(Feats.loc[Targets[index2] == 1, :])
    # So that feats and targets share the random seed
    ThisOnly = random.randint(0, 10000)
    EnrichedFeats = EnrichedFeats.append(Feats.loc[Targets[index2] == 1, :].sample(OutSize, random_state = ThisOnly, replace = Replace))
    EnrichedTargets = Targets.loc[Targets[index1] == 1, :]
    EnrichedTargets = EnrichedTargets.append(Targets.loc[Targets[index2] == 1, :].sample(OutSize, random_state = ThisOnly, replace = Replace))
    print 'Labels', index1, 'with', index2, ': training size reduced from', len(Feats), 'to', len(EnrichedFeats)
    return EnrichedFeats, EnrichedTargets

def trainModel(Feats, Labels, Feats_test, Labels_test):
    print 'Training model...'
    RFC = RandomForestClassifier(n_estimators = 500, max_features = 200)
    RFC.fit(Feats, Labels)
    importances = RFC.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    #print 'Model score:', RFC.score(Feats_test, Labels_test)
    print("Feature ranking:")
    for f in range(0, 50):
        print("%d. feature %s (%f)" % (f + 1, Feats.columns[indices[f]], importances[indices[f]]))
    return RFC

#AgeGender = pd.read_csv('age_gender_bkts.csv')
#Countries = pd.read_csv('countries.csv')
#Sessions = pd.read_csv('sessions.csv')
    
#For now drop 'date_first_booking' - use it later with sessions maybe
Train = pd.read_csv('train_users_2.csv').drop('date_first_booking', axis=1)
Test = pd.read_csv('test_users.csv').drop('date_first_booking', axis=1)
Train_Feats = bf.getFeats(Train, 'Train')
Test_Feats = bf.getFeats(Test, 'Test')
##
# Things that appear in the training set and not in the test set: add a column of all zeros
for col in Train_Feats.columns:
    if(col not in Test_Feats.columns):
        Test_Feats[col] = np.zeros(Test_Feats['id'].shape)
        
# Things that appear in the test set and not in the training set: remove these columns
for col in Test_Feats.columns:
    if(col not in Train_Feats.columns):
        print 'Dropping:', col
        Test_Feats.drop(col, axis = 1, inplace = True)

# Get rid of the columns left after the sessions join
Train_Feats.drop(['user_id_x', 'user_id_y'], axis = 1, inplace = True)
Test_Feats.drop(['user_id_x', 'user_id_y'], axis = 1, inplace = True)

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

#print 'Numer of features used:', len(Train_Feats.columns) - 1

FOLDER = 'Testieren'

pd.read_csv(FOLDER + '\\y_test.csv').to_csv(FOLDER + '/IntTestLabels.csv')
# Train a separate classifier for each label
for li in LabelIndex:
    Probs = pd.DataFrame()
    Ext_Test_Probs = pd.DataFrame()
    print datetime.datetime.now().time()
    print 'Training model no', LabelIndex.index(li)
    # enrich models in their data of interest
    Enriched_X_train, Enriched_y_train = enrich(pd.read_csv(FOLDER + '\\X_train.csv'), pd.read_csv(FOLDER + '\\y_train.csv'), li)
    # Only for reference - for scoring models
    #Brain = trainModel(Enriched_X_train.drop('id', axis = 1), Enriched_y_train[li], X_test.drop('id', axis = 1), y_test[li])
    Brain = trainModel(Enriched_X_train.drop('id', axis = 1), Enriched_y_train[li], None, None)
    print datetime.datetime.now().time()
    print 'Getting probs for model no', LabelIndex.index(li)
    Probs['id'] = pd.read_csv(FOLDER + '\\X_test.csv')['id']
    Probs[li] = Brain.predict_proba(pd.read_csv(FOLDER + '\\X_test.csv').drop('id', axis = 1))[:, 1]
    # External data
    Ext_Test_Probs['id'] = pd.read_csv(FOLDER + '\\Test_Feats.csv')['id']
    Ext_Test_Probs[str(li)] = Brain.predict_proba(pd.read_csv(FOLDER + '\\Test_Feats.csv').drop('id', axis = 1).drop(Remove, axis = 1))[:, 1]
    FileNameIndex = LabelIndex.index(li)
    Probs.to_csv(FOLDER + '/IntTestProbs' + str(FileNameIndex) + '.csv')
    Ext_Test_Probs.to_csv(FOLDER + '/ExtTestProbs' + str(FileNameIndex) + '.csv')
    
# Train classifier for all 2-label combinations
#Freqs = [0.27, 0.66, 0.51, 1.06, 2.50, 1.02, 1.15, 58.34, 0.42, 0.06, 29.55, 4.46]
#Repeat = [15, 15, 15, 15, 10, 15, 15, 3, 15, 15, 5, 8]
#for li1 in LabelIndex:
#    for li2 in LabelIndex:
#        if(li1 is not li2):
#            for i in range(0, int(np.true_divide(Freqs[LabelIndex.index(li2)], Freqs[LabelIndex.index(li1)])) + 1):
#                if(LabelIndex.index(li1) < 7 or 
#                (LabelIndex.index(li1) == 7 and LabelIndex.index(li2) < 4) or
#                (LabelIndex.index(li1) == 7 and LabelIndex.index(li2) == 4 and i + 1 < 2)):
#                    continue
#                name = li1 + 'with' + li2 + str(i + 1)
#                Probs = pd.DataFrame()
#                Ext_Test_Probs = pd.DataFrame()
#                print datetime.datetime.now().time()
#                print 'Training model',name
#                # enrich models in their data of interest
#                Enriched_X_train, Enriched_y_train = enrichPairwise(pd.read_csv(FOLDER + '\\X_train.csv'), pd.read_csv(FOLDER + '\\y_train.csv'), li1, li2)
#                Brain = trainModel(Enriched_X_train.drop('id', axis = 1), Enriched_y_train[li1], None, None)
#                print datetime.datetime.now().time()
#                print 'Getting probs for model no', name
#                Probs['id'] = pd.read_csv(FOLDER + '\\X_test.csv')['id']
#                Probs[name] = Brain.predict_proba(pd.read_csv(FOLDER + '\\X_test.csv').drop('id', axis = 1))[:, 1]
#                # External data
#                Ext_Test_Probs['id'] = pd.read_csv(FOLDER + '\\Test_Feats.csv')['id']
#                Ext_Test_Probs[name] = Brain.predict_proba(pd.read_csv(FOLDER + '\\Test_Feats.csv').drop('id', axis = 1).drop(Remove, axis = 1))[:, 1]
#                FileNameIndex = str(LabelIndex.index(li1)) + 'with' + str(LabelIndex.index(li2)) + 'no' + str(i + 1)
#                Probs.to_csv(FOLDER + '/IntTestProbs' + FileNameIndex + '.csv')
#                Ext_Test_Probs.to_csv(FOLDER + '/ExtTestProbs' + FileNameIndex + '.csv')
#    
print 'Probs written to CSV'
print datetime.datetime.now().time()
