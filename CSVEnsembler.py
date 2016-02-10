# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 09:09:08 2015

@author: piotrgrudzien
"""
def getReverseOrder(row):
    return np.argsort(row) 
    
    
def getScores(scores):
    global Output, points
#    Output = pd.DataFrame(0, index = np.arange(len(Input[0])), columns = OutputColumns)
#    Output['id'] = Input[0]['id']
#    Output.loc[Output['id'] == scores['id'],scores['country']] += points
    Output.loc[scores.name / 5, scores['country']] += weight * points
    if(points == 1):
        points = 5
    else:
        points -= 1
    if(scores.name % 1000 == 0):
        print 'Row', scores.name, datetime.datetime.now().time()

def printOverlap(f1, f2):
    print 'Overlap between', f1, 'and', f2, ':', np.true_divide(sum(Input[f1]['country'] == Input[f2]['country']), len(Input[f1]))

import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime

print 'Starting', datetime.datetime.now().time()

Countries = ['AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NDF', 'NL', 'PT', 'US', 'other']

FullStats = 'EnrichedFullStatsXG100treesReps/'
SubFile1 = 'ExtResults100Trees3040Feats.csv'
SubFile3 = 'ExtResults100Trees90Feats.csv'
SubFile4 = 'ExtResults100Trees76Feats.csv'
SubFile5 = 'ExtResults100Trees305Feats.csv'

NoStatsBefore = 'EnrichedNoStatsBeforeXG100treesReps/'

NoStatsTimes = 'EnrichedNoStatsTimesXG100treesReps/'
SubFile2 = 'BlendingExtResults100Trees1790Feats.csv'
SubFile6 = 'ExtResults100Trees375Feats.csv'
SubFile7 = 'BlendingExtResults100Trees587Feats.csv'

Input = []
Input.append(pd.read_csv(FullStats + SubFile1))
Input.append(pd.read_csv(NoStatsTimes + SubFile2))
Input.append(pd.read_csv(FullStats + SubFile3))
#Input.append(pd.read_csv(FullStats + SubFile4))
#Input.append(pd.read_csv(FullStats + SubFile5))
#Input.append(pd.read_csv(NoStatsTimes + SubFile6))
#Input.append(pd.read_csv(NoStatsTimes + SubFile7))

printOverlap(0, 1) 
printOverlap(0, 2)
printOverlap(1, 2)
   
OutputColumns = Countries
OutputColumns.append('id')
Output = pd.DataFrame(0, index = np.arange(len(Input[0]) / 5), columns = OutputColumns)
Output['id'] = Input[0]['id'].drop_duplicates().reset_index(drop = True)

print 'Starting mapping', datetime.datetime.now().time()
weights = [2, 1, 1]
weight_index = 0
for subfile in Input:
    weight = weights[weight_index]
    weight_index += 1
    points = 5
    subfile.apply(getScores, axis = 1)
    
#    for index, row in subfile.iterrows():
#        Output.loc[Output['id'] == row['id'],row['country']] += points
#        if(points == 1):
#            points = 5
#        else:
#            points -= 1
#        if(index % 1000 == 0):
#            print 'Row', index, datetime.datetime.now().time()
    print 'File done', datetime.datetime.now().time()
    
print 'Output ready', datetime.datetime.now().time()

Output.columns = range(0, 12) + ['id']
    
le = LabelEncoder()
ids = []
cts = []
le.fit(Countries)
for i in range(len(Output)):
    df = pd.DataFrame(columns = range(0, 12))
    df.loc[i, :] = Output.loc[i, range(0, 12)]
#    print df
#    print 'df', df
    Trans1 = df.apply(getReverseOrder, axis = 1)
#    print 'Trans1', Trans1
    Trans2 = Trans1[Trans1.columns[::-1][:5]]
#    print 'Trans2', Trans2
    cts += le.inverse_transform(Trans2).tolist()[0]
#    print 'Transformed:', le.inverse_transform(Trans2).tolist()[0]

ExtResult = pd.DataFrame(np.column_stack((Input[0]['id'], cts)), columns=['id', 'country'])

ExtResult.to_csv('BlendedCSVRow47.csv', index = False)
print 'External result written to CSV!'
print datetime.datetime.now().time()