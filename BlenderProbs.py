# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 13:10:44 2016

@author: piotrgrudzien
"""
def getCountryID(row):
    # Niezly hak!
    if(list(row).index(1) == 0):
        row[0] = 2
    return list(row).index(1) - 1
    
def getSingleScore(row):
    return row.tolist().index(row['label'])

def getReverseOrder(row):
    return np.argsort(row)
    
def scorePreds(Preds, labels):
    Preds = Preds.reset_index(drop = True)
    labels = labels.reset_index(drop = True)
    Preds = Preds.apply(getReverseOrder, axis = 1)
    Together = Preds[Preds.columns[::-1][:5]]
    Together['label'] = labels
    Together['score'] = Together.apply(getSingleScore, axis = 1)
    Result = 0
    DivideBy = 0
    for index, row in Together.iterrows():
        DivideBy += 1
        if(row['score'] < 5):
            Result += np.true_divide(1, np.log2(row['score'] + 2))
    return np.true_divide(Result, DivideBy)
    
def scoreTrain(Weights):
    Input = Sub_Train[0].multiply(Weights[0])
    for x in range(1, len(Folder)):
        Input = Input.add(Sub_Train[x].multiply(Weights[x]))
    res = scorePreds(Input, y_train.drop('IsTest', axis = 1))
    print Weights, ':', res
    global best
    global best_weights
    if(res > best):
        best = res
        best_weights = Weights
    return -res
    
def scoreTest(Weights):
    Input = Sub_Test[0].multiply(Weights[0])
    for x in range(1, len(Folder)):
        Input = Input.add(Sub_Test[x].multiply(Weights[x]))
    res = scorePreds(Input, y_test.drop('IsTest', axis = 1))
    print 'Test:', Weights, ':', res
    global bestCV
    global best_weightsCV
    if(res > best):
        bestCV = res
        best_weightsCV = Weights
    return -res
    
def score(Weights):
    Input = Submissions[0].multiply(Weights[0])
    for x in range(1, len(Folder)):
        Input = Input.add(Submissions[x].multiply(Weights[x]))
    res = scorePreds(Input, TrueLabels.drop('IsTest', axis = 1))
    print Weights, ':', res
    global bestCV
    global best_weightsCV
    if(res > best):
        bestCV = res
        best_weightsCV = Weights
    return -res
    
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from scipy.optimize import minimize
import warnings
import os.path
warnings.filterwarnings("ignore")

print 'Starting', datetime.datetime.now().time()

Name = []
Folder = []

# Row 67 Int: 8502
#Folder.append('FullStats2')
#Name.append('BasicInfo2264FullStats2264Trees100')

# Row 69 Int: 8505
#Folder.append('Times2')
#Name.append('BasicInfo2264Times2264Trees100')

# Row 73 Int: 8499
#Folder.append('Before2')
#Name.append('BasicInfo2Before2')

# Row 76 Int: 8502
#Folder.append('Times2')
#Name.append('BasicInfo2FullStats2Times2')

# Row 77 Int: 8506
#Folder.append('Before2')
#Name.append('BasicInfo2FullStats2Before2')

# Row 78 Int: 8506
#Folder.append('Before2')
#Name.append('BasicInfo2Times2Before2')

# Row 80 Int: 8503
#Folder.append('Before2')
#Name.append('BasicInfo2FullStats2Times2Before2')

# Row 85 Int: 8508
#Folder.append('ActionNgram2')
#Name.append('BasicInfo2Times2Before2ActionNgram2')

# Row 86 Int: 8506
#Folder.append('ActionNgramNorm2')
#Name.append('BasicInfo2Times2Before2ActionNgramNorm2')

# Row 87 Int: 8502 - NIE MA PLIKU EXT OD TEGO
#Folder.append('ActionNgram2')
#Name.append('BasicInfo2FullStats2Times2ActionNgram2')

# Row 94 Int: 8421
#Folder.append('ActionTypeNgram2')
#Name.append('ActionTypeNgram2')

# Row 95 Int: 8419
#Folder.append('ActionTypeNgramNorm2')
#Name.append('ActionTypeNgramNorm2')

# Row 96 Int: 8407
#Folder.append('ActionDetailNgram2')
#Name.append('ActionDetailNgram2')

# Row 98 Int: 8510
#Folder.append('BasicInfoFullStats')
#Name.append('BasicInfoFullStats')

#############################################################################

# Row 108 Int: 8512 Ext: 8794
Folder.append('BasicInfoFullStats3')
Name.append('BasicInfoFullStats3')

# Row 109 Int: 8513 Ext: 8790
Folder.append('BasicInfoTimes3')
Name.append('BasicInfoTimes3')

# Row 110 Int: 8518 Ext: 8798
Folder.append('BasicInfoTimes3')
Name.append('BasicInfoFullStats3BasicInfoTimes3')

# Row 111 Int: 8523 Ext: 8791
Folder.append('BasicInfoBefore3')
Name.append('BasicInfoBefore3')

# Row 112 Int: 8521 Ext: 8793
Folder.append('BasicInfoBefore3')
Name.append('BasicInfoFullStats3BasicInfoBefore3')

# Row 113 Int: 8522 Ext: 8788
#Folder.append('BasicInfoBefore3')
#Name.append('BasicInfoTimes3BasicInfoBefore3')

# Row 114 Int: 8524 Ext: 8795
Folder.append('BasicInfoBefore3')
Name.append('BasicInfoFullStats3BasicInfoTimes3BasicInfoBefore3')

# Row 122 Int: 8517 Ext: 8794
Folder.append('BasicInfoFullStats3')
Name.append('ss0.5BasicInfoFullStats3')

# Row 123 Int: 8518 Ext: 8784
#Folder.append('BasicInfoTimes3')
#Name.append('ss0.5BasicInfoTimes3')

# Row 124 Int: 8518 Ext: 
#Folder.append('BasicInfoBefore3')
#Name.append('ss0.5BasicInfoBefore3')

# Row 125 Int: 8513 Ext: 8795
Folder.append('BasicInfoTimes3')
Name.append('ss0.5BasicInfoFullStats3BasicInfoTimes3')

# Row 126 Int: 8513 Ext:
#Folder.append('BasicInfoBefore3')
#Name.append('ss0.5BasicInfoFullStats3BasicInfoBefore3')

# Row 127 Int: 8515 Ext:
#Folder.append('BasicInfoBefore3')
#Name.append('ss0.5BasicInfoTimes3BasicInfoBefore3')

# Row 131 Int: 8429 Ext:
#Folder.append('FullStatsActionNgram3')
#Name.append('ss1FullStatsActionNgram3')

# Row 132 Int: 8520 Ext:
#Folder.append('FullStatsActionNgram3')
#Name.append('ss1BasicInfoTimes3FullStatsActionNgram3')

# Row 133 Int: 8524 Ext: 8787
#Folder.append('FullStatsActionNgram3')
#Name.append('ss1BasicInfoBefore3FullStatsActionNgram3')

# Row 134 Int: 8519 Ext:
#Folder.append('FullStatsActionNgram3')
#Name.append('ss1BasicInfoActionTypeNgram3BasicInfoTimes3FullStatsActionNgram3')

# Row 135 Int: 8475 Ext:
#Folder.append('BasicInfoActionTypeNgram3')
#Name.append('ss1BasicInfoActionTypeNgram3')

# Row 136 Int: 8482 Ext:
#Folder.append('BasicInfoActionTypeNgram3')
#Name.append('ss1FullStatsActionNgram3BasicInfoActionTypeNgram3')

# Row 137 Int: 8514 Ext:
#Folder.append('BasicInfoActionTypeNgram3')
#Name.append('ss1BasicInfoFullStats3BasicInfoActionTypeNgram3')

# Row 138 Int: 8513 Ext:
#Folder.append('BasicInfoActionTypeNgram3')
#Name.append('ss1BasicInfoFullStats3FullStatsActionDetailNgram3BasicInfoActionTypeNgram3')

# Row 139 Int: 8425 Ext:
#Folder.append('FullStatsActionDetailNgram3')
#Name.append('ss1FullStatsActionDetailNgram3')

# Row 140 Int: 8518 Ext:
#Folder.append('FullStatsActionDetailNgram3')
#Name.append('ss1BasicInfoTimes3FullStatsActionDetailNgram3')

# Row 141 Int: 8519 Ext:
#Folder.append('FullStatsActionDetailNgram3')
#Name.append('ss1BasicInfoBefore3FullStatsActionDetailNgram3')

# Row 142 Int: 8511 Ext:
#Folder.append('FullStatsActionDetailNgram3')
#Name.append('ss1BasicInfoActionTypeNgram3BasicInfoTimes3FullStatsActionDetailNgram3')

# Row 149 Int: 8528 Ext: 8798
Folder.append('FullStatsActionNgram3')
Name.append('ss1BasicInfoFullStats3BasicInfoBefore3FullStatsActionNgram3')

# Row 150 Int: 8512 Ext:
#Folder.append('BasicInfoActionTypeNgram3')
#Name.append('ss1FullStats3BasicInfoTimes3BasicInfoActionTypeNgram3')

# Row 151 Int: 8521 Ext:
#Folder.append('FullStatsActionDetailNgram3')
#Name.append('ss1BasicInfoTimes3BasicInfoBefore3FullStatsActionDetailNgram3')

# Row 153 Int: 8487 Ext:
#Folder.append('FullStatsActionDetailNgram3')
#Name.append('ss1FullStatsActionNgram3BasicInfoActionTypeNgram3FullStatsActionDetailNgram3')

# Row 154 Int: 8524 Ext:
Folder.append('BasicInfoBefore3')
Name.append('ss1BasicInfoFullStats3BasicInfoTimes3BasicInfoBefore3')

# Row 155 Int: 8512 Ext:
#Folder.append('BasicInfoTimes3')
#Name.append('ss1FullStatsActionNgram3BasicInfoFullStats3BasicInfoTimes3')

# Row 156 Int: 8521 Ext:
#Folder.append('BasicInfoBefore3')
#Name.append('ss1BasicInfoActionTypeNgram3BasicInfoFullStats3BasicInfoBefore3')

# Row 157 Int: 8517 Ext:
#Folder.append('BasicInfoTimes3')
#Name.append('ss1FullStatsActionDetailNgram3BasicInfoFullStats3BasicInfoTimes3')

# Row 158 Int: 8524 Ext: 8795
Folder.append('BasicInfoBefore3')
Name.append('ss1FullStatsActionDetailNgram3BasicInfoFullStats3BasicInfoBefore3')

# Row 180 Int: 8515 Ext:
#Folder.append('BasicInfoTimes3')
#Name.append('ss1FullStatsActionNgramNorm3BasicInfoFullStats3BasicInfoTimes3')

# Row 181 Int: 8520 Ext:
#Folder.append('BasicInfoBefore3')
#Name.append('ss1FullStatsActionNgramNorm3BasicInfoFullStats3BasicInfoBefore3')

# Row 182 Int: 8506 Ext:
#Folder.append('BasicInfoActionTypeNgram3')
#Name.append('ss1FullStatsActionNgramNorm3BasicInfoBefore3BasicInfoActionTypeNgram3')

# Row 183 Int: 8518 Ext:
#Folder.append('FullStatsActionDetailNgram3')
#Name.append('ss1FullStatsActionNgramNorm3BasicInfoTimes3FullStatsActionDetailNgram3')

# Row 184 Int: 8523 Ext:
Folder.append('BasicInfoBefore3')
Name.append('ss1FullStatsActionNgramNorm3BasicInfoTimes3BasicInfoBefore3')

# Row 187 Int: 8527 Ext: 8797
Folder.append('FullStatsActionNgram3')
Name.append('ss1BasicInfoFullStats3BasicInfoTimes3BasicInfoBefore3FullStatsActionNgram3')

# Row 188 Int: 8525 Ext: 8791
Folder.append('BasicInfoActionTypeNgram3')
Name.append('ss1BasicInfoFullStats3BasicInfoTimes3BasicInfoBefore3BasicInfoActionTypeNgram3')

# Row 189 Int: 8526 Ext: 8799
Folder.append('FullStatsActionDetailNgram3')
Name.append('ss1BasicInfoFullStats3BasicInfoTimes3BasicInfoBefore3FullStatsActionDetailNgram3')

# Row 190 Int: 8513 Ext:
#Folder.append('FullStatsActionNgramNorm3')
#Name.append('ss1BasicInfoFullStats3BasicInfoTimes3BasicInfoBefore3FullStatsActionNgramNorm3') 

# Row 191 Int: 8524 Ext:8789
#Folder.append('BasicInfoActionDetailNgramNorm3')
#Name.append('ss1BasicInfoFullStats3BasicInfoTimes3BasicInfoBefore3BasicInfoActionDetailNgramNorm3') 

# Row 204 Int: 8509 Ext: 8790
Folder.append('BasicInfoFullStats4')
Name.append('ss1Trees500BasicInfoFullStats4')

# Row 208 Int: 8599 Ext: 8781
Folder.append('BasicInfoTimes4')
Name.append('ss1Trees500BasicInfoTimes4duo')

# Row 211 Int:  Ext: 
Folder.append('BasicInfo4')
Name.append('ss1Trees100BasicInfo4')

# Row 212 Int:  Ext: 
Folder.append('BasicInfoFullStats4')
Name.append('ss1Trees500BasicInfoFullStats4')

# Row 215 Int: 8518  Ext: 
Folder.append('BasicInfoTimes4')
Name.append('ss1Trees500BasicInfoTimes4')

# Row 216 Int: 8549 Ext: 
Folder.append('BasicInfoBefore4')
Name.append('ss1Trees500BasicInfoBefore4')

# Row 217 Int: 8516 Ext: 
#Folder.append('BasicInfoActionNgram4')
#Name.append('ss1Trees500BasicInfoActionNgram4')

# Row 218 Int: 8448 Ext: 
#Folder.append('BasicInfoActionNgramNorm4')
#Name.append('ss1Trees500BasicInfoActionNgramNorm4')

# Row 219 Int: 8500 Ext: 
#Folder.append('BasicInfoActionTypeNgram4')
#Name.append('ss1Trees500BasicInfoActionTypeNgram4')

# Row 225 Int: 8506 Ext: 
Folder.append('BasicInfoFullStats40')
Name.append('ss1Trees500BasicInfoFullStats40')

# Row 231 Int: 8545 Ext: 
Folder.append('BasicInfoFullStats41')
Name.append('ss1Trees500BasicInfoFullStats41')

# Row 232 Int: 8585 Ext: 
Folder.append('BasicInfoTimes41')
Name.append('ss1Trees500BasicInfoTimes41')

# Row 233 Int: 8568 Ext: 
Folder.append('BasicInfoBefore41')
Name.append('ss1Trees500BasicInfoBefore41')

# Row 234 Int: 8515 Ext: 
Folder.append('BasicInfoFullStats42')
Name.append('ss1Trees500BasicInfoFullStats42')
########################################################

#Folder.append('RawResults')
#Name.append('BasicInfo')
#
#Folder.append('RawResults')
#Name.append('FullStats')
#
#Folder.append('RawResults')
#Name.append('Times')
#
#Folder.append('RawResults')
#Name.append('Before')
#
#Folder.append('RawResults')
#Name.append('ActionNgram')
#
#Folder.append('RawResults')
#Name.append('ActionNgramNorm')
#
#Folder.append('RawResults')
#Name.append('ActionTypeNgram')
#
#Folder.append('RawResults')
#Name.append('ActionTypeNgramNorm')
#
#Folder.append('RawResults')
#Name.append('ActionDetailNgram')
#
#Folder.append('RawResults')
#Name.append('ActionDetailNgramNorm')
#
#Folder.append('RawResults')
#Name.append('BasicInfoFullStats')

#########################################################

Subfile = ['/' + Name[x] + 'IntProbs.csv' for x in range(0, len(Name))]

########## FIND OPTIMAL WEIGHTS
Weights = [1 for x in range(0, len(Name))]
Submissions = [pd.read_csv(Folder[x] + Subfile[x]) for x in range(0, len(Folder))]
Sub_Train = [None for x in range(0, len(Name))]
Sub_Test = [None for x in range(0, len(Name))]

Labels = ['id', 'TargetIsAU', 'TargetIsCA', 'TargetIsDE', 'TargetIsES', 'TargetIsFR', 'TargetIsGB', 'TargetIsIT', 'TargetIsNDF', 'TargetIsNL', 'TargetIsPT', 'TargetIsUS', 'TargetIsother']
IntLabels = pd.read_csv(Folder[0] + '/IntTestLabels.csv')[Labels + ['IsTest']]
IntLabels['country'] = IntLabels.apply(getCountryID, axis = 1)
TrueLabels = IntLabels.loc[IntLabels['IsTest'] == 1, ['country', 'IsTest']]
TrueLabels = TrueLabels.reset_index(drop = True)

#for f in range(0, len(Submissions)):
#    Sub_Train[f], Sub_Test[f], y_train, y_test = train_test_split(Submissions[f], TrueLabels, test_size=0.2, random_state = 22)
#    Sub_Train[f] = Sub_Train[f].append(Sub_Test[f].loc[y_test[y_test['country'] == 7].index.tolist()])
#    y_train = y_train.append(y_test.loc[y_test[y_test['country'] == 7].index.tolist()])

best = 0
best_weights = []
bestCV = 0
best_weightsCV = []



#res = minimize(score, Weights, method = 'Nelder-Mead', options = {'eps' : 2, 'maxiter' : 30000, 'maxfun' : 30000, 'bounds' : bnds})
#
#print 'Weights:', best_weights
#print 'Internal test result:', best

score(Weights)

#w = [1 for x in range(0, len(Weights))]
#score(w)
#for i in range(0, len(w)):
#    w = [1 for x in range(0, len(Weights))]
#    w[i] = 0
#    print 'i =', i
#    score(w)
    
#score([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#score([1 for x in range(0, len(Weights))])
            

######## FIND OUTPUT FOR EXTERNAL DATA

Weights = [1 for x in range(0, len(Name))]


FILE_NAME = 'Row236.csv'
if(os.path.isfile(FILE_NAME)):
    raise Exception('Set row correctly! ' + FILE_NAME + ' already exists!')
    
External = [pd.read_csv(Folder[x] + Subfile[x].replace('Int', 'Ext')) for x in range(0, len(Folder))]

Output = External[0].multiply(Weights[0])
for x in range(1, len(Folder)):
    Output = Output.add(External[x].multiply(Weights[x]))

ids = pd.read_csv(Folder[0] + '/' + Name[0] + '.csv')

le = LabelEncoder()
cts = []
#Twisted = [0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9]
Twisted_names = ['AU', 'CA', 'US', 'other', 'DE', 'ES', 'FR', 'GB', 'IT', 'NDF', 'NL', 'PT']
#Names = ['AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NDF', 'NL', 'PT', 'US', 'other']
le.fit(Twisted_names)
for i in range(len(Output)):
    df = pd.DataFrame()
    df = df.append(Output.loc[i, :])
    Trans1 = df.apply(getReverseOrder, axis = 1)
    Trans2 = Trans1[Trans1.columns[::-1][:5]]
    cts += [Twisted_names[int(x)] for x in Trans2.loc[i]]

ExtResult = pd.DataFrame(np.column_stack((ids['id'], cts)), columns=['id', 'country'])



ExtResult.to_csv(FILE_NAME, index = False)
print('External result written to ' + FILE_NAME)
print datetime.datetime.now().time()