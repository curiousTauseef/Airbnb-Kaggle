# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 21:05:35 2015

@author: Piotrek
"""
from dateutil.parser import parse
import pandas as pd
import pickle

def LoadAndMerge(Name, Type, FS):
    print 'Feature matrix shape:', FS.shape
    Data = pickle.load(open( Name + Type + ".p", "rb"))
    print Name + ' shape:', Data.shape
    return pd.merge(FS, Data, left_on = 'id', right_on = 'id')
    
def LoadAndMergeCSV(Name, FS):
    print 'Feature matrix shape:', FS.shape
    Data = pd.read_csv(Name + '.csv')
    print Name + ' shape:', Data.shape
    return pd.merge(FS, Data, how = 'left', left_on = 'id', right_on = 'user_id').fillna(0)
    
    
def getFeats(Data, Type, FOLDER):
    FS = pd.DataFrame()
    FS['id'] = Data['id']
    print 'Id shape:', FS.shape
# Need to include everywhere to do train/test split
    FS = LoadAndMerge('FirstActive', Type, FS)
    if('BasicInfo' in FOLDER):
        FS = LoadAndMerge('AccountCreated', Type, FS)
        
#        FS = LoadAndMerge('FirstBooking', Type, FS)
        
        FS = LoadAndMerge('Gender', Type, FS)
        FS = LoadAndMerge('Age', Type, FS)
        FS = LoadAndMerge('SignupMethod', Type, FS)
        FS = LoadAndMerge('SignupFlow', Type, FS)
        FS = LoadAndMerge('Language', Type, FS)
        FS = LoadAndMerge('AffChannel', Type, FS)
        FS = LoadAndMerge('AffProvider', Type, FS)
        FS = LoadAndMerge('FirstAffTracked', Type, FS)
        FS = LoadAndMerge('SignupApp', Type, FS)
        FS = LoadAndMerge('FirstDeviceType', Type, FS)
        FS = LoadAndMerge('FirstBrowser', Type, FS)
########### SESSION BASED
    # BASED ON action
    GN = 'Action'
    if('FullStats' in FOLDER):
        FS = LoadAndMergeCSV(GN + 'Counts', FS)
        FS = LoadAndMergeCSV(GN + 'Norm', FS)
    if('Times' in FOLDER):
        FS = LoadAndMergeCSV(GN + 'TimesKurtosis', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMax', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMean', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMedian', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMin', FS)
        FS = LoadAndMergeCSV(GN + 'TimesSkewness', FS)
        FS = LoadAndMergeCSV(GN + 'TimesStd', FS)
    if('Before' in FOLDER):
        FS = LoadAndMergeCSV(GN + 'BeforeKurtosis', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMax', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMean', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMedian', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMin', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeSkewness', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeStd', FS)
    if('ActionNgram' in FOLDER):
        FS = LoadAndMergeCSV(GN + '2gram1', FS)
        FS = LoadAndMergeCSV(GN + '2gram2', FS)
        FS = LoadAndMergeCSV(GN + '2gram3', FS)
        FS = LoadAndMergeCSV(GN + '3gram1', FS)
        FS = LoadAndMergeCSV(GN + '3gram2', FS)
        FS = LoadAndMergeCSV(GN + '3gram3', FS)
        FS = LoadAndMergeCSV(GN + '4gram1', FS)
        FS = LoadAndMergeCSV(GN + '4gram2', FS)
        FS = LoadAndMergeCSV(GN + '4gram3', FS)
        FS = LoadAndMergeCSV(GN + '5gram1', FS)
        FS = LoadAndMergeCSV(GN + '5gram2', FS)
        FS = LoadAndMergeCSV(GN + '5gram3', FS)
    if('ActionNgramNorm' in FOLDER):
        FS = LoadAndMergeCSV(GN + '2gram1Norm', FS)
        FS = LoadAndMergeCSV(GN + '2gram2Norm', FS)
        FS = LoadAndMergeCSV(GN + '2gram3Norm', FS)
        FS = LoadAndMergeCSV(GN + '3gram1Norm', FS)
        FS = LoadAndMergeCSV(GN + '3gram2Norm', FS)
        FS = LoadAndMergeCSV(GN + '3gram3Norm', FS)
        FS = LoadAndMergeCSV(GN + '4gram1Norm', FS)
        FS = LoadAndMergeCSV(GN + '4gram2Norm', FS)
        FS = LoadAndMergeCSV(GN + '4gram3Norm', FS)
        FS = LoadAndMergeCSV(GN + '5gram1Norm', FS)
        FS = LoadAndMergeCSV(GN + '5gram2Norm', FS)
        FS = LoadAndMergeCSV(GN + '5gram3Norm', FS)
    # BASED ON action_type
    GN = 'ActionType'
    if('FullStats' in FOLDER):
        FS = LoadAndMergeCSV(GN + 'Counts', FS)
        FS = LoadAndMergeCSV(GN + 'Norm', FS)
    if('Times' in FOLDER):
        FS = LoadAndMergeCSV(GN + 'TimesKurtosis', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMax', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMean', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMedian', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMin', FS)
        FS = LoadAndMergeCSV(GN + 'TimesSkewness', FS)
        FS = LoadAndMergeCSV(GN + 'TimesStd', FS)
    if('Before' in FOLDER):
        FS = LoadAndMergeCSV(GN + 'BeforeKurtosis', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMax', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMean', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMedian', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMin', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeSkewness', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeStd', FS)
    if('ActionTypeNgram' in FOLDER):
        FS = LoadAndMergeCSV(GN + '2gram1', FS)
        FS = LoadAndMergeCSV(GN + '3gram1', FS)
        FS = LoadAndMergeCSV(GN + '3gram2', FS)
        FS = LoadAndMergeCSV(GN + '4gram1', FS)
        FS = LoadAndMergeCSV(GN + '4gram2', FS)
        FS = LoadAndMergeCSV(GN + '5gram1', FS)
        FS = LoadAndMergeCSV(GN + '5gram2', FS)
    if('ActionTypeNgramNorm' in FOLDER):
        FS = LoadAndMergeCSV(GN + '2gram1Norm', FS)
        FS = LoadAndMergeCSV(GN + '3gram1Norm', FS)
        FS = LoadAndMergeCSV(GN + '3gram2Norm', FS)
        FS = LoadAndMergeCSV(GN + '4gram1Norm', FS)
        FS = LoadAndMergeCSV(GN + '4gram2Norm', FS)
        FS = LoadAndMergeCSV(GN + '5gram1Norm', FS)
        FS = LoadAndMergeCSV(GN + '5gram2Norm', FS)
    # BASED ON action_detail
    GN = 'ActionDetail'
    if('FullStats' in FOLDER):
        FS = LoadAndMergeCSV(GN + 'Counts', FS)
        FS = LoadAndMergeCSV(GN + 'Norm', FS)
    if('Times' in FOLDER):
        FS = LoadAndMergeCSV(GN + 'TimesKurtosis', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMax', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMean', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMedian', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMin', FS)
        FS = LoadAndMergeCSV(GN + 'TimesSkewness', FS)
        FS = LoadAndMergeCSV(GN + 'TimesStd', FS)
    if('Before' in FOLDER):
        FS = LoadAndMergeCSV(GN + 'BeforeKurtosis', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMax', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMean', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMedian', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMin', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeSkewness', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeStd', FS)
    if('ActionDetailNgram' in FOLDER):
        FS = LoadAndMergeCSV(GN + '2gram1', FS)
        FS = LoadAndMergeCSV(GN + '2gram2', FS)
        FS = LoadAndMergeCSV(GN + '2gram3', FS)
        FS = LoadAndMergeCSV(GN + '3gram1', FS)
        FS = LoadAndMergeCSV(GN + '3gram2', FS)
        FS = LoadAndMergeCSV(GN + '3gram3', FS)
        FS = LoadAndMergeCSV(GN + '4gram1', FS)
        FS = LoadAndMergeCSV(GN + '4gram2', FS)
        FS = LoadAndMergeCSV(GN + '4gram3', FS)
        FS = LoadAndMergeCSV(GN + '5gram1', FS)
        FS = LoadAndMergeCSV(GN + '5gram2', FS)
    if('ActionDetailNgramNorm' in FOLDER):
        FS = LoadAndMergeCSV(GN + '2gram1Norm', FS)
        FS = LoadAndMergeCSV(GN + '2gram2Norm', FS)
        FS = LoadAndMergeCSV(GN + '2gram3Norm', FS)
        FS = LoadAndMergeCSV(GN + '3gram1Norm', FS)
        FS = LoadAndMergeCSV(GN + '3gram2Norm', FS)
        FS = LoadAndMergeCSV(GN + '3gram3Norm', FS)
        FS = LoadAndMergeCSV(GN + '4gram1Norm', FS)
        FS = LoadAndMergeCSV(GN + '4gram2Norm', FS)
        FS = LoadAndMergeCSV(GN + '4gram3Norm', FS)
        FS = LoadAndMergeCSV(GN + '5gram1Norm', FS)
        FS = LoadAndMergeCSV(GN + '5gram2Norm', FS)
    # BASED ON device_type
    GN = 'DeviceType'
    if('FullStats' in FOLDER):
        FS = LoadAndMergeCSV(GN + 'Counts', FS)
        FS = LoadAndMergeCSV(GN + 'Norm', FS)
    if('Times' in FOLDER):
        FS = LoadAndMergeCSV(GN + 'TimesKurtosis', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMax', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMean', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMedian', FS)
        FS = LoadAndMergeCSV(GN + 'TimesMin', FS)
        FS = LoadAndMergeCSV(GN + 'TimesSkewness', FS)
        FS = LoadAndMergeCSV(GN + 'TimesStd', FS)
    if('Before' in FOLDER):
        FS = LoadAndMergeCSV(GN + 'BeforeKurtosis', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMax', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMean', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMedian', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeMin', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeSkewness', FS)
        FS = LoadAndMergeCSV(GN + 'BeforeStd', FS)
    # LABELS target
    if(Type is 'Train'):
        FS = LoadAndMerge('Target', Type, FS)
    print 'Final ' + Type + ' feature matrix shape:', FS.shape
    return FS