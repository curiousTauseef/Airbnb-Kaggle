# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 21:22:23 2015

@author: Piotrek
"""

import pickle
import pandas as pd
import math
import numpy as np
from scipy import stats

def bigramToCSV(Data, GN, GI):
    Data[GN] = Data[GI] + 'X' + Data.groupby('user_id')[GI].shift(-1)
    whichone = 3
    Rank = pd.read_csv(GN + 'Rank.csv', skiprows = 100 * (whichone - 1), nrows = 100, header=None)
    Rank.columns = ['Item', 'Freq']
    print 'Data initially:', len(Data)
    Data = Data.merge(Rank, how='inner', left_on = GN, right_on = 'Item', sort = False)
    print 'Data afterwards:', len(Data)
    A = Data.groupby('user_id')[GN].value_counts(normalize = True).unstack().fillna(0)
    A.columns = [GN + 'IS' + s for s in A.columns]
#    A = getStats(A, GN + str(whichone))
    A.to_csv(GN + str(whichone) + 'Norm.csv')
    
def trigramToCSV(Data, GN, GI):
    Data[GN] = Data[GI] + 'X' + Data.groupby('user_id')[GI].shift(-1) + 'X' + Data.groupby('user_id')[GI].shift(-2)
    whichone = 3
    Rank = pd.read_csv(GN + 'Rank.csv', skiprows = 100 * (whichone - 1), nrows = 100, header=None)
    Rank.columns = ['Item', 'Freq']
    print 'Data initially:', len(Data)
    Data = Data.merge(Rank, how='inner', left_on = GN, right_on = 'Item', sort = False)
    print 'Data afterwards:', len(Data)
    A = Data.groupby('user_id')[GN].value_counts(normalize = True).unstack().fillna(0)
    A.columns = [GN + 'IS' + s for s in A.columns]
#    A = getStats(A, GN + str(whichone))
    A.to_csv(GN + str(whichone) + 'Norm.csv')
    
def fourgramToCSV(Data, GN, GI):
    Data[GN] = Data[GI] + 'X' + Data.groupby('user_id')[GI].shift(-1) + 'X' + Data.groupby('user_id')[GI].shift(-2) + 'X' + Data.groupby('user_id')[GI].shift(-3)
    whichone = 3
    sizes = 100
    Rank = pd.read_csv(GN + 'Rank.csv', skiprows = sizes * (whichone - 1), nrows = sizes, header=None)
    Rank.columns = ['Item', 'Freq']
    print 'Data initially:', len(Data)
    Data = Data.merge(Rank, how='inner', left_on = GN, right_on = 'Item', sort = False)
    print 'Data afterwards:', len(Data)
    A = Data.groupby('user_id')[GN].value_counts(normalize = True).unstack().fillna(0)
    A.columns = [GN + 'IS' + s for s in A.columns]
#    A = getStats(A, GN + str(whichone))
    A.to_csv(GN + str(whichone) + 'Norm.csv')
    
def fivegramToCSV(Data, GN, GI):
    Data[GN] = Data[GI] + 'X' + Data.groupby('user_id')[GI].shift(-1) + 'X' + Data.groupby('user_id')[GI].shift(-2) + 'X' + Data.groupby('user_id')[GI].shift(-3) + 'X' + Data.groupby('user_id')[GI].shift(-4)
    whichone = 2
    Rank = pd.read_csv(GN + 'Rank.csv', skiprows = 100 * (whichone - 1), nrows = 100, header=None)
    Rank.columns = ['Item', 'Freq']
    print 'Data initially:', len(Data)
    Data = Data.merge(Rank, how='inner', left_on = GN, right_on = 'Item', sort = False)
    print 'Data afterwards:', len(Data)
    A = Data.groupby('user_id')[GN].value_counts().unstack().fillna(0)
    A.columns = [GN + 'IS' + s for s in A.columns]
#    A = getStats(A, GN + str(whichone))
    A.to_csv(GN + str(whichone) + '.csv')

def timesToCSV(Data, GN, GI, GT, sh):
    A = pd.DataFrame()
    Data[GT] = Data.groupby('user_id')[GT].shift(sh)
    
    A = Data.groupby(['user_id', GI])[GT].mean().unstack().fillna(-1)
    A.columns = [GN + s + 'mean' for s in A.columns]
    A.to_csv(GN + 'Mean' + '.csv')
    
    A = Data.groupby(['user_id', GI])[GT].std().unstack().fillna(-1)
    A.columns = [GN + s + 'std' for s in A.columns]
    A.to_csv(GN + 'Std' + '.csv')
    
    A = Data.groupby(['user_id', GI])[GT].max().unstack().fillna(-1)
    A.columns = [GN + s + 'max' for s in A.columns]
    A.to_csv(GN + 'Max' + '.csv')
    
    A = Data.groupby(['user_id', GI])[GT].min().unstack().fillna(-1)
    A.columns = [GN + s + 'min' for s in A.columns]
    A.to_csv(GN + 'Min' + '.csv')
    
    A = Data.groupby(['user_id', GI])[GT].median().unstack().fillna(-1)
    A.columns = [GN + s + 'median' for s in A.columns]
    A.to_csv(GN + 'Median' + '.csv')
    
    A = Data.groupby(['user_id', GI])[GT].skew().unstack().fillna(-1)
    A.columns = [GN + s + 'skewness' for s in A.columns]
    A.to_csv(GN + 'Skewness' + '.csv')
    
    A = Data.groupby(['user_id', GI])[GT].apply(stats.kurtosis).unstack().fillna(-1)
    A.columns = [GN + s + 'kurtosis' for s in A.columns]
    A.to_csv(GN + 'Kurtosis' + '.csv')

def getStats(A, GN):
    
    A[GN + 'Max'] = A.apply(np.max, axis = 1)   
    A[GN + 'Min'] = A.apply(np.min, axis = 1) 
    A[GN + 'Std'] = A.apply(np.std, axis = 1)
    A[GN + 'CountNonZero'] = A.apply(np.count_nonzero, axis = 1) 
    A[GN + 'Mean'] = A.apply(np.mean, axis = 1)  
    A[GN + 'Median'] = A.apply(np.median, axis = 1)  
    A[GN + 'Entropy'] = A.apply(stats.entropy, axis = 1)  
    A[GN + 'Kurtosis'] = A.apply(stats.kurtosis, axis = 1)  
    A[GN + 'Skewness'] = A.apply(stats.skew, axis = 1)  
    return A
    
def addNumeric(A, Col, GroupName):
    A[GroupName] = Col
    return A

def catToBinary(A, Col, GroupName):
    U = np.unique(Col)
    for val in U:
        if(str(val) != 'nan'):
            index = GroupName + 'Is' + str(val)
            A[index] = (Col == val).astype(int)
    index = GroupName + 'IsNull'
    A[index] = (Col.isnull()).astype(int)
    return A
    
def addCountColumns(row):
    U = np.unique(Col)
    for val in U:
        if(str(val) != 'nan'):
            index = GroupName + 'Is' + str(val)
            A[index] = (Col == val).astype(int)
    index = GroupName + 'IsNull'
    A[index] = (Col.isnull()).astype(int)
    return A

def pickleStuff(Data, Type):
### Pickled
    AccountCreated = False
    FirstActive = False
    FirstBooking = False
    Gender = False
    Age = False
    SignupMethod = False
    SignupFlow = False
    Language = False
    AffChannel = False
    AffProvider = False
    FirstAffTracked = False
    SignupApp = False
    FirstDeviceType = False
    FirstBrowser = False

#####BASED ON date_account_created
    if(AccountCreated):
        GN = 'AccountCreated' # group name
        GI = 'date_account_created' # group index
        Data[GI] = pd.to_datetime(Data[GI])
        A = pd.DataFrame()
        A['id'] = Data['id']
        A = addNumeric(A, Data[GI].dt.year, GN + 'Year')
        A = addNumeric(A, Data[GI].dt.month, GN + 'Month')
        A = addNumeric(A, Data[GI].dt.day, GN + 'DayOfMonth')
        A = addNumeric(A, Data[GI].dt.weekofyear, GN + 'WeekOfYear')
        A = addNumeric(A, Data[GI].dt.dayofweek, GN + 'DayOfWeek')
        A = addNumeric(A, Data[GI].dt.quarter, GN + 'Quarter')
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
#### BASED ON timestamp_first_active
    if(FirstActive):
        GN = 'FirstActive' # group name
        GI = 'timestamp_first_active' # group index
        Data[GI] = pd.to_datetime(Data[GI], format = '%Y%m%d%H%M%S')
        A = pd.DataFrame()
        A['id'] = Data['id']
        A = addNumeric(A, Data[GI].dt.year, GN + 'Year')
        A = addNumeric(A, Data[GI].dt.month, GN + 'Month')
        A = addNumeric(A, Data[GI].dt.day, GN + 'DayOfMonth')
        A = addNumeric(A, Data[GI].dt.weekofyear, GN + 'WeekOfYear')
        A = addNumeric(A, Data[GI].dt.dayofweek, GN + 'DayOfWeek')
        A = addNumeric(A, Data[GI].dt.quarter, GN + 'Quarter')
        A = addNumeric(A, Data[GI].dt.hour, GN + 'Hour')
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
##### BASED ON date_first_booking
    if(FirstBooking):
        GN = 'FirstBooking' # group name
        GI = 'date_first_booking' # group index
        A = pd.DataFrame()
        A['id'] = Data['id']
        Data[GI] = pd.to_datetime(Data[GI])
        A = addNumeric(A, Data[GI].dt.year, GN + 'Year')
        A = addNumeric(A, Data[GI].dt.month, GN + 'Month')
        A = addNumeric(A, Data[GI].dt.day, GN + 'DayOfMonth')
        A = addNumeric(A, Data[GI].dt.weekofyear, GN + 'WeekOfYear')
        A = addNumeric(A, Data[GI].dt.dayofweek, GN + 'DayOfWeek')
        A = addNumeric(A, Data[GI].dt.quarter, GN + 'Quarter')
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
##### BASED ON gender
    if(Gender):
        GN = 'Gender' # group name
        GI = 'gender' # group index
        A = pd.DataFrame()
        A['id'] = Data['id']
        A = catToBinary(A, Data[GI], GN)
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
##### BASED ON age
    if(Age):
        GN = 'Age' # group name
        GI = 'age' # group index
        A = pd.DataFrame()
        A['id'] = Data['id']
        A = addNumeric(A, Data[GI], GN)
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
#### BASED ON signup_method
    if(SignupMethod):
        GN = 'SignupMethod' # group name
        GI = 'signup_method' # group index
        A = pd.DataFrame()
        A['id'] = Data['id']
        A = catToBinary(A, Data[GI], GN)
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
##### BASED ON signup_flow
    if(SignupFlow):
        GN = 'SignupFlow' # group name
        GI = 'signup_flow' # group index
        A = pd.DataFrame()
        A['id'] = Data['id']
        A = catToBinary(A, Data[GI], GN)
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
#### BASED ON language
    if(Language):
        GN = 'Language' # group name
        GI = 'language' # group index
        A = pd.DataFrame()
        A['id'] = Data['id']
        A = catToBinary(A, Data[GI], GN)
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
##### BASED ON affiliate_channel
    if(AffChannel):
        GN = 'AffChannel' # group name
        GI = 'affiliate_channel' # group index
        A = pd.DataFrame()
        A['id'] = Data['id']
        A = catToBinary(A, Data[GI], GN)
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
#### BASED ON affiliate_provider
    if(AffProvider):
        GN = 'AffProvider' # group name
        GI = 'affiliate_provider' # group index
        A = pd.DataFrame()
        A['id'] = Data['id']
        A = catToBinary(A, Data[GI], GN)
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
##### BASED ON first_affiliate_tracked
    if(FirstAffTracked):
        GN = 'FirstAffTracked' # group name
        GI = 'first_affiliate_tracked' # group index
        A = pd.DataFrame()
        A['id'] = Data['id']
        A = catToBinary(A, Data[GI], GN)
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
##### BASED ON signup_app
    if(SignupApp):
        GN = 'SignupApp' # group name
        GI = 'signup_app' # group index
        A = pd.DataFrame()
        A['id'] = Data['id']
        A = catToBinary(A, Data[GI], GN)
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
#### BASED ON first_device_type
    if(FirstDeviceType):
        GN = 'FirstDeviceType' # group name
        GI = 'first_device_type' # group index
        A = pd.DataFrame()
        A['id'] = Data['id']
        A = catToBinary(A, Data[GI], GN)
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
##### BASED ON first_browser
    if(FirstBrowser):
        GN = 'FirstBrowser' # group name
        GI = 'first_browser' # group index
        A = pd.DataFrame()
        A['id'] = Data['id']
        A = catToBinary(A, Data[GI], GN)
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
##### LABELS target
    if(Type is 'Train'):
        GN = 'Target' # group name
        GI = 'country_destination' # group index
        A = pd.DataFrame()
        A['id'] = Data['id']
        A = catToBinary(A, Data[GI], GN)
        pickle.dump( A, open( GN + Type + ".p", "wb" ) )
        print "Pickled " + GN + Type + ".p"
        
def pickleSessions(Data):
    
    Action = False
    Action_Times = False
    Action_Before = False
    Action_2gram = False
    Action_3gram = False
    Action_4gram = False
    Action_5gram = False
    
    ActionType = False
    ActionType_Times = False
    ActionType_Before = False
    ActionType_2gram = False
    ActionType_3gram = False
    ActionType_4gram = False
    ActionType_5gram = False
    
    ActionDetail = False
    ActionDetail_Times = False
    ActionDetail_Before = False
    ActionDetail_2gram = False
    ActionDetail_3gram = False
    ActionDetail_4gram = False
    ActionDetail_5gram = True
    
    DeviceType = False
    DeviceType_Times = False
    DeviceType_Before = False

    #A['id'] = Data['id']
##### BASED ON action
    if(Action):
        A = pd.DataFrame()
        GN = 'Action'
        GI = 'action'
        
        A = Data.groupby('user_id')[GI].value_counts().unstack().fillna(0)
        A.columns = [GN + 'IS' + s for s in A.columns]
        A = getStats(A, GN)
        A.to_csv(GN + 'Counts' + '.csv')
        print GN + ' written to CSV'
        
        A = Data.groupby('user_id')[GI].value_counts(normalize = True).unstack().fillna(0)
        A.columns = [GN + 'IS' + s + 'Norm' for s in A.columns]
        A.to_csv(GN + 'Norm' + '.csv')
        print GN + 'Norm' + ' written to CSV'
        
    if(Action_Times):
        GN = 'ActionTimes'
        GI = 'action'
        GT = 'secs_elapsed'
        timesToCSV(Data, GN, GI, GT, 0)
        print GN + 'written to CSV'
        
    if(Action_Before):
        GN = 'ActionBefore'
        GI = 'action'
        GT = 'secs_elapsed'
        timesToCSV(Data, GN, GI, GT, 1)
        print GN + 'written to CSV'
        
    if(Action_2gram):
        GN = 'Action2gram'
        GI = 'action'
        bigramToCSV(Data, GN, GI)
        print GN + 'written to CSV'
        
    if(Action_3gram):
        GN = 'Action3gram'
        GI = 'action'
        trigramToCSV(Data, GN, GI)
        print GN + 'written to CSV'
        
    if(Action_4gram):
        GN = 'Action4gram'
        GI = 'action'
        fourgramToCSV(Data, GN, GI)
        print GN + 'written to CSV'
        
    if(Action_5gram):
        GN = 'Action5gram'
        GI = 'action'
        fivegramToCSV(Data, GN, GI)
        print GN + 'written to CSV'
        
##### BASED ON action_type
    if(ActionType):
        A = pd.DataFrame()
        GN = 'ActionType'
        GI = 'action_type'
        
        A = Data.groupby('user_id')[GI].value_counts().unstack().fillna(0)
        A.columns = [GN + 'IS' + s for s in A.columns]
        A = getStats(A, GN)
        A.to_csv(GN + 'Counts' + '.csv')
        print GN + 'Counts' + ' written to CSV'
        
        A = Data.groupby('user_id')[GI].value_counts(normalize = True).unstack().fillna(0)
        A.columns = [GN + 'IS' + s + 'Norm' for s in A.columns]
        A.to_csv(GN + 'Norm' + '.csv')
        print GN + 'Norm' + ' written to CSV'
        
    if(ActionType_Times):
        GN = 'ActionTypeTimes'
        GI = 'action_type'
        GT = 'secs_elapsed'
        timesToCSV(Data, GN, GI, GT, 0)
        print GN + 'written to CSV'
        
    if(ActionType_Before):
        GN = 'ActionTypeBefore'
        GI = 'action_type'
        GT = 'secs_elapsed'
        timesToCSV(Data, GN, GI, GT, 1)
        print GN + 'written to CSV'
        
    if(ActionType_2gram):
        GN = 'ActionType2gram'
        GI = 'action_type'
        bigramToCSV(Data, GN, GI)
        print GN + 'written to CSV'
        
    if(ActionType_3gram):
        GN = 'ActionType3gram'
        GI = 'action_type'
        trigramToCSV(Data, GN, GI)
        print GN + 'written to CSV'
        
    if(ActionType_4gram):
        GN = 'ActionType4gram'
        GI = 'action_type'
        fourgramToCSV(Data, GN, GI)
        print GN + 'written to CSV'
        
    if(ActionType_5gram):
        GN = 'ActionType5gram'
        GI = 'action_type'
        fivegramToCSV(Data, GN, GI)
        print GN + 'written to CSV'
        
##### BASED ON action_detail
    if(ActionDetail):
        A = pd.DataFrame()
        GN = 'ActionDetail'
        GI = 'action_detail'
        
        A = Data.groupby('user_id')[GI].value_counts().unstack().fillna(0)
        A.columns = [GN + 'IS' + s for s in A.columns]
        A = getStats(A, GN)
        A.to_csv(GN + 'Counts' + '.csv')
        print GN + 'Counts' + ' written to CSV'
        
        A = Data.groupby('user_id')[GI].value_counts(normalize = True).unstack().fillna(0)
        A.columns = [GN + 'IS' + s + 'Norm' for s in A.columns]
        A.to_csv(GN + 'Norm' + '.csv')
        print GN + 'Norm' + ' written to CSV'
        
    if(ActionDetail_Times):
        GN = 'ActionDetailTimes'
        GI = 'action_detail'
        GT = 'secs_elapsed'
        timesToCSV(Data, GN, GI, GT, 0)
        print GN + 'written to CSV'
        
    if(ActionDetail_Before):
        GN = 'ActionDetailBefore'
        GI = 'action_detail'
        GT = 'secs_elapsed'
        timesToCSV(Data, GN, GI, GT, 1)
        print GN + 'written to CSV'
        
    if(ActionDetail_2gram):
        GN = 'ActionDetail2gram'
        GI = 'action_detail'
        bigramToCSV(Data, GN, GI)
        print GN + 'written to CSV'
        
    if(ActionDetail_3gram):
        GN = 'ActionDetail3gram'
        GI = 'action_detail'
        trigramToCSV(Data, GN, GI)
        print GN + 'written to CSV'
        
    if(ActionDetail_4gram):
        GN = 'ActionDetail4gram'
        GI = 'action_detail'
        fourgramToCSV(Data, GN, GI)
        print GN + 'written to CSV'
        
    if(ActionDetail_5gram):
        GN = 'ActionDetail5gram'
        GI = 'action_detail'
        fivegramToCSV(Data, GN, GI)
        print GN + 'written to CSV'
        
#### BASED ON device_type
    if(DeviceType):
        A = pd.DataFrame()
        GN = 'DeviceType'
        GI = 'device_type'
        
        A = Data.groupby('user_id')[GI].value_counts().unstack().fillna(0)
        A.columns = [GN + 'IS' + s for s in A.columns]
        A = getStats(A, GN)
        A.to_csv(GN + 'Counts' + '.csv')
        print GN + 'Counts' + ' written to CSV'
        
        A = Data.groupby('user_id')[GI].value_counts(normalize = True).unstack().fillna(0)
        A.columns = [GN + 'IS' + s + 'Norm' for s in A.columns]
        A.to_csv(GN + 'Norm' + '.csv')
        print GN + 'Norm' + ' written to CSV'
        
    if(DeviceType_Times):
        GN = 'DeviceTypeTimes'
        GI = 'device_type'
        GT = 'secs_elapsed'
        timesToCSV(Data, GN, GI, GT, 0)
        print GN + 'written to CSV' 
        
    if(DeviceType_Before):
        GN = 'DeviceTypeBefore'
        GI = 'device_type'
        GT = 'secs_elapsed'
        timesToCSV(Data, GN, GI, GT, 1)
        print GN + 'written to CSV'
        
        
#Train = pd.read_csv('train_users_2.csv')
#Test = pd.read_csv('test_users.csv')
Sessions = pd.read_csv('sessions.csv')
#pickleStuff(Train, 'Train')
#pickleStuff(Test, 'Test')
pickleSessions(Sessions)