# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 20:54:58 2016

@author: piotrgrudzien
"""

import ReadDataXG as rd
import ReadDataXG4 as rd4

#rd.readDataMain('BasicInfo2')
#rd.readDataMain('FullStats2')
#rd.readDataMain('Times2')
#rd.readDataMain('Before2')
#rd.readDataMain('ActionNgram2')
#rd.readDataMain('ActionNgramNorm2')
#rd.readDataMain('ActionTypeNgram2')
#rd.readDataMain('ActionTypeNgramNorm2')
#rd.readDataMain('ActionDetailNgram2')
#rd.readDataMain('ActionDetailNgramNorm2')
#rd.readDataMain('BasicInfoFullStats')

# z tym sie moze jeszcze chwile wstrzymam
#rd.readDataMain('BasicInfoFullStatsBeforeActionTypeNgramActionDetailNgram')



# tamto sie skonczy to to puszczam
#rd.readDataMain('BasicInfoTimes')
#rd.readDataMain('BasicInfoBefore')
#rd.readDataMain('FullStatsTimes')
#rd.readDataMain('FullStatsBefore')

# folder z 3 w nazwie
#rd.readDataMain('BasicInfo3')
#rd.readDataMain('FullStats3')
#rd.readDataMain('Times3')

#rd.readDataMain('BasicInfoFullStats3')
#rd.readDataMain('BasicInfoTimes3')
#rd.readDataMain('BasicInfoBefore3')
#rd.readDataMain('FullStatsActionNgram3')
#rd.readDataMain('BasicInfoActionTypeNgram3')
#rd.readDataMain('FullStatsActionDetailNgram3')
#rd.readDataMain('FullStatsActionNgramNorm3')
#rd.readDataMain('BeforeActionTypeNgramNorm3', skip = True)
#rd.readDataMain('BasicInfoActionDetailNgramNorm3', skip = False)

## From now on 50/50 split
## 171 + 1112 + 3766 = 5049
#rd.readDataMain('BasicInfoFullStatsTimes3', skip = True) 
## 171 + 1112 + 3766 = 5049
#rd.readDataMain('BasicInfoFullStatsBefore3', skip = False)
## 171 + 1308 + 749 + 1181 = 3409
#rd.readDataMain('BasicInfoActionNgramActionTypeNgramActionDetailNgram3', skip = False)
## 171 + 1112 + 2508 = 3791
#rd.readDataMain('BasicInfoFullStatsActionNgramNorm', skip = False)
## 1112 + 1435 + 2281 = 4828
#rd.readDataMain('FullStatsActionTypeNgramNormActionDetailNgramNorm', skip = False)

for i in range (1, 5):
#    rd.readDataMain('BasicInfo4' + str(i), skip = False)
    rd4.readDataMain('BasicInfoFullStats4' + str(i), skip = False)
    rd4.readDataMain('BasicInfoTimes4' + str(i), skip = False)
    rd4.readDataMain('BasicInfoBefore4' + str(i), skip = False)
#rd4.readDataMain('BasicInfoActionNgram4', skip = False)
#rd4.readDataMain('BasicInfoActionNgramNorm4', skip = False)
#rd4.readDataMain('BasicInfoActionTypeNgram4', skip = False)
#rd4.readDataMain('BasicInfoActionTypeNgramNorm4', skip = False)
#rd4.readDataMain('BasicInfoActionDetailNgram4', skip = False)
#rd4.readDataMain('BasicInfoActionDetailNgramNorm4', skip = False)

#rd.readDataMain('BasicInfoFullStats3duo', skip = False)
#rd.readDataMain('BasicInfoTimes3duo', skip = False)
#rd.readDataMain('BasicInfoBefore3duo', skip = False)
#rd.readDataMain('BasicInfoFullStats3trio', skip = False)
#rd.readDataMain('BasicInfoTimes3trio', skip = False)
#rd.readDataMain('BasicInfoBefore3trio', skip = False)
#rd.readDataMain('BasicInfoFullStats3quatro', skip = False)
#rd.readDataMain('BasicInfoTimes3quatro', skip = False)
#rd.readDataMain('BasicInfoBefore3quatro', skip = False)