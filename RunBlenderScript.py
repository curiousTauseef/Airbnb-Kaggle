# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 20:54:58 2016

@author: piotrgrudzien
"""

import BlenderXGS as b
import BlenderXGS4 as b4
import BlenderRawSimple as brs

#b.blenderXGMain(['BasicInfoFullStats3'], ss = 0.5)
#b.blenderXGMain(['BasicInfoTimes3'], ss = 0.5)
#b.blenderXGMain(['BasicInfoFullStats3', 'BasicInfoTimes3'], ss = 0.5)
#b.blenderXGMain(['BasicInfoBefore3'], ss = 0.5)
#b.blenderXGMain(['BasicInfoFullStats3', 'BasicInfoBefore3'], ss = 0.5)
#b.blenderXGMain(['BasicInfoTimes3', 'BasicInfoBefore3'], ss = 0.5)
#b.blenderXGMain(['BasicInfoFullStats3', 'BasicInfoTimes3', 'BasicInfoBefore3'], ss = 0.5)

#b.blenderXGMain(['FullStatsActionDetailNgram3'])

#br.blenderXGMain('BasicInfo', 'BasicInfo', fraction = 0.1)
#br.blenderXGMain('FullStats', 'FullStats', fraction = 0.1)
#br.blenderXGMain('Times', 'Times', fraction = 0.1)
#br.blenderXGMain('Before', 'Before', fraction = 0.1)
#br.blenderXGMain('ActionNgram', 'ActionNgram', fraction = 0.1)
#br.blenderXGMain('ActionNgramNorm', 'ActionNgramNorm', fraction = 0.1)
#br.blenderXGMain('ActionTypeNgram', 'ActionTypeNgram', fraction = 0.1)
#br.blenderXGMain('ActionTypeNgramNorm', 'ActionTypeNgramNorm', fraction = 0.1)
#br.blenderXGMain('ActionDetailNgram', 'ActionDetailNgram', fraction = 0.1)
#br.blenderXGMain('ActionDetailNgramNorm', 'ActionDetailNgramNorm', fraction = 0.1)
#br.blenderXGMain('BasicInfoFullStats', 'FullStats', fraction = 0.1)

#FOLDER = ['BasicInfo', 'FullStats']
#brs.blenderXGMain(FOLDER, fraction = 1, namepart = 'Row194')
#
#FOLDER = ['BasicInfo', 'FullStats', 'Times']
#brs.blenderXGMain(FOLDER, fraction = 1, namepart = 'Row195')
#
#FOLDER = ['BasicInfo', 'FullStats', 'Times', 'Before']
#brs.blenderXGMain(FOLDER, fraction = 1, namepart = 'Row196')
#
#FOLDER = ['BasicInfo', 'FullStats', 'Times', 'Before', 'ActionNgram', 'ActionNgramNorm', 'ActionTypeNgram', 'ActionTypeNgramNorm', 'ActionDetailNgram', 'ActionDetailNgramNorm']
#brs.blenderXGMain(FOLDER, fraction = 1, namepart = 'Row197')
#
#b.blenderXGMain(['FullStatsActionNgram3'], ss = 1)
#b.blenderXGMain(['BasicInfoTimes3', 'FullStatsActionNgram3'], ss = 1)
#b.blenderXGMain(['BasicInfoBefore3', 'FullStatsActionNgram3'], ss = 1)
#b.blenderXGMain(['BasicInfoActionTypeNgram3', 'BasicInfoTimes3', 'FullStatsActionNgram3'], ss = 1)

#b.blenderXGMain(['BasicInfoActionTypeNgram3'], ss = 1)
#b.blenderXGMain(['FullStatsActionNgram3', 'BasicInfoActionTypeNgram3'], ss = 1)
#b.blenderXGMain(['BasicInfoFullStats3', 'BasicInfoActionTypeNgram3'], ss = 1)
#b.blenderXGMain(['BasicInfoFullStats3', 'FullStatsActionDetailNgram3', 'BasicInfoActionTypeNgram3'], ss = 1)

#b.blenderXGMain(['FullStatsActionDetailNgram3'], ss = 1)
#b.blenderXGMain(['BasicInfoTimes3', 'FullStatsActionDetailNgram3'], ss = 1)
#b.blenderXGMain(['BasicInfoBefore3', 'FullStatsActionDetailNgram3'], ss = 1)
#b.blenderXGMain(['BasicInfoActionTypeNgram3', 'BasicInfoTimes3', 'FullStatsActionDetailNgram3'], ss = 1)

#b.blenderXGMain(['BasicInfoFullStats3', 'BasicInfoBefore3', 'FullStatsActionNgram3'], ss = 1)
#b.blenderXGMain(['FullStats3', 'BasicInfoTimes3', 'BasicInfoActionTypeNgram3'], ss = 1)
#b.blenderXGMain(['BasicInfoTimes3', 'BasicInfoBefore3', 'FullStatsActionDetailNgram3'], ss = 1)

#b.blenderXGMain(['FullStatsActionNgram3', 'BasicInfoActionTypeNgram3', 'FullStatsActionDetailNgram3'], ss = 1)
#b.blenderXGMain(['BasicInfoFullStats3', 'BasicInfoTimes3', 'BasicInfoBefore3'], ss = 1)
#b.blenderXGMain(['FullStatsActionNgram3', 'BasicInfoFullStats3', 'BasicInfoTimes3'], ss = 1)
#b.blenderXGMain(['BasicInfoActionTypeNgram3', 'BasicInfoFullStats3', 'BasicInfoBefore3'], ss = 1)
#b.blenderXGMain(['FullStatsActionDetailNgram3', 'BasicInfoFullStats3', 'BasicInfoTimes3'], ss = 1)
#b.blenderXGMain(['FullStatsActionDetailNgram3', 'BasicInfoFullStats3', 'BasicInfoBefore3'], ss = 1)

#b.blenderXGMain(['FullStatsActionNgramNorm3', 'BasicInfoFullStats3', 'BasicInfoTimes3'], ss = 1)
#b.blenderXGMain(['FullStatsActionNgramNorm3', 'BasicInfoFullStats3', 'BasicInfoBefore3'], ss = 1)
#b.blenderXGMain(['FullStatsActionNgramNorm3', 'BasicInfoBefore3', 'BasicInfoActionTypeNgram3'], ss = 1)
#b.blenderXGMain(['FullStatsActionNgramNorm3', 'BasicInfoTimes3', 'FullStatsActionDetailNgram3'], ss = 1)
#b.blenderXGMain(['FullStatsActionNgramNorm3', 'BasicInfoTimes3', 'BasicInfoBefore3'], ss = 1)

#b.blenderXGMain(['BasicInfoFullStats3', 'BasicInfoTimes3', 'BasicInfoBefore3', 'FullStatsActionNgram3'], ss = 1)
#b.blenderXGMain(['BasicInfoFullStats3', 'BasicInfoTimes3', 'BasicInfoBefore3', 'BasicInfoActionTypeNgram3'], ss = 1)
#b.blenderXGMain(['BasicInfoFullStats3', 'BasicInfoTimes3', 'BasicInfoBefore3', 'FullStatsActionDetailNgram3'], ss = 1)
#b.blenderXGMain(['BasicInfoFullStats3', 'BasicInfoTimes3', 'BasicInfoBefore3', 'FullStatsActionNgramNorm3'], ss = 1)
#b.blenderXGMain(['BasicInfoFullStats3', 'BasicInfoTimes3', 'BasicInfoBefore3', 'BeforeActionTypeNgramNorm3'], ss = 1)
#b.blenderXGMain(['BasicInfoFullStats3', 'BasicInfoTimes3', 'BasicInfoBefore3', 'BasicInfoActionDetailNgramNorm3'], ss = 1)

#b4.blenderXGMain(['BasicInfoFullStats4'], ss = 1, N_trees = 500)
#b4.blenderXGMain(['BasicInfo4'], ss = 1, N_trees = 100)
#b4.blenderXGMain(['BasicInfo4'], ss = 1, N_trees = 100)

#b4.blenderXGMain(['BasicInfoTimes4'], ss = 1, N_trees = 500)
#b4.blenderXGMain(['BasicInfoBefore4'], ss = 1, N_trees = 500)
#b4.blenderXGMain(['BasicInfoActionNgram4'], ss = 1, N_trees = 500)
#b4.blenderXGMain(['BasicInfoActionNgramNorm4'], ss = 1, N_trees = 500)
#b4.blenderXGMain(['BasicInfoActionTypeNgram4'], ss = 1, N_trees = 500)
#b4.blenderXGMain(['BasicInfoActionTypeNgramNorm4'], ss = 1, N_trees = 500)
#b4.blenderXGMain(['BasicInfoActionDetailNgram4'], ss = 1, N_trees = 500)
#b4.blenderXGMain(['BasicInfoActionDetailNgramNorm4'], ss = 1, N_trees = 500)

#b4.blenderXGMain(['BasicInfoFullStats40'], ss = 1, N_trees = 500)
#b4.blenderXGMain(['BasicInfoFullStats3duo'], ss = 1, N_trees = 500)
b4.blenderXGMain(['BasicInfoFullStats41'], ss = 1, N_trees = 500)
b4.blenderXGMain(['BasicInfoTimes41'], ss = 1, N_trees = 500)
b4.blenderXGMain(['BasicInfoBefore41'], ss = 1, N_trees = 500)
b4.blenderXGMain(['BasicInfoFullStats42'], ss = 1, N_trees = 500)
