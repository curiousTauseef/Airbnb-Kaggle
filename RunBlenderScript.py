# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 20:54:58 2016

@author: piotrgrudzien
"""

import BlenderXGS as b
import BlenderRawXG as br

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

FOLDER = ['BasicInfo', 'FullStats', 'Times', 'Before', 'ActionNgram', 'ActionNgramNorm', 'ActionTypeNgram', 'ActionTypeNgramNorm', 'ActionDetailNgram', 'ActionDetailNgramNorm']
TOP = [100 for x in FOLDER]
br.blenderXGMain(FOLDER, TOP, fraction = 0.4)
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