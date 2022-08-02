#!/usr/bin/env python

import sys,math
import numpy as np

from new_ensemble import ensemble as models



def averageTimelines(timelines):
    
    averaged = []
    for i in range(len(timelines[0])):  # timelines have all same length
        valuesum = 0.0
        for j in range(len(timelines)): # number of timelines
            valuesum += timelines[j][i]
        averaged.append(valuesum/len(timelines))

    return averaged



def arithmeticWeightedTimelines(timelines, labels):
    
    # at the beginning, all models have the same weight
    weights = [1.0]*len(timelines)
    averaged = []
    
    for i in range(len(timelines[0])):  # timelines have all same length
        valuesum = 0.0

        for j in range(len(timelines)): # number of timelines
            valuesum += weights[j]*timelines[j][i]
        averaged.append(valuesum/(sum(weights)))

        # update weights by calculating similarity
        for j in range(len(timelines)): # number of timelines
            weights[j] += 1.0/(1+abs(timelines[j][i]-labels[i])*abs(timelines[j][i]-labels[i])) 
        
    return averaged



def loadPredictions(path):

    predictions = []

    for model in models:

        filename = path+'/generated_'+str(model)+'.dat'
        timeline = []

        with open(filename) as fh:
            for line in fh:
                values = line.rstrip('\n').split()
                timeline.append(float(values[4]))
                #timeline.append(float(values[7]))
            predictions.append(timeline)
            
    return predictions



def loadGroundTruth(path):

    truth = []
    filename = path+'/test.dat'

    with open(filename) as fh:
        for line in fh:
            values = line.rstrip('\n').split()
            if float(values[2]) >= 2.0:
                truth.append(1.0)
            else:
                truth.append(0.0)

    return truth

#
# Main
#
if len(sys.argv) != 2:
    sys.stderr.write("USAGE: python %s encnum\n"%sys.argv[0])
    sys.exit(0)


encnum = int(sys.argv[1])
path = 'data/data_'+str(encnum)

predictions = loadPredictions(path)
groundtruth = loadGroundTruth(path)

# check if all have same length
for i in range(0,len(predictions)):
    assert(len(groundtruth)==len(predictions[i]))

np.random.RandomState(np.random.MT19937(np.random.SeedSequence(123456789)))

avg = averageTimelines(predictions)
arithmetic = arithmeticWeightedTimelines(predictions,groundtruth)

for i in range(0,len(groundtruth)):
    
    sys.stdout.write("%d\t%1.2f\t%1.1f\t%f\t%f\t%f\t%f"%(encnum, 
                                                         i*.5,
                                                         groundtruth[i],
                                                         avg[i],
                                                         arithmetic[i]))
    sys.stdout.write("\n")
