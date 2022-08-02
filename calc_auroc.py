from __future__ import print_function

__docformat__ = 'restructedtext en'

import os
import sys
import math
import gzip
import timeit
import numpy

from test_ids import valid_ns,valid_ss


skip_steps = 16 # 8h


def get_values_from_series(ser):

    vals = set()

    #print(ser)
    for pat in ser:
        #print(ser[pat])
        for tt in ser[pat]:
            vals.add(tt[1])

    return sorted(list(vals))



def array_from_series(ser,start,end):

    new_arr = []
    for pat in ser:
        for tt in ser[pat][skip_steps:]:
            if tt[0]>=start and tt[0]<end:
                new_arr.append(tt[1])
                
    return numpy.array(new_arr)



def count_nprows_gt(nparr, threshold):

    count = 0
    for nprow in nparr:
        #print(nprow)
        if len(numpy.where(nprow>=threshold)[0])>0:
            count += 1

    return count



def count_nprows_lt(nparr, threshold):

    count = 0
    for nprow in nparr:
        #print(nprow)
        if len(numpy.where(nprow<threshold)[0])>0:
            count += 1

    return count



def calculate_auroc_avg_range(ns,ss, start,end, printdat=False):

    ns_vals = get_values_from_series(ns)
    ss_vals = get_values_from_series(ss)

    threshes = sorted(ns_vals+ss_vals)

    res = set()
    res.add( (0.0,0.0) )

    ns_all = array_from_series(ns,  -1e6, 1e6)
    ss_out = array_from_series(ss,  -1e6, start)
    ss_in  = array_from_series(ss, start, end)

    for thresh in threshes:

        fp = 0.0
        tp = 0.0
        
        tot1 = len(ns_all)
        fp1  = len(numpy.where(ns_all>=thresh)[0])

        tot2 = len(ss_out)
        fp2  = len(numpy.where(ss_out>=thresh)[0])
        fpr = float(fp1+fp2)/(tot1+tot2)

        tot = len(ss_in)
        tp  = len(numpy.where(ss_in>=thresh)[0])
        tpr = float(tp)/tot

        if (fpr,tpr) not in res:
            res.add( (fpr,tpr) )

    aurocdat = sorted(list(res))

    if printdat:
        for key,value in aurocdat:
            print( key,value )

    auroc = 0.0
    for idx in range(len(aurocdat)-1):
        
        ww = aurocdat[idx+1][0]-aurocdat[idx][0]
        hh = aurocdat[idx][1]+aurocdat[idx+1][1]
        auroc += (ww*hh)/2

    return auroc



def extract_timelines_baseline():

    global valid_encnums, valid_ns, valid_ss, sepsis_first, sepsis_nonincident

    ns = {}
    ss = {}

    for encnum in valid_encnums:

        timeline = []
        with open("data/data_%d/label_pred_all.dat"%encnum,"r") as fh: 
        
            for row in fh:
                values = row.rstrip("\n").split()

                time = float(values[3])
                score = float(values[8])
            
                if encnum in valid_ss and encnum in sepsis_first:
                    time -= sepsis_first[encnum]
            
                timeline.append((time,score))
    
        if encnum in valid_ns:
            ns[encnum] = timeline
        elif encnum in sepsis_nonincident:
            ss[encnum] = timeline
        #else:

    return ns,ss



def extract_timelines_models(column):
    
    global valid_encnums, valid_ns, valid_ss, sepsis_first, sepsis_nonincident
    
    ns = {}
    ss = {}

    for encnum in valid_encnums:

        timeline = []

        with open("data/data_%d/ensemble_test.dat"%encnum,"r") as fh: 
        
            for row in fh:
                values = row.rstrip("\n").split()

                time = float(values[1])
                score = float(values[column])
            
                if encnum in valid_ss and encnum in sepsis_first:
                    time -= sepsis_first[encnum]
            
                timeline.append((time,score))

        if encnum in valid_ns:
            ns[encnum] = timeline
        elif encnum in sepsis_nonincident:
            ss[encnum] = timeline

    return ns,ss


#
# Main 
#

valid_encnums = valid_ns.union(valid_ss)    # test ids

#
# extract first sepsis episode
#
curr_encnum = 0
sepsis_first = {}

with open("test.dat","r") as fh: 

    for row in fh:
        values = row.rstrip("\n").split()
        
        encnum = int(values[0])

        if encnum not in valid_encnums:
            sys.stderr.write("WARN: id=%d not found in test data.\n"%encnum)
            continue
        
        septic = int(values[1])
        label = float(values[2])
        time = float(values[3])
        
        if septic == 1:

            if label >= 2.0 and encnum not in sepsis_first:
                sepsis_first[encnum] = time

sepsis_nonincident = {}
for encnum in sepsis_first:
    sepsis_nonincident[encnum] = sepsis_first[encnum]

#
# calculate AUROC for different intervals
#
intervals = [(-4.25,-3.75),(-8.25,-7.75),(-12.25,-11.75),(-12.25,-7.75),(-24.25,-11.75)]

for start,end in intervals:

  print("Interval: %f to %f"%(start,end))

  ns, ss = extract_timelines_baseline()
  auroc = calculate_auroc_avg_range(ns,ss, start,end, printdat=False)
  print("full model:", auroc)

  ns, ss = extract_timelines_models(3)    # 3rd column: uniform weighting
  auroc = calculate_auroc_avg_range(ns,ss, start,end, printdat=False)
  print("uniform:   ", auroc)

  ns, ss = extract_timelines_models(4)    # 4th column: adaptive weights
  auroc = calculate_auroc_avg_range(ns,ss, start,end, printdat=False)
  print("weighted:  ", auroc)

