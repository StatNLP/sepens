from __future__ import print_function

__docformat__ = 'restructedtext en'

import os
import sys
import math
import gzip
import timeit
import numpy as np

from test_ids import valid_ns,valid_ss
from new_ensemble_data import full_ranges,full_sizes,ensemble_ranges,ensemble_sizes,seeds



skip_steps = 16 # 8h

def get_values_from_series(ser):

    vals = set()

    for pat in ser:
        for tt in ser[pat]:
            vals.add(tt[1])

    return sorted(list(vals))



def array_from_series(ser,start,end):

    new_arr = []
    for pat in ser:
        for tt in ser[pat][skip_steps:]:
            if tt[0]>=start and tt[0]<end:
                new_arr.append(tt[1])
                
    return np.array(new_arr)



def calculate_auroc_range(ns,ss, start,end, printdat=False):

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
        fp1  = len(np.where(ns_all>=thresh)[0])

        tot2 = len(ss_out)
        fp2  = len(np.where(ss_out>=thresh)[0])
        fpr = float(fp1+fp2)/(tot1+tot2)

        tot = len(ss_in)
        tp  = len(np.where(ss_in>=thresh)[0])
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



def extract_timelines_baseline_lap(scale):

    global valid_encnums, valid_ns, valid_ss, sepsis_first, sepsis_nonincident

    ns = {}
    ss = {}

    for encnum in valid_encnums:

        timeline = []
        with open("data/data_%d/label_pred_all.dat"%encnum,"r") as fh: 
        
            for row in fh:
                values = row.rstrip("\n").split()

                time = float(values[3])
                score = float(values[8]) + np.random.laplace(0.0, scale)
            
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
        #else:

    return ns,ss


def extract_timelines_models_lap(column,scale):
    
    global valid_encnums, valid_ns, valid_ss, sepsis_first, sepsis_nonincident
    
    ns = {}
    ss = {}

    for encnum in valid_encnums:

        timeline = []
        with open("data/data_%d/ensemble_test.dat"%encnum,"r") as fh: 
        
            for row in fh:
                values = row.rstrip("\n").split()

                time = float(values[1])
                score = float(values[column]) + np.random.laplace(0.0, scale)

                if encnum in valid_ss and encnum in sepsis_first:
                    time -= sepsis_first[encnum]
            
                timeline.append((time,score))

        if encnum in valid_ns:
            ns[encnum] = timeline
        elif encnum in sepsis_nonincident:
            ss[encnum] = timeline
        #else:

    return ns,ss
    


def find_first_sepsis_episode():
    
    global valid_encnums
    
    sepsis_first = {}
    with open("all_mixed.dat","r") as fh: 

        for row in fh:
            values = row.rstrip("\n").split()
        
            encnum = int(values[0])

            if encnum not in valid_encnums:
                continue
        
            septic = int(values[1])
            label = float(values[2])
            time = float(values[3])
        
            if septic == 1:

                if label >= 2.0 and encnum not in sepsis_first:
                    sepsis_first[encnum] = time

    return sepsis_first



def get_non_incident_sepsis(sepsis_first):
    
    sepsis_nonincident = {}
    for encnum in sepsis_first:

        if True:
            sepsis_nonincident[encnum] = sepsis_first[encnum]
        else:
            print("#",encnum,":",sepsis_first[encnum])

    return sepsis_nonincident



def main():

    global valid_encnums, valid_ns, valid_ss, sepsis_first, sepsis_nonincident

    if len(sys.argv) < 2:
        sys.stderr.write("USAGE: %s splitnum\n"%(sys.argv[0]))
        sys.exit()

    splitnum = int(sys.argv[1])

    splitseed = seeds[splitnum]
    np.random.RandomState(np.random.MT19937(np.random.SeedSequence(splitseed)))
    np.random.seed(splitseed)

    valid_encnums = valid_ns.union(valid_ss)

    sepsis_first = find_first_sepsis_episode()
    print("#",len(sepsis_first))
    
    sepsis_nonincident = get_non_incident_sepsis(sepsis_first)
    print("#",len(sepsis_nonincident),sepsis_nonincident)
    
    intervals = [(-4.25,-3.75),(-8.25,-7.75),(-12.25,-11.75),(-12.25,-7.75),(-24.25,-11.75)]
    epsilons = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000]

    for start,end in intervals:

        print("#Interval: %f to %f"%(start,end))

        ns, ss = extract_timelines_baseline()
        auroc_full = calculate_auroc_range(ns,ss, start,end)

        ns, ss = extract_timelines_models(3)
        auroc_ens = calculate_auroc_range(ns,ss, start,end)

        for i in range(len(epsilons)):

            scale_full = (full_ranges[splitnum] * \
                (1/full_sizes[splitnum])) / epsilons[i]
            scale_ens = (ensemble_ranges[splitnum] * \
                (1/ensemble_sizes[splitnum])) / epsilons[i]

            print("#epsilon,scale_full,auroc_full_noise,acc_loss_full," \
                  "scale_ens,auroc_ens_noise,acc_loss_ens")

            for run in range(10):

                ns, ss = extract_timelines_baseline_lap(scale_full)
                auroc_full_noise = calculate_auroc_range(ns,ss, start,end)
                acc_loss_full = 1 - ((2*auroc_full_noise - 1)/(2*auroc_full - 1))

                ns, ss = extract_timelines_models_lap(3,scale_ens)
                auroc_ens_noise = calculate_auroc_range(ns,ss, start,end)
                acc_loss_ens = 1 - ((2*auroc_ens_noise - 1)/(2*auroc_ens - 1))

                sys.stdout.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(epsilons[i],
                                 scale_full,auroc_full_noise,acc_loss_full,
                                 scale_ens,auroc_ens_noise,acc_loss_ens))


if __name__ == "__main__":
    main()

