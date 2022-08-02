import os,sys
import numpy as np
import gzip, pickle
import os.path

from scipy.stats import pearsonr

#from inc_trainset import septicarr as pool_ids

from inc_trainset import validarr as alltrain_ids
from inc_devset import validarr as alldev_ids
from inc_testset import validarr as alltest_ids

from new_ensemble_data import seeds,full_ranges,full_sizes


def readPrediction(model, candidate): 

    timeline = []
    generated = "data/data_"+str(model)+"/generated_"+str(candidate)+".dat"
    
    with open(generated,'r') as f:
        while (line := f.readline().rstrip('\n')):
            values = line.split('\t')
            timeline.append(float(values[4]))

    if len(timeline) == 0:
        print("WARN: model = %d, candidate = %d IS EMPTY"%(model, candidate) )

    return timeline



def readGroundTruth(model): 

    timeline = []
    generated = "data/data_"+str(model)+"/label_pred.dat"
    
    with open(generated,'r') as f:
        while (line := f.readline().rstrip('\n')):
            values = line.split('\t')
            timeline.append(float(values[2]))

    if len(timeline) == 0:
        print("WARN: groundtruth patient = %d IS EMPTY"%(model) )

    return timeline



def readSepticNonSeptic(valid_ids):

    septic = set()
    nonseptic = set()
    
    with open('all_mixed.dat','r') as f:
        while (line := f.readline().rstrip('\n')):
            values = line.split('\t')
            encnum = int(values[0])
            
            if encnum not in valid_ids:
                continue
            
            if encnum not in nonseptic and encnum not in septic:
                if values[1] == '0':
                    nonseptic.add(encnum)
                else: # values[1] == '1':
                    septic.add(encnum)
    
    return septic,nonseptic



#def readPredictionsAll(devids,pool): 

def readPredictionsAll(model_ids,eval_ids): 
    
    cnt = 0
    dev_preds_all = {}
    
    for candidate in model_ids:
        
        predictions = []
        
        for model in eval_ids:

            predictions += readPrediction(model,candidate)
            
        dev_preds_all[candidate] = predictions

            
        if cnt %10 == 0:
            sys.stderr.write('.')
            sys.stderr.flush()
        cnt+=1
        
    sys.stderr.write('\n')

    return dev_preds_all


def readPredictionslModelsAll(model_ids): 
    
    cnt = 0
    predictions_model = {}
    #dev_preds_all = {}
    
    predictions_all = []
    #predictions_full = []
        
    for model in model_ids:

        predictions_model[model] = readPrediction(model,'all')
        predictions_all += predictions_model[model] 
            
    if cnt %10 == 0:
        sys.stderr.write('.')
        sys.stderr.flush()
    cnt+=1
        
    sys.stderr.write('\n')

    return predictions_model,predictions_all






def readGroundTruthModelsAll(model_ids): 

    groundtruth_model = {}
    groundtruth_all = []

    for model in model_ids:
        groundtruth_model[model] = readGroundTruth(model)
        groundtruth_all += groundtruth_model[model]

    return groundtruth_model,groundtruth_all



def meanSquaredError(labels,predictions):
    
    assert(len(labels) == len(predictions))
    
    mse = (np.square(np.array(labels) - np.array(predictions))).mean(axis=None)

    return mse



def calcEnsemblePredsAll(all_predictions,ensemble_ids):
    
    predictions = []

    for model in ensemble_ids:
        #print(len(all_predictions[model]))
        predictions.append(np.array(all_predictions[model]))
    
    return np.mean(predictions,axis=0)



def check_membership(model_ids,predictions,groundtruth,training_mse):
    
    success = 0
    
    for model in model_ids:
        model_preds = predictions[model]
        model_groundtruth = groundtruth[model] #readGroundTruth(model)
        model_mse = meanSquaredError(model_groundtruth,model_preds)

        #if model in ss_ids:
        #    model_type = '(ss)'
        #else:
        #    model_type = '(ns)'
        
        if model_mse < training_mse:
            #print("Model",model,model_type,"loss (MSE):",model_mse)
            success += 1
        #else:
            #print("Model",model,model_type,"loss (MSE):",model_mse,"NON-MEMBER")

    return success


def check_membership_laplace(model_ids,predictions,groundtruth,
                             training_mse,scale):
    success = 0
    
    for model in model_ids:
        preds = np.array(predictions[model])
        noise = np.random.laplace(0.0, scale, len(preds))
        model_preds = np.add(preds,noise)

        model_groundtruth = groundtruth[model] 
        model_mse = meanSquaredError(model_groundtruth,model_preds)

        if model_mse < training_mse:
            success += 1

    return success





def existsTimelinesFile(fname):
    return os.path.isfile(fname) 
    

def loadAllTimelines(fname):

    fp=gzip.open(fname,'rb')
    ss_ids,ns_ids,dev_predictions,ground_truth = pickle.load(fp)
    fp.close()
    
    return ss_ids,ns_ids,dev_predictions,ground_truth



def saveAllTimelines(fname,ss_ids,ns_ids,dev_predictions,ground_truth):
    
    fp=gzip.open(fname,'wb')
    pickle.dump([
        ss_ids,ns_ids,dev_predictions,ground_truth
        ],fp,1)
    fp.close()




def existsTimelinesFile(fname):
    return os.path.isfile(fname) 
    

def loadAllTimelines(fname):

    fp = gzip.open(fname,'rb')
    ss_ids,ns_ids,train_preds_all,train_groundtruth_all, \
        train_preds,train_groundtruth, \
            test_preds,test_groundtruth = pickle.load(fp)
    fp.close()
    
    return ss_ids,ns_ids,train_preds_all, train_groundtruth_all, \
        train_preds,train_groundtruth,\
        test_preds,test_groundtruth


def saveAllTimelines(fname,ss_ids,ns_ids,
                     train_preds_all,train_groundtruth_all,
                     train_preds,train_groundtruth,
                     test_preds,test_groundtruth):
    
    fp=gzip.open(fname,'wb')
    pickle.dump([ss_ids,ns_ids,train_preds_all,train_groundtruth_all,
                 train_preds,train_groundtruth,
                 test_preds,test_groundtruth
                 ],fp,1)
    fp.close()








def main():

    if len(sys.argv) < 2:
        sys.stderr.write("USAGE: %s splitnum\n"%(sys.argv[0]))
        sys.exit()

    splitnum = int(sys.argv[1])

    trainset_ids = set(alltrain_ids[splitnum])
    devset_ids = set(alldev_ids[splitnum])
    testset_ids = set(alltest_ids[splitnum])
    
    splitseed = seeds[splitnum]
    np.random.RandomState(np.random.MT19937(np.random.SeedSequence(splitseed)))
    np.random.seed(splitseed)

    splitrange = full_ranges[splitnum]
    splitsize = full_sizes[splitnum]
    
    #
    # Cache all data in file
    #
    data_filename = "memberdata_fullmodel.pkl.gz"
    
    if existsTimelinesFile(data_filename):

        sys.stderr.write("INFO: loading model predictions.\n")
        ss_ids,ns_ids,train_preds_all,train_groundtruth_all, \
            train_preds,train_groundtruth, \
                test_preds,test_groundtruth = loadAllTimelines(data_filename)

        sys.stderr.write("ns=%d, ss=%d\n"%(len(ns_ids), len(ss_ids)))
        sys.stderr.write("no.preds=%d\n"%(len(train_preds)+len(test_preds)))
        sys.stderr.write("no.labels=%d\n"%(len(train_groundtruth)
                                           +len(test_groundtruth)))

        sys.stderr.write("INFO: model predictions loaded.\n")

    else:
    
        sys.stderr.write("INFO: loading timelines and predictions.\n")
        
        ss_ids,ns_ids = readSepticNonSeptic(trainset_ids.union(testset_ids))
        sys.stderr.write("ns=%d, ss=%d\n"%(len(ns_ids), len(ss_ids)))

        train_preds,train_preds_all = readPredictionslModelsAll(trainset_ids) 
        train_groundtruth,train_groundtruth_all = readGroundTruthModelsAll(trainset_ids)
        test_preds,_ = readPredictionslModelsAll(testset_ids) 
        test_groundtruth,_ = readGroundTruthModelsAll(testset_ids)

        sys.stderr.write("no.preds=%d\n"%(len(train_preds)+len(test_preds)))
        sys.stderr.write("no.labels=%d\n"%(len(train_groundtruth)
                                           +len(test_groundtruth)))

        saveAllTimelines(data_filename,
                         ss_ids,ns_ids,
                         train_preds_all,train_groundtruth_all,
                         train_preds,train_groundtruth,
                         test_preds,test_groundtruth)
                         
        sys.stderr.write("INFO: model predictions compiled.\n")

    """
    """

    trainset_mse = meanSquaredError(train_groundtruth_all,train_preds_all)

    rounds = 1000

    print("#"*50)
    print("# Full Model Training Loss (MSE):",trainset_mse)
    print("# Ensemble Min. Prediction:",min(train_preds_all))
    print("# Ensemble Max. Prediction:",max(train_preds_all))

    print("#"*50)
    print("# Membership Attack:")
    print("# pos:", len(testset_ids), "sampled",rounds,"times from" ,len(trainset_ids))
    print("# neg:", len(testset_ids))
    print("#"*50)
    
    length = len(testset_ids)

    epsilons = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000]

    for epsilon in epsilons:
        #epsilon = 50.0
        scale = ((splitrange)*(1.0/splitsize))/epsilon

        falsepos = check_membership_laplace(testset_ids,test_preds,
                                        test_groundtruth,trainset_mse,scale)
        trueneg = length - falsepos

        if falsepos == 0:
            falsepos = 1e-9
    
        recalls = []
        precisions = []
        trueposs = []

        trainlist_ids = list(trainset_ids)
    
        for i in range(rounds):

            np.random.shuffle(trainlist_ids)
        
            truepos = check_membership_laplace(trainlist_ids[0:length],train_preds,
                                           train_groundtruth,trainset_mse,scale)

            recalls.append(truepos / length)
            precisions.append(truepos / (truepos+falsepos))
            trueposs.append(truepos)
        
            #if i%10 ==0:
            #    sys.stderr.write('.')
            #    sys.stderr.flush()

        #sys.stderr.write('\n')

        #print(epsilon,np.mean(precisions),np.mean(recalls),
        #      (trueneg+np.mean(trueposs))/(2*length))

        tpr = np.mean(trueposs)/length
        fpr = (falsepos/length)
        privacy_leakage = tpr-fpr
        
        steps = 1 # int(rounds / 1)
        
        for pos in range(0,rounds,steps):

            tpr = np.mean(trueposs[pos:pos+steps])/length
            fpr = falsepos/length
            privacy_leakage = tpr-fpr

            #tpr = trueposs[i]/length
            #fpr = falsepos/length
            #privacy_leakage = tpr-fpr
        
            print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f"%(epsilon,scale,np.mean(precisions[pos:pos+steps]),
                    np.mean(trueposs[pos:pos+steps]) / length, (np.mean(trueposs[pos:pos+steps])+trueneg)/(2*length),
                    tpr,fpr,privacy_leakage))

    #print("Precision:", np.mean(precisions))
    #print("Recall:", np.mean(recalls))

    #print("Accuracy:", (trueneg+np.mean(trueposs))/(2*length))
    
    sys.exit(0)

if __name__ == "__main__":
    main()

