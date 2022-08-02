import os,sys
import numpy as np
import gzip, pickle
import os.path

import numpy as np

from scipy.stats import pearsonr


#from inc_trainset import septicarr as pool_ids

from inc_trainset import validarr as alltrain_ids
from inc_devset import validarr as alldev_ids
from inc_testset import validarr as alltest_ids

from new_ensemble import ensemble as ensemble_ids
from new_ensemble_data import seeds,ensemble_ranges,ensemble_sizes


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


def readGroundTruthAll(ids): 

    ground_truth = []
    
    for model in ids:

        ground_truth += readGroundTruth(model)

    return ground_truth



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





def readPredictionsModelsEns(model_ids,ensemble_ids): 
    
    #cnt = 0
    predictions = {}
    ens_predictions = {}
    
    for model in model_ids:
        
        predictionlist = []
        predictions[model]= []
        
        for ensmodel in ensemble_ids:

            pred = readPrediction(model,ensmodel)

            predictions[model].append(np.array(pred))
            predictionlist.append(np.array(pred))
            
        ens_predictions[model] = np.mean(predictionlist,axis=0)
        
        #if cnt %10 == 0:
        #    sys.stderr.write('.')
        #    sys.stderr.flush()
        #cnt+=1
        
    #sys.stderr.write('\n')

    return predictions, ens_predictions



def readGroundTruthSingle(model_ids): 

    ground_truth = {}
    
    for model in model_ids:

        ground_truth[model] = readGroundTruth(model)

    return ground_truth





def meanSquaredError(labels,predictions):
    
    assert(len(labels) == len(predictions))
    
    mse = (np.square(np.array(labels) - np.array(predictions))).mean(axis=None)

    return mse




def calcEnsemblePredsAll(all_predictions,ensemble_ids):
    
    predictions = []
    predictions_single = []

    for model in ensemble_ids:
        #print(len(all_predictions[model]))
        predictions.append(np.array(all_predictions[model]))
        predictions_single += all_predictions[model]
    
    return np.mean(predictions,axis=0),predictions_single



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


def add_laplace_reduce(predictions,model_ids,scale):

    noisy_preds = {}
    
    for model in model_ids:
        noisy_model = []
        for preds in predictions[model]:
            noise = np.random.laplace(0.0, scale, len(preds))
            noisy_model.append(np.add(preds,noise))

        noisy_preds[model] = np.mean(noisy_model,axis=0)

    return noisy_preds


def add_laplace(predictions,scale):

    noisy_preds = {}
    
    for model,preds in predictions.items():
            
        noise = np.random.laplace(0.0, scale, len(preds))
        noisy_preds[model] = np.add(preds,noise)

    return noisy_preds


def existsTimelinesFile(fname):
    return os.path.isfile(fname) 
    



def existsTimelinesFile(fname):
    return os.path.isfile(fname) 
    

def loadAllTimelines(filename):

    fp=gzip.open(filename,'rb')
    ss_ids,ns_ids,train_predictions,train_groundtruth, \
        ens_predictions,ens_predictions_ens,ens_groundtruth, \
            test_predictions,test_predictions_ens,test_groundtruth \
                = pickle.load(fp)

    #ss_ids,ns_ids,dev_predictions,ground_truth = pickle.load(fp)
    fp.close()
    
    return ss_ids,ns_ids,train_predictions,train_groundtruth, \
        ens_predictions,ens_predictions_ens,ens_groundtruth, \
            test_predictions,test_predictions_ens,test_groundtruth


def saveAllTimelines(filename,ss_ids,ns_ids,
                     train_predictions,train_groundtruth,
                     ens_predictions,ens_predictions_ens,ens_groundtruth,
                     test_predictions,test_predictions_ens,test_groundtruth):
    #saveAllTimelines(fname,ss_ids,ns_ids,dev_predictions,ground_truth):
    
    fp=gzip.open(filename,'wb')
    pickle.dump([
        ss_ids,ns_ids,train_predictions,train_groundtruth,
        ens_predictions,ens_predictions_ens,ens_groundtruth,
        test_predictions,test_predictions_ens,test_groundtruth
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

    data_filename = "memberdata_ensemble.pkl.gz"

    if existsTimelinesFile(data_filename):

        sys.stderr.write("INFO: loading model predictions.\n")
        ss_ids,ns_ids,train_predictions,train_groundtruth, \
            ens_predictions,ens_predictions_ens,ens_groundtruth, \
                test_predictions,test_predictions_ens,test_groundtruth \
                    = loadAllTimelines(data_filename)

        sys.stderr.write("ns=%d, ss=%d\n"%(len(ns_ids), len(ss_ids)))
        sys.stderr.write("no.preds=%d\n"%(len(train_predictions)))
        sys.stderr.write("no.labels=%d\n"%(len(train_groundtruth)))

        sys.stderr.write("INFO: model predictions loaded.\n")

    else:
    
        sys.stderr.write("INFO: loading timelines and predictions.\n")
        
        ss_ids,ns_ids = readSepticNonSeptic(ensemble_ids.union(set(testset_ids)))
        sys.stderr.write("ns=%d, ss=%d\n"%(len(ns_ids), len(ss_ids)))

        train_predictions = readPredictionsAll(ensemble_ids,trainset_ids)
        sys.stderr.write("no.preds=%d\n"%(len(train_predictions)))

        train_groundtruth = readGroundTruthAll(trainset_ids)
        sys.stderr.write("no.labels=%d\n"%(len(train_groundtruth)))

        ens_predictions,ens_predictions_ens = readPredictionsModelsEns(ensemble_ids,ensemble_ids)
        ens_groundtruth = readGroundTruthSingle(ensemble_ids)

        test_predictions,test_predictions_ens = readPredictionsModelsEns(testset_ids,ensemble_ids)
        test_groundtruth = readGroundTruthSingle(testset_ids)

        saveAllTimelines(data_filename,ss_ids,ns_ids,
                         train_predictions,train_groundtruth,
                         ens_predictions,ens_predictions_ens,ens_groundtruth,
                         test_predictions,test_predictions_ens,test_groundtruth)
        sys.stderr.write("INFO: model predictions compiled.\n")


    train_predictions_model,_ = readPredictionsModelsEns(trainset_ids,ensemble_ids)
    train_groundtruth_model = readGroundTruthSingle(trainset_ids)

    trainset_preds,trainset_preds_single = calcEnsemblePredsAll(train_predictions,ensemble_ids)
    trainset_mse = meanSquaredError(train_groundtruth,trainset_preds)

    rounds = 1000
    
    print("#"*50)
    print("# Ensemble Training Loss (MSE):",trainset_mse)
    print("# Ensemble Min. Prediction:",min(trainset_preds_single))
    print("# Ensemble Max. Prediction:",max(trainset_preds_single))
    print("#"*50)
    print("# Membership Attack:")
    print("# pos:", len(testset_ids), "sampled",rounds,"times from" ,len(trainset_ids))
    print("# neg:", len(testset_ids))
    print("#"*50)
    
    #length = len(ensemble_ids)
    length = len(testset_ids)

    epsilons = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000]

    _,train_predictions_ens = readPredictionsModelsEns(trainset_ids,ensemble_ids)

    for epsilon in epsilons:
        
        #scale = ((max(trainset_preds)-min(trainset_preds))/ensemble_sizes[splitnum])/epsilon
        scale = (4.0/ensemble_sizes[splitnum])/epsilon

        """
        noisy_predictions = add_laplace_reduce(ens_predictions,ensemble_ids,scale)

        truepos = check_membership(ensemble_ids,noisy_predictions,
                                    ens_groundtruth,trainset_mse)
        """

        noisy_predictions = add_laplace(test_predictions_ens,scale)

       #truepos = check_membership(ensemble_ids,noisy_predictions,
       #                             ens_groundtruth,trainset_mse)

        
        #noisy_predictions = add_laplace_reduce(test_predictions,testset_ids,scale)
        falsepos = check_membership(testset_ids,noisy_predictions,
                                    test_groundtruth,trainset_mse)
        trueneg = len(testset_ids)-falsepos

        if falsepos == 0:
            falsepos = 1e-9
        
        precisions = []
        recalls = []
        trueposs = []
    
        trainlist_ids = list(trainset_ids)
    
        for i in range(rounds):
        
            np.random.shuffle(trainlist_ids)

            #noisy_predictions = add_laplace_reduce(train_predictions_ens,trainset_ids,scale)
            noisy_predictions = add_laplace(train_predictions_ens,scale)

            truepos = check_membership(trainlist_ids[0:length],noisy_predictions,
                                    train_groundtruth_model,trainset_mse)
            trueposs.append(truepos)
            precisions.append(truepos / (truepos+falsepos))
            recalls.append(truepos / length)
        
        #print(epsilon,np.mean(precisions),np.mean(recalls), (trueneg+np.mean(trueposs))/(2*length))

        tpr = np.mean(trueposs)/length
        fpr = falsepos/length
        privacy_leakage = tpr-fpr
        
        #print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f"%(epsilon,scale,np.mean(precisions),
        #            truepos / length, (np.mean(trueposs)+trueneg)/(2*length),
        #            tpr,fpr,privacy_leakage))
        
        steps = 1 # int(rounds / 10)
        
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

if __name__ == "__main__":
    main()

