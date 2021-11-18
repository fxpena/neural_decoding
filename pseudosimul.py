import numpy as np
from numpy.random import default_rng
#instantiate a Generator
rng = default_rng(12345)
from sklearn import svm
from os import path
import os
import glob
import scipy.io as sio #to load matlab files
#package for imbalanced learning
from imblearn import over_sampling, under_sampling
from sklearn import preprocessing
    
def decode(session_data,n_iter,nTrialsTrain,training_flag=False,shuffle_flag=False): 
    '''Discriminate activity for a pseudo-simultaneous population
    INPUT
    session_data: list of data for each session. each session is a list that 
        contains two matrices with the same # of columns because columns 
        represent neurons. The # rows in each matrix are trials
    n_iter: # of iterations for sampling and testing
    nTrialsTrain: # of trials in the final sampling of trials for each neuron
        on each class (e.g. CS+ class vs non-CS class)
    OUTPUT
    accuracy: accuracy on the held-out data. column vector with 1 value for each
        iteration
    acc_training: accuracy on the training set
    acc_shuffled: accuracy on held-out data, but the labels were shuffled
    '''
    
    allow_balancing = True
    proportion4test = 0.45
    #make the # of trials in the training and testing sets the same
    #this parameter applies to the positive class and the negative class separately
    nTrialsTest = nTrialsTrain
    #initialize outputs as column vectors
    accuracy = np.zeros((n_iter,))
    acc_training= np.zeros((n_iter,))
    acc_shuffled= np.zeros((n_iter,))
    for j in range(n_iter):
        #initialize empty lists for holding arrays
        train=[]
        test=[]
        train_shuf=[]
        test_shuf=[]
        # data={
        #     'X_train': [],
        #     'X_test': [],
        #     'y_train': [],
        #     'y_test': []}
        # shuffled_data={
        #     'X_train': [],
        #     'X_test': [],
        #     'y_train': [],
        #     'y_test': []}
        for session in session_data:
            Xpos = session[0]
            Xneg = session[1]
            X = np.concatenate((Xpos,Xneg),axis=0)
            #create a permutation by shuffling the rows across classes
            X = rng.permutation(X)
            #split the combined matrix to allow the pseudo-population to be
            #created with independent test/train sets
            XposShuffled = X[:len(Xpos)]
            XnegShuffled = X[len(Xpos):]
            #standardize the # of trials so that the matrices can be concatenated
            #across sessions/animals
            Xtrain, Xtest = splitAndOversample(Xpos,Xneg,proportion4test,
                                              nTrialsTrain,
                                              balance=allow_balancing)
            #store the training and testing matrices
            train.append(Xtrain)
            test.append(Xtest)
            #repeat this process for the shuffled data
            Xtrain, Xtest = splitAndOversample(XposShuffled,XnegShuffled,
                                              proportion4test,nTrialsTrain,
                                              balance=allow_balancing)
            train_shuf.append(Xtrain)
            test_shuf.append(Xtest)
        #once all sessions have been sampled
        #concatenate the matrices horizontally
        train = np.concatenate(train,axis=1)
        test= np.concatenate(test,axis=1)
        train_shuf=np.concatenate(train_shuf,axis=1)
        test_shuf=np.concatenate(test_shuf,axis=1)
        #create the labels as a 1d array
        trainingLabels = np.full((len(train),),False)
        #write the first half to true
        trainingLabels[:nTrialsTrain] = True
        #repeat for the test data
        testLabels = np.full((len(test),),False)
        testLabels[:nTrialsTest] = True
        #train an SVM
        SVM=svm.LinearSVC(fit_intercept=False,max_iter=5000)
        SVM = SVM.fit(train,trainingLabels)
        #test the model on the held out data and calculate accuracy
        accuracy[j] = SVM.score(test,testLabels)
        if training_flag:
            #test the model on the training data set
            acc_training[j] = SVM.score(train, trainingLabels)
        if shuffle_flag:
            #train on the shuffled data
            SVM = SVM.fit(train_shuf,trainingLabels)
            #test the model on the shuffled data
            acc_shuffled[j] = SVM.score(test_shuf,testLabels)
    
    return accuracy,acc_training,acc_shuffled

def prepDecoder(path_name,normalize_flag=True): 
    ''' load matrices from matlab files in the given folder
    INPUT
    path_name: full path name for folder of matlab files
    OPTIONAL INPUT
    normalize_flag: true to normalize each neuron to [0,1]. set to False to 
        prevent this min-max scaling
    OUTPUT
    sessionData: a list of session data where each session is a list of 2 matrices
    '''
    #collect only the files names with the .mat extension
    file_names=glob.glob(path.join(path_dir,'*.mat'))
    #instantiate object for scaling all data to 0 to 1
    scaler=preprocessing.MinMaxScaler()
    #initialize a list for the lists
    sessionData = []
    
    #collect the data from each session into 1 list of lists
    for file in file_names:
        #load the matlab file as a dict
        mat_contents = sio.loadmat(file)
        #lick and no lick data come out in separate matrices
        out1=mat_contents['X1']
        out2=mat_contents['X2']
        #if there are trials
        if len(out1)>0:
            #train the scaler on the data concatenated along the columns (neurons)
            scaler.fit(np.concatenate((out1,out2),axis=0))
            if normalize_flag:
                #normalize each array
                out1=scaler.transform(out1)
                out2=scaler.transform(out2)
            #store the data in the output
            sessionData.append([out1,out2])

    return sessionData

def splitAndBalance(Xpos,Xneg,prop): 
    '''Prepare data for classification by splitting for train-test and balancing
    the labels within each split
    INPUTS
    Xpos: matrix of neural data for the positive class (e.g. CS+). rows are
       trials. columns are neurons.
    Xneg: same as Xpos but for the negative class (e.g. CS-). The # of columns should be the
       same for these two inputs. The # rows should be very close to equal
    prop: proportion of the raw data to use for test set
    OUTPUTS
    Xtrain:
    Xtest:
    '''
    trial_min=2 # # of test trials must be at least 2
    #randomize the positions of rows in the matrix
    Xpos = rng.permutation(Xpos)
    Xneg = rng.permutation(Xneg)
    #split each of these sets into training and testing sets
    num4test = round(len(Xpos) * prop)
    #choose the greater between the trial minimum and the calculated #
    num4test = max([trial_min,num4test])
    #use a random subset of the data for testing purposes
    posTest = Xpos[0:num4test,:]
    #use the rest of the data for training purposes
    posTrain = Xpos[num4test:,:]
    #repeat for the negative class
    num4test = round(Xneg.shape[0] * prop)
    num4test = max([trial_min,num4test])
    negTest = Xneg[0:num4test]
    negTrain = Xneg[num4test:]
    #apply sampling strategy to the training data
    X_train,y_train = overAndUnderSampleSK(posTrain,negTrain)
    #apply balancing strategy to the test data
    X_test,y_test = overAndUnderSampleSK(posTest, negTest)
    
    return X_train, X_test, y_train, y_test

def splitAndOversample(Xpos,Xneg,prop,nTrialsTrain,nTrialsTest=None,balance = False): 
    '''Prepare data for classification by partitioning it and oversampling to a
    specific # of trials (rows)
    INPUTS
    Xpos: matrix of neural data for the positive class (e.g. CS+). rows are
       trials. columns are neurons.
    Xneg: same as Xpos but for the negative class (e.g. CS-). The # of columns should be the
       same for these two inputs. The # rows should be very close to equal
    prop: proportion of the raw data to use for test set
    nTrialsTrain: # of trials to sample for each class in the final training set
    nTrialsTest: same as above for the final testing set
    OPTIONAL INPUT
    balance: set to true to balance the training data with oversampling and undersampling
    OUTPUTS
    Xtrain:
    Xtest:
    '''
    if nTrialsTest is None:
        #use the same value as the # of training trials
        nTrialsTest=nTrialsTrain
    trial_min=2 # # of test trials must be at least 2
    #randomize the positions of rows in the matrix
    Xpos = rng.permutation(Xpos)
    Xneg = rng.permutation(Xneg)
    #split each of these sets into training and testing sets
    num4test = round(len(Xpos) * prop)
    #choose the greater between the trial minimum and the calculated #
    num4test = max([trial_min,num4test])
    #use a random subset of the data for testing purposes
    posTest = Xpos[0:num4test,:]
    #use the rest of the data for training purposes
    posTrain = Xpos[num4test:,:]
    #repeat for the negative class
    num4test = round(Xneg.shape[0] * prop)
    num4test = max([trial_min,num4test])
    negTest = Xneg[0:num4test]
    negTrain = Xneg[num4test:]
    if balance:
        #apply sampling strategy to the training data
        posTrain,negTrain = overAndUnderSample(posTrain,negTrain)
        #apply balancing strategy to the test data
        posTest,negTest = overAndUnderSample(posTest, negTest)
    
    #randomly oversample with replacement
    Xtrain = oversample(posTrain,negTrain,nTrialsTrain)
    Xtest = oversample(posTest,negTest,nTrialsTest)
    return Xtrain,Xtest

def oversample(Xpos,Xneg,n_trials): 
    '''Prepare data for classification by oversampling to a specific # of trials
    INPUTS
    Xpos: matrix of neural data for the positive class (e.g. CS+). rows are
       trials. columns are neurons.
    Xneg: same as Xcsplus but for the negative class (e.g. CS-). The # of columns should be the
       same for these two inputs. The # rows should be very close to equal
    n_trials: # of trials to sample for each class in the final training set
    OUTPUTS
    Xtrain: data for both classes
    '''
    
    #draw indices with replacement
    indSample = rng.integers(len(Xpos),size=n_trials)
    #randomly oversample the rows
    posTrain = Xpos[indSample]
    indSample = rng.integers(len(Xneg),size=n_trials)
    negTrain = Xneg[indSample]
    #vertically concatenate the classes into a single training set
    return np.concatenate((posTrain,negTrain),axis=0)

def overAndUnderSample(Xpositive,Xnegative): 
    ''' Balance two sets of trials using random sampling.
    
    '''
    X = np.concatenate((Xpositive,Xnegative),axis=0)
    nPos = len(Xpositive)
    #create the corresponding boolean
    top=np.full((nPos,1),True)
    bottom=np.full((len(Xnegative),1),False)
    isPos = np.concatenate((top,bottom),axis=0)
    propPositive = nPos / len(X)
    #determine whether to resample based on the proportion of trials
    #in the positive class
    if propPositive < 0.4 or propPositive > 0.6:
        #num_neighbors=min([nPos-1,len(Xnegative)-1,5])
        #instantiate a sampling object
        #oversample to 80% the size of the majority label
        #over=over_sampling.SMOTE(sampling_strategy=0.8,k_neighbors=num_neighbors)
        over=over_sampling.RandomOverSampler(sampling_strategy=0.8)
        X_resampled,labels=over.fit_resample(X, isPos)
        #undersample the majority class to be equal
        under=under_sampling.RandomUnderSampler(sampling_strategy=1)
        X_resampled,labels=under.fit_resample(X_resampled, labels)
        #update the outputs
        Xpositive=X_resampled[labels,:]
        Xnegative=X_resampled[labels==False,:]
    return Xpositive,Xnegative

def overAndUnderSampleSK(Xpositive,Xnegative): 
    ''' Balance two sets of trials using random sampling. Outputs are easy to 
    use with sklearn functions
    OUTPUTS:
    X: trials x features
    labels: labels of trials
    '''
    X = np.concatenate((Xpositive,Xnegative),axis=0)
    nPos = len(Xpositive)
    #create the corresponding boolean
    top=np.full((nPos,),True)
    bottom=np.full((len(Xnegative),),False)
    isPos = np.concatenate((top,bottom),axis=0)
    propPositive = nPos / len(X)
    #determine whether to resample based on the proportion of trials
    #in the positive class
    if propPositive < 0.4 or propPositive > 0.6:
        #num_neighbors=min([nPos-1,len(Xnegative)-1,5])
        #instantiate a sampling object
        #oversample to 80% the size of the majority label
        #over=over_sampling.SMOTE(sampling_strategy=0.8,k_neighbors=num_neighbors)
        over=over_sampling.RandomOverSampler(sampling_strategy=0.8)
        X_resampled,labels=over.fit_resample(X, isPos)
        #undersample the majority class to be equal
        under=under_sampling.RandomUnderSampler(sampling_strategy=1)
        X_resampled,labels=under.fit_resample(X_resampled, labels)
        return X_resampled,labels
    else:
        return X,isPos
