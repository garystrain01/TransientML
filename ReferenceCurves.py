import random
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter('ignore')

import statsmodels.api as sm #  the stats module which contains state space modelling

bDebug = False     #  switch debug code on/off
bGenerateCSVFiles = False # to re-generate individual transient CSV files
bDisplayLightCurveSizing = False # used to display number of points per light curve (for reference only)
bDisplayLightCurvePoints = False # for display of test lightcurves
bRandom = False # generate random lightcurves for comparison
bDisplayTransientClasses = False # Display All Transient Classes as processed (only for information)
bOptimiseParameters = False # optimise hyperparameters
bTestReducedData = False # test the trade-off between accuracy and number of time points


DEFAULT_TRAINING_DATA_LOCATION = '/Users/garystrain/AAresearch/training/'
REF_LIGHTCURVE_LOCATION = '/Users/garystrain/AAresearch/TestLightCurves/'
REF_LIGHTCURVE_CSV_LOCATION = '/Users/garystrain/AAresearch/TestLightCurves/GeneratedFiles/'
REF_LIGHTCURVE_CLASSTYPE_NAME = '_class.txt'
REF_LIGHTCURVE_TEST_NAME = '_test.txt'
REF_LIGHTCURVE_FILENAME = 'transient_lc.txt'
REF_LIGHTCURVE_LABELS = 'lightcurveLabels.txt'
RANDOM_DATA_FILENAME = 'Random.txt'
DEFAULT_CLASS_NAME = 'Classification'
DEFAULT_TRAINING_CURVENAME = REF_LIGHTCURVE_CSV_LOCATION+'RefCurveSet_'
DEFAULT_FILENAME_EXT = '.jpg'
DEFAULT_TRANSIENTID_NAME = 'TransientID'
MJD_TO_JD_OFFSET = 2400000.5
DEFAULT_DATA_REDUCTION = 10 # no of time samples to be left for testing 'real-time'
DEFAULT_ALLOWED_TRANSIENTS = ['AGN','CV','Flare','HPM','Var','Blazar','SN']
# Defaults for CNN Model

DEFAULT_VERBOSE_LEVEL = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_NO_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.01

DEFAULT_LABEL1 = 0
DEFAULT_LABEL2 = 1

DEFAULT_BINARY_CLASS1 = 'SN'
DEFAULT_BINARY_CLASS2 = 'HPM'

TRAIN_TEST_RATIO = 0.70  # 70% of the total data should be training

SMALL_FONT_SIZE = 8
MEDIUM_FONT_SIZE = 10
BIGGER_FONT_SIZE = 12



FileHandleDict = {}  #use for storing file handles for all generated CSV files


def createLabelSet(label, numberOfSamples, sizeOfSample):
    a = np.empty((numberOfSamples, sizeOfSample), dtype=np.int8)
    a.fill(label)
    return a


def ExportTrainingCurves(lightCurveFlux,nameOfObject,numberOfLightCurves,sizeOfSample):

# take a lightcurve and save it to a CSV file

    f = open(DEFAULT_TRAINING_DATA_LOCATION+nameOfObject+'.txt','w')

    for lightCurve in range(numberOfLightCurves-1):
        for fluxData in range(sizeOfSample):

            element = str(lightCurveFlux[lightCurve][fluxData])
            if fluxData < (sizeOfSample - 1):
                element = element + ','
            f.write(element)
        f.write('\n')

    # special case for last one - drop the final ,

    for fluxData in range(sizeOfSample):
        element = str(lightCurveFlux[numberOfLightCurves-1][fluxData])
        if fluxData < (sizeOfSample-1):
            element = element+','
        f.write(element)

    f.close()

def GenerateRandomTrainingSet(numberTrainingCurves,numberSampleTimes,maxAmplitude):

    SetOfRandomLightCurves  = []


    print("generating no of random curves = ",numberTrainingCurves)

    for curve in range(numberTrainingCurves):
        strr = "Generating curve No "+str(curve+1)
        print(strr)
        RandomFluxCurve = np.zeros(numberSampleTimes)
        maxSize = random.random() * maxAmplitude

        timeIncrement = 4*np.pi/numberSampleTimes
        for t in range(numberSampleTimes):

            RandomFluxCurve[t] = np.sin(t)*(random.random() * maxSize)
            t += timeIncrement

        SetOfRandomLightCurves.append(RandomFluxCurve)


#    ExportTrainingCurves(SetOfRandomLightCurves, 'Random', numberTrainingCurves, numberSampleTimes)

    return SetOfRandomLightCurves





def ConvertToJulianDay(MJD):

    julianDay = MJD+MJD_TO_JD_OFFSET

    return julianDay

def StoreFileHandle(f,transientClassName):
#    print("storing file handle for ",transientClassName)

    if transientClassName not in FileHandleDict.keys():
        FileHandleDict[transientClassName] = f

def CreateTransientClassFile(transientClassName):

    fullFileName = REF_LIGHTCURVE_CSV_LOCATION+transientClassName+REF_LIGHTCURVE_CLASSTYPE_NAME
    # check if file already exists

    if transientClassName in FileHandleDict:
        f = FileHandleDict[transientClassName]
    else:

        f = open(fullFileName, 'w+')

        StoreFileHandle(f, transientClassName)


    return f

def load_file(filePath):
    dataframe = pd.read_csv(filePath,header=0)
    return dataframe

def loadCSVFile(filePath):
    print("*** Loading CSV File")
    dataframe = pd.read_csv(filePath,header=None)
    print("*** Completed Loading CSV File")
    return dataframe.values

def DisplayTransientNames(transientDict):

    listKeys = list(transientDict.keys())

    for entry in range(len(listKeys)):
        print("Transient Class: ",listKeys[entry])

def AddToDictionary(dict,newKey):

    if (bool(dict) == False):
        #dict is empty
        dict[newKey] = 0
    elif (newKey not in dict.keys()):
        maxValue = max(dict.values())
        dict[newKey] = maxValue+1

def StoreInTransientDictionary(dict,key,value):

    # this records the mapping of Transient ID's to transient types
    if (key not in dict):
       dict[key]= value
    else:
        print("*** Duplicate Transient ID *** ")

def ProcessTransientClasses():

    labels = load_file(REF_LIGHTCURVE_LOCATION + REF_LIGHTCURVE_LABELS)

    transientClasses = labels[DEFAULT_CLASS_NAME].values
    transientID = labels[DEFAULT_TRANSIENTID_NAME].values

    ClassificationDict = {}
    TransientIDDict = {}

    for i in range(len(transientClasses)):
        if ('/' in transientClasses[i]):

            transientClasses[i]= transientClasses[i].replace('/','_')

        AddToDictionary(ClassificationDict,transientClasses[i])

        StoreInTransientDictionary(TransientIDDict,transientID[i],transientClasses[i])

    return ClassificationDict,TransientIDDict

def CreateAllTransientFiles(ClassificationDict):
    # create a CSV file for each transient class

    transientNames = list(ClassificationDict.keys())

    for i in range(len(ClassificationDict)):
        print("Creating CSV File For Class = ",transientNames[i])
        CreateTransientClassFile(transientNames[i])

def CloseAllCSVFiles():

    print("*** Now Closing CSV Files ***")

    fileHandles = list(FileHandleDict.values())

    for f in fileHandles:

        f.close()

def StoreInCSVFile(transientID,transientClassName,obsID,magnitude,magnitudeError,MJD):

    # find file for this transient class

    if (transientClassName in FileHandleDict):

        # now write to this CVS file

        f = FileHandleDict[transientClassName]

        strr = str(transientID)+','+str(obsID)+','+str(magnitude)+','+str(magnitudeError)+','+str(MJD)

        f.write(strr)
        f.write('\n')
    else:
        print("*** Error - No CSV File For This Transient Class ***")

def StoreAllObservations(transientIDDict,lightCurves):
    # for each lightcurve, store the set of observations for a specific transient type in
    # the correct CSV file

    numberObservations = len(lightCurves['ID'])
    transientID = lightCurves['ID']
    obsID = lightCurves['observation_id']
    Mag = lightCurves['Mag']
    Magerr = lightCurves['Magerr']
    MJD = lightCurves['MJD']

    print("***Storing Observations in CSV Files ***")
    iTotalCount = 0
    iCount = 0
    for obs in range(numberObservations):
        # for each observation, store in relevant CSV file

        transientID[obs] = int(transientID[obs].replace('TranID',''))
        transientClassName = transientIDDict[transientID[obs]]

        StoreInCSVFile(transientID[obs],transientClassName,obsID[obs],Mag[obs],Magerr[obs],MJD[obs])
        iCount += 1

        if (iCount == 1000):
            iTotalCount = iTotalCount+iCount
            iCount = 0
            strr = 'Completed '+str(iTotalCount)+' Observations'
            print(strr)
    strr = 'Final Count = ' + str(iTotalCount) + ' Observations'
    print(strr)


def DisplayLightCurveIndices(transientName,X,y):


    plt.bar(X,y)
    strr = 'No Points in Lightcurve ('+transientName+') by LightCurve Index'
    plt.title(strr)


    plt.show()

def DisplayLightCurvePoints(transientName,lightCurveNumber,times,y):


    plt.scatter(times, y, marker='+')

    strr = 'Lightcurve No '+str(lightCurveNumber)+'('+transientName+') '
    plt.title(strr)
    plt.show()

def SetPlotParameters():

    plt.rc('axes',labelsize=SMALL_FONT_SIZE)
    plt.rc('xtick',labelsize=SMALL_FONT_SIZE)
    plt.rc('ytick',labelsize=SMALL_FONT_SIZE)

def DisplaySelectionLightCurves(transientName,setNumber,times,finalTrainingSet):


    DEFAULT_DISPLAY_CURVES = 3
    fig, axs = plt.subplots(DEFAULT_DISPLAY_CURVES)

    for curve in range(0, DEFAULT_DISPLAY_CURVES):
        randomCurveNo = int(random.random() * len(finalTrainingSet))
        axs[curve].scatter(times, finalTrainingSet[randomCurveNo], marker='+')

        axs[curve].tick_params(axis='x',labelsize=SMALL_FONT_SIZE)
        axs[curve].set_xlabel('Days',fontsize=SMALL_FONT_SIZE)
        axs[curve].set_ylabel('Mag')
        strr = 'Lightcurve No '+str(randomCurveNo)+'('+transientName+') '

    plt.show()

    fig_filename = DEFAULT_TRAINING_CURVENAME + str(setNumber)+DEFAULT_FILENAME_EXT
    fig.savefig(fig_filename)


def CreateTransientTrainingData(transientName,TotalLightCurves):

    # create a suitable np array

    numberObjects = len(TotalLightCurves)
    strr = 'Processing '+str(numberObjects)+' lightcurves of '+transientName
    print(strr)

    completeTrainingData = []

    # now populate the training data

    NumberPointsInCurve = []
    LightCurveNumber = []

    lightCurveIndex = 0
    maxTimeDelta = 0
    maxSize = 0

    for lc in range(numberObjects):

        lightCurve = TotalLightCurves[lc]
        numberPoints = len(lightCurve)

        #store this data for display only
        NumberPointsInCurve.append(numberPoints)
        LightCurveNumber.append(lightCurveIndex)
        lightCurveIndex += 1

        timeSeriesData = []
        magSeriesData = []
        lightCurveTraining = []
        if (bDebug):
            print("number points in this light curve = ",numberPoints)

        for point in range(numberPoints):
            pointData = lightCurve[point]
            if (point ==0):
                # lets get the start time for this sample

                startSampleTime = ConvertToJulianDay(pointData[3])

            if (bDebug):

                print("transientID = ", pointData[0])
                print("magnitude = ", pointData[1])
                print("magnitude error = ", pointData[2])
                print("MJD= ", pointData[3])

            magnitudeValue = pointData[1]

            thisTimeValue = ConvertToJulianDay(pointData[3])
            thisTimeDelta = round(thisTimeValue-startSampleTime)

            if (thisTimeDelta <0):
                # make it zero
                thisTimeDelta = 0.0
            if (thisTimeDelta > maxTimeDelta):
                maxTimeDelta = thisTimeDelta
            if (magnitudeValue > maxSize):
                maxSize = magnitudeValue

            timeSeriesData.append(thisTimeDelta)
            magSeriesData.append(magnitudeValue)

        lightCurveTraining.append(timeSeriesData)
        lightCurveTraining.append(magSeriesData)
        completeTrainingData.append(lightCurveTraining)

    if (bDisplayLightCurveSizing):

        DisplayLightCurveIndices(transientName,LightCurveNumber,NumberPointsInCurve)

    return completeTrainingData, round(maxTimeDelta), maxSize #this is a linked list of 2 lists of time series and mag series data

def DisplayLightCurveEntries(trainingDataSet):
    listOfData = []
    listOfEntries = []

    for entry in range(len(trainingDataSet)):
        if (trainingDataSet[entry] != 0):
            listOfEntries.append(entry)
            listOfData.append(trainingDataSet[entry])

    for entry in range(len(listOfEntries)):
        strr = '('+str(listOfEntries[entry])+','+str(listOfData[entry])+')'
        print(strr)


def StandardiseDataSet(trainingData,maxTimeDelta,largestTimeDelta):

    # transform raw dataset into consistent length dataset and use interpolation for
    # missing values

    if (largestTimeDelta > maxTimeDelta):
        maxTime = int(largestTimeDelta)
    else:
        maxTime = int(maxTimeDelta)

    newTrainingSet = np.zeros((maxTime+1))

    timeSeriesData = trainingData[0]
    magSeriesData = trainingData[1]

    maxTimeDelta = round(max(timeSeriesData))

    if (bDebug):
        print("max time delta ", maxTimeDelta)
        print("maxTime = ", maxTime)
        print("no of entries in timeseries = ",len(timeSeriesData))
        print("no of entries in magseries = ", len(magSeriesData))
        print("max time delta in this series = ",maxTimeDelta)

        print(timeSeriesData)
        print(magSeriesData)

    # the training data is in 2 lists - one for the original time series and one for
    # the magnitude time series

    for entry in range(len(timeSeriesData)):
        if (bDebug):
            print("entry = ",entry)
            print("time value =",int(timeSeriesData[entry]))
            print("mag value =",magSeriesData[entry])

        if (newTrainingSet[int(timeSeriesData[entry])] != 0):
            prevSeriesData = newTrainingSet[int(timeSeriesData[entry])]
            #now average out any duplicates
            newTrainingSet[int(timeSeriesData[entry])] = (magSeriesData[entry]+prevSeriesData)/2
        else:
           newTrainingSet[int(timeSeriesData[entry])] = magSeriesData[entry]

    if (bDebug):
        DisplayLightCurveEntries(newTrainingSet)

    return newTrainingSet

def ProcessClass(transientClassName):

    fullFileName = REF_LIGHTCURVE_CSV_LOCATION + transientClassName + REF_LIGHTCURVE_CLASSTYPE_NAME
    print("Loading...", fullFileName)

    trainingTransientData = loadCSVFile(fullFileName)

    numberObservations = len(trainingTransientData)

    print("No Observations Loaded = ",numberObservations)

    currentTransientID = 0
    numberTransients = 0

    TotalLightCurves = []  #this will contain a list of individual light curves
    TotalTransientList = [] # list of all processed transients

    for i in range(numberObservations):

        obs = trainingTransientData[i]
        transientID = obs[0]
        obsID = obs[1]
        magnitude = obs[2]
        magnitudeError = obs[3]
        MJD = obs[4]

        if (currentTransientID == 0):
            # first time through

            numberTransients += 1
            currentTransientID = transientID
            TotalLightCurveEntry = []
            TotalTransientList.append(transientID)

        lightCurvePoint = []   # this contains one point on a specific light curve

        lightCurvePoint.append(transientID)
        lightCurvePoint.append(magnitude)
        lightCurvePoint.append(magnitudeError)
        lightCurvePoint.append(MJD)

        if (transientID != currentTransientID):
            # new transient set of observations
            if (bDebug):
                print("new transient ID = ",transientID)
            numberTransients += 1
            currentTransientID = transientID

            TotalLightCurves.append(TotalLightCurveEntry)
            # create a new list
            TotalLightCurveEntry = []
            TotalTransientList.append(transientID)

        TotalLightCurveEntry.append(lightCurvePoint)

    TotalLightCurves.append(TotalLightCurveEntry)

    trainingDataSet, maxTimeDelta, maxSize = CreateTransientTrainingData(transientClassName,TotalLightCurves)

    return trainingDataSet, maxTimeDelta, maxSize


def evaluateCNNModel(Xtrain,ytrain,Xtest,ytest,n_timesteps,n_features,n_outputs,epochs):

    verbose, batch_size = DEFAULT_VERBOSE_LEVEL,DEFAULT_BATCH_SIZE

    if (bDebug):
        print("input train shape = ", Xtrain.shape)
        print("label train shape = ", ytrain.shape)
        print("input test shape = ", Xtest.shape)
        print("label test shape = ", ytest.shape)
        print("n_timesteps = ",n_timesteps)
        print("n_features = ", n_features)
        print("n_outputs = ", n_outputs)
        print("epochs = ",epochs)
        print("batch size =",batch_size)


    model=keras.Sequential()
    model.add(tf.keras.layers.Conv1D(64,DEFAULT_KERNEL_SIZE,activation='relu',input_shape=(n_timesteps,n_features)))
    model.add(tf.keras.layers.Conv1D(64,DEFAULT_KERNEL_SIZE,activation='relu'))

    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(100,activation='relu'))

    model.add(tf.keras.layers.Dense(n_outputs,activation='sigmoid'))


    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

    model.fit(Xtrain,ytrain,epochs=epochs,batch_size=batch_size,verbose=verbose)

    model.summary()
    _,accuracy = model.evaluate(Xtest,ytest,batch_size=batch_size,verbose=verbose)

    return accuracy,model

def ScaleInputData(X):

    scaler = MinMaxScaler()
    scaler.fit(X)
    normalised = scaler.transform(X)

    return normalised

def StandardLightCurves(setNumber,trainingDataSet,thisMaxTimeDelta,maxTimeDelta):

    finalTrainingSet = []

    if (bDebug):
        print("len of training set = ", len(trainingDataSet))
        print("Standardising Set Number ", setNumber)
    for curve in range(len(trainingDataSet)):
        standardTrainingData = StandardiseDataSet(trainingDataSet[curve],thisMaxTimeDelta,maxTimeDelta)
        finalTrainingSet.append(standardTrainingData)

    return finalTrainingSet

def DisplayNonZeroPoints(text,dataSet):

    print(text)
    numberOfCurves = len(dataSet)

    AverageNoNonZeroPoints = []

    for curve in range(numberOfCurves):
        numberNonZeroPoints = 0
        lightCurve = dataSet[curve]

        for point in range(len(lightCurve)):

            if (lightCurve[point] != 0):
                numberNonZeroPoints += 1

        AverageNoNonZeroPoints.append(numberNonZeroPoints/len(lightCurve))

    print("Final % Average number non zero points =",(sum(AverageNoNonZeroPoints)/numberOfCurves)*100)
    print("For Number of curves = ",numberOfCurves)





def testCNNModel(n_epochs,learning_rate,Xtrain,ytrain,Xtest,ytest):


    n_timesteps = Xtrain.shape[1]
    n_features = Xtrain.shape[2]

    n_outputs = 1

    verbose, batch_size = DEFAULT_VERBOSE_LEVEL,DEFAULT_BATCH_SIZE

    model=keras.Sequential()
    model.add(tf.keras.layers.Conv1D(64,DEFAULT_KERNEL_SIZE,activation='relu',input_shape=(n_timesteps,n_features)))
    model.add(tf.keras.layers.Conv1D(64,DEFAULT_KERNEL_SIZE,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    model.add(tf.keras.layers.Dense(n_outputs,activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

    model.fit(Xtrain,ytrain,epochs=n_epochs,batch_size=batch_size,verbose=verbose)
    _,accuracy = model.evaluate(Xtest,ytest,batch_size=batch_size,verbose=verbose)

    return accuracy,model

def DisplayHyperTable(Accuracy,Epochs,LearningRates):

    from astropy.table import QTable, Table, Column

    t = Table([Accuracy,Epochs,LearningRates],names=('Accuracy','Epochs','Learning Rate'))
    print(t)

def OptimiseCNNHyperparameters(Xtrain, ytrain, Xtest, ytest):
    numberEpochs = [1, 10, 100]
    learningRateSchedule = [0.1, 0.01, 0.001]

    ExperimentAccuracy = []
    ExperimentEpochs = []
    ExperimentLearningRates = []

    for epoch in numberEpochs:
        for learning_rate in learningRateSchedule:

            print("Testing Number Epochs = ", epoch)
            print("Testing Learning Rate = ", learning_rate)

            accuracy, model = testCNNModel(epoch, learning_rate, Xtrain, ytrain, Xtest, ytest)
            ExperimentAccuracy.append(accuracy)
            ExperimentEpochs.append(epoch)
            ExperimentLearningRates.append(learning_rate)

            print("Accuracy = ", accuracy)

    return ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates

def DisplayReducedDataTable(numberDataPoints,totalData,trueData,falseData,accuracyData):


    from astropy.table import Table

    t = Table([numberDataPoints,totalData,trueData,falseData,accuracyData],names=('No Data points','Total','No True','No False','Accuracy'))
    print(t)


def CheckForAccuracy(dataReduced,ypred,ytest):

    iTrue = 0
    iFalse = 0
    iTotal = 0
    numberZeros = 0

    for i in range(len(ytest)):
        iTotal += 1
        ypredicted = np.round(ypred[i])  # we're dealing with probabilities, so anything >0.5 becomes 1
        yoriginal = np.round(ytest[i])

        if (ytest[i]==0):
            numberZeros += 1
        if (ypredicted == yoriginal):
            iTrue += 1
        else:
            iFalse += 1

    print("Total No = ",iTotal)
    print("Total No Zero Data =", numberZeros)
    print("Total Correct =", iTrue)
    print("Total Incorrect = ",iFalse)
    accuracy = np.round((iTrue/iTotal),2)
    strr = "For Data Points Reduced to "+str(dataReduced)+", New Accuracy = "+str(accuracy)
    print(strr)
    return dataReduced,iTotal, iTrue, iFalse, accuracy

def RemoveData(lightCurve,maxLength,nullStart):

    for i in range(maxLength-nullStart):
        lightCurve[0][i + nullStart][0] = 0

    return lightCurve

def RemoveDataFromTest(Xtest,maxLength,nullStart):


    for curve in range(len(Xtest)):
        for i in range(maxLength-nullStart):
            Xtest[curve][i + nullStart][0] = 0

    return Xtest



def TestDatasets(totalSampleSize,model,Xtest, ytest):

    numberDataPoints = []
    totalResult= []
    trueResult = []
    falseResult = []
    accuracyResult = []

    print("max length of curve = ",totalSampleSize)

    #start half way through the dataset and see the effect on the accuracy

    nullStart = int(totalSampleSize/2)
    Xtest = RemoveDataFromTest(Xtest, totalSampleSize, nullStart)
    ypred = model.predict(Xtest)
    dataReduced,iTotal, iTrue, iFalse, accuracy = CheckForAccuracy(nullStart, ypred, ytest)
    numberDataPoints.append(totalSampleSize-dataReduced)
    totalResult.append(iTotal)
    trueResult.append(iTrue)
    falseResult.append(iFalse)
    accuracyResult.append(accuracy)


    for i in range(nullStart-2):
        Xtest = RemoveDataFromTest(Xtest,totalSampleSize,nullStart-i)
        ypred = model.predict(Xtest)
        dataReduced,iTotal, iTrue, iFalse, accuracy = CheckForAccuracy(nullStart+i,ypred, ytest)
        numberDataPoints.append(totalSampleSize-dataReduced)
        totalResult.append(iTotal)
        trueResult.append(iTrue)
        falseResult.append(iFalse)
        accuracyResult.append(accuracy)

    return  numberDataPoints, totalResult,trueResult, falseResult, accuracyResult

def CheckTransients(tran1,tran2):
    bPassTest=True

    if (tran1 == tran2):
        print("Transient Cannot be Identical")
        bPassTest= False
    elif (tran1 not in DEFAULT_ALLOWED_TRANSIENTS):
        print("Unknown Transient ")
        bPassTest=False
    elif (tran1 not in DEFAULT_ALLOWED_TRANSIENTS):
        print("Unknown Transient ")
        bPassTest=False

    return bPassTest

def SelectTransients():
    bInput=False

    while (bInput==False):

        transient1 = input("Select 1st Transient From "+str(DEFAULT_ALLOWED_TRANSIENTS)+" :")
        transient2 = input("Select 1st Transient From "+str(DEFAULT_ALLOWED_TRANSIENTS)+" :")
        if (CheckTransients(transient1,transient2)):
            bInput=True

    return transient1,transient2

def main():


    DEFAULT_BINARY_CLASS1,DEFAULT_BINARY_CLASS2 = SelectTransients()

    print("Processing "+DEFAULT_BINARY_CLASS1+" versus "+DEFAULT_BINARY_CLASS2)

    lightCurves = load_file(REF_LIGHTCURVE_LOCATION+REF_LIGHTCURVE_FILENAME)

    if (bGenerateCSVFiles):

        print("Generating Individual CSV Files For Transients...")

        ClassificationDict, TransientIDDict = ProcessTransientClasses()
        if (bDisplayTransientClasses):
            DisplayTransientNames(ClassificationDict)

        CreateAllTransientFiles(ClassificationDict)

        StoreAllObservations(TransientIDDict,lightCurves)

        CloseAllCSVFiles()


    # now process each a CSV file to create a training and test set of data


    trainingDataSet1, maxTimeDelta1,maxAmplitude = ProcessClass(DEFAULT_BINARY_CLASS1)
    trainingDataSet2, maxTimeDelta2, maxAmplitude = ProcessClass(DEFAULT_BINARY_CLASS2)

    if (maxTimeDelta1 > maxTimeDelta2):
        maxTimeDelta = maxTimeDelta1
    else:
        maxTimeDelta = maxTimeDelta2

    numberTrainingCurves1 = len(trainingDataSet1)
    numberTrainingCurves2 = len(trainingDataSet2)

    finalTrainingSet1 = StandardLightCurves(DEFAULT_LABEL1+1,trainingDataSet1,maxTimeDelta1,maxTimeDelta)
    finalTrainingSet2 = StandardLightCurves(DEFAULT_LABEL2+1,trainingDataSet2,maxTimeDelta2,maxTimeDelta)

    times = np.arange(0,maxTimeDelta+1)

    if (bDisplayLightCurvePoints):
        DisplaySelectionLightCurves(DEFAULT_BINARY_CLASS1,DEFAULT_LABEL1+1,times,finalTrainingSet1)
        DisplaySelectionLightCurves(DEFAULT_BINARY_CLASS2,DEFAULT_LABEL2+1,times,finalTrainingSet2)

    # scale all data to be between 0-1

    XData1 = ScaleInputData(finalTrainingSet1)
    XData2 = ScaleInputData(finalTrainingSet2)

    print("Shape of Two Datasets ...")
    print(XData1.shape)
    print(XData2.shape)

    numberInTrainSet1 = int(round(XData1.shape[0]*TRAIN_TEST_RATIO))
    numberInTrainSet2 = int(round(XData2.shape[0]*TRAIN_TEST_RATIO))

    Xtrain1 = XData1[:numberInTrainSet1,:]
    Xtrain2 = XData2[:numberInTrainSet2,:]

    Xtest1 = XData1[numberInTrainSet1:,:]
    Xtest2 = XData2[numberInTrainSet2:,:]

    completeXtrain = np.concatenate((Xtrain1,Xtrain2))
    completeXtest = np.concatenate((Xtest1,Xtest2))


    #now create the train label data

    if (bDebug):
        print(" number transient1 training light curves = ",len(Xtrain1))
        print(" number transient 2 training light curves = ",len(Xtrain2))

        print("complete training light curves = ",len(completeXtrain))

        print(" number transient1 test light curves = ",len(Xtest1))
        print(" number transient 2 test light curves = ",len(Xtest2))

        print("complete test light curves = ",len(completeXtest))


    train1_y = createLabelSet(DEFAULT_LABEL1,numberInTrainSet1,1)
    test1_y = createLabelSet(DEFAULT_LABEL1,(numberTrainingCurves1-numberInTrainSet1),1)

    train2_y = createLabelSet(DEFAULT_LABEL2,numberInTrainSet2,1)
    test2_y = createLabelSet(DEFAULT_LABEL2,(numberTrainingCurves2-numberInTrainSet2),1)

    complete_ytrain = np.concatenate((train1_y,train2_y))
    complete_ytest = np.concatenate((test1_y,test2_y))


    # create an integrated set of training data which includes the transient and the random data
    # ensure that the sequence numbers are kept to manage the label data

    if (bDebug):
        print("size of complete y train = ", len(complete_ytrain))
        print("size of complete y test = ", len(complete_ytest))

        print("size of complete x train = ", len(completeXtrain))
        print("size of complete x test = ", len(completeXtest))


    index = np.random.choice(completeXtrain.shape[0],len(completeXtrain),replace=False)

    Xtrain = completeXtrain[index]
    ytrain = complete_ytrain[index]

    index = np.random.choice(completeXtest.shape[0],len(completeXtest),replace=False)

    Xtest = completeXtest[index]
    ytest = complete_ytest[index]

    if (bDebug):
        print("Final Training Data = ",Xtrain.shape)
        print("Final Test Data = ",Xtest.shape)

        print("Final Training Label Data = ",ytrain.shape)
        print("Final Test Label Data = ",ytest.shape)


    Xtrain = np.reshape(Xtrain,(Xtrain.shape[0],Xtrain.shape[1],1))
    Xtest = np.reshape(Xtest,(Xtest.shape[0],Xtest.shape[1],1))

    if (bDebug):
        print("shape of Xtrain =",Xtrain.shape)
        print("shape of Xtest =",Xtest.shape)
        print("shape of ytrain =",ytrain.shape)
        print("shape of ytest =",ytest.shape)

    n_timesteps = Xtrain.shape[1]
    n_features = Xtrain.shape[2]
    n_outputs = 1 # for binary classification

    if (bDebug):
        print(n_timesteps,n_features,n_outputs)

    DisplayNonZeroPoints("For Training Data..",Xtrain)
    DisplayNonZeroPoints("For Test Data..",Xtest)

    Accuracy, CNNModel = evaluateCNNModel(Xtrain,ytrain,Xtest,ytest,n_timesteps,n_features,n_outputs,DEFAULT_NO_EPOCHS)
    print("CNN Accuracy = ",Accuracy)


    if (bTestReducedData==True):

        Xtest = RemoveDataFromTest(Xtest,n_timesteps,DEFAULT_DATA_REDUCTION)
        ypred = CNNModel.predict(Xtest)
        CheckForAccuracy(DEFAULT_DATA_REDUCTION, ypred, ytest)

    if (bOptimiseParameters == True):
        ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates = OptimiseCNNHyperparameters(Xtrain, ytrain, Xtest,ytest)
        DisplayHyperTable(ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates)

    if (bTestReducedData == True):
        numberDataPoints, totalData, trueData, falseData, accuracyData = TestDatasets(n_timesteps, CNNModel, Xtest,ytest)
        DisplayReducedDataTable(numberDataPoints, totalData, trueData, falseData, accuracyData)



if __name__== '__main__':
    main()

