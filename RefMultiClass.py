import random
import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
from support import ProcessResultsFile


warnings.simplefilter('ignore')

import statsmodels.api as sm  # the stats module which contains state space modelling

bDebug = False  # switch debug code on/off
bGenerateCSVFiles = False  # to re-generate individual transient CSV files
bDisplayLightCurveSizing = False  # used to display number of points per light curve (for reference only)
bDisplayLightCurvePoints = False  # for display of test lightcurves
bRandom = False  # generate random lightcurves for comparison
bDisplayTransientClasses = False  # Display All Transient Classes as processed (only for information)
bOptimiseParameters = False  # optimise hyperparameters
bTestReducedData = False  # test the trade-off between accuracy and number of time points
bCalcAveragesFromFile = False # used to calulate averages from results file
bInspectObservations = False # used to inspect the observation CSV files
bInspectLightCurveLength = False # switch this on if inspecting light curves and want to see all of their raw lengths (bInspectObservations must be True also)
bDetailedLowPoints = False # switch this on if you want detail for low points (bInspectObservations must be True also)
bRecordLightCurves = True # record number observations held in corresponding CSV file

DEFAULT_TRAINING_DATA_LOCATION = '/Users/garystrain/AAresearch/training/'
REF_LIGHTCURVE_LOCATION = '/Users/garystrain/AAresearch/TestLightCurves/'
REF_LIGHTCURVE_CSV_LOCATION = '/Users/garystrain/AAresearch/TestLightCurves/GeneratedFiles/'
DEFAULT_RESULTS_FILE = 'results'
REF_LIGHTCURVE_CLASSTYPE_NAME = '_class.txt'
REF_LIGHTCURVE_TEST_NAME = '_test.txt'
REF_LIGHTCURVE_FILENAME = 'transient_lc.txt'
REF_LIGHTCURVE_LABELS = 'lightcurveLabels.txt'
RANDOM_DATA_FILENAME = 'Random.txt'
OBSERVATION_DATA_FILENAME = 'Observations'
DEFAULT_CLASS_NAME = 'Classification'
DEFAULT_TRAINING_CURVENAME = REF_LIGHTCURVE_CSV_LOCATION + 'RefCurveSet_'
DEFAULT_FILENAME_EXT = '.jpg'
DEFAULT_TRANSIENTID_NAME = 'TransientID'
MJD_TO_JD_OFFSET = 2400000.5
DEFAULT_DATA_REDUCTION = 10  # no of time samples to be left for testing 'real-time'
DEFAULT_MULTIPLE_CHOICE = 'M'
DEFAULT_ALL_CHOICE = '*'
FAILED_RETURN = -1
MIN_LIMIT_OBSERVATIONS = 5 # ignore all CSV files with less than 5 observations

DEFAULT_TEST_SET1_ALLOWED_TRANSIENTS \
    = ['AGN']


DEFAULT_TEST_SET_ALLOWED_TRANSIENTS \
    = ['AGN',
       'CV']

DEFAULT_SMALL_SET_ALLOWED_TRANSIENTS \
    = ['AGN',
       'CV',
       'FLARE',
       'HPM',
       'VAR',
       'BLAZAR',
       'SN']

DEFAULT_MEDIUM_SET_ALLOWED_TRANSIENTS \
    = ['AGN',
       'CV',
       'FLARE',
       'HPM',
       'VAR',
       'BLAZAR',
       'SN',
       'AGN?',
       'SN?',
       'MIRA',
       'YSO',
       'BLAZAR?',
       'CV?',
       'LPV']

DEFAULT_FULL_SET_ALLOWED_TRANSIENTS \
    = ['AGN',
       'CV',
       'FLARE',
       'HPM',
       'Var',
       'BLAZAR',
       'SN',
       'YSO',
       'CV_SN',
       'SN?',
       'CV?',
       'LPV',
       'SDSS',
       'AGN?',
       'MERGER_CV?',
       'NOVA?',
       'BLAZAR?',
       'AST?',
       'SN_CV',
       'COMET',
       'QSO',
       'SN_AST',
       'NOVA',
       'MIRA',
       'VAR',
       'LHS_5157',
       'YSO?',
       'SN_AGN',
       'BLAZAR_AGN',
       'FLARE?',
       'RRL',
       'RCORB',
       'HPM?',
       'AGN_SN',
       'GRB',
       'CV_AGN',
       'SN_AGN?',
       'AST',
       'FLARE_SN?',
       'FLARE_SN',
       'AGN_CV',
       'CV_VAR',
       'OH_IR',
       'RRLYRAE',
       'CV_AGN?',
       'CV_AST',
       'VAR_SN',
       'CARBON',
       'SN_AST?',
       'AGN_BLAZAR',
       'CV_BLAZAR',
       'AGN_SN?',
       'BLAZAR_SN',
       'NOVA_CV',
       'SN_VAR',
       'VAR_AST?',
       'CV_VAR?',
       'AGN_VAR',
       'AST_SN',
       'VAR?',
       'VAR_SN?',
       'AMCVN?',
       'FU',
       'VAR_AGN',
       'AST_VAR',
       'AGN_VAR?',
       'CARB',
       'TDE?',
       'UVES',
       'AST_FLARE',
       'SN_CV?',
       'CV_FLARE',
       'VAR_FLARE?',
       'CV_AST?',
       'COMET_AST?',
       'O_NE',
       'HPM_VAR?',
       'AGN_FLARE?',
       'NOTHING_LENSING',
       'CV_VAR_AST',
       'MASER',
       'AST_FLARE?',
       'CV_SN?',
       'AGN_NOTHING?',
       'SN_TDE',
       'OH_IR',
       'VAR_NOTHING?',
       'AST_CV?',
       'VERY',
       'RED',
       'SN_NOTHING?',
       'VAR_NOTHING',
       'VARIABLE',
       'SN_TDE?',
       'AST_VAR??',
       'AST_VAR?',
       'SN_VAR?',
       'AST_SN?',
       'VAR_NOVA',
       'UNKNOWN']

# Defaults for CNN Model

DEFAULT_VERBOSE_LEVEL = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_NO_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT_RATE = 0.30
DEFAULT_NO_NEURONS = 100

TRAIN_TEST_RATIO = 0.80  # ratio of the total data for training

SMALL_FONT_SIZE = 8
MEDIUM_FONT_SIZE = 10
BIGGER_FONT_SIZE = 12

FileHandleDict = {}  # use for storing file handles for all generated CSV files



def assignLabelSet(label, numberOfSamples):

    shape = (numberOfSamples, len(label))

    a = np.empty((shape))

    a[:]=label

    return a

def createOneHotEncodedSet(listOfLabels):

    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.preprocessing import OneHotEncoder

    # use label to get one hot encoded value

    ordinalEncoder = OrdinalEncoder()
    oneHotEncoder = OneHotEncoder()

    listOfLabels = np.array(listOfLabels)
    listOfLabels= listOfLabels.reshape(-1,1)
    integerEncoded = ordinalEncoder.fit_transform(listOfLabels)

    oneHotEncoded = oneHotEncoder.fit_transform(integerEncoded)
    oneHotEncoded = oneHotEncoded.toarray()

    return oneHotEncoded



def ExportTrainingCurves(lightCurveFlux, nameOfObject, numberOfLightCurves, sizeOfSample):
    # take a lightcurve and save it to a CSV file

    f = open(DEFAULT_TRAINING_DATA_LOCATION + nameOfObject + '.txt', 'w')

    for lightCurve in range(numberOfLightCurves - 1):
        for fluxData in range(sizeOfSample):

            element = str(lightCurveFlux[lightCurve][fluxData])
            if fluxData < (sizeOfSample - 1):
                element = element + ','
            f.write(element)
        f.write('\n')

    # special case for last one - drop the final ,

    for fluxData in range(sizeOfSample):
        element = str(lightCurveFlux[numberOfLightCurves - 1][fluxData])
        if fluxData < (sizeOfSample - 1):
            element = element + ','
        f.write(element)

    f.close()


def GenerateRandomTrainingSet(numberTrainingCurves, numberSampleTimes, maxAmplitude):
    SetOfRandomLightCurves = []

    print("generating no of random curves = ", numberTrainingCurves)

    for curve in range(numberTrainingCurves):
        strr = "Generating curve No " + str(curve + 1)
        print(strr)
        RandomFluxCurve = np.zeros(numberSampleTimes)
        maxSize = random.random() * maxAmplitude

        timeIncrement = 4 * np.pi / numberSampleTimes
        for t in range(numberSampleTimes):
            RandomFluxCurve[t] = np.sin(t) * (random.random() * maxSize)
            t += timeIncrement

        SetOfRandomLightCurves.append(RandomFluxCurve)

    #    ExportTrainingCurves(SetOfRandomLightCurves, 'Random', numberTrainingCurves, numberSampleTimes)

    return SetOfRandomLightCurves


def ConvertToJulianDay(MJD):
    julianDay = MJD + MJD_TO_JD_OFFSET

    return julianDay


def StoreFileHandle(f, transientClassName):
    #    print("storing file handle for ",transientClassName)

    if transientClassName not in FileHandleDict.keys():
        FileHandleDict[transientClassName] = f


def CreateTransientClassFile(transientClassName):
    fullFileName = REF_LIGHTCURVE_CSV_LOCATION + transientClassName + REF_LIGHTCURVE_CLASSTYPE_NAME
    # check if file already exists

    if transientClassName in FileHandleDict:
        f = FileHandleDict[transientClassName]
    else:

        f = open(fullFileName, 'w+')

        StoreFileHandle(f, transientClassName)

    return f


def load_file(filePath):
    dataframe = pd.read_csv(filePath, header=0)
    return dataframe


def loadCSVFile(filePath):

    bDataValid = True

    if (os.path.isfile(filePath)):
        print("*** Loading CSV File")
        dataframe = pd.read_csv(filePath, header=None)
        dataReturn = dataframe.values
        print("*** Completed Loading CSV File")
    else:
        print("*** CSV File Does Not Exist ***")
        bDataValid = False
        dataReturn = []

    return bDataValid,dataReturn



def DisplayTransientNames(transientDict):
    listKeys = list(transientDict.keys())

    for entry in range(len(listKeys)):
        print("Transient Class: ", listKeys[entry])


def AddToDictionary(dict, newKey):
    if (bool(dict) == False):
        # dict is empty
        dict[newKey] = 0
    elif (newKey not in dict.keys()):
        maxValue = max(dict.values())
        dict[newKey] = maxValue + 1


def StoreInTransientDictionary(dict, key, value):
    # this records the mapping of Transient ID's to transient types
    if (key not in dict):
        dict[key] = value
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
            transientClasses[i] = transientClasses[i].replace('/', '_')
        transientClasses[i] = transientClasses[i].upper()

        AddToDictionary(ClassificationDict, transientClasses[i])

        StoreInTransientDictionary(TransientIDDict, transientID[i], transientClasses[i])

    return ClassificationDict, TransientIDDict


def CreateAllTransientFiles(ClassificationDict):
    # create a CSV file for each transient class

    transientNames = list(ClassificationDict.keys())

    for i in range(len(ClassificationDict)):
        transientNames[i] = transientNames[i].upper()
        print("Creating CSV File For Class = ", transientNames[i])
        CreateTransientClassFile(transientNames[i])


def CloseAllCSVFiles():
    print("*** Now Closing CSV Files ***")

    fileHandles = list(FileHandleDict.values())

    for f in fileHandles:
        f.close()


def StoreInCSVFile(transientID, transientClassName, obsID, magnitude, magnitudeError, MJD):
    # find file for this transient class

    if (transientClassName in FileHandleDict):

        # now write to this CVS file

        f = FileHandleDict[transientClassName]

        strr = str(transientID) + ',' + str(obsID) + ',' + str(magnitude) + ',' + str(magnitudeError) + ',' + str(MJD)

        f.write(strr)
        f.write('\n')
    else:
        print("*** Error - No CSV File For This Transient Class ***")


def StoreAllObservations(transientIDDict, lightCurves):
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

        transientID[obs] = int(transientID[obs].replace('TranID', ''))
        transientClassName = transientIDDict[transientID[obs]]

        StoreInCSVFile(transientID[obs], transientClassName, obsID[obs], Mag[obs], Magerr[obs], MJD[obs])
        iCount += 1

        if (iCount == 1000):
            iTotalCount = iTotalCount + iCount
            iCount = 0
            strr = 'Completed ' + str(iTotalCount) + ' Observations'
            print(strr)
    strr = 'Final Count = ' + str(iTotalCount) + ' Observations'
    print(strr)


def DisplayLightCurveIndices(transientName, X, y):
    plt.bar(X, y)
    strr = 'No Points in Lightcurve (' + transientName + ') by LightCurve Index'
    plt.title(strr)

    plt.show()


def DisplayLightCurvePoints(transientName, lightCurveNumber, times, y):
    plt.scatter(times, y, marker='+')

    strr = 'Lightcurve No ' + str(lightCurveNumber) + '(' + transientName + ') '
    plt.title(strr)
    plt.show()


def SetPlotParameters():
    plt.rc('axes', labelsize=SMALL_FONT_SIZE)
    plt.rc('xtick', labelsize=SMALL_FONT_SIZE)
    plt.rc('ytick', labelsize=SMALL_FONT_SIZE)


def DisplaySelectionLightCurves(transientName, setNumber, times, finalTrainingSet):
    DEFAULT_DISPLAY_CURVES = 3
    fig, axs = plt.subplots(DEFAULT_DISPLAY_CURVES)

    for curve in range(0, DEFAULT_DISPLAY_CURVES):
        randomCurveNo = int(random.random() * len(finalTrainingSet))
        axs[curve].scatter(times, finalTrainingSet[randomCurveNo], marker='+')

        axs[curve].tick_params(axis='x', labelsize=SMALL_FONT_SIZE)
        axs[curve].set_xlabel('Days', fontsize=SMALL_FONT_SIZE)
        axs[curve].set_ylabel('Mag')
        strr = 'Lightcurve No ' + str(randomCurveNo) + '(' + transientName + ') '

    plt.show()

    fig_filename = DEFAULT_TRAINING_CURVENAME + str(setNumber) + DEFAULT_FILENAME_EXT
    fig.savefig(fig_filename)


def CreateTransientTrainingData(transientName, TotalLightCurves):
    # create a suitable np array

    numberObjects = len(TotalLightCurves)
    strr = 'Processing ' + str(numberObjects) + ' lightcurves of ' + transientName
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

        # store this data for display only
        NumberPointsInCurve.append(numberPoints)
        LightCurveNumber.append(lightCurveIndex)
        lightCurveIndex += 1

        timeSeriesData = []
        magSeriesData = []
        lightCurveTraining = []
        if (bDebug):
            print("number points in this light curve = ", numberPoints)

        for point in range(numberPoints):
            pointData = lightCurve[point]
            if (point == 0):
                # lets get the start time for this sample

                startSampleTime = ConvertToJulianDay(pointData[3])

            if (bDebug):
                print("transientID = ", pointData[0])
                print("magnitude = ", pointData[1])
                print("magnitude error = ", pointData[2])
                print("MJD= ", pointData[3])

            magnitudeValue = pointData[1]

            thisTimeValue = ConvertToJulianDay(pointData[3])
            thisTimeDelta = round(thisTimeValue - startSampleTime)

            if (thisTimeDelta < 0):
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
        DisplayLightCurveIndices(transientName, LightCurveNumber, NumberPointsInCurve)

    return completeTrainingData, round(maxTimeDelta), maxSize  # this is a linked list of 2 lists of time series and mag series data


def DisplayLightCurveEntries(trainingDataSet):
    listOfData = []
    listOfEntries = []

    for entry in range(len(trainingDataSet)):
        if (trainingDataSet[entry] != 0):
            listOfEntries.append(entry)
            listOfData.append(trainingDataSet[entry])

    for entry in range(len(listOfEntries)):
        strr = '(' + str(listOfEntries[entry]) + ',' + str(listOfData[entry]) + ')'
        print(strr)


def StandardiseDataSet(trainingData, maxTimeDelta, largestTimeDelta):
    # transform raw dataset into consistent length dataset and use interpolation for
    # missing values

    if (largestTimeDelta > maxTimeDelta):
        maxTime = int(largestTimeDelta)
    else:
        maxTime = int(maxTimeDelta)

    newTrainingSet = np.zeros((maxTime + 1))

    timeSeriesData = trainingData[0]
    magSeriesData = trainingData[1]

    maxTimeDelta = round(max(timeSeriesData))

    if (bDebug):
        print("max time delta ", maxTimeDelta)
        print("maxTime = ", maxTime)
        print("no of entries in timeseries = ", len(timeSeriesData))
        print("no of entries in magseries = ", len(magSeriesData))
        print("max time delta in this series = ", maxTimeDelta)

        print(timeSeriesData)
        print(magSeriesData)

    # the training data is in 2 lists - one for the original time series and one for
    # the magnitude time series

    for entry in range(len(timeSeriesData)):
        if (bDebug):
            print("entry = ", entry)
            print("time value =", int(timeSeriesData[entry]))
            print("mag value =", magSeriesData[entry])

        if (newTrainingSet[int(timeSeriesData[entry])] != 0):
            prevSeriesData = newTrainingSet[int(timeSeriesData[entry])]
            # now average out any duplicates
            newTrainingSet[int(timeSeriesData[entry])] = (magSeriesData[entry] + prevSeriesData) / 2
        else:
            newTrainingSet[int(timeSeriesData[entry])] = magSeriesData[entry]

    if (bDebug):
        DisplayLightCurveEntries(newTrainingSet)

    return newTrainingSet


def InspectObservationFile(f,transientClassName):


    print("Inspecting...", transientClassName)

    bValidData, lightCurveData = InspectTransientClass(transientClassName)

    if (bValidData):
        numberlightCurves = len(lightCurveData)

        f.write("Transient = "+transientClassName)
        f.write("\n")

        if (numberlightCurves < MIN_LIMIT_OBSERVATIONS):
            f.write("*** Too Few LightCurve Observations For Transient"+transientClassName+" ***")
            f.write("\n")
        else:
            f.write("No. LightCurves Loaded = "+str(numberlightCurves))
            f.write("\n")

            totalNoPoints = 0
            noSmallCurves = 0

            for entry in range(numberlightCurves):

                lightCurve = lightCurveData[entry]
                if (bInspectLightCurveLength == True):
                    f.write("*** Length lightcurve = "+str(len(lightCurve)))
                    f.write("\n")


                if (len(lightCurve) < MIN_LIMIT_OBSERVATIONS):
                    if (bDetailedLowPoints):
                        f.write("*** Too Few POINTS "+str(len(lightCurve))+" For LightCurve No: "+str(entry)+" For Transient " + transientClassName + " ***")
                        f.write("\n")
                    noSmallCurves += 1

                totalNoPoints += len(lightCurve)



            f.write("\n")
            f.write("*** Average No Points Per LightCurve =  "+ str(round(totalNoPoints/numberlightCurves))+" For Transient "+transientClassName+" ***")
            f.write("\n")
            f.write("*** Small POINTS lightcurves for "+str(noSmallCurves)+" out of Total No Curves: "+str(numberlightCurves)+" ***")

            f.write("\n")


def InspectAllObservationFiles():

    f = open(DEFAULT_TRAINING_DATA_LOCATION + OBSERVATION_DATA_FILENAME + '.txt', 'w')

    for entry in range(len(DEFAULT_FULL_SET_ALLOWED_TRANSIENTS)):

        InspectObservationFile(f,DEFAULT_FULL_SET_ALLOWED_TRANSIENTS[entry])
    f.close()
    print("exiting...")
    sys.exit()


def ProcessClass(transientClassName):

    fullFileName = REF_LIGHTCURVE_CSV_LOCATION + transientClassName + REF_LIGHTCURVE_CLASSTYPE_NAME
    print("Loading...", fullFileName)

    bValidData,trainingTransientData = loadCSVFile(fullFileName)
    if (bValidData):

        numberObservations = len(trainingTransientData)

        print("No Observations Loaded = ", numberObservations)
        currentTransientID = 0
        numberTransients = 0

        TotalLightCurves = []  # this will contain a list of individual light curves
        TotalTransientList = []  # list of all processed transients

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

            lightCurvePoint = []  # this contains one point on a specific light curve

            lightCurvePoint.append(transientID)
            lightCurvePoint.append(magnitude)
            lightCurvePoint.append(magnitudeError)
            lightCurvePoint.append(MJD)

            if (transientID != currentTransientID):
                # new transient set of observations
                if (bDebug):
                    print("new transient ID = ", transientID)
                numberTransients += 1
                currentTransientID = transientID

                TotalLightCurves.append(TotalLightCurveEntry)

                # create a new list
                TotalLightCurveEntry = []
                TotalTransientList.append(transientID)

            TotalLightCurveEntry.append(lightCurvePoint)

        TotalLightCurves.append(TotalLightCurveEntry)

        trainingDataSet, maxTimeDelta, maxSize = CreateTransientTrainingData(transientClassName, TotalLightCurves)
    else:
        trainingDataSet = []
        maxTimeDelta = 0
        maxSize = 0

    return bValidData,trainingDataSet, maxTimeDelta, maxSize


def InspectTransientClass(transientClassName):

    fullFileName = REF_LIGHTCURVE_CSV_LOCATION + transientClassName + REF_LIGHTCURVE_CLASSTYPE_NAME
    print("Loading...", fullFileName)

    bValidData, trainingTransientData = loadCSVFile(fullFileName)
    if (bValidData):

        numberObservations = len(trainingTransientData)

        print("No Observations Loaded = ", numberObservations)

        currentTransientID = 0
        numberTransients = 0

        TotalLightCurves = []  # this will contain a list of individual light curves
        TotalTransientList = []  # list of all processed transients

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

            lightCurvePoint = []  # this contains one point on a specific light curve

            lightCurvePoint.append(transientID)
            lightCurvePoint.append(magnitude)
            lightCurvePoint.append(magnitudeError)
            lightCurvePoint.append(MJD)

            if (transientID != currentTransientID):
                # new transient set of observations
                if (bDebug):
                    print("new transient ID = ", transientID)
                numberTransients += 1
                currentTransientID = transientID

                TotalLightCurves.append(TotalLightCurveEntry)
                # create a new list
                TotalLightCurveEntry = []
                TotalTransientList.append(transientID)

            TotalLightCurveEntry.append(lightCurvePoint)

        TotalLightCurves.append(TotalLightCurveEntry)


    else:
        TotalLightCurves = []

    return bValidData, TotalLightCurves



def evaluateCNNModel(Xtrain, ytrain, Xtest, ytest, n_timesteps, n_features, n_outputs, epochs):
    verbose, batchSize, learningRate = DEFAULT_VERBOSE_LEVEL, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE

    if (bDebug):
        print("input train shape = ", Xtrain.shape)
        print("label train shape = ", ytrain.shape)
        print("input test shape = ", Xtest.shape)
        print("label test shape = ", ytest.shape)
        print("n_timesteps = ", n_timesteps)
        print("n_features = ", n_features)
        print("n_outputs = ", n_outputs)
        print("epochs = ", epochs)
        print("batch size =", batchSize)

    model = keras.Sequential()
    model.add(tf.keras.layers.Conv1D(64, DEFAULT_KERNEL_SIZE, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(tf.keras.layers.Conv1D(64, DEFAULT_KERNEL_SIZE, activation='relu'))

    model.add(tf.keras.layers.Dropout(DEFAULT_DROPOUT_RATE))
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(DEFAULT_NO_NEURONS, activation='relu'))

    model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=learningRate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(Xtrain, ytrain, epochs=epochs, batch_size=batchSize, verbose=verbose)

    model.summary()
    _, accuracy = model.evaluate(Xtest, ytest, batch_size=batchSize, verbose=verbose)

    return accuracy, model


def ScaleInputData(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    normalised = scaler.transform(X)

    return normalised


def StandardLightCurves(trainingDataSet, thisMaxTimeDelta, maxTimeDelta):
    finalTrainingSet = []

    for curve in range(len(trainingDataSet)):
        standardTrainingData = StandardiseDataSet(trainingDataSet[curve], thisMaxTimeDelta, maxTimeDelta)
        finalTrainingSet.append(standardTrainingData)

    return finalTrainingSet


def DisplayNonZeroPoints(text, dataSet):
    print(text)
    numberOfCurves = len(dataSet)

    AverageNoNonZeroPoints = []

    for curve in range(numberOfCurves):
        numberNonZeroPoints = 0
        lightCurve = dataSet[curve]

        for point in range(len(lightCurve)):

            if (lightCurve[point] != 0):
                numberNonZeroPoints += 1

        AverageNoNonZeroPoints.append(numberNonZeroPoints / len(lightCurve))

    print("Final % Average number non zero points =", (sum(AverageNoNonZeroPoints) / numberOfCurves) * 100)
    print("For Number of curves = ", numberOfCurves)


def testCNNModel(n_epochs, learningRate, dropoutRate,XTrain, ytrain, Xtest, ytest):

    n_timesteps = XTrain.shape[1]
    n_features = XTrain.shape[2]
    n_outputs = ytrain.shape[1]

    verbose, batchSize = DEFAULT_VERBOSE_LEVEL, DEFAULT_BATCH_SIZE

    model = keras.Sequential()
    model.add(tf.keras.layers.Conv1D(64, DEFAULT_KERNEL_SIZE, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(tf.keras.layers.Conv1D(64, DEFAULT_KERNEL_SIZE, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropoutRate))
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(DEFAULT_NO_NEURONS, activation='relu'))
    model.add(tf.keras.layers.Dense(n_outputs, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=learningRate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(XTrain, ytrain, epochs=n_epochs, batch_size=batchSize, verbose=verbose)
    _, accuracy = model.evaluate(Xtest, ytest, batch_size=batchSize, verbose=verbose)

    return accuracy, model


def DisplayHyperTable(Accuracy, Epochs, LearningRates,DropoutRate):
    from astropy.table import QTable, Table, Column

    t = Table([Accuracy, Epochs, LearningRates,DropoutRate], names=('Accuracy', 'Epochs', 'Learning Rate','Dropout Rate'))
    print(t)


def OptimiseCNNHyperparameters(XTrain, ytrain, Xtest, ytest):
    numberEpochs = [1, 5,10]
    learningRateSchedule = [0.1, 0.01, 0.001]
    dropoutRateSchedule = [0.20,0.30,0.50]


    ExperimentAccuracy = []
    ExperimentEpochs = []
    ExperimentLearningRates = []
    ExperimentDropoutRates = []


    for epoch in numberEpochs:
        for learningRate in learningRateSchedule:
            for dropoutRate in dropoutRateSchedule:

                print("Testing Number Epochs = ", epoch)
                print("Testing Learning Rate = ", learningRate)
                print("Testing Dropout Rate  = ", dropoutRate)


                accuracy, model = testCNNModel(epoch, learningRate, dropoutRate, XTrain, ytrain, Xtest, ytest)

                ExperimentAccuracy.append(accuracy)
                ExperimentEpochs.append(epoch)
                ExperimentLearningRates.append(learningRate)
                ExperimentDropoutRates.append(dropoutRate)


                print("Accuracy = ", accuracy)

    return ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates,ExperimentDropoutRates


def DisplayReducedDataTable(numberDataPoints, totalData, trueData, falseData, accuracyData):
    from astropy.table import Table

    t = Table([numberDataPoints, totalData, trueData, falseData, accuracyData],
              names=('No Data points', 'Total', 'No True', 'No False', 'Accuracy'))
    print(t)


def CheckForAccuracy(dataReduced, ypred, ytest):
    iTrue = 0
    iFalse = 0
    iTotal = 0
    numberZeros = 0

    for i in range(len(ytest)):
        iTotal += 1
        ypredicted = np.round(ypred[i])  # we're dealing with probabilities, so anything >0.5 becomes 1
        yoriginal = np.round(ytest[i])

        if (ytest[i] == 0):
            numberZeros += 1
        if (ypredicted == yoriginal):
            iTrue += 1
        else:
            iFalse += 1

    print("Total No = ", iTotal)
    print("Total No Zero Data =", numberZeros)
    print("Total Correct =", iTrue)
    print("Total Incorrect = ", iFalse)
    accuracy = np.round((iTrue / iTotal), 2)
    strr = "For Data Points Reduced to " + str(dataReduced) + ", New Accuracy = " + str(accuracy)
    print(strr)
    return dataReduced, iTotal, iTrue, iFalse, accuracy


def RemoveData(lightCurve, maxLength, nullStart):
    for i in range(maxLength - nullStart):
        lightCurve[0][i + nullStart][0] = 0

    return lightCurve


def RemoveDataFromTest(Xtest, maxLength, nullStart):
    for curve in range(len(Xtest)):
        for i in range(maxLength - nullStart):
            Xtest[curve][i + nullStart][0] = 0

    return Xtest


def TestDatasets(totalSampleSize, model, Xtest, ytest):
    numberDataPoints = []
    totalResult = []
    trueResult = []
    falseResult = []
    accuracyResult = []

    print("max length of curve = ", totalSampleSize)

    # start half way through the dataset and see the effect on the accuracy

    nullStart = int(totalSampleSize / 2)
    Xtest = RemoveDataFromTest(Xtest, totalSampleSize, nullStart)
    ypred = model.predict(Xtest)
    dataReduced, iTotal, iTrue, iFalse, accuracy = CheckForAccuracy(nullStart, ypred, ytest)
    numberDataPoints.append(totalSampleSize - dataReduced)
    totalResult.append(iTotal)
    trueResult.append(iTrue)
    falseResult.append(iFalse)
    accuracyResult.append(accuracy)

    for i in range(nullStart - 2):
        Xtest = RemoveDataFromTest(Xtest, totalSampleSize, nullStart - i)
        ypred = model.predict(Xtest)
        dataReduced, iTotal, iTrue, iFalse, accuracy = CheckForAccuracy(nullStart + i, ypred, ytest)
        numberDataPoints.append(totalSampleSize - dataReduced)
        totalResult.append(iTotal)
        trueResult.append(iTrue)
        falseResult.append(iFalse)
        accuracyResult.append(accuracy)

    return numberDataPoints, totalResult, trueResult, falseResult, accuracyResult


def CheckTransients(tran1, tran2):
    bPassTest = True

    if (tran1 == tran2):
        print("Transient Cannot be Identical")
        bPassTest = False
    elif (tran1 not in DEFAULT_MEDIUM_SET_ALLOWED_TRANSIENTS):

        print("Unknown Transient ")
        bPassTest = False
    elif (tran2 not in DEFAULT_MEDIUM_SET_ALLOWED_TRANSIENTS):
        print("Unknown Transient ")
        bPassTest = False

    return bPassTest


def CheckTransientProcessing(tranProc):
    bPassTest = True

    if ((tranProc == DEFAULT_MULTIPLE_CHOICE) or (tranProc == DEFAULT_MULTIPLE_CHOICE.lower())):
        print("Multiple Selection Chosen")

    elif (tranProc == DEFAULT_ALL_CHOICE):
        print("Full Selection Chosen")

    else:
        print("Unknown Choice ")
        bPassTest = False

    return bPassTest


def SelectTransients():
    bInput = False
    transientList = []
    bValidData = True

    while (bInput == False):

        transientName = input("Select Transient From " + str(DEFAULT_MEDIUM_SET_ALLOWED_TRANSIENTS) + " (or enter to complete) :")
        if (transientName == ''):
            #finished list
            bInput=True

            #check list

            for transient in transientList:
                if transient not in DEFAULT_MEDIUM_SET_ALLOWED_TRANSIENTS:
                    bValidData=False

                    return bValidData,transientList

        else:

            transientName =  transientName.upper()

            if (transientList.count(transientName) > 0):
                print("Duplicate Entry")
                bValidData = False

                return bValidData, transientList
            else:
                transientList.append(transientName)

    return bValidData, transientList



def SelectTransientProcessing():
    bInput = False

    while (bInput == False):

        transientProcessing = input("Enter Multiple Transients (M) or All (*) : ")
        if ((transientProcessing == DEFAULT_MULTIPLE_CHOICE) or (transientProcessing == DEFAULT_MULTIPLE_CHOICE.lower())):
            transientProcessing =DEFAULT_MULTIPLE_CHOICE
            print("Multiple Selection Chosen")
            bInput = True
            bPassTest = True

        elif (transientProcessing == DEFAULT_ALL_CHOICE):
            print("Full Selection Chosen")
            bInput = True
            bPassTest = True
        else:
            print("Unknown Choice ")
            bPassTest = False

    return bPassTest, transientProcessing

    if (CheckTransientProcessing(transientProcessing)):
            bInput = True

    return transientProcessing

def CreateCSVFiles(lightCurves):

    print("Generating Individual CSV Files For Transients...")

    ClassificationDict, TransientIDDict = ProcessTransientClasses()
    if (bDisplayTransientClasses):
        DisplayTransientNames(ClassificationDict)

    CreateAllTransientFiles(ClassificationDict)

    StoreAllObservations(TransientIDDict, lightCurves)

    CloseAllCSVFiles()

    print("Exiting...")
    sys.exit()

def SaveNumberLightCurves(transientClassName, numberLightCurves):

    f = open(DEFAULT_TRAINING_DATA_LOCATION + transientClassName + '_obs.txt', 'w')

    if (f):
        f.write("Total Number of Light Curves = " + str(numberLightCurves))

        f.close()


def ProcessMultipleClasses(transientClassList):

    trainingDataSets = []
    maxTimeDeltas = []
    maxAmplitudes = []
    numberTrainingCurves = []
    finalTrainingSet = []

    print("Processing "+str(transientClassList))


    # now process each a CSV file to create a training and test set of data
    for transient in transientClassList:

        bValidData,transientTrainingData, maxTimeDelta, maxAmplitude = ProcessClass(transient)

        if (bValidData == False) :
             print("Invalid Data")
             return FAILED_RETURN
        elif (len(transientTrainingData) < MIN_LIMIT_OBSERVATIONS):
            print("Ignoring this transient due to too few observations")
        else:
            trainingDataSets.append(transientTrainingData)
            numberTrainingCurves.append(len(transientTrainingData))
            maxTimeDeltas.append(maxTimeDelta)
            maxAmplitudes.append(maxAmplitude)
            if (bRecordLightCurves):
                SaveNumberLightCurves(transient,len(transientTrainingData))



    print("Processed Individual Transients")


    overallMaxTimeDelta = max(maxTimeDeltas)

    XData = [[] for _ in range(len(trainingDataSets))]
    ydata = [[] for _ in range(len(trainingDataSets))]

    XTrain = [[] for _ in range(len(trainingDataSets))]
    ytrain = [[] for _ in range(len(trainingDataSets))]

    Xtest = [[] for _ in range(len(trainingDataSets))]
    ytest = [[] for _ in range(len(trainingDataSets))]



    for set in range(len(trainingDataSets)):

        XData[set] = StandardLightCurves(trainingDataSets[set], maxTimeDeltas[set], overallMaxTimeDelta)


#    times = np.arange(0, overallMaxTimeDelta + 1)

    # create the one hot encoded labels

    OHELabels = createOneHotEncodedSet(transientClassList)
    # scale all data to be between 0-1

    for entry in range(len(trainingDataSets)):

        ydata[entry] = np.asarray(assignLabelSet(OHELabels[entry], len(trainingDataSets[entry])))
        XData[entry] = ScaleInputData(XData[entry])


    print("Shape of Datasets ...")

    for entry in range(len(XData)):

        numberCurvesInTrainingSet = int(round(XData[entry].shape[0] * TRAIN_TEST_RATIO))
        print("no curves in training set = ",numberCurvesInTrainingSet)

        XTrain[entry] = XData[entry][:numberCurvesInTrainingSet, :]
        Xtest[entry] = XData[entry][numberCurvesInTrainingSet:, :]

        ytrain[entry] = ydata[entry][:numberCurvesInTrainingSet]
        ytest[entry] = ydata[entry][numberCurvesInTrainingSet:]


    # XTrain and Xtest now contain (in a list) all of the split data for each transient
    # now concatenate these slices into one contiguous data set for training and testing


    completeXTrain = XTrain[0]
    completeXtest = Xtest[0]

    completeytrain = ytrain[0]
    completeytest = ytest[0]

    for entry in range(1,len(XTrain)):

        completeXTrain = np.concatenate((completeXTrain,XTrain[entry]))
        completeXtest = np.concatenate((completeXtest,Xtest[entry]))

        completeytrain = np.concatenate((completeytrain,ytrain[entry]))
        completeytest = np.concatenate((completeytest,ytest[entry]))

    print("no of training light curves = ", len(completeXTrain))
    print("no of test light curves = ", len(completeXtest))

    print("no of training light curve labels = ", len(completeytrain))
    print("no of test light curve labels = ", len(completeytest))


    # create an integrated set of training data which includes the transient and the random data
    # ensure that the sequence numbers are kept to manage the label data



    index = np.random.choice(completeXTrain.shape[0], len(completeXTrain), replace=False)

    XTrain = completeXTrain[index]
    ytrain = completeytrain[index]


    index = np.random.choice(completeXtest.shape[0], len(completeXtest), replace=False)

    Xtest = completeXtest[index]
    ytest = completeytest[index]

    if (bDebug):
        print("Final Training Data = ", XTrain.shape)
        print("Final Test Data = ", Xtest.shape)

        print("Final Training Label Data = ", ytrain.shape)
        print("Final Test Label Data = ", ytest.shape)

    XTrain = np.reshape(XTrain, (XTrain.shape[0], XTrain.shape[1], 1))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))

    if (bDebug):
        print("shape of XTrain =", XTrain.shape)
        print("shape of Xtest =", Xtest.shape)
        print("shape of ytrain =", ytrain.shape)
        print("shape of ytest =", ytest.shape)


    n_timesteps = XTrain.shape[1]
    n_features = XTrain.shape[2]
    n_outputs = ytrain.shape[1]


    if (bDebug):
        print(n_timesteps, n_features, n_outputs)

    Accuracy, CNNModel = evaluateCNNModel(XTrain, ytrain, Xtest, ytest, n_timesteps, n_features, n_outputs,
                                          DEFAULT_NO_EPOCHS)
    print("CNN Accuracy = ",Accuracy)

 #   if (bTestReducedData == True):
 #       Xtest = RemoveDataFromTest(Xtest, n_timesteps, DEFAULT_DATA_REDUCTION)
 #       ypred = CNNModel.predict(Xtest)
 #       CheckForAccuracy(DEFAULT_DATA_REDUCTION, ypred, ytest)
    bOptimiseParameters=True
    if (bOptimiseParameters == True):
        ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates, ExperimentDropoutRates = OptimiseCNNHyperparameters(XTrain, ytrain,
                                                                                                   Xtest, ytest)
        DisplayHyperTable(ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates, ExperimentDropoutRates)

    if (bTestReducedData == True):
        numberDataPoints, totalData, trueData, falseData, accuracyData = TestDatasets(n_timesteps, CNNModel, Xtest,
                                                                                      ytest)
        DisplayReducedDataTable(numberDataPoints, totalData, trueData, falseData, accuracyData)

    return Accuracy


def DisplayAllAccuracies(f, MultiClass1, MultiClass2, MultiClassAccuracy):
    totalAccuracy = 0

    print("in display all accuracies")
    numberEntries = len(MultiClass1)

    print("no of entries = ", numberEntries)

    for entry in range(numberEntries):
        totalAccuracy += MultiClassAccuracy[entry]
        print("For Transient " + MultiClass1[entry] + " versus " + MultiClass2[entry] + " Accuracy = " + str(
            MultiClassAccuracy[entry]))

    if (numberEntries >0):

        f.write("\n")
        strr = "Average Accuracy Across " + str(len(MultiClass1)) + " Objects = " + str(totalAccuracy/numberEntries)
        print(strr)
        f.write(strr)
        f.write("\n")


def SaveAccuracyResults(f, class1, class2, accuracy):
    strr = "Results (" + class1 + "," + class2 + ") = " + str(accuracy)
    f.write(strr)
    f.write("\n")

def SaveNumberLightCurves(transientClassName,numberLightCurves)  :

        f = open(DEFAULT_TRAINING_DATA_LOCATION +transientClassName  + '_obs.txt', 'w')

        if (f):
            f.write("Total Number of Light Curves = "+str(numberLightCurves))

            f.close()


def ProcessAllClasses():
    MultiClass1 = []
    MultiClass2 = []
    MultiClassAccuracy = []


    transientSet = DEFAULT_FULL_SET_ALLOWED_TRANSIENTS


    f = open(DEFAULT_TRAINING_DATA_LOCATION + DEFAULT_RESULTS_FILE + '.txt', 'w')

    numberTransientClasses = len(transientSet)

    print("Number of Transient Classes to Process =", numberTransientClasses)
    print("number of combinations =", int(numberTransientClasses * (numberTransientClasses - 1) / 2))

    numberProcessed = 0

    for tran1 in range(numberTransientClasses):
        class1ToProcess = transientSet[tran1]
        print("class1 to process =", class1ToProcess)
        for tran2 in range(tran1 + 1, numberTransientClasses):
            binarySet = []

            class2ToProcess = transientSet[tran2]
            binarySet.append(class1ToProcess)
            binarySet.append(class2ToProcess)

            print("class2 to process =", class2ToProcess)
            numberProcessed += 1
            accuracy = ProcessMultipleClasses(binarySet)
            if (accuracy != FAILED_RETURN):
                print("accuracy = ", accuracy)
                MultiClass1.append(class1ToProcess)
                MultiClass2.append(class2ToProcess)
                MultiClassAccuracy.append(accuracy)

                SaveAccuracyResults(f, class1ToProcess, class2ToProcess, accuracy)
            else:
                print("Failed In Processing Binary Classes")
                sys.exit()

    DisplayAllAccuracies(f, MultiClass1, MultiClass2, MultiClassAccuracy)

    f.close()
    print("no processed = ", numberProcessed)


def main():

    lightCurves = load_file(REF_LIGHTCURVE_LOCATION + REF_LIGHTCURVE_FILENAME)

    if (bCalcAveragesFromFile):
        # just used to retrospectively calculate overall accuracy average from results file

        ProcessResultsFile()

    if (bInspectObservations):

        InspectAllObservationFiles()

    if (bGenerateCSVFiles):

        CreateCSVFiles(lightCurves)

    bValidData, tranProc = SelectTransientProcessing()

    if (bValidData):
        if (tranProc == DEFAULT_MULTIPLE_CHOICE):

            bValidData, transientClassList = SelectTransients()

            if (bValidData):

                print(transientClassList)

                ProcessMultipleClasses(transientClassList)
            else:
                print("Invalid Input")

        else:
            # we're going to do them all

            ProcessAllClasses()


if __name__ == '__main__':
    main()
