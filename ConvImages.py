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
from astropy.io import fits


DEFAULT_IMAGE_LOCATION = '/Volumes/ExtraDisk/AGN/'
DEFAULT_AGN_SOURCE_LOCATION = '/Volumes/ExtraDisk/AGN/'
DEFAULT_BLAZAR_SOURCE_LOCATION = '/Volumes/ExtraDisk/BLAZAR/'
DEFAULT_TEST_SOURCE_LOCATION = '/Volumes/ExtraDisk/TESTFITS/'
DEFAULT_HYPER_FILENAME = 'CNNHyper.txt'
DEFAULT_AGN_SOURCES_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'AGN_sources.txt'
DEFAULT_BLAZAR_SOURCES_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'BLAZAR_sources.txt'
DEFAULT_HYPERPARAMETERS_FILE = DEFAULT_TEST_SOURCE_LOCATION+DEFAULT_HYPER_FILENAME
DEFAULT_FITS_NO_TESTS = 10
DEFAULT_OUTPUT_FITS_AGN_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'AGN_FITS.png'
DEFAULT_OUTPUT_FITS_BLAZAR_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'BLAZAR_FITS.png'

FAILED_TO_OPEN_FILE = -1
DS_STORE_FILENAME = '.'
DEFAULT_CSV_DIR = 'CSVFiles'
FOLDER_IDENTIFIER = '/'
SOURCE_TITLE_TEXT = ' Source :'
UNDERSCORE = '_'
DEFAULT_CSV_FILETYPE = UNDERSCORE+'data.txt'
XSIZE_FITS_IMAGE= 120
YSIZE_FITS_IMAGE = 120

DEFAULT_AGN_CLASS = "AGN"
DEFAULT_BLAZAR_CLASS = "BLAZAR"
DEFAULT_CLASSIC_AGN_CLASS = 0
DEFAULT_CLASSIC_BLAZAR_CLASS = 1

DEFAULT_CLASSIC_MODEL = True
DEFAULT_NN_MODEL = False
# Defaults for CNN Model

DEFAULT_VERBOSE_LEVEL = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_NO_EPOCHS = 5
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT_RATE = 0.30
DEFAULT_NO_NEURONS = 100

TRAIN_TEST_RATIO = 0.80  # ratio of the total data for training

SMALL_FONT_SIZE = 8
MEDIUM_FONT_SIZE = 10
BIGGER_FONT_SIZE = 12



bCreateCSVFiles = False # flag set to create all CSV files from FITS images
bDebug = False # swich on debug code
bOptimiseHyperParameters = False # optimise hyper parameters used on convolutional model
bTestFITSFile = False # test CNN model with individual FITS files
bTestClassicModels = False # test selection of other models
bTestRandomForestModel = True # test random forest model only
bSaveImageFiles = False # save FITS files for presentations

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


def testCNNModel(n_epochs, learningRate, dropoutRate,XTrain, ytrain, XTest, ytest):

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
    _, accuracy = model.evaluate(XTest, ytest, batch_size=batchSize, verbose=verbose)

    return accuracy, model



def OptimiseCNNHyperparameters(f,XTrain, ytrain, XTest, ytest):
    numberEpochs = [5, 10,25]
    learningRateSchedule = [0.01, 0.001, 0.0001]
    dropoutRateSchedule = [0.20,0.30,0.50]


    ExperimentAccuracy = []
    ExperimentEpochs = []
    ExperimentLearningRates = []
    ExperimentDropoutRates = []

    experimentNumber = 1
    totalNoExperiments = len(numberEpochs)*len(learningRateSchedule)*len(dropoutRateSchedule)


    for epoch in numberEpochs:
        for learningRate in learningRateSchedule:
            for dropoutRate in dropoutRateSchedule:

                strr = 'Experiment No: '+str(experimentNumber)+' of '+str(totalNoExperiments)
                print(strr)
                print("Testing Number Epochs = ", epoch)
                print("Testing Learning Rate = ", learningRate)
                print("Testing Dropout Rate  = ", dropoutRate)


                accuracy, model = testCNNModel(epoch, learningRate, dropoutRate, XTrain, ytrain, XTest, ytest)

                ExperimentAccuracy.append(accuracy)
                ExperimentEpochs.append(epoch)
                ExperimentLearningRates.append(learningRate)
                ExperimentDropoutRates.append(dropoutRate)

                WriteToOptimsationFile(f,experimentNumber,accuracy,epoch,learningRate,dropoutRate)
                experimentNumber += 1

                print("Accuracy = ", accuracy)

    return ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates,ExperimentDropoutRates



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

def labelDecoder(listOfLabels,labelValue):

    from sklearn.preprocessing import OrdinalEncoder


    # use label to get one hot encoded value

    ordinalEncoder = OrdinalEncoder()

    listOfLabels = np.array(listOfLabels)
    listOfLabels= listOfLabels.reshape(-1,1)
    integerEncoded = ordinalEncoder.fit_transform(listOfLabels)

    label = ordinalEncoder.inverse_transform([[labelValue]])

    return label

def ScaleInputData(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    normalised = scaler.transform(X)

    return normalised
def DisplayFITSImage(imageData,figHeader):

    plt.title(figHeader)
    plt.imshow(imageData,cmap='gray')
    plt.colorbar()

    plt.show()



def SaveSampleImages(imageData,titleData,filename):

    plt.rc('axes', titlesize=SMALL_FONT_SIZE)
    numberRows = 1

    fig,axs = plt.subplots()
    numberImages = len(imageData)
    for i in range(numberImages):
        plt.subplot(numberRows,numberImages,i+1)
        plt.title(titleData[i])
        plt.imshow(imageData[i],cmap='gray')
        #plt.colorbar()

    plt.show()

    fig.savefig(filename)

def OpenFITSFile(filename):
    bValidData = True

    if (filename):

        hdul = fits.open(filename)

        imageData = hdul[0].data


        if ((imageData.shape[0] != XSIZE_FITS_IMAGE) or (imageData.shape[1] != YSIZE_FITS_IMAGE)):
            print("invalid FITS image size")
            sys.exit()
    else:
        print("Failed To Open FITS File")
        bValidData = False


    return bValidData, imageData

def DisplayImageContents(imageData):

    for i in range(imageData.shape[0]):
        for j in range(imageData.shape[1]):
            strr = 'Value at ['+str(i)+','+str(j)+'] = '+str(imageData[i,j])
            print(strr)


def ScanForSources(sourceLocation):
    dirList = []

    sourceList = os.scandir(sourceLocation)
    for entry in sourceList:
        if entry.is_dir():
            if (entry.name != DEFAULT_CSV_DIR):
                dirList.append(entry.name)
            else:
                if (bDebug):
                    print("**** IGNORING CSV Files ***")

    return dirList

def ScanForImages(sourceLocation,sourceNumber):

    imageList = []


    imageLocation = sourceLocation+sourceNumber

    fileList = os.scandir(imageLocation)
    for entry in fileList:
        if entry.is_dir():
            if (bDebug):
                print("dir = ",entry.name)

        elif entry.is_file():
            if entry.name[0] != DS_STORE_FILENAME :
                imageList.append(entry.name)

            else:
                if (bDebug):
                    print("File Entry Ignored For ",entry.name)

    return imageLocation, imageList

def ScanForTestImages(imageLocation):

    imageList = []

    print("image location = ",imageLocation)

    fileList = os.scandir(imageLocation)
    for entry in fileList:
        if entry.is_dir():
            if (bDebug):
                print("dir = ",entry.name)

        elif entry.is_file():
            if entry.name[0] != DS_STORE_FILENAME :
                imageList.append(entry.name)

                if (bDebug):
                    print("entry is file ",entry.name)
            else:
                if (bDebug):
                    print("File Entry Ignored For ",entry.name)

    return imageList

def OpenOptimsationFile():

    f = open(DEFAULT_HYPERPARAMETERS_FILE, "w")
    return f

def WriteToOptimsationFile(f,experimentNumber,accuracy,noEpochs,learningRate,dropoutRate):

    if (f):

        strr = 'Experiment Number: '+str(experimentNumber)+'\n'
        f.write(strr)
        strr = 'For Epochs = '+str(noEpochs)+' , Learning Rate = '+str(learningRate)+' , Dropout Rate = '+str(dropoutRate)+'\n'
        f.write(strr)
        strr = 'Accuracy = '+str(accuracy)+'\n'
        f.write(strr)
        f.write('\n')

def CreateAllCSVFiles(sourceDir,sourceList):

    sourceFileDict = {}
    for source in sourceList:
        # create a CSV file using the source name

        imageLocation, imageList = ScanForImages(sourceDir, source)
        numberImages = len(imageList)

        for imageNo in range(numberImages):
            sourceCSVFileName = sourceDir + DEFAULT_CSV_DIR + FOLDER_IDENTIFIER + source +'_'+str(imageNo)+ DEFAULT_CSV_FILETYPE
            if (bDebug):
                print(sourceCSVFileName)

            f = open(sourceCSVFileName,"w")
            if (f):
                if source in sourceFileDict:
                    print("*** Error - Source File Already Been Created ***")
                else:
                    if (bDebug):
                        print("Success in creating csv file")
                    sourceFileDict[source+UNDERSCORE+str(imageNo)] = f

    return sourceFileDict

def StoreImageContents(f,imageData):

    for i in range(imageData.shape[0]):
        for j in range(imageData.shape[1]):
            strr = str(imageData[i,j])
            f.write(strr)
            if (j+1 ==imageData.shape[1]):
                # we're at the end of a row
                f.write('\n')
            else:
                f.write(',')



def StoreInCSVFile(imageLocation,image,f):

    imageLocation += FOLDER_IDENTIFIER

    bValidData,imageData = OpenFITSFile(imageLocation+image)
    if (bValidData):
        # ok - found the image data, now store in the correct CSV file

        StoreImageContents(f, imageData)

def GetFITSFile(imageLocation,imageName):

    imageLocation += FOLDER_IDENTIFIER

    bValidData,imageData = OpenFITSFile(imageLocation+imageName)

    return bValidData, imageData


def ProcessAllCSVFiles(sourceDir,fileHandleDict,sourceList):


    for source in sourceList:
        # get list of all files for this source
        fileNumber = 0
        imageLocation,imageList = ScanForImages(sourceDir,source)

        if (len(imageList) >0):
            for image in imageList:

                imageCSVFile = source+UNDERSCORE+str(fileNumber)

                f = fileHandleDict[imageCSVFile]
                fileNumber += 1
                StoreInCSVFile(imageLocation, image,f)
                f.close()

def loadCSVFile(sourceDir,source,imageNo):

    filePath =  sourceDir + DEFAULT_CSV_DIR + FOLDER_IDENTIFIER + source +'_'+str(imageNo)+ DEFAULT_CSV_FILETYPE

    bDataValid = True

    if (os.path.isfile(filePath)):
        if (bDebug):
            print("*** Loading CSV File "+filePath+" ***")
        dataframe = pd.read_csv(filePath, header=None)
        dataReturn = dataframe.values
        if (bDebug):
            print("*** Completed Loading CSV File")
    else:
        print("*** CSV File Does Not Exist ***")
        bDataValid = False
        dataReturn = []

    return bDataValid,dataReturn


def StoreSourcesToFile(sourceLocation,sourceList):

    if (sourceLocation == DEFAULT_AGN_SOURCE_LOCATION):
        filename = DEFAULT_AGN_SOURCES_FILENAME
        sourceType = DEFAULT_AGN_CLASS
    else:

        filename = DEFAULT_BLAZAR_SOURCES_FILENAME
        sourceType = DEFAULT_BLAZAR_CLASS

    f = open(filename, "w")
    if (f):
        for source in range(len(sourceList)):

            strr = 'Source: '+sourceType+' : '+str(sourceList[source])+' \n'
            f.write(strr)

        f.close()

def ProcessTransientData(sourceLocation):

    trainingData = []

    sourceList = ScanForSources(sourceLocation)

    StoreSourcesToFile(sourceLocation,sourceList)

    numberSources = len(sourceList)

    for source in range(numberSources):

        imageLocation, imageList = ScanForImages(sourceLocation, sourceList[source])

        for imageNo in range(len(imageList)):
            bValidData, sourceData = loadCSVFile(sourceLocation, sourceList[source], imageNo)
            if (bValidData):
                sourceData = np.reshape(sourceData, (1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE))

                trainingData.append(sourceData)


    return trainingData



def createLabels(label1,label2):

    transientClassList = []

    transientClassList.append(label1)
    transientClassList.append(label2)

    OHELabels = createOneHotEncodedSet(transientClassList)

    return OHELabels

def decodeLabels(label1,label2,predictions):

    transientClassList = []

    transientClassList.append(label1)
    transientClassList.append(label2)

    label = transientClassList[np.argmax(predictions)]


    return label


def assignLabelSet(label, numberOfSamples):


    shape = (numberOfSamples, len(label))

    a = np.empty((shape))

    a[:]=label

    return a

def createLabelSet(label, numberOfSamples):


    a = np.empty((numberOfSamples,))

    a[:]=label

    return a

def DisplayHyperTable(Accuracy, Epochs, LearningRates,DropoutRate):
    from astropy.table import QTable, Table, Column

    t = Table([Accuracy, Epochs, LearningRates,DropoutRate], names=('Accuracy', 'Epochs', 'Learning Rate','Dropout Rate'))
    print(t)




def GetOptimalParameters(Accuracy,Epochs, LearningRates, DropoutRates):

    largestEntry = Accuracy.index(max(Accuracy))

    print("Best accuracy = ",Accuracy[largestEntry])
    print("Best epochs = ",Epochs[largestEntry])
    print("Best LR = ",LearningRates[largestEntry])
    print("Best dropout = ",DropoutRates[largestEntry])


    return Epochs[largestEntry],LearningRates[largestEntry],DropoutRates[largestEntry]


def CreateTrainingAndTestData(trainingAGNData,trainingBLAZARData):

    # create the one hot encoded labels

    OHELabels = createLabels(DEFAULT_AGN_CLASS, DEFAULT_BLAZAR_CLASS)

    # create label set and scale all data to be between 0-1

    agnLabel = np.asarray(assignLabelSet(OHELabels[0], len(trainingAGNData)))
    blazarLabel = np.asarray(assignLabelSet(OHELabels[1], len(trainingBLAZARData)))

    AGNData = np.asarray(trainingAGNData)
    AGNData = np.reshape(trainingAGNData, (AGNData.shape[0], AGNData.shape[2]))

    BLAZARData = np.asarray(trainingBLAZARData)
    BLAZARData = np.reshape(BLAZARData, (BLAZARData.shape[0], BLAZARData.shape[2]))

    numberVectorsInAGNTrainingSet = int(round(AGNData.shape[0] * TRAIN_TEST_RATIO))
    numberVectorsInBLAZARTrainingSet = int(round(BLAZARData.shape[0] * TRAIN_TEST_RATIO))

    AGNTrain = AGNData[:numberVectorsInAGNTrainingSet]
    AGNTest = AGNData[numberVectorsInAGNTrainingSet:]

    AGNLabelTrain = agnLabel[:numberVectorsInAGNTrainingSet]
    AGNLabelTest = agnLabel[numberVectorsInAGNTrainingSet:]

    BLAZARTrain = BLAZARData[:numberVectorsInBLAZARTrainingSet]
    BLAZARTest = BLAZARData[numberVectorsInBLAZARTrainingSet:]

    BLAZARLabelTrain = blazarLabel[:numberVectorsInBLAZARTrainingSet]
    BLAZARLabelTest = blazarLabel[numberVectorsInBLAZARTrainingSet:]


    XTrain = np.concatenate((AGNTrain, BLAZARTrain))
    XTest = np.concatenate((AGNTest, BLAZARTest))

    ytrain = np.concatenate((AGNLabelTrain, BLAZARLabelTrain))
    ytest = np.concatenate((AGNLabelTest, BLAZARLabelTest))
    if (bDebug):
        print("Final train/test shape...")

        print("XTrain = ", XTrain.shape)
        print("XTest = ", XTest.shape)
        print("ytrain = ", ytrain.shape)
        print("ytest = ", ytest.shape)

    # create an integrated set of training data which includes the transient and the random data
    # ensure that the sequence numbers are kept to manage the label data

    index = np.random.choice(XTrain.shape[0], len(XTrain), replace=False)

    XTrain = XTrain[index]
    ytrain = ytrain[index]

    index = np.random.choice(XTest.shape[0], len(XTest), replace=False)

    XTest = XTest[index]
    ytest = ytest[index]

    if (bDebug):
        print("Final Training Data = ", XTrain.shape)
        print("Final Test Data = ", XTest.shape)

        print("Final Training Label Data = ", ytrain.shape)
        print("Final Test Label Data = ", ytest.shape)

    XTrain = np.reshape(XTrain, (XTrain.shape[0], XTrain.shape[1], 1))
    XTest = np.reshape(XTest, (XTest.shape[0], XTest.shape[1], 1))

    if (bDebug):
        print("shape of XTrain =", XTrain.shape)
        print("shape of Xtest =", XTest.shape)
        print("shape of ytrain =", ytrain.shape)
        print("shape of ytest =", ytest.shape)

    return XTrain, XTest, ytrain, ytest

def CreateClassicTrainingAndTestData(trainingAGNData,trainingBLAZARData):

    # create label set and scale all data to be between 0-1

    agnLabel = np.asarray(createLabelSet(DEFAULT_CLASSIC_AGN_CLASS, len(trainingAGNData)))
    blazarLabel = np.asarray(createLabelSet(DEFAULT_CLASSIC_BLAZAR_CLASS, len(trainingBLAZARData)))

    AGNData = np.asarray(trainingAGNData)
    AGNData = np.reshape(trainingAGNData, (AGNData.shape[0], AGNData.shape[2]))

    BLAZARData = np.asarray(trainingBLAZARData)
    BLAZARData = np.reshape(BLAZARData, (BLAZARData.shape[0], BLAZARData.shape[2]))

    numberVectorsInAGNTrainingSet = int(round(AGNData.shape[0] * TRAIN_TEST_RATIO))
    numberVectorsInBLAZARTrainingSet = int(round(BLAZARData.shape[0] * TRAIN_TEST_RATIO))

    AGNTrain = AGNData[:numberVectorsInAGNTrainingSet]
    AGNTest = AGNData[numberVectorsInAGNTrainingSet:]

    AGNLabelTrain = agnLabel[:numberVectorsInAGNTrainingSet]
    AGNLabelTest = agnLabel[numberVectorsInAGNTrainingSet:]

    BLAZARTrain = BLAZARData[:numberVectorsInBLAZARTrainingSet]
    BLAZARTest = BLAZARData[numberVectorsInBLAZARTrainingSet:]

    BLAZARLabelTrain = blazarLabel[:numberVectorsInBLAZARTrainingSet]
    BLAZARLabelTest = blazarLabel[numberVectorsInBLAZARTrainingSet:]


    XTrain = np.concatenate((AGNTrain, BLAZARTrain))
    XTest = np.concatenate((AGNTest, BLAZARTest))

    ytrain = np.concatenate((AGNLabelTrain, BLAZARLabelTrain))
    ytest = np.concatenate((AGNLabelTest, BLAZARLabelTest))

    # create an integrated set of training data which includes the transient and the random data
    # ensure that the sequence numbers are kept to manage the label data

    index = np.random.choice(XTrain.shape[0], len(XTrain), replace=False)

    XTrain = XTrain[index]
    ytrain = ytrain[index]

    index = np.random.choice(XTest.shape[0], len(XTest), replace=False)

    XTest = XTest[index]
    ytest = ytest[index]

    if (bDebug):
        print("Final Training Data = ", XTrain.shape)
        print("Final Test Data = ", XTest.shape)

        print("Final Training Label Data = ", ytrain.shape)
        print("Final Test Label Data = ", ytest.shape)

    return XTrain, XTest, ytrain, ytest


def ProcessRandomFile(bClassicOrNN,sourceDir,model):

    sourceList = ScanForSources(sourceDir)

    randomEntry = int(random.random() * len(sourceList))

    source = sourceList[randomEntry]

    imageLocation, imageList = ScanForImages(sourceDir, source)

    randomImage = int(random.random() * len(imageList))
    randomImageName = imageList[randomImage]

    # ok - have got a valid FITS file, lets classify according to the model

    bValidData, sourceData = loadCSVFile(sourceDir, source, randomImage)
    if (bValidData):
        if (bDebug):
            print("*** Successfully Loaded Individual FITS File ",randomImageName)
        sourceData = np.reshape(sourceData, (1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE))
        if (bClassicOrNN):
            ypredicted = model.predict(sourceData)
        else:
            ypredicted = model.predict(sourceData)

        if (bClassicOrNN):
            if (ypredicted[0] == DEFAULT_CLASSIC_AGN_CLASS):
                label = DEFAULT_AGN_CLASS
            else:
                label = DEFAULT_BLAZAR_CLASS

        else:
            label = decodeLabels(DEFAULT_AGN_CLASS, DEFAULT_BLAZAR_CLASS, ypredicted)

    return bValidData, label


def SaveRandomImageFiles(numberSamples,sourceDir):

    if (sourceDir == DEFAULT_AGN_SOURCE_LOCATION):
        filename = DEFAULT_OUTPUT_FITS_AGN_FILENAME
        fileTitle = DEFAULT_AGN_CLASS+SOURCE_TITLE_TEXT
    else:
        filename = DEFAULT_OUTPUT_FITS_BLAZAR_FILENAME
        fileTitle = DEFAULT_BLAZAR_CLASS+SOURCE_TITLE_TEXT

    imageFigData = []
    titleData = []

    for sample in range(numberSamples):

        sourceList = ScanForSources(sourceDir)

        randomEntry = int(random.random() * len(sourceList))

        source = sourceList[randomEntry]

        imageLocation, imageList = ScanForImages(sourceDir, source)

        randomImage = int(random.random() * len(imageList))
        randomImageName = imageList[randomImage]
        print("image name = ",randomImageName)

        filePath = sourceDir + source +FOLDER_IDENTIFIER+randomImageName

        bValidImage,fitsImage = OpenFITSFile(filePath)

        if (bValidImage):

            imageFigData.append(fitsImage)
            titleData.append(fileTitle+source)


    SaveSampleImages(imageFigData,titleData,filename)


def TestRandomFITSFiles(numberFiles,bClassicOrNN,model):

    # test model with specific (but random) files
    print("*** Testing Random FITS Files ***")
    agnPredictions = []
    correctAGN = 0
    blazarPredictions = []
    correctBLAZAR = 0


    for i in range(numberFiles):

        bValidData, label = ProcessRandomFile(bClassicOrNN,DEFAULT_AGN_SOURCE_LOCATION, model)
        if (bValidData):
            agnPredictions.append(label)

        bValidData, label = ProcessRandomFile(bClassicOrNN,DEFAULT_BLAZAR_SOURCE_LOCATION, model)

        if (bValidData):
            blazarPredictions.append(label)


    print("*** Individual AGN Predictions ***")

    for i in range(numberFiles):
        if (agnPredictions[i] == DEFAULT_AGN_CLASS):
            correctAGN += 1
        if (bDebug):
            print(agnPredictions[i])

    print("*** Individual BLAZAR Predictions ***")
    for i in range(numberFiles):
        if (blazarPredictions[i] == DEFAULT_BLAZAR_CLASS):
            correctBLAZAR += 1
        if (bDebug):
            print(blazarPredictions[i])

    # summary
    print("*** SUMMARY ***")
    print(" No correct AGN = ",correctAGN)
    print(" No correct BLAZAR = ",correctBLAZAR)

def MultipleClassicModels(XTrain,ytrain,XTest,ytest):

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score


    logClf = LogisticRegression()
    rndClf = RandomForestClassifier()
    svmClf = SVC()

    votingClf = VotingClassifier(
        estimators=[('lr',logClf),('rf',rndClf),('svc',svmClf)],
        voting='hard')
    votingClf.fit(XTrain,ytrain)

    for clf in (logClf,rndClf,svmClf,votingClf):
        clf.fit(XTrain,ytrain)
        y_pred = clf.predict(XTest)
        print(clf.__class__.__name__,accuracy_score(ytest,y_pred))


def RandomForestModel(XTrain,ytrain,XTest,ytest):

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    rndClf = RandomForestClassifier()

    rndClf.fit(XTrain,ytrain)
    y_pred = rndClf.predict(XTest)
    print(rndClf.__class__.__name__,accuracy_score(ytest,y_pred))

    return rndClf

def main():

    if (bCreateCSVFiles):
        print("*** Processing ALl AGN Files ***")

        sourceAGNList = ScanForSources(DEFAULT_AGN_SOURCE_LOCATION)
        agnSourceFileDict = CreateAllCSVFiles(DEFAULT_AGN_SOURCE_LOCATION,sourceAGNList)

        ProcessAllCSVFiles(DEFAULT_AGN_SOURCE_LOCATION,agnSourceFileDict,sourceAGNList)

        print("*** Processing ALl BLAZAR Files ***")

        sourceBLAZARList = ScanForSources(DEFAULT_BLAZAR_SOURCE_LOCATION)
        blazarSourceFileDict = CreateAllCSVFiles(DEFAULT_BLAZAR_SOURCE_LOCATION, sourceBLAZARList)

        ProcessAllCSVFiles(DEFAULT_BLAZAR_SOURCE_LOCATION,blazarSourceFileDict, sourceBLAZARList)
    else:


        # now process all images per AGN and BLAZAR class

        print("*** Loading Training Data ***")

        trainingAGNData = ProcessTransientData(DEFAULT_AGN_SOURCE_LOCATION)
        trainingBLAZARData = ProcessTransientData(DEFAULT_BLAZAR_SOURCE_LOCATION)

        if (bSaveImageFiles):

            SaveRandomImageFiles(3,DEFAULT_AGN_SOURCE_LOCATION)
            SaveRandomImageFiles(3,DEFAULT_BLAZAR_SOURCE_LOCATION)

        print("*** Creating Training and Test Data Sets ***")
        if (bTestClassicModels):


            XTrain, XTest, ytrain, ytest = CreateClassicTrainingAndTestData(trainingAGNData, trainingBLAZARData)
            print("*** Evaluating Multiple Classic Models ***")
            MultipleClassicModels(XTrain, ytrain, XTest, ytest)


        if (bTestRandomForestModel):




                XTrain, XTest, ytrain, ytest = CreateClassicTrainingAndTestData(trainingAGNData, trainingBLAZARData)
                print("*** Evaluating Random Forest Model ***")
                randomForestModel = RandomForestModel(XTrain, ytrain, XTest, ytest)



                if (bTestFITSFile):
                        TestRandomFITSFiles(DEFAULT_FITS_NO_TESTS, DEFAULT_CLASSIC_MODEL, randomForestModel)
                        TestRandomFITSFiles(100,DEFAULT_CLASSIC_MODEL,randomForestModel)
                        TestRandomFITSFiles(1000, DEFAULT_CLASSIC_MODEL, randomForestModel)


        else:

            XTrain, XTest, ytrain, ytest = CreateTrainingAndTestData(trainingAGNData, trainingBLAZARData)

            n_timesteps = XTrain.shape[1]
            n_features = XTrain.shape[2]
            n_outputs = ytrain.shape[1]

            if (bDebug):
                print(n_timesteps, n_features, n_outputs)

            if (bOptimiseHyperParameters == True):

                fOptimisationFile = OpenOptimsationFile()

                if (fOptimisationFile):
                    print("*** Optimising CNN Hyper Parameters ***")
                    ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates, ExperimentDropoutRates = OptimiseCNNHyperparameters(fOptimisationFile,XTrain, ytrain,XTest, ytest)
                    fOptimisationFile.close()
                    DisplayHyperTable(ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates, ExperimentDropoutRates)
                    BestEpochs, BestLearningRate, BestDropoutRate = GetOptimalParameters(ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates, ExperimentDropoutRates)

                    Accuracy, CNNModel = testCNNModel(BestEpochs,BestLearningRate,BestDropoutRate, XTrain, ytrain, XTest, ytest)
                else:
                    print("*** Failed To Open CNN Hyper Parameters File ***")

            else:
                print("*** Evaluating CNN Model ***")
                Accuracy, CNNModel = evaluateCNNModel(XTrain, ytrain, XTest, ytest, n_timesteps, n_features, n_outputs,DEFAULT_NO_EPOCHS)

            print("Final CNN Accuracy = ", Accuracy)

            if (bTestFITSFile):


                TestRandomFITSFiles(DEFAULT_FITS_NO_TESTS,DEFAULT_NN_MODEL,CNNModel)
                TestRandomFITSFiles(100,DEFAULT_NN_MODEL, CNNModel)
                TestRandomFITSFiles(1000,DEFAULT_NN_MODEL, CNNModel)


if __name__ == '__main__':
    main()
