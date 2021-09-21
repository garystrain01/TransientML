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
from sklearn.model_selection import StratifiedShuffleSplit


DEFAULT_IMAGE_LOCATION = '/Volumes/ExtraDisk/AGN/'
DEFAULT_AGN_SOURCE_LOCATION = '/Volumes/ExtraDisk/AGN/'
DEFAULT_PULSAR_SOURCE_LOCATION = '/Volumes/ExtraDisk/PULSAR/'
DEFAULT_BLAZAR_SOURCE_LOCATION = '/Volumes/ExtraDisk/BLAZAR/'
DEFAULT_SEYFERT_SOURCE_LOCATION = '/Volumes/ExtraDisk/Seyferts/'
DEFAULT_QUASAR_SOURCE_LOCATION = '/Volumes/ExtraDisk/Quasar/'
DEFAULT_TEST_SOURCE_LOCATION = '/Volumes/ExtraDisk/TESTFITS/'
DEFAULT_HYPER_FILENAME = 'CNNHyper.txt'
DEFAULT_PULSAR_SOURCES_FILENAME = DEFAULT_PULSAR_SOURCE_LOCATION+'ATNF.txt'
DEFAULT_AGN_SOURCES_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'AGN_sources.txt'

DEFAULT_BLAZAR_SOURCES_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'BLAZAR_sources.txt'
DEFAULT_SEYFERT_SOURCES_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'SEYFERT_sources.txt'
DEFAULT_QUASAR_SOURCES_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'QUASAR_sources.txt'
DEFAULT_HYPERPARAMETERS_FILE = DEFAULT_TEST_SOURCE_LOCATION+DEFAULT_HYPER_FILENAME
DEFAULT_FITS_NO_TESTS = 10

DEFAULT_OUTPUT_FITS_AGN_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'AGN_FITS.png'
DEFAULT_OUTPUT_FITS_BLAZAR_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'BLAZAR_FITS.png'
DEFAULT_OUTPUT_FITS_SEYFERT_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'SEYFERTS_FITS.png'
DEFAULT_OUTPUT_FITS_QUASAR_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'QUASAR_FITS.png'
DEFAULT_OUTPUT_FITS_PULSAR_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'PULSAR_FITS.png'
DEFAULT_PULSARCOORDS_FILE = DEFAULT_PULSAR_SOURCE_LOCATION+'PulsarCoords.txt'

# Offset data for processing PULSAR text files

PULSAR_RA_LOC = 4
PULSAR_DEC_LOC = 5

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
DEFAULT_SEYFERT_CLASS = "SEYFERT"
DEFAULT_QUASAR_CLASS = "QUASAR"
DEFAULT_PULSAR_CLASS = "PULSAR"

AGN_DATA_SELECTED = "A"
SEYFERT_DATA_SELECTED = "S"
BLAZAR_DATA_SELECTED = "B"
QUASAR_DATA_SELECTED = "Q"
PULSAR_DATA_SELECTED = "P"

DEFAULT_CLASSIC_SEYFERT_CLASS = 0
DEFAULT_CLASSIC_BLAZAR_CLASS = 1
DEFAULT_CLASSIC_QUASAR_CLASS = 2
DEFAULT_CLASSIC_AGN_CLASS = 3
DEFAULT_CLASSIC_PULSAR_CLASS = 4


DEFAULT_CLASSIC_MODEL = True
DEFAULT_NN_MODEL = False
# Defaults for CNN Model

DEFAULT_VERBOSE_LEVEL = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_NO_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT_RATE = 0.20
DEFAULT_NO_NEURONS = 100

TRAIN_TEST_RATIO = 0.80  # ratio of the total data for training

SMALL_FONT_SIZE = 8
MEDIUM_FONT_SIZE = 10
BIGGER_FONT_SIZE = 12


bStratifiedSplit = False # use stratified split to create random train/test sets
bScaleInputs = True # use minmax scaler for inputs
bCreateCSVFiles = False # flag set to create all CSV files from FITS images

bCreateSEYFERTFiles = True
bCreateQUASARFiles = True
bCreateBLAZARFiles = True
bCreateAGNFiles = True
bCreatePULSARFiles = True

bCreatePulsarData = False # needed for special PULSAR dataset



bSelectAGN = True
bSelectQUASAR = True
bSelectBLAZAR = False
bSelectSEYFERTS = True
bSelectPULSAR = True


bDebug = False # swich on debug code
bOptimiseHyperParameters = False # optimise hyper parameters used on convolutional model
bTestFITSFile = False # test CNN model with individual FITS files
bTestClassicModels = False # test selection of other models
bTestRandomForestModel = True # test random forest model only
bSaveImageFiles = False # save FITS files for presentations
bTestRandomFiles= True # test individual (random) FITS files
bDisplayProbs = False # display predicted probbilities
bDisplayIndividualPredictions = True

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
    labelDict = {}


    # use label to get one hot encoded value

    ordinalEncoder = OrdinalEncoder()
    oneHotEncoder = OneHotEncoder()

    listOfLabels = np.array(listOfLabels)
    listOfLabels= listOfLabels.reshape(-1,1)
    integerEncoded = ordinalEncoder.fit_transform(listOfLabels)

    print("integer encoded = ",integerEncoded)
    oneHotEncoded = oneHotEncoder.fit_transform(integerEncoded)
    oneHotEncoded = oneHotEncoded.toarray()


    for i in range(len(listOfLabels)):
        labelDict[integerEncoded[i][0]] = listOfLabels[i]


    return oneHotEncoded, labelDict



def ConvertOHE(labelValue,labelDict):

  #  print("OHE = ",labelValue)

    val  = np.argmax(labelValue)
 #   print("integer value = ",val)

  #  print("dict entry = ",labelDict[val])

    return labelDict[val]

def labelDecoder(listOfLabels,labelValue):

    from sklearn.preprocessing import OrdinalEncoder

    # use label to get one hot encoded value

    ordinalEncoder = OrdinalEncoder()

    listOfLabels = np.array(listOfLabels)
    listOfLabels= listOfLabels.reshape(-1,1)
    integerEncoded = ordinalEncoder.fit_transform(listOfLabels)

    label = ordinalEncoder.inverse_transform([[labelValue]])

    return label[0]

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
                if (bDebug):
                    print("Creating File Entry For ", entry.name)

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

def loadPULSARData():


    filePath =  DEFAULT_PULSAR_SOURCES_FILENAME
    bDataValid = True

    if (os.path.isfile(filePath)):
        if (bDebug):
            print("*** Loading PULSAR data  File "+filePath+" ***")
        dataframe = pd.read_csv(filePath, header=None)
        dataReturn = dataframe.values
        if (bDebug):
            print("*** Completed Loading PULSAR File")
    else:
        print("*** PULSAR File Does Not Exist ***")
        bDataValid = False
        dataReturn = []

    return bDataValid,dataReturn


def StoreSourcesToFile(sourceLocation,sourceList):

    if (sourceLocation == DEFAULT_AGN_SOURCE_LOCATION):
        filename = DEFAULT_AGN_SOURCES_FILENAME
        sourceType = DEFAULT_AGN_CLASS
    elif (sourceLocation == DEFAULT_BLAZAR_SOURCE_LOCATION):

        filename = DEFAULT_BLAZAR_SOURCES_FILENAME
        sourceType = DEFAULT_BLAZAR_CLASS

    elif (sourceLocation == DEFAULT_SEYFERT_SOURCE_LOCATION):
        filename = DEFAULT_SEYFERT_SOURCES_FILENAME
        sourceType = DEFAULT_SEYFERT_CLASS
    elif (sourceLocation == DEFAULT_QUASAR_SOURCE_LOCATION):
        filename = DEFAULT_QUASAR_SOURCES_FILENAME
        sourceType = DEFAULT_QUASAR_CLASS
    elif (sourceLocation == DEFAULT_PULSAR_SOURCE_LOCATION):
        filename = DEFAULT_PULSAR_SOURCES_FILENAME
        sourceType = DEFAULT_PULSAR_CLASS


    f = open(filename, "w")
    if (f):
        for source in range(len(sourceList)):

            strr = 'Source: '+sourceType+' : '+str(sourceList[source])+' \n'
            f.write(strr)

        f.close()

def ProcessTransientData(sourceClass):

    trainingData = []

    if (sourceClass == DEFAULT_AGN_CLASS):
        sourceLocation = DEFAULT_AGN_SOURCE_LOCATION
        print("*** Loading AGN Data ***")
    elif (sourceClass == DEFAULT_SEYFERT_CLASS):
        sourceLocation = DEFAULT_SEYFERT_SOURCE_LOCATION
        print("*** Loading SEYFERT Data ***")
    elif (sourceClass == DEFAULT_BLAZAR_CLASS):
        sourceLocation = DEFAULT_BLAZAR_SOURCE_LOCATION
        print("*** Loading BLAZAR Data ***")
    elif (sourceClass == DEFAULT_QUASAR_CLASS):
        sourceLocation = DEFAULT_QUASAR_SOURCE_LOCATION
        print("*** Loading QUASAR Data ***")
    elif (sourceClass == DEFAULT_PULSAR_CLASS):
        sourceLocation = DEFAULT_PULSAR_SOURCE_LOCATION
        print("*** Loading PULSAR Data ***")

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

    print("No of Samples Loaded For "+sourceClass+ " = "+str(len(trainingData)))
    return trainingData



def createLabels(labelList):


    OHELabels,labelDict = createOneHotEncodedSet(labelList)

    return OHELabels,labelDict

def decodeLabels(labelList,predictions):



    label = labelList[np.argmax(predictions)]


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



def CreateTrainingAndTestData(bNNorClassic,labelList,completeTrainingData,trainingDataSizes):


    # create label set and scale all data to be between 0-1

    OHELabels,labelDict = createLabels(labelList)


    datasetLabels = []
    finalTrainingData = []

    #create labels and scale data

    print("no datasets = ",len(completeTrainingData))

    for dataset in range(len(completeTrainingData)):
        print("length of dataset = ",trainingDataSizes[dataset])

        datasetLabels.append(np.asarray(assignLabelSet(OHELabels[dataset], trainingDataSizes[dataset])))

        dataAsArray = np.asarray(completeTrainingData[dataset])
        dataAsArray = np.reshape(dataAsArray, (dataAsArray.shape[0], dataAsArray.shape[2]))

        if (bScaleInputs):
            dataAsArray = ScaleInputData(dataAsArray)

        finalTrainingData.append(dataAsArray)

    print("no of training data samples =",len(finalTrainingData))

    # create the training and test sets

    combinedTrainingSet = []
    combinedTestSet = []
    combinedTrainingLabels = []
    combinedTestLabels = []

    for dataset in range(len(finalTrainingData)):
        classTrainingData = finalTrainingData[dataset]
        classLabels = datasetLabels[dataset]


        print("shape of class data = ",classTrainingData.shape)
        print("shape of class labels = ", classLabels.shape)


        numberVectorsInTrainingSet = int(round(classTrainingData.shape[0] * TRAIN_TEST_RATIO))
        numberVectorsInTestSet = int(round(classTrainingData.shape[0]-numberVectorsInTrainingSet))

        combinedTrainingSet.append(classTrainingData[:numberVectorsInTrainingSet])
        combinedTrainingLabels.append(classLabels[:numberVectorsInTrainingSet])

        combinedTestSet.append(classTrainingData[numberVectorsInTrainingSet:])
        combinedTestLabels.append(classLabels[numberVectorsInTrainingSet:])

        print(" no in training set = ",numberVectorsInTrainingSet)
        print(" no in test set = ", numberVectorsInTestSet)


    print("no of data sets to combine = ",len(combinedTrainingSet))
    # now concatenate all training and test sets to create one combined training and test set

    XTrain = combinedTrainingSet[0]
    XTest = combinedTestSet[0]
    ytrain = combinedTrainingLabels[0]
    ytest = combinedTestLabels[0]

    for dataset in range(1,len(combinedTrainingSet)):

        XTrain = np.concatenate((XTrain, combinedTrainingSet[dataset]))
        XTest = np.concatenate((XTest, combinedTestSet[dataset]))
        ytrain = np.concatenate((ytrain,combinedTrainingLabels[dataset]))
        ytest = np.concatenate((ytest, combinedTestLabels[dataset]))



    # create an integrated set of training data which includes the transient and the random data
    # ensure that the sequence numbers are kept to manage the label data


    print("XTrain shape = ",XTrain.shape)
    print("XTest shape = ", XTest.shape)
    print("ytrain shape = ", ytrain.shape)
    print("ytest shape = ", ytest.shape)


    index = np.random.choice(XTrain.shape[0], len(XTrain), replace=False)

    XTrain = XTrain[index]
    ytrain = ytrain[index]

    index = np.random.choice(XTest.shape[0], len(XTest), replace=False)

    XTest = XTest[index]
    ytest = ytest[index]

    if (bNNorClassic):
        XTrain = np.reshape(XTrain, (XTrain.shape[0], XTrain.shape[1], 1))
        XTest = np.reshape(XTest, (XTest.shape[0], XTest.shape[1], 1))

    #  if (bDebug):
    print("Final Training Data = ", XTrain.shape)
    print("Final Test Data = ", XTest.shape)

    print("Final Training Label Data = ", ytrain.shape)
    print("Final Test Label Data = ", ytest.shape)


    return XTrain, XTest, ytrain, ytest,labelDict



def SaveRandomImageFiles(numberSamples,sourceDir):

    if (sourceDir == DEFAULT_AGN_SOURCE_LOCATION):
        filename = DEFAULT_OUTPUT_FITS_AGN_FILENAME
        fileTitle = DEFAULT_AGN_CLASS+SOURCE_TITLE_TEXT
    elif (sourceDir == DEFAULT_BLAZAR_SOURCE_LOCATION):
        filename = DEFAULT_OUTPUT_FITS_BLAZAR_FILENAME
        fileTitle = DEFAULT_BLAZAR_CLASS+SOURCE_TITLE_TEXT
    elif (sourceDir == DEFAULT_SEYFERT_SOURCE_LOCATION):
        filename = DEFAULT_OUTPUT_FITS_SEYFERT_FILENAME
        fileTitle = DEFAULT_SEYFERT_CLASS + SOURCE_TITLE_TEXT
    elif (sourceDir == DEFAULT_QUASAR_SOURCE_LOCATION):
        filename = DEFAULT_OUTPUT_FITS_QUASAR_FILENAME
        fileTitle = DEFAULT_QUASAR_CLASS + SOURCE_TITLE_TEXT
    elif (sourceDir == DEFAULT_PULSAR_SOURCE_LOCATION):
        filename = DEFAULT_OUTPUT_FITS_PULSAR_FILENAME
        fileTitle = DEFAULT_PULSAR_CLASS + SOURCE_TITLE_TEXT

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


def GetSelectedDataSets():

    classLabel= []
    bCorrectInput=False

    choiceList = ["AGN(A)","SEYFERT(S)","BLAZAR(B)", "QUASAR(Q)", "PULSAR (P)"]


    while (bCorrectInput == False):
        numberClasses= int(input("Number of Classes : "))
        if (numberClasses <=1) or (numberClasses > len(choiceList)):
            print("*** Invalid Selection - Enter again... ***")
        else:
            print("*** Number Classes Chosen = " + str(numberClasses) + " ***")
            bCorrectInput = True

    for i in range(numberClasses):
        bCorrectInput = False
        while (bCorrectInput==False):
            strr = 'Select '+choiceList[0]+', '+choiceList[1]+', '+choiceList[2]+', '+choiceList[3]+', '+choiceList[4]+' : '
            classData = input(strr)
            classData= classData.upper()
            if (classData == AGN_DATA_SELECTED):
                if (DEFAULT_AGN_CLASS) in classLabel:
                    print("*** Cannot Choose Same Datasets For Classification ***")
                else:
                    classLabel.append(DEFAULT_AGN_CLASS)
                    bCorrectInput = True
            elif (classData == SEYFERT_DATA_SELECTED):
                if (DEFAULT_SEYFERT_CLASS) in classLabel:
                    print("*** Cannot Choose Same Datasets For Classification ***")
                else:
                    classLabel.append(DEFAULT_SEYFERT_CLASS)
                    bCorrectInput = True
            elif (classData == BLAZAR_DATA_SELECTED):
                if (DEFAULT_BLAZAR_CLASS) in classLabel:
                    print("*** Cannot Choose Same Datasets For Classification ***")
                else:
                    classLabel.append(DEFAULT_BLAZAR_CLASS)
                    bCorrectInput = True
            elif (classData == QUASAR_DATA_SELECTED):
                if (DEFAULT_QUASAR_CLASS) in classLabel:
                    print("*** Cannot Choose Same Datasets For Classification ***")
                else:
                    classLabel.append(DEFAULT_QUASAR_CLASS)
                    bCorrectInput = True

            elif (classData == PULSAR_DATA_SELECTED):
                if (DEFAULT_PULSAR_CLASS) in classLabel:
                    print("*** Cannot Choose Same Datasets For Classification ***")
                else:
                    classLabel.append(DEFAULT_PULSAR_CLASS)
                    bCorrectInput = True
            else:
                bCorrectInput = False



    return classLabel


def TestRandomFITSFiles(numberFiles,labelList,model,XTest,ytest,labelDict):

    # test model with specific (but random) files
    print("*** Testing "+str(numberFiles)+ " Random FITS Files ***")

    totalnoIncorrect = 0

    for dataset in range(numberFiles):

        randomEntry = int(random.random() * len(XTest))

        RandomSample = XTest[randomEntry].reshape(1, -1)
        correctLabel = ytest[randomEntry]

        y_pred = model.predict(RandomSample)

        if (bDisplayProbs):
            pred_proba = model.predict_proba(RandomSample)

        noInCorrectValues = 0

        for i in range (len(correctLabel)):
            if (y_pred[0][i] != correctLabel[i]):
                noInCorrectValues += 1

        labelText = ConvertOHE(correctLabel, labelDict)
        if (noInCorrectValues == 0):
            if (bDisplayIndividualPredictions):
                print("correct prediction for test number "+str(dataset)+ "= "+labelText[0]+" !!")

        else:
            if (bDisplayIndividualPredictions):
                print("incorrect prediction for test number " + str(dataset) + "= " + labelText[0] + " !!")
            totalnoIncorrect += 1



    print("Total No Correct Predictions = ",numberFiles-totalnoIncorrect)
    print("Total No Incorrect Predictions = ", totalnoIncorrect)

def ProcessPulsarData():


    bValidData, pulsarData = loadPULSARData()

    if (bValidData):


        pulsarCoord = []


        numberPulsars = len(pulsarData)
        print("no of pulsars = ",numberPulsars)

        for pulsar in range(len(pulsarData)):
            skyCoord = []

            pulsarValues = pulsarData[[pulsar]]
            splitString = pulsarValues[0][0].split()
        #    print(splitString)

            Text1 = splitString[0]
            Text2 = splitString[1]
            Text3 = splitString[2]
            Text4 = splitString[3]

            print("Text1 = ",Text1)
            print("Text2 = ",Text2)
            print("Text3 = ",Text3)
            print("Text4 = ",Text4)


            RA = float(splitString[PULSAR_RA_LOC])
            DEC = float(splitString[PULSAR_DEC_LOC])

            print("pulsar = ",pulsar)
            print("RA = ",RA)
            print("DEC = ", DEC)


            skyCoord.append(Text1)
            skyCoord.append(Text2)
            skyCoord.append(Text3)
            skyCoord.append(Text4)


            skyCoord.append(RA)
            skyCoord.append(DEC)


            pulsarCoord.append((skyCoord))

    return bValidData, pulsarCoord



def StorePulsarData(pulsarCoord):


    f = open(DEFAULT_PULSARCOORDS_FILE, "w")
    if (f):
        f.write('pulsarData = [')
        f.write('\n')

        for i in range(len(pulsarCoord)):
            RADec = pulsarCoord[i]


            f.write('("')
            f.write(RADec[0])
            f.write('","')
            f.write(RADec[1])
            f.write('","')
            f.write(RADec[2])
            f.write('","')
            f.write(RADec[3])
            f.write('",')
            f.write(str(RADec[4]))
            f.write(',')
            f.write(str(RADec[5]))
            f.write('),')
            f.write('\n')

            f.write(']')
            f.write('\n')
            f.write('\n')
            f.write('\n')

            f.write('pulsarRA = [')
            f.write('\n')

        for i in range(len(pulsarCoord)):
            RADec = pulsarCoord[i]


            f.write('(')
            f.write(str(RADec[4]))
            f.write('),')
            f.write('\n')

            f.write(']')
            f.write('\n')

            f.write('\n')
            f.write('\n')

            f.write('pulsarDEC = [')
            f.write('\n')

        for i in range(len(pulsarCoord)):
            RADec = pulsarCoord[i]

            f.write('(')
            f.write(str(RADec[5]))
            f.write('),')
            f.write('\n')

            f.write(']')
            f.write('\n')



        f.close()

def main():

    if (bCreatePulsarData):

        bValidData, pulsarCoord = ProcessPulsarData()

        if (bValidData):

            StorePulsarData(pulsarCoord)

    if (bCreateCSVFiles):


        if (bCreateAGNFiles):

            print("*** Processing All AGN Files ***")
            sourceAGNList = ScanForSources(DEFAULT_AGN_SOURCE_LOCATION)
            agnSourceFileDict = CreateAllCSVFiles(DEFAULT_AGN_SOURCE_LOCATION,sourceAGNList)

            ProcessAllCSVFiles(DEFAULT_AGN_SOURCE_LOCATION,agnSourceFileDict,sourceAGNList)

        if (bCreatePULSARFiles):
            print("*** Processing All PULSAR Files ***")
            sourcePULSARList = ScanForSources(DEFAULT_PULSAR_SOURCE_LOCATION)
            pulsarSourceFileDict = CreateAllCSVFiles(DEFAULT_PULSAR_SOURCE_LOCATION, sourcePULSARList)

            ProcessAllCSVFiles(DEFAULT_PULSAR_SOURCE_LOCATION, pulsarSourceFileDict, sourcePULSARList)

        if (bCreateBLAZARFiles):
            print("*** Processing All BLAZAR Files ***")

            sourceBLAZARList = ScanForSources(DEFAULT_BLAZAR_SOURCE_LOCATION)
            blazarSourceFileDict = CreateAllCSVFiles(DEFAULT_BLAZAR_SOURCE_LOCATION, sourceBLAZARList)

            ProcessAllCSVFiles(DEFAULT_BLAZAR_SOURCE_LOCATION,blazarSourceFileDict, sourceBLAZARList)

        if (bCreateSEYFERTFiles):
            print("*** Processing All SEYFERT Files ***")

            sourceSEYFERTList = ScanForSources(DEFAULT_SEYFERT_SOURCE_LOCATION)
            seyfertSourceFileDict = CreateAllCSVFiles(DEFAULT_SEYFERT_SOURCE_LOCATION, sourceSEYFERTList)

            ProcessAllCSVFiles(DEFAULT_SEYFERT_SOURCE_LOCATION, seyfertSourceFileDict, sourceSEYFERTList)

        if (bCreateQUASARFiles):
            print("*** Processing All QUASAR Files ***")

            sourceQUASARList = ScanForSources(DEFAULT_QUASAR_SOURCE_LOCATION)
            quasarSourceFileDict = CreateAllCSVFiles(DEFAULT_QUASAR_SOURCE_LOCATION, sourceQUASARList)

            ProcessAllCSVFiles(DEFAULT_QUASAR_SOURCE_LOCATION, quasarSourceFileDict, sourceQUASARList)
    else:


        # now process all images per chosen datasets

        labelList = GetSelectedDataSets()
        print("Datasets to be classified are ",labelList)

        completeTrainingData = []
        trainingDataSizes = []

        print("*** Loading Training Data ***")

        for classes in range(len(labelList)):

            trainingData = ProcessTransientData(labelList[classes])
            trainingDataSizes.append(len(trainingData))
            completeTrainingData.append(trainingData)

        print("*** Creating Training and Test Data Sets ***")
        if (bTestClassicModels):

            XTrain, XTest, ytrain, ytest,labelDict = CreateTrainingAndTestData(False,labelList,completeTrainingData,trainingDataSizes)

            print("*** Evaluating Multiple Classic Models ***")
            MultipleClassicModels(XTrain, ytrain, XTest, ytest)


        if (bTestRandomForestModel):
                XTrain, XTest, ytrain, ytest,labelDict = CreateTrainingAndTestData(False,labelList, completeTrainingData,trainingDataSizes)

                print("*** Evaluating Random Forest Model ***")
                randomForestModel = RandomForestModel(XTrain, ytrain, XTest, ytest)


                if (bTestRandomFiles):

                    TestRandomFITSFiles(100*DEFAULT_FITS_NO_TESTS,labelList,randomForestModel,XTest, ytest,labelDict)

        else:

            XTrain, XTest, ytrain, ytest,labelDict = CreateTrainingAndTestData(True,labelList, completeTrainingData,trainingDataSizes)

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


if __name__ == '__main__':
    main()
