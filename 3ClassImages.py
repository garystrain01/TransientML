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
import astroquery
from sklearn.model_selection import StratifiedShuffleSplit


DEFAULT_IMAGE_LOCATION = '/Volumes/ExtraDisk/AGN/'
DEFAULT_NVVS_DATA_ROOT   = '/Volumes/ExtraDisk/NVSS_DATA/'
DEFAULT_VAST_DATA_ROOT   = '/Volumes/ExtraDisk/VAST_DATA/'


DEFAULT_AGN_SOURCE_LOCATION = 'AGN/'
DEFAULT_PULSAR_SOURCE_LOCATION = 'PULSAR/'
DEFAULT_BLAZAR_SOURCE_LOCATION = 'BLAZAR/'
DEFAULT_SEYFERT_SOURCE_LOCATION = 'SEYFERT/'
DEFAULT_QUASAR_SOURCE_LOCATION = 'QUASAR/'
DEFAULT_TEST_SOURCE_LOCATION = 'TEST_DATA/'
DEFAULT_SOURCES_SOURCE_LOCATION = 'SOURCES/'

DEFAULT_NVSS_DATASET =  "NVSS"
DEFAULT_VAST_DATASET = "VAST"

NVSS_SHORT_NAME = 'N'
VAST_SHORT_NAME = 'V'

DEFAULT_EXISTING_MODEL_LOCATION = '/Volumes/ExtraDisk/EXISTING_MODELS/'

DEFAULT_HYPER_FILENAME = 'CNNHyper.txt'
DEFAULT_PULSAR_SOURCES_FILENAME = DEFAULT_PULSAR_SOURCE_LOCATION+'ATNF.txt'

DEFAULT_NVSS_CATALOG_FILENAME = '/Volumes/ExtraDisk/NVSS/CATALOG.FIT'

DEFAULT_AGN_SOURCES_FILENAME = DEFAULT_SOURCES_SOURCE_LOCATION+'AGN_sources.txt'
DEFAULT_BLAZAR_SOURCES_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'BLAZAR_sources.txt'
DEFAULT_SEYFERT_SOURCES_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'SEYFERT_sources.txt'
DEFAULT_QUASAR_SOURCES_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'QUASAR_sources.txt'
DEFAULT_PULSAR_SOURCES_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'PULSAR_sources.txt'

DEFAULT_HYPERPARAMETERS_FILE = DEFAULT_TEST_SOURCE_LOCATION+DEFAULT_HYPER_FILENAME

DEFAULT_FITS_NO_TESTS = 10

DEFAULT_OUTPUT_FITS_AGN_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'AGN_FITS.png'
DEFAULT_OUTPUT_FITS_BLAZAR_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'BLAZAR_FITS.png'
DEFAULT_OUTPUT_FITS_SEYFERT_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'SEYFERTS_FITS.png'
DEFAULT_OUTPUT_FITS_QUASAR_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'QUASAR_FITS.png'
DEFAULT_OUTPUT_FITS_PULSAR_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'PULSAR_FITS.png'
DEFAULT_PULSARCOORDS_FILE = DEFAULT_PULSAR_SOURCE_LOCATION+'PulsarCoords.txt'
DEFAULT_STACKED_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'StackedImages.png'
DEFAULT_DUPLICATES_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'DuplicateSources.txt'
DEFAULT_MODEL_FILE_LOCATION = DEFAULT_TEST_SOURCE_LOCATION
MODEL_FILENAME_EXTENSION = '.pkl'
DEFAULT_VIZIER_SOURCES_FILENAME = DEFAULT_TEST_SOURCE_LOCATION+'VizierSources.txt'

DEFAULT_NVSS_SOURCE_LOCATION = '/Volumes/ExtraDisk/NVSS/'
DEFAULT_NVSS_SOURCES_FILENAME = DEFAULT_NVSS_SOURCE_LOCATION+'NVSSDetections.txt'
DEFAULT_NVSS_PULSAR_SOURCES_FILENAME = DEFAULT_NVSS_SOURCE_LOCATION+'NVSSPulsarsFinal.txt'
DEFAULT_NVSS_SEYFERT_SOURCES_FILENAME = DEFAULT_NVSS_SOURCE_LOCATION+'NVSSSeyfertsFinal.txt'
DEFAULT_NVSS_BLAZARS_SOURCES_FILENAME = DEFAULT_NVSS_SOURCE_LOCATION+'NVSSBlazarsFinal.txt'
DEFAULT_NVSS_QUASARS_SOURCES_FILENAME = DEFAULT_NVSS_SOURCE_LOCATION+'NVSSQuasarsFinal.txt'

DEFAULT_NVSS_CATALOG_LOCATION = '/Volumes/ExtraDisk/NVSS_CATALOG/'
NVSS_CATALOG_IMAGE_LOCATION = '/Volumes/ExtraDisk/NVSS_CATALOG/NVSS_IMAGES/'
NVSS_CATALOG_IMAGE_FILENAME = NVSS_CATALOG_IMAGE_LOCATION +'NVSS_CATALOG_IMAGE_'
NVSS_CATALOG_DETECTIONS_FILENAME = DEFAULT_NVSS_CATALOG_LOCATION+'NVSS_CATALOG_DETECTIONS.txt'

NVSS_CATALOG_CSV_LOCATION = '/Volumes/ExtraDisk/NVSS_CATALOG/NVSS_CSVFILES/'
NVSS_CATALOG_CSV_FILENAME = NVSS_CATALOG_CSV_LOCATION+'NVSS_CSV_IMAGE_'

DEFAULT_NVSS_CATALOG_FILE = DEFAULT_NVSS_CATALOG_LOCATION+'NVSSCatalog.text'
FINAL_NVSS_CATALOG_FILE = DEFAULT_NVSS_CATALOG_LOCATION+'NVSS_NewCatalog.txt'
FINAL_SELECTED_NVSS_SOURCES_LOCATION = DEFAULT_NVSS_SOURCE_LOCATION+'NVSS_RandomSamples.txt'

DEFAULT_VIZIER_PULSAR_SOURCES_FILENAME = DEFAULT_NVSS_SOURCE_LOCATION+'PulsarSources.txt'
DEFAULT_NVSS_FITS_FILENAMES = DEFAULT_TEST_SOURCE_LOCATION+'NVSS_FITS_FILENAMES.txt'

DEFAULT_RANDOM_SAMPLES_DIR = 'RANDOM_SAMPLES/'
DEFAULT_RANDOM_NVSS_SOURCE_LOCATION = DEFAULT_NVSS_SOURCE_LOCATION+ DEFAULT_RANDOM_SAMPLES_DIR

# Offset data for processing PULSAR text files

PULSAR_RA_LOC = 4
PULSAR_DEC_LOC = 5

MAX_NVSS_RA_DEC_STRINGS = 6
NVSS_CAT_RA_POS = 0
NVSS_CAT_DEC_POS = 3
NVSS_FIELD_ID_POS = 11

DEFAULT_MAX_NO_RANDOM_NVSS = 1000
DEFAULT_MAX_NO_NVSS_ENTRIES = 1773484


FAILED_TO_OPEN_FILE = -1
FITS_FILE_EXTENSION = '.fits'
TEXT_FILE_EXTENSION = '.txt'
DS_STORE_FILENAME = '.'
DEFAULT_CSV_DIR = 'CSVFiles'
FOLDER_IDENTIFIER = '/'
SOURCE_TITLE_TEXT = ' Source :'
UNDERSCORE = '_'
DEFAULT_CSV_FILETYPE = UNDERSCORE+'data.txt'
XSIZE_FITS_IMAGE= 120
YSIZE_FITS_IMAGE = 120
MAX_NUMBER_RANDOM_SAMPLES = 10 # when random samples independently tested

DEFAULT_NO_INDEPENDENT_TESTS = 10


XSIZE_SMALL_FITS_IMAGE = 20
YSIZE_SMALL_FITS_IMAGE = 20

DEFAULT_AGN_CLASS = "AGN"
DEFAULT_BLAZAR_CLASS = "BLAZAR"
DEFAULT_SEYFERT_CLASS = "SEYFERT"
DEFAULT_QUASAR_CLASS = "QUASAR"
DEFAULT_PULSAR_CLASS = "PULSAR"
DEFAULT_TEST_CLASS = "TEST"

AGN_DATA_SELECTED = "A"
SEYFERT_DATA_SELECTED = "S"
BLAZAR_DATA_SELECTED = "B"
QUASAR_DATA_SELECTED = "Q"
PULSAR_DATA_SELECTED = "P"

NVSS_AGN_DATA_SELECTED = "NA"
NVSS_SEYFERT_DATA_SELECTED = "NS"
NVSS_BLAZAR_DATA_SELECTED = "NB"
NVSS_QUASAR_DATA_SELECTED = "NQ"
NVSS_PULSAR_DATA_SELECTED = "NP"


DEFAULT_CLASSIC_SEYFERT_CLASS = 0
DEFAULT_CLASSIC_BLAZAR_CLASS = 1
DEFAULT_CLASSIC_QUASAR_CLASS = 2
DEFAULT_CLASSIC_AGN_CLASS = 3
DEFAULT_CLASSIC_PULSAR_CLASS = 4

DEFAULT_CLASSIC_NVSS_AGN_CLASS = 5


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


OP_BUILD_MODEL = 'B'
OP_TEST_MODEL = 'T'

OP_MODEL_RANDOM= 'R'
OP_MODEL_CNN = 'C'

DEFAULT_DICT_TYPE = '_dict.txt'

bStratifiedSplit = False # use stratified split to create random train/test sets
bScaleInputs = True # use minmax scaler for inputs

bCreateVASTFiles = True # process all VAST related FITS Images
bCreateNVSSFiles = False # process all NVSS related FITS Images

bCreateSEYFERTFiles = True
bCreateQUASARFiles = True
bCreateBLAZARFiles = True
bCreateAGNFiles = True
bCreatePULSARFiles = True
bCreateTESTFiles = True




bCreatePulsarData = False # needed for special PULSAR dataset
bAstroqueryAGN = False # to access astroquery AGN data
bAstroqueryPULSAR = False # to access astroquery PULSAR data
bAstroqueryBLAZAR = False # to access astroquery BLAZAR data
bAstroqueryQUASAR = False # to access astroquery QUASAR data
bAstroquerySEYFERT = False # to access astroquery SEYFERT data


bSelectAGN = True
bSelectQUASAR = True
bSelectBLAZAR = True
bSelectSEYFERTS = True
bSelectPULSAR = True

bSelectNVSS_AGN = False
bSelectNVSS_QUASAR = False
bSelectNVSS_BLAZAR = False
bSelectNVSS_SEYFERTS = False
bSelectNVSS_PULSAR = False

bCreateNVSSCatalog = False # for inital pre-procesing of NVSS Catalog
bTestNVSSCatalog = False # for testing againt NVSS catalogue
bTestNVSSFiles = False # for testing againt a select set of NVSS FITS files
bTestIndSamples = False # test some independent samples
bDataCreation = False # for general data creation functions
bCreateCSVFiles = False # flag set to create all CSV files from FITS images

bDebug = False # swich on debug code
bOptimiseHyperParameters = False # optimise hyper parameters used on convolutional model
bTestFITSFile = False # test CNN model with individual FITS files
bTestClassicModels = False # test selection of other models
bTestRandomForestModel = False # test random forest model only
bSaveImageFiles = False # save FITS files for presentations
bTestRandomFiles= False # test individual (random) FITS files
bDisplayProbs = False # display predicted probabilities
bDisplayIndividualPredictions = True
bStackImages = False # stack all source images
bShrinkImages = False  # to experiment with smalleer FITS images
bRandomSelectData = True # set false if you want to eliminate randomizing of training and test data
bCollectRandomSamples = False # for extra tests of random samples
bCollectRawRandomSamples = False # for extra tests of raw random samples
bSaveModel = True # save model for future use
bAccessNVSS = False # access NVSS data
bCheckForDuplicates = False # check if we have duplicate sources across classes
bCheckFITSIntegrity = False # check if randomly selected FITS files are identical (or not)
bGetSkyview = False # download skyview images
bProcessNVSSCatalog = False # edit NVSS Catalog
bSelectFromNVSSCatalog = False # do a random selection test from NVSS catalog
bSelectRandomNVSS = False # Do a random selection from NVSS Catalog (or not)
bStoreSourcesToFile = False # retain source list as processing
bCheckTrainingAndTestSet = False # check that training and test sets don't overlap
bCheckDupRandomSamples = False # check that random samples are not in test or training set

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

    oneHotEncoded = oneHotEncoder.fit_transform(integerEncoded)
    oneHotEncoded = oneHotEncoded.toarray()


    for i in range(len(listOfLabels)):
        labelDict[integerEncoded[i][0]] = listOfLabels[i]


    return oneHotEncoded, labelDict



def ConvertOHE(labelValue,labelDict):

    val  = np.argmax(labelValue)

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

    return normalised,scaler

def StoreNVSSFITSImage(transientType,sourceDir,sourceName,fitsImage,FITSFileNumber):

    import os

    try:
        sourceDir = sourceDir+FOLDER_IDENTIFIER+sourceName

        print("NVSS source dir = ",sourceDir)

        print("Creating Source Directory = ",sourceDir)

        os.mkdir(sourceDir)

        filename = sourceDir + FOLDER_IDENTIFIER + transientType+ '_' + str(FITSFileNumber) + FITS_FILE_EXTENSION

        print(filename)

        fitsImage.writeto(filename)
    except:

        print("Failed in StoreNVSSFITSImage")

def StoreNVSSCatalogImage(fitsImage,sourceNumber):

    bValidData=True

    try:
        imageFileName = NVSS_CATALOG_IMAGE_FILENAME+str(sourceNumber)+FITS_FILE_EXTENSION

        print("NVSS image file = ",imageFileName)

        fitsImage.writeto(imageFileName)

    except:

        bValidData=False
        print("Failed in StoreNVSSCatalogImage")


    return bValidData,imageFileName

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



def ReshapeFITSImage(imageData,newXSize,newYSize):

    currentXSize = imageData.shape[0]
    currentYSize = imageData.shape[1]

    newImageData = np.zeros((newXSize,newYSize))

    for x in range(XSIZE_FITS_IMAGE):
        for y in range(YSIZE_FITS_IMAGE):
            srcX = int(((currentXSize / 2) - (XSIZE_FITS_IMAGE / 2)) + x)
            srcY = int(((currentYSize / 2) - (YSIZE_FITS_IMAGE / 2)) + y)

            newImageData[x][y] = imageData[srcX][srcY]

    return newImageData

def OpenFITSFile(filename):

    bValidData = True

    if (filename):

        hdul = fits.open(filename)

        imageData = hdul[0].data

        if (imageData.shape[0] > XSIZE_FITS_IMAGE) or (imageData.shape[1] > YSIZE_FITS_IMAGE):
            # reshape the image - deal with NVSS images

            imageData = ReshapeFITSImage(imageData,XSIZE_FITS_IMAGE,YSIZE_FITS_IMAGE)


        elif ((imageData.shape[0] != XSIZE_FITS_IMAGE) or (imageData.shape[1] != YSIZE_FITS_IMAGE)):
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
    bValidData=False

    print("image location = ",imageLocation)

    fileList = os.scandir(imageLocation)
    for entry in fileList:
        if entry.is_dir():
            if (bDebug):
                print("dir = ",entry.name)

        elif entry.is_file():
            if entry.name[0] != DS_STORE_FILENAME :
                imageList.append(entry.name)
                bValidData = True
                if (bDebug):
                    print("entry is file ",entry.name)
            else:
                if (bDebug):
                    print("File Entry Ignored For ",entry.name)

    return bValidData, imageList

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
                        print("Success in creating csv file",sourceCSVFileName)
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

def StoreIndividualCSVFile(fitsImageFile,source):

    bResultOK= False

    bValidData,imageData = OpenFITSFile(fitsImageFile)
    if (bValidData):
        # ok - found the image data, now store in the correct CSV file
        csvFilename = NVSS_CATALOG_CSV_FILENAME + str(source) + TEXT_FILE_EXTENSION
        print("csvFilename = ",csvFilename)
        f = open(csvFilename,"w")
        if (f):
            bResultOK = True
            StoreImageContents(f, imageData)

    return bResultOK,csvFilename

def StoreTestCSVFile(fitsImageFile):

    bResultOK= False

    bValidData,imageData = OpenFITSFile(DEFAULT_TEST_NVSS_SOURCE_LOCATION+fitsImageFile)
    if (bValidData):
        # ok - found the image data, now store in the correct CSV file
        csvFilename = fitsImageFile.split('.')
        csvFilename = DEFAULT_TEST_NVSS_CSV_SOURCE_LOCATION+csvFilename[0]+TEXT_FILE_EXTENSION

        f = open(csvFilename,"w")
        if (f):
            bResultOK = True
            StoreImageContents(f, imageData)

    return bResultOK,csvFilename



def GetFITSFile(imageLocation,imageName):

    imageLocation += FOLDER_IDENTIFIER

    bValidData,imageData = OpenFITSFile(imageLocation+imageName)

    return bValidData, imageData


def ProcessAllCSVFiles(sourceDir,fileHandleDict,sourceList):

    totalNumberFilesProcessed = 0

    for source in sourceList:
        # get list of all files for this source
        fileNumber = 0
        imageLocation,imageList = ScanForImages(sourceDir,source)

        if (len(imageList) >0):
            for image in imageList:
                totalNumberFilesProcessed += 1
                imageCSVFile = source+UNDERSCORE+str(fileNumber)
                f = fileHandleDict[imageCSVFile]

                fileNumber += 1
                StoreInCSVFile(imageLocation, image,f)
                f.close()

    return  totalNumberFilesProcessed

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


def StoreSourcesToFile(rootData,sourceLocation,sourceList):

    if (sourceLocation == DEFAULT_AGN_SOURCE_LOCATION):
        filename = DEFAULT_AGN_SOURCES_FILENAME
        sourceType = DEFAULT_AGN_CLASS
    elif (sourceLocation == DEFAULT_NVSS_AGN_SOURCE_LOCATION):
        filename = DEFAULT_NVSS_AGN_SOURCES_FILENAME
        sourceType = DEFAULT_NVSS_AGN_CLASS
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

def ProcessTransientData(dataSet,sourceClass):

    trainingData = []

    if (dataSet == DEFAULT_NVSS_DATASET):
        print("Processing NVSS Data")
        # we're doing NVSS data
        rootData = DEFAULT_NVVS_DATA_ROOT
    else:
        # we're doing VAST data
        print("Processing VAST Data")
        rootData = DEFAULT_VAST_DATA_ROOT

    if (sourceClass == DEFAULT_TEST_CLASS):
        sourceLocation = rootData+DEFAULT_TEST_SOURCE_LOCATION
        print("*** Loading TEST Data From "+sourceLocation+ "***")

    else:

        if (sourceClass == DEFAULT_AGN_CLASS):
            sourceLocation = rootData + DEFAULT_AGN_SOURCE_LOCATION
            print("*** Loading AGN Data ***")
        elif (sourceClass == DEFAULT_SEYFERT_CLASS):
            sourceLocation = rootData+DEFAULT_SEYFERT_SOURCE_LOCATION
            print("*** Loading SEYFERT Data ***")
        elif (sourceClass == DEFAULT_BLAZAR_CLASS):
            sourceLocation = rootData+DEFAULT_BLAZAR_SOURCE_LOCATION
            print("*** Loading BLAZAR Data ***")
        elif (sourceClass == DEFAULT_QUASAR_CLASS):
            sourceLocation = rootData+DEFAULT_QUASAR_SOURCE_LOCATION
            print("*** Loading QUASAR Data ***")
        elif (sourceClass == DEFAULT_PULSAR_CLASS):
            sourceLocation = rootData+DEFAULT_PULSAR_SOURCE_LOCATION
            print("*** Loading PULSAR Data ***")

    sourceList = ScanForSources(sourceLocation)

    if (bStoreSourcesToFile):
        StoreSourcesToFile(rootData,sourceLocation,sourceList)

    numberSources = len(sourceList)

    for source in range(numberSources):

        imageLocation, imageList = ScanForImages(sourceLocation, sourceList[source])

        for imageNo in range(len(imageList)):

            bValidData, sourceData = loadCSVFile(sourceLocation, sourceList[source], imageNo)
            if (bValidData):

                if (bShrinkImages):
                    sourceData = np.reshape(sourceData, (1, XSIZE_SMALL_FITS_IMAGE * YSIZE_SMALL_FITS_IMAGE))
                else:
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

def TransformTrainingData(trainingData):

    dataAsArray = np.asarray(trainingData)

    dataAsArray = np.reshape(dataAsArray, (dataAsArray.shape[0], dataAsArray.shape[2]))

    if (bScaleInputs):
        dataAsArray,scaler = ScaleInputData(dataAsArray)

    return dataAsArray,scaler


def CreateTrainingAndTestData(bNNorClassic,labelList,completeTrainingData,trainingDataSizes, randomTrainingData):


    # create label set and scale all data to be between 0-1

    OHELabels,labelDict = createLabels(labelList)

    datasetLabels = []
    finalTrainingData = []

    #create labels and scale data

    print("no datasets = ",len(completeTrainingData))
    randomSelectedSamples = []
    randomSelectedSampleLabels = []

    numberSamplesCollected = 0

    print("no of datasets = ",len(completeTrainingData))

    dataScaler = []

    for dataset in range(len(completeTrainingData)):

        print("length of dataset = ",trainingDataSizes[dataset])

        dataAsArray,scaler = TransformTrainingData(completeTrainingData[dataset])

        dataScaler.append(scaler)

        datasetLabels.append(np.asarray(assignLabelSet(OHELabels[dataset], trainingDataSizes[dataset])))

        finalTrainingData.append(dataAsArray)

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

    if (bRandomSelectData):

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

    if (bCollectRandomSamples):

        for entry in range(MAX_NUMBER_RANDOM_SAMPLES):

            XChoice = int(random.random()*len(XTest))

            randomSample = XTest[XChoice]
            randomLabel = ytest[XChoice]

            randomSelectedSamples.append(randomSample)

            randomSelectedSampleLabels.append(randomLabel)

            XTest = np.delete(XTest,XChoice,axis=0)
            ytest = np.delete(ytest, XChoice,axis=0)


        print("final XTest shape = ",XTest.shape)
        print("final ytest shape = ", ytest.shape)


    return XTrain, XTest, ytrain, ytest,labelDict,randomSelectedSamples,randomSelectedSampleLabels


def CheckTrainingAndTestSet(XTrain,XTest):

    print("Checking Train and Test Set")

    numberInTestSet = len(XTest)
    numberInTrainSet = len(XTrain)

    numberDuplicates = 0

    for entry in range(numberInTestSet):
        for trainSample in range(numberInTrainSet):

            comparison = (XTest[entry]== XTrain[trainSample])

            if (comparison.all() == True):

                print("DUPLICATE !!")
                numberDuplicates += 1

    print("Total No Of Duplicates = ",numberDuplicates)

def CheckDupRandomSamples(XTrain,XTest,randomSamples):

    print("Checking Random Samples In Train and Test Set")

    numberInTestSet = len(XTest)
    numberInTrainSet = len(XTrain)
    numberSamples = len(randomSamples)

    print("No of samples = ",numberSamples)
    numberDuplicates = 0

    for sample in range(numberSamples):
        for entry in range(numberInTestSet):

            comparison = (XTest[entry]== randomSamples[sample])
            if (comparison.all() == True):

                print("DUPLICATE IN TEST SET !!")
                numberDuplicates += 1

    for sample in range(numberSamples):
        for entry in range(numberInTrainSet):

            comparison = (XTrain[entry] == randomSamples[sample])
            if (comparison.all() == True):
                print("DUPLICATE IN TRAIN SET !!")
                numberDuplicates += 1

    print("Total No Of Duplicates = ",numberDuplicates)



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

    classLabels= []
    bCorrectInput=False

    datasetChoice = ['NVSS','VAST']
    shortDataSetChoice = [NVSS_SHORT_NAME,VAST_SHORT_NAME]

    choiceList = ["AGN(A)","SEYFERT(S)","BLAZAR(B)", "QUASAR(Q)", "PULSAR (P)"]

    while (bCorrectInput == False):
        dataSet = input('Choose Dataset For Model '+datasetChoice[0]+','+datasetChoice[1]+'  ('+datasetChoice[0][0]+'/'+datasetChoice[1][0]+') :')
        dataSet = dataSet.upper()
        if (dataSet not in shortDataSetChoice):
            print("*** Invalid Selection - Enter again... ***")
        else:
            print("*** Dataset Chosen = " + dataSet + " ***")

            if (dataSet == NVSS_SHORT_NAME):
                dataClass = DEFAULT_NVSS_DATASET
            else:
                dataClass = DEFAULT_VAST_DATASET
            bCorrectInput = True

    bCorrectInput = False
    while (bCorrectInput == False):
        numberClasses= int(input("Number of Classes : "))
        if (numberClasses < 2) or (numberClasses > len(choiceList)):
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
                if (DEFAULT_AGN_CLASS) in classLabels:
                    print("*** Cannot Choose Same Datasets For Classification ***")
                else:
                    classLabels.append(DEFAULT_AGN_CLASS)
                    bCorrectInput = True
            elif (classData == SEYFERT_DATA_SELECTED):
                if (DEFAULT_SEYFERT_CLASS) in classLabels:
                    print("*** Cannot Choose Same Datasets For Classification ***")
                else:
                    classLabels.append(DEFAULT_SEYFERT_CLASS)
                    bCorrectInput = True
            elif (classData == BLAZAR_DATA_SELECTED):
                if (DEFAULT_BLAZAR_CLASS) in classLabels:
                    print("*** Cannot Choose Same Datasets For Classification ***")
                else:
                    classLabels.append(DEFAULT_BLAZAR_CLASS)
                    bCorrectInput = True
            elif (classData == QUASAR_DATA_SELECTED):
                if (DEFAULT_QUASAR_CLASS) in classLabels:
                    print("*** Cannot Choose Same Datasets For Classification ***")
                else:
                    classLabels.append(DEFAULT_QUASAR_CLASS)
                    bCorrectInput = True

            elif (classData == PULSAR_DATA_SELECTED):
                if (DEFAULT_PULSAR_CLASS) in classLabels:
                    print("*** Cannot Choose Same Datasets For Classification ***")
                else:
                    classLabels.append(DEFAULT_PULSAR_CLASS)
                    bCorrectInput = True
            else:
                bCorrectInput = False

    return dataClass,classLabels

def GetModelType():

    bCorrectInput = False

    while (bCorrectInput == False):
        selectedOperation = input("Select Random Forest (R) or CNN (C) Model : ")
        selectedOperation = selectedOperation.upper()
        if (selectedOperation == OP_MODEL_RANDOM) or (selectedOperation == OP_MODEL_CNN):
            bCorrectInput = True
        else:
            print("*** Incorrect Input - try again ***")

    return selectedOperation


def GetOperationMode():

    bCorrectInput=False

    while (bCorrectInput == False):
        selectedOperation= input("Select Build Model (B) or Test Existing Model (T) : ")
        selectedOperation = selectedOperation.upper()
        if (selectedOperation == OP_BUILD_MODEL) or (selectedOperation == OP_TEST_MODEL):
            bCorrectInput = True
        else:
            print("*** Incorrect Input - try again ***")

    return selectedOperation



def DecodePredictedLabel(labelDict,prediction):

    bDetection=False
    elementNo = 0

    for predictionElement in prediction:
        elementNo += 1

        predictVal =predictionElement>0

        if (predictVal.any() == True):

            bDetection=True
            label = labelDict[np.argmax(predictionElement)]

            print("Predicted Value For Entry "+str(elementNo)+" : "+str(label))



 #   return bDetection,label



def TestIndividualFile(model,fileData):

    fileData = fileData.reshape(1, -1)
    y_pred = model.predict(fileData)
    print("y_pred =", y_pred)

    return y_pred

def TestRawRandomSamples(XTest,XTrain,labelDict,model,randomSamples,sampleLabels):

    rawTestSamples = []

    completeData = np.concatenate((XTrain,XTest))

    # test model with random samples of raw image data

    print(completeData.shape)
    sys.exit()
    dataAsArray = np.asarray(randomSamples)
    sampleLabels = np.asarray(sampleLabels)

    print(dataAsArray.shape)

    dataAsArray = np.reshape(dataAsArray, (dataAsArray.shape[0], dataAsArray.shape[2]))
    print(dataAsArray.shape)

    if (bScaleInputs):
        dataAsArray = ScaleInputData(dataAsArray)
    print(dataAsArray.shape)
    sys.exit()
    for sample in range(len(dataAsArray)):

        rawTestSamples.append(dataAsArray[sample])

        pred = TestIndividualFile(model,dataAsArray[sample])

        DecodePredictedLabel(labelDict, pred)

        print("versus ACTUAL label:",sampleLabels[sample])

    return rawTestSamples,sampleLabels

def TestRandomSamples(labelDict,model,randomSamples,sampleLabels):

    # test model with random samples of raw image data

    print("no of random samples = ",len(randomSamples))
    print("no labels = ",len(sampleLabels))
    print("sample labels = ",sampleLabels)

    for sample in range(len(randomSamples)):

        pred = TestIndividualFile(model,randomSamples[sample])

        DecodePredictedLabel(labelDict, pred)

        print("versus ACTUAL label:",sampleLabels[sample])



def TestIndependentSamples(XTrain,XTest,dataSet,model,labelDict):


    testData = ProcessTransientData(dataSet, DEFAULT_TEST_CLASS)

    dataAsArray = TransformTrainingData(testData)

    for sample in range(len(dataAsArray)):

        pred = TestIndividualFile(model, dataAsArray[sample])

        DecodePredictedLabel(labelDict, pred)

def TestRandomFITSFiles(numberFiles,model,XTest,ytest,labelDict):

    # test model with specific (but random) files

    print("*** Testing "+str(numberFiles)+ " Random FITS Files ***")

    totalnoIncorrect = 0

    for dataset in range(numberFiles):

        randomEntry = int(random.random() * len(XTest))

        RandomSample = XTest[randomEntry].reshape(1, -1)
        print("shape = ",RandomSample.shape)
        correctLabel = ytest[randomEntry]

        y_pred = model.predict(RandomSample)
        print("y_pred=",y_pred)
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



        StorePulsarData(pulsarCoord)


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



def StackTrainingData(allTrainingData):

    stackedTransientImage = []

    numberDatasets = len(allTrainingData)
    print("number datasets to stack = ",numberDatasets)

    for dataset in range(numberDatasets):
        trainingData = allTrainingData[dataset]
        print("size of training data = ",len(trainingData))


        numberImagesToStack = len(trainingData)
        print("no images to stack = ",numberImagesToStack)
        stackedImage = trainingData[0]
        for image in range(1,len(trainingData)):
            stackedImage += trainingData[image]


        stackedImage = stackedImage/numberImagesToStack


        stackedTransientImage.append(stackedImage)


    return stackedTransientImage


def SetPlotParameters():
    plt.rc('axes', titlesize=SMALL_FONT_SIZE)
    plt.rc('axes', labelsize=SMALL_FONT_SIZE)
    plt.rc('xtick', labelsize=SMALL_FONT_SIZE)
    plt.rc('ytick', labelsize=SMALL_FONT_SIZE)

def DisplayStackedImages(stackedData,labelList):

    SetPlotParameters()

   # fig, axs = plt.subplots(len(stackedData))

    numberPlots = len(stackedData)

    fig, axs = plt.subplots(3,2)

    figx = 0
    figy = 0

    for imageNo in range(0, len(stackedData)):
        imageData = stackedData[imageNo]
        x = np.arange(len(imageData[0]))

        axs[figx,figy].scatter(x, imageData[0], marker='+')
        axs[figx,figy].set_title(labelList[imageNo])
        axs[figx,figy].scatter(x, imageData[0], marker='+')
        axs[figx,figy].tick_params(axis='x', labelsize=SMALL_FONT_SIZE)

        figy += 1

        if (figy > 1):
            figx += 1
            figy = 0

 #   plt.show()

    fig.savefig(DEFAULT_STACKED_FILENAME)


def CreateSetCSVFiles(dataSet):

    if (dataSet == DEFAULT_NVSS_DATASET):
        # we're doing NVSS data
        rootData = DEFAULT_NVVS_DATA_ROOT
        print("Processing NVSS DATA")
    else:
        # we're doing VAST data
        rootData = DEFAULT_VAST_DATA_ROOT
        print("Processing VAST DATA")


    if (bCreateTESTFiles):

        print("*** Processing All TEST Files ***")

        sourceTESTList = ScanForSources(rootData+DEFAULT_TEST_SOURCE_LOCATION)

        testSourceFileDict = CreateAllCSVFiles(rootData+DEFAULT_TEST_SOURCE_LOCATION, sourceTESTList)

        totalNoFiles = ProcessAllCSVFiles(rootData+DEFAULT_TEST_SOURCE_LOCATION, testSourceFileDict, sourceTESTList)

        print("*** Processed "+str(totalNoFiles)+" TEST FILES")


    if (bCreateAGNFiles):

        print("*** Processing All AGN Files ***")

        sourceAGNList = ScanForSources(rootData+DEFAULT_AGN_SOURCE_LOCATION)

        agnSourceFileDict = CreateAllCSVFiles(rootData+DEFAULT_AGN_SOURCE_LOCATION, sourceAGNList)

        totalNoFiles = ProcessAllCSVFiles(rootData+DEFAULT_AGN_SOURCE_LOCATION, agnSourceFileDict,sourceAGNList)

        print("*** Processed " + str(totalNoFiles) + " AGN FILES")

    if (bCreatePULSARFiles):

        print("*** Processing All PULSAR Files ***")

        sourcePULSARList = ScanForSources(rootData+DEFAULT_PULSAR_SOURCE_LOCATION)

        pulsarSourceFileDict = CreateAllCSVFiles(rootData+DEFAULT_PULSAR_SOURCE_LOCATION, sourcePULSARList)

        totalNoFiles = ProcessAllCSVFiles(rootData+DEFAULT_PULSAR_SOURCE_LOCATION, pulsarSourceFileDict,sourcePULSARList)

        print("*** Processed " + str(totalNoFiles) + " PULSAR FILES ")

    if (bCreateQUASARFiles):

        print("*** Processing All QUASAR Files ***")

        sourceQUASARList = ScanForSources(rootData + DEFAULT_QUASAR_SOURCE_LOCATION)

        quasarSourceFileDict = CreateAllCSVFiles(rootData + DEFAULT_QUASAR_SOURCE_LOCATION, sourceQUASARList)

        totalNoFiles = ProcessAllCSVFiles(rootData + DEFAULT_QUASAR_SOURCE_LOCATION, quasarSourceFileDict,sourceQUASARList)

        print("*** Processed " + str(totalNoFiles) + " QUASAR FILES ")

    if (bCreateSEYFERTFiles):

        print("*** Processing All SEYFERT Files ***")

        sourceSEYFERTList = ScanForSources(rootData + DEFAULT_SEYFERT_SOURCE_LOCATION)

        seyfertSourceFileDict = CreateAllCSVFiles(rootData + DEFAULT_SEYFERT_SOURCE_LOCATION, sourceSEYFERTList)

        totalNoFiles = ProcessAllCSVFiles(rootData + DEFAULT_SEYFERT_SOURCE_LOCATION, seyfertSourceFileDict, sourceSEYFERTList)

        print("*** Processed " + str(totalNoFiles) + " SEYFERT FILES ")

    if (bCreateBLAZARFiles):

        print("*** Processing All BLAZAR Files ***")

        sourceBLAZARList = ScanForSources(rootData + DEFAULT_BLAZAR_SOURCE_LOCATION)

        blazarSourceFileDict = CreateAllCSVFiles(rootData + DEFAULT_BLAZAR_SOURCE_LOCATION, sourceBLAZARList)

        totalNoFiles = ProcessAllCSVFiles(rootData + DEFAULT_BLAZAR_SOURCE_LOCATION, blazarSourceFileDict, sourceBLAZARList)

        print("*** Processed " + str(totalNoFiles) + " BLAZAR FILES ")


def CreateModelFileName(dataSet,labelList):


    if (dataSet==DEFAULT_NVSS_DATASET):
        filename= NVSS_SHORT_NAME+UNDERSCORE
    else:
        filename = VAST_SHORT_NAME+UNDERSCORE

    filename += labelList[0]

    for labelNo in range(1, len(labelList)):
        filename = filename + '_' + labelList[labelNo]

    fullFilename = DEFAULT_EXISTING_MODEL_LOCATION+ filename + MODEL_FILENAME_EXTENSION

    return filename,fullFilename


def SaveLabelDict(fileName,labelDict):
    bSavedOK = False

    filename = DEFAULT_EXISTING_MODEL_LOCATION+fileName+DEFAULT_DICT_TYPE

    f = open(filename,'w')
    if (f):
        bSavedOK = True

        f.write("LabelValue,LabelName")
        f.write("\n")

        for key in labelDict.keys():
            f.write("%s,%s\n"%(key,labelDict[key]))


        f.close()


    return bSavedOK

def GetLabelDict(filename):
    from os.path import splitext

    labelDict = {}

    filename,ext = splitext(filename)

    filename = DEFAULT_EXISTING_MODEL_LOCATION+filename+DEFAULT_DICT_TYPE

    print("retrieving label dict ...",filename)

    f = open(filename)
    if (f):
        dataframe = pd.read_csv(filename)
        labelList = dataframe.values

        for label in range(len(labelList)):

            labelDict[labelList[label][0]] = labelList[label][1]

        f.close()


    return labelDict



def SaveModel(dataSet,labelList,labelDict,model):

    import pickle

    filename,fullFilename = CreateModelFileName(dataSet,labelList)

    with open(fullFilename,'wb')as file:
        pickle.dump(model,file)

    SaveLabelDict(filename,labelDict)


def GetExistingModels():
    from os.path import splitext

    modelList = []

    fileList = os.scandir(DEFAULT_EXISTING_MODEL_LOCATION)
    for entry in fileList:
       if entry.is_file():
            if entry.name[0] != DS_STORE_FILENAME:
                filename, ext = splitext(entry.name)
                if (ext == MODEL_FILENAME_EXTENSION):
                    modelList.append(entry.name)

    return modelList

def SelectExistingModel():
    import pickle

    bCorrectInput=False

    modelList = GetExistingModels()

    for model in range(len(modelList)):
        strr= str(model+1)+' : '+modelList[model]
        print(strr)

    while (bCorrectInput == False):

        modelNumber = int(input('Select Model Number: '))
        if (modelNumber > 0) and (modelNumber < (len(modelList)+1)):
            bCorrectInput= True
        else:
            print("*** Incorrect Selection - try again ***")

    print('*** SELECTED MODEL: '+modelList[modelNumber-1]+' ***')

    modelName = modelList[modelNumber - 1]
    labelDict = GetLabelDict(modelName)

    if (modelName[0] == VAST_SHORT_NAME):
        dataSet = DEFAULT_VAST_DATASET

    else:
        dataSet = DEFAULT_NVSS_DATASET

    modelName = DEFAULT_EXISTING_MODEL_LOCATION + modelList[modelNumber - 1]
    with open(modelName, 'rb')as file:
        pickleModel = pickle.load(file)

    return pickleModel,labelDict,dataSet


def TestRetrievedModel(XTest,ytest,model):

    score = model.score(XTest,ytest)

    print("Test Score (Retrieved Model): {0:.2f} %".format(100*score))


def AccessNVSSData():

    from astropy import units as u
    from astropy.coordinates import SkyCoord


    hdul = fits.open(DEFAULT_NVSS_CATALOG_FILENAME)

    nvssData = hdul[1].data
    print("size of data = ",len(nvssData))


 #   for entry in range(len(nvssData)):
    for entry in range(10):

        nvssDataEntry = nvssData[entry]
        print("RA, DEC = ",nvssDataEntry[0],nvssDataEntry[1])
        c = SkyCoord(ra=nvssDataEntry[0]*u.degree,dec=nvssDataEntry[1]*u.degree)



    sys.exit()


def OpenDuplicateFile():

    f = open(DEFAULT_DUPLICATES_FILENAME, "w")

    return f

def SaveInDuplicateFile(f, source):

    if (f):

        f.write("DUPLICATE FOR SOURCE "+str(source))
        f.write("\n")

def FindSourceInList(f,source,sourceList):

    NoFound = 0

    if (source in sourceList):
        SaveInDuplicateFile(f, source)
        NoFound += 1

    return NoFound

def CompareFITSImages(FITSImage1,FITSImage2):

    bEqual = np.array_equal(FITSImage1,FITSImage2)
    if (bEqual):
        print("FITS Images are IDENTICAL ")
    else:
        print("FITS Images are NOT IDENTICAL ")

    return bEqual


def CheckFITSIntegrity(sourceLocation,sourceList):

    # select random file from sourceList

    randomEntry = int(random.random() * len(sourceList))

    source = sourceList[randomEntry]

    imageLocation, imageList = ScanForImages(sourceLocation, source)

    print("image location = ",imageLocation)
    print("image  list = ",imageList)

    randomImage1  = int(random.random() * len(imageList))
    randomImage2 = int(random.random() * len(imageList))

    if(randomImage1 == randomImage2):
        print("Same random image chosen - exit")
        sys.exit()
    else:
        imageLocation += FOLDER_IDENTIFIER
        bValidImage1,FITSImage1 = OpenFITSFile(imageLocation+imageList[randomImage1])
        bValidImage2,FITSImage2 = OpenFITSFile(imageLocation + imageList[randomImage2])

        if ((bValidImage1) and (bValidImage2)):
            CompareFITSImages(FITSImage1,FITSImage2)
        else:
            print("*** INVALID FITS IMAGES ***")
    sys.exit()



def CheckForDuplicatedSources():


    f = OpenDuplicateFile()

    agnSourceList = ScanForSources(DEFAULT_AGN_SOURCE_LOCATION)
    print("NO AGN = ",len(agnSourceList))
    seyfertSourceList= ScanForSources(DEFAULT_SEYFERT_SOURCE_LOCATION)
    blazarSourceList= ScanForSources(DEFAULT_BLAZAR_SOURCE_LOCATION)
    quasarSourceList= ScanForSources(DEFAULT_QUASAR_SOURCE_LOCATION)
    print("NO QUASAR = ",len(quasarSourceList))
    pulsarSourceList = ScanForSources(DEFAULT_PULSAR_SOURCE_LOCATION)
    print("NO PULSAR = ",len(pulsarSourceList))

    if (bCheckFITSIntegrity):
        bCheckFITS = CheckFITSIntegrity(DEFAULT_AGN_SOURCE_LOCATION,agnSourceList)

    noFoundInSEYFERT = 0
    noFoundInBLAZAR = 0
    noFoundInQUASAR = 0
    noFoundInPULSAR = 0

    for agn in range(len(agnSourceList)):
        agnSource = agnSourceList[agn]

        noFoundInSEYFERT += FindSourceInList(f,agnSource,seyfertSourceList)
        noFoundInBLAZAR += FindSourceInList(f,agnSource, blazarSourceList)
        noFoundInQUASAR += FindSourceInList(f,agnSource, quasarSourceList)
        noFoundInPULSAR += FindSourceInList(f,agnSource, pulsarSourceList)


    if (noFoundInSEYFERT >0):
        print("*** NO. DUPLICATES FOUND FOR AGNs in SEYFERTS = "+str(noFoundInSEYFERT)+"***")
    else:
        print("*** NO DUPLICATES FOUND FOR AGNS IN SEYFERTS ***")
    if (noFoundInBLAZAR > 0):
            print("*** NO. DUPLICATES FOUND FOR AGNs in BLAZARS = " + str(noFoundInBLAZAR)+" ***")
    else:
        print("*** NO DUPLICATES FOUND FOR AGNS IN BLAZARS ***")

    if (noFoundInQUASAR > 0):
            print("*** NO. DUPLICATES FOUND FOR AGNs in QUASARS = " + str(noFoundInQUASAR)+" ***")
    else:
        print("*** NO DUPLICATES FOUND FOR AGNS IN QUASARS ***")
    if (noFoundInPULSAR > 0):
            print("*** NO. DUPLICATES FOUND FOR AGNs in PULSARS = " + str(noFoundInPULSAR)+"***")
    else:
        print("*** NO DUPLICATES FOUND FOR AGNS IN PULSARS ***")

    print("\n")

    noFoundInAGN = 0
    noFoundInBLAZAR = 0
    noFoundInQUASAR = 0
    noFoundInPULSAR = 0

    for seyfert in range(len(seyfertSourceList)):
        seyfertSource = seyfertSourceList[seyfert]

        noFoundInAGN += FindSourceInList(f,seyfertSource, agnSourceList)
        noFoundInBLAZAR += FindSourceInList(f,seyfertSource, blazarSourceList)
        noFoundInQUASAR += FindSourceInList(f,seyfertSource, quasarSourceList)
        noFoundInPULSAR += FindSourceInList(f,seyfertSource, pulsarSourceList)

    if (noFoundInAGN > 0):
        print("*** NO. DUPLICATES FOUND FOR SEYFERTS IN AGN = " + str(noFoundInAGN) + "***")
    else:
        print("*** NO DUPLICATES FOUND FOR SEYFERTS IN AGN ***")
    if (noFoundInBLAZAR > 0):
        print("*** NO. DUPLICATES FOUND FOR SEYFERTS in BLAZARS = " + str(noFoundInBLAZAR) + " ***")
    else:
        print("*** NO DUPLICATES FOUND FOR SEYFERTS IN BLAZARS ***")
    if (noFoundInQUASAR > 0):
        print("*** NO. DUPLICATES FOUND FOR SEYFERTS in QUASARS = " + str(noFoundInQUASAR) + " ***")
    else:
        print("*** NO DUPLICATES FOUND FOR SEYFERTS IN QUASARS ***")
    if (noFoundInPULSAR > 0):
        print("*** NO. DUPLICATES FOUND FOR SEYFERTS in PULSARS = " + str(noFoundInPULSAR) + "***")
    else:
        print("*** NO DUPLICATES FOUND FOR SEYFERTS IN PULSARS ***")

    print("\n")

    noFoundInAGN = 0
    noFoundInSEYFERT = 0
    noFoundInQUASAR = 0
    noFoundInPULSAR = 0

    for blazar in range(len(blazarSourceList)):
        blazarSource = blazarSourceList[blazar]

        noFoundInAGN += FindSourceInList(f,blazarSource, agnSourceList)
        noFoundInSEYFERT += FindSourceInList(f,blazarSource, seyfertSourceList)
        noFoundInQUASAR += FindSourceInList(f,blazarSource, quasarSourceList)
        noFoundInPULSAR += FindSourceInList(f,blazarSource, pulsarSourceList)

    if (noFoundInAGN > 0):
        print("*** NO. DUPLICATES FOUND FOR BLAZARS IN AGN = " + str(noFoundInAGN) + "***")
    else:
        print("*** NO DUPLICATES FOUND FOR BLAZARS IN AGN ***")
    if (noFoundInSEYFERT > 0):
        print("*** NO. DUPLICATES FOUND FOR BLAZARS IN SEYFERTS  = " + str(noFoundInSEYFERT) + " ***")
    else:
        print("*** NO DUPLICATES FOUND FOR BLAZARS IN SEYFERTS ***")
    if (noFoundInQUASAR > 0):
        print("*** NO. DUPLICATES FOUND FOR BLAZARS IN QUASARS = " + str(noFoundInQUASAR) + " ***")
    else:
        print("*** NO DUPLICATES FOUND FOR BLAZARS IN QUASARS ***")
    if (noFoundInPULSAR > 0):
        print("*** NO. DUPLICATES FOUND FOR BLAZARS in PULSARS = " + str(noFoundInPULSAR) + "***")
    else:
        print("*** NO DUPLICATES FOUND FOR BLAZARS IN PULSARS ***")

    print("\n")

    noFoundInAGN = 0
    noFoundInSEYFERT = 0
    noFoundInBLAZAR = 0
    noFoundInPULSAR = 0

    for quasar in range(len(quasarSourceList)):
        quasarSource = quasarSourceList[quasar]

        noFoundInAGN += FindSourceInList(f,quasarSource, agnSourceList)
        noFoundInBLAZAR += FindSourceInList(f,quasarSource, blazarSourceList)
        noFoundInSEYFERT += FindSourceInList(f,quasarSource, seyfertSourceList)
        noFoundInPULSAR += FindSourceInList(f,quasarSource, pulsarSourceList)

    if (noFoundInAGN > 0):
        print("*** NO. DUPLICATES FOUND FOR QUASARS IN AGN = " + str(noFoundInAGN) + "***")
    else:
        print("*** NO DUPLICATES FOUND FOR QUASARS IN AGN ***")
    if (noFoundInBLAZAR > 0):
        print("*** NO. DUPLICATES FOUND FOR QUASARS in BLAZARS = " + str(noFoundInBLAZAR) + " ***")
    else:
        print("*** NO DUPLICATES FOUND FOR QUASARS IN BLAZARS ***")
    if (noFoundInSEYFERT > 0):
        print("*** NO. DUPLICATES FOUND FOR QUASARS IN SEYFERTS  = " + str(noFoundInSEYFERT) + " ***")
    else:
        print("*** NO DUPLICATES FOUND FOR QUASARS IN SEYFERTS ***")
    if (noFoundInPULSAR > 0):
        print("*** NO. DUPLICATES FOUND FOR QUASARS IN PULSARS = " + str(noFoundInPULSAR) + "***")
    else:
        print("*** NO DUPLICATES FOUND FOR QUASARS IN PULSARS ***")

    print("\n")

    noFoundInAGN = 0
    noFoundInSEYFERT = 0
    noFoundInBLAZAR = 0
    noFoundInQUASAR = 0


    for pulsar in range(len(pulsarSourceList)):
        pulsarSource = pulsarSourceList[pulsar]

        noFoundInAGN += FindSourceInList(f,pulsarSource, agnSourceList)
        noFoundInBLAZAR += FindSourceInList(f,pulsarSource, blazarSourceList)
        noFoundInQUASAR += FindSourceInList(f,pulsarSource, quasarSourceList)
        noFoundInSEYFERT += FindSourceInList(f,pulsarSource, seyfertSourceList)

    if (noFoundInAGN > 0):
        print("*** NO. DUPLICATES FOUND FOR PULSARS IN AGN = " + str(noFoundInAGN) + "***")
    else:
        print("*** NO DUPLICATES FOUND FOR PULSARS IN AGN ***")
    if (noFoundInBLAZAR > 0):
        print("*** NO. DUPLICATES FOUND FOR PULSARS in BLAZARS = " + str(noFoundInBLAZAR) + " ***")
    else:
        print("*** NO DUPLICATES FOUND FOR PULSARS IN BLAZARS ***")
    if (noFoundInQUASAR > 0):
        print("*** NO. DUPLICATES FOUND FOR PULSARS in QUASARS = " + str(noFoundInQUASAR) + " ***")
    else:
        print("*** NO DUPLICATES FOUND FOR PULSARS IN QUASARS ***")
    if (noFoundInSEYFERT > 0):
        print("*** NO. DUPLICATES FOUND FOR PULSARS IN SEYFERT = " + str(noFoundInSEYFERT) + "***")
    else:
        print("*** NO DUPLICATES FOUND FOR PULSARS IN SEYFERTS ***")

    print("\n")
    f.close()

    sys.exit()


def StoreNVSSSources(f,name,ra,dec):

    f.write(name+','+ra+','+dec)
    f.write('\n')





def ViewImagesToRetrieve(df):

    for entry in range(len(df.RA)):

        strr = 'Skyview Source '+str(entry)+' '+str(df.SOURCE[entry]) + ' = ' + str(df.RA[entry]) + ' ' + str(df.DEC[entry])

        print(strr)


def GetSkyviewImages(transientType):

    from astroquery.skyview import SkyView
    import pandas as pd
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    if (transientType == DEFAULT_NVSS_PULSAR_CLASS):
        sourceDir = DEFAULT_NVSS_PULSAR_SOURCE_LOCATION
        sourcesFilename = DEFAULT_NVSS_PULSAR_SOURCES_FILENAME

    elif (transientType == DEFAULT_NVSS_QUASAR_CLASS):
        sourceDir = DEFAULT_NVSS_QUASAR_SOURCE_LOCATION
        sourcesFilename = DEFAULT_NVSS_QUASARS_SOURCES_FILENAME

    elif (transientType == DEFAULT_NVSS_SEYFERT_CLASS):
        sourceDir = DEFAULT_NVSS_SEYFERT_SOURCE_LOCATION
        sourcesFilename =DEFAULT_NVSS_SEYFERT_SOURCES_FILENAME

    elif (transientType == DEFAULT_NVSS_BLAZAR_CLASS):
        sourceDir = DEFAULT_NVSS_BLAZAR_SOURCE_LOCATION
        sourcesFilename =DEFAULT_NVSS_BLAZARS_SOURCES_FILENAME


    print("Reading NVSS CSV file ..."+sourcesFilename)

    df = pd.read_csv(sourcesFilename)

    print("Total No of Images To Be Processed = ",len(df.RA))
    fitsFileNumber = 0

    for entry in range(len(df.RA)):
        bValidData = True

        strr = 'Skyview Source '+str(df.SOURCE[entry])+' = '+str(df.RA[entry]) + ' ' + str(df.DEC[entry])

        print(strr)

        coordStrr = str(df.RA[entry])+' '+str(df.DEC[entry])

        sv = SkyView()

        try:

            imagePaths = sv.get_images(position=coordStrr, coordinates="J2000", survey='NVSS')

        except:
            bValidData = False
            print("*** Error in Skyview call - moving on ***")

        if (bValidData):
            for fitsImage in imagePaths:
                print('new file for source',df.SOURCE[entry])
                fitsFileNumber += 1
                print("Storing FITS NVSS File Number: ",fitsFileNumber)
                StoreNVSSFITSImage(transientType, sourceDir, df.SOURCE[entry], fitsImage, fitsFileNumber)


    print("Finished...exiting")
    sys.exit()

def SelectSkyviewImages(ra,dec):

    from astroquery.skyview import SkyView
    import pandas as pd
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import ssl

    imageList = []
    bValidData = True

    ssl._create_default_https_context = ssl._create_unverified_context
    coordStrr = ra+' '+dec

    print('Skyview Coordinates = '+coordStrr)

    sv = SkyView()

    try:

        imagePaths = sv.get_images(position=coordStrr, coordinates="J2000", survey='NVSS')

        for fitsImage in imagePaths:
            bValidData = True
            imageList.append(fitsImage)

    except:

        bValidData = False

        print("*** Error in NVSS Skyview call ***")

    return bValidData, imageList

def ProcessAstroqueryTransient(startSelection,catalogName,transientName,OutputSourcesFilename,entryRA,entryDEC,bDegree):

    from astroquery.vizier import Vizier
    from astroquery.nrao import Nrao
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.coordinates import Angle
    from astropy.coordinates import SkyCoord
    from astroquery.image_cutouts.first import First
    import time

    print('NVSS Processing '+transientName+'....')
    print(catalogName)
    catalog_list = Vizier.find_catalogs(catalogName)

    print({k:v.description for k,v in catalog_list.items()})
    Vizier.ROW_LIMIT = -1

    Catalog = Vizier.get_catalogs(catalog_list.keys())

    print(Catalog)

    transientTable = Catalog[0]

    numberTransients = len(transientTable)

    print('Total Number '+transientName+' in Vizier Catalog = '+str(numberTransients))

    allVizierCoords = []

    for entry in range(numberTransients):


        print("Processing entry...",entry)

        try:
            if (bDegree):

                skyCoords = SkyCoord(transientTable[entry][entryRA],transientTable[entry][entryDEC],unit=("deg"))
            else:
                skyCoords = SkyCoord(transientTable[entry][entryRA] + ' ' + transientTable[entry][entryDEC],unit=(u.hourangle, u.deg))

            allVizierCoords.append(skyCoords)
        except:

            print("*** Error in Vizier coordinates  - moving on ***")

    sourcesFound = 0

    NVSS_RA_Coords = []
    NVSS_DEC_Coords = []

    print('Opening Output File '+OutputSourcesFilename+'....')

    fNVSS = open(OutputSourcesFilename, "w")
    bComplete = False

    entry = startSelection
    noErrors = 0

    while (bComplete == False):

        try:
            print("Querying entry "+str(entry)+" from total of "+str(numberTransients))

            results_table = Nrao.query_region(coord.SkyCoord(allVizierCoords[entry].ra,allVizierCoords[entry].dec,unit=u.degree),radius=15*u.arcsec)
            print("RESULT OK")
            if (len(results_table) > 0):

                print(transientName+' SOURCE FOUND')
                sourcesFound += 1
                print(results_table)

                ra = results_table['RA'][0]
                dec = results_table['DEC'][0]

                NVSS_RA_Coords.append(ra)
                NVSS_DEC_Coords.append(dec)

                StoreNVSSSources(fNVSS, results_table['Source'][0], ra, dec)
            else:
                print('NO '+transientName + ' SOURCE FOUND')

            print('TOTAL NO ' +transientName+ ' SOURCES FOUND SO FAR = '+str(sourcesFound))

            print('TOTAL NO ERRORS SO FAR = '+str(noErrors))

            if (sourcesFound >= 600):
                print('600 '+transientName+' SOURCES FOUND AT ENTRY No. ='+str(entry))
                bComplete = True
            else:
                entry += 1
                if (entry >= numberTransients):
                    print("AT END OF CATALOG")
                    bComplete = True

        except :
            print("Invalid return - carrying on")
            noErrors += 1
            entry += 1
            if (entry >= numberTransients):
                print("AT END OF CATALOG")
                bComplete = True

    print('TOTAL NO '+transientName+' SOURCES FOUND = '+str(sourcesFound))

    fNVSS.close()
    sys.exit()




def BuildandTestCNNModel(labelList,completeTrainingData,trainingDataSizes):


    XTrain, XTest, ytrain, ytest, labelDict, randomSamples,randomSampleLabels = CreateTrainingAndTestData(True, labelList,
                                                                                       completeTrainingData,
                                                                                       trainingDataSizes)

    n_timesteps = XTrain.shape[1]
    n_features = XTrain.shape[2]
    n_outputs = ytrain.shape[1]

    if (bDebug):
        print(n_timesteps, n_features, n_outputs)

    if (bOptimiseHyperParameters == True):

        fOptimisationFile = OpenOptimsationFile()

        if (fOptimisationFile):
            print("*** Optimising CNN Hyper Parameters ***")
            ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates, ExperimentDropoutRates = OptimiseCNNHyperparameters(
                fOptimisationFile, XTrain, ytrain, XTest, ytest)
            fOptimisationFile.close()
            DisplayHyperTable(ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates, ExperimentDropoutRates)
            BestEpochs, BestLearningRate, BestDropoutRate = GetOptimalParameters(ExperimentAccuracy, ExperimentEpochs,
                                                                                 ExperimentLearningRates,
                                                                                 ExperimentDropoutRates)

            Accuracy, CNNModel = testCNNModel(BestEpochs, BestLearningRate, BestDropoutRate, XTrain, ytrain, XTest,
                                              ytest)
        else:
            print("*** Failed To Open CNN Hyper Parameters File ***")

    else:
        print("*** Evaluating CNN Model ***")
        Accuracy, CNNModel = evaluateCNNModel(XTrain, ytrain, XTest, ytest, n_timesteps, n_features, n_outputs,
                                              DEFAULT_NO_EPOCHS)

    print("Final CNN Accuracy = ", Accuracy)


def ProcessNVSSCatalog():

    print("Transforming NVSS Catalog")
    bComplete =False

    fIn = open(DEFAULT_NVSS_CATALOG_LOCATION)
    fOut = open(FINAL_NVSS_CATALOG_LOCATION,'w')

    if (fIn) and (fOut):

        sourceNumber = 0
        # read one line at a time and parse to output file
        while (bComplete==False):
            line = fIn.readline()
            if not line:
                bComplete=True
            else:
                print(line)
                if ('##') in line:
                    print("Ignoring this line..." + line)
                elif ('NVSS') in line:
                    print("Ignoring this line..."+line)
                elif ('RA(2000)') in line :
                    print("Ignoring this line..." + line)
                elif ('mJ') in line:
                    print("Ignoring this line..." + line)
                else:

                    sourceNumber +=1

                    splitLine = line.split()
                    if (len(splitLine) >= MAX_NVSS_RA_DEC_STRINGS):
                        ra = splitLine[NVSS_CAT_RA_POS]+' '+splitLine[NVSS_CAT_RA_POS+1]+' '+splitLine[NVSS_CAT_RA_POS+2]
                        dec = splitLine[NVSS_CAT_DEC_POS]+' ' + splitLine[NVSS_CAT_DEC_POS+1] + ' ' + splitLine[NVSS_CAT_DEC_POS+2]
                        newLine = str(sourceNumber) + ' , '+ra+' '+dec

                        fOut.write(newLine)
                        fOut.write('\n')

                    line = fIn.readline() # throw away this line
                    if not line:
                        bComplete = True

    fIn.close()
    fOut.close()

    print("Finished transforming NVSS Catalog")
    sys.exit()

def ProcessNewNVSSCatalog(bRandomOrNot):

    if (bRandomOrNot):
        print("Randomly Selecting from NVSS Catalog....")
    else:
        print("Sequentially Selecting from NVSS Catalog....")

    if (bRandomOrNot):

        # create the random selection of 1000 entries
        entrySelected = []

        for entry in range(DEFAULT_MAX_NO_RANDOM_NVSS):
            chosenEntry = int(random.random()*DEFAULT_MAX_NO_NVSS_ENTRIES)

            entrySelected.append(chosenEntry)

    # now select each of these random lines from the NVSS Catalog

    fIn = open(FINAL_NVSS_CATALOG_LOCATION)
    fOut = open(FINAL_SELECTED_NVSS_SOURCES_LOCATION,"w")
    entryNo = 1
    bComplete=False

    if (fIn) and (fOut):

        while (bComplete == False):
            line = fIn.readline()
            if not line:
                bComplete = True
            else:
                if (entryNo) in entrySelected:
                     # found one - now store in a separate file
                    fOut.write(line)
                    entrySelected.remove(entryNo)
                entryNo+=1

        fIn.close()
        fOut.close()

def SelectRandomNVSSSource():


    chosenEntry = int(random.random()*DEFAULT_MAX_NO_NVSS_ENTRIES)

    # now select each of these random lines from the NVSS Catalog

    fIn = open(FINAL_NVSS_CATALOG_LOCATION)

    bComplete=False
    bValidData = False
    entryNo = 1

    if (fIn):

        while (bComplete == False):
            line = fIn.readline()
            if not line:
                bComplete = True
            else:
                if (entryNo ==chosenEntry):
                    bComplete=True
                    bValidData= True
                    # now get the ra and dec

                    splitText = line.split()

                    ra = splitText[2]+' '+splitText[3]+' '+splitText[4]
                    dec = splitText[5]+' '+ splitText[6]+' '+splitText[7]

                entryNo+=1

        fIn.close()

    return bValidData,ra,dec


def OpenNVSSDetections():

    f = open(NVSS_CATALOG_DETECTIONS_FILENAME,'w')

    return f


def StoreNVSSDetections(f,source,ra,dec,label):

    if (f):

        print('Detection For Source No: '+source+' = '+label)
        f.write('Detection For Source No: '+source+','+'ra= '+ra+', dec= '+dec+' = '+label)
        f.write('\n')



def TestNVSSCatalog(model,labelDict,startEntry):

    print("Testing from NVSS Catalog....")
    totalNoSources = 0
    entry =0
    fIn = open(FINAL_NVSS_CATALOG_FILE)
    fDetect = OpenNVSSDetections()

    bComplete = False

    if (fIn) and (fDetect):
        MAX_NVSS_TO_PROCESS = 10000

        while (bComplete == False):
            line = fIn.readline()
            entry +=1
            if not line:
                bComplete = True
            else:
                    if (entry >= startEntry):
                        # now get the ra and dec

                        splitText = line.split()

                        source = splitText[0]
                        ra = splitText[2] + ' ' + splitText[3] + ' ' + splitText[4]
                        dec = splitText[5] + ' ' + splitText[6] + ' ' + splitText[7]


                        # now get the image from skyview and test it

                        bValidData, imageList = SelectSkyviewImages(ra, dec)
                        if (bValidData):
                            for fitsImage in imageList:
                                bValidFile,fitsImageFile = StoreNVSSCatalogImage(fitsImage,source)
                                if (bValidFile):

                                    bStoreOK, csvFilename = StoreIndividualCSVFile(fitsImageFile,source)
                                    if (bStoreOK):
                                        dataframe = pd.read_csv(csvFilename, header=None)
                                        dataValues = dataframe.values
                                        sourceData = np.reshape(dataValues, (1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE))
                                        if (bScaleInputs):
                                            dataAsArray = ScaleInputData(sourceData)

                                        ypredict = model.predict(dataAsArray)
                                        print("ypredict = ",ypredict)

                                        bDetection,predictedLabel = DecodePredictedLabel(labelDict, ypredict)
                                        if (bDetection):
                                            StoreNVSSDetections(fDetect, source, ra,dec,predictedLabel)
                                            totalNoSources += 1
                                            if (totalNoSources >= MAX_NVSS_TO_PROCESS):
                                                bComplete = True


    fIn.close()
    fDetect.close()

def TestNVSSFiles(model,labelDict):

    print("Testing from NVSS Files....")

    bValidImageList,fitsImageList = ScanForTestImages(DEFAULT_TEST_NVSS_SOURCE_LOCATION)

    if (bValidImageList):
        for fitsImage in fitsImageList:
                print("fits image = ", fitsImage)
                bValidData, fitsData = OpenFITSFile(DEFAULT_TEST_NVSS_SOURCE_LOCATION+fitsImage)
                if (bValidData):
                    fitsData = np.reshape(fitsData, (1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE))

                    if (bScaleInputs):
                        fitsData = ScaleInputData(fitsData)
                    ypredict = model.predict(fitsData)
                    print("ypredict = ",ypredict)

                    bDetection,predictedLabel = DecodePredictedLabel(labelDict, ypredict)




def CreateData():

    if (bGetSkyview):
        GetSkyviewImages(DEFAULT_NVSS_AGN_CLASS)
        GetSkyviewImages(DEFAULT_NVSS_BLAZAR_CLASS)
        GetSkyviewImages(DEFAULT_NVSS_SEYFERT_CLASS)
        GetSkyviewImages(DEFAULT_NVSS_QUASAR_CLASS)
        GetSkyviewImages(DEFAULT_NVSS_PULSAR_CLASS)

    if (bAstroqueryAGN):
        ProcessAstroqueryTransient(0, 'VII/258', 'AGN', DEFAULT_NVSS_AGN_SOURCES_FILENAME, 'RAJ2000', 'DEJ2000',
                                   False)

    if (bAstroqueryPULSAR):
        ProcessAstroqueryTransient(0, 'B/psr', 'PULSAR', DEFAULT_NVSS_PULSAR_SOURCES_FILENAME, 'RAJ2000', 'DEJ2000',
                                   False)

    if (bAstroquerySEYFERT):
        ProcessAstroqueryTransient(0, 'II/221A', 'SEYFERT', DEFAULT_NVSS_SEYFERT_SOURCES_FILENAME, 'RA1950',
                                   'DE1950', False)

    if (bAstroqueryQUASAR):
        ProcessAstroqueryTransient(0, 'J/ApJ/873/132', 'QUASAR', DEFAULT_NVSS_QUASARS_SOURCES_FILENAME, 'RAICRS',
                                   'DEICRS', True)

    if (bAstroqueryBLAZAR):
        #  ProcessAstroqueryTransient(0,'VII/157','BLAZAR',DEFAULT_NVSS_BLAZARS_SOURCES_FILENAME,'RA1950', 'DE1950',False)
        ProcessAstroqueryTransient(0, 'VII/274', 'BLAZAR', DEFAULT_NVSS_BLAZARS_SOURCES_FILENAME, 'RAJ2000',
                                   'DEJ2000', False)

    if (bCheckForDuplicates):
        CheckForDuplicatedSources()

    if (bAccessNVSS):
        AccessNVSSData()

    if (bCreatePulsarData):
        ProcessPulsarData()


def main():


    if (bDataCreation):
        CreateData()

    elif (bCreateNVSSCatalog):

        if (bProcessNVSSCatalog):

            ProcessNVSSCatalog()


        if (bSelectFromNVSSCatalog):

            ProcessNewNVSSCatalog(True)

    elif (bCreateCSVFiles):

        if (bCreateVASTFiles):
            CreateSetCSVFiles(DEFAULT_VAST_DATASET)

        if (bCreateNVSSFiles):
            CreateSetCSVFiles(DEFAULT_NVSS_DATASET)

    else:

        selectedOperation = GetOperationMode()

        if (selectedOperation == OP_BUILD_MODEL):

            modelType = GetModelType()

            if (modelType == OP_MODEL_RANDOM):
                bTestRandomForestModel = True
            else:
                bTestRandomForestModel = False

            # now process all images per chosen datasets

            dataSet,labelList = GetSelectedDataSets()
            print("Datasets to be classified are ",dataSet)
            print("Classes to be classified are ",labelList)

            completeTrainingData = []
            trainingDataSizes = []
            randomTrainingData = []
            randomTrainingLabel = []

            print("*** Loading Training Data ***")

            for classes in range(len(labelList)):

                trainingData = ProcessTransientData(dataSet,labelList[classes])

                trainingDataSizes.append(len(trainingData))
                completeTrainingData.append(trainingData)

                if (bCollectRawRandomSamples):
                    # collect some random data entry from training samples

                    for i in range(5):
                        randomChoice = int(random.random() * len(completeTrainingData[classes]))

                        randomTrainingData.append(completeTrainingData[classes][randomChoice])
                        randomTrainingLabel.append(labelList[classes])

                        completeTrainingData[classes].pop(randomChoice)


            if (bStackImages):
                stackedImages = StackTrainingData(completeTrainingData)

                print("no of stacked images = ",len(stackedImages))
                DisplayStackedImages(stackedImages, labelList)

            print("*** Creating Training and Test Data Sets ***")
            if (bTestClassicModels):

                XTrain, XTest, ytrain, ytest,labelDict,randomSamples,randomSampleLabels = CreateTrainingAndTestData(False,labelList,completeTrainingData,trainingDataSizes,randomTrainingData)

                print("*** Evaluating Multiple Classic Models ***")
                MultipleClassicModels(XTrain, ytrain, XTest, ytest)


            if (bTestRandomForestModel):
                    XTrain, XTest, ytrain, ytest,labelDict,randomSamples,randomSampleLabels = CreateTrainingAndTestData(False,labelList, completeTrainingData,trainingDataSizes,randomTrainingData)

                    if (bCheckTrainingAndTestSet):
                        CheckTrainingAndTestSet(XTrain, XTest)
                    if (bCheckDupRandomSamples):

                         CheckDupRandomSamples(XTrain, XTest, randomSamples)

                    print("*** Evaluating Random Forest Model ***")

                    randomForestModel = RandomForestModel(XTrain, ytrain, XTest, ytest)

                    if (bTestIndSamples):
                        TestIndependentSamples(XTrain,XTest,dataSet,randomForestModel, labelDict)

                    if (bSaveModel):
                        SaveModel(dataSet,labelList,labelDict,randomForestModel)

                    if (bCollectRandomSamples):
                        TestRandomSamples(labelDict, randomForestModel,randomSamples,randomSampleLabels)

                    if (bCollectRawRandomSamples):
                        rawTestSamples,rawTestLabels = TestRawRandomSamples(XTrain,XTest,labelDict, randomForestModel,randomTrainingData,randomTrainingLabel)
                        #TestRandomSamples(labelDict, randomForestModel, rawTestSamples,rawTestLabels)


                    if (bTestRandomFiles):

                        TestRandomFITSFiles(DEFAULT_FITS_NO_TESTS, randomForestModel, XTest, ytest, labelDict)

            else:

                BuildandTestCNNModel(labelList, completeTrainingData, trainingDataSizes)
        else:

                model,labelDict,dataSet = SelectExistingModel()
                if (bTestNVSSFiles):
                    TestNVSSFiles(model, labelDict)
                if (bTestNVSSCatalog):
                    TestNVSSCatalog(model,labelDict,2549)
                if (bTestIndSamples):
                      TestIndependentSamples(dataSet,DEFAULT_TEST_CLASS,model,labelDict)






if __name__ == '__main__':
    main()
