import math
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
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
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
DEFAULT_SOURCES_LOCATION = 'SOURCES/'
DEFAULT_RANDOM_FILE_TEST_LOCATION = '/Volumes/ExtraDisk/VAST_RANDOM_FILE_TESTS/'

DEFAULT_VAST_RANDOM_FILE_TEST_LOCATION = '/Volumes/ExtraDisk/TESTFITS/'
DEFAULT_EXISTING_MODEL_LOCATION = '/Volumes/ExtraDisk/OUTPUT_MODELS/'

VAST_OTHER_BINARY_MODELS_LOCATION = '/Volumes/ExtraDisk/VAST_DATA/BINARY_OTHER_MODELS/'
VAST_FULL_BINARY_MODELS_LOCATION = '/Volumes/ExtraDisk/VAST_DATA/MULTI_BINARY_MODELS/'

NVSS_OTHER_BINARY_MODELS_LOCATION = '/Volumes/ExtraDisk/NVSS_DATA/BINARY_OTHER_MODELS/'
NVSS_FULL_BINARY_MODELS_LOCATION = '/Volumes/ExtraDisk/NVSS_DATA/MULTI_BINARY_MODELS/'

DEFAULT_NVSS_DATASET =  "NVSS"
DEFAULT_VAST_DATASET = "VAST"

NVSS_SHORT_NAME = 'N'
VAST_SHORT_NAME = 'V'


DEFAULT_HYPER_FILENAME = 'CNNHyper.txt'
ORIGINAL_PULSAR_SOURCES_FILENAME = DEFAULT_PULSAR_SOURCE_LOCATION+'ATNF.txt'

DEFAULT_NVSS_CATALOG_FILENAME = '/Volumes/ExtraDisk/NVSS/CATALOG.FIT'

DEFAULT_AGN_SOURCES_FILENAME = 'AGN_sources.txt'
DEFAULT_BLAZAR_SOURCES_FILENAME = 'BLAZAR_sources.txt'
DEFAULT_SEYFERT_SOURCES_FILENAME = 'SEYFERT_sources.txt'
DEFAULT_QUASAR_SOURCES_FILENAME = 'QUASAR_sources.txt'
DEFAULT_PULSAR_SOURCES_FILENAME = 'PULSAR_sources.txt'

DEFAULT_HYPERPARAMETERS_FILE = DEFAULT_VAST_RANDOM_FILE_TEST_LOCATION+DEFAULT_HYPER_FILENAME

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

BINARY_OTHER_MODELS = 'O'
BINARY_FULL_MODELS = 'F'

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

MAX_NUMBER_MODELS = 5


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
DEFAULT_MODEL_EXTENSION = '.pkl'
DS_STORE_FILENAME = '.'
DEFAULT_CSV_DIR = 'CSVFiles'
FOLDER_IDENTIFIER = '/'
SOURCE_TITLE_TEXT = ' Source :'
UNDERSCORE = '_'
DEFAULT_CSV_FILETYPE = UNDERSCORE+'data.txt'
XSIZE_FITS_IMAGE= 120
YSIZE_FITS_IMAGE = 120

MAX_NUMBER_RAW_SAMPLES = 10

DEFAULT_NO_INDEPENDENT_TESTS = 10


XSIZE_SMALL_FITS_IMAGE = 20
YSIZE_SMALL_FITS_IMAGE = 20

DEFAULT_AGN_CLASS = "AGN"
DEFAULT_BLAZAR_CLASS = "BLAZAR"
DEFAULT_SEYFERT_CLASS = "SEYFERT"
DEFAULT_QUASAR_CLASS = "QUASAR"
DEFAULT_PULSAR_CLASS = "PULSAR"
DEFAULT_TEST_CLASS = "TEST"
DEFAULT_OTHER_CLASS = "OTHER_TRANSIENT"

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


DEFAULT_CLASSIC_SEYFERT_CLASS = 1
DEFAULT_CLASSIC_BLAZAR_CLASS = 2
DEFAULT_CLASSIC_QUASAR_CLASS = 3
DEFAULT_CLASSIC_AGN_CLASS = 4
DEFAULT_CLASSIC_PULSAR_CLASS = 5


DEFAULT_CLASSIC_NVSS_AGN_CLASS = 5


DEFAULT_CLASSIC_MODEL = True
DEFAULT_NN_MODEL = False
# Defaults for CNN Model

DEFAULT_VERBOSE_LEVEL = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_NO_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_DROPOUT_RATE = 0.20
DEFAULT_NO_NEURONS = 100

TRAIN_TEST_RATIO = 0.80  # ratio of the total data for training

DEFAULT_NUMBER_MODELS  = 5

SMALL_FONT_SIZE = 8
MEDIUM_FONT_SIZE = 10
BIGGER_FONT_SIZE = 12



OP_BUILD_FULL_MODEL = '1'
OP_BUILD_BINARY_MODELS = '2'
OP_TEST_MODEL = '3'
OP_TEST_BLIND = '4'
OP_BUILD_STATS = '5'

OP_MODEL_RANDOM= 'R'
OP_MODEL_CNN = 'C'
OP_MODEL_NAIVE = 'N'
OP_MODEL_SVM = 'O'

DEFAULT_DICT_TYPE = '_dict.txt'
DEFAULT_SCALER_TYPE = '_scaler.bin'
DEFAULT_ENCODER_TYPE = '.zip'
MODEL_FILENAME_EXTENSION = '.pkl'
MODEL_TXT_EXTENSION = '.txt'
MODEL_BIN_EXTENSION = '.bin'
MODEL_ENC_EXTENSION = '.zip'
DEFAULT_RANDOM_TYPE = 'test.txt'


DEFAULT_TEST_V_Q_O = 'V_Q_O'
DEFAULT_TEST_V_A_O = 'V_A_O'
DEFAULT_TEST_V_P_O = 'V_P_O'
DEFAULT_TEST_V_S_O = 'V_S_O'
DEFAULT_TEST_V_B_O = 'V_B_O'

DEFAULT_TEST_N_Q_O = 'N_Q_O'
DEFAULT_TEST_N_A_O = 'N_A_O'
DEFAULT_TEST_N_P_O = 'N_P_O'
DEFAULT_TEST_N_S_O = 'N_S_O'
DEFAULT_TEST_N_B_O = 'N_B_O'


# used for stat comparison on images



bStratifiedSplit = False # use stratified split to create random train/test sets
bScaleInputs = True # use minmax scaler for inputs
bNormalizeData = False # or normalize instead
bStandardScaler = False # or normalize instead


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
bFullTestNVSSCatalog = False # for testing againt NVSS catalogue
bTestNVSSFiles = False # for testing againt a select set of NVSS FITS files
bTestIndSamples = False # test some independent samples
bDataCreation = False # for general data creation functions
bCreateCSVFiles = False # flag set to create all CSV files from FITS images
bTakeTop2Models = True # if false, will continue to searh below top 2 models
bCNNBinary = True  # for sigmoid vs softmax activation
bCollectPossibilities = False # log potential outcomes
bDebug = False # swich on debug code
bLoadFullBinaryModels = True # load full binary models for comparison
bOptimiseHyperParameters = False # optimise hyper parameters used on convolutional model
bTestFITSFile = False # test CNN model with individual FITS files
bTestRandomForestModel = False # test random forest model only

bSaveImageFiles = False # save FITS files for presentations

bDisplayProbs = False # display predicted probabilities
bDisplayIndividualPredictions = True
bStackImages = False # stack all source images
bDisplayStackedImages = False
bShrinkImages = False  # to experiment with smalleer FITS images
bRandomSelectData = True # set false if you want to eliminate randomizing of training and test data
bConstrainSamples=False # constrain samples to size of primary dataset
bEqualiseAllSamples=True # equalise all data samples
bBlindTestAllSamples = True
bBlindTestAGN = False
bBlindTestQUASAR = True
bBlindTestPULSAR = False
bBlindTestSEYFERT = False
bBlindTestBLAZAR = False


bReduceFalseNegatives = True # reduce false negatives by picking largest probability
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

    if (bCNNBinary):
        model.add(tf.keras.layers.Dense(n_outputs, activation='sigmoid'))
    else:
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

    OriglistOfLabels = np.array(listOfLabels)
    listOfLabels = np.array(listOfLabels)

    listOfLabels= listOfLabels.reshape(-1,1)
 #   print(listOfLabels)
    integerEncoded = ordinalEncoder.fit_transform(listOfLabels)
 #   print(integerEncoded)

    oneHotEncoded = oneHotEncoder.fit_transform(integerEncoded)
    oneHotEncoded = oneHotEncoded.toarray()

  #  print(oneHotEncoded)

    for i in range(len(listOfLabels)):
        labelDict[integerEncoded[i][0]] = OriglistOfLabels[i]


    return oneHotEncoded, labelDict,oneHotEncoder



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
   # scaler = StandardScaler()
    scaler.fit(X)
    normalised = scaler.transform(X)

    return normalised,scaler

def StandardizeInputData(X):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X)
    normalised = scaler.transform(X)

    return normalised


def NormalizeInputData(X):

    normalised = normalize(X)

    return normalised


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



def ScanForModels(modelLocation):
    import pathlib

    bValidData= False

    modelList = []
    dictList = []
    scalerList = []
    encoderList = []

    fileList = os.scandir(modelLocation)
    for entry in fileList:
        if entry.is_dir():
            if (bDebug):
                print("dir = ",entry.name)

        elif entry.is_file():

            if entry.name[0] != DS_STORE_FILENAME :
                extension = pathlib.Path(entry.name).suffix

                if (extension == DEFAULT_MODEL_EXTENSION):
                    # found a model
                    print("found model = ",entry.name)
                    modelList.append(entry.name)

                elif (extension == MODEL_TXT_EXTENSION):
                    # found a dict
                    print("found dict = ",entry.name)
                    dictList.append(entry.name)
                elif (extension == MODEL_BIN_EXTENSION):
                    # found a scaler
                    print("found scaler = ",entry.name)
                    scalerList.append(entry.name)
                elif (extension == MODEL_ENC_EXTENSION):
                    # found an encoder
                    print("found encoder = ",entry.name)
                    encoderList.append(entry.name)
            else:
                if (bDebug):
                    print("File Entry Ignored For ",entry.name)

    if (len(modelList) >0) and (len(modelList) == len(dictList)) and (len(modelList) == len(scalerList)) and (len(modelList) == len(encoderList)):
        bValidData=True


    return bValidData,modelList,dictList,scalerList,encoderList


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

def OpenOptimisationFile():

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
        print("Storing NVSS Image In ",csvFilename)
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

def loadNVSSCSVFile(imageNo):

    filePath =  NVSS_CATALOG_CSV_FILENAME+str(imageNo+1)+ TEXT_FILE_EXTENSION

    bDataValid = True

    if (os.path.isfile(filePath)):

     #   print("*** Loading NVSS CSV File "+filePath+" ***")
        dataframe = pd.read_csv(filePath, header=None)
        dataReturn = dataframe.values
      #  print("*** Completed Loading CSV File")
    else:
        print("*** CSV File Does Not Exist ***")
        bDataValid = False
        dataReturn = []

    return bDataValid,dataReturn



def loadPULSARData():


    filePath =  ORIGINAL_PULSAR_SOURCES_FILENAME
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


def OpenSourcesFile(rootData,sourceClass,sourceLocation,source):

    if (sourceClass == DEFAULT_AGN_CLASS):
        filename = DEFAULT_AGN_SOURCES_FILENAME
    elif (sourceClass == DEFAULT_BLAZAR_CLASS):
        filename = DEFAULT_BLAZAR_SOURCES_FILENAME
    elif (sourceClass == DEFAULT_QUASAR_CLASS):
        filename = DEFAULT_QUASAR_SOURCES_FILENAME
    elif (sourceClass == DEFAULT_SEYFERT_CLASS):
        filename = DEFAULT_SEYFERT_SOURCES_FILENAME
    elif (sourceClass == DEFAULT_PULSAR_CLASS):
        filename = DEFAULT_PULSAR_SOURCES_FILENAME

    else:
        print("*** UNKNOWN Source Class ***")
        sys.exit()


    filename = rootData+DEFAULT_SOURCES_LOCATION+source+UNDERSCORE+filename

    f = open(filename, "w")

    return f

def StoreSourcesToFile(f,sourceType,source, imageList):


    if (f):

        strr = 'Source: '+sourceType+' : '+source+' \n'
        f.write(strr)
        for imageNo in range(len(imageList)):
            strr = 'Image No '+str(imageNo)+' (For Source: '+source+')'+' - '+imageList[imageNo]+' \n'
            print(strr)
            f.write(strr)

    else:
        print("*** Unable To Access SOURCES file ***")
        sys.exit()

def ProcessTransientData(dataSet,sourceClass,maxNumberSamples):

    trainingData = []
    sourceDetails = []

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
        else:
            print("*** Unknown Data Class "+sourceClass+" ***")
            sys.exit()

    sourceList = ScanForSources(sourceLocation)

    numberSources = len(sourceList)

    for source in range(numberSources):

        imageLocation, imageList = ScanForImages(sourceLocation, sourceList[source])

        if (bStoreSourcesToFile):

            fSourceFile = OpenSourcesFile(rootData,sourceClass, sourceLocation, sourceList[source])
            StoreSourcesToFile(fSourceFile,sourceClass, sourceList[source],imageList)
            fSourceFile.close()

        for imageNo in range(len(imageList)):

            bValidData, sourceData = loadCSVFile(sourceLocation, sourceList[source], imageNo)
            if (bValidData):

                if (bShrinkImages):
                    sourceData = np.reshape(sourceData, (1, XSIZE_SMALL_FITS_IMAGE * YSIZE_SMALL_FITS_IMAGE))
                else:
                    sourceData = np.reshape(sourceData, (1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE))

                sourceDetails.append(str(source)+UNDERSCORE+str(imageNo))
                trainingData.append(sourceData)

    if (bConstrainSamples):

        if (maxNumberSamples >0):
            if (len(trainingData) > maxNumberSamples):

                # delete the excess samples

                del trainingData[maxNumberSamples:len(trainingData)]
                del sourceDetails[maxNumberSamples:len(trainingData)]

    print("No of Samples Loaded For "+sourceClass+ " = "+str(len(trainingData)))


    return trainingData,sourceDetails



def createLabels(primaryLabel):


    labelList = [primaryLabel,DEFAULT_OTHER_CLASS]

    OHELabels,labelDict,OHE = createOneHotEncodedSet(labelList)

    return OHELabels,labelDict,labelList,OHE

def decodeLabels(labelList,predictions):



    label = labelList[np.argmax(predictions)]


    return label


def assignLabelSet(label, numberOfSamples):


    shape = (numberOfSamples, len(label))

    a = np.empty((shape))

    a[:]=label

    return a

def assignLabelValue(labelDict, numberOfSamples):

    labelValue = [1.0,0.0]

    shape = (numberOfSamples,len(labelValue))

    a = np.empty((shape))

    a[:]=labelValue

    return a

def createClassicLabelSet(label, numberOfSamples):

    print("creating label")

    if (label == DEFAULT_AGN_CLASS):
        labelValue = DEFAULT_CLASSIC_AGN_CLASS
    elif (label == DEFAULT_BLAZAR_CLASS):
        labelValue = DEFAULT_CLASSIC_BLAZAR_CLASS
    elif (label == DEFAULT_SEYFERT_CLASS):
        labelValue = DEFAULT_CLASSIC_SEYFERT_CLASS
    elif (label == DEFAULT_QUASAR_CLASS):
        labelValue = DEFAULT_CLASSIC_QUASAR_CLASS
    elif (label == DEFAULT_PULSAR_CLASS):
        labelValue = DEFAULT_CLASSIC_PULSAR_CLASS
    else:
        print("*** UNKNOWN CLASS ***")


    a = np.empty((numberOfSamples,))

    a[:]=labelValue

    return a

def decodeClassicLabelSet(labelValue):


    if (labelValue == DEFAULT_CLASSIC_AGN_CLASS):
        label = DEFAULT_AGN_CLASS
    elif (labelValue == DEFAULT_CLASSIC_BLAZAR_CLASS):
        label = DEFAULT_BLAZAR_CLASS
    elif (labelValue == DEFAULT_CLASSIC_SEYFERT_CLASS):
        label = DEFAULT_SEYFERT_CLASS
    elif (labelValue == DEFAULT_CLASSIC_QUASAR_CLASS):
        label = DEFAULT_QUASAR_CLASS
    elif (labelValue == DEFAULT_CLASSIC_PULSAR_CLASS):
        label = DEFAULT_PULSAR_CLASS
    else:
        label = -1
        print("*** UNKNOWN LABEL VALUE ***")

    return label

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

    return dataAsArray


def createFullLabels(labelList):


    OHELabels,labelDict,OHE = createOneHotEncodedSet(labelList)

    return OHELabels,labelDict,OHE


def CreateFullTrainingAndTestData(bNNorClassic,labelList,completeTrainingData,trainingDataSizes):

    finalTrainingData = []
    datasetLabels = []

    #create labels and scale data

    OHELabels, labelDict,OHE = createFullLabels(labelList)

    print("no of datasets = ",len(completeTrainingData))
    print("labelDict = ",labelDict)

    for dataset in range(len(completeTrainingData)):

        print("length of dataset = ",trainingDataSizes[dataset])

        dataAsArray = TransformTrainingData(completeTrainingData[dataset])
        print("shape = ",dataAsArray.shape)

        datasetLabels.append(np.asarray(assignLabelSet(OHELabels[dataset], trainingDataSizes[dataset])))

        finalTrainingData.append(dataAsArray)

    # create the training and test sets

    combinedTrainingSet = []
    combinedTestSet = []
    combinedTrainingLabels = []
    combinedTestLabels = []

    # scale the data

    joinedTrainingData = finalTrainingData[0]
    for dataset in range(1,len(finalTrainingData)):
        classTrainingData = finalTrainingData[dataset]
        joinedTrainingData = np.concatenate((joinedTrainingData,classTrainingData))

    joinedTrainingData,scaler = ScaleInputData(joinedTrainingData)

    #now split back up again

    startPos = 0
    for dataset in range(len(finalTrainingData)):

        finalTrainingData[dataset] = joinedTrainingData[startPos:startPos+trainingDataSizes[dataset]]
        startPos = startPos+trainingDataSizes[dataset]


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



    index = np.random.choice(XTrain.shape[0], len(XTrain), replace=False)

    XTrain = XTrain[index]
    ytrain = ytrain[index]

    index = np.random.choice(XTest.shape[0], len(XTest), replace=False)

    XTest = XTest[index]
    ytest = ytest[index]

    if (bNNorClassic):
        XTrain = np.reshape(XTrain, (XTrain.shape[0], XTrain.shape[1], 1))
        XTest = np.reshape(XTest, (XTest.shape[0], XTest.shape[1], 1))


    return XTrain, XTest, ytrain, ytest,labelDict,scaler,labelList,OHE


def CreateTrainingAndTestData(bNNorClassic,primaryLabel,completeTrainingData,trainingDataSizes):

    OHELabels,labelDict,labelList,OHE = createLabels(primaryLabel)

    datasetLabels = []
    finalTrainingData = []


    #create labels and scale data


    print("no of datasets = ",len(completeTrainingData))

    for dataset in range(len(completeTrainingData)):

   #     print("length of dataset = ",trainingDataSizes[dataset])

        dataAsArray = TransformTrainingData(completeTrainingData[dataset])

     #   datasetLabels.append(createClassicLabelSet(labelList[dataset],trainingDataSizes[dataset]))

        datasetLabels.append(np.asarray(assignLabelSet(OHELabels[dataset], trainingDataSizes[dataset])))

        finalTrainingData.append(dataAsArray)

    # create the training and test sets

    combinedTrainingSet = []
    combinedTestSet = []
    combinedTrainingLabels = []
    combinedTestLabels = []

    # scale the data

    joinedTrainingData = finalTrainingData[0]
    for dataset in range(1,len(finalTrainingData)):
        classTrainingData = finalTrainingData[dataset]
        joinedTrainingData = np.concatenate((joinedTrainingData,classTrainingData))

    joinedTrainingData,scaler = ScaleInputData(joinedTrainingData)

    #now split back up again
 #   print("len joinedTrainingData = ",len(joinedTrainingData))


    startPos = 0
    for dataset in range(len(finalTrainingData)):

        finalTrainingData[dataset] = joinedTrainingData[startPos:startPos+trainingDataSizes[dataset]]
    #    print(finalTrainingData[dataset])
        startPos = startPos+trainingDataSizes[dataset]


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


    if (bNNorClassic):
        XTrain = np.reshape(XTrain, (XTrain.shape[0], XTrain.shape[1], 1))
        XTest = np.reshape(XTest, (XTest.shape[0], XTest.shape[1], 1))



    index = np.random.choice(XTrain.shape[0], len(XTrain), replace=False)

    XTrain = XTrain[index]
    ytrain = ytrain[index]

    index = np.random.choice(XTest.shape[0], len(XTest), replace=False)

    XTest = XTest[index]
    ytest = ytest[index]

    #  if (bDebug):
    print("Final Training Data = ", XTrain.shape)
    print("Final Test Data = ", XTest.shape)

    print("Final Training Label Data = ", ytrain.shape)
    print("Final Test Label Data = ", ytest.shape)

    return XTrain, XTest, ytrain, ytest,labelDict,scaler,labelList,OHE





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




def RandomForestModel(XTrain,ytrain,XTest,ytest):

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    rndClf = RandomForestClassifier(n_estimators=30,max_depth=9,min_samples_leaf=15)

    rndClf.fit(XTrain,ytrain)
    y_pred = rndClf.predict(XTest)
    print(rndClf.__class__.__name__,accuracy_score(ytest,y_pred))

    return rndClf




def NaiveBayesModel(XTrain,ytrain,XTest,ytest):

    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    gnb = GaussianNB()

    gnb.fit(XTrain,ytrain)
    y_pred = gnb.predict(XTest)
    print(gnb.__class__.__name__,accuracy_score(ytest,y_pred))

    return gnb


def SelectDataset():

    bCorrectInput=False

    datasetChoice = ['NVSS', 'VAST']
    shortDataSetChoice = [NVSS_SHORT_NAME, VAST_SHORT_NAME]

    while (bCorrectInput == False):
        dataSet = input(
            'Choose Dataset For Model ' + datasetChoice[0] + ',' + datasetChoice[1] + '  (' + datasetChoice[0][
                0] + '/' + datasetChoice[1][0] + ') :')
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

    return dataClass

def GetBinaryLabels(testClass):


    if (testClass == DEFAULT_AGN_CLASS):

        otherLabels = [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS]
    elif (testClass == DEFAULT_PULSAR_CLASS):

        otherLabels = [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_AGN_CLASS, DEFAULT_QUASAR_CLASS]

    elif (testClass == DEFAULT_QUASAR_CLASS):

        otherLabels = [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_AGN_CLASS]
    elif (testClass == DEFAULT_BLAZAR_CLASS):
        otherLabels = [DEFAULT_SEYFERT_CLASS, DEFAULT_AGN_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS]

    elif (testClass == DEFAULT_SEYFERT_CLASS):

        otherLabels = [DEFAULT_AGN_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS]


    return otherLabels


def ChooseDataSetsToTest():

    bCorrectInput=False

    choiceList = ["AGN(A)","SEYFERT(S)","BLAZAR(B)", "QUASAR(Q)", "PULSAR (P)"]

    while (bCorrectInput==False):
        strr = 'Select '+choiceList[0]+', '+choiceList[1]+', '+choiceList[2]+', '+choiceList[3]+', '+choiceList[4]+' : '
        classData = input(strr)
        classData= classData.upper()
        bCorrectInput=True
        if (classData == AGN_DATA_SELECTED):
            classLabel = DEFAULT_AGN_CLASS
        elif (classData == SEYFERT_DATA_SELECTED):
            classLabel = DEFAULT_SEYFERT_CLASS
        elif (classData == BLAZAR_DATA_SELECTED):
            classLabel = DEFAULT_BLAZAR_CLASS
        elif (classData == PULSAR_DATA_SELECTED):
            classLabel = DEFAULT_PULSAR_CLASS
        elif (classData == QUASAR_DATA_SELECTED):
            classLabel = DEFAULT_QUASAR_CLASS

        else:
            print("Invalid Input - try again")
            bCorrectInput=False

    return classLabel

def GetSelectedBinaryDataSets():

    classLabels= []
    bCorrectInput=False
    otherLabels = []

    choiceList = ["AGN(A)","SEYFERT(S)","BLAZAR(B)", "QUASAR(Q)", "PULSAR (P)"]
 #   choiceList = ["SEYFERT(S)","BLAZAR(B)", "QUASAR(Q)", "PULSAR (P)"]

    dataClass = SelectDataset()

    while (bCorrectInput==False):
        strr = 'Select '+choiceList[0]+', '+choiceList[1]+', '+choiceList[2]+', '+choiceList[3]+', '+choiceList[4]+' : '
    #    strr = 'Select '+choiceList[0]+', '+choiceList[1]+', '+choiceList[2]+', '+choiceList[3]+' : '
        classData = input(strr)
        classData= classData.upper()
        if (classData == AGN_DATA_SELECTED):
            classLabels.append(DEFAULT_AGN_CLASS)
            otherLabels = [DEFAULT_SEYFERT_CLASS,DEFAULT_BLAZAR_CLASS,DEFAULT_PULSAR_CLASS,DEFAULT_QUASAR_CLASS]
            bCorrectInput = True
        elif (classData == SEYFERT_DATA_SELECTED):

            classLabels.append(DEFAULT_SEYFERT_CLASS)
            otherLabels = [DEFAULT_AGN_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS,DEFAULT_QUASAR_CLASS]
       #     otherLabels = [ DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS,DEFAULT_QUASAR_CLASS]
            bCorrectInput = True
        elif (classData == BLAZAR_DATA_SELECTED):
            classLabels.append(DEFAULT_BLAZAR_CLASS)
            otherLabels = [DEFAULT_SEYFERT_CLASS, DEFAULT_AGN_CLASS, DEFAULT_PULSAR_CLASS,DEFAULT_QUASAR_CLASS]
        #    otherLabels = [DEFAULT_SEYFERT_CLASS, DEFAULT_PULSAR_CLASS,DEFAULT_QUASAR_CLASS]
            bCorrectInput = True
        elif (classData == QUASAR_DATA_SELECTED):

            classLabels.append(DEFAULT_QUASAR_CLASS)
            otherLabels = [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS,DEFAULT_AGN_CLASS]
        #    otherLabels = [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS]
            bCorrectInput = True

        elif (classData == PULSAR_DATA_SELECTED):

            classLabels.append(DEFAULT_PULSAR_CLASS)
            otherLabels= [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_AGN_CLASS,DEFAULT_QUASAR_CLASS]
        #    otherLabels= [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS,DEFAULT_QUASAR_CLASS]
            bCorrectInput = True
        else:
            bCorrectInput = False

    return dataClass,classLabels,otherLabels

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
        print("1:  Build Multi-Classifier Models")
        print("2:  Build Binary Models")
        print("3:  Test Existing Models")
        print("4:  Blind Test Models")

        selOP = input("Select Operation:")
        if (selOP == OP_BUILD_FULL_MODEL) or (selOP == OP_BUILD_BINARY_MODELS) or (selOP == OP_TEST_MODEL) or (selOP == OP_TEST_BLIND):
            bCorrectInput = True
        else:
            print("*** Incorrect Input - try again ***")

    return selOP




def DecodePredictedLabel(labelDict,prediction,predProbability):

    bDetection=False


    for predictionElement in prediction:

        predictVal =predictionElement>0

        if (predictVal.any() == True):

            bDetection=True


            val = int(np.argmax(predictionElement))

            label = labelDict[np.argmax(predictionElement)]

            finalProbability = max(max(predProbability[val]))



        else:

            return bDetection,-1


    return bDetection,label,finalProbability





def TestIndividualFile(model,fileData):


    fileData = fileData.reshape(1, -1)

    y_pred = model.predict(fileData)
    if (bDebug):
        print("y_pred =", y_pred)

    y_pred_proba = model.predict_proba(fileData)
    if (bDebug):
     print("y_pred_proba =", y_pred_proba)

    return y_pred,y_pred_proba


def DecodeProbabilities(labelDict,classProb):

    print("number of class probabilities  = ",len(classProb))
    print("number of classes = ", len(labelDict))

    print("class prob = ",classProb)

    maxProb = max(classProb[0])
    print("max probability = ",maxProb)
    return maxProb


def TestRawRandomSamples(labelDict,model,randomSamples,sampleLabels,dataScaler):

    rawTestSamples = []

    print("*** Testing Unseen RAW samples ***")

    # test model with random samples of raw image data


    dataAsArray = np.asarray(randomSamples)
    dataAsArray = np.reshape(dataAsArray, (dataAsArray.shape[0], dataAsArray.shape[2]))

    numberCorrectPredictions = 0
    numberIncorrectPredictions = 0
    for sample in range(len(dataAsArray)):

    #    print("** FOR SAMPLE NO : ",sample)

        x = dataAsArray[sample]

   #     print(x.shape)

        x = x.reshape(1, -1)

        scaledData = dataScaler.transform(x)

        pred,predProbability = TestIndividualFile(model, scaledData)

        bDetection,predictedlabel,finalProbability = DecodePredictedLabel(labelDict, pred,predProbability)

        if (bDetection):
            print("predicted label =",predictedlabel)
            print("sample label =",sampleLabels[sample])

            if (predictedlabel==sampleLabels[sample]):
                numberCorrectPredictions += 1
            else:
                numberIncorrectPredictions +=1
   #     print("versus ACTUAL label", sampleLabels[sample])


    print("No Correct Predictions = ",numberCorrectPredictions)
    print("No Incorrect Predictions = ", numberIncorrectPredictions)



    return rawTestSamples,sampleLabels


def TestRawSample(labelDict,model,sampleData,dataScaler):


    dataAsArray = np.asarray(sampleData)
    dataAsArray = dataAsArray.reshape(1, -1)
    scaledData = dataScaler.transform(dataAsArray)

    pred,predProb = TestIndividualFile(model, scaledData)

    bDetection,predictedLabel,finalProbability = DecodePredictedLabel(labelDict, pred,predProb)

    return bDetection,predictedLabel,finalProbability



def DecodePossibilities(className,testPossibilities):
    noSinglePossibilities = 0
    noDoublePossibilities = 0
    noTriplePossibilities = 0
    noQuadPossibilities = 0

    for entry in range(len(testPossibilities)):
        if (testPossibilities[entry] == 1):
            noSinglePossibilities += 1
        elif (testPossibilities[entry] == 2):
            noDoublePossibilities +=1
        elif (testPossibilities[entry] == 3):
             noTriplePossibilities +=1
        elif (testPossibilities[entry] == 4):

            noQuadPossibilities +1

    print("For class "+className)
    print('total no. of Single possibilities = '+str(noSinglePossibilities))
    print('total no. of Double possibilities = ' + str(noDoublePossibilities))
    print('total no. of Triple possibilities = ' + str(noTriplePossibilities))
    print('total no. of Quad possibilities = ' + str(noQuadPossibilities))

def StoreResultsPossibilities(f,testPossibilities):
    noSinglePossibilities = 0
    noDoublePossibilities = 0
    noTriplePossibilities = 0
    noQuadPossibilities = 0

    for entry in range(len(testPossibilities)):
        if (testPossibilities[entry] == 1):
            noSinglePossibilities += 1
        elif (testPossibilities[entry] == 2):
            noDoublePossibilities +=1
        elif (testPossibilities[entry] == 3):
             noTriplePossibilities +=1
        elif (testPossibilities[entry] == 4):

            noQuadPossibilities +1

    if (f):

        f.write('Total no. of Single possibilities = '+str(noSinglePossibilities))
        f.write('\n')
        f.write('total no. of Double possibilities = ' + str(noDoublePossibilities))
        f.write('\n')
        f.write('total no. of Triple possibilities = ' + str(noTriplePossibilities))
        f.write('\n')
        f.write('total no. of Quad possibilities = ' + str(noQuadPossibilities))
        f.write('\n')

    else:
        print("Invalid Sumamry File Handle ")
        sys.exit()


def ConvertClassTypeToClassValue(classType):

    if (classType ==  DEFAULT_BLAZAR_CLASS):
        classValue = BLAZAR_DATA_SELECTED

    elif (classType == DEFAULT_AGN_CLASS):
        classValue = AGN_DATA_SELECTED

    elif (classType == DEFAULT_QUASAR_CLASS):
        classValue = QUASAR_DATA_SELECTED

    elif (classType == DEFAULT_PULSAR_CLASS):
        classValue = PULSAR_DATA_SELECTED

    elif (classType == DEFAULT_SEYFERT_CLASS):
        classValue = SEYFERT_DATA_SELECTED

    else:
        print("*** UNKNOWN Class Type ***")
        sys.exit()

    return classValue




def ExecuteSavedBinaryModel(testData, finalBinaryModelDict, classType1, classType2, modelList, dictList,scalerList):



    classValue1 = ConvertClassTypeToClassValue(classType1)
    classValue2 = ConvertClassTypeToClassValue(classType2)


    modelName = classValue1 + classValue2

    if (modelName in finalBinaryModelDict):
        modelNo = finalBinaryModelDict[modelName]

        bDetection, predictedLabel, finalProbability = TestRawSample(dictList[modelNo], modelList[modelNo],
                                                                     testData,
                                                                     scalerList[modelNo])

    else:
        print("*** UNKNOWN MODEL TYPE " + class1 + class2 + " ***")
        sys.exit()

    return bDetection,predictedLabel,finalProbability

def LoadMultiClassModels(dataset):
    import os

    models = []
    dicts = []
    scalers = []
    encoders = []

    finalBinaryModelDict = {}

    print("*** Building Full Binary Model Dictionary ***")

    if (dataset == DEFAULT_NVSS_DATASET):
        modelLocation =NVSS_FULL_BINARY_MODELS_LOCATION
    elif (dataset == DEFAULT_VAST_DATASET):
        modelLocation = VAST_FULL_BINARY_MODELS_LOCATION
    else:
        print("*** UNKNOWN DAT SET To extract Models ***")
        sys.exit()
    bValidData, modelList, dictList, scalerList,encoderList = ScanForModels(modelLocation)

    if (bValidData):

        print("*** Total No Models = " + str(len(modelList)) + " ***")
        for entry in range(len(modelList)):
            model = GetSavedModel(modelLocation, modelList[entry])
            bDictOK,dict = GetSavedDict(modelLocation, modelList[entry])
            if (bDictOK):
                dicts.append(dict)
            else:
                print("*** FAILED TO LOAD FULL BINARY DICT ***")
                sys.exit()

            scaler = GetSavedScaler(modelLocation, modelList[entry])
            encoder = GetSavedEncoder(modelLocation, modelList[entry])

            models.append(model)
            scalers.append(scaler)
            encoders.append(encoder)

    else:
        print("*** No FULL BINARY Models Could Be Found ***")
        sys.exit()


    for entry in range(len(modelList)):
        rootExt = os.path.splitext(modelList[entry])

        modelParams = rootExt[0].split(UNDERSCORE)

        finalBinaryModelDict[modelParams[1] + modelParams[2]] = entry
        finalBinaryModelDict[modelParams[2] + modelParams[1]] = entry

    return finalBinaryModelDict, models, dicts, scalers,encoders


def TestForAlternatives(fOutputFile,testData,testDataDetails,dictNo,dictList,testModelList,testClass,scalerList,classList,finalBinaryModelDict, fullModelList, fullDictList, fullScalerList):

    testPossibilities = []

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    numberAltFN = 0


    numberChangedPredictions = 0

    for dataEntry in range(len(testData)):

        noPossibilities = 0
        bChangedPrediction = False
        bFalseNegative=False

        sourceNo = testDataDetails[dataEntry]
        StoreHeaderInFile(fOutputFile, sourceNo, testClass, testClass)

        bDetection, predictedLabel, finalProbability = TestRawSample(dictList[dictNo], testModelList[dictNo],
                                                                      testData[dataEntry],
                                                                      scalerList[dictNo])

        if (bDetection):

            StoreModelResultsInFile(fOutputFile,sourceNo, testClass, testClass, predictedLabel, finalProbability)


            if (predictedLabel==testClass):
                noPossibilities+=1
                TP += 1
                probCorrectPrediction = finalProbability
                outcomeProbability = finalProbability
                outcomeClass = testClass
                bFalseNegative = False

            else:

                FN += 1
                outcomeClass = predictedLabel
                outcomeProbability = finalProbability
                probCorrectPrediction = 0
                bFalseNegative=True
        else:
            print("*** INVALID RESULT ***")
            sys.exit()
        # now try this data against all other models

        for modelNo in range(len(testModelList)):
            if (modelNo != dictNo):

                bDetection, predictedLabel,finalProbability = TestRawSample(dictList[modelNo], testModelList[modelNo],
                                                                            testData[dataEntry],
                                                                            scalerList[modelNo])


                if (bDetection == True):

                    StoreModelResultsInFile(fOutputFile, sourceNo, testClass, classList[modelNo], predictedLabel,
                                            finalProbability)
                    # possible conflict

                    if (predictedLabel != DEFAULT_OTHER_CLASS):

                        # this is where an alternative other than 'other' has been identified

                        if (finalProbability > probCorrectPrediction):

                            # this is a stronger candidate

                            outcomeProbability = finalProbability
                            outcomeClass = classList[modelNo]
                            noPossibilities+=1

                            # can we determine most likely of these candidates ?
                            if (bLoadFullBinaryModels):
                                bFullDetection, fullPredLabel, fullPredProb = ExecuteSavedBinaryModel(testData[dataEntry], finalBinaryModelDict, testClass,outcomeClass,
                                                                                fullModelList, fullDictList, fullScalerList)

                                if (bFullDetection):
                                    if (fullPredLabel != testClass):

                                        # otherwise it is a true FP

                                        FP += 1
                                    else:
                                        # it is detecting our test class

                                        numberChangedPredictions += 1
                                        bChangedPrediction=True

                                        #### TEST !
                                        if (bReduceFalseNegatives) and (bFalseNegative):
                                            FN -= 1




                            else:
                                FP += 1
                        else:
                            TN += 1
                    else:
                        TN += 1


        StoreOutcomeInFile(fOutputFile,outcomeClass,outcomeProbability,bChangedPrediction)

        testPossibilities.append(noPossibilities)

    return testPossibilities,TP,FN,FP,TN,numberChangedPredictions



def findResultantClass(probability,predictionProbabilities,predictionModels,classList):

    index = predictionProbabilities.index(probability)
    modelNo = predictionModels[index]
    return classList[modelNo]

def CalculateMeanValue(dataSample):
    import statistics

    bStatsCheckOK = False

    meanValue = statistics.mean(dataSample[0])

    return meanValue


def TestAgainstStats(dataSample, mean, stdev):
    import statistics

    bStatsCheckOK = False

    meanValue = statistics.mean(dataSample[0])

    if ((meanValue >= mean - stdev) and (meanValue <= mean + stdev)):
        bStatsCheckOK = True

    return bStatsCheckOK


def TestImageAgainstStats(numberPosStatSuccess,numberPosStatFailures,dataSample,classMean,classStdev):

    numberPosStatS = numberPosStatSuccess
    numberPosStatF = numberPosStatFailures

    bCheckStats = TestAgainstStats(dataSample, classMean, classStdev)
    if (bCheckStats):
        numberPosStatS += 1
    else:
        numberPosStatF += 1

    return numberPosStatS, numberPosStatF


def ConvertToClassLabel(possibleClasses):



    classes = []
    for i in range(len(possibleClasses)):
        if (possibleClasses[i]== AGN_DATA_SELECTED):
            classes.append(DEFAULT_AGN_CLASS)
        elif (possibleClasses[i]== PULSAR_DATA_SELECTED):
            classes.append(DEFAULT_PULSAR_CLASS)
        elif (possibleClasses[i]== BLAZAR_DATA_SELECTED):
            classes.append(DEFAULT_BLAZAR_CLASS)
        elif (possibleClasses[i] == SEYFERT_DATA_SELECTED):
            classes.append(DEFAULT_SEYFERT_CLASS)
        elif (possibleClasses[i] == QUASAR_DATA_SELECTED):
            classes.append(DEFAULT_QUASAR_CLASS)


    return classes[0],classes[1]


def TestIndividualSampleSet(fOutputFile,testData,testDataDetails,dictNo,dictList,testModelList,testClass,scalerList,classList,finalBinaryModelDict, fullModelList,
                            fullDictList, fullScalerList,meanDict,stddevDict):

    TP = 0
    FN = 0

    numberNegStatSuccess = 0
    numberNegStatFailures = 0
    numberPosStatSuccess = 0
    numberPosStatFailures = 0

    StatDict = {}
    StatDict['POSSUCC'] = 0
    StatDict['NEGSUCC'] = 0
    StatDict['POSFAIL'] = 0
    StatDict['NEGFAIL'] = 0

    TPDict= {}

    TPDict['TPCase1'] =0
    TPDict['TPCase2'] = 0
    TPDict['TPCase3'] = 0
    TPDict['TPCase4'] = 0
    TPDict['TPCase5'] = 0

    TPDict['FNCase1'] = 0
    TPDict['FNCase2'] = 0
    TPDict['FNCase3'] = 0
    TPDict['FNCase4'] = 0
    TPDict['FNCase5'] = 0



    for dataEntry in range(len(testData)):

        truePredictionProbabilities = []
        truePredictionModels = []
        truePredictionLabels = []

        otherPredictionProbabilities = []
        otherPredictionModels = []

        for modelNo in range(len(testModelList)):

            bDetection, predictedLabel,thisProbability = TestRawSample(dictList[modelNo], testModelList[modelNo],
                                                                            testData[dataEntry],
                                                                            scalerList[modelNo])


            if (bDetection == True):

                if (predictedLabel != DEFAULT_OTHER_CLASS):

                    truePredictionProbabilities.append(thisProbability)
                    truePredictionLabels.append(predictedLabel)
                    truePredictionModels.append(modelNo)
                else:
                    otherPredictionProbabilities.append(thisProbability)
                    otherPredictionModels.append(modelNo)

            else:
                print("INVALID Prediction")
                sys.exit()



        numberTruePredictions = len(truePredictionProbabilities)
        numberOtherPredictions = len(otherPredictionProbabilities)

        if (numberTruePredictions>0):
            sortedTruePredictionProbabilities = sorted(truePredictionProbabilities, reverse=True)
        else:
            sortedTruePredictionProbabilities = truePredictionProbabilities
        if (numberOtherPredictions > 0):
            sortedOtherPredictionProbabilities = sorted(otherPredictionProbabilities)
        else:
            sortedOtherPredictionProbabilities =otherPredictionProbabilities

        if (numberTruePredictions==1):

            if (truePredictionLabels[0] == testClass):

                # ok - it has predicted the right class

                TP += 1

                TPDict['TPCase1'] += 1

                # now check the stats

                numberPosStatSuccess,numberPosStatFailures = TestImageAgainstStats(numberPosStatSuccess,numberPosStatFailures,testData[dataEntry],
                                                                                   meanDict[testClass],stddevDict[testClass])



            else:


                possibleClass1 = findResultantClass(sortedTruePredictionProbabilities[0], truePredictionProbabilities,
                                                    truePredictionModels, classList)
                possibleClass2 = findResultantClass(sortedOtherPredictionProbabilities[0], otherPredictionProbabilities,
                                                    otherPredictionModels, classList)


                bFullDetection, fullPredLabel, fullPredProb = ExecuteSavedBinaryModel(testData[dataEntry],
                                                                                      finalBinaryModelDict,
                                                                                      possibleClass1, possibleClass2,
                                                                                      fullModelList,
                                                                                      fullDictList, fullScalerList)

                if (bFullDetection):

                    if (fullPredLabel == testClass):

                        TPDict['TPCase2'] += 1

                        TP +=1


                        numberPosStatSuccess, numberPosStatFailures = TestImageAgainstStats(numberPosStatSuccess,
                                                                                            numberPosStatFailures,
                                                                                            testData[dataEntry],
                                                                                            meanDict[testClass],
                                                                                            stddevDict[testClass])


                    else:

                        TPDict['FNCase2'] += 1
                        FN +=1

                        numberNegStatSuccess, numberNegStatFailures = TestImageAgainstStats(numberNegStatSuccess,
                                                                                        numberNegStatFailures,
                                                                                        testData[dataEntry],
                                                                                        meanDict[fullPredLabel],
                                                                                        stddevDict[fullPredLabel])


                else:
                    print("FAILED TO GET A PREDICTION")
                    sys.exit()



        elif (numberTruePredictions >= 2):

            if (bTakeTop2Models):


                possibleClass1 = findResultantClass(sortedTruePredictionProbabilities[0], truePredictionProbabilities,
                                                          truePredictionModels, classList)
                possibleClass2 = findResultantClass(sortedTruePredictionProbabilities[1], truePredictionProbabilities,
                                                          truePredictionModels, classList)

                bFullDetection, fullPredLabel, fullPredProb = ExecuteSavedBinaryModel(testData[dataEntry],
                                                                                  finalBinaryModelDict,
                                                                                  possibleClass1, possibleClass2,
                                                                                  fullModelList,
                                                                                  fullDictList, fullScalerList)

                if (bFullDetection):

                    if (fullPredLabel==testClass):
                        TP +=1

                        TPDict['TPCase3'] += 1

                        numberPosStatSuccess, numberPosStatFailures = TestImageAgainstStats(numberPosStatSuccess,
                                                                                            numberPosStatFailures,
                                                                                            testData[dataEntry],
                                                                                            meanDict[testClass],
                                                                                            stddevDict[testClass])


                    else:
                        FN +=1

                        TPDict['FNCase3'] += 1

                        numberNegStatSuccess, numberNegStatFailures = TestImageAgainstStats(numberNegStatSuccess,
                                                                                            numberNegStatFailures,
                                                                                            testData[dataEntry],
                                                                                            meanDict[fullPredLabel],
                                                                                            stddevDict[fullPredLabel])


                else:

                    print("FAILED TO GET A PREDICTION")
                    sys.exit()

            else:


                allPossibleClasses = GenerateListOfPairsToTest(truePredictionLabels)
                possibleClasses = list(allPossibleClasses.keys())


                predLabelsList = []
                predProbList = []

                for classPair in range(len(possibleClasses)):

                        possibleClass1,possibleClass2 = ConvertToClassLabel(possibleClasses[classPair])

                        bFullDetection, fullPredLabel, fullPredProb = ExecuteSavedBinaryModel(testData[dataEntry],
                                                                                          finalBinaryModelDict,
                                                                                          possibleClass1, possibleClass2,
                                                                                          fullModelList,
                                                                                          fullDictList, fullScalerList)

                        if (bFullDetection):

                            # add to a list

                            predLabelsList.append(fullPredLabel)
                            predProbList.append(fullPredProb)

                        else:
                            print("FAILED TO GET A PREDICTION")
                            sys.exit()

                if (len(predProbList) >0):

                    # we have a potential result - find the highest probability and its corresponding label

                    highestProb = max(predProbList)

                    highestClassIndex = predProbList.index(highestProb)
                    highestClass = predLabelsList[highestClassIndex]
                    if (highestClass==testClass):

                        TP+=1
                        StatDict['TPCase4'] += 1

                        numberPosStatSuccess, numberPosStatFailures = TestImageAgainstStats(numberPosStatSuccess,
                                                                                            numberPosStatFailures,
                                                                                            testData[dataEntry],
                                                                                            meanDict[testClass],
                                                                                            stddevDict[testClass])

                    else:
                        FN +=1
                        StatDict['FNCase4'] += 1


                        numberNegStatSuccess, numberNegStatFailures = TestImageAgainstStats(numberNegStatSuccess,
                                                                                            numberNegStatFailures,
                                                                                            testData[dataEntry],
                                                                                            meanDict[fullPredLabel],
                                                                                            stddevDict[fullPredLabel])

        elif ((numberTruePredictions==0) and (numberOtherPredictions>0)):

            possibleClass1 = findResultantClass(sortedOtherPredictionProbabilities[0],otherPredictionProbabilities,otherPredictionModels,classList)

            possibleClass2 = findResultantClass(sortedOtherPredictionProbabilities[1],otherPredictionProbabilities,otherPredictionModels,classList)

            bFullDetection, fullPredLabel, fullPredProb = ExecuteSavedBinaryModel(testData[dataEntry],
                                                                                  finalBinaryModelDict,
                                                                                  possibleClass1, possibleClass2,
                                                                                  fullModelList,
                                                                                  fullDictList, fullScalerList)

            if (bFullDetection):
                if (fullPredLabel == testClass):

                    # ok - it has predicted the right class

                    TP += 1
                    TPDict['TPCase5'] += 1


                    numberPosStatSuccess, numberPosStatFailures = TestImageAgainstStats(numberPosStatSuccess,
                                                                                        numberPosStatFailures,
                                                                                        testData[dataEntry],
                                                                                        meanDict[testClass],
                                                                                        stddevDict[testClass])


                else:

                    FN += 1

                    TPDict['FNCase5'] += 1

                    numberNegStatSuccess, numberNegStatFailures = TestImageAgainstStats(numberNegStatSuccess,
                                                                                    numberNegStatFailures,
                                                                                    testData[dataEntry],
                                                                                    meanDict[fullPredLabel],
                                                                                    stddevDict[fullPredLabel])


            else:
                print("FAILED TO GET A PREDICTION")
                sys.exit()

        else:
            print("NO TRUE OR OTHER PREDICTIONS ")
            sys.exit()

        truePredictionProbabilities.clear()
        truePredictionModels.clear()
        truePredictionLabels.clear()

        otherPredictionProbabilities.clear()
        otherPredictionModels.clear()

    print("For test class = " + testClass + " TP = " + str(TP) + ", FN = " + str(FN))
    print("Total No Samples tested = ",str(len(testData)))
    accuracy = round((TP/len(testData)),2)
    print("Accuracy = ",str(accuracy))

    StatDict['POSSUCC'] =numberPosStatSuccess
    StatDict['POSFAIL'] =numberPosStatFailures
    StatDict['NEGSUCC']= numberNegStatSuccess
    StatDict['NEGFAIL'] =numberNegStatFailures


    print("No Pos Stat Success = ",str(StatDict['POSSUCC']))
    print("No Pos Stat Failures = ",str(StatDict['POSFAIL']))
    print("No Neg Stat Success = ", str(StatDict['NEGSUCC']))
    print("No Neg Stat Failures = ",str(StatDict['NEGFAIL']))

    print("No TP (Case 1) = ", str(TPDict['TPCase1']))
    print("No TP (Case 2) = ", str(TPDict['TPCase2']))
    print("No TP (Case 3) = ", str(TPDict['TPCase3']))
    print("No TP (Case 4) = ", str(TPDict['TPCase4']))
    print("No TP (Case 5) = ", str(TPDict['TPCase5']))

    print("No FN (Case 1) = ", str(TPDict['FNCase1']))
    print("No FN (Case 2) = ", str(TPDict['FNCase2']))
    print("No FN (Case 3) = ", str(TPDict['FNCase3']))
    print("No FN (Case 4) = ", str(TPDict['FNCase4']))
    print("No FN (Case 5) = ", str(TPDict['FNCase5']))

    return TP, FN,StatDict,TPDict

def ClassifyAnImage(testData,dictList,testModelList,scalerList,classList,finalBinaryModelDict, fullModelList,fullDictList, fullScalerList):



    truePredictionProbabilities = []
    truePredictionModels = []
    truePredictionLabels = []

    otherPredictionProbabilities = []
    otherPredictionModels = []

    for modelNo in range(len(testModelList)):

        bDetection, predictedLabel,thisProbability = TestRawSample(dictList[modelNo], testModelList[modelNo],
                                                                            testData,
                                                                            scalerList[modelNo])


        if (bDetection == True):

            if (predictedLabel != DEFAULT_OTHER_CLASS):

                truePredictionProbabilities.append(thisProbability)
                truePredictionLabels.append(predictedLabel)
                truePredictionModels.append(modelNo)
            else:
                otherPredictionProbabilities.append(thisProbability)
                otherPredictionModels.append(modelNo)

        else:
            print("INVALID Prediction")
            sys.exit()



    numberTruePredictions = len(truePredictionProbabilities)
    numberOtherPredictions = len(otherPredictionProbabilities)

    if (numberTruePredictions>0):
        sortedTruePredictionProbabilities = sorted(truePredictionProbabilities, reverse=True)
    else:
        sortedTruePredictionProbabilities = truePredictionProbabilities
    if (numberOtherPredictions > 0):
        sortedOtherPredictionProbabilities = sorted(otherPredictionProbabilities)
    else:
        sortedOtherPredictionProbabilities =otherPredictionProbabilities

    if (numberTruePredictions==1):
        print("*** ONE Prediction Only ***")
        identifiedClass =truePredictionLabels[0]
        transientProb = sortedTruePredictionProbabilities[0]
        possibleClass = 0
    elif (numberTruePredictions >= 2):

        print("*** TWO or MORE Predictions ***")
        possibleClass1 = findResultantClass(sortedTruePredictionProbabilities[0], truePredictionProbabilities,
                                                          truePredictionModels, classList)
        possibleClass2 = findResultantClass(sortedTruePredictionProbabilities[1], truePredictionProbabilities,
                                                          truePredictionModels, classList)

        bFullDetection, fullPredLabel, fullPredProb = ExecuteSavedBinaryModel(testData,
                                                                                  finalBinaryModelDict,
                                                                                  possibleClass1, possibleClass2,
                                                                                  fullModelList,
                                                                                  fullDictList, fullScalerList)

        if (bFullDetection):

            identifiedClass = fullPredLabel
            if (fullPredLabel==possibleClass1):
                possibleClass = possibleClass2
            else:
                possibleClass = possibleClass1
            transientProb = fullPredProb

        else:

            print("FAILED TO GET A PREDICTION")
            sys.exit()


    elif ((numberTruePredictions==0) and (numberOtherPredictions>0)):

        print("*** NO TRUE Predictions ***")

        possibleClass1 = findResultantClass(sortedOtherPredictionProbabilities[0],otherPredictionProbabilities,otherPredictionModels,classList)

        possibleClass2 = findResultantClass(sortedOtherPredictionProbabilities[1],otherPredictionProbabilities,otherPredictionModels,classList)

        bFullDetection, fullPredLabel, fullPredProb = ExecuteSavedBinaryModel(testData,
                                                                                  finalBinaryModelDict,
                                                                                  possibleClass1, possibleClass2,
                                                                                  fullModelList,
                                                                                  fullDictList, fullScalerList)

        if (bFullDetection):

            identifiedClass = fullPredLabel
            transientProb = fullPredProb
            if (fullPredLabel==possibleClass1):
                possibleClass = possibleClass2
            else:
                possibleClass = possibleClass1

        else:
            print("FAILED TO GET A PREDICTION")
            sys.exit()

    else:
        print("NO TRUE OR OTHER PREDICTIONS ")
        sys.exit()


    return identifiedClass,transientProb,possibleClass



def SoakTestFullModel(testData,model,testClass,scaler,labelDict):


    noCorrect = 0
    noIncorrect = 0

    print('*** Soak Testing '+testClass+' ***')
    for dataEntry in range(len(testData)):


        bDetection, predictedLabel, finalProbability = TestRawSample(labelDict, model,
                                                                      testData[dataEntry],
                                                                      scaler)

        if (bDetection):
            print("predicted label = ",predictedLabel)
            if (predictedLabel==testClass):
                noCorrect += 1
            else:
                noIncorrect += 1

        else:
            print("*** INVALID RESULT ***")
            sys.exit()

    return noCorrect, noIncorrect

def OpenTestResultsFile():

    f = open(DEFAULT_VAST_RANDOM_FILE_TEST_LOCATION + 'RandomTestResults.txt','w')
    if not (f):
        print("*** Unable to open test results file ***")
        sys.exit()

    return f

def OpenNVSSTestResultsFile():

    f = open(DEFAULT_VAST_RANDOM_FILE_TEST_LOCATION + 'NVSSTestResults.txt','w')
    if not (f):
        print("*** Unable to open NVSS test results file ***")
        sys.exit()

    return f

def OpenBlindTestSummaryFile(dataset):

    f = open(DEFAULT_VAST_RANDOM_FILE_TEST_LOCATION + 'BlindTestResults.txt','w')
    if not (f):
        print("*** Unable to open blind test results file ***")
        sys.exit()
    else:
        f.write('BLIND TEST RESULTS FOR DATASET :'+dataset)
        f.write('\n\n')


    return f

def OpenNVSSTestSummaryFile():

    f = open(DEFAULT_VAST_RANDOM_FILE_TEST_LOCATION + 'NVSSTestSummary.txt','w')
    if not (f):
        print("*** Unable to open NVSS test summary file ***")
        sys.exit()
    else:
        f.write('**** TEST SUMMARY FOR NVSS CATALOG SAMPLES ****')
        f.write('\n\n')


    return f

def OpenTestSummaryFile():

    f = open(DEFAULT_VAST_RANDOM_FILE_TEST_LOCATION + 'RandomTestSummary.txt','w')
    if (f):
        f.write('**** INTERPRETATION OF RESULTS ****\n\n')
        f.write('TP => A samples model correctly detected a sample e.g. AGN model with AGN data (CORRECT)\n')
        f.write('FN =>  A samples model did not correctly detect a sample e.g. AGN model with AGN data (INCORRECT)\n')
        f.write('FP => An Alternative Model Tested Positive to this sample with a GREATER probability than the TP case (INCORRECT)\n')
        f.write('TN => An Alternative Model Tested Negative to this sample (CORRECT)\n\n')


        if (bReduceFalseNegatives):
            f.write('REDUCE FALSE NEGATIVES = TRUE\n')
        else:
            f.write('REDUCE FALSE NEGATIVES = FALSE\n')

    else:
        print("*** Unable to open test results summary file ***")
        sys.exit()

    return f


def  StoreOutcomeInFile(f,outcomeClass,outcomeProbability,bChangedPrediction):
    if (f):
        f.write('\n')
        f.write('\n')
        f.write('OUTCOME Class = ' + outcomeClass)
        f.write(',')
        f.write('Probability= ' + str(round(outcomeProbability,4)))
        f.write('\n')
        if (bChangedPrediction):
            f.write('CHANGED PREDICTION')

        f.write('\n\n')

def StoreNVSSTestResults(f, imageNo, transientClass,transientProb,possibleClass):

    if (f):
        f.write(str(imageNo))
        f.write(',')
        f.write(transientClass)
        f.write(',')
        f.write(str(round(transientProb,4)))
        if (possibleClass != 0):
            f.write(', Possible = '+possibleClass)
        f.write('\n')

    else:
        print("*** INVALID File Handle For NVSS Test results ***")
        sys.exit()

def StoreHeaderInFile(f,sourceNo,trueClass,testModel):

    if (f):
        f.write('\n')
        f.write('\n')
        f.write('SOURCE NO = '+sourceNo)
        f.write(',')
        f.write('TRUE CLASS = '+trueClass)
        f.write('\n\n')

def StoreResultsInFile(f,sourceNo,trueClass,dictResults,rankDict):



    results = list(dictResults.values())
    highestProbability = max(results)

    for className,prob in dictResults.items():
        if (prob == highestProbability):
            highestClass = className
    if (f):

        f.write(sourceNo)
        f.write(',')
        f.write(trueClass)
        f.write(',')
        f.write('MODEL= ')
        f.write(dictResults['TEST_MODEL'])
        f.write(',')
        d = sorted(dictResults.items(),key=lambda x:x[1],reverse=True)

        rank1 = d[0][0]
        rank2 = d[1][0]
        rank3 = d[2][0]
        rank4 = d[3][0]
        rank5 = d[4][0]
        rank6 = d[5][0]


        for className, prob in dictResults.items():
            if (className==rank1):
                rank = '(1)'
                rankDict[className]  = 1
            elif (className==rank2):
                rank = '(2)'
                rankDict[className] = 2
            elif (className==rank3):
                rank = '(3)'
                rankDict[className] = 3
            elif (className==rank4):
                rank = '(4)'
                rankDict[className] = 4
            elif (className == rank5):
                rank = '(5)'
                rankDict[className] = 5
            elif (className == rank6):
                rank = '(6)'
                rankDict[className] = 6

            if (prob> 0.0):
                strr = className+rank+' = '+str(round(prob,4))
                f.write(strr)
                f.write(',')

        f.write('RESULT='+highestClass)
        f.write('\n')
    else:
        print("*** Invalid File Handle For results File ***")
        sys.exit()

    return rankDict


def  StoreSummaryResults(f, testClass,numberSamples,TP,FN,FP,TN,NC):


    if (f):
        classAccuracy = round((TP/(TP+FN)),2)
        nullAccuracy = round(TN/((MAX_NUMBER_MODELS-1)*numberSamples),2)
        f.write("**** True Class : ")
        f.write(testClass+' ****')
        f.write('\n')
        f.write("Number of Samples Tested: "+str(numberSamples)+' = (TP+FN)')
        f.write('\n\n')
        f.write('TP =  '+str(TP)+', TARGET = '+str(numberSamples)+' FN =  '+str(FN)+', TARGET = 0'+' Class Accuracy = '+str(classAccuracy))
        f.write('\n')
        f.write('TN =  '+str(TN)+', TARGET = '+str(int(MAX_NUMBER_MODELS-1)*numberSamples)+' FP =  '+str(FP)+', TARGET = 0'+' NULL Accuracy = '+str(nullAccuracy))
        f.write('\n')
        f.write('No Changed Predictions = '+str(NC)+' (Reduces FP)')
        f.write('\n')
        f.write('NOTE: TN TARGET = Changed Predictions+TN+FP = '+str(NC+TN+FP))
        f.write('\n\n')
    else:
        print("*** Invalid File Handle For results File ***")
        sys.exit()


def  StoreBlindTestSummaryResults(f, testClass,numberSamples,TP,FN,StatsDict,TPDict):

    PosSucc= StatsDict['POSSUCC']
    PosFail=StatsDict['POSFAIL']
    NegSucc=StatsDict['NEGSUCC']
    NegFail=StatsDict['NEGFAIL']

    TP1 = TPDict['TPCase1']
    TP2 = TPDict['TPCase2']
    TP3 = TPDict['TPCase3']
    TP4 = TPDict['TPCase4']
    TP5 = TPDict['TPCase5']

    FN1 = TPDict['FNCase1']
    FN2 = TPDict['FNCase2']
    FN3 = TPDict['FNCase3']
    FN4 = TPDict['FNCase4']
    FN5 = TPDict['FNCase5']

    accuracy = round((TP/numberSamples),2)
    if (f):
        f.write("**** True Class : ")
        f.write(testClass+' ****')
        f.write('\n')
        f.write("Number of Samples Tested: "+str(numberSamples)+' = (TP+FN)')
        f.write('\n\n')
        f.write('TP =  '+str(TP)+', TARGET = '+str(numberSamples)+' FN =  '+str(FN)+', TARGET = 0'+' Accuracy = '+str(accuracy))
        f.write('\n')
        f.write('Stats Checking')
        f.write('\n')
        f.write('For Pos Cases (In Range ='+str(PosSucc)+' ) (Out of Range = '+str(PosFail)+')')
        f.write('\n')
        f.write('For Neg Cases (In Range =' + str(NegSucc) + ') (Out of Range = ' + str(NegFail)+ ')')
        f.write('\n')
        f.write('TPCase1 = '+str(TP1)+' TPCase2 = '+str(TP2)+' TPCase3 = '+str(TP3)+' TPCase4 = '+str(TP4)+' TPCase5 = '+str(TP5))
        f.write('\n')
        f.write('FNCase1 = '+str(FN1)+' FNCase2 = '+str(FN2)+' FNCase3 = '+str(FN3)+' FNCase4 = '+str(FN4)+' FNCase5 = '+str(FN5))
        f.write('\n\n')

    else:
        print("*** Invalid File Handle For results File ***")
        sys.exit()




def  StoreNVSSTestSummaryResults(f,numberSamples,pulsarCount,agnCount,blazarCount,seyfertCount,quasarCount,unknownCount,numberPossibles):



    if (f):
        print("*** Storing NVSS Summary Results ....")
        f.write('\n')
        f.write("Number of Samples Tested: "+str(numberSamples))
        f.write('\n')
        f.write("Number of Classifications with Possibles : "+str(numberPossibles))
        f.write('\n\n')
        f.write('AGN Count =  '+str(agnCount))
        f.write('\n')
        f.write('PULSAR Count =  ' + str(pulsarCount))
        f.write('\n')
        f.write('BLAZAR Count =  ' + str(blazarCount))
        f.write('\n')
        f.write('QUASAR Count =  ' + str(quasarCount))
        f.write('\n')
        f.write('SEYFERT Count =  ' + str(seyfertCount))
        f.write('\n')
        f.write('UNKNOWN Count =  ' + str(unknownCount))
        f.write('\n')

        print("*** NVSS Summary Results Completed....")


    else:
        print("*** Invalid File Handle For NVSS Results File ***")
        sys.exit()




def StoreModelResultsInFile(f,sourceNo,trueClass,modelClass,predictedLabel,probability):

    if (f):
        f.write("True Class :")
        f.write(trueClass)
        f.write(',')
        f.write("Using Model : ")
        f.write(modelClass)
        f.write('\n')
        f.write('Predicted Outcome = ')
        f.write(predictedLabel)
        f.write(',')
        f.write('prob = ')
        f.write(str(round(probability,4)))
        f.write('\n')
    else:
        print("*** Invalid File Handle For results File ***")
        sys.exit()


def BlindTestAllSamples(dataSet,testModelList,dictList,scalerList,classList):

    agnTestData,agnSourceDetails = ProcessTransientData(dataSet, DEFAULT_AGN_CLASS,0)
    quasarTestData,quasarSourceDetails = ProcessTransientData(dataSet, DEFAULT_QUASAR_CLASS,0)
    pulsarTestData,pulsarSourceDetails = ProcessTransientData(dataSet, DEFAULT_PULSAR_CLASS,0)
    seyfertTestData,seyfertSourceDetails = ProcessTransientData(dataSet, DEFAULT_SEYFERT_CLASS,0)
    blazarTestData,blazarSourceDetails = ProcessTransientData(dataSet, DEFAULT_BLAZAR_CLASS,0)


    finalBinaryModelDict, fullModelList, fullDictList, fullScalerList,fullEncoderList = LoadMultiClassModels(dataSet)


    fTestResults = OpenTestResultsFile()
    fSummaryResults = OpenTestSummaryFile()

    for dictNo in range(len(dictList)):

        labelList = list(dictList[dictNo].values())
        print("Soak Testing "+labelList[0]+ " Class")
        if (labelList[0] == DEFAULT_AGN_CLASS):

            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING AGN SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')


            # now test test agn sample againt the agn model
            testAGNPossibilities,TP,FN,FP,TN,NC = TestForAlternatives(fTestResults,agnTestData,agnSourceDetails, dictNo, dictList, testModelList,
                                                  DEFAULT_AGN_CLASS,
                                                  scalerList,classList,finalBinaryModelDict, fullModelList, fullDictList, fullScalerList)


            StoreSummaryResults(fSummaryResults,DEFAULT_AGN_CLASS, len(agnTestData),TP,FN,FP,TN,NC )
            if (bCollectPossibilities):
                StoreResultsPossibilities(fSummaryResults, testAGNPossibilities)

        elif (labelList[0] == DEFAULT_BLAZAR_CLASS):

            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING BLAZAR SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')
            testBLAZARPossibilities,TP,FN,FP,TN,NC = TestForAlternatives(fTestResults,blazarTestData,blazarSourceDetails, dictNo, dictList, testModelList,
                                                      DEFAULT_BLAZAR_CLASS,
                                                      scalerList,classList,finalBinaryModelDict, fullModelList, fullDictList, fullScalerList)


            StoreSummaryResults(fSummaryResults, DEFAULT_BLAZAR_CLASS,len(blazarTestData),TP,FN,FP,TN,NC )
            if (bCollectPossibilities):
                StoreResultsPossibilities(fSummaryResults, testBLAZARPossibilities)

        elif (labelList[0] == DEFAULT_SEYFERT_CLASS):

            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING SEYFERT SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')

            testSEYFERTPossibilities,TP,FN,FP,TN,NC= TestForAlternatives(fTestResults,seyfertTestData, seyfertSourceDetails,dictNo, dictList, testModelList, DEFAULT_SEYFERT_CLASS,
                                                  scalerList,classList,finalBinaryModelDict, fullModelList, fullDictList, fullScalerList)

            StoreSummaryResults(fSummaryResults, DEFAULT_SEYFERT_CLASS,len(seyfertTestData), TP,FN ,FP,TN,NC)
            if (bCollectPossibilities):
              StoreResultsPossibilities(fSummaryResults, testSEYFERTPossibilities)

        elif (labelList[0] == DEFAULT_QUASAR_CLASS):


            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING QUASAR SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')

            testQUASARPossibilities,TP,FN,FP,TN,NC =  TestForAlternatives(fTestResults,quasarTestData, quasarSourceDetails,dictNo, dictList, testModelList,
                                                                       DEFAULT_QUASAR_CLASS, scalerList,classList,finalBinaryModelDict, fullModelList, fullDictList, fullScalerList)

            StoreSummaryResults(fSummaryResults, DEFAULT_QUASAR_CLASS,len(quasarTestData), TP,FN,FP,TN,NC )
            if (bCollectPossibilities):
                  StoreResultsPossibilities(fSummaryResults, testQUASARPossibilities)
        elif (labelList[0] == DEFAULT_PULSAR_CLASS):

            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING PULSAR SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')
            testPULSARPossibilities,TP,FN,FP,TN,NC = TestForAlternatives(fTestResults,pulsarTestData,pulsarSourceDetails, dictNo, dictList, testModelList,
                                                  DEFAULT_PULSAR_CLASS,
                                                  scalerList,classList,finalBinaryModelDict, fullModelList, fullDictList, fullScalerList)

            StoreSummaryResults(fSummaryResults,DEFAULT_PULSAR_CLASS,len(pulsarTestData),TP,FN,FP,TN,NC)
            if (bCollectPossibilities):
                StoreResultsPossibilities(fSummaryResults, testPULSARPossibilities)

        else:
            print("*** Unknown Transient Class To Process ***")
            sys.exit()

    fTestResults.close()
    fSummaryResults.close()

    if (bCollectPossibilities):
        DecodePossibilities(DEFAULT_QUASAR_CLASS, testQUASARPossibilities)
        DecodePossibilities(DEFAULT_PULSAR_CLASS, testPULSARPossibilities)
        DecodePossibilities(DEFAULT_SEYFERT_CLASS, testSEYFERTPossibilities)
        DecodePossibilities(DEFAULT_AGN_CLASS, testAGNPossibilities)
        DecodePossibilities(DEFAULT_BLAZAR_CLASS, testBLAZARPossibilities)

    sys.exit()



def CalcIndStats(sampleData):
    import statistics

    mean =statistics.mean(sampleData)
    stdev = statistics.stdev(sampleData)
    var = statistics.pvariance(sampleData)

    return mean,stdev,var


def CalculateClassMean(classType,testData):
    import statistics
    meanEntry = []

    print("*** Calculating "+classType+" Mean ***")
    for entry in range(len(testData)):
        meanEntry.append(statistics.mean(testData[entry][0]))

    return meanEntry



def PredictClassBasedOnStats(mean,agnMean,agnStdev,quasarMean,quasarStdev,pulsarMean,pulsarStdev,seyfertMean,seyfertStdev,blazarMean,blazarStdev):

    potentialClasses = []

    print("mean = ",mean)
    print("agn mean = ",agnMean)
    print("agn std dev = ", agnStdev)
    if (mean>= agnMean-agnStdev) and (mean <= agnMean+agnStdev):

        potentialClasses.append(DEFAULT_AGN_CLASS)
    if (mean >= quasarMean - quasarStdev) and (mean <= quasarMean + quasarStdev):

        potentialClasses.append(DEFAULT_QUASAR_CLASS)
    if (mean >= pulsarMean - pulsarStdev) and (mean <= pulsarMean + pulsarStdev):

        potentialClasses.append(DEFAULT_PULSAR_CLASS)
    if (mean >= seyfertMean - seyfertStdev) and (mean <= seyfertMean + seyfertStdev):

        potentialClasses.append(DEFAULT_SEYFERT_CLASS)
    if (mean >= blazarMean - blazarStdev) and (mean <= blazarMean + blazarStdev):

        potentialClasses.append(DEFAULT_BLAZAR_CLASS)

    return potentialClasses


def StackImages(imageData):

    numberImagesToStack = len(imageData)

    stackedImage = imageData[0]
    for image in range(1, len(imageData)):
        stackedImage += imageData[image]

    stackedImage = stackedImage / numberImagesToStack

    mean, stddev, var = CalcIndStats(stackedImage[0])

    return mean, stddev



def GenerateListOfPairsToTest(listofClasses):

    import itertools
    import operator

    processDict = {}
    modelNumber = 0

    if (len(listofClasses) >2):

        entriesToProcess = list(itertools.product(listofClasses,listofClasses))

        # now examine each entry and check that we have no duplicates


        for entry in range(len(entriesToProcess)):
            if not ((entriesToProcess[entry][0] == entriesToProcess[entry][1])):
                firstClass = entriesToProcess[entry][0]
                secondClass = entriesToProcess[entry][1]

                combinedModel = firstClass[0]+secondClass[0]
                otherModel = secondClass[0] + firstClass[0]

                if not ((combinedModel in processDict) or (otherModel in processDict)):
                    processDict[combinedModel] = modelNumber
                    modelNumber +=1

    elif (len(listofClasses) == 2):

        firstClass = listofClasses[0]
        secondClass = listofClasses[1]

        combinedModel = firstClass[0] + secondClass[0]
        otherModel = secondClass[0] + firstClass[0]

        if not ((combinedModel in processDict) or (otherModel in processDict)):
            processDict[combinedModel] = modelNumber


    return processDict


def FullTestAllSamplesOnStats(dataSet,testModelList,dictList,scalerList,classList):

    meanDict = {}
    stddevDict = {}

    if (bBlindTestAllSamples) or (bBlindTestAGN):
        agnTestData,agnSourceDetails = ProcessTransientData(dataSet, DEFAULT_AGN_CLASS,0)
        meanDict[DEFAULT_AGN_CLASS], stddevDict[DEFAULT_AGN_CLASS] = StackImages(agnTestData)
    if (bBlindTestAllSamples) or (bBlindTestQUASAR):
        quasarTestData,quasarSourceDetails = ProcessTransientData(dataSet, DEFAULT_QUASAR_CLASS,0)
        meanDict[DEFAULT_QUASAR_CLASS], stddevDict[DEFAULT_QUASAR_CLASS] = StackImages(quasarTestData)
    if (bBlindTestAllSamples) or (bBlindTestPULSAR):
        pulsarTestData,pulsarSourceDetails = ProcessTransientData(dataSet, DEFAULT_PULSAR_CLASS,0)
        meanDict[DEFAULT_PULSAR_CLASS], stddevDict[DEFAULT_PULSAR_CLASS] = StackImages(pulsarTestData)
    if (bBlindTestAllSamples) or (bBlindTestSEYFERT):
        seyfertTestData,seyfertSourceDetails = ProcessTransientData(dataSet, DEFAULT_SEYFERT_CLASS,0)
        meanDict[DEFAULT_SEYFERT_CLASS], stddevDict[DEFAULT_SEYFERT_CLASS] = StackImages(seyfertTestData)
    if (bBlindTestAllSamples) or (bBlindTestBLAZAR):
        blazarTestData,blazarSourceDetails = ProcessTransientData(dataSet, DEFAULT_BLAZAR_CLASS,0)
        meanDict[DEFAULT_BLAZAR_CLASS], stddevDict[DEFAULT_BLAZAR_CLASS] = StackImages(blazarTestData)

    finalBinaryModelDict, fullModelList, fullDictList, fullScalerList,fullEncoderList = LoadMultiClassModels(dataSet)

    fTestResults = OpenTestResultsFile()
    fSummaryResults = OpenBlindTestSummaryFile(dataSet)

    for dictNo in range(len(dictList)):

        labelList = list(dictList[dictNo].values())
        print("Testing All Data Samples Of "+labelList[0]+ " Class")
        if (labelList[0] == DEFAULT_AGN_CLASS):

            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING AGN SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')

            # now test test agn sample against agn stats
            if (bBlindTestAllSamples) or (bBlindTestAGN):
                TP, FN,StatsDict,TPDict = TestIndividualSampleSet(fTestResults,agnTestData,
                                                    agnSourceDetails, dictNo, dictList, testModelList,DEFAULT_AGN_CLASS,
                                                  scalerList,classList,finalBinaryModelDict, fullModelList, fullDictList, fullScalerList,meanDict,stddevDict)


                StoreBlindTestSummaryResults(fSummaryResults,DEFAULT_AGN_CLASS,len(agnTestData), TP, FN, StatsDict,TPDict)


        elif (labelList[0] == DEFAULT_BLAZAR_CLASS):

            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING BLAZAR SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')

            if (bBlindTestAllSamples) or (bBlindTestBLAZAR):
                TP, FN,StatsDict,TPDict =TestIndividualSampleSet(fTestResults, blazarTestData, blazarSourceDetails, dictNo, dictList, testModelList,
                                    DEFAULT_BLAZAR_CLASS,scalerList, classList, finalBinaryModelDict, fullModelList, fullDictList,fullScalerList,meanDict,stddevDict)

                StoreBlindTestSummaryResults(fSummaryResults, DEFAULT_BLAZAR_CLASS, len(blazarTestData), TP, FN, StatsDict, TPDict)

        elif (labelList[0] == DEFAULT_SEYFERT_CLASS):

            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING SEYFERT SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')
            if (bBlindTestAllSamples) or (bBlindTestSEYFERT):
                    TP, FN,StatsDict,TPDict = TestIndividualSampleSet(fTestResults, seyfertTestData, seyfertSourceDetails, dictNo, dictList, testModelList,
                                    DEFAULT_SEYFERT_CLASS, scalerList, classList, finalBinaryModelDict, fullModelList,
                                    fullDictList,fullScalerList,meanDict,stddevDict)

                    StoreBlindTestSummaryResults(fSummaryResults, DEFAULT_SEYFERT_CLASS, len(seyfertTestData), TP, FN, StatsDict,TPDict)



        elif (labelList[0] == DEFAULT_QUASAR_CLASS):


            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING QUASAR SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')
            if (bBlindTestAllSamples) or (bBlindTestQUASAR):
                TP, FN,StatsDict,TPDict = TestIndividualSampleSet(fTestResults, quasarTestData, quasarSourceDetails, dictNo, dictList, testModelList,
                                    DEFAULT_QUASAR_CLASS, scalerList, classList, finalBinaryModelDict, fullModelList,
                                    fullDictList, fullScalerList,meanDict,stddevDict)

          
                StoreBlindTestSummaryResults(fSummaryResults, DEFAULT_QUASAR_CLASS, len(quasarTestData), TP, FN, StatsDict, TPDict)



        elif (labelList[0] == DEFAULT_PULSAR_CLASS):

            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING PULSAR SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')
            if (bBlindTestAllSamples) or (bBlindTestPULSAR):
                TP, FN,StatsDict,TPDict = TestIndividualSampleSet(fTestResults, pulsarTestData, pulsarSourceDetails, dictNo, dictList, testModelList,
                                    DEFAULT_PULSAR_CLASS, scalerList, classList, finalBinaryModelDict, fullModelList,
                                    fullDictList, fullScalerList,meanDict,stddevDict)
                StoreBlindTestSummaryResults(fSummaryResults, DEFAULT_PULSAR_CLASS, len(pulsarTestData), TP, FN, StatsDict,TPDict)


        else:
            print("*** Unknown Transient Class To Process ***")
            sys.exit()

    fTestResults.close()
    fSummaryResults.close()


def FullTestNVSSCatalog():


    testModelList, dictList, scalerList, classList, encoderList = GetAllTestModels(DEFAULT_NVSS_DATASET, BINARY_OTHER_MODELS)

    finalBinaryModelDict, fullModelList, fullDictList, fullScalerList, fullEncoderList = LoadMultiClassModels(DEFAULT_NVSS_DATASET)

    fTestResults = OpenNVSSTestResultsFile()
    fSummaryResults = OpenNVSSTestSummaryFile()

    MAX_NUMBER_NVSS_TEST_IMAGES = 8568
    numberSamples = 0
    agnCount = 0
    blazarCount = 0
    quasarCount = 0
    seyfertCount = 0
    pulsarCount = 0
    unknownCount =0
    numberPossibles = 0
    MIN_PROB_LIMIT = 0.6

    for imageNo in range(MAX_NUMBER_NVSS_TEST_IMAGES):

        bValidData,testData = loadNVSSCSVFile(imageNo)
        if (bValidData):
            transientClass,transientProb,possibleClass = ClassifyAnImage(testData, dictList, testModelList, scalerList, classList, finalBinaryModelDict, fullModelList,
                                            fullDictList, fullScalerList)
            numberSamples += 1
            if (possibleClass != 0):
                numberPossibles +=1
            if (transientClass==DEFAULT_AGN_CLASS):
                agnCount+=1
            elif (transientClass==DEFAULT_BLAZAR_CLASS):
                blazarCount+=1
            elif (transientClass==DEFAULT_QUASAR_CLASS):
                quasarCount+=1
            elif (transientClass==DEFAULT_SEYFERT_CLASS):
                seyfertCount+=1
            elif (transientClass==DEFAULT_PULSAR_CLASS):
                pulsarCount+=1
            else:
                unknownCount +=1

            print("Transient Class = "+transientClass+", with Prob = "+str(round(transientProb,4)))
            if (transientProb > MIN_PROB_LIMIT):
                StoreNVSSTestResults(fTestResults, imageNo, transientClass,transientProb,possibleClass)


    StoreNVSSTestSummaryResults(fSummaryResults, numberSamples, pulsarCount, agnCount, blazarCount, seyfertCount, quasarCount,unknownCount,numberPossibles)
    fTestResults.close()
    fSummaryResults.close()


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
        if (bDebug):
            print("y_pred=",y_pred)

        pred_proba = model.predict_proba(RandomSample)
        if (bDebug):
            print(pred_proba)
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

def DisplayStackedImages(stackedData,labelList,imageMean,imageStddev,imageVar):

    SetPlotParameters()

   # fig, axs = plt.subplots(len(stackedData))

    numberPlots = len(stackedData)

    fig, axs = plt.subplots(3,2)

    figx = 0
    figy = 0

    DEFAULT_STATS_ROUNDING = 5

    for imageNo in range(0, len(stackedData)):
        imageData = stackedData[imageNo]
        x = np.arange(len(imageData[0]))

        axs[figx,figy].scatter(x, imageData[0], marker='+')
        axs[figx,figy].set_title(labelList[imageNo]+' mean = '+str(round(imageMean[imageNo],DEFAULT_STATS_ROUNDING))+' stddev = '+str(round(imageStddev[imageNo],DEFAULT_STATS_ROUNDING)))
        axs[figx,figy].scatter(x, imageData[0], marker='+')
        axs[figx,figy].tick_params(axis='x', labelsize=SMALL_FONT_SIZE)

        figy += 1

        if (figy > 1):
            figx += 1
            figy = 0

    plt.show()

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


def CreateModelFileName(modelLocation,dataSet,labelList):


    if (dataSet==DEFAULT_NVSS_DATASET):
        filename= NVSS_SHORT_NAME+UNDERSCORE
    else:
        filename = VAST_SHORT_NAME+UNDERSCORE

    filename += labelList[0][0]

    for labelNo in range(1, len(labelList)):
        filename = filename + '_' + labelList[labelNo][0]

    fullFilename = modelLocation+ filename + MODEL_FILENAME_EXTENSION

    return filename,fullFilename


def SaveModelDict(modelLocation,fileName,labelDict):
    bSavedOK = False

    filename = modelLocation+fileName+DEFAULT_DICT_TYPE

    f = open(filename,'w')
    if (f):
        bSavedOK = True

        f.write("LabelValue,LabelName")
        f.write("\n")


        for key in labelDict.keys():
            f.write("%s,%s\n"%(key,labelDict[key]))

        print("***Saving Label Dictionary ...***")
        f.close()


    return bSavedOK

def SaveModelScaler(modelLocation,fileName,scaler):
    import pickle

    bSavedOK = False

    filename = modelLocation+fileName+DEFAULT_SCALER_TYPE

    f = open(filename,'wb')
    if (f):
        pickle.dump(scaler,f)
        bSavedOK = True
        print("***Saving Scaler...***")

        f.close()

    return bSavedOK

def SaveModelEncoder(modelLocation,fileName,encoder):

    import pickle

    bSavedOK = False

    filename = modelLocation+fileName+DEFAULT_ENCODER_TYPE

    f = open(filename,'wb')
    if (f):
        pickle.dump(encoder,f)
        bSavedOK = True
        print("***Saving Encoder...***")

        f.close()

    return bSavedOK




def GetSavedDict(dictLocation,modelName):
    from os.path import splitext
    bDictOK= False


    filename, ext = splitext(modelName)

    labelDict = {}

    filename = dictLocation+filename+DEFAULT_DICT_TYPE

    print("Retrieving Label Dict ...",filename)

    f = open(filename)
    if (f):
        bDictOK = True
        dataframe = pd.read_csv(filename)
        labelList = dataframe.values

        for label in range(len(labelList)):

            labelDict[labelList[label][0]] = labelList[label][1]

        f.close()

    return bDictOK,labelDict

def GetSavedScaler(scalerLocation,modelName):
    from os.path import splitext
    import pickle


    filename, ext = splitext(modelName)

    filename = scalerLocation + filename + DEFAULT_SCALER_TYPE

    print("Retrieving Scaler ...",filename)

    with open(filename,'rb') as file:
        scaler = pickle.load(file)

    return scaler

def GetSavedEncoder(encoderLocation,modelName):
    from os.path import splitext
    import pickle


    filename, ext = splitext(modelName)

    filename = encoderLocation + filename + DEFAULT_ENCODER_TYPE

    print("Retrieving encoder ...",filename)

    with open(filename,'rb') as file:
        encoder = pickle.load(file)

    return encoder



def SaveModel(modelLocation,dataSet,labelDict,model,scaler,labelList,encoder):

    import pickle

    print("*** Saving Model ***")

    filename,fullFilename = CreateModelFileName(modelLocation,dataSet,labelList)

    print("*** as ...."+fullFilename+" ***")
    with open(fullFilename,'wb')as file:
        pickle.dump(model,file)

    SaveModelDict(modelLocation,filename,labelDict)

    SaveModelScaler(modelLocation,filename,scaler)

    SaveModelEncoder(modelLocation, filename, encoder)

def SaveCNNModel(modelLocation,dataSet,labelDict,model,scaler,labelList,encoder):

    print("*** Saving CNN Model ***")

    filename,fullFilename = CreateModelFileName(modelLocation,dataSet,labelList)

    print("*** as ...."+fullFilename+" ***")

    model.save(modelLocation)

    SaveModelDict(modelLocation,filename,labelDict)

    SaveModelScaler(modelLocation,filename,scaler)

    SaveModelEncoder(modelLocation, filename, encoder)



def GetExistingModels():
    from os.path import splitext

    modelList = []

    print("*** Scanning Models In "+DEFAULT_EXISTING_MODEL_LOCATION+" ***")
    fileList = os.scandir(DEFAULT_EXISTING_MODEL_LOCATION)
    for entry in fileList:
       if entry.is_file():
            if entry.name[0] != DS_STORE_FILENAME:
                filename, ext = splitext(entry.name)
                if (ext == MODEL_FILENAME_EXTENSION):
                    modelList.append(entry.name)

    return modelList


def GetSavedModel(modelLocation,modelFilename):
    import pickle

    filename = modelLocation+modelFilename
    print("Retrieving Model...",filename)

    with open(filename,'rb') as file:
        pickleModel = pickle.load(file)

    return pickleModel

def GetSavedCNNModel(modelLocation,modelFilename):
    from tensorflow import keras

    filename = modelLocation+modelFilename
    print("Retrieving CNN Model...",filename)

    model = keras.models.load_model(filename)


    return model


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

    bdictOK,labelDict = GetSavedDict(DEFAULT_EXISTING_MODEL_LOCATION,modelName)

    scaler = GetSavedScaler(DEFAULT_EXISTING_MODEL_LOCATION,modelName)

    encoder = GetSavedEncoder(DEFAULT_EXISTING_MODEL_LOCATION,modelName)

    if (AGN_DATA_SELECTED in modelName):
        classToTest = DEFAULT_AGN_CLASS
    elif (SEYFERT_DATA_SELECTED in modelName):
        classToTest = DEFAULT_SEYFERT_CLASS
    elif (BLAZAR_DATA_SELECTED in modelName):
        classToTest = DEFAULT_BLAZAR_CLASS
    elif (PULSAR_DATA_SELECTED in modelName):
        classToTest = DEFAULT_PULSAR_CLASS
    elif (QUASAR_DATA_SELECTED in modelName):
        classToTest = DEFAULT_QUASAR_CLASS
    else:
        print("*** UNKNOWN CLASS ***")
        sys.exit()

    if (modelName[0] == VAST_SHORT_NAME):
        dataSet = DEFAULT_VAST_DATASET

    else:
        dataSet = DEFAULT_NVSS_DATASET

    modelName = modelList[modelNumber - 1]

    pickleModel = GetSavedModel(DEFAULT_EXISTING_MODEL_LOCATION,modelName)

    return pickleModel,labelDict,scaler,dataSet,classToTest,encoder


def GetAllTestModels(dataset,modelType):

    import pickle

    testModelList = []
    dictList = []
    scalerList = []
    encoderList = []

    classList = [DEFAULT_AGN_CLASS,DEFAULT_QUASAR_CLASS,DEFAULT_PULSAR_CLASS,DEFAULT_SEYFERT_CLASS,DEFAULT_BLAZAR_CLASS,DEFAULT_OTHER_CLASS]
    if (dataset == DEFAULT_VAST_DATASET):
        modelList = [DEFAULT_TEST_V_A_O, DEFAULT_TEST_V_Q_O, DEFAULT_TEST_V_P_O, DEFAULT_TEST_V_S_O, DEFAULT_TEST_V_B_O]
        if (modelType== BINARY_OTHER_MODELS):
            modelLocation = VAST_OTHER_BINARY_MODELS_LOCATION
        elif (modelType== BINARY_FULL_MODELS):
            modelLocation = VAST_FULL_BINARY_MODELS_LOCATION
        else:
            print("*** UNKNOWN Model Type To Extract ***")
            sys.exit()

    elif (dataset == DEFAULT_NVSS_DATASET):
        modelList = [DEFAULT_TEST_N_A_O, DEFAULT_TEST_N_Q_O, DEFAULT_TEST_N_P_O, DEFAULT_TEST_N_S_O, DEFAULT_TEST_N_B_O]
        if (modelType == BINARY_OTHER_MODELS):
            modelLocation = NVSS_OTHER_BINARY_MODELS_LOCATION
        elif (modelType == BINARY_FULL_MODELS):
            modelLocation = NVSS_FULL_BINARY_MODELS_LOCATION
        else:
            print("*** UNKNOWN Model Type to Extract ***")
            sys.exit()

    print("Extracting Models From "+modelLocation+" ***")

    for testModel in modelList:

        modelName = testModel+MODEL_FILENAME_EXTENSION

        testModelList.append(GetSavedModel(modelLocation, modelName))
        bDictOK, dict = GetSavedDict(modelLocation, modelName)
        if (bDictOK):
            dictList.append(dict)
        scalerList.append(GetSavedScaler(modelLocation, modelName))
        encoderList.append(GetSavedEncoder(modelLocation, modelName))

    return testModelList,dictList,scalerList,classList,encoderList

def TestRetrievedModel(XTest,ytest,model,encoder):


    score = model.score(XTest,ytest)
    print("Test Score (Retrieved Model): {0:.2f} %".format(100*score))
    sys.exit()

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




def BuildandTestCNNModel(primaryLabel,completeTrainingData,trainingDataSizes):


    XTrain, XTest, ytrain, ytest, labelDict,scaler,labelList,encoderList = CreateTrainingAndTestData(True, primaryLabel[0],
                                                                                       completeTrainingData,
                                                                                       trainingDataSizes)

    n_timesteps = XTrain.shape[1]
    n_features = XTrain.shape[2]
    n_outputs = ytrain.shape[1]

    if (bDebug):
        print(n_timesteps, n_features, n_outputs)

    if (bOptimiseHyperParameters == True):

        fOptimisationFile = OpenOptimisationFile()

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

    return CNNModel,labelDict,scaler,labelList,encoderList


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



def TestNVSSCatalog(startEntry):

    print("Extracting and Storing Samples From NVSS Catalog....")
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
                                #        if (bScaleInputs):
                                #            dataAsArray = ScaleInputData(sourceData)

                                 #       ypredict = model.predict(dataAsArray)
                                 #       print("ypredict = ",ypredict)

                                  #      bDetection,predictedLabel = DecodePredictedLabel(labelDict, ypredict)
                                  #      if (bDetection):
                                  #          StoreNVSSDetections(fDetect, source, ra,dec,predictedLabel)
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



def RetrieveRandomImages(label):

    print("*** Retrieving Randomly Selected Images ...***")

    filepath = DEFAULT_RANDOM_FILE_TEST_LOCATION + label + DEFAULT_RANDOM_TYPE

    bDataValid = True

    if (os.path.isfile(filePath)):
        if (bDebug):
            print("*** Loading CSV File "+filePath+" ***")
        dataframe = pd.read_csv(filePath, header=None)
        dataReturn = dataframe.values
        if (bDebug):
            print("*** Completed Loading Random Image CSV File")
    else:
        print("*** Random Image CSV File Does Not Exist ***")
        bDataValid = False
        dataReturn = []

    return bDataValid,dataReturn

def StoreRandomImages(primaryLabel,randomTrainingData) :

    print("*** Storing Randomly Selected Images ...***")

    filename = DEFAULT_RANDOM_FILE_TEST_LOCATION+primaryLabel[0]+DEFAULT_RANDOM_TYPE

    f = open(filename,'w')
    if (f):
        for imageNo in range(len(randomTrainingData)):
            StoreImageContents(f, randomTrainingData[imageNo])
        f.close()


def EqualiseAllDataSamples(completeTrainingData,trainingDataSizes):

    reducedTrainingData = []

    maxSampleSize = max(trainingDataSizes)
    minSampleSize = min(trainingDataSizes)

    print("max = ",maxSampleSize)
    print("min = ",minSampleSize)


    # now make all data samples the same size

    for sample in range(len(completeTrainingData)):
        trainingData = completeTrainingData[sample]
        if (trainingDataSizes[sample] > minSampleSize):
            del trainingData[minSampleSize:trainingDataSizes[sample]]
            trainingDataSizes[sample] = minSampleSize
  #      reducedTrainingData.append(trainingData)

  #  return reducedTrainingData, trainingDataSize
    return completeTrainingData, trainingDataSizes


def ProcessAllTransientModelData(dataSet,labelList,bEqualise):

    completeTrainingData = []
    trainingDataSizes = []

    print("*** Loading Training Data ***")

    for classes in range(len(labelList)):


    #    trainingData,sourceDetails = ProcessTransientData(dataSet, labelList[classes],MAX_NUMBER_DATA_SAMPLES)
        trainingData,sourceDetails = ProcessTransientData(dataSet, labelList[classes],0)

        trainingDataSizes.append(len(trainingData))
        completeTrainingData.append(trainingData)

    if (bEqualise):
        completeTrainingData,trainingDataSizes = EqualiseAllDataSamples(completeTrainingData,trainingDataSizes)

    return completeTrainingData,trainingDataSizes


def ProcessBinaryModelData(dataSet,primaryLabel,otherLabels):


    completeTrainingData = []
    trainingDataSizes = []
    otherTrainingData = []

    print("*** Loading Primary Training Data ***")

    trainingData, sourceDetails = ProcessTransientData(dataSet, primaryLabel[0], 0)

    maxNumberSamples = int(len(trainingData) / (DEFAULT_NUMBER_MODELS-1))

    trainingDataSizes.append(len(trainingData))
    completeTrainingData.append(trainingData)

    print("*** Loading Secondary Training Data ***")

    for classes in range(len(otherLabels)):
        trainingData, sourceDetails = ProcessTransientData(dataSet, otherLabels[classes], maxNumberSamples)

        otherTrainingData.append(trainingData)

    # now concatenate all the 'other' training sets into one

    joinedTrainingData = otherTrainingData[0]
    for dataClass in range(1, len(otherTrainingData)):
        joinedTrainingData = joinedTrainingData + otherTrainingData[dataClass]

    completeTrainingData.append(joinedTrainingData)
    trainingDataSizes.append(len(joinedTrainingData))

    return completeTrainingData,trainingDataSizes

def ProcessSingleModelData(dataSet,primaryLabel,scaler,labelDict,encoder):
    testLabels = []

    print("*** Loading Primary Training Data For Single Model ***")

    testData, sourceDetails = ProcessTransientData(dataSet, primaryLabel, 0)
    testData = TransformTrainingData(testData)

    scaledData = scaler.transform(testData)

    OHELabels,lDict,labelList,OHE = createLabels(primaryLabel)

    index = labelList.index(primaryLabel)

    testLabels = assignLabelSet(OHELabels[index], len(scaledData))

    return scaledData,testLabels


def GetEqualisationStrategy():

    bCorrectInput = False

    while (bCorrectInput == False):
        equaLise = input('Equalise All Data Samples (Y/N) :')
        equaLise = equaLise.upper()
        if (equaLise == 'Y'):
            bEqualise = True
            bCorrectInput = True
        elif (equaLise == 'N'):
            bEqualise = False
            bCorrectInput = True

    return bEqualise

def GetSelectedMultiDataSets():

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


def main():

    bTestRandomForestModel=False
    bTestCNNModel = False
    bTestSVMModel = False

    if (bTestNVSSFiles):
        TestNVSSFiles(model, labelDict)

    if (bTestNVSSCatalog):
        TestNVSSCatalog(7527)

    if (bFullTestNVSSCatalog):
        FullTestNVSSCatalog()


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

        if (selectedOperation == OP_TEST_MODEL):

            model, labelDict, scaler, dataSet,primaryLabel,encoder = SelectExistingModel()
            primaryLabel = ChooseDataSetsToTest()

            testData,testLabels = ProcessSingleModelData(dataSet, primaryLabel,scaler,labelDict,encoder)

            TestRetrievedModel(testData,testLabels, model,encoder)

            sys.exit()

        elif (selectedOperation == OP_TEST_BLIND):

            print("*** Selected Blind Testing***")

            dataSet = SelectDataset()

            testModelList, dictList, scalerList, classList,encoderList = GetAllTestModels(dataSet, BINARY_OTHER_MODELS)

         #   BlindTestAllSamples(dataSet, testModelList, dictList, scalerList, classList)
            FullTestAllSamplesOnStats(dataSet, testModelList, dictList, scalerList, classList)


            sys.exit()


        elif (selectedOperation == OP_BUILD_FULL_MODEL):

            modelType = GetModelType()

            if (modelType == OP_MODEL_RANDOM):
                bTestRandomForestModel = True
            elif (modelType == OP_MODEL_SVM):
                bTestSVMModel = True

            else:
                bTestCNNModel = True

            dataSet,labelList = GetSelectedMultiDataSets()
            bEqualise = GetEqualisationStrategy()

            completeTrainingData, trainingDataSizes = ProcessAllTransientModelData(dataSet,labelList,bEqualise)

            print("*** Creating Training and Test Data Sets ***")

            XTrain, XTest, ytrain, ytest, labelDict, scaler, labelList,encoder = CreateFullTrainingAndTestData(
                False,labelList, completeTrainingData, trainingDataSizes)


        elif (selectedOperation == OP_BUILD_BINARY_MODELS):

            modelType = GetModelType()

            if (modelType == OP_MODEL_RANDOM):
                bTestRandomForestModel = True
            elif (modelType == OP_MODEL_SVM):
                bTestSVMModel = True

            else:
                 bTestCNNModel = True

            # now process all images per chosen datasets

            dataSet,primaryLabel,otherLabels = GetSelectedBinaryDataSets()

            completeTrainingData,trainingDataSizes = ProcessBinaryModelData(dataSet, primaryLabel, otherLabels)

            print("*** Creating Training and Test Data Sets ***")

            XTrain, XTest, ytrain, ytest, labelDict, scaler, labelList, encoder = CreateTrainingAndTestData(
                False, primaryLabel[0], completeTrainingData, trainingDataSizes)

        if (bTestRandomForestModel):

            if (bCheckTrainingAndTestSet):
                CheckTrainingAndTestSet(XTrain, XTest)
            if (bCheckDupRandomSamples):
                CheckDupRandomSamples(XTrain, XTest, randomSamples)

            print("*** Evaluating Random Forest Model ***")

            newModel = RandomForestModel(XTrain, ytrain, XTest, ytest)

            if (bSaveModel):
                SaveModel(DEFAULT_EXISTING_MODEL_LOCATION,dataSet, labelDict, newModel, scaler, labelList,encoder)

            sys.exit()

        elif (bTestSVMModel):

            print("*** Evaluating One Class SVM Model ***")

            newModel = OneClassSVMModel(XTrain, ytrain, XTest, ytest)

        elif (bTestCNNModel):

            newModel,labelDict, scaler, labelList, encoder = BuildandTestCNNModel(primaryLabel, completeTrainingData, trainingDataSizes)

            if (bSaveModel):
                SaveCNNModel(DEFAULT_EXISTING_MODEL_LOCATION, dataSet, labelDict, newModel, scaler, labelList, encoder)

        else:

            print("*** Unknown Operation...exiting ***")
            sys.exit()








if __name__ == '__main__':
    main()
