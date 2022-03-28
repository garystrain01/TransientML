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


bCreateTransientCVSFiles = True # build transient CSV Files

DEFAULT_UNISYDNEY_DATA = '/Volumes/ExtraDisk/UNISYDNEY_RESEARCH/TransientClassifier/'
DEFAULT_ARTEFACT_DATA = DEFAULT_UNISYDNEY_DATA + 'ARTEFACTS/'
DEFAULT_POOR_QUALITY_DATA = DEFAULT_UNISYDNEY_DATA + 'POOR_QUALITY/'


DEFAULT_HYPER_FILENAME = 'CNNHyper.txt'

DEFAULT_HYPERPARAMETERS_FILE = DEFAULT_VAST_RANDOM_FILE_TEST_LOCATION + DEFAULT_HYPER_FILENAME

DEFAULT_FITS_NO_TESTS = 10

DEFAULT_OUTPUT_FITS_AGN_FILENAME = DEFAULT_TEST_SOURCE_LOCATION + 'AGN_FITS.png'
DEFAULT_OUTPUT_FITS_BLAZAR_FILENAME = DEFAULT_TEST_SOURCE_LOCATION + 'BLAZAR_FITS.png'
DEFAULT_OUTPUT_FITS_SEYFERT_FILENAME = DEFAULT_TEST_SOURCE_LOCATION + 'SEYFERTS_FITS.png'
DEFAULT_OUTPUT_FITS_QUASAR_FILENAME = DEFAULT_TEST_SOURCE_LOCATION + 'QUASAR_FITS.png'
DEFAULT_OUTPUT_FITS_PULSAR_FILENAME = DEFAULT_TEST_SOURCE_LOCATION + 'PULSAR_FITS.png'
DEFAULT_PULSARCOORDS_FILE = DEFAULT_PULSAR_SOURCE_LOCATION + 'PulsarCoords.txt'

DEFAULT_STACKED_FILENAME_LOC = DEFAULT_VAST_DATA_ROOT + 'STACKED_IMAGE_DATA/'

STACKED_FILENAME = 'StackedImages'
DEFAULT_DUPLICATES_FILENAME = DEFAULT_TEST_SOURCE_LOCATION + 'DuplicateSources.txt'
DEFAULT_MODEL_FILE_LOCATION = DEFAULT_TEST_SOURCE_LOCATION

DEFAULT_VIZIER_SOURCES_FILENAME = DEFAULT_TEST_SOURCE_LOCATION + 'VizierSources.txt'

DEFAULT_NVSS_SOURCE_LOCATION = '/Volumes/ExtraDisk/NVSS/'
DEFAULT_NVSS_SOURCES_FILENAME = DEFAULT_NVSS_SOURCE_LOCATION + 'NVSSDetections.txt'
DEFAULT_NVSS_PULSAR_SOURCES_FILENAME = DEFAULT_NVSS_SOURCE_LOCATION + 'NVSSPulsarsFinal.txt'
DEFAULT_NVSS_SEYFERT_SOURCES_FILENAME = DEFAULT_NVSS_SOURCE_LOCATION + 'NVSSSeyfertsFinal.txt'
DEFAULT_NVSS_BLAZARS_SOURCES_FILENAME = DEFAULT_NVSS_SOURCE_LOCATION + 'NVSSBlazarsFinal.txt'
DEFAULT_NVSS_QUASARS_SOURCES_FILENAME = DEFAULT_NVSS_SOURCE_LOCATION + 'NVSSQuasarsFinal.txt'

DEFAULT_NVSS_CATALOG_LOCATION = '/Volumes/ExtraDisk/NVSS_CATALOG/'
NVSS_CATALOG_IMAGE_LOCATION = '/Volumes/ExtraDisk/NVSS_CATALOG/NVSS_IMAGES/'
NVSS_CATALOG_IMAGE_FILENAME = NVSS_CATALOG_IMAGE_LOCATION + 'NVSS_CATALOG_IMAGE_'
NVSS_CATALOG_DETECTIONS_FILENAME = DEFAULT_NVSS_CATALOG_LOCATION + 'NVSS_CATALOG_DETECTIONS.txt'

NVSS_CATALOG_CSV_LOCATION = '/Volumes/ExtraDisk/NVSS_CATALOG/NVSS_CSVFILES/'
NVSS_CATALOG_CSV_FILENAME = NVSS_CATALOG_CSV_LOCATION + 'NVSS_CSV_IMAGE_'

DEFAULT_NVSS_CATALOG_FILE = DEFAULT_NVSS_CATALOG_LOCATION + 'NVSSCatalog.text'
FINAL_NVSS_CATALOG_FILE = DEFAULT_NVSS_CATALOG_LOCATION + 'NVSS_NewCatalog.txt'
FINAL_SELECTED_NVSS_SOURCES_LOCATION = DEFAULT_NVSS_SOURCE_LOCATION + 'NVSS_RandomSamples.txt'

DEFAULT_VIZIER_PULSAR_SOURCES_FILENAME = DEFAULT_NVSS_SOURCE_LOCATION + 'PulsarSources.txt'
DEFAULT_NVSS_FITS_FILENAMES = DEFAULT_TEST_SOURCE_LOCATION + 'NVSS_FITS_FILENAMES.txt'

DEFAULT_RANDOM_SAMPLES_DIR = 'RANDOM_SAMPLES/'
DEFAULT_RANDOM_NVSS_SOURCE_LOCATION = DEFAULT_NVSS_SOURCE_LOCATION + DEFAULT_RANDOM_SAMPLES_DIR

MAX_NUMBER_MODELS = 5
MAX_NUMBER_FEATURES = 6

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
DEFAULT_CSV_FILETYPE = UNDERSCORE + 'data.txt'
XSIZE_FITS_IMAGE = 120
YSIZE_FITS_IMAGE = 120

MULTI_BINARY_MODEL_TYPE = 'M'
BINARY_MODEL_TYPE = 'B'

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

OTHER_DATA_SELECTED = "O"
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
DEFAULT_BATCH_SIZE = 16  # 32
DEFAULT_KERNEL_SIZE = 1  # 3 7
DEFAULT_NUMBER_FILTERS = 32
DEFAULT_NO_EPOCHS = 5000  # 100
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_DROPOUT_RATE = 0.50
DEFAULT_NO_NEURONS = 100
DEFAULT_MIN_CNN_DATA = 2

bPCA = False  # invoke PCA
bAddBatchNormalisation = False

bTestCNNTest1 = False
bTestCNNTest2 = False

TRAIN_TEST_RATIO = 0.80  # ratio of the total data for training

DEFAULT_NUMBER_MODELS = 5

SMALL_FONT_SIZE = 8
MEDIUM_FONT_SIZE = 10
BIGGER_FONT_SIZE = 12

OP_BUILD_MULTI_MODEL = '1'
OP_BUILD_BINARY_MODELS = '2'
OP_TEST_MODEL = '3'
OP_TEST_BLIND = '4'
OP_AUTO_TEST_BINARY_CNN = '5'
OP_AUTO_TEST_MULTI_CNN = '6'
OP_AUTO_MULTI_TEST_RF = '7'
OP_AUTO_BINARY_TEST_RF = '8'
OP_AUTO_TEST_MLP = '9'
OP_STACK_IMAGES = '1' # note was 10


OP_MODEL_RANDOM = 'R'
OP_MODEL_1DCNN = 'C1'
OP_MODEL_2DCNN = 'C2'
OP_MODEL_ALEXNET = "A"
OP_MODEL_MLP = 'M'
OP_MODEL_NAIVE = 'N'
OP_MODEL_SVM = 'O'

_1DCNN_DESC = "1D CNN"
_2D_CNN_DESC = "2D CNN"
_ALEXNET_DESC = "Alex Net"

DEFAULT_DICT_TYPE = '_dict.txt'
DEFAULT_SCALER_TYPE = '_scaler.bin'
DEFAULT_ENCODER_TYPE = '.zip'
MODEL_FILENAME_EXTENSION = '.pkl'
MODEL_TXT_EXTENSION = '.txt'
MODEL_BIN_EXTENSION = '.bin'
MODEL_ENC_EXTENSION = '.zip'
DEFAULT_RANDOM_TYPE = 'test.txt'

DEFAULT_MODEL_Q_O = 'Q_O'
DEFAULT_MODEL_A_O = 'A_O'
DEFAULT_MODEL_P_O = 'P_O'
DEFAULT_MODEL_S_O = 'S_O'
DEFAULT_MODEL_B_O = 'B_O'

# used for stat comparison on images


bStratifiedSplit = False  # use stratified split to create random train/test sets
bScaleInputs = True  # use minmax scaler for inputs
bNormalizeData = False  # or normalize instead
bStandardScaler = False  # or normalize instead

bCreateVASTFiles = True  # process all VAST related FITS Images
bCreateNVSSFiles = False  # process all NVSS related FITS Images

bCreateSEYFERTFiles = True
bCreateQUASARFiles = True
bCreateBLAZARFiles = True
bCreateAGNFiles = True
bCreatePULSARFiles = True
bCreateTESTFiles = True

bCreatePulsarData = False  # needed for special PULSAR dataset
bAstroqueryAGN = False  # to access astroquery AGN data
bAstroqueryPULSAR = False  # to access astroquery PULSAR data
bAstroqueryBLAZAR = False  # to access astroquery BLAZAR data
bAstroqueryQUASAR = False  # to access astroquery QUASAR data
bAstroquerySEYFERT = False  # to access astroquery SEYFERT data

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

bCreateNVSSCatalog = False  # for inital pre-procesing of NVSS Catalog
bTestNVSSCatalog = False  # for testing againt NVSS catalogue
bFullTestNVSSCatalog = False  # for testing againt NVSS catalogue
bTestNVSSFiles = False  # for testing againt a select set of NVSS FITS files
bTestIndSamples = False  # test some independent samples
bDataCreation = False  # for general data creation functions
bCreateCSVFiles = False  # flag set to create all CSV files from FITS images
bTakeTop2Models = True  # if false, will continue to searh below top 2 models
bCollectPossibilities = False  # log potential outcomes
bDebug = False  # swich on debug code
bLoadFullBinaryModels = True  # load full binary models for comparison
bOptimiseHyperParameters = False  # optimise hyper parameters used on convolutional model
bTestFITSFile = False  # test CNN model with individual FITS files
bTestRandomForestModel = False  # test random forest model only
bSoakTest = True
MAX_NUMBER_SOAK_TESTS = 1  # 1000

bSaveImageFiles = False  # save FITS files for presentations

bDisplayProbs = False  # display predicted probabilities
bDisplayIndividualPredictions = True
bShrinkImages = False  # to experiment with smalleer FITS images
bRandomSelectData = True  # set false if you want to eliminate randomizing of training and test data
bBlindTestAllSamples = True
bBlindTestAGN = False
bBlindTestQUASAR = False
bBlindTestPULSAR = False
bBlindTestSEYFERT = False
bBlindTestBLAZAR = False

bReduceFalseNegatives = True  # reduce false negatives by picking largest probability
bSaveModel = True  # save model for future use
bAccessNVSS = False  # access NVSS data
bCheckForDuplicates = False  # check if we have duplicate sources across classes
bCheckFITSIntegrity = False  # check if randomly selected FITS files are identical (or not)
bGetSkyview = False  # download skyview images
bProcessNVSSCatalog = False  # edit NVSS Catalog
bSelectFromNVSSCatalog = False  # do a random selection test from NVSS catalog
bSelectRandomNVSS = False  # Do a random selection from NVSS Catalog (or not)
bStoreSourcesToFile = False  # retain source list as processing
bCheckTrainingAndTestSet = False  # check that training and test sets don't overlap
bCheckDupRandomSamples = False  # check that random samples are not in test or training set


def evaluateMLPModel(Xtrain, ytrain, Xtest, ytest, noEpochs, learningRate):
    from sklearn.neural_network import MLPClassifier

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1]))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1]))

    print(Xtrain.shape)

    #   model = MLPClassifier(max_iter=noEpochs, shuffle=True, verbose=True,learning_rate_init=learningRate).fit(Xtrain,ytrain)
    model = MLPClassifier(shuffle=True, verbose=True, learning_rate_init=learningRate).fit(Xtrain, ytrain)
    accuracy = model.score(Xtest, ytest)
    print("accuracy MLP = ", accuracy)
    sys.exit()
    return accuracy, model


def SaveCNNModelAnalysis(modelType, dataset, history, labelList, numberEpochs, accuracy):
    # list all data in history
    plt.figure()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy - number epochs = ' + str(numberEpochs) + ' Accuracy = ' + str(round(accuracy, 2)))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    if (dataset == DEFAULT_VAST_DATASET):
        fname = VAST_SHORT_NAME + UNDERSCORE
        fname = fname + modelType + UNDERSCORE
        analysisLocation = DEFAULT_VAST_DATA_ROOT
    else:

        fname = NVSS_SHORT_NAME + UNDERSCORE
        fname = fname + modelType + UNDERSCORE
        analysisLocation = DEFAULT_NVSS_DATA_ROOT

    analysisLocation = analysisLocation + DEFAULT_CNN_MODEL_ANALYSIS

    for label in range(len(labelList)):
        fname = fname + labelList[label][0]
        fname = fname + UNDERSCORE

    fnameAccuracy = analysisLocation + fname + UNDERSCORE + str(
        numberEpochs) + UNDERSCORE + DEFAULT_CNN_MODEL_ACCURACY_FNAME
    fnameLoss = analysisLocation + fname + UNDERSCORE + str(numberEpochs) + UNDERSCORE + DEFAULT_CNN_MODEL_LOSS_FNAME

    plt.savefig(fnameAccuracy)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss - number epochs = ' + str(numberEpochs) + ' Accuracy = ' + str(round(accuracy, 2)))

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')

    plt.savefig(fnameLoss)


def evaluate1DCNNModel(dataset, labelList, Xtrain, ytrain, Xtest, ytest, n_timesteps, n_features, n_outputs,
                       numberEpochs, learningRate, dropoutRate, numberNeurons):
    from keras.callbacks import EarlyStopping
    import statistics

    verbose, batchSize = DEFAULT_VERBOSE_LEVEL, DEFAULT_BATCH_SIZE

    print("number epochs = ", numberEpochs)
    print("learning rate = ", learningRate)
    print("Dropout Rate = ", dropoutRate)
    print("number neurons = ", numberNeurons)
    print("timesteps = ", n_timesteps)
    print("no features = ", n_features)
    print("no outputs = ", n_outputs)
    print("labellist = ", labelList)

    model = keras.Sequential()

    model.add(tf.keras.layers.Conv1D(filters=DEFAULT_NUMBER_FILTERS, kernel_size=DEFAULT_KERNEL_SIZE, activation='relu',
                                     input_shape=(n_features, n_timesteps)))
    model.add(
        tf.keras.layers.Conv1D(filters=DEFAULT_NUMBER_FILTERS, kernel_size=DEFAULT_KERNEL_SIZE, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=DEFAULT_KERNEL_SIZE))

    model.add(tf.keras.layers.Dropout(dropoutRate))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(numberNeurons, activation='relu'))

    if (n_outputs == 1):
        model.add(tf.keras.layers.Dense(n_outputs, activation='sigmoid'))
    else:
        model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=learningRate)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=6)

    history = model.fit(Xtrain, ytrain, epochs=numberEpochs, batch_size=batchSize, verbose=1, validation_split=0.2,
                        callbacks=[es])

    model.summary()

    DEFAULT_NUMBER_CNN_TESTS = 1

    cnnAccuracy = []

    for testNo in range(DEFAULT_NUMBER_CNN_TESTS):
        accuracy = model.evaluate(Xtest, ytest, batch_size=batchSize, verbose=verbose)
        cnnAccuracy.append(accuracy)

        index = np.random.choice(Xtest.shape[0], len(Xtest), replace=False)

        Xtest = Xtest[index]
        ytest = ytest[index]

    print("CNN Accuracies = ", cnnAccuracy)
    print("CNN Best Accuracy = ", max(cnnAccuracy))
    print("CNN Worst Accuracy = ", min(cnnAccuracy))

    SaveCNNModelAnalysis(OP_MODEL_1DCNN, dataset, history, labelList, numberEpochs, accuracy[1])

    print("*** TEST RESULTS ***")
    print("Test Loss = ", accuracy[0])
    print("Test Accuracy = ", accuracy[1])

    return accuracy[1], model


def evaluate2DCNNModel(dataset, labelList, Xtrain, ytrain, Xtest, ytest, n_outputs, numberEpochs, learningRate,
                       dropoutRate, numberNeurons):
    from keras.callbacks import EarlyStopping
    import statistics

    verbose, batchSize = DEFAULT_VERBOSE_LEVEL, DEFAULT_BATCH_SIZE

    print("number epochs = ", numberEpochs)
    print("learning rate = ", learningRate)
    print("Dropout Rate = ", dropoutRate)
    print("number neurons = ", numberNeurons)
    print("no outputs = ", n_outputs)
    print("labellist = ", labelList)

    model = keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=DEFAULT_NUMBER_FILTERS, kernel_size=DEFAULT_KERNEL_SIZE, activation='relu',
                                     input_shape=(XSIZE_FITS_IMAGE, YSIZE_FITS_IMAGE, 1)))
    model.add(
        tf.keras.layers.Conv2D(filters=2 * DEFAULT_NUMBER_FILTERS, kernel_size=DEFAULT_KERNEL_SIZE, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=DEFAULT_KERNEL_SIZE))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(numberNeurons, activation='relu'))

    if (n_outputs == 1):
        model.add(tf.keras.layers.Dense(n_outputs, activation='sigmoid'))
    else:
        model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=learningRate)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    history = model.fit(Xtrain, ytrain, epochs=numberEpochs, shuffle=True, batch_size=batchSize, verbose=1,
                        validation_split=0.2, callbacks=[es])

    model.summary()

    DEFAULT_NUMBER_CNN_TESTS = 1

    cnnAccuracy = []

    for testNo in range(DEFAULT_NUMBER_CNN_TESTS):
        accuracy = model.evaluate(Xtest, ytest, batch_size=batchSize, verbose=verbose)
        cnnAccuracy.append(accuracy)

        index = np.random.choice(Xtest.shape[0], len(Xtest), replace=False)

        Xtest = Xtest[index]
        ytest = ytest[index]

    print("2D CNN Accuracies = ", cnnAccuracy)
    print("2D CNN Best Accuracy = ", max(cnnAccuracy))
    print("2D CNN Worst Accuracy = ", min(cnnAccuracy))

    SaveCNNModelAnalysis(OP_MODEL_2DCNN, dataset, history, labelList, numberEpochs, accuracy[1])

    print("*** TEST RESULTS ***")
    print("2D Test Loss = ", accuracy[0])
    print("2D Test Accuracy = ", accuracy[1])

    return accuracy[1], model


def test1DCNNModel(n_epochs, learningRate, dropoutRate, XTrain, ytrain, XTest, ytest):
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
    #   history = model.fit(Xtrain, ytrain, epochs=n_epochs, batch_size=batchSize, verbose=1, validation_split=0.2,callbacks=[es])
    model.fit(XTrain, ytrain, epochs=n_epochs, batch_size=batchSize, verbose=verbose)
    _, accuracy = model.evaluate(XTest, ytest, batch_size=batchSize, verbose=verbose)

    return accuracy, model


def OptimiseCNNHyperparameters(f, XTrain, ytrain, XTest, ytest):
    numberEpochs = [5, 10, 25]
    learningRateSchedule = [0.01, 0.001, 0.0001]
    dropoutRateSchedule = [0.20, 0.30, 0.50]

    ExperimentAccuracy = []
    ExperimentEpochs = []
    ExperimentLearningRates = []
    ExperimentDropoutRates = []

    experimentNumber = 1
    totalNoExperiments = len(numberEpochs) * len(learningRateSchedule) * len(dropoutRateSchedule)

    for epoch in numberEpochs:
        for learningRate in learningRateSchedule:
            for dropoutRate in dropoutRateSchedule:
                strr = 'Experiment No: ' + str(experimentNumber) + ' of ' + str(totalNoExperiments)
                print(strr)
                print("Testing Number Epochs = ", epoch)
                print("Testing Learning Rate = ", learningRate)
                print("Testing Dropout Rate  = ", dropoutRate)

                accuracy, model = test1DCNNModel(epoch, learningRate, dropoutRate, XTrain, ytrain, XTest, ytest)

                ExperimentAccuracy.append(accuracy)
                ExperimentEpochs.append(epoch)
                ExperimentLearningRates.append(learningRate)
                ExperimentDropoutRates.append(dropoutRate)

                WriteToOptimsationFile(f, experimentNumber, accuracy, epoch, learningRate, dropoutRate)
                experimentNumber += 1

                print("Accuracy = ", accuracy)

    return ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates, ExperimentDropoutRates


def reEncodeLabels(listOfLabels, encoder):
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.preprocessing import OneHotEncoder

    ordinalEncoder = OrdinalEncoder()
    oneHotEncoder = OneHotEncoder()

    labelDict = {}

    listOfLabels = np.array(listOfLabels)
    listOfLabels = listOfLabels.reshape(-1, 1)
    print("labels = ", listOfLabels)
    integerEncoded = ordinalEncoder.fit_transform(listOfLabels)
    categories = ordinalEncoder.categories_

    oneHotEncoded = OneHotEncoder.fit_transform(integerEncoded)
    #  oneHotEncoded = encoder.transform(integerEncoded)
    oneHotEncodedArray = oneHotEncoded.toarray()

    for i in range(len(categories[0])):
        labelDict[categories[0][i]] = oneHotEncodedArray[i]

    return labelDict


def createOneHotEncodedSet(listOfLabels):
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.preprocessing import OneHotEncoder
    labelDict = {}

    # use label to get one hot encoded value

    ordinalEncoder = OrdinalEncoder()
    oneHotEncoder = OneHotEncoder()

    listOfLabels = np.array(listOfLabels)

    listOfLabels = listOfLabels.reshape(-1, 1)

    integerEncoded = ordinalEncoder.fit_transform(listOfLabels)
    categories = ordinalEncoder.categories_

    oneHotEncoded = oneHotEncoder.fit_transform(integerEncoded)
    oneHotEncodedArray = oneHotEncoded.toarray()

    for i in range(len(categories[0])):
        labelDict[categories[0][i]] = oneHotEncodedArray[i]

    return labelDict, oneHotEncoder


def ConvertOHE(labelValue, labelDict):
    val = np.argmax(labelValue)

    return labelDict[val]


def labelDecoder(listOfLabels, labelValue):
    from sklearn.preprocessing import OrdinalEncoder

    # use label to get one hot encoded value

    ordinalEncoder = OrdinalEncoder()

    listOfLabels = np.array(listOfLabels)
    listOfLabels = listOfLabels.reshape(-1, 1)
    integerEncoded = ordinalEncoder.fit_transform(listOfLabels)

    label = ordinalEncoder.inverse_transform([[labelValue]])

    return label[0]


def ScaleInputData(X):
    from sklearn.decomposition import PCA

    scaler = MinMaxScaler()
    scaler.fit(X)
    normalised = scaler.transform(X)

    if (bPCA):
        pca = PCA(n_components=0.95)  # dimensions that account for 95% of variation
        normalised = pca.fit_transform(normalised)

    return normalised, scaler


def NormalizeInputData(X):
    normalised = normalize(X)

    return normalised


def StoreNVSSFITSImage(transientType, sourceDir, sourceName, fitsImage, FITSFileNumber):
    import os

    try:
        sourceDir = sourceDir + FOLDER_IDENTIFIER + sourceName

        print("NVSS source dir = ", sourceDir)

        print("Creating Source Directory = ", sourceDir)

        os.mkdir(sourceDir)

        filename = sourceDir + FOLDER_IDENTIFIER + transientType + '_' + str(FITSFileNumber) + FITS_FILE_EXTENSION

        print(filename)

        fitsImage.writeto(filename)
    except:

        print("Failed in StoreNVSSFITSImage")


def StoreNVSSCatalogImage(fitsImage, sourceNumber):
    bValidData = True

    try:
        imageFileName = NVSS_CATALOG_IMAGE_FILENAME + str(sourceNumber) + FITS_FILE_EXTENSION

        print("NVSS image file = ", imageFileName)

        fitsImage.writeto(imageFileName)

    except:

        bValidData = False
        print("Failed in StoreNVSSCatalogImage")

    return bValidData, imageFileName


def DisplayFITSImage(imageData, figHeader):
    plt.title(figHeader)
    plt.imshow(imageData, cmap='gray')
    plt.colorbar()

    plt.show()


def SaveSampleImages(imageData, titleData, filename):
    plt.rc('axes', titlesize=SMALL_FONT_SIZE)
    numberRows = 1

    fig, axs = plt.subplots()
    numberImages = len(imageData)
    for i in range(numberImages):
        plt.subplot(numberRows, numberImages, i + 1)
        plt.title(titleData[i])
        plt.imshow(imageData[i], cmap='gray')

    plt.show()

    fig.savefig(filename)


def ReshapeFITSImage(imageData, newXSize, newYSize):
    currentXSize = imageData.shape[0]
    currentYSize = imageData.shape[1]

    newImageData = np.zeros((newXSize, newYSize))

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

            imageData = ReshapeFITSImage(imageData, XSIZE_FITS_IMAGE, YSIZE_FITS_IMAGE)


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
            strr = 'Value at [' + str(i) + ',' + str(j) + '] = ' + str(imageData[i, j])
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


def ScanForImages(sourceLocation, sourceNumber):
    imageList = []

    imageLocation = sourceLocation + sourceNumber

    fileList = os.scandir(imageLocation)
    for entry in fileList:
        if entry.is_dir():
            if (bDebug):
                print("dir = ", entry.name)

        elif entry.is_file():
            if entry.name[0] != DS_STORE_FILENAME:
                imageList.append(entry.name)
                if (bDebug):
                    print("Creating File Entry For ", entry.name)

            else:
                if (bDebug):
                    print("File Entry Ignored For ", entry.name)

    return imageLocation, imageList


def ScanForModels(modelLocation):
    import pathlib

    bValidData = False

    modelList = []
    dictList = []
    scalerList = []
    encoderList = []

    fileList = os.scandir(modelLocation)
    for entry in fileList:
        if entry.is_dir():
            if (bDebug):
                print("dir = ", entry.name)

        elif entry.is_file():

            if entry.name[0] != DS_STORE_FILENAME:
                extension = pathlib.Path(entry.name).suffix

                if (extension == DEFAULT_MODEL_EXTENSION):
                    # found a model
                    print("found model = ", entry.name)
                    modelList.append(entry.name)

                elif (extension == MODEL_TXT_EXTENSION):
                    # found a dict
                    print("found dict = ", entry.name)
                    dictList.append(entry.name)
                elif (extension == MODEL_BIN_EXTENSION):
                    # found a scaler
                    print("found scaler  = ", entry.name)
                    scalerList.append(entry.name)

                elif (extension == MODEL_ENC_EXTENSION):
                    # found an encoder
                    print("found encoder = ", entry.name)
                    encoderList.append(entry.name)


            else:
                if (bDebug):
                    print("File Entry Ignored For ", entry.name)

    if (len(modelList) > 0) and (len(modelList) == len(dictList)) and (len(modelList) == len(scalerList)) and (
            len(modelList) == len(encoderList)):
        bValidData = True

    return bValidData, modelList, dictList, scalerList, encoderList


def ScanForTestImages(imageLocation):
    imageList = []
    bValidData = False

    print("image location = ", imageLocation)

    fileList = os.scandir(imageLocation)
    for entry in fileList:
        if entry.is_dir():
            if (bDebug):
                print("dir = ", entry.name)

        elif entry.is_file():
            if entry.name[0] != DS_STORE_FILENAME:
                imageList.append(entry.name)
                bValidData = True
                if (bDebug):
                    print("entry is file ", entry.name)
            else:
                if (bDebug):
                    print("File Entry Ignored For ", entry.name)

    return bValidData, imageList


def OpenOptimisationFile():
    f = open(DEFAULT_HYPERPARAMETERS_FILE, "w")
    return f


def WriteToOptimsationFile(f, experimentNumber, accuracy, noEpochs, learningRate, dropoutRate):
    if (f):
        strr = 'Experiment Number: ' + str(experimentNumber) + '\n'
        f.write(strr)
        strr = 'For Epochs = ' + str(noEpochs) + ' , Learning Rate = ' + str(learningRate) + ' , Dropout Rate = ' + str(
            dropoutRate) + '\n'
        f.write(strr)
        strr = 'Accuracy = ' + str(accuracy) + '\n'
        f.write(strr)
        f.write('\n')


def CreateAllCSVFiles(sourceDir, sourceList):
    sourceFileDict = {}
    for source in sourceList:
        # create a CSV file using the source name

        imageLocation, imageList = ScanForImages(sourceDir, source)
        numberImages = len(imageList)

        for imageNo in range(numberImages):
            sourceCSVFileName = sourceDir + DEFAULT_CSV_DIR + FOLDER_IDENTIFIER + source + '_' + str(
                imageNo) + DEFAULT_CSV_FILETYPE

            if (bDebug):
                print(sourceCSVFileName)

            f = open(sourceCSVFileName, "w")
            if (f):
                if source in sourceFileDict:
                    print("*** Error - Source File Already Been Created ***")
                else:
                    if (bDebug):
                        print("Success in creating csv file", sourceCSVFileName)
                    sourceFileDict[source + UNDERSCORE + str(imageNo)] = f

    return sourceFileDict


def StoreImageContents(f, imageData):
    for i in range(imageData.shape[0]):
        for j in range(imageData.shape[1]):
            strr = str(imageData[i, j])
            f.write(strr)
            if (j + 1 == imageData.shape[1]):
                # we're at the end of a row
                f.write('\n')
            else:
                f.write(',')


def StoreInCSVFile(imageLocation, image, f):
    imageLocation += FOLDER_IDENTIFIER

    bValidData, imageData = OpenFITSFile(imageLocation + image)
    if (bValidData):
        # ok - found the image data, now store in the correct CSV file
        StoreImageContents(f, imageData)


def StoreIndividualCSVFile(fitsImageFile, source):
    bResultOK = False

    bValidData, imageData = OpenFITSFile(fitsImageFile)
    if (bValidData):
        # ok - found the image data, now store in the correct CSV file
        csvFilename = NVSS_CATALOG_CSV_FILENAME + str(source) + TEXT_FILE_EXTENSION
        print("Storing NVSS Image In ", csvFilename)
        f = open(csvFilename, "w")
        if (f):
            bResultOK = True
            StoreImageContents(f, imageData)

    return bResultOK, csvFilename


def GetFITSFile(imageLocation, imageName):
    imageLocation += FOLDER_IDENTIFIER

    bValidData, imageData = OpenFITSFile(imageLocation + imageName)

    return bValidData, imageData


def ProcessAllCSVFiles(sourceDir, fileHandleDict, sourceList):
    totalNumberFilesProcessed = 0

    for source in sourceList:
        # get list of all files for this source
        fileNumber = 0
        imageLocation, imageList = ScanForImages(sourceDir, source)

        if (len(imageList) > 0):
            for image in imageList:
                totalNumberFilesProcessed += 1
                imageCSVFile = source + UNDERSCORE + str(fileNumber)
                f = fileHandleDict[imageCSVFile]

                fileNumber += 1
                StoreInCSVFile(imageLocation, image, f)
                f.close()

    return totalNumberFilesProcessed


def loadCSVFile(sourceDir, source, imageNo):
    filePath = sourceDir + DEFAULT_CSV_DIR + FOLDER_IDENTIFIER + source + '_' + str(imageNo) + DEFAULT_CSV_FILETYPE

    bDataValid = True

    if (os.path.isfile(filePath)):
        if (bDebug):
            print("*** Loading CSV File " + filePath + " ***")
        dataframe = pd.read_csv(filePath, header=None)
        dataReturn = dataframe.values
        if (bDebug):
            print("*** Completed Loading CSV File")
    else:
        print("*** CSV File Does Not Exist ***")
        bDataValid = False
        dataReturn = []

    return bDataValid, dataReturn


def loadNVSSCSVFile(imageNo):
    filePath = NVSS_CATALOG_CSV_FILENAME + str(imageNo + 1) + TEXT_FILE_EXTENSION

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

    return bDataValid, dataReturn


def loadPULSARData():
    filePath = ORIGINAL_PULSAR_SOURCES_FILENAME
    bDataValid = True

    if (os.path.isfile(filePath)):
        if (bDebug):
            print("*** Loading PULSAR data  File " + filePath + " ***")
        dataframe = pd.read_csv(filePath, header=None)
        dataReturn = dataframe.values
        if (bDebug):
            print("*** Completed Loading PULSAR File")
    else:
        print("*** PULSAR File Does Not Exist ***")
        bDataValid = False
        dataReturn = []

    return bDataValid, dataReturn


def OpenSourcesFile(rootData, sourceClass, sourceLocation, source):
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

    filename = rootData + DEFAULT_SOURCES_LOCATION + source + UNDERSCORE + filename

    f = open(filename, "w")

    return f


def StoreSourcesToFile(f, sourceType, source, imageList):
    if (f):

        strr = 'Source: ' + sourceType + ' : ' + source + ' \n'
        f.write(strr)
        for imageNo in range(len(imageList)):
            strr = 'Image No ' + str(imageNo) + ' (For Source: ' + source + ')' + ' - ' + imageList[imageNo] + ' \n'
            print(strr)
            f.write(strr)

    else:
        print("*** Unable To Access SOURCES file ***")
        sys.exit()


def ProcessTransientData(dataSet, sourceClass, maxNumberSamples):
    trainingData = []
    sourceDetails = []

    if (dataSet == DEFAULT_POORQUALITY_DATA):
        print("Processing Poor Quality Images")
        # we're doing NVSS data
        rootData = DEFAULT_POOR_QUALITY_ROOT
    elif (dataSet == DEFAULT_ARTEFACT_DATA):
            print("Processing Artefact Images")
            # we're doing NVSS data
            rootData = DEFAULT_ARTEFACT_ROOT
    else:
        # we're doing VAST data
        print("Processing VAST Data")
        rootData = DEFAULT_VAST_DATA_ROOT

    if (sourceClass == DEFAULT_TEST_CLASS):
        sourceLocation = rootData + DEFAULT_TEST_SOURCE_LOCATION
        print("*** Loading TEST Data From " + sourceLocation + "***")

    else:

        if (sourceClass == DEFAULT_AGN_CLASS):
            sourceLocation = rootData + DEFAULT_AGN_SOURCE_LOCATION
            print("*** Loading AGN Data ***")
        elif (sourceClass == DEFAULT_SEYFERT_CLASS):
            sourceLocation = rootData + DEFAULT_SEYFERT_SOURCE_LOCATION
            print("*** Loading SEYFERT Data ***")
        elif (sourceClass == DEFAULT_BLAZAR_CLASS):
            sourceLocation = rootData + DEFAULT_BLAZAR_SOURCE_LOCATION
            print("*** Loading BLAZAR Data ***")
        elif (sourceClass == DEFAULT_QUASAR_CLASS):
            sourceLocation = rootData + DEFAULT_QUASAR_SOURCE_LOCATION
            print("*** Loading QUASAR Data ***")
        elif (sourceClass == DEFAULT_PULSAR_CLASS):
            sourceLocation = rootData + DEFAULT_PULSAR_SOURCE_LOCATION
            print("*** Loading PULSAR Data ***")
        else:
            print("*** Unknown Data Class " + sourceClass + " ***")
            sys.exit()

    sourceList = ScanForSources(sourceLocation)

    numberSources = len(sourceList)

    for source in range(numberSources):

        imageLocation, imageList = ScanForImages(sourceLocation, sourceList[source])

        if (bStoreSourcesToFile):
            fSourceFile = OpenSourcesFile(rootData, sourceClass, sourceLocation, sourceList[source])
            StoreSourcesToFile(fSourceFile, sourceClass, sourceList[source], imageList)
            fSourceFile.close()

        for imageNo in range(len(imageList)):

            bValidData, sourceData = loadCSVFile(sourceLocation, sourceList[source], imageNo)
            if (bValidData):

                if (bShrinkImages):
                    sourceData = np.reshape(sourceData, (1, XSIZE_SMALL_FITS_IMAGE * YSIZE_SMALL_FITS_IMAGE))
                else:
                    sourceData = np.reshape(sourceData, (1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE))

                sourceDetails.append(str(source) + UNDERSCORE + str(imageNo))
                trainingData.append(sourceData)

    if (maxNumberSamples > 0):
        if (len(trainingData) > maxNumberSamples):
            # delete the excess samples

            del trainingData[maxNumberSamples:len(trainingData)]
            del sourceDetails[maxNumberSamples:len(trainingData)]

    print("No of Samples Loaded For " + sourceClass + " = " + str(len(trainingData)))

    return trainingData, sourceDetails

def ProcessClassificationData(dataSet, sourceClass, maxNumberSamples):
    trainingData = []
    sourceDetails = []

    if (dataSet == DEFAULT_POOR_QUALITY_DATA):
        print("Processing Poor Quality Images")

        rootData = DEFAULT_POOR_QUALITY_ROOT
    elif (dataSet == DEFAULT_ARTEFACT_DATA):
            print("Processing Artefact Images")
            # we're doing NVSS data
            rootData = DEFAULT_ARTEFACT_ROOT
    else:

        print("Unknown Dataset, exiting....")
        sys.exit()


    sourceList = ScanForSources(rootData)

    numberSources = len(sourceList)

    for source in range(numberSources):

        imageLocation, imageList = ScanForImages(sourceLocation, sourceList[source])

        if (bStoreSourcesToFile):
            fSourceFile = OpenSourcesFile(rootData, sourceClass, sourceLocation, sourceList[source])
            StoreSourcesToFile(fSourceFile, sourceClass, sourceList[source], imageList)
            fSourceFile.close()

        for imageNo in range(len(imageList)):

            bValidData, sourceData = loadCSVFile(sourceLocation, sourceList[source], imageNo)
            if (bValidData):

                if (bShrinkImages):
                    sourceData = np.reshape(sourceData, (1, XSIZE_SMALL_FITS_IMAGE * YSIZE_SMALL_FITS_IMAGE))
                else:
                    sourceData = np.reshape(sourceData, (1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE))

                sourceDetails.append(str(source) + UNDERSCORE + str(imageNo))
                trainingData.append(sourceData)

    if (maxNumberSamples > 0):
        if (len(trainingData) > maxNumberSamples):
            # delete the excess samples

            del trainingData[maxNumberSamples:len(trainingData)]
            del sourceDetails[maxNumberSamples:len(trainingData)]

    print("No of Samples Loaded For " + sourceClass + " = " + str(len(trainingData)))

    return trainingData, sourceDetails





def createLabels(labelList):
    labelDict, OHE = createOneHotEncodedSet(labelList)

    return labelDict, OHE


def decodeLabels(labelList, predictions):
    label = labelList[np.argmax(predictions)]

    return label


def assignLabelValues(label, numberOfSamples):
    shape = (numberOfSamples, len(label))

    a = np.empty((shape))

    a[:] = label

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

    a[:] = labelValue

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


def DisplayHyperTable(Accuracy, Epochs, LearningRates, DropoutRate):
    from astropy.table import QTable, Table, Column

    t = Table([Accuracy, Epochs, LearningRates, DropoutRate],
              names=('Accuracy', 'Epochs', 'Learning Rate', 'Dropout Rate'))
    print(t)


def GetOptimalParameters(Accuracy, Epochs, LearningRates, DropoutRates):
    largestEntry = Accuracy.index(max(Accuracy))

    print("Best accuracy = ", Accuracy[largestEntry])
    print("Best epochs = ", Epochs[largestEntry])
    print("Best LR = ", LearningRates[largestEntry])
    print("Best dropout = ", DropoutRates[largestEntry])

    return Epochs[largestEntry], LearningRates[largestEntry], DropoutRates[largestEntry]


def ReduceDimensions(XData):
    from sklearn.decomposition import PCA

    XData = np.asarray(XData)

    print("shape before = ", XData.shape)

    pca = PCA(n_components=0.95)
    XData = pca.fit_transform(XData)

    print("shape after = ", XData.shape)

    return XData, pca


def TransformTrainingData(trainingData):
    dataAsArray = np.asarray(trainingData)
    dataAsArray = np.reshape(dataAsArray, (dataAsArray.shape[0], dataAsArray.shape[2]))

    return dataAsArray


def CreateMultiTrainingAndTestData(cnnModel, labelList, completeTrainingData, trainingDataSizes):
    finalTrainingData = []
    datasetLabels = []

    # create labels and scale data

    labelDict, OHE = createLabels(labelList)
    OHELabelValues = list(labelDict.values())

    for dataset in range(len(completeTrainingData)):
        dataAsArray = TransformTrainingData(completeTrainingData[dataset])

        datasetLabels.append(np.asarray(assignLabelValues(OHELabelValues[dataset], trainingDataSizes[dataset])))

        finalTrainingData.append(dataAsArray)

    # create the training and test sets

    combinedTrainingSet = []
    combinedTestSet = []
    combinedTrainingLabels = []
    combinedTestLabels = []

    # scale the data

    joinedTrainingData = finalTrainingData[0]
    for dataset in range(1, len(finalTrainingData)):
        classTrainingData = finalTrainingData[dataset]
        joinedTrainingData = np.concatenate((joinedTrainingData, classTrainingData))

    joinedTrainingData, scaler = ScaleInputData(joinedTrainingData)

    # now split back up again

    startPos = 0
    for dataset in range(len(finalTrainingData)):
        finalTrainingData[dataset] = joinedTrainingData[startPos:startPos + trainingDataSizes[dataset]]
        startPos = startPos + trainingDataSizes[dataset]

    for dataset in range(len(finalTrainingData)):
        classTrainingData = finalTrainingData[dataset]
        classLabels = datasetLabels[dataset]

        numberVectorsInTrainingSet = int(round(classTrainingData.shape[0] * TRAIN_TEST_RATIO))
        numberVectorsInTestSet = int(round(classTrainingData.shape[0] - numberVectorsInTrainingSet))

        combinedTrainingSet.append(classTrainingData[:numberVectorsInTrainingSet])
        combinedTrainingLabels.append(classLabels[:numberVectorsInTrainingSet])

        combinedTestSet.append(classTrainingData[numberVectorsInTrainingSet:])
        combinedTestLabels.append(classLabels[numberVectorsInTrainingSet:])

    # now concatenate all training and test sets to create one combined training and test set

    XTrain = combinedTrainingSet[0]
    XTest = combinedTestSet[0]
    ytrain = combinedTrainingLabels[0]
    ytest = combinedTestLabels[0]

    for dataset in range(1, len(combinedTrainingSet)):
        XTrain = np.concatenate((XTrain, combinedTrainingSet[dataset]))
        XTest = np.concatenate((XTest, combinedTestSet[dataset]))
        ytrain = np.concatenate((ytrain, combinedTrainingLabels[dataset]))
        ytest = np.concatenate((ytest, combinedTestLabels[dataset]))

    # create an integrated set of training data which includes the transient and the random data
    # ensure that the sequence numbers are kept to manage the label data

    index = np.random.choice(XTrain.shape[0], len(XTrain), replace=False)

    XTrain = XTrain[index]
    ytrain = ytrain[index]

    index = np.random.choice(XTest.shape[0], len(XTest), replace=False)

    XTest = XTest[index]
    ytest = ytest[index]

    if (cnnModel == OP_MODEL_1DCNN) or (cnnModel == OP_MODEL_MLP):
        XTrain = np.reshape(XTrain, (XTrain.shape[0], XTrain.shape[1], 1))
        XTest = np.reshape(XTest, (XTest.shape[0], XTest.shape[1], 1))

    if (cnnModel == OP_MODEL_2DCNN) or (cnnModel == OP_MODEL_ALEXNET):
        # check shape

        XTrain = np.reshape(XTrain, (XTrain.shape[0], XSIZE_FITS_IMAGE, YSIZE_FITS_IMAGE, 1))
        XTest = np.reshape(XTest, (XTest.shape[0], XSIZE_FITS_IMAGE, YSIZE_FITS_IMAGE, 1))

    if (bTestCNNTest2):
        # shuffle labels as test case

        index = np.random.choice(ytest.shape[0], len(ytest), replace=False)
        ytest = ytest[index]
        index = np.random.choice(ytrain.shape[0], len(ytrain), replace=False)
        ytrain = ytrain[index]

    return XTrain, XTest, ytrain, ytest, labelDict, scaler, labelList, OHE


def CreateTrainingAndTestData(cnnModel, primaryLabel, completeTrainingData, trainingDataSizes):
    labelDict, OHE = createLabels([primaryLabel, DEFAULT_OTHER_CLASS])

    OHELabelValues = list(labelDict.values())
    labelList = list(labelDict.keys())

    datasetLabels = []
    finalTrainingData = []

    # create labels and scale data

    for dataset in range(len(completeTrainingData)):
        dataAsArray = TransformTrainingData(completeTrainingData[dataset])

        datasetLabels.append(np.asarray(assignLabelValues(OHELabelValues[dataset], trainingDataSizes[dataset])))

        finalTrainingData.append(dataAsArray)

    # create the training and test sets

    combinedTrainingSet = []
    combinedTestSet = []
    combinedTrainingLabels = []
    combinedTestLabels = []

    # scale the data

    joinedTrainingData = finalTrainingData[0]
    for dataset in range(1, len(finalTrainingData)):
        classTrainingData = finalTrainingData[dataset]
        joinedTrainingData = np.concatenate((joinedTrainingData, classTrainingData))

    joinedTrainingData, scaler = ScaleInputData(joinedTrainingData)

    # now split back up again

    startPos = 0
    for dataset in range(len(finalTrainingData)):
        finalTrainingData[dataset] = joinedTrainingData[startPos:startPos + trainingDataSizes[dataset]]

        startPos = startPos + trainingDataSizes[dataset]

    for dataset in range(len(finalTrainingData)):
        classTrainingData = finalTrainingData[dataset]
        classLabels = datasetLabels[dataset]

        numberVectorsInTrainingSet = int(round(classTrainingData.shape[0] * TRAIN_TEST_RATIO))
        numberVectorsInTestSet = int(round(classTrainingData.shape[0] - numberVectorsInTrainingSet))

        combinedTrainingSet.append(classTrainingData[:numberVectorsInTrainingSet])
        combinedTrainingLabels.append(classLabels[:numberVectorsInTrainingSet])

        combinedTestSet.append(classTrainingData[numberVectorsInTrainingSet:])
        combinedTestLabels.append(classLabels[numberVectorsInTrainingSet:])

    # now concatenate all training and test sets to create one combined training and test set

    XTrain = combinedTrainingSet[0]
    XTest = combinedTestSet[0]
    ytrain = combinedTrainingLabels[0]
    ytest = combinedTestLabels[0]

    for dataset in range(1, len(combinedTrainingSet)):
        XTrain = np.concatenate((XTrain, combinedTrainingSet[dataset]))
        XTest = np.concatenate((XTest, combinedTestSet[dataset]))
        ytrain = np.concatenate((ytrain, combinedTrainingLabels[dataset]))
        ytest = np.concatenate((ytest, combinedTestLabels[dataset]))

    # create an integrated set of training data which includes the transient and the random data
    # ensure that the sequence numbers are kept to manage the label data

    print("XTrain shape = ", XTrain.shape)
    print("XTest shape = ", XTest.shape)
    print("ytrain shape = ", ytrain.shape)
    print("ytest shape = ", ytest.shape)

    index = np.random.choice(XTrain.shape[0], len(XTrain), replace=False)

    XTrain = XTrain[index]
    ytrain = ytrain[index]

    index = np.random.choice(XTest.shape[0], len(XTest), replace=False)

    XTest = XTest[index]
    ytest = ytest[index]

    if (cnnModel == OP_MODEL_1DCNN):
        XTrain = np.reshape(XTrain, (XTrain.shape[0], XTrain.shape[1], 1))
        XTest = np.reshape(XTest, (XTest.shape[0], XTest.shape[1], 1))
    elif (cnnModel == OP_MODEL_2DCNN):
        XTrain = np.reshape(XTrain, (XTrain.shape[0], XSIZE_FITS_IMAGE, YSIZE_FITS_IMAGE, 1))
        XTest = np.reshape(XTest, (XTest.shape[0], XSIZE_FITS_IMAGE, YSIZE_FITS_IMAGE, 1))

    print("Final Training Data = ", XTrain.shape)
    print("Final Test Data = ", XTest.shape)

    print("Final Training Label Data = ", ytrain.shape)
    print("Final Test Label Data = ", ytest.shape)

    return XTrain, XTest, ytrain, ytest, labelDict, scaler, labelList, OHE


def CheckTrainingAndTestSet(XTrain, XTest):
    print("Checking Train and Test Set")

    numberInTestSet = len(XTest)
    numberInTrainSet = len(XTrain)

    numberDuplicates = 0

    for entry in range(numberInTestSet):
        for trainSample in range(numberInTrainSet):

            comparison = (XTest[entry] == XTrain[trainSample])

            if (comparison.all() == True):
                print("DUPLICATE !!")
                numberDuplicates += 1

    print("Total No Of Duplicates = ", numberDuplicates)


def CheckDupRandomSamples(XTrain, XTest, randomSamples):
    print("Checking Random Samples In Train and Test Set")

    numberInTestSet = len(XTest)
    numberInTrainSet = len(XTrain)
    numberSamples = len(randomSamples)

    print("No of samples = ", numberSamples)
    numberDuplicates = 0

    for sample in range(numberSamples):
        for entry in range(numberInTestSet):

            comparison = (XTest[entry] == randomSamples[sample])
            if (comparison.all() == True):
                print("DUPLICATE IN TEST SET !!")
                numberDuplicates += 1

    for sample in range(numberSamples):
        for entry in range(numberInTrainSet):

            comparison = (XTrain[entry] == randomSamples[sample])
            if (comparison.all() == True):
                print("DUPLICATE IN TRAIN SET !!")
                numberDuplicates += 1

    print("Total No Of Duplicates = ", numberDuplicates)


def SaveRandomImageFiles(numberSamples, sourceDir):
    if (sourceDir == DEFAULT_AGN_SOURCE_LOCATION):
        filename = DEFAULT_OUTPUT_FITS_AGN_FILENAME
        fileTitle = DEFAULT_AGN_CLASS + SOURCE_TITLE_TEXT
    elif (sourceDir == DEFAULT_BLAZAR_SOURCE_LOCATION):
        filename = DEFAULT_OUTPUT_FITS_BLAZAR_FILENAME
        fileTitle = DEFAULT_BLAZAR_CLASS + SOURCE_TITLE_TEXT
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
        print("image name = ", randomImageName)

        filePath = sourceDir + source + FOLDER_IDENTIFIER + randomImageName

        bValidImage, fitsImage = OpenFITSFile(filePath)

        if (bValidImage):
            imageFigData.append(fitsImage)
            titleData.append(fileTitle + source)

    SaveSampleImages(imageFigData, titleData, filename)


def RandomForestModel(labelDict, XTrain, ytrain, XTest, ytest):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    rndClf = RandomForestClassifier(n_estimators=30, max_depth=9, min_samples_leaf=15)

    models = []
    accuracyScores = []

    if (bSoakTest):
        print("*** Soak testing RF Model ***")
        for testNo in range(MAX_NUMBER_SOAK_TESTS):
            rndClf.fit(XTrain, ytrain)
            y_pred = rndClf.predict(XTest)

            #       bDetection,predictedLabels = DecodePredictedLabel2(labelDict,y_pred)

            accuracy = accuracy_score(ytest, y_pred)

            #            print(rndClf.__class__.__name__,accuracy)

            accuracyScores.append(accuracy)
            models.append(rndClf)
            currentHighest = max(accuracyScores)
            currentHighest = round((currentHighest * 100), 2)
            accuracy = round((accuracy * 100), 2)
            if (accuracy >= currentHighest):
                print("Random Forest Classifier, Test No " + str(testNo) + " = " + str(
                    accuracy) + " against current highest = " + str(currentHighest))
            else:
                print("Random Forest Classifier, Completed Test No " + str(testNo))

        highestAccuracy = max(accuracyScores)
        highestIndex = accuracyScores.index(highestAccuracy)

        highestAccuracy = round((highestAccuracy * 100), 2)
        print("Highest Accuracy = ", highestAccuracy)

        bestModel = models[highestIndex]

    else:
        rndClf.fit(XTrain, ytrain)
        y_pred = rndClf.predict(XTest)

        print(rndClf.__class__.__name__, accuracy_score(ytest, y_pred))
        bestModel = rndClf

    models.clear()
    accuracyScores.clear()

    return bestModel, highestAccuracy


def NaiveBayesModel(XTrain, ytrain, XTest, ytest):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    gnb = GaussianNB()

    gnb.fit(XTrain, ytrain)
    y_pred = gnb.predict(XTest)
    print(gnb.__class__.__name__, accuracy_score(ytest, y_pred))

    return gnb


def SelectDataset():
    bCorrectInput = False

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


def SelectTestDataset(dataset, modelType):
    bCorrectInput = False

    if (dataset == DEFAULT_VAST_DATASET):
        if (modelType == MULTI_BINARY_MODEL_TYPE):
            datasetChoice = 'SET A - Data Equalised Models (A), SET B - Data Not Equalised Models (B), SET C - Optimised Set Models (C)'
            shortDataSetChoice = ['A', 'B', 'C']
        else:
            datasetChoice = 'SET D - Data Equalised Models (D), SET E - Data Not Equalised Models (E)'
            shortDataSetChoice = ['D', 'E']
    elif (dataset == DEFAULT_NVSS_DATASET):
        if (modelType == MULTI_BINARY_MODEL_TYPE):
            datasetChoice = 'SET F - Data Equalised Models (F), SET G - Data Not Equalised Models (G)'
            shortDataSetChoice = ['F', 'G']
        else:
            datasetChoice = 'SET H - Data Equalised Models (H), SET K - Data Not Equalised Models (K)'
            shortDataSetChoice = ['H', 'K']
    else:
        print("*** UNKNOWN Dataset to process ***")
        sys.exit()

    while (bCorrectInput == False):
        testDataSet = input(datasetChoice)
        testDataSet = testDataSet.upper()
        if (testDataSet not in shortDataSetChoice):
            print("*** Invalid Selection - Enter again... ***")
        else:
            print("*** Test Dataset Chosen = " + testDataSet + " ***")
            bCorrectInput = True

    if (dataset == DEFAULT_VAST_DATASET) and (modelType == MULTI_BINARY_MODEL_TYPE):
        dataLocation = VAST_FULL_BINARY_MODELS_LOCATION + 'TEST_SET' + testDataSet + FOLDER_IDENTIFIER
    elif (dataset == DEFAULT_NVSS_DATASET) and (modelType == MULTI_BINARY_MODEL_TYPE):
        dataLocation = NVSS_FULL_BINARY_MODELS_LOCATION + 'TEST_SET' + testDataSet + FOLDER_IDENTIFIER
    elif (dataset == DEFAULT_VAST_DATASET) and (modelType == BINARY_MODEL_TYPE):
        dataLocation = VAST_OTHER_BINARY_MODELS_LOCATION + 'TEST_SET' + testDataSet + FOLDER_IDENTIFIER
    elif (dataset == DEFAULT_NVSS_DATASET) and (modelType == BINARY_MODEL_TYPE):
        dataLocation = NVSS_OTHER_BINARY_MODELS_LOCATION + 'TEST_SET' + testDataSet + FOLDER_IDENTIFIER
    else:
        print("*** INVALID Model Type and Dataset Selection ***")
        sys.exit()

    return dataLocation, testDataSet


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
    bCorrectInput = False

    choiceList = ["AGN(A)", "SEYFERT(S)", "BLAZAR(B)", "QUASAR(Q)", "PULSAR (P)"]

    while (bCorrectInput == False):
        strr = 'Select ' + choiceList[0] + ', ' + choiceList[1] + ', ' + choiceList[2] + ', ' + choiceList[3] + ', ' + \
               choiceList[4] + ' : '
        classData = input(strr)
        classData = classData.upper()
        bCorrectInput = True
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
            bCorrectInput = False

    return classLabel


def GetSelectedBinaryDataSets():
    classLabels = []
    bCorrectInput = False
    otherLabels = []

    choiceList = ["AGN(A)", "SEYFERT(S)", "BLAZAR(B)", "QUASAR(Q)", "PULSAR (P)"]

    dataClass = SelectDataset()

    while (bCorrectInput == False):
        strr = 'Select ' + choiceList[0] + ', ' + choiceList[1] + ', ' + choiceList[2] + ', ' + choiceList[3] + ', ' + \
               choiceList[4] + ' : '
        classData = input(strr)
        classData = classData.upper()
        if (classData == AGN_DATA_SELECTED):
            classLabels.append(DEFAULT_AGN_CLASS)
            otherLabels = [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS]
            bCorrectInput = True
        elif (classData == SEYFERT_DATA_SELECTED):

            classLabels.append(DEFAULT_SEYFERT_CLASS)
            otherLabels = [DEFAULT_AGN_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS]
            #     otherLabels = [ DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS,DEFAULT_QUASAR_CLASS]
            bCorrectInput = True
        elif (classData == BLAZAR_DATA_SELECTED):
            classLabels.append(DEFAULT_BLAZAR_CLASS)
            otherLabels = [DEFAULT_SEYFERT_CLASS, DEFAULT_AGN_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS]
            #    otherLabels = [DEFAULT_SEYFERT_CLASS, DEFAULT_PULSAR_CLASS,DEFAULT_QUASAR_CLASS]
            bCorrectInput = True
        elif (classData == QUASAR_DATA_SELECTED):

            classLabels.append(DEFAULT_QUASAR_CLASS)
            otherLabels = [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_AGN_CLASS]
            #    otherLabels = [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS]
            bCorrectInput = True

        elif (classData == PULSAR_DATA_SELECTED):

            classLabels.append(DEFAULT_PULSAR_CLASS)
            otherLabels = [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_AGN_CLASS, DEFAULT_QUASAR_CLASS]
            #    otherLabels= [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS,DEFAULT_QUASAR_CLASS]
            bCorrectInput = True
        else:
            bCorrectInput = False

    return dataClass, classLabels, otherLabels


def GetModelType():
    bCorrectInput = False

    while (bCorrectInput == False):
        selectedOperation = input(
            'Select Random Forest (' + OP_MODEL_RANDOM + ') or AlexNet (' + OP_MODEL_ALEXNET + ') Model or 1D CNN (' + OP_MODEL_1DCNN + ') Model or 2D CNN (' + OP_MODEL_2DCNN + ') Model or MLP (' + OP_MODEL_MLP + ') Model : ')
        selectedOperation = selectedOperation.upper()
        if (selectedOperation == OP_MODEL_RANDOM) or (selectedOperation == OP_MODEL_ALEXNET) or (
                selectedOperation == OP_MODEL_1DCNN) or (selectedOperation == OP_MODEL_2DCNN) or (
                selectedOperation == OP_MODEL_MLP) or (selectedOperation == OP_MODEL_MLP):
            bCorrectInput = True
        else:
            print("*** Incorrect Input - try again ***")

    return selectedOperation


def GetOperationMode():
    bCorrectInput = False

    while (bCorrectInput == False):
        print("Transient Classifier ....")
        print("1 - Create/Save Stacked Images")

        allowedOps = [OP_STACK_IMAGES]
        selOP = input("Select Operation:")
        if (selOP in allowedOps):
            bCorrectInput = True
        else:
            print("*** Incorrect Input - try again ***")

    return selOP


def DecodePredictedLabel(labelDict, prediction, predProbability):
    bDetection = False

    for predictionElement in prediction:

        predictVal = predictionElement > 0

        if (predictVal.any() == True):

            bDetection = True

            val = int(np.argmax(predictionElement))

            label = labelDict[np.argmax(predictionElement)]

            finalProbability = max(max(predProbability[val]))



        else:

            return bDetection, 0, -1

    return bDetection, label, finalProbability


def DecodePredictedLabel2(labelDict, predictions):
    bDetection = False
    predictedLabels = []

    allPossiblePredictions = list(labelDict.values())
    allPossibleLabels = list(labelDict.keys())

    for predNo in range(len(predictions)):
        prediction = predictions[predNo]

        noMatch = 0
        bLabelOK = False
        possPred = 0
        while (bLabelOK == False) and (possPred < len(allPossiblePredictions)):

            possiblePrediction = allPossiblePredictions[possPred]

            for element in range(len(prediction)):

                if (prediction[element] == possiblePrediction[element]):
                    noMatch += 1

            if (noMatch == len(prediction)):
                bLabelOK = True
                label = allPossibleLabels[possPred]
                predictedLabels.append(label)

            else:
                possPred += 1

    if (len(predictedLabels) > 0):
        bDetection = True

    return bDetection, predictedLabels


def TestIndividualFile(model, fileData):
    fileData = fileData.reshape(1, -1)

    y_pred = model.predict(fileData)
    if (bDebug):
        print("y_pred =", y_pred)

    y_pred_proba = model.predict_proba(fileData)
    if (bDebug):
        print("y_pred_proba =", y_pred_proba)

    return y_pred, y_pred_proba


def DecodeProbabilities(labelDict, classProb):
    print("number of class probabilities  = ", len(classProb))
    print("number of classes = ", len(labelDict))

    print("class prob = ", classProb)

    maxProb = max(classProb[0])
    print("max probability = ", maxProb)
    return maxProb


def TestRawSample(labelDict, model, sampleData, dataScaler):
    dataAsArray = np.asarray(sampleData)

    dataAsArray = dataAsArray.reshape(1, -1)
    scaledData = dataScaler.transform(dataAsArray)

    prediction, predProbability = TestIndividualFile(model, scaledData)

    #    print("prediction = ",prediction)
    #   print("labelDict = ",labelDict)

    bDetection, predictedLabelList = DecodePredictedLabel2(labelDict, prediction)

    #   print("predictedlabel list = ",predictedLabelList)

    return bDetection, predictedLabelList, predProbability


def DecodePossibilities(className, testPossibilities):
    noSinglePossibilities = 0
    noDoublePossibilities = 0
    noTriplePossibilities = 0
    noQuadPossibilities = 0

    for entry in range(len(testPossibilities)):
        if (testPossibilities[entry] == 1):
            noSinglePossibilities += 1
        elif (testPossibilities[entry] == 2):
            noDoublePossibilities += 1
        elif (testPossibilities[entry] == 3):
            noTriplePossibilities += 1
        elif (testPossibilities[entry] == 4):

            noQuadPossibilities + 1

    print("For class " + className)
    print('total no. of Single possibilities = ' + str(noSinglePossibilities))
    print('total no. of Double possibilities = ' + str(noDoublePossibilities))
    print('total no. of Triple possibilities = ' + str(noTriplePossibilities))
    print('total no. of Quad possibilities = ' + str(noQuadPossibilities))


def StoreResultsPossibilities(f, testPossibilities):
    noSinglePossibilities = 0
    noDoublePossibilities = 0
    noTriplePossibilities = 0
    noQuadPossibilities = 0

    for entry in range(len(testPossibilities)):
        if (testPossibilities[entry] == 1):
            noSinglePossibilities += 1
        elif (testPossibilities[entry] == 2):
            noDoublePossibilities += 1
        elif (testPossibilities[entry] == 3):
            noTriplePossibilities += 1
        elif (testPossibilities[entry] == 4):

            noQuadPossibilities + 1

    if (f):

        f.write('Total no. of Single possibilities = ' + str(noSinglePossibilities))
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
    if (classType == DEFAULT_BLAZAR_CLASS):
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


def ExecuteMultiModel(testData, multiNameToModelDict, classType1, classType2, modelList, labelDictList, scalerList):
    classValue1 = ConvertClassTypeToClassValue(classType1)
    classValue2 = ConvertClassTypeToClassValue(classType2)

    modelName = classValue1 + classValue2

    if (modelName in multiNameToModelDict):

        modelNo = multiNameToModelDict[modelName]

        bDetection, predictedLabel, finalProbability = TestRawSample(labelDictList[modelNo], modelList[modelNo],
                                                                     testData,
                                                                     scalerList[modelNo])

    else:
        print("*** UNKNOWN MODEL TYPE " + class1 + class2 + " ***")
        sys.exit()

    return bDetection, predictedLabel, finalProbability


def LoadModels(dataSet, modelLocation, modelClassTxt):
    import os

    modelList = []
    dictList = []
    scalerList = []
    encoderList = []
    nameToModelDict = {}

    if (dataSet == DEFAULT_VAST_DATASET):
        print('Loading ' + modelClassTxt + ' Models Using VAST Dataset...')
    elif (dataset == DEFAULT_NVSS_DATASET):
        print('NVSS Multi Class Models Not Supported Yet')
        sys.exit()
    else:
        print("Unknown Dataset ...")
        sys.exit()

    print('*** Building ' + modelClassTxt + ' Model Dictionary ***')

    bValidData, modelNameList, labelDictList, scalers, encoders = ScanForModels(modelLocation)

    if (bValidData):

        print("*** Total No Models = " + str(len(modelNameList)) + " ***")

        for entry in range(len(modelNameList)):
            model = GetSavedModel(modelLocation, modelNameList[entry])
            encoder = GetSavedEncoder(modelLocation, modelNameList[entry])
            dict = GetSavedDict(modelLocation, modelNameList[entry])

            scaler = GetSavedScaler(modelLocation, modelNameList[entry])

            dictList.append(dict)
            modelList.append(model)
            scalerList.append(scaler)
            encoderList.append(encoder)

    else:
        print('*** No ' + multiClassTxt + ' Models Could Be Found ***')
        sys.exit()

    for entry in range(len(modelNameList)):
        rootExt = os.path.splitext(modelNameList[entry])

        modelParams = rootExt[0].split(UNDERSCORE)

        nameToModelDict[modelParams[2] + modelParams[3]] = entry
        nameToModelDict[modelParams[3] + modelParams[2]] = entry

    return nameToModelDict, modelList, dictList, scalerList, encoderList


def LoadFullMultiClassModel(modelLocation):
    import os

    models = []
    dicts = []
    scalers = []
    encoders = []

    bValidData, modelList, dictList, scalerList, encoderList = ScanForModels(modelLocation)

    if (bValidData):

        print("*** Total No Models = " + str(len(modelList)) + " ***")

        for entry in range(len(modelList)):
            model = GetSavedModel(modelLocation, modelList[entry])
            encoder = GetSavedEncoder(modelLocation, modelList[entry])

            dict = GetSavedDict(modelLocation, modelList[entry])

            scaler = GetSavedScaler(modelLocation, modelList[entry])

            dicts.append(dict)
            models.append(model)
            scalers.append(scaler)
            encoders.append(encoder)


    else:
        print("*** No FULL BINARY Models Could Be Found ***")
        sys.exit()

    rootExt = os.path.splitext(modelList[0])
    print("rootExt = ", rootExt)
    modelDetails = rootExt[0].split(UNDERSCORE)
    print("Model Details = ", modelDetails)

    modelDataset = modelDetails[0]
    modelType = modelDetails[1]
    labelList = []

    for classNo in range(len(modelDetails) - 2):
        if (modelDetails[2 + classNo] == AGN_DATA_SELECTED):
            labelList.append(DEFAULT_AGN_CLASS)
        elif (modelDetails[2 + classNo] == BLAZAR_DATA_SELECTED):
            labelList.append(DEFAULT_BLAZAR_CLASS)
        elif (modelDetails[2 + classNo] == QUASAR_DATA_SELECTED):
            labelList.append(DEFAULT_QUASAR_CLASS)
        elif (modelDetails[2 + classNo] == SEYFERT_DATA_SELECTED):
            labelList.append(DEFAULT_SEYFERT_CLASS)
        elif (modelDetails[2 + classNo] == PULSAR_DATA_SELECTED):
            labelList.append(DEFAULT_PULSAR_CLASS)
        else:
            print("UNKNOWN Model Class - exiting")
            sys.exit()

    print("label list=", labelList)

    return labelList, models, dicts, scalers, encoders


def TestForAlternatives(fOutputFile, testData, testDataDetails, dictNo, dictList, testModelList, testClass, scalerList,
                        classList, finalBinaryModelDict, fullModelList, fullDictList, fullScalerList):
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
        bFalseNegative = False

        sourceNo = testDataDetails[dataEntry]
        StoreHeaderInFile(fOutputFile, sourceNo, testClass, testClass)

        bDetection, predictedLabel, finalProbability = TestRawSample(dictList[dictNo], testModelList[dictNo],
                                                                     testData[dataEntry],
                                                                     scalerList[dictNo])

        if (bDetection):

            StoreModelResultsInFile(fOutputFile, sourceNo, testClass, testClass, predictedLabel, finalProbability)

            if (predictedLabel == testClass):
                noPossibilities += 1
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
                bFalseNegative = True
        else:
            print("*** INVALID RESULT ***")
            sys.exit()
        # now try this data against all other models

        for modelNo in range(len(testModelList)):
            if (modelNo != dictNo):

                bDetection, predictedLabel, finalProbability = TestRawSample(dictList[modelNo], testModelList[modelNo],
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
                            noPossibilities += 1

                            # can we determine most likely of these candidates ?
                            if (bLoadFullBinaryModels):
                                bFullDetection, fullPredLabel, fullPredProb = ExecuteMultiModel(testData[dataEntry],
                                                                                                finalBinaryModelDict,
                                                                                                testClass, outcomeClass,
                                                                                                fullModelList,
                                                                                                fullDictList,
                                                                                                fullScalerList)

                                if (bFullDetection):
                                    if (fullPredLabel != testClass):

                                        # otherwise it is a true FP

                                        FP += 1
                                    else:
                                        # it is detecting our test class

                                        numberChangedPredictions += 1
                                        bChangedPrediction = True

                                        #### TEST !
                                        if (bReduceFalseNegatives) and (bFalseNegative):
                                            FN -= 1




                            else:
                                FP += 1
                        else:
                            TN += 1
                    else:
                        TN += 1

        StoreOutcomeInFile(fOutputFile, outcomeClass, outcomeProbability, bChangedPrediction)

        testPossibilities.append(noPossibilities)

    return testPossibilities, TP, FN, FP, TN, numberChangedPredictions


def findResultantClass(probability, predictionProbabilities, predictionModels, classList):
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


def TestImageAgainstStats(numberPosStatSuccess, numberPosStatFailures, dataSample, classMean, classStdev):
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
        if (possibleClasses[i] == AGN_DATA_SELECTED):
            classes.append(DEFAULT_AGN_CLASS)
        elif (possibleClasses[i] == PULSAR_DATA_SELECTED):
            classes.append(DEFAULT_PULSAR_CLASS)
        elif (possibleClasses[i] == BLAZAR_DATA_SELECTED):
            classes.append(DEFAULT_BLAZAR_CLASS)
        elif (possibleClasses[i] == SEYFERT_DATA_SELECTED):
            classes.append(DEFAULT_SEYFERT_CLASS)
        elif (possibleClasses[i] == QUASAR_DATA_SELECTED):
            classes.append(DEFAULT_QUASAR_CLASS)

    return classes[0], classes[1]


def TestIndividualSampleSet(testData, testClass, binaryNameToModelList, binaryLabelDictList, binaryScalerList,
                            multiNameToModelDict,
                            multiModelList, multiLabelDictList, multiScalerList, meanDict, stddevDict):
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

    TPDict = {}

    TPDict['TPCase1'] = 0
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

        for modelNo in range(len(binaryModelList)):

            bDetection, predictedLabel, thisProbability = TestRawSample(binaryLabelDictList[modelNo],
                                                                        binaryModelList[modelNo],
                                                                        testData[dataEntry],
                                                                        binaryScalerList[modelNo])

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

        if (numberTruePredictions > 0):
            sortedTruePredictionProbabilities = sorted(truePredictionProbabilities, reverse=True)
        else:
            sortedTruePredictionProbabilities = truePredictionProbabilities
        if (numberOtherPredictions > 0):
            sortedOtherPredictionProbabilities = sorted(otherPredictionProbabilities)
        else:
            sortedOtherPredictionProbabilities = otherPredictionProbabilities

        if (numberTruePredictions == 1):

            if (truePredictionLabels[0] == testClass):

                # ok - it has predicted the right class

                TP += 1

                TPDict['TPCase1'] += 1

                # now check the stats

                numberPosStatSuccess, numberPosStatFailures = TestImageAgainstStats(numberPosStatSuccess,
                                                                                    numberPosStatFailures,
                                                                                    testData[dataEntry],
                                                                                    meanDict[testClass],
                                                                                    stddevDict[testClass])



            else:

                possibleClass1 = findResultantClass(sortedTruePredictionProbabilities[0], truePredictionProbabilities,
                                                    truePredictionModels, classList)
                possibleClass2 = findResultantClass(sortedOtherPredictionProbabilities[0], otherPredictionProbabilities,
                                                    otherPredictionModels, classList)

                bFullDetection, fullPredLabel, fullPredProb = ExecuteMultiModel(testData[dataEntry],
                                                                                multiNameToModelDict,
                                                                                possibleClass1, possibleClass2,
                                                                                multiModelList,
                                                                                multiLabelDictList, multiScalerList)

                if (bFullDetection):

                    if (fullPredLabel == testClass):

                        TPDict['TPCase2'] += 1

                        TP += 1

                        numberPosStatSuccess, numberPosStatFailures = TestImageAgainstStats(numberPosStatSuccess,
                                                                                            numberPosStatFailures,
                                                                                            testData[dataEntry],
                                                                                            meanDict[testClass],
                                                                                            stddevDict[testClass])


                    else:

                        TPDict['FNCase2'] += 1
                        FN += 1

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

                bFullDetection, fullPredLabel, fullPredProb = ExecuteMultiModel(testData[dataEntry],
                                                                                finalBinaryModelDict,
                                                                                possibleClass1, possibleClass2,
                                                                                fullModelList,
                                                                                fullDictList, fullScalerList)

                if (bFullDetection):

                    if (fullPredLabel == testClass):
                        TP += 1

                        TPDict['TPCase3'] += 1

                        numberPosStatSuccess, numberPosStatFailures = TestImageAgainstStats(numberPosStatSuccess,
                                                                                            numberPosStatFailures,
                                                                                            testData[dataEntry],
                                                                                            meanDict[testClass],
                                                                                            stddevDict[testClass])


                    else:
                        FN += 1

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

                    possibleClass1, possibleClass2 = ConvertToClassLabel(possibleClasses[classPair])

                    bFullDetection, fullPredLabel, fullPredProb = ExecuteMultiModel(testData[dataEntry],
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

                if (len(predProbList) > 0):

                    # we have a potential result - find the highest probability and its corresponding label

                    highestProb = max(predProbList)

                    highestClassIndex = predProbList.index(highestProb)
                    highestClass = predLabelsList[highestClassIndex]
                    if (highestClass == testClass):

                        TP += 1
                        StatDict['TPCase4'] += 1

                        numberPosStatSuccess, numberPosStatFailures = TestImageAgainstStats(numberPosStatSuccess,
                                                                                            numberPosStatFailures,
                                                                                            testData[dataEntry],
                                                                                            meanDict[testClass],
                                                                                            stddevDict[testClass])

                    else:
                        FN += 1
                        StatDict['FNCase4'] += 1

                        numberNegStatSuccess, numberNegStatFailures = TestImageAgainstStats(numberNegStatSuccess,
                                                                                            numberNegStatFailures,
                                                                                            testData[dataEntry],
                                                                                            meanDict[fullPredLabel],
                                                                                            stddevDict[fullPredLabel])

        elif ((numberTruePredictions == 0) and (numberOtherPredictions > 0)):

            possibleClass1 = findResultantClass(sortedOtherPredictionProbabilities[0], otherPredictionProbabilities,
                                                otherPredictionModels, classList)

            possibleClass2 = findResultantClass(sortedOtherPredictionProbabilities[1], otherPredictionProbabilities,
                                                otherPredictionModels, classList)

            bFullDetection, fullPredLabel, fullPredProb = ExecuteMultiModel(testData[dataEntry],
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
    print("Total No Samples tested = ", str(len(testData)))
    accuracy = round((TP / len(testData)), 2)
    print("Accuracy = ", str(accuracy))

    StatDict['POSSUCC'] = numberPosStatSuccess
    StatDict['POSFAIL'] = numberPosStatFailures
    StatDict['NEGSUCC'] = numberNegStatSuccess
    StatDict['NEGFAIL'] = numberNegStatFailures

    print("No Pos Stat Success = ", str(StatDict['POSSUCC']))
    print("No Pos Stat Failures = ", str(StatDict['POSFAIL']))
    print("No Neg Stat Success = ", str(StatDict['NEGSUCC']))
    print("No Neg Stat Failures = ", str(StatDict['NEGFAIL']))

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

    return TP, FN, StatDict, TPDict


def IdentifyCorrectBinaryModel(testClass, binaryNameToModelDict):
    testModel = testClass[0] + OTHER_DATA_SELECTED

    if (testModel in binaryNameToModelDict):
        modelNo = binaryNameToModelDict[testModel]
    else:
        print("NO MODEL IDENTIFIED  - exiting...")
        sys.exit()

    return modelNo


def TestIndividualSampleSet2(testData, testClass, binaryNameToModelDict, binaryModelList, binaryLabelDictList,
                             binaryScalerList,
                             multiNameToModelDict, multiModelList, multiLabelDictList, multiScalerList):
    TP = 0
    TPP = 0
    FN = 0
    FNN = 0

    thisClassModelNo = IdentifyCorrectBinaryModel(testClass, binaryNameToModelDict)

    for dataEntry in range(len(testData)):
        #  for dataEntry in range(1):

        truePredictionProbabilities = []
        truePredictionModels = []
        truePredictionLabels = []

        otherPredictionProbabilities = []
        otherPredictionModels = []

        # first of all test the data sample against it's own model - this should be TRUE

        bDetection, predictedLabels, thisProbability = TestRawSample(binaryLabelDictList[thisClassModelNo],
                                                                     binaryModelList[thisClassModelNo],
                                                                     testData[dataEntry],
                                                                     binaryScalerList[thisClassModelNo])

        if (bDetection):
            if (predictedLabels[0] == testClass):
                TP += 1
            else:
                FN += 1
        else:
            print("INVALID Detection - exiting...")
            sys.exit()

        # now test against all the other models - these should be FALSE
        for modelNo in range(len(binaryModelList)):

            if (modelNo != thisClassModelNo):

                bDetection, predictedLabels, thisProbability = TestRawSample(binaryLabelDictList[modelNo],
                                                                             binaryModelList[modelNo],
                                                                             testData[dataEntry],
                                                                             binaryScalerList[modelNo])

                if (bDetection == True):
                    # print("predicted labels = ",predictedLabels)

                    if (predictedLabels[0] == DEFAULT_OTHER_CLASS):

                        TPP += 1

                    else:

                        FNN += 1

                else:
                    print("INVALID Prediction")
                    sys.exit()

    print("For test class = " + testClass + " TP = " + str(TP) + ", FN = " + str(FN) + " TPP = " + str(
        TPP) + ", FNN = " + str(FNN))
    print("Total No Samples tested = ", str(len(testData)))
    coreAccuracy = round((TP / (len(testData))), 2)
    print("Core Accuracy = ", str(coreAccuracy))

    otherModelAccuracy = round((TPP / ((MAX_NUMBER_MODELS - 1) * len(testData))), 2)
    print("Other Model Accuracy = ", str(otherModelAccuracy))

    return TP, FN, TPP, FNN


def ClassifyAnImage(testData, dictList, testModelList, scalerList, classList, finalBinaryModelDict, fullModelList,
                    fullDictList, fullScalerList):
    truePredictionProbabilities = []
    truePredictionModels = []
    truePredictionLabels = []

    otherPredictionProbabilities = []
    otherPredictionModels = []

    for modelNo in range(len(testModelList)):

        bDetection, predictedLabel, thisProbability = TestRawSample(dictList[modelNo], testModelList[modelNo],
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

    if (numberTruePredictions > 0):
        sortedTruePredictionProbabilities = sorted(truePredictionProbabilities, reverse=True)
    else:
        sortedTruePredictionProbabilities = truePredictionProbabilities
    if (numberOtherPredictions > 0):
        sortedOtherPredictionProbabilities = sorted(otherPredictionProbabilities)
    else:
        sortedOtherPredictionProbabilities = otherPredictionProbabilities

    if (numberTruePredictions == 1):
        print("*** ONE Prediction Only ***")
        identifiedClass = truePredictionLabels[0]
        transientProb = sortedTruePredictionProbabilities[0]
        possibleClass = 0
    elif (numberTruePredictions >= 2):

        print("*** TWO or MORE Predictions ***")
        possibleClass1 = findResultantClass(sortedTruePredictionProbabilities[0], truePredictionProbabilities,
                                            truePredictionModels, classList)
        possibleClass2 = findResultantClass(sortedTruePredictionProbabilities[1], truePredictionProbabilities,
                                            truePredictionModels, classList)

        bFullDetection, fullPredLabel, fullPredProb = ExecuteMultiModel(testData,
                                                                        finalBinaryModelDict,
                                                                        possibleClass1, possibleClass2,
                                                                        fullModelList,
                                                                        fullDictList, fullScalerList)

        if (bFullDetection):

            identifiedClass = fullPredLabel
            if (fullPredLabel == possibleClass1):
                possibleClass = possibleClass2
            else:
                possibleClass = possibleClass1
            transientProb = fullPredProb

        else:

            print("FAILED TO GET A PREDICTION")
            sys.exit()


    elif ((numberTruePredictions == 0) and (numberOtherPredictions > 0)):

        print("*** NO TRUE Predictions ***")

        possibleClass1 = findResultantClass(sortedOtherPredictionProbabilities[0], otherPredictionProbabilities,
                                            otherPredictionModels, classList)

        possibleClass2 = findResultantClass(sortedOtherPredictionProbabilities[1], otherPredictionProbabilities,
                                            otherPredictionModels, classList)

        bFullDetection, fullPredLabel, fullPredProb = ExecuteMultiModel(testData,
                                                                        finalBinaryModelDict,
                                                                        possibleClass1, possibleClass2,
                                                                        fullModelList,
                                                                        fullDictList, fullScalerList)

        if (bFullDetection):

            identifiedClass = fullPredLabel
            transientProb = fullPredProb
            if (fullPredLabel == possibleClass1):
                possibleClass = possibleClass2
            else:
                possibleClass = possibleClass1

        else:
            print("FAILED TO GET A PREDICTION")
            sys.exit()

    else:
        print("NO TRUE OR OTHER PREDICTIONS ")
        sys.exit()

    return identifiedClass, transientProb, possibleClass


def SoakTestFullModel(testData, model, testClass, scaler, labelDict):
    noCorrect = 0
    noIncorrect = 0

    print('*** Soak Testing ' + testClass + ' ***')
    for dataEntry in range(len(testData)):

        bDetection, predictedLabel, finalProbability = TestRawSample(labelDict, model,
                                                                     testData[dataEntry],
                                                                     scaler)

        if (bDetection):
            print("predicted label = ", predictedLabel)
            if (predictedLabel == testClass):
                noCorrect += 1
            else:
                noIncorrect += 1

        else:
            print("*** INVALID RESULT ***")
            sys.exit()

    return noCorrect, noIncorrect


def OpenTestResultsFile():
    f = open(DEFAULT_VAST_RANDOM_FILE_TEST_LOCATION + 'RandomTestResults.txt', 'w')
    if not (f):
        print("*** Unable to open test results file ***")
        sys.exit()

    return f


def OpenNVSSTestResultsFile():
    f = open(DEFAULT_VAST_RANDOM_FILE_TEST_LOCATION + 'NVSSTestResults.txt', 'w')
    if not (f):
        print("*** Unable to open NVSS test results file ***")
        sys.exit()

    return f


def OpenBlindTestSummaryFile(dataSet):
    if (dataSet == DEFAULT_VAST_DATASET):
        blindTestFilename = DEFAULT_VAST_DATA_ROOT + DEFAULT_BLIND_TEST_LOCATION + DEFAULT_BLIND_TEST_FILENAME
    else:
        blindTestFilename = DEFAULT_NVSS_DATA_ROOT + DEFAULT_BLIND_TEST_LOCATION + DEFAULT_BLIND_TEST_FILENAME

    f = open(blindTestFilename, 'w')
    if not (f):
        print("*** Unable to open blind test results file ***")
        sys.exit()
    else:
        f.write('BLIND TEST RESULTS FOR DATASET :' + dataSet)
        f.write('\n\n')

    return f


def OpenNVSSTestSummaryFile():
    f = open(DEFAULT_VAST_RANDOM_FILE_TEST_LOCATION + 'NVSSTestSummary.txt', 'w')
    if not (f):
        print("*** Unable to open NVSS test summary file ***")
        sys.exit()
    else:
        f.write('**** TEST SUMMARY FOR NVSS CATALOG SAMPLES ****')
        f.write('\n\n')

    return f


def OpenTestSummaryFile():
    f = open(DEFAULT_VAST_RANDOM_FILE_TEST_LOCATION + 'RandomTestSummary.txt', 'w')
    if (f):
        f.write('**** INTERPRETATION OF RESULTS ****\n\n')
        f.write('TP => A samples model correctly detected a sample e.g. AGN model with AGN data (CORRECT)\n')
        f.write('FN =>  A samples model did not correctly detect a sample e.g. AGN model with AGN data (INCORRECT)\n')
        f.write(
            'FP => An Alternative Model Tested Positive to this sample with a GREATER probability than the TP case (INCORRECT)\n')
        f.write('TN => An Alternative Model Tested Negative to this sample (CORRECT)\n\n')

        if (bReduceFalseNegatives):
            f.write('REDUCE FALSE NEGATIVES = TRUE\n')
        else:
            f.write('REDUCE FALSE NEGATIVES = FALSE\n')

    else:
        print("*** Unable to open test results summary file ***")
        sys.exit()

    return f


def StoreOutcomeInFile(f, outcomeClass, outcomeProbability, bChangedPrediction):
    if (f):
        f.write('\n')
        f.write('\n')
        f.write('OUTCOME Class = ' + outcomeClass)
        f.write(',')
        f.write('Probability= ' + str(round(outcomeProbability, 4)))
        f.write('\n')
        if (bChangedPrediction):
            f.write('CHANGED PREDICTION')

        f.write('\n\n')


def StoreNVSSTestResults(f, imageNo, transientClass, transientProb, possibleClass):
    if (f):
        f.write(str(imageNo))
        f.write(',')
        f.write(transientClass)
        f.write(',')
        f.write(str(round(transientProb, 4)))
        if (possibleClass != 0):
            f.write(', Possible = ' + possibleClass)
        f.write('\n')

    else:
        print("*** INVALID File Handle For NVSS Test results ***")
        sys.exit()


def StoreHeaderInFile(f, sourceNo, trueClass, testModel):
    if (f):
        f.write('\n')
        f.write('\n')
        f.write('SOURCE NO = ' + sourceNo)
        f.write(',')
        f.write('TRUE CLASS = ' + trueClass)
        f.write('\n\n')


def StoreResultsInFile(f, sourceNo, trueClass, dictResults, rankDict):
    results = list(dictResults.values())
    highestProbability = max(results)

    for className, prob in dictResults.items():
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
        d = sorted(dictResults.items(), key=lambda x: x[1], reverse=True)

        rank1 = d[0][0]
        rank2 = d[1][0]
        rank3 = d[2][0]
        rank4 = d[3][0]
        rank5 = d[4][0]
        rank6 = d[5][0]

        for className, prob in dictResults.items():
            if (className == rank1):
                rank = '(1)'
                rankDict[className] = 1
            elif (className == rank2):
                rank = '(2)'
                rankDict[className] = 2
            elif (className == rank3):
                rank = '(3)'
                rankDict[className] = 3
            elif (className == rank4):
                rank = '(4)'
                rankDict[className] = 4
            elif (className == rank5):
                rank = '(5)'
                rankDict[className] = 5
            elif (className == rank6):
                rank = '(6)'
                rankDict[className] = 6

            if (prob > 0.0):
                strr = className + rank + ' = ' + str(round(prob, 4))
                f.write(strr)
                f.write(',')

        f.write('RESULT=' + highestClass)
        f.write('\n')
    else:
        print("*** Invalid File Handle For results File ***")
        sys.exit()

    return rankDict


def StoreSummaryResults(f, testClass, numberSamples, TP, FN, FP, TN, NC):
    if (f):
        classAccuracy = round((TP / (TP + FN)), 2)
        nullAccuracy = round(TN / ((MAX_NUMBER_MODELS - 1) * numberSamples), 2)
        f.write("**** True Class : ")
        f.write(testClass + ' ****')
        f.write('\n')
        f.write("Number of Samples Tested: " + str(numberSamples) + ' = (TP+FN)')
        f.write('\n\n')
        f.write('TP =  ' + str(TP) + ', TARGET = ' + str(numberSamples) + ' FN =  ' + str(
            FN) + ', TARGET = 0' + ' Class Accuracy = ' + str(classAccuracy))
        f.write('\n')
        f.write('TN =  ' + str(TN) + ', TARGET = ' + str(int(MAX_NUMBER_MODELS - 1) * numberSamples) + ' FP =  ' + str(
            FP) + ', TARGET = 0' + ' NULL Accuracy = ' + str(nullAccuracy))
        f.write('\n')
        f.write('No Changed Predictions = ' + str(NC) + ' (Reduces FP)')
        f.write('\n')
        f.write('NOTE: TN TARGET = Changed Predictions+TN+FP = ' + str(NC + TN + FP))
        f.write('\n\n')
    else:
        print("*** Invalid File Handle For results File ***")
        sys.exit()


def StoreBlindTestSummaryResults(f, testClass, numberSamples, TP, FN, TPP, FNN):
    accuracy = round((TP / numberSamples), 2)
    otherAccuracy = round(TPP / ((MAX_NUMBER_MODELS - 1) * numberSamples), 2)

    print("For Class: " + testClass + " Accuracy = " + str(accuracy) + " , Other Accuracy = " + str(otherAccuracy))
    if (f):
        f.write("**** True Class : ")
        f.write(testClass + ' ****')
        f.write('\n')
        f.write("Number of Samples Tested: " + str(numberSamples) + ' = (TP+FN)')
        f.write('\n\n')
        f.write('TP =  ' + str(TP) + ', TARGET = ' + str(numberSamples) + ' FN =  ' + str(
            FN) + ', TARGET = 0' + ' Core Accuracy = ' + str(accuracy))
        f.write('\n')
        f.write('TPP =  ' + str(TPP) + ', TARGET = ' + str((MAX_NUMBER_MODELS - 1) * numberSamples) + ' FNN =  ' + str(
            FNN) + ', TARGET = 0' + ' Other Model Accuracy = ' + str(otherAccuracy))
        f.write('\n\n')
    else:
        print("*** Invalid File Handle For results File ***")
        sys.exit()

    return accuracy


def StoreNVSSTestSummaryResults(f, numberSamples, pulsarCount, agnCount, blazarCount, seyfertCount, quasarCount,
                                unknownCount, numberPossibles):
    if (f):
        print("*** Storing NVSS Summary Results ....")
        f.write('\n')
        f.write("Number of Samples Tested: " + str(numberSamples))
        f.write('\n')
        f.write("Number of Classifications with Possibles : " + str(numberPossibles))
        f.write('\n\n')
        f.write('AGN Count =  ' + str(agnCount))
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


def StoreModelResultsInFile(f, sourceNo, trueClass, modelClass, predictedLabel, probability):
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
        f.write(str(round(probability, 4)))
        f.write('\n')
    else:
        print("*** Invalid File Handle For results File ***")
        sys.exit()


def BlindTestAllSamples(dataSet, testModelList, dictList, scalerList, classList):
    agnTestData, agnSourceDetails = ProcessTransientData(dataSet, DEFAULT_AGN_CLASS, 0)
    quasarTestData, quasarSourceDetails = ProcessTransientData(dataSet, DEFAULT_QUASAR_CLASS, 0)
    pulsarTestData, pulsarSourceDetails = ProcessTransientData(dataSet, DEFAULT_PULSAR_CLASS, 0)
    seyfertTestData, seyfertSourceDetails = ProcessTransientData(dataSet, DEFAULT_SEYFERT_CLASS, 0)
    blazarTestData, blazarSourceDetails = ProcessTransientData(dataSet, DEFAULT_BLAZAR_CLASS, 0)

    #### fix up data location in next call
    #   finalBinaryModelDict, fullModelList, fullDictList, fullScalerList,fullEncoderList = LoadMultiClassModels(dataSet)

    fTestResults = OpenTestResultsFile()
    fSummaryResults = OpenTestSummaryFile()

    for dictNo in range(len(dictList)):

        labelList = list(dictList[dictNo].values())
        print("Soak Testing " + labelList[0] + " Class")
        if (labelList[0] == DEFAULT_AGN_CLASS):

            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING AGN SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')

            # now test test agn sample againt the agn model
            testAGNPossibilities, TP, FN, FP, TN, NC = TestForAlternatives(fTestResults, agnTestData, agnSourceDetails,
                                                                           dictNo, dictList, testModelList,
                                                                           DEFAULT_AGN_CLASS,
                                                                           scalerList, classList, finalBinaryModelDict,
                                                                           fullModelList, fullDictList, fullScalerList)

            StoreSummaryResults(fSummaryResults, DEFAULT_AGN_CLASS, len(agnTestData), TP, FN, FP, TN, NC)
            if (bCollectPossibilities):
                StoreResultsPossibilities(fSummaryResults, testAGNPossibilities)

        elif (labelList[0] == DEFAULT_BLAZAR_CLASS):

            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING BLAZAR SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')
            testBLAZARPossibilities, TP, FN, FP, TN, NC = TestForAlternatives(fTestResults, blazarTestData,
                                                                              blazarSourceDetails, dictNo, dictList,
                                                                              testModelList,
                                                                              DEFAULT_BLAZAR_CLASS,
                                                                              scalerList, classList,
                                                                              finalBinaryModelDict, fullModelList,
                                                                              fullDictList, fullScalerList)

            StoreSummaryResults(fSummaryResults, DEFAULT_BLAZAR_CLASS, len(blazarTestData), TP, FN, FP, TN, NC)
            if (bCollectPossibilities):
                StoreResultsPossibilities(fSummaryResults, testBLAZARPossibilities)

        elif (labelList[0] == DEFAULT_SEYFERT_CLASS):

            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING SEYFERT SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')

            testSEYFERTPossibilities, TP, FN, FP, TN, NC = TestForAlternatives(fTestResults, seyfertTestData,
                                                                               seyfertSourceDetails, dictNo, dictList,
                                                                               testModelList, DEFAULT_SEYFERT_CLASS,
                                                                               scalerList, classList,
                                                                               finalBinaryModelDict, fullModelList,
                                                                               fullDictList, fullScalerList)

            StoreSummaryResults(fSummaryResults, DEFAULT_SEYFERT_CLASS, len(seyfertTestData), TP, FN, FP, TN, NC)
            if (bCollectPossibilities):
                StoreResultsPossibilities(fSummaryResults, testSEYFERTPossibilities)

        elif (labelList[0] == DEFAULT_QUASAR_CLASS):

            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING QUASAR SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')

            testQUASARPossibilities, TP, FN, FP, TN, NC = TestForAlternatives(fTestResults, quasarTestData,
                                                                              quasarSourceDetails, dictNo, dictList,
                                                                              testModelList,
                                                                              DEFAULT_QUASAR_CLASS, scalerList,
                                                                              classList, finalBinaryModelDict,
                                                                              fullModelList, fullDictList,
                                                                              fullScalerList)

            StoreSummaryResults(fSummaryResults, DEFAULT_QUASAR_CLASS, len(quasarTestData), TP, FN, FP, TN, NC)
            if (bCollectPossibilities):
                StoreResultsPossibilities(fSummaryResults, testQUASARPossibilities)
        elif (labelList[0] == DEFAULT_PULSAR_CLASS):

            fTestResults.write('\n')
            fTestResults.write('\n')
            fTestResults.write('*** TESTING PULSAR SAMPLES ***')
            fTestResults.write('\n')
            fTestResults.write('\n')
            testPULSARPossibilities, TP, FN, FP, TN, NC = TestForAlternatives(fTestResults, pulsarTestData,
                                                                              pulsarSourceDetails, dictNo, dictList,
                                                                              testModelList,
                                                                              DEFAULT_PULSAR_CLASS,
                                                                              scalerList, classList,
                                                                              finalBinaryModelDict, fullModelList,
                                                                              fullDictList, fullScalerList)

            StoreSummaryResults(fSummaryResults, DEFAULT_PULSAR_CLASS, len(pulsarTestData), TP, FN, FP, TN, NC)
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

    mean = statistics.mean(sampleData)
    stdev = statistics.stdev(sampleData)
    var = statistics.pvariance(sampleData)

    return mean, stdev, var


def CalculateClassMean(classType, testData):
    import statistics
    meanEntry = []

    print("*** Calculating " + classType + " Mean ***")
    for entry in range(len(testData)):
        meanEntry.append(statistics.mean(testData[entry][0]))

    return meanEntry


def PredictClassBasedOnStats(mean, agnMean, agnStdev, quasarMean, quasarStdev, pulsarMean, pulsarStdev, seyfertMean,
                             seyfertStdev, blazarMean, blazarStdev):
    potentialClasses = []

    print("mean = ", mean)
    print("agn mean = ", agnMean)
    print("agn std dev = ", agnStdev)
    if (mean >= agnMean - agnStdev) and (mean <= agnMean + agnStdev):
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


def StackImagesForStats(imageData):
    numberImagesToStack = len(imageData)

    stackedImage = imageData[0]
    for image in range(1, len(imageData)):
        stackedImage += imageData[image]

    stackedImage = stackedImage / numberImagesToStack

    mean, stddev, var = CalcIndStats(stackedImage[0])

    return mean, stddev, stackedImage


def StackImagesNoStats(imageData):
    numberImagesToStack = len(imageData)

    stackedImage = imageData[0]
    for image in range(1, len(imageData)):
        stackedImage += imageData[image]

    stackedImage = stackedImage / numberImagesToStack

    return stackedImage


def GenerateListOfPairsToTest(listofClasses):
    import itertools
    import operator

    processDict = {}
    modelNumber = 0

    if (len(listofClasses) > 2):

        entriesToProcess = list(itertools.product(listofClasses, listofClasses))

        # now examine each entry and check that we have no duplicates

        for entry in range(len(entriesToProcess)):
            if not ((entriesToProcess[entry][0] == entriesToProcess[entry][1])):
                firstClass = entriesToProcess[entry][0]
                secondClass = entriesToProcess[entry][1]

                combinedModel = firstClass[0] + secondClass[0]
                otherModel = secondClass[0] + firstClass[0]

                if not ((combinedModel in processDict) or (otherModel in processDict)):
                    processDict[combinedModel] = modelNumber
                    modelNumber += 1

    elif (len(listofClasses) == 2):

        firstClass = listofClasses[0]
        secondClass = listofClasses[1]

        combinedModel = firstClass[0] + secondClass[0]
        otherModel = secondClass[0] + firstClass[0]

        if not ((combinedModel in processDict) or (otherModel in processDict)):
            processDict[combinedModel] = modelNumber

    return processDict


def StackAllImages(dataSet):
    StackedImageData = []
    StackedImageLabel = []

    agnTestData, agnSourceDetails = ProcessTransientData(dataSet, DEFAULT_AGN_CLASS, 0)
    StackedImageData.append(StackImagesNoStats(agnTestData))
    StackedImageLabel.append(DEFAULT_AGN_CLASS)
    quasarTestData, quasarSourceDetails = ProcessTransientData(dataSet, DEFAULT_QUASAR_CLASS, 0)
    StackedImageData.append(StackImagesNoStats(quasarTestData))
    StackedImageLabel.append(DEFAULT_QUASAR_CLASS)
    pulsarTestData, pulsarSourceDetails = ProcessTransientData(dataSet, DEFAULT_PULSAR_CLASS, 0)
    StackedImageData.append(StackImagesNoStats(pulsarTestData))
    StackedImageLabel.append(DEFAULT_PULSAR_CLASS)
    seyfertTestData, seyfertSourceDetails = ProcessTransientData(dataSet, DEFAULT_SEYFERT_CLASS, 0)
    StackedImageData.append(StackImagesNoStats(seyfertTestData))
    StackedImageLabel.append(DEFAULT_SEYFERT_CLASS)
    blazarTestData, blazarSourceDetails = ProcessTransientData(dataSet, DEFAULT_BLAZAR_CLASS, 0)
    StackedImageData.append(StackImagesNoStats(blazarTestData))
    StackedImageLabel.append(DEFAULT_BLAZAR_CLASS)

    return StackedImageLabel, StackedImageData


def FullTestAllSamples(dataSet, classList, binaryNameToModelDict, binaryModelList, binaryLabelDictList,
                       binaryScalerList, binaryEncoderList, multiNameToModelDict,
                       multiModelList, multiLabelDictList, multiScalerList, multiEncoderList):
    if (bBlindTestAllSamples) or (bBlindTestAGN):
        agnTestData, agnSourceDetails = ProcessTransientData(dataSet, DEFAULT_AGN_CLASS, 0)

    if (bBlindTestAllSamples) or (bBlindTestQUASAR):
        quasarTestData, quasarSourceDetails = ProcessTransientData(dataSet, DEFAULT_QUASAR_CLASS, 0)

    if (bBlindTestAllSamples) or (bBlindTestPULSAR):
        pulsarTestData, pulsarSourceDetails = ProcessTransientData(dataSet, DEFAULT_PULSAR_CLASS, 0)

    if (bBlindTestAllSamples) or (bBlindTestSEYFERT):
        seyfertTestData, seyfertSourceDetails = ProcessTransientData(dataSet, DEFAULT_SEYFERT_CLASS, 0)

    if (bBlindTestAllSamples) or (bBlindTestBLAZAR):
        blazarTestData, blazarSourceDetails = ProcessTransientData(dataSet, DEFAULT_BLAZAR_CLASS, 0)

    fSummaryResults = OpenBlindTestSummaryFile(dataSet)
    accuracy = 0.0

    for labelNo in range(len(classList)):

        print("Testing All Data Samples Of " + classList[labelNo] + " Class")
        if (classList[labelNo] == DEFAULT_PULSAR_CLASS):

            print('*** TESTING PULSAR SAMPLES ***')

            # now test test agn sample against agn stats
            if (bBlindTestAllSamples) or (bBlindTestPULSAR):
                TP, FN, TPP, FNN = TestIndividualSampleSet2(pulsarTestData, DEFAULT_PULSAR_CLASS, binaryNameToModelDict,
                                                            binaryModelList, binaryLabelDictList,
                                                            binaryScalerList, multiNameToModelDict, multiModelList,
                                                            multiLabelDictList, multiScalerList)

                accuracy += StoreBlindTestSummaryResults(fSummaryResults, DEFAULT_PULSAR_CLASS, len(pulsarTestData), TP,
                                                         FN, TPP, FNN)
                sys.exit()

        elif (classList[labelNo] == DEFAULT_BLAZAR_CLASS):

            print('*** TESTING BLAZAR SAMPLES ***')

            if (bBlindTestAllSamples) or (bBlindTestBLAZAR):
                TP, FN, TPP, FNN = TestIndividualSampleSet2(blazarTestData, DEFAULT_BLAZAR_CLASS, binaryNameToModelDict,
                                                            binaryModelList, binaryLabelDictList,
                                                            binaryScalerList, multiNameToModelDict, multiModelList,
                                                            multiLabelDictList, multiScalerList)

                accuracy += StoreBlindTestSummaryResults(fSummaryResults, DEFAULT_BLAZAR_CLASS, len(blazarTestData), TP,
                                                         FN, TPP, FNN)

        elif (classList[labelNo] == DEFAULT_SEYFERT_CLASS):

            print('*** TESTING SEYFERT SAMPLES ***')

            if (bBlindTestAllSamples) or (bBlindTestSEYFERT):
                TP, FN, TPP, FNN = TestIndividualSampleSet2(seyfertTestData, DEFAULT_SEYFERT_CLASS,
                                                            binaryNameToModelDict, binaryModelList, binaryLabelDictList,
                                                            binaryScalerList, multiNameToModelDict, multiModelList,
                                                            multiLabelDictList, multiScalerList)

                accuracy += StoreBlindTestSummaryResults(fSummaryResults, DEFAULT_SEYFERT_CLASS, len(seyfertTestData),
                                                         TP, FN, TPP, FNN)



        elif (classList[labelNo] == DEFAULT_QUASAR_CLASS):

            print('*** TESTING QUASAR SAMPLES ***')

            if (bBlindTestAllSamples) or (bBlindTestQUASAR):
                TP, FN, TPP, FNN = TestIndividualSampleSet2(quasarTestData, DEFAULT_QUASAR_CLASS, binaryNameToModelDict,
                                                            binaryModelList, binaryLabelDictList,
                                                            binaryScalerList, multiNameToModelDict, multiModelList,
                                                            multiLabelDictList, multiScalerList)

                accuracy += StoreBlindTestSummaryResults(fSummaryResults, DEFAULT_QUASAR_CLASS, len(quasarTestData), TP,
                                                         FN, TPP, FNN)



        elif (classList[labelNo] == DEFAULT_AGN_CLASS):

            print('*** TESTING AGN SAMPLES ***')

            if (bBlindTestAllSamples) or (bBlindTestPULSAR):
                TP, FN, TPP, FNN = TestIndividualSampleSet2(agnTestData, DEFAULT_AGN_CLASS, binaryNameToModelDict,
                                                            binaryModelList, binaryLabelDictList,
                                                            binaryScalerList, multiNameToModelDict, multiModelList,
                                                            multiLabelDictList, multiScalerList)

                accuracy += StoreBlindTestSummaryResults(fSummaryResults, DEFAULT_AGN_CLASS, len(agnTestData), TP, FN,
                                                         TPP, FNN)


        else:
            print("*** Unknown Transient Class To Process ***")
            sys.exit()

    print("Average Accuracy = ", round((accuracy / MAX_NUMBER_MODELS), 2))
    fSummaryResults.close()


def FullTestNVSSCatalog():
    testDataLocation, testDataSet = SelectTestDataset(dataset, BINARY_MODEL_TYPE)

    #   testModelList, dictList, scalerList, classList, encoderList = LoadBinaryModels(DEFAULT_NVSS_DATASET,testDataLocation)

    #  finalBinaryModelDict, fullModelList, fullDictList, fullScalerList, fullEncoderList = LoadMultiClassModels(DEFAULT_NVSS_DATASET)

    fTestResults = OpenNVSSTestResultsFile()
    fSummaryResults = OpenNVSSTestSummaryFile()

    MAX_NUMBER_NVSS_TEST_IMAGES = 8568
    numberSamples = 0
    agnCount = 0
    blazarCount = 0
    quasarCount = 0
    seyfertCount = 0
    pulsarCount = 0
    unknownCount = 0
    numberPossibles = 0
    MIN_PROB_LIMIT = 0.6

    for imageNo in range(MAX_NUMBER_NVSS_TEST_IMAGES):

        bValidData, testData = loadNVSSCSVFile(imageNo)
        if (bValidData):
            transientClass, transientProb, possibleClass = ClassifyAnImage(testData, dictList, testModelList,
                                                                           scalerList, classList, finalBinaryModelDict,
                                                                           fullModelList,
                                                                           fullDictList, fullScalerList)
            numberSamples += 1
            if (possibleClass != 0):
                numberPossibles += 1
            if (transientClass == DEFAULT_AGN_CLASS):
                agnCount += 1
            elif (transientClass == DEFAULT_BLAZAR_CLASS):
                blazarCount += 1
            elif (transientClass == DEFAULT_QUASAR_CLASS):
                quasarCount += 1
            elif (transientClass == DEFAULT_SEYFERT_CLASS):
                seyfertCount += 1
            elif (transientClass == DEFAULT_PULSAR_CLASS):
                pulsarCount += 1
            else:
                unknownCount += 1

            print("Transient Class = " + transientClass + ", with Prob = " + str(round(transientProb, 4)))
            if (transientProb > MIN_PROB_LIMIT):
                StoreNVSSTestResults(fTestResults, imageNo, transientClass, transientProb, possibleClass)

    StoreNVSSTestSummaryResults(fSummaryResults, numberSamples, pulsarCount, agnCount, blazarCount, seyfertCount,
                                quasarCount, unknownCount, numberPossibles)
    fTestResults.close()
    fSummaryResults.close()


def TestRandomFITSFiles(numberFiles, model, XTest, ytest, labelDict):
    # test model with specific (but random) files

    print("*** Testing " + str(numberFiles) + " Random FITS Files ***")

    totalnoIncorrect = 0

    for dataset in range(numberFiles):

        randomEntry = int(random.random() * len(XTest))

        RandomSample = XTest[randomEntry].reshape(1, -1)

        print("shape = ", RandomSample.shape)
        correctLabel = ytest[randomEntry]

        y_pred = model.predict(RandomSample)
        if (bDebug):
            print("y_pred=", y_pred)

        pred_proba = model.predict_proba(RandomSample)
        if (bDebug):
            print(pred_proba)
        noInCorrectValues = 0

        for i in range(len(correctLabel)):
            if (y_pred[0][i] != correctLabel[i]):
                noInCorrectValues += 1

        labelText = ConvertOHE(correctLabel, labelDict)
        if (noInCorrectValues == 0):
            if (bDisplayIndividualPredictions):
                print("correct prediction for test number " + str(dataset) + "= " + labelText[0] + " !!")

        else:
            if (bDisplayIndividualPredictions):
                print("incorrect prediction for test number " + str(dataset) + "= " + labelText[0] + " !!")
            totalnoIncorrect += 1

    print("Total No Correct Predictions = ", numberFiles - totalnoIncorrect)
    print("Total No Incorrect Predictions = ", totalnoIncorrect)


def ProcessPulsarData():
    bValidData, pulsarData = loadPULSARData()

    if (bValidData):

        pulsarCoord = []

        numberPulsars = len(pulsarData)
        print("no of pulsars = ", numberPulsars)

        for pulsar in range(len(pulsarData)):
            skyCoord = []

            pulsarValues = pulsarData[[pulsar]]
            splitString = pulsarValues[0][0].split()
            #    print(splitString)

            Text1 = splitString[0]
            Text2 = splitString[1]
            Text3 = splitString[2]
            Text4 = splitString[3]

            print("Text1 = ", Text1)
            print("Text2 = ", Text2)
            print("Text3 = ", Text3)
            print("Text4 = ", Text4)

            RA = float(splitString[PULSAR_RA_LOC])
            DEC = float(splitString[PULSAR_DEC_LOC])

            print("pulsar = ", pulsar)
            print("RA = ", RA)
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
    print("number datasets to stack = ", numberDatasets)

    for dataset in range(numberDatasets):
        trainingData = allTrainingData[dataset]
        print("size of training data = ", len(trainingData))

        numberImagesToStack = len(trainingData)
        print("no images to stack = ", numberImagesToStack)
        stackedImage = trainingData[0]
        for image in range(1, len(trainingData)):
            stackedImage += trainingData[image]

        stackedImage = stackedImage / numberImagesToStack
        stackedTransientImage.append(stackedImage)

    return stackedTransientImage


def SetPlotParameters():
    plt.rc('axes', titlesize=SMALL_FONT_SIZE)
    plt.rc('axes', labelsize=SMALL_FONT_SIZE)
    plt.rc('xtick', labelsize=SMALL_FONT_SIZE)
    plt.rc('ytick', labelsize=SMALL_FONT_SIZE)


def DisplayStackedImages(stackedData, labelList):
    SetPlotParameters()

    fig, axs = plt.subplots(3, 2)

    figx = 0
    figy = 0

    for imageNo in range(0, len(stackedData)):
        imageData = stackedData[imageNo]
        x = np.arange(len(imageData[0]))

        axs[figx, figy].scatter(x, imageData[0], marker='+')
        axs[figx, figy].set_title(labelList[imageNo])
        axs[figx, figy].scatter(x, imageData[0], marker='+')
        axs[figx, figy].tick_params(axis='x', labelsize=SMALL_FONT_SIZE)

        figy += 1

        if (figy > 1):
            figx += 1
            figy = 0

    plt.show()

    fig.savefig(DEFAULT_STACKED_FILENAME_LOC)


def DisplayAllStackedImages(stackedData, labelList):
    SetPlotParameters()

    for imageNo in range(0, len(stackedData)):
        imageData = stackedData[imageNo]
        fig = plt.figure(imageNo)
        x = np.arange(len(imageData[0]))

        plt.scatter(x, imageData[0], marker='+')
        plt.title(labelList[imageNo])
        plt.scatter(x, imageData[0], marker='+')
        plt.tick_params(axis='x', labelsize=SMALL_FONT_SIZE)

        plt.show()
        filename = DEFAULT_STACKED_FILENAME_LOC + STACKED_FILENAME + UNDERSCORE + labelList[imageNo] + '.png'
        print("saving..." + filename)
        fig.savefig(filename)


def CreateSetCSVFiles(dataSet):
    if (dataSet == DEFAULT_NVSS_DATASET):
        # we're doing NVSS data
        rootData = DEFAULT_NVSS_DATA_ROOT
        print("Processing NVSS DATA")
    else:
        # we're doing VAST data
        rootData = DEFAULT_VAST_DATA_ROOT
        print("Processing VAST DATA")

    if (bCreateTESTFiles):
        print("*** Processing All TEST Files ***")

        sourceTESTList = ScanForSources(rootData + DEFAULT_TEST_SOURCE_LOCATION)

        testSourceFileDict = CreateAllCSVFiles(rootData + DEFAULT_TEST_SOURCE_LOCATION, sourceTESTList)

        totalNoFiles = ProcessAllCSVFiles(rootData + DEFAULT_TEST_SOURCE_LOCATION, testSourceFileDict, sourceTESTList)

        print("*** Processed " + str(totalNoFiles) + " TEST FILES")

    if (bCreateAGNFiles):
        print("*** Processing All AGN Files ***")

        sourceAGNList = ScanForSources(rootData + DEFAULT_AGN_SOURCE_LOCATION)

        agnSourceFileDict = CreateAllCSVFiles(rootData + DEFAULT_AGN_SOURCE_LOCATION, sourceAGNList)

        totalNoFiles = ProcessAllCSVFiles(rootData + DEFAULT_AGN_SOURCE_LOCATION, agnSourceFileDict, sourceAGNList)

        print("*** Processed " + str(totalNoFiles) + " AGN FILES")

    if (bCreatePULSARFiles):
        print("*** Processing All PULSAR Files ***")

        sourcePULSARList = ScanForSources(rootData + DEFAULT_PULSAR_SOURCE_LOCATION)

        pulsarSourceFileDict = CreateAllCSVFiles(rootData + DEFAULT_PULSAR_SOURCE_LOCATION, sourcePULSARList)

        totalNoFiles = ProcessAllCSVFiles(rootData + DEFAULT_PULSAR_SOURCE_LOCATION, pulsarSourceFileDict,
                                          sourcePULSARList)

        print("*** Processed " + str(totalNoFiles) + " PULSAR FILES ")

    if (bCreateQUASARFiles):
        print("*** Processing All QUASAR Files ***")

        sourceQUASARList = ScanForSources(rootData + DEFAULT_QUASAR_SOURCE_LOCATION)

        quasarSourceFileDict = CreateAllCSVFiles(rootData + DEFAULT_QUASAR_SOURCE_LOCATION, sourceQUASARList)

        totalNoFiles = ProcessAllCSVFiles(rootData + DEFAULT_QUASAR_SOURCE_LOCATION, quasarSourceFileDict,
                                          sourceQUASARList)

        print("*** Processed " + str(totalNoFiles) + " QUASAR FILES ")

    if (bCreateSEYFERTFiles):
        print("*** Processing All SEYFERT Files ***")

        sourceSEYFERTList = ScanForSources(rootData + DEFAULT_SEYFERT_SOURCE_LOCATION)

        seyfertSourceFileDict = CreateAllCSVFiles(rootData + DEFAULT_SEYFERT_SOURCE_LOCATION, sourceSEYFERTList)

        totalNoFiles = ProcessAllCSVFiles(rootData + DEFAULT_SEYFERT_SOURCE_LOCATION, seyfertSourceFileDict,
                                          sourceSEYFERTList)

        print("*** Processed " + str(totalNoFiles) + " SEYFERT FILES ")

    if (bCreateBLAZARFiles):
        print("*** Processing All BLAZAR Files ***")

        sourceBLAZARList = ScanForSources(rootData + DEFAULT_BLAZAR_SOURCE_LOCATION)

        blazarSourceFileDict = CreateAllCSVFiles(rootData + DEFAULT_BLAZAR_SOURCE_LOCATION, sourceBLAZARList)

        totalNoFiles = ProcessAllCSVFiles(rootData + DEFAULT_BLAZAR_SOURCE_LOCATION, blazarSourceFileDict,
                                          sourceBLAZARList)

        print("*** Processed " + str(totalNoFiles) + " BLAZAR FILES ")


def CreateTransientCSVFiles(dataSet):
    if (dataSet == DEFAULT_POORQUALITY_DATA):

        rootData = DEFAULT_POOR_QUALITY_DATA_ROOT
        print("Processing POOR QUALITY DATA")
    elif (dataSet == DEFAULT_ARTEFACT_DATA):

        rootData = DEFAULT_ARTEFACT_DATA_ROOT
        print("Processing ARTEFACT DATA")

    else:
        print("Unknown Dataset, exiting...")
        sys.exit()

    sourceList = ScanForSources(rootData)

    sourceFileDict = CreateAllCSVFiles(rootData, sourceList)

    totalNoFiles = ProcessAllCSVFiles(rootData , sourceFileDict, sourceList)

    print("*** Processed " + str(totalNoFiles) + " FITS FILES")



def CreateModelFileName(dataSet, modelType, labelList):
    if (dataSet == DEFAULT_NVSS_DATASET):

        filename = NVSS_SHORT_NAME + UNDERSCORE
        modelLocation = DEFAULT_NVSS_DATA_ROOT
    else:

        filename = VAST_SHORT_NAME + UNDERSCORE
        modelLocation = DEFAULT_VAST_DATA_ROOT

    if (modelType == OP_MODEL_1DCNN):
        modelLocation = modelLocation + DEFAULT_1DCNN_OUTPUT_MODEL
    if (modelType == OP_MODEL_2DCNN):
        modelLocation = modelLocation + DEFAULT_2DCNN_OUTPUT_MODEL
    elif (modelType == OP_MODEL_RANDOM):
        modelLocation = modelLocation + DEFAULT_RF_OUTPUT_MODEL
    elif (modelType == OP_MODEL_MLP):
        modelLocation = modelLocation + DEFAULT_MLP_OUTPUT_MODEL

    filename = filename + modelType + UNDERSCORE

    filename += labelList[0][0]

    for labelNo in range(1, len(labelList)):
        filename = filename + UNDERSCORE + labelList[labelNo][0]
    if (modelType == OP_MODEL_RANDOM):
        fullFilename = modelLocation + filename + MODEL_FILENAME_EXTENSION
    else:
        fullFilename = modelLocation + filename

    return filename, fullFilename, modelLocation


def SaveModelDict(modelLocation, fileName, labelDict):
    import pandas as pd

    filename = modelLocation + fileName + DEFAULT_DICT_TYPE

    df = pd.DataFrame(labelDict)

    df.to_csv(filename)


def SaveModelScaler(modelLocation, fileName, scaler):
    import pickle

    bSavedOK = False

    filename = modelLocation + fileName + DEFAULT_SCALER_TYPE

    f = open(filename, 'wb')
    if (f):
        pickle.dump(scaler, f)
        bSavedOK = True
        print("***Saving Scaler...***")

        f.close()

    return bSavedOK


def SaveModelEncoder(modelLocation, fileName, encoder):
    import pickle

    bSavedOK = False

    filename = modelLocation + fileName + DEFAULT_ENCODER_TYPE

    f = open(filename, 'wb')
    if (f):
        pickle.dump(encoder, f)
        bSavedOK = True
        print("***Saving Encoder...***")

        f.close()

    return bSavedOK


def GetSavedDict(dictLocation, modelName):
    from os.path import splitext

    labelDict = {}

    filename, ext = splitext(modelName)

    filename = dictLocation + filename + DEFAULT_DICT_TYPE

    print("Retrieving Label Dict ...", filename)

    dataframe = pd.read_csv(filename, usecols=range(1, 3))

    colNames = list(dataframe.columns)

    for col in colNames:
        fullOHE = np.array([dataframe[col][0]])
        for element in range(1, len(dataframe[col])):
            fullOHE = np.append(fullOHE, dataframe[col][element])

        labelDict[col] = fullOHE

    return labelDict


def GetSavedScaler(scalerLocation, modelName):
    from os.path import splitext
    import pickle

    filename, ext = splitext(modelName)

    filename = scalerLocation + filename + DEFAULT_SCALER_TYPE

    print("Retrieving Scaler ...", filename)

    with open(filename, 'rb') as file:
        scaler = pickle.load(file)

    return scaler


def GetSavedEncoder(encoderLocation, modelName):
    from os.path import splitext
    import pickle

    filename, ext = splitext(modelName)

    filename = encoderLocation + filename + DEFAULT_ENCODER_TYPE

    print("Retrieving encoder ...", filename)

    with open(filename, 'rb') as file:
        encoder = pickle.load(file)

    return encoder


def SaveModel(dataSet, labelDict, model, scaler, labelList, encoder):
    import pickle

    print("*** Saving Model ***")

    filename, fullFilename, modelLocation = CreateModelFileName(dataSet, OP_MODEL_RANDOM, labelList)

    print("*** as ...." + fullFilename + " ***")
    with open(fullFilename, 'wb') as file:
        pickle.dump(model, file)

    SaveModelDict(modelLocation, filename, labelDict)

    SaveModelScaler(modelLocation, filename, scaler)
    SaveModelEncoder(modelLocation, filename, encoder)


def SaveCNNModel(modelType, dataSet, labelDict, model, scaler, labelList, encoder, numberEpochs):
    print("*** Saving CNN Model ***")

    filename, fullFilename, modelLocation = CreateModelFileName(dataSet, modelType, labelList)
    if (numberEpochs > 0):
        filename = filename + UNDERSCORE + str(numberEpochs)
    model.save(modelLocation + filename)

    SaveModelDict(modelLocation, filename, labelDict)

    SaveModelScaler(modelLocation, filename, scaler)

    SaveModelEncoder(modelLocation, filename, encoder)


def GetExistingModels():
    from os.path import splitext

    modelList = []

    print("*** Scanning Models In " + DEFAULT_EXISTING_MODEL_LOCATION + " ***")
    fileList = os.scandir(DEFAULT_EXISTING_MODEL_LOCATION)
    for entry in fileList:
        if entry.is_file():
            if entry.name[0] != DS_STORE_FILENAME:
                filename, ext = splitext(entry.name)
                if (ext == MODEL_FILENAME_EXTENSION):
                    modelList.append(entry.name)

    return modelList


def GetSavedModel(modelLocation, modelFilename):
    import pickle

    filename = modelLocation + modelFilename
    print("Retrieving Model...", filename)

    with open(filename, 'rb') as file:
        pickleModel = pickle.load(file)

    return pickleModel


def GetSavedCNNModel(modelLocation, modelFilename):
    from tensorflow import keras

    filename = modelLocation + modelFilename
    print("Retrieving CNN Model...", filename)

    model = keras.models.load_model(filename)

    return model


def SelectExistingModel():
    import pickle

    bCorrectInput = False

    modelList = GetExistingModels()

    for model in range(len(modelList)):
        strr = str(model + 1) + ' : ' + modelList[model]
        print(strr)

    while (bCorrectInput == False):

        modelNumber = int(input('Select Model Number: '))
        if (modelNumber > 0) and (modelNumber < (len(modelList) + 1)):
            bCorrectInput = True
        else:
            print("*** Incorrect Selection - try again ***")

    print('*** SELECTED MODEL: ' + modelList[modelNumber - 1] + ' ***')

    modelName = modelList[modelNumber - 1]

    encoder = GetSavedEncoder(DEFAULT_EXISTING_MODEL_LOCATION, modelName)

    labelDict = GetSavedDict(DEFAULT_EXISTING_MODEL_LOCATION, modelName)

    scaler = GetSavedScaler(DEFAULT_EXISTING_MODEL_LOCATION, modelName)

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

    pickleModel = GetSavedModel(DEFAULT_EXISTING_MODEL_LOCATION, modelName)

    return classToTest, pickleModel, labelDict, scaler, dataSet, encoder


def TestRetrievedModel(labelDict, XTest, ytest, model):
    score = model.score(XTest, ytest)
    print("Test Score (Retrieved Model): {0:.2f} %".format(100 * score))
    sys.exit()
    i = 0
    for sampleNo in range(len(XTest)):
        X = XTest[sampleNo]

        X = X.reshape(1, -1)

        prediction = model.predict(X)

        #  print("pred = ",prediction[0])
        bDetection, predictedLabelList = DecodePredictedLabel2(labelDict, prediction)
        if (predictedLabelList[0] != DEFAULT_PULSAR_CLASS):
            i += 1
    #   print("Test Score (Retrieved Model): {0:.2f} %".format(100*score))

    print(" incorrect = ", i)

    sys.exit()


def AccessNVSSData():
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    hdul = fits.open(DEFAULT_NVSS_CATALOG_FILENAME)

    nvssData = hdul[1].data
    print("size of data = ", len(nvssData))

    #   for entry in range(len(nvssData)):
    for entry in range(10):
        nvssDataEntry = nvssData[entry]
        print("RA, DEC = ", nvssDataEntry[0], nvssDataEntry[1])
        c = SkyCoord(ra=nvssDataEntry[0] * u.degree, dec=nvssDataEntry[1] * u.degree)

    sys.exit()


def OpenDuplicateFile():
    f = open(DEFAULT_DUPLICATES_FILENAME, "w")

    return f


def SaveInDuplicateFile(f, source):
    if (f):
        f.write("DUPLICATE FOR SOURCE " + str(source))
        f.write("\n")


def FindSourceInList(f, source, sourceList):
    NoFound = 0

    if (source in sourceList):
        SaveInDuplicateFile(f, source)
        NoFound += 1

    return NoFound


def CompareFITSImages(FITSImage1, FITSImage2):
    bEqual = np.array_equal(FITSImage1, FITSImage2)
    if (bEqual):
        print("FITS Images are IDENTICAL ")
    else:
        print("FITS Images are NOT IDENTICAL ")

    return bEqual


def CheckFITSIntegrity(sourceLocation, sourceList):
    # select random file from sourceList

    randomEntry = int(random.random() * len(sourceList))

    source = sourceList[randomEntry]

    imageLocation, imageList = ScanForImages(sourceLocation, source)

    randomImage1 = int(random.random() * len(imageList))
    randomImage2 = int(random.random() * len(imageList))

    if (randomImage1 == randomImage2):
        print("Same random image chosen - exit")
        sys.exit()
    else:
        imageLocation += FOLDER_IDENTIFIER
        bValidImage1, FITSImage1 = OpenFITSFile(imageLocation + imageList[randomImage1])
        bValidImage2, FITSImage2 = OpenFITSFile(imageLocation + imageList[randomImage2])

        if ((bValidImage1) and (bValidImage2)):
            CompareFITSImages(FITSImage1, FITSImage2)
        else:
            print("*** INVALID FITS IMAGES ***")
    sys.exit()


def CheckForDuplicatedSources():
    f = OpenDuplicateFile()

    agnSourceList = ScanForSources(DEFAULT_AGN_SOURCE_LOCATION)
    print("NO AGN = ", len(agnSourceList))
    seyfertSourceList = ScanForSources(DEFAULT_SEYFERT_SOURCE_LOCATION)
    blazarSourceList = ScanForSources(DEFAULT_BLAZAR_SOURCE_LOCATION)
    quasarSourceList = ScanForSources(DEFAULT_QUASAR_SOURCE_LOCATION)
    print("NO QUASAR = ", len(quasarSourceList))
    pulsarSourceList = ScanForSources(DEFAULT_PULSAR_SOURCE_LOCATION)
    print("NO PULSAR = ", len(pulsarSourceList))

    if (bCheckFITSIntegrity):
        bCheckFITS = CheckFITSIntegrity(DEFAULT_AGN_SOURCE_LOCATION, agnSourceList)

    noFoundInSEYFERT = 0
    noFoundInBLAZAR = 0
    noFoundInQUASAR = 0
    noFoundInPULSAR = 0

    for agn in range(len(agnSourceList)):
        agnSource = agnSourceList[agn]

        noFoundInSEYFERT += FindSourceInList(f, agnSource, seyfertSourceList)
        noFoundInBLAZAR += FindSourceInList(f, agnSource, blazarSourceList)
        noFoundInQUASAR += FindSourceInList(f, agnSource, quasarSourceList)
        noFoundInPULSAR += FindSourceInList(f, agnSource, pulsarSourceList)

    if (noFoundInSEYFERT > 0):
        print("*** NO. DUPLICATES FOUND FOR AGNs in SEYFERTS = " + str(noFoundInSEYFERT) + "***")
    else:
        print("*** NO DUPLICATES FOUND FOR AGNS IN SEYFERTS ***")
    if (noFoundInBLAZAR > 0):
        print("*** NO. DUPLICATES FOUND FOR AGNs in BLAZARS = " + str(noFoundInBLAZAR) + " ***")
    else:
        print("*** NO DUPLICATES FOUND FOR AGNS IN BLAZARS ***")

    if (noFoundInQUASAR > 0):
        print("*** NO. DUPLICATES FOUND FOR AGNs in QUASARS = " + str(noFoundInQUASAR) + " ***")
    else:
        print("*** NO DUPLICATES FOUND FOR AGNS IN QUASARS ***")
    if (noFoundInPULSAR > 0):
        print("*** NO. DUPLICATES FOUND FOR AGNs in PULSARS = " + str(noFoundInPULSAR) + "***")
    else:
        print("*** NO DUPLICATES FOUND FOR AGNS IN PULSARS ***")

    print("\n")

    noFoundInAGN = 0
    noFoundInBLAZAR = 0
    noFoundInQUASAR = 0
    noFoundInPULSAR = 0

    for seyfert in range(len(seyfertSourceList)):
        seyfertSource = seyfertSourceList[seyfert]

        noFoundInAGN += FindSourceInList(f, seyfertSource, agnSourceList)
        noFoundInBLAZAR += FindSourceInList(f, seyfertSource, blazarSourceList)
        noFoundInQUASAR += FindSourceInList(f, seyfertSource, quasarSourceList)
        noFoundInPULSAR += FindSourceInList(f, seyfertSource, pulsarSourceList)

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

        noFoundInAGN += FindSourceInList(f, blazarSource, agnSourceList)
        noFoundInSEYFERT += FindSourceInList(f, blazarSource, seyfertSourceList)
        noFoundInQUASAR += FindSourceInList(f, blazarSource, quasarSourceList)
        noFoundInPULSAR += FindSourceInList(f, blazarSource, pulsarSourceList)

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

        noFoundInAGN += FindSourceInList(f, quasarSource, agnSourceList)
        noFoundInBLAZAR += FindSourceInList(f, quasarSource, blazarSourceList)
        noFoundInSEYFERT += FindSourceInList(f, quasarSource, seyfertSourceList)
        noFoundInPULSAR += FindSourceInList(f, quasarSource, pulsarSourceList)

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

        noFoundInAGN += FindSourceInList(f, pulsarSource, agnSourceList)
        noFoundInBLAZAR += FindSourceInList(f, pulsarSource, blazarSourceList)
        noFoundInQUASAR += FindSourceInList(f, pulsarSource, quasarSourceList)
        noFoundInSEYFERT += FindSourceInList(f, pulsarSource, seyfertSourceList)

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


def StoreNVSSSources(f, name, ra, dec):
    f.write(name + ',' + ra + ',' + dec)
    f.write('\n')


def ViewImagesToRetrieve(df):
    for entry in range(len(df.RA)):
        strr = 'Skyview Source ' + str(entry) + ' ' + str(df.SOURCE[entry]) + ' = ' + str(df.RA[entry]) + ' ' + str(
            df.DEC[entry])

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
        sourcesFilename = DEFAULT_NVSS_SEYFERT_SOURCES_FILENAME

    elif (transientType == DEFAULT_NVSS_BLAZAR_CLASS):
        sourceDir = DEFAULT_NVSS_BLAZAR_SOURCE_LOCATION
        sourcesFilename = DEFAULT_NVSS_BLAZARS_SOURCES_FILENAME

    print("Reading NVSS CSV file ..." + sourcesFilename)

    df = pd.read_csv(sourcesFilename)

    print("Total No of Images To Be Processed = ", len(df.RA))
    fitsFileNumber = 0

    for entry in range(len(df.RA)):
        bValidData = True

        strr = 'Skyview Source ' + str(df.SOURCE[entry]) + ' = ' + str(df.RA[entry]) + ' ' + str(df.DEC[entry])

        print(strr)

        coordStrr = str(df.RA[entry]) + ' ' + str(df.DEC[entry])

        sv = SkyView()

        try:

            imagePaths = sv.get_images(position=coordStrr, coordinates="J2000", survey='NVSS')

        except:
            bValidData = False
            print("*** Error in Skyview call - moving on ***")

        if (bValidData):
            for fitsImage in imagePaths:
                print('new file for source', df.SOURCE[entry])
                fitsFileNumber += 1
                print("Storing FITS NVSS File Number: ", fitsFileNumber)
                StoreNVSSFITSImage(transientType, sourceDir, df.SOURCE[entry], fitsImage, fitsFileNumber)

    print("Finished...exiting")
    sys.exit()


def SelectSkyviewImages(ra, dec):
    from astroquery.skyview import SkyView
    import pandas as pd
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import ssl

    imageList = []
    bValidData = True

    ssl._create_default_https_context = ssl._create_unverified_context
    coordStrr = ra + ' ' + dec

    print('Skyview Coordinates = ' + coordStrr)

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


def ProcessAstroqueryTransient(startSelection, catalogName, transientName, OutputSourcesFilename, entryRA, entryDEC,
                               bDegree):
    from astroquery.vizier import Vizier
    from astroquery.nrao import Nrao
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.coordinates import Angle
    from astropy.coordinates import SkyCoord
    from astroquery.image_cutouts.first import First
    import time

    print('NVSS Processing ' + transientName + '....')
    print(catalogName)
    catalog_list = Vizier.find_catalogs(catalogName)

    print({k: v.description for k, v in catalog_list.items()})
    Vizier.ROW_LIMIT = -1

    Catalog = Vizier.get_catalogs(catalog_list.keys())

    print(Catalog)

    transientTable = Catalog[0]

    numberTransients = len(transientTable)

    print('Total Number ' + transientName + ' in Vizier Catalog = ' + str(numberTransients))

    allVizierCoords = []

    for entry in range(numberTransients):

        print("Processing entry...", entry)

        try:
            if (bDegree):

                skyCoords = SkyCoord(transientTable[entry][entryRA], transientTable[entry][entryDEC], unit=("deg"))
            else:
                skyCoords = SkyCoord(transientTable[entry][entryRA] + ' ' + transientTable[entry][entryDEC],
                                     unit=(u.hourangle, u.deg))

            allVizierCoords.append(skyCoords)
        except:

            print("*** Error in Vizier coordinates  - moving on ***")

    sourcesFound = 0

    NVSS_RA_Coords = []
    NVSS_DEC_Coords = []

    print('Opening Output File ' + OutputSourcesFilename + '....')

    fNVSS = open(OutputSourcesFilename, "w")
    bComplete = False

    entry = startSelection
    noErrors = 0

    while (bComplete == False):

        try:
            print("Querying entry " + str(entry) + " from total of " + str(numberTransients))

            results_table = Nrao.query_region(
                coord.SkyCoord(allVizierCoords[entry].ra, allVizierCoords[entry].dec, unit=u.degree),
                radius=15 * u.arcsec)
            print("RESULT OK")
            if (len(results_table) > 0):

                print(transientName + ' SOURCE FOUND')
                sourcesFound += 1
                print(results_table)

                ra = results_table['RA'][0]
                dec = results_table['DEC'][0]

                NVSS_RA_Coords.append(ra)
                NVSS_DEC_Coords.append(dec)

                StoreNVSSSources(fNVSS, results_table['Source'][0], ra, dec)
            else:
                print('NO ' + transientName + ' SOURCE FOUND')

            print('TOTAL NO ' + transientName + ' SOURCES FOUND SO FAR = ' + str(sourcesFound))

            print('TOTAL NO ERRORS SO FAR = ' + str(noErrors))

            if (sourcesFound >= 600):
                print('600 ' + transientName + ' SOURCES FOUND AT ENTRY No. =' + str(entry))
                bComplete = True
            else:
                entry += 1
                if (entry >= numberTransients):
                    print("AT END OF CATALOG")
                    bComplete = True

        except:
            print("Invalid return - carrying on")
            noErrors += 1
            entry += 1
            if (entry >= numberTransients):
                print("AT END OF CATALOG")
                bComplete = True

    print('TOTAL NO ' + transientName + ' SOURCES FOUND = ' + str(sourcesFound))

    fNVSS.close()
    sys.exit()


def BuildandTest1DCNNModel(dataset, labelList, XTrain, XTest, ytrain, ytest, numberEpochs, learningRate, dropoutRate,
                           numberNeurons):
    n_timesteps = XTrain.shape[2]
    n_features = XTrain.shape[1]
    n_outputs = ytrain.shape[1]

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

            Accuracy, CNNModel = test1DCNNModel(BestEpochs, BestLearningRate, BestDropoutRate, XTrain, ytrain, XTest,
                                                ytest)
        else:
            print("*** Failed To Open CNN Hyper Parameters File ***")

    else:
        print("*** Evaluating 1D CNN Model ***")

        Accuracy, CNNModel = evaluate1DCNNModel(dataset, labelList, XTrain, ytrain, XTest, ytest, n_timesteps,
                                                n_features, n_outputs,
                                                numberEpochs, learningRate, dropoutRate, numberNeurons)

    return CNNModel


def BuildandTest2DCNNModel(dataset, labelList, XTrain, XTest, ytrain, ytest, numberEpochs, learningRate, dropoutRate,
                           numberNeurons):
    n_outputs = ytrain.shape[1]

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

            Accuracy, CNNModel = test2DCNNModel(BestEpochs, BestLearningRate, BestDropoutRate, XTrain, ytrain, XTest,
                                                ytest)
        else:
            print("*** Failed To Open CNN Hyper Parameters File ***")

    else:
        print("*** Evaluating 2D CNN Model ***")

        Accuracy, CNNModel = evaluate2DCNNModel(dataset, labelList, XTrain, ytrain, XTest, ytest, n_outputs,
                                                numberEpochs, learningRate, dropoutRate, numberNeurons)

    return CNNModel


def BuildandTestAlexNetModel(dataset, labelList, XTrain, XTest, ytrain, ytest, numberEpochs, learningRate):
    n_outputs = ytrain.shape[1]

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

        #  Accuracy, CNNModel = testAlexNetModel(BestEpochs, BestLearningRate, BestDropoutRate, XTrain, ytrain, XTest,
        #                                   ytest)
        else:
            print("*** Failed To Open CNN Hyper Parameters File ***")

    else:
        print("*** Evaluating ALEXNet CNN Model ***")

        Accuracy, CNNModel = evaluateAlexNetModel(dataset, numberEpochs, learningRate, labelList, XTrain, ytrain, XTest,
                                                  ytest)

    return CNNModel


def BuildandTestMLPModel(XTrain, XTest, ytrain, ytest):
    if (bOptimiseHyperParameters == True):

        fOptimisationFile = OpenOptimisationFile()

        if (fOptimisationFile):
            print("*** Optimising MLP Hyper Parameters ***")
            ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates, ExperimentDropoutRates = OptimiseMLPHyperparameters(
                fOptimisationFile, XTrain, ytrain, XTest, ytest)
            fOptimisationFile.close()
            DisplayHyperTable(ExperimentAccuracy, ExperimentEpochs, ExperimentLearningRates, ExperimentDropoutRates)
            BestEpochs, BestLearningRate, BestDropoutRate = GetOptimalParameters(ExperimentAccuracy, ExperimentEpochs,
                                                                                 ExperimentLearningRates,
                                                                                 ExperimentDropoutRates)

            Accuracy, MLPModel = testMLPModel(BestEpochs, BestLearningRate, BestDropoutRate, XTrain, ytrain, XTest,
                                              ytest)
        else:
            print("*** Failed To Open MLP Hyper Parameters File ***")

    else:
        print("*** Evaluating MLP Model ***")

        Accuracy, CNNModel = evaluateMLPModel(XTrain, ytrain, XTest, ytest, DEFAULT_NO_EPOCHS, DEFAULT_LEARNING_RATE)

    print("Final MLP Accuracy = ", Accuracy)

    return MLPModel


def evaluateAlexNetModel(dataset, n_epochs, learningRate, labelList, XTrain, ytrain, XTest, ytest):
    from keras.callbacks import EarlyStopping
    from keras.backend import clear_session

    verbose, batchSize = DEFAULT_VERBOSE_LEVEL, DEFAULT_BATCH_SIZE

    clear_session()
    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(XSIZE_FITS_IMAGE, YSIZE_FITS_IMAGE, 1),
                                     use_bias=True))
    model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', use_bias=True))

    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.BatchNormalization())

    ## change from 0.01 to 0.02
    DEFAULT_REGULARIZER = 0.04
    #   model.add(tf.keras.layers.Conv2D(128, 3, activation='relu',use_bias=True,kernel_regularizer = tf.keras.regularizers.l1(l= DEFAULT_REGULARIZER)))
    model.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    #  model.add(tf.keras.layers.BatchNormalization())

    #  model.add(tf.keras.layers.Conv2D(256, 3, activation='relu',use_bias=True,kernel_regularizer = tf.keras.regularizers.l1(l= DEFAULT_REGULARIZER)))
    model.add(tf.keras.layers.Conv2D(256, 3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, 3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, 3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    # model.add(tf.keras.layers.BatchNormalization())

    #  model.add(tf.keras.layers.Conv2D(512, 3, activation='relu', use_bias=True,kernel_regularizer=tf.keras.regularizers.l1(l=DEFAULT_REGULARIZER)))
    model.add(tf.keras.layers.Conv2D(512, 3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, 3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, 3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    # model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(DEFAULT_DROPOUT_RATE))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(DEFAULT_DROPOUT_RATE))

    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=learningRate)
    #  opt = tf.keras.optimizers.SGD(momentum=0.9,learning_rate=learningRate)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print("Training Model...")

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=12)
    #   history = model.fit(XTrain, ytrain, epochs=n_epochs, batch_size=batchSize, verbose=1, validation_split=0.2,callbacks=[es])
    history = model.fit(XTrain, ytrain, epochs=n_epochs, batch_size=batchSize, shuffle=True, verbose=1,
                        validation_split=0.2, callbacks=[es])
    accuracy = model.evaluate(XTest, ytest, batch_size=batchSize, verbose=verbose)
    SaveCNNModelAnalysis(OP_MODEL_ALEXNET, dataset, history, labelList, n_epochs, accuracy[1])
    print("Testing Model...")
    _, accuracy = model.evaluate(XTest, ytest, batch_size=batchSize, verbose=verbose)

    return accuracy, model


def ProcessNVSSCatalog():
    print("Transforming NVSS Catalog")
    bComplete = False

    fIn = open(DEFAULT_NVSS_CATALOG_LOCATION)
    fOut = open(FINAL_NVSS_CATALOG_LOCATION, 'w')

    if (fIn) and (fOut):

        sourceNumber = 0
        # read one line at a time and parse to output file
        while (bComplete == False):
            line = fIn.readline()
            if not line:
                bComplete = True
            else:
                print(line)
                if ('##') in line:
                    print("Ignoring this line..." + line)
                elif ('NVSS') in line:
                    print("Ignoring this line..." + line)
                elif ('RA(2000)') in line:
                    print("Ignoring this line..." + line)
                elif ('mJ') in line:
                    print("Ignoring this line..." + line)
                else:

                    sourceNumber += 1

                    splitLine = line.split()
                    if (len(splitLine) >= MAX_NVSS_RA_DEC_STRINGS):
                        ra = splitLine[NVSS_CAT_RA_POS] + ' ' + splitLine[NVSS_CAT_RA_POS + 1] + ' ' + splitLine[
                            NVSS_CAT_RA_POS + 2]
                        dec = splitLine[NVSS_CAT_DEC_POS] + ' ' + splitLine[NVSS_CAT_DEC_POS + 1] + ' ' + splitLine[
                            NVSS_CAT_DEC_POS + 2]
                        newLine = str(sourceNumber) + ' , ' + ra + ' ' + dec

                        fOut.write(newLine)
                        fOut.write('\n')

                    line = fIn.readline()  # throw away this line
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
            chosenEntry = int(random.random() * DEFAULT_MAX_NO_NVSS_ENTRIES)

            entrySelected.append(chosenEntry)

    # now select each of these random lines from the NVSS Catalog

    fIn = open(FINAL_NVSS_CATALOG_LOCATION)
    fOut = open(FINAL_SELECTED_NVSS_SOURCES_LOCATION, "w")
    entryNo = 1
    bComplete = False

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
                entryNo += 1

        fIn.close()
        fOut.close()


def SelectRandomNVSSSource():
    chosenEntry = int(random.random() * DEFAULT_MAX_NO_NVSS_ENTRIES)

    # now select each of these random lines from the NVSS Catalog

    fIn = open(FINAL_NVSS_CATALOG_LOCATION)

    bComplete = False
    bValidData = False
    entryNo = 1

    if (fIn):

        while (bComplete == False):
            line = fIn.readline()
            if not line:
                bComplete = True
            else:
                if (entryNo == chosenEntry):
                    bComplete = True
                    bValidData = True
                    # now get the ra and dec

                    splitText = line.split()

                    ra = splitText[2] + ' ' + splitText[3] + ' ' + splitText[4]
                    dec = splitText[5] + ' ' + splitText[6] + ' ' + splitText[7]

                entryNo += 1

        fIn.close()

    return bValidData, ra, dec


def OpenNVSSDetections():
    f = open(NVSS_CATALOG_DETECTIONS_FILENAME, 'w')

    return f


def StoreNVSSDetections(f, source, ra, dec, label):
    if (f):
        print('Detection For Source No: ' + source + ' = ' + label)
        f.write('Detection For Source No: ' + source + ',' + 'ra= ' + ra + ', dec= ' + dec + ' = ' + label)
        f.write('\n')


def TestNVSSCatalog(startEntry):
    print("Extracting and Storing Samples From NVSS Catalog....")
    totalNoSources = 0
    entry = 0
    fIn = open(FINAL_NVSS_CATALOG_FILE)
    fDetect = OpenNVSSDetections()

    bComplete = False

    if (fIn) and (fDetect):
        MAX_NVSS_TO_PROCESS = 10000

        while (bComplete == False):
            line = fIn.readline()
            entry += 1
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
                            bValidFile, fitsImageFile = StoreNVSSCatalogImage(fitsImage, source)
                            if (bValidFile):

                                bStoreOK, csvFilename = StoreIndividualCSVFile(fitsImageFile, source)
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


def TestNVSSFiles(model, labelDict):
    print("Testing from NVSS Files....")

    bValidImageList, fitsImageList = ScanForTestImages(DEFAULT_TEST_NVSS_SOURCE_LOCATION)

    if (bValidImageList):
        for fitsImage in fitsImageList:
            print("fits image = ", fitsImage)
            bValidData, fitsData = OpenFITSFile(DEFAULT_TEST_NVSS_SOURCE_LOCATION + fitsImage)
            if (bValidData):
                fitsData = np.reshape(fitsData, (1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE))

                if (bScaleInputs):
                    fitsData = ScaleInputData(fitsData)
                ypredict = model.predict(fitsData)
                print("ypredict = ", ypredict)

                bDetection, predictedLabel = DecodePredictedLabel(labelDict, ypredict)


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
            print("*** Loading CSV File " + filePath + " ***")
        dataframe = pd.read_csv(filePath, header=None)
        dataReturn = dataframe.values
        if (bDebug):
            print("*** Completed Loading Random Image CSV File")
    else:
        print("*** Random Image CSV File Does Not Exist ***")
        bDataValid = False
        dataReturn = []

    return bDataValid, dataReturn


def StoreRandomImages(primaryLabel, randomTrainingData):
    print("*** Storing Randomly Selected Images ...***")

    filename = DEFAULT_RANDOM_FILE_TEST_LOCATION + primaryLabel[0] + DEFAULT_RANDOM_TYPE

    f = open(filename, 'w')
    if (f):
        for imageNo in range(len(randomTrainingData)):
            StoreImageContents(f, randomTrainingData[imageNo])
        f.close()


def EqualiseAllDataSamples(completeTrainingData, trainingDataSizes):
    reducedTrainingData = []

    maxSampleSize = max(trainingDataSizes)
    minSampleSize = min(trainingDataSizes)

    # now make all data samples the same size

    for sample in range(len(completeTrainingData)):
        trainingData = completeTrainingData[sample]

        if (trainingDataSizes[sample] > minSampleSize):
            del trainingData[minSampleSize:trainingDataSizes[sample]]
            trainingDataSizes[sample] = minSampleSize
        if (bTestCNNTest1):
            del trainingData[DEFAULT_MIN_CNN_DATA:minSampleSize]
            trainingDataSizes[sample] = DEFAULT_MIN_CNN_DATA

    return completeTrainingData, trainingDataSizes


def ProcessAllTransientModelData(dataSet, labelList, bEqualise):
    completeTrainingData = []
    trainingDataSizes = []

    print("*** Loading Training Data ***")

    for classes in range(len(labelList)):
        trainingData, sourceDetails = ProcessTransientData(dataSet, labelList[classes], 0)

        trainingDataSizes.append(len(trainingData))
        completeTrainingData.append(trainingData)

    if (bEqualise):
        completeTrainingData, trainingDataSizes = EqualiseAllDataSamples(completeTrainingData, trainingDataSizes)

    return completeTrainingData, trainingDataSizes


def ProcessBinaryModelData(dataSet, primaryLabel, otherLabels, bEqualise):
    completeTrainingData = []
    trainingDataSizes = []
    otherTrainingData = []

    print("*** Loading Primary Training Data ***")

    trainingData, sourceDetails = ProcessTransientData(dataSet, primaryLabel, 0)
    if (bEqualise):
        maxNumberSamples = int(len(trainingData) / (DEFAULT_NUMBER_MODELS - 1))
    else:
        maxNumberSamples = 0

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

    return completeTrainingData, trainingDataSizes


def ProcessSingleModelData(dataSet, primaryLabel, scaler, labelDict):
    print("*** Loading Primary Training Data For Single Model ***")

    testData, sourceDetails = ProcessTransientData(dataSet, primaryLabel, 0)
    testData = TransformTrainingData(testData)

    scaledData = scaler.transform(testData)

    if primaryLabel in labelDict:
        OHELabelValue = labelDict[primaryLabel]
    else:
        print("No associated label value - exiting...")
        sys.exit()

    testLabels = assignLabelValues(OHELabelValue, len(scaledData))

    return scaledData, testLabels


def GetCNNParameters(defaultModel):
    bCompleteInput = False
    print("*** Change CNN Parameters ***")
    while (bCompleteInput == False):
        numberEpochs = input(
            'Default No Epochs = ' + str(DEFAULT_NO_EPOCHS) + ' (Enter to accept, number to change) : ')
        if (numberEpochs == ""):
            numberEpochs = DEFAULT_NO_EPOCHS
        else:
            numberEpochs = int(numberEpochs)

        learningRate = input(
            'Default Learning Rate = ' + str(DEFAULT_LEARNING_RATE) + ' (Enter to accept, number to change) : ')
        if (learningRate == ""):
            learningRate = DEFAULT_LEARNING_RATE
        else:
            learningRate = float(learningRate)

        dropoutRate = input(
            'Default Dropout Rate = ' + str(DEFAULT_DROPOUT_RATE) + ' (Enter to accept, number to change) : ')
        if (dropoutRate == ""):
            dropoutRate = DEFAULT_DROPOUT_RATE
        else:
            dropoutRate = float(dropoutRate)

        numberNeurons = input(
            'Default No Neurons = ' + str(DEFAULT_NO_NEURONS) + ' (Enter to accept, number to change) : ')
        if (numberNeurons == ""):
            numberNeurons = DEFAULT_NO_NEURONS
        else:
            numberNeurons = int(numberNeurons)

        cnnModel = input(
            'Default CNN Model = ' + defaultModel + ' (Enter to accept, or ' + OP_MODEL_1DCNN + '/' + OP_MODEL_2DCNN + ' to change) : ')
        cnnModel = cnnModel.upper()
        if (cnnModel == ""):
            cnnModel = defaultModel

        if (cnnModel == OP_MODEL_1DCNN):
            modelDesc = _1DCNN_DESC
        elif (cnnModel == OP_MODEL_2DCNN):
            modelDesc = _2D_CNN_DESC
        elif (cnnModel == OP_MODEL_ALEXNET):

            modelDesc = _ALEXNET_DESC
        else:
            print("*** Unknown Choice ... exiting***")
            sys.exit()

        print("*** CNN Final Selection ***")
        print("Number Epochs = ", numberEpochs)
        print("Learning Rate = ", learningRate)
        print("Dropout Rate = ", dropoutRate)
        print("Number Neurons = ", numberNeurons)
        print("CNN Model = ", modelDesc)
        complete = input('Accept (Y/N) :')
        if (complete == ""):
            complete = 'Y'
        complete = complete.upper()
        if (complete == 'Y'):
            bCompleteInput = True
        elif (complete != 'N'):
            print('*** Invalid Input ***')

    return numberEpochs, learningRate, dropoutRate, numberNeurons, cnnModel


def GetEqualisationStrategy():
    bCorrectInput = False

    while (bCorrectInput == False):
        equaLise = input('Equalise All Data Samples (Y/N) :')
        equaLise = equaLise.upper()
        if (equaLise == 'Y') or (equaLise == ''):
            print("*** Equalising Data Samples ***")
            bEqualise = True
            bCorrectInput = True
        elif (equaLise == 'N'):
            print("*** NOT Equalising Data Samples ***")
            bEqualise = False
            bCorrectInput = True

    return bEqualise


def GetDataset():
    datasetChoice = ['NVSS', 'VAST']
    shortDataSetChoice = [NVSS_SHORT_NAME, VAST_SHORT_NAME]
    bCorrectInput = False

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


def GetSelectedMultiDataSets():
    classLabels = []
    bCorrectInput = False
    choiceList = ["AGN(A)", "SEYFERT(S)", "BLAZAR(B)", "QUASAR(Q)", "PULSAR (P)"]

    dataClass = GetDataset()

    while (bCorrectInput == False):
        numberClasses = int(input("Number of Classes : "))
        if (numberClasses < 2) or (numberClasses > len(choiceList)):
            print("*** Invalid Selection - Enter again... ***")
        else:
            print("*** Number Classes Chosen = " + str(numberClasses) + " ***")
            bCorrectInput = True

    for i in range(numberClasses):
        bCorrectInput = False
        while (bCorrectInput == False):
            strr = 'Select ' + choiceList[0] + ', ' + choiceList[1] + ', ' + choiceList[2] + ', ' + choiceList[
                3] + ', ' + choiceList[4] + ' : '
            classData = input(strr)
            classData = classData.upper()
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

    return dataClass, classLabels


def AutoTestMultiCNNModels(dataSet):
    # get params for all models

    print("*** Auto Testing All CNN Models ***")

    numberEpochs, learningRate, dropoutRate, numberNeurons, cnnModel = GetCNNParameters(OP_MODEL_1DCNN)

    completeModelSelections = [[DEFAULT_AGN_CLASS, DEFAULT_PULSAR_CLASS], [DEFAULT_AGN_CLASS, DEFAULT_BLAZAR_CLASS],
                               [DEFAULT_AGN_CLASS, DEFAULT_QUASAR_CLASS], [DEFAULT_AGN_CLASS, DEFAULT_SEYFERT_CLASS],
                               [DEFAULT_PULSAR_CLASS, DEFAULT_BLAZAR_CLASS],
                               [DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS],
                               [DEFAULT_PULSAR_CLASS, DEFAULT_SEYFERT_CLASS],
                               [DEFAULT_BLAZAR_CLASS, DEFAULT_QUASAR_CLASS],
                               [DEFAULT_BLAZAR_CLASS, DEFAULT_SEYFERT_CLASS],
                               [DEFAULT_QUASAR_CLASS, DEFAULT_SEYFERT_CLASS]]

    for label in range(len(completeModelSelections)):
        labelList = []

        labelList.append(completeModelSelections[label][0])
        labelList.append(completeModelSelections[label][1])

        print('*** Testing ' + labelList[0] + ' versus ' + labelList[1] + ' ***')

        completeTrainingData, trainingDataSizes = ProcessAllTransientModelData(dataSet, labelList, False)

        print("*** Creating Training and Test Data Sets ***")

        XTrain, XTest, ytrain, ytest, labelDict, scaler, labelList, encoder = CreateMultiTrainingAndTestData(
            cnnModel, labelList, completeTrainingData, trainingDataSizes)

        if (cnnModel == OP_MODEL_1DCNN):
            newModel = BuildandTest1DCNNModel(dataSet, labelList, XTrain, XTest, ytrain, ytest, numberEpochs,
                                              learningRate,
                                              dropoutRate, numberNeurons)

        else:

            newModel = BuildandTest2DCNNModel(dataSet, labelList, XTrain, XTest, ytrain, ytest, numberEpochs,
                                              learningRate, dropoutRate, numberNeurons)

        SaveCNNModel(cnnModel, dataSet, labelDict, newModel, scaler, labelList, encoder, numberEpochs)

        labelList.clear()
        completeTrainingData.clear()
        trainingDataSizes.clear()


def AutoTestBinaryCNNModels(dataSet):
    # get params for all models

    print("*** Auto Testing All Binary CNN Models ***")

    bEqualise = GetEqualisationStrategy()
    numberEpochs, learningRate, dropoutRate, numberNeurons, cnnModel = GetCNNParameters(OP_MODEL_1DCNN)

    primaryLabels = [DEFAULT_AGN_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS, DEFAULT_BLAZAR_CLASS,
                     DEFAULT_SEYFERT_CLASS]

    otherLabels = [[DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS],
                   [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_QUASAR_CLASS, DEFAULT_AGN_CLASS],
                   [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_AGN_CLASS],
                   [DEFAULT_SEYFERT_CLASS, DEFAULT_AGN_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS],
                   [DEFAULT_AGN_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS]]

    for label in range(len(primaryLabels)):

        print('*** Testing ' + primaryLabels[label] + ' versus OTHER ***')

        completeTrainingData, trainingDataSizes = ProcessBinaryModelData(dataSet, primaryLabels[label],
                                                                         otherLabels[label], bEqualise)

        print("*** Creating Training and Test Data Sets ***")

        XTrain, XTest, ytrain, ytest, labelDict, scaler, labelList, encoder = CreateTrainingAndTestData(
            cnnModel, primaryLabels[label], completeTrainingData, trainingDataSizes)

        if (cnnModel == OP_MODEL_1DCNN):
            newModel = BuildandTest1DCNNModel(dataSet, labelList, XTrain, XTest, ytrain, ytest, numberEpochs,
                                              learningRate,
                                              dropoutRate, numberNeurons)

        else:

            newModel = BuildandTest2DCNNModel(dataSet, labelList, XTrain, XTest, ytrain, ytest, numberEpochs,
                                              learningRate, dropoutRate, numberNeurons)

        SaveCNNModel(cnnModel, dataSet, labelDict, newModel, scaler, labelList, encoder, numberEpochs)

        labelList.clear()
        completeTrainingData.clear()
        trainingDataSizes.clear()


def AutoTestBinaryRFModels(dataSet):
    print("*** Auto Testing All Binary RF Models ***")

    f = OpenAutoResultsFile('AutoBinaryRF_Results.txt')

    bEqualise = GetEqualisationStrategy()

    primaryLabels = [DEFAULT_AGN_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS, DEFAULT_BLAZAR_CLASS,
                     DEFAULT_SEYFERT_CLASS]

    otherLabels = [[DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS],
                   [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_QUASAR_CLASS, DEFAULT_AGN_CLASS],
                   [DEFAULT_SEYFERT_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_AGN_CLASS],
                   [DEFAULT_SEYFERT_CLASS, DEFAULT_AGN_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS],
                   [DEFAULT_AGN_CLASS, DEFAULT_BLAZAR_CLASS, DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS]]

    for labelNo in range(len(primaryLabels)):
        print('*** Testing ' + primaryLabels[labelNo] + ' versus OTHER ***')

        completeTrainingData, trainingDataSizes = ProcessBinaryModelData(dataSet, primaryLabels[labelNo],
                                                                         otherLabels[labelNo], bEqualise)

        print("*** Creating Training and Test Data Sets ***")

        cnnModel = ""
        XTrain, XTest, ytrain, ytest, labelDict, scaler, labelList, encoder = CreateTrainingAndTestData(
            cnnModel, primaryLabels[labelNo], completeTrainingData, trainingDataSizes)

        newModel, highestAccuracy = RandomForestModel(labelDict, XTrain, ytrain, XTest, ytest)

        WriteAutoResultsFile(f, primaryLabels[labelNo], 'OTHER', MAX_NUMBER_SOAK_TESTS, highestAccuracy)

        SaveModel(dataSet, labelDict, newModel, scaler, labelList, encoder)

        labelList.clear()
        completeTrainingData.clear()
        trainingDataSizes.clear()

    f.close()


def OpenAutoResultsFile(filename):
    f = open(DEFAULT_AUTO_RF_TEST_LOCATION + filename, "w")

    f.write('\n\n')
    f.write('*** Auto Testing Random Forest Models Results ***')
    f.write('\n\n')

    return f


def WriteAutoResultsFile(f, testClass1, testClass2, totalNoSoakTests, bestAccuracy):
    f.write('\n')
    if not (bSoakTest):
        totalNoSoakTests = 0

    strr = 'For Test Class: ' + testClass1 + ' versus Test Class: ' + testClass2 + ', Total No Tests = ' + str(
        totalNoSoakTests) + ' , Best Accuracy = ' + str(bestAccuracy)
    print(strr)
    f.write(strr)
    f.write('\n\n')


def AutoTestMultiRFModels(dataSet):
    print("*** Auto Testing All RF Models ***")

    completeModelSelections = [[DEFAULT_AGN_CLASS, DEFAULT_SEYFERT_CLASS], [DEFAULT_AGN_CLASS, DEFAULT_PULSAR_CLASS],
                               [DEFAULT_AGN_CLASS, DEFAULT_BLAZAR_CLASS], [DEFAULT_AGN_CLASS, DEFAULT_QUASAR_CLASS],
                               [DEFAULT_PULSAR_CLASS, DEFAULT_BLAZAR_CLASS],
                               [DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS],
                               [DEFAULT_PULSAR_CLASS, DEFAULT_SEYFERT_CLASS],
                               [DEFAULT_BLAZAR_CLASS, DEFAULT_QUASAR_CLASS],
                               [DEFAULT_BLAZAR_CLASS, DEFAULT_SEYFERT_CLASS],
                               [DEFAULT_QUASAR_CLASS, DEFAULT_SEYFERT_CLASS]]

    # assume equalisation of all datasets

    f = OpenAutoResultsFile('AutoMultiRF_Results.txt')
    bEqualise = GetEqualisationStrategy()
    for label in range(len(completeModelSelections)):
        labelList = []

        labelList.append(completeModelSelections[label][0])
        labelList.append(completeModelSelections[label][1])

        print('*** Testing ' + labelList[0] + ' versus ' + labelList[1] + ' ***')

        completeTrainingData, trainingDataSizes = ProcessAllTransientModelData(dataSet, labelList, bEqualise)

        print("*** Creating Training and Test Data Sets ***")

        cnnModel = ""
        XTrain, XTest, ytrain, ytest, labelDict, scaler, labelList, encoder = CreateMultiTrainingAndTestData(
            cnnModel, labelList, completeTrainingData, trainingDataSizes)

        newModel, highestAccuracy = RandomForestModel(labelDict, XTrain, ytrain, XTest, ytest)

        WriteAutoResultsFile(f, completeModelSelections[label][0], completeModelSelections[label][1],
                             MAX_NUMBER_SOAK_TESTS, highestAccuracy)

        SaveModel(dataSet, labelDict, newModel, scaler, labelList, encoder)

        labelList.clear()
        completeTrainingData.clear()
        trainingDataSizes.clear()

    f.close()


def AutoTestMLPModels(dataSet):
    # get params for all models

    print("*** Auto Testing All MLP Models ***")

    completeModelSelections = [[DEFAULT_AGN_CLASS, DEFAULT_PULSAR_CLASS], [DEFAULT_AGN_CLASS, DEFAULT_BLAZAR_CLASS],
                               [DEFAULT_AGN_CLASS, DEFAULT_QUASAR_CLASS], [DEFAULT_AGN_CLASS, DEFAULT_SEYFERT_CLASS],
                               [DEFAULT_PULSAR_CLASS, DEFAULT_BLAZAR_CLASS],
                               [DEFAULT_PULSAR_CLASS, DEFAULT_QUASAR_CLASS],
                               [DEFAULT_PULSAR_CLASS, DEFAULT_SEYFERT_CLASS],
                               [DEFAULT_BLAZAR_CLASS, DEFAULT_QUASAR_CLASS],
                               [DEFAULT_BLAZAR_CLASS, DEFAULT_SEYFERT_CLASS],
                               [DEFAULT_QUASAR_CLASS, DEFAULT_SEYFERT_CLASS]]

    # assume equalisation of all datasets

    for label in range(len(completeModelSelections)):
        labelList = []

        labelList.append(completeModelSelections[label][0])
        labelList.append(completeModelSelections[label][1])

        print('*** Testing ' + labelList[0] + ' versus ' + labelList[1] + ' ***')

        completeTrainingData, trainingDataSizes = ProcessAllTransientModelData(dataSet, labelList, True)

        print("*** Creating Training and Test Data Sets ***")
        cnnModel = ""
        XTrain, XTest, ytrain, ytest, labelDict, scaler, labelList, encoder = CreateMultiTrainingAndTestData(
            cnnModel, labelList, completeTrainingData, trainingDataSizes)

        newModel = BuildandTestMLPModel(XTrain, XTest, ytrain, ytest)

        SaveCNNModel(OP_MODEL_MLP, dataSet, labelDict, newModel, scaler, labelList, encoder, 0)

        labelList.clear()
        completeTrainingData.clear()
        trainingDataSizes.clear()


def BuildClassList(binaryLabelDictList):
    classList = []

    for dictNo in range(len(binaryLabelDictList)):

        possibleClasses = list(binaryLabelDictList[dictNo].keys())
        for classNo in range(len(possibleClasses)):
            if (possibleClasses[classNo] != DEFAULT_OTHER_CLASS):
                classList.append(possibleClasses[classNo])

    return classList


def BlindTest():
    dataSet = SelectDataset()
    if (dataSet == DEFAULT_VAST_DATASET):
        testBinaryLocation = DEFAULT_VAST_DATA_ROOT + DEFAULT_RF_BINARY_MODEL_ANALYSIS
        testMultiLocation = DEFAULT_VAST_DATA_ROOT + DEFAULT_RF_MULTI_MODEL_ANALYSIS

    elif (dataSet == DEFAULT_NVSS_DATASET):
        print("*** NVSS Dataset Not Supported Yet***")
        sys.exit()
    else:
        print("*** UNKNOWN Dataset ***")
        sys.exit()

    binaryNameToModelDict, binaryModelList, binaryLabelDictList, binaryScalerList, binaryEncoderList = LoadModels(
        dataSet, testBinaryLocation, 'BINARY MODELS')

    multiNameToModelDict, multiModelList, multiLabelDictList, multiScalerList, multiEncoderList = LoadModels(dataSet,
                                                                                                             testMultiLocation,
                                                                                                             'MULTI-CLASS MODELS')

    classList = BuildClassList(binaryLabelDictList)

    FullTestAllSamples(dataSet, classList, binaryNameToModelDict, binaryModelList, binaryLabelDictList,
                       binaryScalerList,
                       binaryEncoderList, multiNameToModelDict, multiModelList, multiLabelDictList, multiScalerList,
                       multiEncoderList)

    sys.exit()


def main():

    if (bCreateTransientCVSFiles):
        CreateTransientCSVFiles(dataSet)
        sys.exit()
    else:

        selectedOperation = GetOperationMode()

        if (selectedOperation == OP_STACK_IMAGES):

            print("*** Stacking Images ***")
            stackedLabelData, stackedImageData = StackAllImages(dataset)
            DisplayAllStackedImages(stackedImageData, stackedLabelData)
            sys.exit()
        else:

            print("*** Unknown Operation...exiting ***")
            sys.exit()


if __name__ == '__main__':
    main()
