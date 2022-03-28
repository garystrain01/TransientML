import os
import sys
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
#import tensorflow as tf
#import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from astropy.io import fits
from sklearn.model_selection import StratifiedShuffleSplit




bDebug = False # to activate debug
bShrinkImages = False # to shrink standard FITS images
bSoakModelTest = True # soak test models

# for poor quality image processing
bTestForETAV = False # use eta and v
bTestEpochPairs = True # use epoch pairs to determine criticality

DEFAULT_IMAGE_COEFFICIENT = 2.0
DEFAULT_MAX_COEFFICIENT = 1.7
DEFAULT_ARTEFACT_BOUNDARY = 5.0
DEFAULT_PQ_LABEL_NAME = 'POOR' # 1
DEFAULT_NORMAL_LABEL_NAME = 'NORMAL' # 0

DEFAULT_PQ_LABEL = 1
DEFAULT_NORMAL_LABEL = 0

lightCurveSourceList = []
binaryClassifierModelList = []

lightCurveStokesData = {}
lightCurveRAData = {}
lightCurveDECData= {}
lightCurveInitETAData= {}
lightCurveInitVData= {}
lightCurveFluxData= {}
lightCurveRMSData= {}
lightCurveDetectionData= {}
lightCurveCoefficientData= {}
lightCurveCorrelationData= {}
lightCurveRevalEtaData= {}
lightCurveRevalVData= {}
lightCurveFITSImages= {}
lightCurveWiseDataDict= {}
lightCurveSimbadDataDict ={}
lightCurvePulsarDataDict = {}
lightCurveSDSSDataDict = {}
lightCurveQuasarDataDict = {}
lightCurveFIRSTDataDict = {}
lightCurveNVSSDataDict = {}
lightCurvePossibleClasses= {}
lightCurvePossibleRationale= {}
lightCurveSimbadClass = {}
lightCurvePoorQualityImages= {}
lightCurvePoorQualityFlags= {}
lightCurveCriticalFlag= {}
summaryObjectClasses= {}
lightCurveManualClassification= {}
lightCurveFractionalNegativeData= {}
coefficientThreshold = DEFAULT_IMAGE_COEFFICIENT
bBinaryModelCreated=False



bStoreUnmatchedOnly = True
bUseBinaryClassifierModel=True # use model rather than coefficient

DEFAULT_NUMBER_TEST_IMAGES = 10

DEFAULT_FORCED_POINT = 'F'
DEFAULT_TRUE_DETECTION = 'D'
DEFAULT_CRITICAL_DETECTION = 'C'
DEFAULT_NON_CRITICAL_DETECTION = 'N'

DEFAULT_CRITICAL_DETECTION_TEXT = 'CRITICAL'
DEFAULT_NON_CRITICAL_DETECTION_TEXT = 'NON-CRITICAL'

RF_BINARY_CLASSIFIER = 'RF_BINARY_CLASSIFIER'

DEFAULT_OK_IMAGE = -1
# For WISE Color-Color Plot

DEFAULT_WISE_RADIUS = 5 # arcsec
DEFAULT_SIMBAD_RADIUS = 5 # arcsec
DEFAULT_STANDARD_RADIUS = 5 # arcsec
DEFAULT_FIRST_RADIUS = 5 # arcsec
DEFAULT_NVSS_RADIUS = 5 # arcsec
DEFAULT_PULSAR_RADIUS = 10 # arcsec

# MILLIQUAS LEGEND

milliQuasLegend = ['Q','A','B','K','N','2']

# SDSS CLASSIFICATION CODES

DEFAULT_SDSS_GALAXY_CODE = 3
DEFAULT_SDSS_STAR_CODE = 6
DEFAULT_SDSS_UNKNOWN_CODE = 0
DEFAULT_SDSS_GHOST_CODE = 4
DEFAULT_SDSS_KNOWN_CODE = 5

# MILLIQUAS CLASSIFICATION CODES

DEFAULT_QUASAR_QSO_CODE = 'Q'
DEFAULT_QUASAR_AGN_CODE = 'A'
DEFAULT_QUASAR_BL_CODE = 'B'

# FIRST CLASSIFICATION CODE

DEFAULT_FIRST_GALAXY_CODE = 'G'
DEFAULT_FIRST_STAR_CODE = 'S'

# CIRCLES

DEFAULT_STAR_CENTRE_X = 0.6
DEFAULT_STAR_CENTRE_Y = 0.25
DEFAULT_STAR_RADIUS = 0.80

DEFAULT_SEYFERT_CENTRE_X = 3.5
DEFAULT_SEYFERT_CENTRE_Y = 1.2
DEFAULT_SEYFERT_RADIUS = 1.15

DEFAULT_ELLIPTICAL_CENTRE_X = 0.9
DEFAULT_ELLIPTICAL_CENTRE_Y = 0.1
DEFAULT_ELLIPTICAL_RADIUS= 0.50

# ELLIPSES - ANGLES IN DEGREES (ROTATED COUNTER CLOCKWISE)

DEFAULT_SPIRAL_CENTRE_X = 2.80
DEFAULT_SPIRAL_CENTRE_Y = 0.25
DEFAULT_SPIRAL_MAJOR = 4.0
DEFAULT_SPIRAL_MINOR =  1.0
DEFAULT_SPIRAL_ANGLE =  260

DEFAULT_QSO_CENTRE_X = 3.0
DEFAULT_QSO_CENTRE_Y = 1.4
DEFAULT_QSO_MAJOR = 2.0
DEFAULT_QSO_MINOR =  1.0
DEFAULT_QSO_ANGLE =  200

DEFAULT_ULIRGS_CENTRE_X = 4.8
DEFAULT_ULIRGS_CENTRE_Y = 1.6
DEFAULT_ULIRGS_MAJOR = 4.0
DEFAULT_ULIRGS_MINOR = 1.0
DEFAULT_ULIRGS_ANGLE = 160

DEFAULT_LIRGS_CENTRE_X = 4.0
DEFAULT_LIRGS_CENTRE_Y = 0.8
DEFAULT_LIRGS_MAJOR = 4.0
DEFAULT_LIRGS_MINOR = 3.0
DEFAULT_LIRGS_ANGLE = 160

DEFAULT_OAGN_CENTRE_X = 1.75
DEFAULT_OAGN_CENTRE_Y = 4.7
DEFAULT_OAGN_MAJOR = 1.5
DEFAULT_OAGN_MINOR = 0.5
DEFAULT_OAGN_ANGLE = 110


# VIZIER CATALOGS

DEFAULT_WISE_CATALOG_NAME = "II/328"
DEFAULT_PULSAR_CATALOG_NAME = "B/psr"
DEFAULT_NVSS_CATALOG_NAME = "VIII/65/nvss"
DEFAULT_FIRST_CATALOG_NAME = "VIII/92/first14"
DEFAULT_MILLIQUAS_CATALOG = "VII/290"
DEFAULT_SDSS_CATALOG_NAME ="V/154"






DEFAULT_UNISYDNEY_DATA = '/Volumes/ExtraDisk/UNISYDNEY_RESEARCH/ClassificationData/'
DEFAULT_UNISYDNEY_ANALYSIS_DATA = '/Volumes/ExtraDisk/UNISYDNEY_RESEARCH/STATISTICAL_DATA/'
DEFAULT_UNISYDNEY_NOTEBOOK_DATA = '/Volumes/ExtraDisk/UNISYDNEY_RESEARCH/NOTEBOOK/'

DEFAULT_FITS_REFERENCE_FILE = DEFAULT_UNISYDNEY_NOTEBOOK_DATA+'TextFiles/classifyFITSFiles.txt'
DEFAULT_STOKES_REFERENCE_FILE = DEFAULT_UNISYDNEY_NOTEBOOK_DATA+'TextFiles/CompleteStokesData.txt'
#DEFAULT_MANUAL_CLASSIFIED_FILE = DEFAULT_UNISYDNEY_NOTEBOOK_DATA+'TextFiles/ManualResults.csv'
DEFAULT_MANUAL_CLASSIFIED_FILE = DEFAULT_UNISYDNEY_NOTEBOOK_DATA+'TextFiles/CompleteManualResults.txt'
#DEFAULT_MANUAL_CLASSIFIED_FILE = DEFAULT_UNISYDNEY_NOTEBOOK_DATA+'TextFiles/TestManualResults.txt'
DEFAULT_FITS_SRC_FOLDER = DEFAULT_UNISYDNEY_NOTEBOOK_DATA+'FITSFiles/FITSFILES3/FITSImages/'
DEFAULT_FITS_IMAGE_FOLDER = DEFAULT_UNISYDNEY_NOTEBOOK_DATA+'FITSImages/'
#DEFAULT_ALL_LC_MEASUREMENTS_FILE = DEFAULT_UNISYDNEY_NOTEBOOK_DATA+'TextFiles/TestClassifyMeasurements.txt'
DEFAULT_ALL_LC_MEASUREMENTS_FILE = DEFAULT_UNISYDNEY_NOTEBOOK_DATA+'TextFiles/CompleteClassifyMeasurements.txt'
DEFAULT_BINARY_DATA_FOLDER = DEFAULT_UNISYDNEY_NOTEBOOK_DATA+'BinaryData/'
DEFAULT_BINARY_MODEL_FOLDER = DEFAULT_UNISYDNEY_NOTEBOOK_DATA+'Models/'
DEFAULT_BINARY_MODEL_NAME = 'RF_CLASSIFIER'
DEFAULT_RESULTS_FOLDER = DEFAULT_UNISYDNEY_NOTEBOOK_DATA+'Results/'
DEFAULT_CATALOG_RESULTS_FOLDER = DEFAULT_UNISYDNEY_NOTEBOOK_DATA+'Results/Catalogs/'
DEFAULT_PULSAR_RESULTS_FILE  = DEFAULT_CATALOG_RESULTS_FOLDER+'PulsarResults'
DEFAULT_SDSS_RESULTS_FILE = DEFAULT_CATALOG_RESULTS_FOLDER+'SDSSResults'
DEFAULT_MILLIQUAS_RESULTS_FILE = DEFAULT_CATALOG_RESULTS_FOLDER+'MilliquasResults'
DEFAULT_FIRST_RESULTS_FILE = DEFAULT_CATALOG_RESULTS_FOLDER+'FIRSTResults'
DEFAULT_RESULTS_FILE = DEFAULT_RESULTS_FOLDER+'ClassificationResults'
DEFAULT_MISMATCHED_FILE = DEFAULT_RESULTS_FOLDER+'MismatchSummary'
DEFAULT_COMPARE_RESULTS_FILE = DEFAULT_RESULTS_FOLDER+'CompareResults'
DEFAULT_PQ_RESULTS_FILE = DEFAULT_RESULTS_FOLDER+'PoorQuality'
DEFAULT_TRANSIENTS_DROPPED_FILE = DEFAULT_RESULTS_FOLDER+'TransientsDropped'
DEFAULT_POOR_QUALITY_SUMMARY_FILE = DEFAULT_RESULTS_FOLDER+'ListPoorQualityImages'
DEFAULT_NORMAL_IMAGE_FILE =DEFAULT_RESULTS_FOLDER+'ListGoodImages'

DEFAULT_HISTOGRAM_COLLECTION = DEFAULT_RESULTS_FOLDER +'HistCollection/'
DEFAULT_IMAGES_COLLECTION = DEFAULT_RESULTS_FOLDER +'ImagesCollection/'
DEFAULT_IMAGE_IDENTIFIER = '_IMAGES.PNG'
DEFAULT_HIST_IDENTIFIER = '_HIST.PNG'
DEFAULT_TXT_FILENAME = '.txt'

DEFAULT_IMAGE_DATA = DEFAULT_UNISYDNEY_DATA +'IMAGE_DATA/'
DEFAULT_LIGHTCURVE_DATA = DEFAULT_UNISYDNEY_DATA +'LIGHT_CURVE_DATA/'

DEFAULT_ANALYSIS_DATA = DEFAULT_UNISYDNEY_ANALYSIS_DATA +'IMAGE_DATA/'

# Transient Classes
DEFAULT_NOT_SPECIFIED_CLASS = "UNCLASSIFIED"
DEFAULT_UNKNOWN_CLASS = "UNKNOWN"
DEFAULT_GALAXY_CLASS = "GALAXY"
DEFAULT_ARTEFACT_CLASS = "ARTEFACT"
DEFAULT_POOR_QUALITY_CLASS = "POOR QUALITY"
DEFAULT_STAR_CLASS = "STAR"
DEFAULT_ELLIPTICAL_CLASS ="ELLIPTICAL"
DEFAULT_QSO_CLASS = "QSO"
DEFAULT_SEYFERT_CLASS = "SEYFERT"
DEFAULT_AGN_CLASS = "AGN"
DEFAULT_BL_CLASS = "BL"
DEFAULT_SPIRAL_CLASS = "SPIRAL"
DEFAULT_ULIRGS_CLASS = "ULIRGS"
DEFAULT_LIRGS_CLASS = "LIRGS"
DEFAULT_OAGN_CLASS = "OBSCURED_AGN"
DEFAULT_PLANET_CLASS = "PLANET"
DEFAULT_PULSAR_CLASS = "PULSAR"

# cutdown labels for confusion matrix

DEFAULT_NOT_SPECIFIED_CLASS_LBL = "NC"
DEFAULT_UNKNOWN_CLASS_LBL = "UN"
DEFAULT_GALAXY_CLASS_LBL = "GA"
DEFAULT_ARTEFACT_CLASS_LBL = "AR"
DEFAULT_POOR_QUALITY_CLASS_LBL = "PQ"
DEFAULT_STAR_CLASS_LBL = "ST"
DEFAULT_ELLIPTICAL_CLASS_LBL ="EL"
DEFAULT_QSO_CLASS_LBL = "QS"
DEFAULT_SEYFERT_CLASS_LBL = "SY"
DEFAULT_AGN_CLASS_LBL = "AG"
DEFAULT_BL_CLASS_LBL = "BL"
DEFAULT_SPIRAL_CLASS_LBL = "SP"
DEFAULT_ULIRGS_CLASS_LBL = "UL"
DEFAULT_LIRGS_CLASS_LBL = "LI"
DEFAULT_OAGN_CLASS_LBL = "OA"
DEFAULT_PLANET_CLASS_LBL = "PL"
DEFAULT_PULSAR_CLASS_LBL = "PU"

setOfPossibleClasses = [DEFAULT_NOT_SPECIFIED_CLASS,
DEFAULT_UNKNOWN_CLASS,
DEFAULT_GALAXY_CLASS,
DEFAULT_ARTEFACT_CLASS,
DEFAULT_POOR_QUALITY_CLASS,
DEFAULT_STAR_CLASS,
DEFAULT_ELLIPTICAL_CLASS,
DEFAULT_QSO_CLASS,
DEFAULT_SEYFERT_CLASS,
DEFAULT_AGN_CLASS,
DEFAULT_BL_CLASS,
DEFAULT_SPIRAL_CLASS,
DEFAULT_ULIRGS_CLASS,
DEFAULT_LIRGS_CLASS,
DEFAULT_OAGN_CLASS,
DEFAULT_PLANET_CLASS,
DEFAULT_PULSAR_CLASS]

setOfPossibleClassLabels = [DEFAULT_NOT_SPECIFIED_CLASS_LBL,
DEFAULT_UNKNOWN_CLASS_LBL,
DEFAULT_GALAXY_CLASS_LBL,
DEFAULT_ARTEFACT_CLASS_LBL,
DEFAULT_POOR_QUALITY_CLASS_LBL,
DEFAULT_STAR_CLASS_LBL,
DEFAULT_ELLIPTICAL_CLASS_LBL,
DEFAULT_QSO_CLASS_LBL,
DEFAULT_SEYFERT_CLASS_LBL,
DEFAULT_AGN_CLASS_LBL,
DEFAULT_BL_CLASS_LBL,
DEFAULT_SPIRAL_CLASS_LBL,
DEFAULT_ULIRGS_CLASS_LBL,
DEFAULT_LIRGS_CLASS_LBL,
DEFAULT_OAGN_CLASS_LBL,
DEFAULT_PLANET_CLASS_LBL,
DEFAULT_PULSAR_CLASS_LBL]



DEFAULT_NUMBER_HIST_BINS = 100

DEFAULT_ARTEFACT_DATA = 'A'
DEFAULT_AGN_DATA = 'G'
DEFAULT_POOR_QUALITY_DATA = 'P'
DEFAULT_RANDOM_DATA = 'R'

DEFAULT_ARTEFACT_DATA_NAME = 'ARTEFACT'
DEFAULT_AGN_DATA_NAME = 'AGN'
DEFAULT_POOR_QUALITY_DATA_NAME = 'POOR QUALITY'
DEFAULT_RANDOM_DATA_NAME = 'RANDOM'

DEFAULT_HEADER_LENGTH_CSV = 3 # to allow for source ID, RA and DEC
DEFAULT_HEADER_LENGTH_CSV_FULL_LC = 5 # to allow for source ID, RA and DEC, ETA and V
DEFAULT_ENTRY_LENGTH_CSV_FULL_LC = 6 # to allow for flux,rms,f/d, recalc ETA and V, Critical Flag

DEFAULT_CLASSIFY_DATA = 'C'
DEFAULT_STATISTICAL_DATA = 'S'


DEFAULT_ARTEFACT_DATA_ROOT = 'ARTEFACTS/'
DEFAULT_AGN_DATA_ROOT = 'AGN/'
DEFAULT_POOR_QUALITY_DATA_ROOT = 'POOR_QUALITY/'
DEFAULT_RANDOM_DATA_ROOT = 'RANDOM/'

DEFAULT_POOR_QUALITY_IMAGE = 'P'
DEFAULT_NORMAL_IMAGE = 'N'


#imageClassificationLabels = [DEFAULT_ARTEFACT_DATA,DEFAULT_AGN_DATA,DEFAULT_POOR_QUALITY_DATA]
imageClassificationLabels = [DEFAULT_POOR_QUALITY_DATA,DEFAULT_AGN_DATA]
imageClassificationRoots = [DEFAULT_AGN_DATA_ROOT,DEFAULT_POOR_QUALITY_DATA_ROOT]
LCClassificationLabels = [DEFAULT_AGN_DATA,DEFAULT_RANDOM_DATA]

qualityClassificationLabels = [DEFAULT_POOR_QUALITY_IMAGE,DEFAULT_NORMAL_IMAGE]



#OP_STACK_IMAGES = '1'
OP_BUILD_IMAGE_MODELS = '1'
#OP_BUILD_LC_MODELS = '3'
OP_ANALYSE_IMAGES = '2'
OP_PROCESS_FITS_FILES = '3'
OP_BUILD_CLASSIFY_DATA = '4'
OP_INTERROGATE_DATA = '5'
#OP_FIND_IMAGES = '8'
OP_STORE_BINARY_DATA = '6'
OP_LOAD_BINARY_DATA = '7'
OP_ANALYSE_MAN_CLASSES = '8'
OP_RECLASSIFY = '9'
OP_TEST_MODEL = '10'
OP_SOAK_TEST = '11'
OP_MISMATCH = '12'
OP_DETERMINE_OBJECT = '13'
OP_FIND_IMAGES = '14'
OP_COMPARE_RESULTS = '15'
OP_EXIT = '16'

DEFAULT_STATUS_MATCHED = 'MATCHED'
DEFAULT_STATUS_NOT_MATCHED = 'NOT MATCHED'


SMALL_FONT_SIZE = 8
MEDIUM_FONT_SIZE = 10
BIGGER_FONT_SIZE = 12

DEFAULT_LC_MEASUREMENTS_FILE = 'measurements.txt'

OP_MODEL_1DCNN = '1D_CNN'
OP_MODEL_2DCNN = '2D_CNN'
NO_CNN = 'NO_CNN'


SIMULATED_PERCENTAGE_DEVIATION = 5
DEFAULT_NUMBER_SIMULATIONS = 50
DEFAULT_PERIOD_SYM = '.'
DEFAULT_SIM_FILE = '_sim'
TRAIN_TEST_RATIO = 0.80
FAILED_TO_OPEN_FILE = -1
FITS_FILE_EXTENSION = '.fits'
DEFAULT_PKL_EXTENSION = '.pkl'
DS_STORE_FILENAME = '.'
DEFAULT_CSV_DIR = 'CSVFiles'
FOLDER_IDENTIFIER = '/'
SOURCE_TITLE_TEXT = ' Source :'
UNDERSCORE = '_'
DEFAULT_CSV_FILETYPE = UNDERSCORE + 'data.txt'
XSIZE_FITS_IMAGE = 120
YSIZE_FITS_IMAGE = 120

def TestIndividualImage(model,imageData,dataScaler):

    dataAsArray = np.asarray(imageData)

    dataAsArray = dataAsArray.reshape(1, -1)
    scaledData = dataScaler.transform(dataAsArray)

    y_pred = model.predict(scaledData)

 #   print("y_pred =", y_pred)

    y_pred_proba = model.predict_proba(scaledData)

   # print("y_pred_proba =", y_pred_proba)

    return y_pred[0]




def evaluate1DCNNModel(dataset,labelList,Xtrain, ytrain, Xtest, ytest, n_timesteps, n_features, n_outputs,numberEpochs, learningRate, dropoutRate, numberNeurons):

    from keras.callbacks import EarlyStopping
    import statistics

    verbose, batchSize = DEFAULT_VERBOSE_LEVEL, DEFAULT_BATCH_SIZE

    print("number epochs = ",numberEpochs)
    print("learning rate = ",learningRate)
    print("Dropout Rate = ",dropoutRate)
    print("number neurons = ",numberNeurons)
    print("timesteps = ",n_timesteps)
    print("no features = ",n_features)
    print("no outputs = ",n_outputs)
    print("labellist = ",labelList)



    model = keras.Sequential()

    model.add(tf.keras.layers.Conv1D(filters=DEFAULT_NUMBER_FILTERS, kernel_size=DEFAULT_KERNEL_SIZE, activation='relu', input_shape=(n_features,n_timesteps)))
    model.add(tf.keras.layers.Conv1D(filters=DEFAULT_NUMBER_FILTERS, kernel_size=DEFAULT_KERNEL_SIZE, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=DEFAULT_KERNEL_SIZE))

    model.add(tf.keras.layers.Dropout(dropoutRate))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(numberNeurons, activation='relu'))

    if (n_outputs==1):
        model.add(tf.keras.layers.Dense(n_outputs, activation='sigmoid'))
    else:
        model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=learningRate)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=6)

    history = model.fit(Xtrain, ytrain, epochs=numberEpochs, batch_size=batchSize, verbose=1,validation_split=0.2,callbacks=[es])

    model.summary()

    DEFAULT_NUMBER_CNN_TESTS = 1

    cnnAccuracy = []

    for testNo in range(DEFAULT_NUMBER_CNN_TESTS):

        accuracy = model.evaluate(Xtest, ytest, batch_size=batchSize, verbose=verbose)
        cnnAccuracy.append(accuracy)

        index = np.random.choice(Xtest.shape[0], len(Xtest), replace=False)

        Xtest = Xtest[index]
        ytest = ytest[index]


    print("CNN Accuracies = ",cnnAccuracy)
    print("CNN Best Accuracy = ", max(cnnAccuracy))
    print("CNN Worst Accuracy = ", min(cnnAccuracy))


#    SaveCNNModelAnalysis(OP_MODEL_1DCNN,dataset, history, labelList, numberEpochs, accuracy[1])

    print("*** TEST RESULTS ***")
    print("Test Loss = ",accuracy[0])
    print("Test Accuracy = ", accuracy[1])

    return accuracy[1], model




def DisplayFITSImage(imageData, figHeader):
    plt.title(figHeader)
    plt.imshow(imageData, cmap='gray')
    plt.colorbar()

    plt.show()

def DisplayAllFITSImages(sourceID,fitsImages):

    global coefficientThreshold
    numberPlots = len(fitsImages)
    numberRows = int(numberPlots/2)

    excess = int(numberPlots- (numberRows*2))

    f, axarr = plt.subplots(numberRows+excess, 2, figsize=(15, 15))

    strr = "SOURCE ID : "+str(sourceID)+" RA = "+lightCurveRAData[sourceID]+" DEC = "+lightCurveDECData[sourceID]
#    strr = strr+'\n Initial ETA: '+str(lightCurveInitETAData[sourceID])+' , Initial V : '+str(lightCurveInitVData[sourceID])
    if (lightCurveWiseDataDict[sourceID] ==0):
        strr = strr+" \n NO WISE SRC DETECTED"
    else:
        wiseData = lightCurveWiseDataDict[sourceID]
    #    print(wiseData)
        strr = strr+"\n  WISE SRC: "+str(wiseData[0])+','+str(wiseData[1])+','+str(wiseData[2])+','+str(wiseData[3])+','+str(wiseData[4])+','+str(wiseData[5])

    possibleClasses = lightCurvePossibleClasses[sourceID]
    strr = strr+"\n POSSIBLE CLASS: "
    strr = strr+str(possibleClasses)

    if (lightCurveStokesData[sourceID] !=0):
        strr = strr + "      STOKES : "+lightCurveStokesData[sourceID]

    f.suptitle(strr)

    detectionData = lightCurveDetectionData[sourceID]
    poorQualityImages = lightCurvePoorQualityImages[sourceID]
    poorQualityFlags = lightCurvePoorQualityFlags[sourceID]
    coefficients = lightCurveCoefficientData[sourceID]
    correlations = lightCurveCorrelationData[sourceID]
    revisedETA = lightCurveRevalEtaData[sourceID]
    revisedV = lightCurveRevalVData[sourceID]
    criticalFlag = lightCurveCriticalFlag[sourceID]
    fractionalNegatives = lightCurveFractionalNegativeData[sourceID]

    lcPoint = 0
    for rowNo in range(numberRows):
        for colNo in range(2):

            axarr[rowNo, colNo].imshow(fitsImages[lcPoint])
            axarr[rowNo,colNo].axis('off')
            axarr[rowNo, colNo].text(120, 30, str(sourceID) + '-' + str(lcPoint))
            if (detectionData[lcPoint]==DEFAULT_FORCED_POINT):
                axarr[rowNo, colNo].text(120, 50, 'FORCED')
                if (criticalFlag[lcPoint] == DEFAULT_CRITICAL_DETECTION):
                    strr = DEFAULT_CRITICAL_DETECTION_TEXT
                else:
                    strr = DEFAULT_NON_CRITICAL_DETECTION_TEXT

                axarr[rowNo, colNo].text(120, 70, strr)
            axarr[rowNo, colNo].text(120, 90,'Coefficient: ' + str(coefficients[lcPoint]))

            if (poorQualityFlags[lcPoint] == True):
                axarr[rowNo, colNo].text(120, 110, 'POOR QUALITY '+str(coefficients[lcPoint]))

            lcPoint+=1

    # now do excess
    if (excess >0):
        rowNo+=1
        colNo = 0
        for excessNo in range(excess):

            axarr[rowNo, colNo].imshow(fitsImages[lcPoint])
            axarr[rowNo, colNo].axis('off')
            axarr[rowNo, colNo].text(120, 30, str(sourceID) + '-' + str(lcPoint))

            if (detectionData[lcPoint] == DEFAULT_FORCED_POINT):
                axarr[rowNo, colNo].text(120, 50, 'FORCED')
                if (criticalFlag[lcPoint] == DEFAULT_CRITICAL_DETECTION):
                    strr = DEFAULT_CRITICAL_DETECTION_TEXT
                else:
                    strr = DEFAULT_NON_CRITICAL_DETECTION_TEXT
                axarr[rowNo, colNo].text(120, 70, strr)

            axarr[rowNo, colNo].text(120, 90, 'Coefficient: ' + str(coefficients[lcPoint]))

            if (poorQualityFlags[lcPoint] == True):
                axarr[rowNo, colNo].text(120, 110, 'POOR QUALITY ' + str(coefficients[lcPoint]))



            colNo +=1
            lcPoint +=1

        # delete the extra subplot
        f.delaxes(axarr[rowNo,1])


    plt.show()

    f.savefig(DEFAULT_IMAGES_COLLECTION+str(sourceID)+DEFAULT_IMAGE_IDENTIFIER)


def DisplayAllFITSHistograms(sourceID,fitsImages):

    numberPlots = len(fitsImages)
    numberRows = int(numberPlots/2)

    excess = int(numberPlots- (numberRows*2))

    f, axarr = plt.subplots(numberRows+excess, 2, figsize=(15, 15))

    strr = "SOURCE ID : "+str(sourceID)+" RA = "+lightCurveRAData[sourceID]+" DEC = "+lightCurveDECData[sourceID]
    strr = strr+'\n Initial ETA: '+str(lightCurveInitETAData[sourceID])+' , Initial V : '+str(lightCurveInitVData[sourceID])
    if (lightCurveWiseDataDict[sourceID] ==0):
        strr = strr+" \n NO WISE SRC DETECTED"
    else:
        wiseData = lightCurveWiseDataDict[sourceID]
        strr = strr+"\n  WISE SRC: "+wiseData[0]
    if (lightCurvePossibleClasses[sourceID] != 0):
        possibleClasses = lightCurvePossibleClasses[sourceID]
        numberPossibleClasses =len(possibleClasses)
    else:
        numberPossibleClasses = 0

    if (numberPossibleClasses>0):
        strr = strr+"\n POSSIBLE CLASSES: "
        for objectNo in range(numberPossibleClasses):

            strr = strr+possibleClasses[objectNo]

            if (objectNo+1 < numberPossibleClasses):
                strr = strr+','
    else:
        strr = strr+"\n UNKNOWN CLASS"
    if (lightCurveStokesData[sourceID] != 0):
        strr = strr +"      STOKES : "+lightCurveStokesData[sourceID]


    f.suptitle(strr)

    detectionData = lightCurveDetectionData[sourceID]
    poorQualityFlags = lightCurvePoorQualityFlags[sourceID]
    coefficients = lightCurveCoefficientData[sourceID]
    revisedETA = lightCurveRevalEtaData[sourceID]
    revisedV = lightCurveRevalVData[sourceID]
    criticalFlag = lightCurveCriticalFlag[sourceID]
    fractionalNegatives = lightCurveFractionalNegativeData[sourceID]

    lcPoint = 0
    for rowNo in range(numberRows):
        for colNo in range(2):

            imageData = fitsImages[imageNo]
            imageData = np.reshape(imageData, [1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE])
            image = imageData[0]
            axarr[rowNo, colNo].hist(image,bins=DEFAULT_NUMBER_HIST_BINS)
            axarr[rowNo, colNo].text(0.1, 0.9, 'Image No: ' + str(lcPoint), horizontalalignment='center',
                                     verticalalignment='center', transform=axarr[rowNo, colNo].transAxes)
            if (detectionData[lcPoint]==DEFAULT_FORCED_POINT):
                axarr[rowNo, colNo].text(0.85,0.9, 'FORCED',horizontalalignment='center',
                                         verticalalignment='center',transform=axarr[rowNo, colNo].transAxes)
                if (criticalFlag[lcPoint] == DEFAULT_CRITICAL_DETECTION):
                    strr = DEFAULT_CRITICAL_DETECTION_TEXT
                else:
                    strr = DEFAULT_NON_CRITICAL_DETECTION_TEXT
                axarr[rowNo, colNo].text(0.85, 0.75, strr, horizontalalignment='center',
                                         verticalalignment='center', transform=axarr[rowNo, colNo].transAxes)

            if (poorQualityFlags[lcPoint]):

                axarr[rowNo, colNo].text(0.85, 0.6, 'POOR QUALITY '+str(coefficients[lcPoint]), horizontalalignment='center',
                                             verticalalignment='center', transform=axarr[rowNo, colNo].transAxes)
            axarr[rowNo, colNo].text(0.85, 0.45, 'FRAC NEG FLUX ' + str(round(fractionalNegatives[lcPoint],2)),
                                         horizontalalignment='center',
                                         verticalalignment='center', transform=axarr[rowNo, colNo].transAxes)


            bNormal,bShapiro = IsNormaltest(image)

            axarr[rowNo, colNo].text(0.85, 0.30, 'NORMAL = '+str(bNormal),
                                         horizontalalignment='center',
                                         verticalalignment='center', transform=axarr[rowNo, colNo].transAxes)

            axarr[rowNo, colNo].text(0.85, 0.15, 'SHAPIRO = ' + str(bShapiro),
                                     horizontalalignment='center',
                                     verticalalignment='center', transform=axarr[rowNo, colNo].transAxes)

            lcPoint+=1

    # now do excess
    if (excess >0):
        rowNo+=1
        colNo = 0
        imageData = fitsImages[imageNo]
        imageData = np.reshape(imageData, [1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE])
        image = imageData[0]
        for excessNo in range(excess):
            axarr[rowNo, colNo].hist(image, bins=DEFAULT_NUMBER_HIST_BINS)
            axarr[rowNo, colNo].text(0.1, 0.9, 'Image No: ' + str(imageNo), horizontalalignment='center',
                                     verticalalignment='center', transform=axarr[rowNo, colNo].transAxes)

            if (detectionData[lcPoint]==DEFAULT_FORCED_POINT):
                axarr[rowNo, colNo].text(0.85,0.9, 'FORCED',horizontalalignment='center',
                                         verticalalignment='center',transform=axarr[rowNo, colNo].transAxes)
                if (criticalFlag[lcPoint] == DEFAULT_CRITICAL_DETECTION):
                    strr = DEFAULT_CRITICAL_DETECTION_TEXT
                else:
                    strr = DEFAULT_NON_CRITICAL_DETECTION_TEXT
                axarr[rowNo, colNo].text(0.85, 0.75, strr, horizontalalignment='center',
                                         verticalalignment='center', transform=axarr[rowNo, colNo].transAxes)


            if (poorQualityFlags[lcPoint]):

                axarr[rowNo, colNo].text(0.85, 0.6, 'POOR QUALITY '+str(coefficients[lcPoint]), horizontalalignment='center',
                                             verticalalignment='center', transform=axarr[rowNo, colNo].transAxes)

            axarr[rowNo, colNo].text(0.85, 0.45, 'FRAC NEG FLUX ' + str(round(fractionalNegatives[lcPoint],2)),
                                         horizontalalignment='center',
                                         verticalalignment='center', transform=axarr[rowNo, colNo].transAxes)

            bNormal,bShapiro = IsNormaltest(image)

            axarr[rowNo, colNo].text(0.85, 0.3, 'NORMAL = '+str(bNormal),
                                         horizontalalignment='center',
                                         verticalalignment='center', transform=axarr[rowNo, colNo].transAxes)

            axarr[rowNo, colNo].text(0.85, 0.15, 'SHAPIRO = ' + str(bShapiro),
                                     horizontalalignment='center',
                                     verticalalignment='center', transform=axarr[rowNo, colNo].transAxes)

            colNo +=1
            lcPoint += 1

        # delete the extra subplot
        f.delaxes(axarr[rowNo,1])


    plt.show()

    f.savefig(DEFAULT_HISTOGRAM_COLLECTION+str(sourceID)+DEFAULT_HIST_IDENTIFIER)


def GetAllFITSData(sourceID):
    fitsImages = []

    if (sourceID not in lightCurveSourceList):
        print("Unknown Source ID")

    else:
        FITSImages = lightCurveFITSImages[sourceID]

        imageLocation =DEFAULT_FITS_IMAGE_FOLDER+sourceID

        for imageNo in range(len(FITSImages)):

            bValidData,fitsImage =  GetFITSFile(imageLocation,
                                FITSImages[str(imageNo)])
            if (bValidData):
                fitsImages.append(fitsImage)

    return fitsImages


def DisplayAllFITSData(sourceID):
    fitsImages = []


    if (sourceID not in lightCurveSourceList):
        print("Unknown Source ID")

    else:

        fitsImages = GetAllFITSData(sourceID)

        DisplayAllFITSImages(sourceID,fitsImages)




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


def ScanForCSVFiles(sourceLocation):
    fileList = []

    sourceList = os.scandir(sourceLocation)

    for entry in sourceList:
        if not entry.is_dir():
            fileList.append(entry.name)


    return fileList


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


def loadImage_CSVFile(sourceDir, source, imageNo):
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


def loadLC_CSVFile(rootData,fileName):

    bDataValid = True
    fileName = rootData+fileName

  #  print(fileName)
    sys.exit()
    if (os.path.isfile(fileName)):
        if (bDebug):
            print("*** Loading CSV File " + fileName + " ***")
        dataframe = pd.read_csv(fileName, header=None)
        dataReturn = dataframe.values
        if (bDebug):
            print("*** Completed Loading CSV File")
    else:
        print("*** CSV File Does Not Exist ***")
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



def GetOperationMode():
    bCorrectInput = False

    while (bCorrectInput == False):
        print("Transient Classifier ....")

  #      print("1 - Create/Save Stacked Images")
        print("1 - Build Image Models")
   #     print("3 - Build LC  Models")
        print("2 - Analyse Images")
        print("3 - Process FITS Files")
        print("4 - Build Classification Data")
        print("5 - Interrogate Classification Data")
       # print(" - Get Image Test Set")
        print("6 - Store Binary Data")
        print("7 - Load Binary Data")
        print("8 - Establish Lowest Coefficients")
        print("9 - Reclassify Sources")
        print("10 - Load/Test Models")
        print("11 - Soak Test Coefficients")
        print("12 - Create Mismatch File")
        print("13 - Determine Likely Object")
        print("14 - Find Images")
        print("15 - Compare Against Manual Classes")
        print("16 - Exit")

        allowedOps = [OP_BUILD_IMAGE_MODELS,
                      OP_ANALYSE_IMAGES,OP_PROCESS_FITS_FILES,
                      OP_BUILD_CLASSIFY_DATA,OP_INTERROGATE_DATA,
                      OP_STORE_BINARY_DATA,OP_LOAD_BINARY_DATA,OP_ANALYSE_MAN_CLASSES,OP_RECLASSIFY,OP_TEST_MODEL,OP_SOAK_TEST,OP_MISMATCH ,
                      OP_DETERMINE_OBJECT,OP_FIND_IMAGES,OP_COMPARE_RESULTS,OP_EXIT]

        selOP = input("Select Operation:")
        if (selOP in allowedOps):
            bCorrectInput = True
        else:
            print("*** Incorrect Input - try again ***")

    return selOP


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


def GetDataRoot(dataRoot,dataSet):

    if (dataSet == DEFAULT_POOR_QUALITY_DATA):
        print("Processing Poor Quality Images")

        rootData = DEFAULT_POOR_QUALITY_DATA_ROOT
    elif (dataSet == DEFAULT_ARTEFACT_DATA):
        print("Processing Artefact Images")

        rootData = DEFAULT_ARTEFACT_DATA_ROOT

    elif (dataSet == DEFAULT_AGN_DATA):
        print("Processing AGN Images")

        rootData = DEFAULT_AGN_DATA_ROOT
    else:

        print("Unknown Dataset, exiting....")
        sys.exit()

    finalRoot = dataRoot+rootData

    return finalRoot



def CreateTransientImageCSVFiles(analysisChoice,dataSet):

    # open the text file containing all sources and their respective FITS files
    # for each source, get a FITS filename, locate in the correct folder
    # then create a new folder with that source ID and copy the FITS files to that folder

    sourceList = ScanForSources(rootData)

    sourceFileDict = CreateAllCSVFiles(rootData, sourceList)

    totalNoFiles = ProcessAllCSVFiles(rootData , sourceFileDict, sourceList)

    print("*** Processed " + str(totalNoFiles) + " FITS FILES")



def StoreLCCSVFile(f,fluxEntry):

    if (len(fluxEntry)>0):
        for entry in range(len(fluxEntry)-1):
            strr = str(fluxEntry[entry])+','
            f.write(strr)

        strr = str(fluxEntry[entry+1])
        f.write(strr)
    else:
        print("Invalid flux data to write to csv file, exiting ...")
        sys.exit()


def CreateLCCSVFile(rootData,lineText):
    fluxEntry = []
    # extract the SOURCEID and create a file for it

    splitText = lineText.split(",")
    print(splitText)

    sourceID = splitText[0]
    csvFileName = rootData + DEFAULT_CSV_DIR + FOLDER_IDENTIFIER + sourceID + DEFAULT_CSV_FILETYPE

    f = open(csvFileName,"w")
    if (f):

        print("Storing LC Data in "+csvFileName)

        StoreLCCSVFile(f, splitText)
        f.close()



    else:
        print("Unable to create csv file, exiting...")
        sys.exit()

def CalculateImageCoefficient(image):

    correlation = CalculateAutocorrelation(image)
    pqCoefficientWithMax = correlation / np.max(image)

    return pqCoefficientWithMax,correlation


def CalculateNegativeFluxValues(imageData):
    numberNegativeValues= 0
    numberPositiveValues= 0
    negativeValues = 0
    positiveValues = 0
    totalValues = 0

    for imageValue in range(len(imageData)):
        totalValues += abs(imageData[imageValue])
        if (imageData[imageValue] <0):
            negativeValues += abs(imageData[imageValue])
            numberNegativeValues +=1
        else:
            numberPositiveValues += 1
            positiveValues += abs(imageData[imageValue])

  #  fractionNegative = numberNegativeValues/len(imageData)
    fractionNegative = negativeValues / totalValues

    return fractionNegative


def decodeImageNumber(filename):

    fName = filename.split('_')
    imageString = fName[2].split('.')

    imageNumber = len(imageString[0]) - 5

    imageValue = imageString[0][5:5 + imageNumber]
    imageValueNo = int(imageValue)
    imageValueNo = imageValueNo-1

    return str(imageValueNo)

def StorePoorQualityRecords(fPQ,fTransientsDropped, sourceID, lcPoint, coefficient,bTestForETAV,bTestEpochPairs,bDroppedTransient,numberDropped):

    if (fPQ) and (fTransientsDropped):
        fPQ.write('\n')
        strr = 'POSSIBLE POOR QUALITY ('+str(coefficient)+') FOR SOURCE: ' + str(sourceID) + ' in Image No ' + str(lcPoint)
        fPQ.write(strr)
        fPQ.write('\n')
        if (bDroppedTransient) and (numberDropped==0):
            strr = "TRANSIENT SOURCE "+str(sourceID)+" DROPPED "
            if (bTestForETAV):
                strr = strr +" DUE TO ETA AND V RECALCULATION"
            if (bTestEpochPairs):
                strr = strr +" DUE TO EPOCH PAIR RE_CALCULATION (POINT "+str(lcPoint)+" )"

            fTransientsDropped.write(strr)
            fTransientsDropped.write('\n')

    else:
        print("POOR QUALITY/TRANSIENTS DROPPED FILE NOT OPENED, exiting....")
        sys.exit()


def StoreSummaryInResultsFile(f, numberMatched,numberMissing,numberNotMatched,poorQualitySources,poorQualityClassification):

    numberMismatched = 0

    if (f):
        f.write('\n\n')
        f.write('**** SUMMARY RESULTS ***')
        f.write('\n\n')
        f.write("Total Number of Sources = "+str(len(lightCurveSourceList)))
        f.write('\n\n')
        f.write("Number Missing Classifications from Manual List =" + str(numberMissing))
        f.write('\n\n')
        f.write("Number Matched = "+str(numberMatched))
        f.write('\n\n')
        f.write("Number Not Matched = "+str(numberNotMatched))
        f.write('\n\n')
        correctMatchRatio= round((numberMatched/(numberMatched+numberNotMatched))*100,2)
        f.write('\n\n')
        f.write("Correctly Matched Ratio = "+str(correctMatchRatio))
        f.write('\n\n')



        f.write("*** POOR QUALITY ANALYSIS - ALL DEFINED MANUALLY AS POOR QUALITY, RESULTS FROM AUTO-CLASSIFICATION***")
        f.write('\n')
        for sourceID in range(len(poorQualitySources)):
            if (poorQualityClassification[sourceID][0]== DEFAULT_POOR_QUALITY_CLASS):
                f.write(str(poorQualitySources[sourceID])+" "+"CORRECT")
            else:
                numberMismatched += 1
                f.write(str(poorQualitySources[sourceID])+" "+str(poorQualityClassification[sourceID]))

            f.write('\n')
        f.write('\n')
        f.write('\n')

        f.write('Number Incorrectly Auto Mismatched on Poor Quality ='+str(numberMismatched))
        f.write('\n')
        f.write('Total Number Manually Classified as Poor Quality =' + str(len(poorQualitySources)))

        f.write('\n')
        f.write('\n')
        if (len(poorQualitySources) >0):
            totalNumberPercentage = round((numberMismatched / len(poorQualitySources)),2)*100
        else:
            totalNumberPercentage = 0.0

        f.write('Percentage Mismatched Poor Quality Detected vs Manually Detected = '+str(round(totalNumberPercentage,2)))

        f.write('\n')
        f.write('\n')
    else:
        print("Manual vs Auto File Not Opened, exiting...")
        sys.exit()


def StoreInResultsFile(f,sourceID, manualClassification, autoClassification, matchedStatus):

    if (f):

        if len(manualClassification) > 1:
            completeManualClass = manualClassification[0] + '/'
            completeManualClass = completeManualClass + manualClassification[1]
        else:
            completeManualClass = manualClassification

        if (bStoreUnmatchedOnly):
            if (matchedStatus==DEFAULT_STATUS_NOT_MATCHED):
                f.write(str(sourceID)+" "+str(completeManualClass)+" "+str(autoClassification)+" "+str(matchedStatus))
                f.write('\n')
        else:

            f.write(str(sourceID) + " " + str(completeManualClass) + " " + str(autoClassification) + " " + str(matchedStatus))
            f.write('\n')

    else:
        print("Manual vs Auto File Not Opened, exiting...")
        sys.exit()

def CreateVersionNumber(threshold):

    versionCoefficient = str(round(threshold, 2))

    if (bUseBinaryClassifierModel):
        versionCoefficient = versionCoefficient+'_M'

    return versionCoefficient




def CreateMismatchedDetail():
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    numberMismatched = 0

    coefficientThreshold = float(input("Input Coefficient Value:"))

    versionCoefficient = CreateVersionNumber(coefficientThreshold)

    f = open(DEFAULT_MISMATCHED_FILE + UNDERSCORE + versionCoefficient + DEFAULT_TXT_FILENAME,"w")
    if (f):
        print("creating Mismatch File...")
        f.write('\n')
        f.write('\n')
        f.write('MISMATCHED SUMMARY, COEFFICIENT = '+str(round(coefficientThreshold,2)))
        f.write('\n\n')
        numberOfClasses = len(setOfPossibleClasses)
        confusionMatrix = np.zeros((numberOfClasses,numberOfClasses))
        print(confusionMatrix.shape)

        for classType in setOfPossibleClasses:
            f.write('*** FOR MANUAL CLASS TYPE = '+classType+' ***')
            f.write('\n\n')
            for sourceID in lightCurveSourceList:
                manualClasses = lightCurveManualClassification[sourceID]

                autoClasses = lightCurvePossibleClasses[sourceID]


                if (classType in manualClasses):

                    # this is one we should be considering - now check all manual classes against all auto classes
                    numberMatches = 0

                    for classNo in range(len(manualClasses)):
                        manIndex = setOfPossibleClasses.index(classType)

                        autoIndex = setOfPossibleClasses.index(autoClasses)

                        confusionMatrix[manIndex, autoIndex] += 1
                        if manualClasses[classNo] in autoClasses:
                            # this is a match
                            numberMatches += 1

                    if (numberMatches==0):
                        numberMismatched+=1
                        f.write(str(sourceID))
                        f.write(' '+str(manualClasses)+' ')
                        f.write(str(lightCurvePossibleClasses[sourceID]))
                        f.write('\n')

        f.write('\n\n')
        f.write('** TOTAL NUMBER MISMATCHED  = '+str(numberMismatched))
        print('** TOTAL NUMBER MISMATCHED  = ' + str(numberMismatched))

        print(confusionMatrix)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,
                                      display_labels=setOfPossibleClassLabels)
        disp.plot(values_format='.5g')

        plt.show()
    else:
        print("Unable to open mismatched file, exiting...")
        sys.exit()

def StoreClassificationSummary(f):
    if (f):
        f.write('\n')
        f.write('\n')
        strr = '*** OVERALL CLASSIFICATION SUMMARY ***'
        f.write(strr)
        f.write('\n')
        strr ="Class POOR QUALITY :"+str(summaryObjectClasses[DEFAULT_POOR_QUALITY_CLASS])
        f.write(strr)
        f.write('\n')
        strr ="Class PLANET :"+str(summaryObjectClasses[DEFAULT_PLANET_CLASS])
        f.write(strr)
        f.write('\n')
        strr ="Class UNKNOWN :"+str(summaryObjectClasses[DEFAULT_UNKNOWN_CLASS])
        f.write(strr)
        f.write('\n')
        strr = "Class STAR :" + str(summaryObjectClasses[DEFAULT_STAR_CLASS])
        f.write(strr)
        f.write('\n')
        strr ="Class QSO :"+str(summaryObjectClasses[DEFAULT_QSO_CLASS])
        f.write(strr)
        f.write('\n')
        strr ="Class SEYFERT :"+str(summaryObjectClasses[DEFAULT_SEYFERT_CLASS])
        f.write(strr)
        f.write('\n')
        strr = "Class AGN :" + str(summaryObjectClasses[DEFAULT_AGN_CLASS])
        f.write(strr)
        f.write('\n')
        strr = "Class BL :" + str(summaryObjectClasses[DEFAULT_BL_CLASS])
        f.write(strr)
        f.write('\n')
        strr ="Class SPIRAL :"+str(summaryObjectClasses[DEFAULT_SPIRAL_CLASS])
        f.write(strr)
        f.write('\n')
        strr ="Class ULIRGS :"+str(summaryObjectClasses[DEFAULT_ULIRGS_CLASS])
        f.write(strr)
        f.write('\n')
        strr ="Class LIRGS :"+str(summaryObjectClasses[DEFAULT_LIRGS_CLASS])
        f.write(strr)
        f.write('\n')
        strr = "Class PULSAR :" + str(summaryObjectClasses[DEFAULT_PULSAR_CLASS])
        f.write(strr)
        f.write('\n')
        strr = "Class GALAXY :" + str(summaryObjectClasses[DEFAULT_GALAXY_CLASS])
        f.write(strr)
        f.write('\n')
        strr ="Class OBSCURED AGN :"+str(summaryObjectClasses[DEFAULT_OAGN_CLASS])
        f.write(strr)
        f.write('\n')
        f.write('\n')





def StoreClassificationRecords(f):
    objectClass = 0
    rationale = 0

    if (f):
        for sourceID in lightCurveSourceList:

            f.write(str(sourceID))
            f.write('   ')

            if (sourceID in lightCurvePossibleClasses) :
                if (lightCurvePossibleClasses[sourceID] != 0):
                    objectClass = lightCurvePossibleClasses[sourceID]

            if (sourceID in lightCurvePossibleRationale):
                if (lightCurvePossibleRationale[sourceID] != 0):
                    rationale = lightCurvePossibleRationale[sourceID]

            if (objectClass !=0):
                summaryObjectClasses[objectClass] += 1
                f.write(objectClass)

            else:
                f.write("   NOT CLASSIFIED")

            f.write('\n\n')
            if (rationale != 0):
                for objectNo in range(len(rationale)):
                    f.write(rationale[objectNo])
                    f.write('\n')
            else:
                f.write("NO RATIONALE")
                f.write('\n')
            f.write('\n\n')



        StoreClassificationSummary(f)

def CheckImpactOnEtaV(sourceID,lcPoint):
    bNotATransient=False


    print("By removing point ",str(lcPoint)+" from sourceID "+str(sourceID))
    print("Original ETA = ", lightCurveInitETAData[sourceID])
    print("Revised ETA = ",lightCurveRevalEtaData[sourceID][lcPoint])

    print("Original V  = ", lightCurveInitVData[sourceID])
    print("Revised V = ",lightCurveRevalVData[sourceID][lcPoint])

    return bNotATransient

def ProcessPoorEpochPairsForImage(sourceID,lcPoint):
    bDropTransient=False

    strr = "Poor quality image detected for lc point "+str(lcPoint)+" for source: "+str(sourceID)
    print(strr)

    criticalFlags = lightCurveCriticalFlag[sourceID]
    if (criticalFlags[lcPoint] == DEFAULT_CRITICAL_DETECTION):

        bDropTransient=True

    return bDropTransient


def ProcessPoorETAVForImage(sourceID,lcPoint):
    bDropTransient=False

    strr = "Poor quality image detected for lc point "+str(lcPoint)+" for source: "+str(sourceID)
    print(strr)

    if CheckImpactOnEtaV(sourceID, lcPoint):
        bDropTransient=True

    return bDropTransient



def InitialiseSummaryStats():

    summaryObjectClasses[DEFAULT_NOT_SPECIFIED_CLASS] = 0
    summaryObjectClasses[DEFAULT_ARTEFACT_CLASS] = 0
    summaryObjectClasses[DEFAULT_UNKNOWN_CLASS]= 0
    summaryObjectClasses[DEFAULT_AGN_CLASS] = 0
    summaryObjectClasses[DEFAULT_BL_CLASS] = 0
    summaryObjectClasses[DEFAULT_STAR_CLASS] = 0
    summaryObjectClasses[DEFAULT_POOR_QUALITY_CLASS] = 0
    summaryObjectClasses[DEFAULT_QSO_CLASS] = 0
    summaryObjectClasses[DEFAULT_SEYFERT_CLASS] = 0
    summaryObjectClasses[DEFAULT_ELLIPTICAL_CLASS] = 0
    summaryObjectClasses[DEFAULT_GALAXY_CLASS] = 0
    summaryObjectClasses[DEFAULT_SPIRAL_CLASS] = 0
    summaryObjectClasses[DEFAULT_ULIRGS_CLASS] = 0
    summaryObjectClasses[DEFAULT_LIRGS_CLASS] = 0
    summaryObjectClasses[DEFAULT_OAGN_CLASS] = 0
    summaryObjectClasses[DEFAULT_PLANET_CLASS] = 0
    summaryObjectClasses[DEFAULT_PULSAR_CLASS] = 0



def  DisplayPossibleClasses(sourceID):

    if (lightCurvePossibleClasses[sourceID] != 0 ):
        possibleObjects = lightCurvePossibleClasses[sourceID]
        possibleRationale = lightCurvePossibleRationale[sourceID]

  #      print("Number Possible Objects = ",len(possibleObjects))
  #      print("Number Possible Rationale = ",len(possibleRationale))

        for objectNo in range(len(possibleObjects)):
            print(possibleObjects[objectNo])
            print("Rationale: "+possibleRationale[objectNo])

    else:
        print("No objects to display for sourceID: "+str(sourceID))

def StoreImageRecord(f, fullFitsImageLocation):

    if (f):
        f.write(fullFitsImageLocation)
        f.write('\n')

    else:
        print("Image Record File Not Opened, exiting...")
        sys.exit()


def storeList(listData,listName):
    bValid=False

    print("Saving List "+listName)
    versionCoefficient = CreateVersionNumber(coefficientThreshold)
  #  versionCoefficient = str(round(coefficientThreshold,2))

    fileName = DEFAULT_BINARY_DATA_FOLDER+listName+UNDERSCORE+versionCoefficient+DEFAULT_PKL_EXTENSION

    f = open(fileName,"wb")
    if (f):
        bValid=True
        pickle.dump(listData,f)
        f.close()

    return bValid


def storeDictionary(dictData, dictName):
    bValid = False

    print("Saving Dictionary " + dictName)
    versionCoefficient = CreateVersionNumber(coefficientThreshold)
 #   versionCoefficient = str(round(coefficientThreshold, 2))
    fileName = DEFAULT_BINARY_DATA_FOLDER + dictName + UNDERSCORE+versionCoefficient+DEFAULT_PKL_EXTENSION

    f = open(fileName, "wb")
    if (f):
        bValid = True
        pickle.dump(dictData, f)
        f.close()

    return bValid

def storeModel(model, modelName,precision,recall,f1,scaler):
    bValid = False
    modelData = []
    global binaryClassifierModelList
    global bBinaryModelCreated

    print("Saving Model " + modelName)
    bBinaryModelCreated=True

    binaryClassifierModelList.append(model)
    binaryClassifierModelList.append(scaler)

    modelData.append(model)
    modelData.append(modelName)
    modelData.append(precision)
    modelData.append(recall)
    modelData.append(f1)
    modelData.append(scaler)
    versionCoefficient = CreateVersionNumber(coefficientThreshold)
#    versionCoefficient = str(round(coefficientThreshold, 2))
    fileName = DEFAULT_BINARY_MODEL_FOLDER + modelName + UNDERSCORE+versionCoefficient+DEFAULT_PKL_EXTENSION

    f = open(fileName, "wb")
    if (f):
        bValid = True
        pickle.dump(modelData, f)
        f.close()
    else:
        print("Could not save Binary Classifier Model, exiting...")
        sys.exit()

    return bValid

def loadDictionary(dictName):
    bValid = False

    print("Loading Dictionary " + dictName)
    versionCoefficient = CreateVersionNumber(coefficientThreshold)
 #   versionCoefficient = str(round(coefficientThreshold, 2))
    fileName = DEFAULT_BINARY_DATA_FOLDER + dictName + UNDERSCORE+versionCoefficient+DEFAULT_PKL_EXTENSION

    f = open(fileName, "rb")
    if (f):
        bValid=True
        restoredDictData = pickle.load(f)
        f.close()


    return bValid, restoredDictData

def loadModel(modelName):
    bValid = False
    global binaryClassifierModelList
    global bBinaryModelCreated

    print("Loading Model " + modelName)

    versionCoefficient = str(round(coefficientThreshold, 2))
    fileName = DEFAULT_BINARY_MODEL_FOLDER + modelName + UNDERSCORE+versionCoefficient+DEFAULT_PKL_EXTENSION

    f = open(fileName, "rb")
    if (f):
        bValid=True
        bBinaryModelCreated = True
        restoredModelData = pickle.load(f)
        f.close()

        model = restoredModelData[0]
        modelName = restoredModelData[1]
        precision = restoredModelData[2]
        recall = restoredModelData[3]
        f1 = restoredModelData[4]
        scaler = restoredModelData[5]

        binaryClassifierModelList.clear()

        binaryClassifierModelList.append(model)
        binaryClassifierModelList.append(scaler)

    return bValid,model,modelName,precision,recall,f1,scaler

def loadList(listName):
    bValid = False

    print("Loading List " + listName)
    versionCoefficient = CreateVersionNumber(coefficientThreshold)
  #  versionCoefficient = str(round(coefficientThreshold, 2))
    fileName = DEFAULT_BINARY_DATA_FOLDER + listName + UNDERSCORE+versionCoefficient+DEFAULT_PKL_EXTENSION

    f = open(fileName, "rb")
    if (f):
        bValid = True
        restoredListData = pickle.load(f)
        f.close()

    return bValid, restoredListData

def checkDataExists(dataName):
    from os.path import exists

    print("Checking List/Dict " + dataName)
    versionCoefficient = CreateVersionNumber(coefficientThreshold)
 #   versionCoefficient = str(round(coefficientThreshold, 2))
    fileName = DEFAULT_BINARY_DATA_FOLDER + dataName + UNDERSCORE+versionCoefficient+DEFAULT_PKL_EXTENSION
    f = exists(fileName)

    return f


def StoreBinaryData():


    bValid = storeList(lightCurveSourceList,"lightCurveSourceList")
    bValid = storeList(coefficientThreshold, "coefficientThreshold")
    if (bBinaryModelCreated):
        bValid = storeList(binaryClassifierModelList,"binaryModelClassifierList")

    # store each dictionary into its own  file

    bValid = storeDictionary(lightCurveStokesData,"lightCurveStokesData")
    bValid = storeDictionary(lightCurveRAData,"lightCurveRAData")
    bValid = storeDictionary(lightCurveDECData,"lightCurveDECData")
    bValid = storeDictionary(lightCurveInitETAData,"lightCurveInitETAData")
    bValid = storeDictionary(lightCurveInitVData,"lightCurveInitVData")
    bValid = storeDictionary(lightCurveFluxData,"lightCurveFluxData")
    bValid = storeDictionary(lightCurveRMSData,"lightCurveRMSData")
    bValid = storeDictionary(lightCurveDetectionData,"lightCurveDetectionData")
    bValid = storeDictionary(lightCurveCoefficientData,"lightCurveCoefficientData")
    bValid = storeDictionary(lightCurveCorrelationData,"lightCurveCorrelationData")
    bValid = storeDictionary(lightCurveRevalEtaData,"lightCurveRevalEtaData")
    bValid = storeDictionary(lightCurveRevalVData,"lightCurveRevalVData")
    bValid = storeDictionary(lightCurveFITSImages,"lightCurveFITSImages")
    bValid = storeDictionary(lightCurveWiseDataDict,"lightCurveWiseDataDict")
    bValid = storeDictionary(lightCurveSimbadDataDict,"lightCurveSimbadDataDict")
    bValid = storeDictionary(lightCurvePulsarDataDict,"lightCurvePulsarDataDict")
    bValid = storeDictionary(lightCurveSDSSDataDict, "lightCurveSDSSDataDict")
    bValid = storeDictionary(lightCurveFIRSTDataDict, "lightCurveFIRSTDataDict")
    bValid = storeDictionary(lightCurveNVSSDataDict, "lightCurveNVSSDataDict")
    bValid = storeDictionary(lightCurveQuasarDataDict, "lightCurveQuasarDataDict")
    bValid = storeDictionary(lightCurvePossibleClasses,"lightCurvePossibleClasses")
    bValid = storeDictionary(lightCurvePossibleRationale,"lightCurvePossibleRationale")
    bValid = storeDictionary(lightCurveSimbadClass, "lightCurveSimbadClass")
    bValid = storeDictionary(lightCurvePoorQualityImages,"lightCurvePoorQualityImages")
    bValid = storeDictionary(lightCurvePoorQualityFlags, "lightCurvePoorQualityFlags")
    bValid = storeDictionary(lightCurveCriticalFlag,"lightCurveCriticalFlag")
    bValid = storeDictionary(summaryObjectClasses,"summaryObjectClasses")
    bValid = storeDictionary(lightCurveManualClassification,"lightCurveManualClassification")
    bValid = storeDictionary(lightCurveFractionalNegativeData,"lightCurveFractionalNegativeData")
#    bValid = storeDictionary(imageClassModelDict, "imageClassModelDict")


def LoadBinaryData():
    global lightCurveSourceList
    global binaryClassifierModelList
    global lightCurveStokesData
    global lightCurveRAData
    global lightCurveDECData
    global lightCurveInitETAData
    global lightCurveInitVData
    global lightCurveFluxData
    global lightCurveRMSData
    global lightCurveDetectionData
    global lightCurveCoefficientData
    global lightCurveCorrelationData
    global lightCurveRevalEtaData
    global lightCurveRevalVData
    global lightCurveFITSImages
    global lightCurveWiseDataDict
    global lightCurveSimbadDataDict
    global lightCurvePulsarDataDict
    global lightCurveSDSSDataDict
    global lightCurveQuasarDataDict
    global lightCurveFIRSTDataDict
    global lightCurveNVSSDataDict
    global lightCurvePossibleClasses
    global lightCurvePossibleRationale
    global lightCurveSimbadClass
    global lightCurvePoorQualityImages
    global lightCurvePoorQualityFlags
    global lightCurveCriticalFlag
    global summaryObjectClasses
    global lightCurveManualClassification
    global lightCurveFractionalNegativeData
    global coefficientThreshold
    global bBinaryModelCreated



    if checkDataExists("lightCurveSourceList"):
        lightCurveSourceList.clear()
    bValid, lightCurveSourceList = loadList("lightCurveSourceList")

    if checkDataExists("binaryModelClassifierList"):
        binaryClassifierModelList.clear()
    bValid, binaryClassifierModelList = loadList("binaryModelClassifierList")

    bValid, coefficientThreshold = loadList("coefficientThreshold")

    # store each dictionary into its own  file

    if checkDataExists("lightCurveStokesData"):
        lightCurveStokesData.clear()

    bValid,lightCurveStokesData= loadDictionary("lightCurveStokesData")

    if checkDataExists("lightCurveRAData"):
        lightCurveRAData.clear()
    bValid,lightCurveRAData= loadDictionary("lightCurveRAData")

    if checkDataExists("lightCurveDECData"):
        lightCurveDECData.clear()
    bValid,lightCurveDECData=loadDictionary("lightCurveDECData")

    if checkDataExists("lightCurveInitETAData"):
        lightCurveInitETAData.clear()
    bValid,lightCurveInitETAData=loadDictionary("lightCurveInitETAData")

    if checkDataExists("lightCurveInitVData"):
        lightCurveInitVData.clear()
    bValid,lightCurveInitVData=loadDictionary("lightCurveInitVData")

    if checkDataExists("lightCurveFluxData"):
        lightCurveFluxData.clear()
    bValid,lightCurveFluxData=loadDictionary("lightCurveFluxData")

    if checkDataExists("lightCurveRMSData"):
        lightCurveRMSData.clear()
    bValid,lightCurveRMSData=loadDictionary("lightCurveRMSData")

    if checkDataExists("lightCurveDetectionData"):
         lightCurveDetectionData.clear()
    bValid,lightCurveDetectionData=loadDictionary("lightCurveDetectionData")

    if checkDataExists("lightCurveCoefficientData"):
        lightCurveCoefficientData.clear()
    bValid,lightCurveCoefficientData=loadDictionary("lightCurveCoefficientData")

    if checkDataExists("lightCurveCorrelationData"):
        lightCurveCorrelationData.clear()
    bValid, lightCurveCorrelationData = loadDictionary("lightCurveCorrelationData")

    if checkDataExists("lightCurverevalEtaData"):
        lightCurveRevalEtaData.clear()
    bValid,lightCurveRevalEtaData=loadDictionary("lightCurveRevalEtaData")

    if checkDataExists("lightCurveRevalVData"):
        lightCurveRevalVData.clear()
    bValid,lightCurveRevalVData=loadDictionary("lightCurveRevalVData")

    if checkDataExists("lightCurveFITSImages"):
        lightCurveFITSImages.clear()
    bValid,lightCurveFITSImages=loadDictionary("lightCurveFITSImages")

    if checkDataExists("lightCurveWiseData"):
        lightCurveWiseDataDict.clear()
    bValid,lightCurveWiseDataDict=loadDictionary("lightCurveWiseDataDict")

    if checkDataExists("lightCurvePulsarData"):
        lightCurveSDSSDataDict.clear()
    bValid,lightCurveSDSSDataDict=loadDictionary("lightCurveSDSSDataDict")

    if checkDataExists("lightCurveSDSSData"):
        lightCurvePulsDataDict.clear()
    bValid,lightCurvePulsarDataDict=loadDictionary("lightCurvePulsarDataDict")

    if checkDataExists("lightCurveQuasarData"):
        lightCurveQuasarDataDict.clear()
    bValid,lightCurveQuasarDataDict=loadDictionary("lightCurveQuasarDataDict")

    if checkDataExists("lightCurveFIRSTData"):
        lightCurveFIRSTDataDict.clear()
    bValid,lightCurveFIRSTDataDict=loadDictionary("lightCurveFIRSTDataDict")

    if checkDataExists("lightCurveNVSSData"):
        lightCurveNVSSDataDict.clear()
    bValid,lightCurveNVSSDataDict=loadDictionary("lightCurveNVSSDataDict")

    if checkDataExists("lightCurveSimbadData"):
        lightCurveSimbadDataDict.clear()
    bValid,lightCurveSimbadDataDict=loadDictionary("lightCurveSimbadDataDict")

    if checkDataExists("lightCurvePossibleClasses"):
        lightCurvePossibleClasses.clear()
    bValid,lightCurvePossibleClasses=loadDictionary("lightCurvePossibleClasses")

    if checkDataExists("lightCurvePossibleRationale"):
        lightCurvePossibleRationale.clear()
    bValid,lightCurvePossibleRationale=loadDictionary("lightCurvePossibleRationale")

    if checkDataExists("lightCurveSimbadClass"):
        lightCurveSimbadClass.clear()
    bValid, lightCurveSimbadClass = loadDictionary("lightCurveSimbadClass")

    if checkDataExists("lightCurvePoorQualityImages"):
        lightCurvePoorQualityImages.clear()
    bValid,lightCurvePoorQualityImages=loadDictionary("lightCurvePoorQualityImages")

    if checkDataExists("lightCurvePoorQualityFlags"):
        lightCurvePoorQualityFlags.clear()
    bValid,lightCurvePoorQualityFlags=loadDictionary("lightCurvePoorQualityFlags")

    if checkDataExists("lightCurveCriticalFlag"):
        lightCurveCriticalFlag.clear()
    bValid,lightCurveCriticalFlag=loadDictionary("lightCurveCriticalFlag")

    if checkDataExists("summaryObjectClasses"):
        summaryObjectClasses.clear()
    bValid,summaryObjectClasses=loadDictionary("summaryObjectClasses")

    if checkDataExists("lightCurveManualClassification"):
        lightCurveManualClassification.clear()
    bValid,lightCurveManualClassification=loadDictionary("lightCurveManualClassification")

    if checkDataExists("lightCurveFractionalNegativeData"):
        lightCurveFractionalNegativeData.clear()
    bValid,lightCurveFractionalNegativeData=loadDictionary("lightCurveFractionalNegativeData")

 #   if checkDataExists("imageClassModelDict"):
 #        imageClassModelDict.clear()
 #   bValid,imageClassModelDict=loadDictionary("imageClassModelDict")




def FindImagesForExamination():
    import random


    selectedImageList = []
    selectedSourceList = []
    coefficientValue = []
    lowestCoefficientsToFind = 1.40
    highestCoefficientsToFind = 1.80
    bFoundAllImages=False
    numberSourcesExamined = 0
    numberImagesFound = 0

    totalNumberSources = len(lightCurveSourceList)

    while (bFoundAllImages==False) and (numberSourcesExamined < totalNumberSources):
        # select a random source

        randomSourceNo = int((random.random()) * totalNumberSources)

        numberSourcesExamined += 1
        sourceID = lightCurveSourceList[randomSourceNo]

     #   print("sourceID = ",sourceID)
        imageList = lightCurveFITSImages[sourceID]
        coefficientList = lightCurveCoefficientData[sourceID]
        imageNo = 0
        bFoundImages =False

        while ((imageNo < len(imageList)) and (bFoundImages == False)):

                if (coefficientList[imageNo] > lowestCoefficientsToFind) and (coefficientList[imageNo] < lowestCoefficientsToFind+0.1):

                # found one
              #      print("found one")
                    selectedImageList.append(imageNo)
                    selectedSourceList.append(sourceID)
                    coefficientValue.append(coefficientList[imageNo])
                    numberImagesFound += 1
                    bFoundImages = True

                imageNo += 1

        if (bFoundImages==True):
            lowestCoefficientsToFind = lowestCoefficientsToFind+0.1
            if (lowestCoefficientsToFind > highestCoefficientsToFind):
                bFoundAllImages = True

    for sourceNo in range(len(selectedSourceList)):
        print("SOURCEID = ",selectedSourceList[sourceNo])
        print("IMAGE NO = ",selectedImageList[sourceNo])
        print("COEFFICIENT VALUE = ",coefficientValue[sourceNo])



    return selectedSourceList,selectedImageList


def ComputeFFT(imageData):
    from scipy.fft import fft,fftfreq

    yf = fft(imageData)
    xf = fftfreq(len(imageData))

    plt.plot(xf,np.abs(yf))
    plt.show()

def TestForPoorImage(imageData,coefficient):
    bPoorImage=False
    global binaryClassifierModelList

    model = binaryClassifierModelList[0]
    scaler = binaryClassifierModelList[1]

    prediction = int(TestIndividualImage(model, imageData,scaler ))

    if (prediction == DEFAULT_PQ_LABEL) and (coefficient > DEFAULT_MAX_COEFFICIENT):
        bPoorImage=True



    return bPoorImage


def ExamineAllCatalogs(sourceID):
    global lightCurvePossibleClasses
    global lightCurvePossibleRationale
    possibleRationale = []
    bWiseObjectDetected=False
    bSimbadDetected=False
    bMilliquasDetected=False
    bATNFObjectDetected=False

    # initialise it

    lightCurvePossibleClasses[sourceID] = 0
    lightCurvePossibleRationale[sourceID] = 0

    # WISE Color-Color Plot
    if (lightCurveWiseDataDict[sourceID] != 0):
        print("possible WISE source")
        wiseObjects,rationale = DetermineWiseObject(lightCurveWiseDataDict[sourceID])

        lightCurvePossibleClasses[sourceID] = wiseObjects
        possibleRationale.append(rationale)
        if (len(wiseObjects) == 1) and (wiseObjects != DEFAULT_UNKNOWN_CLASS):
            print("found a wise source")
            bWiseObjectDetected= True


    if (lightCurveSimbadDataDict[sourceID] != 0):
        # SIMBAD Catalog
        print("possible SIMBAD source")
        possibleObjects,rationale = DetermineSimbadObjects(lightCurveSimbadDataDict[sourceID])
        possibleRationale.append(rationale)
        if (bWiseObjectDetected==False):

            if (lightCurvePossibleClasses[sourceID] != 0):
                existingObjects = list(lightCurvePossibleClasses[sourceID])

                lightCurvePossibleClasses[sourceID] = possibleObjects+existingObjects

            else:
                lightCurvePossibleClasses[sourceID] = possibleObjects
            bSimbadDetected=True



    if (lightCurveQuasarDataDict[sourceID] != 0):
        # Milliquas Catalog
        print("possible MILLIQUAS source")
        possibleObjects,rationale = DetermineQuasarObjects(lightCurveQuasarDataDict[sourceID])
        possibleRationale.append(rationale)
        if (bWiseObjectDetected==False) and (bSimbadDetected==False):
            if (lightCurvePossibleClasses[sourceID] != 0):
                existingObjects = list(lightCurvePossibleClasses[sourceID])

                lightCurvePossibleClasses[sourceID] = possibleObjects+existingObjects

            else:
                lightCurvePossibleClasses[sourceID] = possibleObjects

            bMilliquasDetected=True

 #   if (lightCurveSDSSDataDict[sourceID] != 0):
        # SDSS Catalog
 #       print("possible SDSS source")
 #       possibleObjects,rationale = DetermineSDSSObjects(lightCurveSDSSDataDict[sourceID])
 #       possibleRationale.append(rationale)
 #       if (bWiseObjectDetected == False):
 #           if (lightCurvePossibleClasses[sourceID] != 0):
 #               existingObjects = list(lightCurvePossibleClasses[sourceID])

  #              lightCurvePossibleClasses[sourceID] = possibleObjects+existingObjects

  #          else:
  #              lightCurvePossibleClasses[sourceID] = possibleObjects



    if (lightCurvePulsarDataDict[sourceID] != 0):
        # ATNF Catalog
        print("possible ATNF source")
        possibleObjects,rationale = DetermineATNFObjects(lightCurvePulsarDataDict[sourceID])
        possibleRationale.append(rationale)
        if (bWiseObjectDetected==False) and (bSimbadDetected==False) and (bMilliquasObjectDetected==False):
            if (lightCurvePossibleClasses[sourceID] != 0):

                existingObjects = list(lightCurvePossibleClasses[sourceID])
                lightCurvePossibleClasses[sourceID] = possibleObjects+existingObjects

            else:
                lightCurvePossibleClasses[sourceID] = possibleObjects
            bATNFObjectDetected=True



    if (lightCurveFIRSTDataDict[sourceID] != 0):
        # FIRST Catalog
        print("possible FIRST source")
        possibleObjects,rationale = DetermineFIRSTObjects(lightCurveFIRSTDataDict[sourceID])
        possibleRationale.append(rationale)
        if (bWiseObjectDetected==False) and (bSimbadDetected==False) and (bMilliquasDetected==False) and (bATNFObjectDetected==False):
            if (lightCurvePossibleClasses[sourceID] != 0):
                existingObjects = list(lightCurvePossibleClasses[sourceID])
                lightCurvePossibleClasses[sourceID] =possibleObjects+existingObjects

            else:
                lightCurvePossibleClasses[sourceID] = possibleObjects

    if (lightCurvePossibleClasses[sourceID] == 0):

        # haven't found anything in the catalogues - assign UNKNOWN
        possibleObjects = []

        possibleObjects.append(DEFAULT_UNKNOWN_CLASS)
        lightCurvePossibleClasses[sourceID] = possibleObjects
        rationale = "No Catalog source detected - classified as UNKNOWN"
        print(rationale)
        possibleRationale.append(rationale)

    if (possibleRationale):
        lightCurvePossibleRationale[sourceID] = possibleRationale
    DisplayPossibleClasses(sourceID)

    return lightCurvePossibleClasses[sourceID]


def ClassifyIndividualLightCurve(fPQ,fTransients,fPQListFile,fNormalImageList,sourceID):
    global lightCurvePoorQualityImages
    global lightCurvePoorQualityFlags
    global lightCurveCoefficientData
    global lightCurveCorrelationData
    global lightCurveFractionalNegativeData
    global coefficientThreshold
    global lightCurvePossibleRationale
    global lightCurvePossibleClasses


    poorQualityImages = []
    coefficientValues = []
    correlationValues = []
    possibleObjects = []
    fractionalNegativeValues = []
    poorQualityFlags = []
    rationale = []


    bDroppedTransient = False
    numberDropped = 0

    fitsLocation = DEFAULT_FITS_IMAGE_FOLDER+sourceID
    fitsImageDict = lightCurveFITSImages[sourceID]
    detectionData = lightCurveDetectionData[sourceID]

    for lcPoint in range(len(detectionData)):
        bPoorImage=False
        fitsImageName = fitsImageDict[str(lcPoint)]
        fullFitsImageLocation = fitsLocation + FOLDER_IDENTIFIER + fitsImageName

        bValidData, imageData = OpenFITSFile(fullFitsImageLocation)
        if (bValidData):
            imageData = np.reshape(imageData, [1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE])

            fractionalNegativeValues.append(CalculateNegativeFluxValues(imageData[0]))
            coefficient, correlation = CalculateImageCoefficient(imageData[0])
            coefficientValues.append(coefficient)
            correlationValues.append(correlation)
            if (np.isnan(coefficient)):
                bPoorImage = True

            elif (bUseBinaryClassifierModel):
                bPoorImage = TestForPoorImage(imageData[0],coefficient)
            elif (coefficient > coefficientThreshold):
                bPoorImage = True

            poorQualityFlags.append(bPoorImage)

            if (detectionData[lcPoint]== DEFAULT_FORCED_POINT) and (bPoorImage):
            # we're only looking at quality of images on forced points
                # determine which image it is

                strr = "POSSIBLE POOR QUALITY FOR SOURCE: "+str(sourceID)+" in Image No "+str(lcPoint)+" = "+str(coefficient)
                print(strr)
                if (bTestForETAV):
                    if ProcessPoorETAVForImage(sourceID,lcPoint):
                    # this source will be dropped as it is no longer a transient
                        print(sourceID+" no longer considered a transient")
                        bDroppedTransient = True
                elif (bTestEpochPairs):

                    if ProcessPoorEpochPairsForImage(sourceID, lcPoint):
                        # this source will be dropped as it is no longer a transient
                        print(sourceID + " no longer considered a transient")
                        bDroppedTransient = True


                StorePoorQualityRecords(fPQ, fTransients,sourceID, lcPoint, coefficient,bTestForETAV,bTestEpochPairs,bDroppedTransient,numberDropped)
                numberDropped +=1

                poorQualityImages.append(lcPoint)

                StoreImageRecord(fPQListFile, fullFitsImageLocation)
            else:

                # store this as a good image

                StoreImageRecord(fNormalImageList,fullFitsImageLocation)

                poorQualityImages.append(DEFAULT_OK_IMAGE)


        else:
            print("Invalid FITS Data, exiting...")
            sys.exit()

    lightCurvePoorQualityImages[sourceID] = poorQualityImages
    lightCurvePoorQualityFlags[sourceID] = poorQualityFlags
    lightCurveCoefficientData[sourceID] = coefficientValues
    lightCurveCorrelationData[sourceID] = correlationValues
    lightCurveFractionalNegativeData[sourceID] = fractionalNegativeValues


    if (bDroppedTransient == False):

        if (min(coefficientValues) > DEFAULT_ARTEFACT_BOUNDARY):
            # suspect this source images may constitute an artefact

            possibleObjects.append(DEFAULT_ARTEFACT_CLASS)
            lightCurvePossibleClasses[sourceID] = possibleObjects
            rationaleText = "SOURCE " + str(sourceID) + "- ARTEFACT DETECTED"
            rationale.append(rationaleText)
            lightCurvePossibleRationale[sourceID] = rationale


        else:

            possibleObjects = ExamineAllCatalogs(sourceID)

    else:
        # this transient was dropped due to poor quality image (with forced points and re-evaluation metrics)
        possibleObjects.append(DEFAULT_POOR_QUALITY_CLASS)
        lightCurvePossibleClasses[sourceID] = possibleObjects

        rationaleText = "SOURCE "+str(sourceID)+" DROPPED DUE TO POOR QUALITY"
        rationale.append(rationaleText)
        lightCurvePossibleRationale[sourceID]= rationale


def ReEvaluatePoorQuality(sourceID):

    coefficientValues = []
    bValidThreshold = False

    fitsLocation = DEFAULT_FITS_IMAGE_FOLDER+sourceID
    fitsImageDict = lightCurveFITSImages[sourceID]
    detectionData = lightCurveDetectionData[sourceID]
    criticalFlag = lightCurveCriticalFlag[sourceID]

    for lcPoint in range(len(detectionData)):

        fitsImageName = fitsImageDict[str(lcPoint)]
        fullFitsImageLocation = fitsLocation + FOLDER_IDENTIFIER + fitsImageName

        bValidData, imageData = OpenFITSFile(fullFitsImageLocation)
        if (bValidData):
            imageData = np.reshape(imageData, [1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE])

            coefficient, correlation = CalculateImageCoefficient(imageData[0])

            if (detectionData[lcPoint]== DEFAULT_FORCED_POINT) and (ProcessPoorEpochPairsForImage(sourceID, lcPoint)==True):

                coefficientValues.append(coefficient)

                bDroppedTransient = True

        else:
            print("Invalid FITS Data, exiting...")
            sys.exit()

    if (len(coefficientValues)>0):
        lowestCoefficient = np.min(coefficientValues)
        bValidThreshold = True
    else:
        lowestCoefficient = 0.0

    return bValidThreshold,lowestCoefficient


def AnalyseManualClasses():
    coefficientThresholds = []

    for sourceID in lightCurveSourceList:
        manualClasses = lightCurveManualClassification[sourceID]
        if (DEFAULT_POOR_QUALITY_CLASS in manualClasses):
            print("POOR QUALITY FOR SOURCE "+str(sourceID))

            bValidThreshold,newCoefficientThreshold = ReEvaluatePoorQuality(sourceID)
            if (bValidThreshold):
                print("New Coefficient Threshold for sourceID "+str(sourceID)+" = "+str(newCoefficientThreshold))
                coefficientThresholds.append(newCoefficientThreshold)


    print("Lowest threshold = ",min(coefficientThresholds))
    n,bins,patches = plt.hist(coefficientThresholds,bins=25)
    plt.title("Coefficient Value Spread")
    plt.xlabel("Coefficient Values")
    plt.ylabel("Number Instances")
    plt.xticks(bins)
    plt.show()
    plt.savefig(DEFAULT_RESULTS_FOLDER+'coefficients.png')
    print(coefficientThresholds)

def FinalClassify(sourceID):
    global lightCurvePossibleClasses

    print(sourceID)

    possibleObjects = lightCurvePossibleClasses[sourceID]
    if (possibleObjects != 0):
        print(possibleObjects)
        objectDict = {i:possibleObjects.count(i) for i in possibleObjects}
        if len(objectDict) > 1:
            if DEFAULT_UNKNOWN_CLASS in objectDict.keys():
                del objectDict[DEFAULT_UNKNOWN_CLASS]

        lightCurvePossibleClasses[sourceID] = max(objectDict,key=objectDict.get)

        print("max  class = ",lightCurvePossibleClasses[sourceID])


def ClassifyAllLightCurves():
    bAllClassified = False
    global coefficientThreshold

    InitialiseSummaryStats()

    resultsVersion = CreateVersionNumber(coefficientThreshold)
 #   resultsVersion = str(round(coefficientThreshold,2))

    poorQualityFilename = DEFAULT_POOR_QUALITY_SUMMARY_FILE+UNDERSCORE+resultsVersion+DEFAULT_TXT_FILENAME
    normalImageListFilename = DEFAULT_NORMAL_IMAGE_FILE+UNDERSCORE+resultsVersion+DEFAULT_TXT_FILENAME
    resultsFilename = DEFAULT_RESULTS_FILE+UNDERSCORE+resultsVersion+DEFAULT_TXT_FILENAME
    droppedTransientsFilename = DEFAULT_TRANSIENTS_DROPPED_FILE+UNDERSCORE+resultsVersion+DEFAULT_TXT_FILENAME
    poorQualityResultsFilename = DEFAULT_PQ_RESULTS_FILE+UNDERSCORE+resultsVersion+DEFAULT_TXT_FILENAME

    if (bUseBinaryClassifierModel):
        bValid, model, modelName, precision, recall, f1, scaler = loadModel(RF_BINARY_CLASSIFIER)
        if (bValid):
            binaryClassifierModelList.append(model)
            binaryClassifierModelList.append(scaler)
        else:
            print("Unable to load binary classifier model, exiting...")
            sys.exit()

    fPQListFile = open(poorQualityFilename, "w+")
    fNormalImageList = open(normalImageListFilename, "w+")
    fClassification = open(resultsFilename, "w+")
    fTransients = open(droppedTransientsFilename, "w+")
    fPQ = open(poorQualityResultsFilename, "w+")
    if (fPQ) and (fClassification) and (fPQListFile) and (fNormalImageList):
        for sourceID in lightCurveSourceList:
            ClassifyIndividualLightCurve(fPQ,fTransients,fPQListFile,fNormalImageList,sourceID)
        for sourceID in lightCurveSourceList:
            FinalClassify(sourceID)
        bAllClassified = True
        StoreClassificationRecords(fClassification)

        fPQListFile.close()
        fNormalImageList.close()
        fClassification.close()
        fPQ.close()
    else:
        print("Unable to open Results or Poor Quality File(s), exiting...")
        sys.exit()


    return bAllClassified

def ProcessSingleLCEntry(lineText):

    fluxData = []
    fluxRMS = []
    fluxFlag = []
    fluxCrit = []
    detectionETA = []
    detectionV = []

    FITSImageDict = {}

    # extract the SOURCEID and associated data

    splitText = lineText.split(",")

    sourceID = splitText[0]
    RA = splitText[1]
    DEC = splitText[2]
    sourceEta = float(splitText[3])
    sourceV = float(splitText[4])

    lightCurveRAData[sourceID]= RA
    lightCurveDECData[sourceID] = DEC
    lightCurveInitETAData[sourceID] = sourceEta
    lightCurveInitVData[sourceID] = sourceV

    # now for each image - check if its of poor quality

    numberPoints = int((len(splitText)-DEFAULT_HEADER_LENGTH_CSV_FULL_LC)/DEFAULT_ENTRY_LENGTH_CSV_FULL_LC)
 #   print("number points = ",numberPoints)

    textEntry = DEFAULT_HEADER_LENGTH_CSV_FULL_LC
    for fluxEntry in range(numberPoints):

        fluxData.append(float(splitText[textEntry]))
        fluxRMS.append(float(splitText[textEntry+1]))
        fluxFlag.append(splitText[textEntry+2])
        fluxCrit.append(splitText[textEntry + 3])
        detectionETA.append(float(splitText[textEntry+4]))
        detectionV.append(float(splitText[textEntry+5]))

        textEntry += DEFAULT_ENTRY_LENGTH_CSV_FULL_LC

    lightCurveFluxData[sourceID] = fluxData
    lightCurveRMSData[sourceID] = fluxRMS
    lightCurveDetectionData[sourceID] = fluxFlag
    lightCurveRevalEtaData[sourceID] = detectionETA
    lightCurveRevalVData[sourceID] =detectionV
    lightCurveCriticalFlag[sourceID] = fluxCrit

    fitsLocation =  DEFAULT_FITS_IMAGE_FOLDER+str(sourceID)

    imageList = os.scandir(fitsLocation)

    for entry in imageList:
        imageNumber = decodeImageNumber(entry.name)
        FITSImageDict[imageNumber] = entry.name

    lightCurveFITSImages[sourceID] = FITSImageDict

    return sourceID





def AddOrSubtract():
    import random

    randomSign = random.random()
    if (randomSign >= 0.5):
        bSign = True
    else:
        bSign = False

    return bSign

def CreateSimulatedRandomLC(fluxData):
    import statistics
    import random

    newFluxData = []

    meanValue = statistics.mean(fluxData)
    var = statistics.pvariance(fluxData)

    for entry in range(len(fluxData)):
        if (AddOrSubtract()):
            newFluxEntry = meanValue + (random.random() * meanValue)
        else:
            newFluxEntry = meanValue - (random.random() * meanValue)

        newFluxData.append(newFluxEntry)


    return newFluxData


def CreateSimulatedLCCSVFile(classDataRoot,fileName,fluxData):

    splitText = fileName.split(DEFAULT_PERIOD_SYM)

    for simulatedFile in range(DEFAULT_NUMBER_SIMULATIONS):
        newFluxData = []

        simulatedFileName = classDataRoot+splitText[0]+DEFAULT_SIM_FILE+str(simulatedFile)+UNDERSCORE+DEFAULT_PERIOD_SYM+splitText[1]

        f = open(simulatedFileName,"w")
        if (f):

            #now  create simulated values and store
            for entry in range(len(fluxData)):
                if (AddOrSubtract()):
                    newFluxEntry = fluxData[entry]+(random.random()*SIMULATED_PERCENTAGE_DEVIATION/100)*fluxData[entry]
                else:
                    newFluxEntry = fluxData[entry]-(random.random() * SIMULATED_PERCENTAGE_DEVIATION/100)*fluxData[entry]

                newFluxData.append(newFluxEntry)

            randomFlux = CreateSimulatedRandomLC(fluxData)

            print("Storing Simulated LC Data in "+simulatedFileName)
            StoreLCCSVFile(f,newFluxData)
            f.close()

        else:
            print("Unable to create csv file, exiting...")
            sys.exit()


def CreateRandomLC_CSVFile(randomDataRoot,fileName,fluxData):

    splitText = fileName.split(DEFAULT_PERIOD_SYM)

    for simulatedFile in range(DEFAULT_NUMBER_SIMULATIONS):
        newFluxData = []
        randomFluxData = []

        simulatedRandomName = randomDataRoot+splitText[0]+DEFAULT_SIM_FILE+str(simulatedFile)+UNDERSCORE+'R'+DEFAULT_PERIOD_SYM+splitText[1]

        print(simulatedRandomName)
        f = open(simulatedRandomName, "w")
        if (f):
            randomFlux = CreateSimulatedRandomLC(fluxData)
            StoreLCCSVFile(f, randomFlux)
            f.close()
        else:
            print("Unable to create random csv file, exiting...")
            sys.exit()


def ProcessLCData(dataSet,rootData):

    LCDataFile = rootData+dataSet+UNDERSCORE+DEFAULT_LC_MEASUREMENTS_FILE
    print(LCDataFile)

    numEntries = 0
    with open(LCDataFile) as f:
        while (True):
            lineText = f.readline()
            if not lineText:
                break
            else:

                if (lineText != '\n'):
                    numEntries +=1
                    CreateLCCSVFile(rootData,lineText)

    f.close()

    return numEntries


def DisplayAllLightCurveData(sourceID):

    if (sourceID == 0):
        # display all light curves
        print("total number sources = ",len(lightCurveSourceList))

        for entry in range(len(lightCurveSourceList)):
            sourceID = lightCurveSourceList[entry]

            print("sourceID,RA,DEC, initial ETA, initial V = ",sourceID,lightCurveRAData[sourceID],
                  lightCurveDECData[sourceID],lightCurveInitialETAData[sourceID],lightCurveInitialVData[sourceID])

            print("Flux Data...",lightCurveFluxData[sourceID])
            print("RMS Data...", lightCurveRMSData[sourceID])
            print("Detection Data...", lightCurveDetectionData[sourceID])
            print("Reval ETA Data...", lightCurveRevalETAData[sourceID])
            print("Reval V Data...", lightCurveRevalVData[sourceID])
            print("Associated FITS Images...", lightCurveFITSImages[sourceID])
            if (wiseData[sourceID] != 0):
                print("Associated WISE Data= ",wiseData[sourceID])
    else:

        # just display one light curve
        print("sourceID,RA,DEC, initial ETA, initial V = ", sourceID, lightCurveRAData[sourceID],
              lightCurveDECData[sourceID], lightCurveInitialETAData[sourceID], lightCurveInitialVData[sourceID])

        print("Flux Data...", lightCurveFluxData[sourceID])
        print("RMS Data...", lightCurveRMSData[sourceID])
        print("Detection Data...", lightCurveDetectionData[sourceID])
        print("Reval ETA Data...", lightCurveRevalEtaData[sourceID])
        print("Reval V Data...", lightCurveRevalVData[sourceID])
        print("Associated FITS Images...", lightCurveFITSImages[sourceID])
        if (wiseData[sourceID] != 0):
            print("Associated WISE Data= ", wiseData[sourceID])



def ProcessAstroqueryWISE(sourceID):

     from astroquery.vizier import Vizier
     import astropy.units as u
     import astropy.coordinates as coord
     from astropy.coordinates import Angle
     from astropy.coordinates import SkyCoord
     global lightCurveWISEDataDict

     bValidData = False

     skyCoords = SkyCoord(lightCurveRAData[sourceID],
                          lightCurveDECData[sourceID], unit=("deg"))


     resultsTable = Vizier.query_region(skyCoords, radius=DEFAULT_WISE_RADIUS * u.arcsec,
                                        catalog=DEFAULT_WISE_CATALOG_NAME)



     if (len(resultsTable) == 0):

         lightCurveWiseDataDict[sourceID]= 0
     else:
         bValidData = True

         df = resultsTable[0].to_pandas()

         df = df.iloc[0]

         wiseID = df['AllWISE']
         wiseRA = df['RAJ2000']
         wiseDEC = df['DEJ2000']
         w1Mag = df['W1mag']
         w2Mag = df['W2mag']
         w3Mag = df['W3mag']
         w4Mag = df['W4mag']


         lightCurveWiseDataDict[sourceID] = (wiseID, wiseRA, wiseDEC, w1Mag, w2Mag, w3Mag, w4Mag)



     return bValidData


def ProcessAstroqueryFIRST(sourceID):
    from astroquery.vizier import Vizier
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.coordinates import Angle
    from astropy.coordinates import SkyCoord
    global lightCurveFIRSTDataDict

    bValidData = False

    skyCoords = SkyCoord(lightCurveRAData[sourceID],
                         lightCurveDECData[sourceID], unit=("deg"))

    resultsTable = Vizier.query_region(skyCoords, radius=DEFAULT_FIRST_RADIUS * u.arcsec,
                                       catalog=DEFAULT_FIRST_CATALOG_NAME)

    if (len(resultsTable) == 0):

        lightCurveFIRSTDataDict[sourceID] = 0

    else:

        df = resultsTable[0].to_pandas()

        df = df.iloc[0]

        firstID = df['FIRST']
        objectClass = df['c1']
        no2Mass = df['N2']
      #  print("ID= ",firstID)
      #  print("object class = ",objectClass)
      #  print("N2 = ",no2Mass)

        if (objectClass):
            bValidData=True
            lightCurveFIRSTDataDict[sourceID] = (firstID, objectClass)
        else:
            lightCurveFIRSTDataDict[sourceID] = 0


    return bValidData

def ProcessAstroqueryNVSS(sourceID):
    from astroquery.vizier import Vizier
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.coordinates import Angle
    from astropy.coordinates import SkyCoord
    global lightCurveNVSSDataDict

    bValidData = False

    skyCoords = SkyCoord(lightCurveRAData[sourceID],
                         lightCurveDECData[sourceID], unit=("deg"))

    resultsTable = Vizier.query_region(skyCoords, radius=DEFAULT_NVSS_RADIUS * u.arcsec,
                                       catalog=DEFAULT_NVSS_CATALOG_NAME)

    if (len(resultsTable) == 0):

        lightCurveNVSSDataDict[sourceID] = 0

    else:
        bValidData = True
        df = resultsTable[0].to_pandas()
      #  print(df.info())
        df = df.iloc[0]

        nvssSource = df['NVSS']

   #     print("For SourceID = "+str(sourceID)+", NVSS Source= "+str(nvssSource))

        lightCurveNVSSDataDict[sourceID] = (nvssSource)

    return bValidData

def ProcessAstroqueryMilliquas(sourceID):
    from astroquery.vizier import Vizier
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.coordinates import Angle
    from astropy.coordinates import SkyCoord
    global lightCurveQuasarDataDict
    bValidData = False

    skyCoords = SkyCoord(lightCurveRAData[sourceID],
                         lightCurveDECData[sourceID], unit=("deg"))

    resultsTable = Vizier.query_region(skyCoords, radius=DEFAULT_STANDARD_RADIUS * u.arcsec,
                                       catalog=DEFAULT_MILLIQUAS_CATALOG)


    if (len(resultsTable) == 0):

        lightCurveQuasarDataDict[sourceID] = 0

    else:


        df = resultsTable[0].to_pandas()

        df = df.iloc[0]

        bValidData = True

        lightCurveQuasarDataDict[sourceID] = (df['Name'], df['Type'])


    return bValidData


def ProcessAstroqueryATNF(sourceID):
    from astroquery.vizier import Vizier
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.coordinates import Angle
    from astropy.coordinates import SkyCoord
    global lightCurvePulsarDataDict

    bValidData = False

    skyCoords = SkyCoord(lightCurveRAData[sourceID],
                         lightCurveDECData[sourceID], unit=("deg"))

    resultsTable = Vizier.query_region(skyCoords, radius=DEFAULT_PULSAR_RADIUS * u.arcsec,
                                       catalog=DEFAULT_PULSAR_CATALOG_NAME)

    if (len(resultsTable) == 0):

        lightCurvePulsarDataDict[sourceID] = 0
    else:
        bValidData = True

        df = resultsTable[0].to_pandas()
     #   print(df.info())

        df = df.iloc[0]
        atnfID = df['PSRJ']
        distance = df['Dist']
   #     print("For source "+str(sourceID)+" atnfID = "+str(atnfID)+" at distance "+str(distance))

        lightCurvePulsarDataDict[sourceID] = (atnfID,distance)

    return bValidData

def ConvertToSDSSClass(sdssClass):

    if (sdssClass == DEFAULT_SDSS_GALAXY_CODE):
        className = "SDSS GALAXY"

    elif (sdssClass == DEFAULT_SDSS_STAR_CODE):
        className = "SDSS STAR"

    elif (sdssClass == DEFAULT_SDSS_UNKNOWN_CODE):
        className = "SDSS UNKNOWN"

    elif (sdssClass == DEFAULT_SDSS_GHOST_CODE):
        className = "SDSS GHOST"

    elif (sdssClass == DEFAULT_SDSS_KNOWN_CODE):
        className = "SDSS KNOWN"
    else:
        className = "UNKNOWN SDSS CODE"

    return className

def ProcessAstroquerySDSS(sourceID):
    from astroquery.vizier import Vizier
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.coordinates import Angle
    from astropy.coordinates import SkyCoord
    global lightCurveSDSSDataDict

    bValidData = False

    skyCoords = SkyCoord(lightCurveRAData[sourceID],
                         lightCurveDECData[sourceID], unit=("deg"))

    resultsTable = Vizier.query_region(skyCoords, radius=DEFAULT_PULSAR_RADIUS * u.arcsec,
                                       catalog=DEFAULT_SDSS_CATALOG_NAME)

    if (len(resultsTable) == 0):
        lightCurveSDSSDataDict[sourceID] = 0

    else:
        bValidData = True

        df = resultsTable[0].to_pandas()

        df = df.iloc[0]
        sdssID = df['objID']
        sdssClass = df['class']

   #     sdssClassName = ConvertToSDSSClass(sdssClass)

        lightCurveSDSSDataDict[sourceID] = (sdssID,sdssClass)

    return bValidData




def ProcessAstroquerySIMBAD(sourceID):
    from astroquery.simbad import Simbad
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.coordinates import Angle
    from astropy.coordinates import SkyCoord

    bValidData = False

    skyCoords = SkyCoord(lightCurveRAData[sourceID],
                         lightCurveDECData[sourceID], unit=("deg"))
    customSimbad = Simbad()

    customSimbad.add_votable_fields('otype')

    resultsTable = customSimbad.query_region(skyCoords, radius=DEFAULT_SIMBAD_RADIUS * u.arcsec)

    if (resultsTable is None):

        lightCurveSimbadDataDict[sourceID] = 0
    else:
        bValidData = True

        df = resultsTable.to_pandas()

        df = df.iloc[0]
   #     print("For Source "+str(sourceID)+" Identified nearby object of type "+str(df['OTYPE']))
        lightCurveSimbadDataDict[sourceID] = (df['MAIN_ID'],df['OTYPE'])

    return bValidData


def InterrogateLCData():

    sourceID = input("Input Source ID:")

    DisplayAllFITSData(sourceID)

def QueryWISEData():
    numberWISESources = 0

    print("Querying WISE For Sources...")
    for sourceID in lightCurveSourceList:

        bValidData = ProcessAstroqueryWISE(sourceID)
        if (bValidData):

            numberWISESources += 1

    print("Number WISE Sources Identified (From Total of " + str(len(lightCurveSourceList)) + ") = " + str(numberWISESources))

    return numberWISESources


def QueryMilliquasData():
    numberMilliquasSources = 0

    print("Querying Milliquas Catalogue For Sources...")
    for sourceID in lightCurveSourceList:

        bValidData = ProcessAstroqueryMilliquas(sourceID)
        if (bValidData):

            numberMilliquasSources += 1

    print("Number Milliquas Sources Identified (From Total of " + str(len(lightCurveSourceList)) + ") = " + str(numberMilliquasSources))

    StoreInMilliquasResults()

    return numberMilliquasSources


def QueryFIRSTData():
    numberFIRSTSources = 0

    print("Querying FIRST/NVSS/SDSS For Sources...")
    for sourceID in lightCurveSourceList:

        bValidData = ProcessAstroqueryFIRST(sourceID)
        if (bValidData):

            numberFIRSTSources += 1

    print("Number FIRST Sources Identified (From Total of " + str(len(lightCurveSourceList)) + ") = " + str(numberFIRSTSources))

    return numberFIRSTSources




def StoreInPulsarResults(numberPulsarSources):

    f = open(DEFAULT_PULSAR_RESULTS_FILE,"w")
    if (f):

        f.write("*** PULSAR SOURCES DETECTED ***")
        f.write('\n')
        f.write('\n')

        for sourceID in lightCurveSourceList:
            if (lightCurvePulsarDataDict[sourceID] != 0):
                pulsarData =  lightCurvePulsarDataDict[sourceID]
                atnfID = pulsarData[0]
                distance = pulsarData[1]
                f.write(" For Source "+sourceID+"ID = "+str(atnfID)+" at distance "+str(distance))
                f.write('\n\n')


        f.close()
    else:
        print("Can't open Pulsar results file, exiting...")
        sys.exit()


def StoreInSDSSResults():

    f = open(DEFAULT_SDSS_RESULTS_FILE,"w")
    if (f):

        f.write("*** SDSS SOURCES DETECTED ***")
        f.write('\n')
        f.write('\n')

        for sourceID in lightCurveSourceList:
            if (lightCurveSDSSDataDict[sourceID] != 0):
                sdssData =  lightCurveSDSSDataDict[sourceID]
                sdssName =sdssData[0]
                sdssClass = sdssData[1]
                f.write(" For Source "+sourceID+", ID = "+str(sdssName)+", type = "+str(sdssClass))
                f.write('\n\n')


        f.close()
    else:
        print("Can't open SDSS results file, exiting...")
        sys.exit()


def StoreInMilliquasResults():

    f = open(DEFAULT_MILLIQUAS_RESULTS_FILE,"w")
    if (f):

        f.write("*** MILLIQUAS SOURCES DETECTED ***")
        f.write('\n')
        f.write('\n')

        for sourceID in lightCurveSourceList:
            if (lightCurveQuasarDataDict[sourceID] != 0):
                milliQuasData =  lightCurveQuasarDataDict[sourceID]
                milliQuasName = milliQuasData[0]
                milliQuasType  = milliQuasData[1]
                f.write(" For Source "+sourceID+", Milliquas Name = "+str(milliQuasName)+" type= "+str(milliQuasType))
                f.write('\n\n')


        f.close()
    else:
        print("Can't open Milliquas results file, exiting...")
        sys.exit()


def StoreInFIRSTResults(numberPulsarSources):

    f = open(DEFAULT_FIRST_RESULTS_FILE,"w")
    if (f):

        f.write("*** FIRST/NVSS/SDSS AGN SOURCES DETECTED ***")
        f.write('\n')
        f.write('\n')

        for sourceID in lightCurveSourceList:
            if (lightCurveFIRSTDataDict[sourceID] != 0):
                firstData =  lightCurveFIRSTDataDict[sourceID]

              #  f.write(" For Source "+sourceID+"ID = "+str(atnfID)+" at distance "+str(distance))
                f.write('\n\n')


        f.close()
    else:
        print("Can't open FIRST results file, exiting...")


def QueryPulsarData():
    numberPulsarSources = 0

    print("Querying ATNF For Sources...")
    for sourceID in lightCurveSourceList:

        bValidData = ProcessAstroqueryATNF(sourceID)
        if (bValidData):

            numberPulsarSources += 1

    print("Number Pulsar Sources Identified (From Total of " + str(len(lightCurveSourceList)) + ") = " + str(numberPulsarSources))
    StoreInPulsarResults(numberPulsarSources)
    return numberPulsarSources

def QueryNVSSData():
    numberNVSSSources = 0

    print("Querying NVSS For Sources...")
    for sourceID in lightCurveSourceList:

        bValidData = ProcessAstroqueryNVSS(sourceID)
        if (bValidData):

            numberNVSSSources += 1

    print("Number NVSS Sources Identified (From Total of " + str(len(lightCurveSourceList)) + ") = " + str(numberNVSSSources))
  #  StoreInPulsarResults(numberPulsarSources)
    return numberNVSSSources






def QuerySDSSData():
    numberSDSSSources = 0

    print("Querying SDSS Catalog For Sources...")
    for sourceID in lightCurveSourceList:

        bValidData = ProcessAstroquerySDSS(sourceID)
        if (bValidData):

            numberSDSSSources += 1

    print("Number SDSS Sources Identified (From Total of " + str(len(lightCurveSourceList)) + ") = " + str(numberSDSSSources))
    StoreInSDSSResults()

    return numberSDSSSources




def QuerySIMBADData():
    numberSIMBADSources = 0

    print("Querying SIMBAD For Sources...")
    for sourceID in lightCurveSourceList:

        bValidData = ProcessAstroquerySIMBAD(sourceID)
        if (bValidData):

            numberSIMBADSources += 1

    print("Number SIMBAD Sources Identified (From Total of " + str(len(lightCurveSourceList)) + ") = " + str(numberSIMBADSources))

    return numberSIMBADSources



def ProcessCompleteLCData():
    global lightCurveSourceList

    LCDataFile = DEFAULT_ALL_LC_MEASUREMENTS_FILE
    print("Processing..."+LCDataFile)

    numEntries = 0
    with open(LCDataFile) as f:
        while (True):
            lineText = f.readline()
            if not lineText:
                break
            else:

                if (lineText != '\n'):
                    numEntries +=1
                   # print(lineText)
                    sourceID = ProcessSingleLCEntry(lineText)
                    print("processing sourceID ",sourceID)
                    lightCurveSourceList.append(sourceID)


    f.close()

    print("Processing complete for "+str(len(lightCurveSourceList))+" sources")

def ReadLCCSVFile(rootData,fileName):
    fluxData = []
    fluxFlag = []

    fullFileName =rootData+FOLDER_IDENTIFIER+fileName
    f = open(fullFileName,"r")
    if (f):
        textLine = f.readline()
        if (len(textLine)>0):
            splitText = textLine.split(',')
            numberPoints = int((len(splitText)-DEFAULT_HEADER_LENGTH_CSV)/2)
            print("number points = ",numberPoints)
            textEntry = DEFAULT_HEADER_LENGTH_CSV
            for fluxEntry in range(numberPoints):

                fluxData.append(float(splitText[textEntry]))
                fluxFlag.append(splitText[textEntry+1])
                textEntry += 2

    else:
        print("Cannot open LC CSV File, exiting....")
        sys.exit()

    return fullFileName,fluxData,fluxFlag

def CreateSourceAndFITSFiles(sourceID,fitsFileNames):
    import shutil

    # create source directory
    print("Creating source folder: "+str(sourceID))
    try:
        os.mkdir(DEFAULT_FITS_IMAGE_FOLDER+str(sourceID))
        # then copy all the files across


    except OSError as error:
        strr = "SOURCE "+str(sourceID)+" ALREADY EXISTS"
        print(strr)
        print(error)

    # then copy all the files across

    for entry in range(len(fitsFileNames)):
        srcFile = DEFAULT_FITS_SRC_FOLDER+fitsFileNames[entry]
        print(srcFile)
        destFile = DEFAULT_FITS_IMAGE_FOLDER + str(sourceID) + FOLDER_IDENTIFIER +fitsFileNames[entry]
        print(destFile)
        try:
            shutil.copy(srcFile, destFile)

        except OSError as error:
            strr = "FAILED TO COPY "+srcFile+" TO " +destFile
            print(strr)
            print(error)

def RenameAllFileExtensions():
    fileList = []

    fileList = os.scandir(DEFAULT_FITS_SRC_FOLDER)

    for entry in fileList:
        if entry.is_dir():
            print("ignore folders")
        else:
            fileName = DEFAULT_FITS_SRC_FOLDER+entry.name
            print(fileName)
            newfileName=fileName+'.fits'
            print(newfileName)
            os.rename(fileName,newfileName)



def ReadFITSReference():

    numberSourcesProcessed = 0

    with open(DEFAULT_FITS_REFERENCE_FILE,"r") as f:
        while (True):
            fitsFileNames = []

            textLine = f.readline()
            if not textLine:
               break
            else:
                numberSourcesProcessed +=1
                textLine = textLine.strip()
                splitText = textLine.split(',')
                print(splitText)
                sourceID = splitText[0]

                numberFITSFiles = int((len(splitText)-1))
                print("number files = ",numberFITSFiles)

                textEntry = 1
                for fileName in range(numberFITSFiles):

                    fitsFileNames.append(splitText[textEntry] + '.fits')
                    textEntry += 1

                print("fits filenames ...")
                print(fitsFileNames)
                CreateSourceAndFITSFiles(sourceID,fitsFileNames)



    print("No of sources processed = ",numberSourcesProcessed)


def ReadStokesFiles():

    global lightCurveStokesData

    numberSourcesProcessed = 0
    print("Processing Stokes Data")
    with open(DEFAULT_STOKES_REFERENCE_FILE,"r") as f:
        while (True):
            textLine = f.readline()
            if not textLine:
               break
            else:
                numberSourcesProcessed +=1
                textLine = textLine.rstrip()
                splitText = textLine.split(',')

                sourceID = splitText[0]
                stokesValue = splitText[1]
                # add to the dictionary

                lightCurveStokesData[sourceID]=stokesValue

    print("No of sources processed (stokes data) = ",numberSourcesProcessed)

def ReadImageListFiles(imageListFile):

    imageListFilenames = []

    numberSourcesProcessed = 0
    print("Processing Image List Data")
    with open(imageListFile,"r") as f:
        while (True):
            textLine = f.readline()
            if not textLine:
               break
            else:
                numberSourcesProcessed +=1
                textLine = textLine.rstrip()
                fileName = textLine
                imageListFilenames.append(fileName)



    print("No of filenames processed (image list data) = ",numberSourcesProcessed)

    return imageListFilenames

def CheckForGalaxies(manualClassification,autoClassification):
    bGalaxyMatched=False

    if (manualClassification==DEFAULT_GALAXY_CLASS) and ((autoClassification==DEFAULT_SPIRAL_CLASS) or (autoClassification==DEFAULT_LIRGS_CLASS) or
        (autoClassification==DEFAULT_ULIRGS_CLASS)):
        bGalaxyMatched=True

    return bGalaxyMatched


def CompareAgainstManualResults():
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    numberMatched = 0
    numberNotMatched =0
    numberManualMissing = 0
    poorQualitySources = []
    poorQualityClassification = []
    global coefficientThreshold
    expectedClasses = []
    predictedClasses= []


    print("coeff threshold = ",coefficientThreshold)
    versionCoefficient = CreateVersionNumber(coefficientThreshold)
    compareResultsFilename = DEFAULT_COMPARE_RESULTS_FILE+UNDERSCORE+versionCoefficient+DEFAULT_TXT_FILENAME

    f = open(compareResultsFilename,"w+")
    if not f:
        print("Failed to open auto vs manual results file,exiting...")
        sys.exit()
    else:

        for sourceID in lightCurveSourceList:

            if sourceID in lightCurveManualClassification:
                manualClasses = lightCurveManualClassification[sourceID]
                print("manual classes = ",manualClasses)
                if sourceID in lightCurvePossibleClasses:
                    autoClasses = lightCurvePossibleClasses[sourceID]
                    print("auto classes = ", autoClasses)
                else:
                    print("SourceID " + sourceID + " not found in auto classification,exiting...")
                    sys.exit()
            else:
                print("SourceID " + sourceID + " not found in manual classification,exiting...")
                sys.exit()

            bMatched=False

            if (manualClasses== DEFAULT_NOT_SPECIFIED_CLASS):

                numberManualMissing += 1

            elif (autoClasses in manualClasses):
                bMatched=True

            if (bMatched):

                if (DEFAULT_POOR_QUALITY_CLASS in manualClasses):
                    poorQualitySources.append(sourceID)
                    poorQualityClassification.append(autoClasses)


                StoreInResultsFile(f,sourceID,manualClasses,autoClasses,DEFAULT_STATUS_MATCHED)
                numberMatched +=1

            else:
                StoreInResultsFile(f,sourceID, manualClasses, autoClasses,DEFAULT_STATUS_NOT_MATCHED)

                numberNotMatched +=1

        StoreSummaryInResultsFile(f, numberMatched, numberManualMissing, numberNotMatched,poorQualitySources,poorQualityClassification)

        f.close()

def ReadManualResults():
    global lightCurveManualClassification

    numberSourcesProcessed = 0
    print("Processing Manual Classified Data")
    with open(DEFAULT_MANUAL_CLASSIFIED_FILE,"r") as f:
        while (True):
            manualClasses = []
            textLine = f.readline()
            if not textLine:
               break
            else:
                numberSourcesProcessed +=1

                textLine = textLine.rstrip()
                splitText = textLine.split(',')
                sourceID = splitText[0]

                manualClassification = splitText[1].upper()

                if '/' in manualClassification:
                    manualsplit = manualClassification.split('/')
                    numberOptions = len(manualsplit)
                    for optionNo in range(numberOptions):

                        if (manualsplit[optionNo] not in setOfPossibleClasses):
                            print("UNKNOWN CLASS IN MANUAL SET")
                            manualClasses.append(DEFAULT_NOT_SPECIFIED_CLASS)

                        else:
                            manualClasses.append(manualsplit[optionNo] )

                else:

                    if (manualClassification not in setOfPossibleClasses):
                        print("UNKNOWN CLASS IN MANUAL SET")
                        manualClasses.append(DEFAULT_NOT_SPECIFIED_CLASS)

                    else:
                        manualClasses.append(manualClassification)


                lightCurveManualClassification[sourceID]=manualClasses

    print("No of sources processed (manual classification) = ",numberSourcesProcessed)




def GetDataset():
    datasetChoice = ['Poor Quality Images', 'Artefact Images','AGN Images']
    shortDataSetChoice = [DEFAULT_POOR_QUALITY_DATA,DEFAULT_ARTEFACT_DATA,DEFAULT_AGN_DATA]


    bCorrectInput = False

    while (bCorrectInput == False):
        dataSet = input(
            'Choose Dataset For Model ' + datasetChoice[0] +'('+ shortDataSetChoice[0]+'), ' + datasetChoice[1] +'('+ shortDataSetChoice[1]+'),'+ datasetChoice[2] +'('+ shortDataSetChoice[2]+')'':')
        dataSet = dataSet.upper()
        if (dataSet not in shortDataSetChoice):
            print("*** Invalid Selection - Enter again... ***")
        else:
            print("*** Dataset Chosen = " + dataSet + " ***")
            bCorrectInput = True

    return dataSet



def GetAnalysisChoice():
    analysisChoice = ['Classification','Statistical']

    shortAnalysisChoice = [DEFAULT_CLASSIFY_DATA,DEFAULT_STATISTICAL_DATA]


    bCorrectInput = False

    while (bCorrectInput == False):
        analysis = input(
            'Choose Dataset For Model ' + analysisChoice[0] +'('+ shortAnalysisChoice[0]+'), ' + analysisChoice[1] +'('+ shortAnalysisChoice[1]+') :')
        analysis = analysis.upper()
        if (analysis not in shortAnalysisChoice):
            print("*** Invalid Selection - Enter again... ***")
        else:
            print("*** Analysis Chosen = " + analysis + " ***")
            bCorrectInput = True

    return analysis

def ConvertTypeToName(sourceType):

    if (sourceType == DEFAULT_ARTEFACT_DATA):
        sourceName = DEFAULT_ARTEFACT_DATA_NAME
    elif (sourceType == DEFAULT_AGN_DATA):
        sourceName = DEFAULT_AGN_DATA_NAME
    elif (sourceType == DEFAULT_RANDOM_DATA):
        sourceName = DEFAULT_RANDOM_DATA_NAME
    elif (sourceType == DEFAULT_POOR_QUALITY_DATA):
        sourceName = DEFAULT_POOR_QUALITY_DATA_NAME
    else:
        print("Unknown Source Type to Convert, exiting...")
        sys.exit()

    return sourceName


def  PlotWiseColourPlot(w1Mag,w2Mag,w3Mag,w4Mag):

    xTicks = [-1,0,1,2,3,4,5,6,7]
    yTicks = [-1, 0, 1, 2, 3, 4]

  #  plt.figure(figsize=(8,5))
    plt.title("WISE")
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    xValue = w3Mag - w4Mag
    yValue = w1Mag - w2Mag
    plt.plot(xValue,yValue,"bo")

    plt.xticks(xTicks)
    plt.yticks(yTicks)

    plt.show()


def TestForCircularObjects(x,y,centreX,centreY,radius):
    bObject=False

    distanceFromCentre = np.sqrt(((x-centreX)**2)+((y-centreY)**2))
 #   print("distance from centre = ",distanceFromCentre)
    if (distanceFromCentre <=radius):
        bObject=True

    return bObject


def pointInEllipse(x,y,xp,yp,d,D,angle):
    import math
    #tests if a point[xp,yp] is within
    #boundaries defined by the ellipse
    #of center[x,y], diameter d D, and tilted at angle(in degrees)

    cosa=math.cos(math.radians(angle))
    sina=math.sin(math.radians(angle))

    dd=d/2*d/2
    DD=D/2*D/2

    a =(cosa*(xp-x)+sina*(yp-y))**2
    b =(sina*(xp-x)-cosa*(yp-y))**2
    ellipse=(a/dd)+(b/DD)

    if ellipse <= 1:
        return True
    else:
        return False

def TestForEllipticalObjects(x,y,centreX,centreY,major,minor,angle):
    bObject=False

    if pointInEllipse(centreX, centreY, x, y, minor, major, angle):
        bObject=True

    return bObject



def  DetermineLikelyWISEObject(wiseID,w1Mag,w2Mag,w3Mag):

    objectType = []
    numberClasses = 0


    xValue = w2Mag - w3Mag
    yValue = w1Mag - w2Mag

    if TestForCircularObjects(xValue, yValue, DEFAULT_ELLIPTICAL_CENTRE_X, DEFAULT_ELLIPTICAL_CENTRE_Y,
                              DEFAULT_ELLIPTICAL_RADIUS):

        print("elliptical class")
        objectType.append(DEFAULT_GALAXY_CLASS)
        numberClasses += 1

    if TestForCircularObjects(xValue,yValue,DEFAULT_STAR_CENTRE_X,DEFAULT_STAR_CENTRE_Y,DEFAULT_STAR_RADIUS):

        if DEFAULT_GALAXY_CLASS not in objectType:
            objectType.append(DEFAULT_STAR_CLASS)
            print("star class")
            numberClasses += 1


    if TestForEllipticalObjects(xValue, yValue, DEFAULT_SPIRAL_CENTRE_X, DEFAULT_SPIRAL_CENTRE_Y,DEFAULT_SPIRAL_MAJOR,DEFAULT_SPIRAL_MINOR,DEFAULT_SPIRAL_ANGLE):

        if (DEFAULT_GALAXY_CLASS not in objectType) and (DEFAULT_STAR_CLASS not in objectType):
            print("spiral class")
            # only append this if an elliptical had not been detected
            objectType.append(DEFAULT_GALAXY_CLASS)
       # if (DEFAULT_SPIRAL_CLASS) not in objectType:
       #     objectType.append(DEFAULT_SPIRAL_CLASS)
            numberClasses += 1

    if TestForEllipticalObjects(xValue, yValue, DEFAULT_QSO_CENTRE_X, DEFAULT_QSO_CENTRE_Y, 2*DEFAULT_QSO_MAJOR,2*DEFAULT_QSO_MINOR,DEFAULT_QSO_ANGLE):

        if (DEFAULT_AGN_CLASS not in objectType) and (DEFAULT_GALAXY_CLASS not in objectType):
            print("qso class")
            objectType.append(DEFAULT_AGN_CLASS)
  #      if (DEFAULT_QSO_CLASS) not in objectType:
  #          objectType.append(DEFAULT_QSO_CLASS)
            numberClasses += 1

    if TestForCircularObjects(xValue, yValue, DEFAULT_SEYFERT_CENTRE_X, DEFAULT_SEYFERT_CENTRE_Y,DEFAULT_SEYFERT_RADIUS):

        if ((DEFAULT_AGN_CLASS) not in objectType) and ((DEFAULT_GALAXY_CLASS) not in objectType) :
            print("seyfert class")
            objectType.append(DEFAULT_AGN_CLASS)    # could only have come from QSO
      #  if (DEFAULT_SEYFERT_CLASS) not in objectType:
      #      objectType.append(DEFAULT_SEYFERT_CLASS)
            numberClasses += 1



    if TestForEllipticalObjects(xValue, yValue, DEFAULT_ULIRGS_CENTRE_X, DEFAULT_ULIRGS_CENTRE_Y, 2*DEFAULT_ULIRGS_MAJOR,2*DEFAULT_ULIRGS_MINOR,DEFAULT_ULIRGS_ANGLE):
        if ((DEFAULT_GALAXY_CLASS) not in objectType) and ((DEFAULT_AGN_CLASS) not in objectType):
            print("ulirgs class")
            objectType.append(DEFAULT_GALAXY_CLASS)
       # if (DEFAULT_ULIRGS_CLASS) not in objectType:
       #     objectType.append(DEFAULT_ULIRGS_CLASS)
            numberClasses += 1

    if TestForEllipticalObjects(xValue, yValue, DEFAULT_LIRGS_CENTRE_X, DEFAULT_LIRGS_CENTRE_Y, DEFAULT_LIRGS_MAJOR,
                                DEFAULT_LIRGS_MINOR, DEFAULT_LIRGS_ANGLE):
        if ((DEFAULT_GALAXY_CLASS) not in objectType) and ((DEFAULT_AGN_CLASS) not in objectType):
            print("lirgs class")
            objectType.append(DEFAULT_GALAXY_CLASS)

        #if (DEFAULT_LIRGS_CLASS) not in objectType:
         #   objectType.append(DEFAULT_LIRGS_CLASS)
            numberClasses += 1

    if TestForEllipticalObjects(xValue, yValue, DEFAULT_OAGN_CENTRE_X, DEFAULT_OAGN_CENTRE_Y, 2*DEFAULT_OAGN_MAJOR,
                                2*DEFAULT_OAGN_MINOR, DEFAULT_OAGN_ANGLE):
        if (DEFAULT_AGN_CLASS) not in objectType:
            print("oagn class")
            objectType.append(DEFAULT_AGN_CLASS)
   #     if (DEFAULT_OAGN_CLASS) not in objectType:
   #         objectType.append(DEFAULT_OAGN_CLASS)
            numberClasses +=1


    if (numberClasses == 0):
        rationale = "NO WISE Source Detected"
        objectType.append(DEFAULT_UNKNOWN_CLASS)
    else:
        if (wiseID != 0):
           rationale = "WISE Source ("+str(wiseID)+"), Type:  "+str(objectType[0]+" detected")
        else:
            rationale = "WISE Source Type:  " + str(objectType[0] + " detected")

    return objectType,rationale

def DetermineWiseObject(wiseDictData):

        wiseID = wiseDictData[0]
        wiseRA = wiseDictData[1]
        wiseDEC = wiseDictData[2]
        w1Mag = wiseDictData[3]
        w2Mag = wiseDictData[4]
        w3Mag = wiseDictData[5]
        w4Mag = wiseDictData[6]

        objectClasses,rationale = DetermineLikelyWISEObject(wiseID,w1Mag,w2Mag,w3Mag)

        return objectClasses,rationale


def ConvertQuasarToKnownClass(objectClass):


    objectClass = objectClass.upper()

    print("quasar object class = ",objectClass)
    if (objectClass[0] == DEFAULT_QUASAR_QSO_CODE):
 #       classType = DEFAULT_QSO_CLASS
        classType = DEFAULT_AGN_CLASS
    elif (objectClass[0] == DEFAULT_QUASAR_AGN_CODE):
        classType = DEFAULT_AGN_CLASS
    elif (objectClass[0] == DEFAULT_QUASAR_BL_CODE):
        classType = DEFAULT_BL_CLASS
    else:
        classType = DEFAULT_AGN_CLASS
   #     classType = DEFAULT_UNKNOWN_CLASS
     #   print("Unknown milliquas code"+str(objectClass))

    if classType in setOfPossibleClasses:
        print("identified a quasar class = ",classType)


    return classType


def ConvertSimbadToKnownClass(objectClass):

    objectClass = objectClass.upper()
    if (objectClass == DEFAULT_QSO_CLASS):
        objectClass = DEFAULT_AGN_CLASS

    if objectClass not in setOfPossibleClasses:
        objectName = DEFAULT_UNKNOWN_CLASS
    else:
        objectName = objectClass


    return objectName

def ConvertATNFToKnownClass(objectClass):
    bKnownClass = False

    objectClass = objectClass.upper()

    if objectClass in setOfPossibleClasses:
        bKnownClass = True

    return bKnownClass, objectClass

def ConvertFIRSTToKnownClass(objectClass):


    objectClass = objectClass.upper()

    if (objectClass == DEFAULT_FIRST_GALAXY_CODE):
        objectName = DEFAULT_GALAXY_CLASS
    elif (objectClass == DEFAULT_FIRST_STAR_CODE):
        objectName = DEFAULT_STAR_CLASS
    else:
        objectName = DEFAULT_UNKNOWN_CLASS

    return objectName

def ConvertATNFToKnownClass(objectClass):
    bKnownClass = False

    objectClass = objectClass.upper()

    if objectClass in setOfPossibleClasses:
        bKnownClass = True

    return bKnownClass, objectClass


def ConvertSDSSToKnownClass(sdssClass):

    if (sdssClass==DEFAULT_SDSS_GALAXY_CODE):

        objectClass = DEFAULT_GALAXY_CLASS
    elif (sdssClass==DEFAULT_SDSS_STAR_CODE):

        objectClass = DEFAULT_STAR_CLASS
    else:

        objectClass = DEFAULT_UNKNOWN_CLASS


    return objectClass


def DetermineSimbadObjects(simbadDictData):
    objectName = []

    mainID = simbadDictData[0]
    objectType = simbadDictData[1]

    objectClass = ConvertSimbadToKnownClass(objectType)

    rationale = "SIMBAD source ("+str(mainID)+") detected, Original Type: "+str(objectType)
    objectName.append(objectClass)


    return objectName,rationale

def DetermineQuasarObjects(quasarDictData):
    objectName = []

    quasarName = quasarDictData[0]
    quasarType  = quasarDictData[1]

    objectClass = ConvertQuasarToKnownClass(quasarType)
    rationale = "MILLIQUAS source (" + str(quasarName) + ") detected, Original Type: " + str(quasarType)

    objectName.append(objectClass)

    return objectName,rationale

def DetermineSDSSObjects(sdssDictData):
    sdssObject = []

    sdssName = sdssDictData[0]
    sdssClass  = sdssDictData[1]

    objectClass = ConvertSDSSToKnownClass(sdssClass)

    sdssObject.append(objectClass)

    rationale = "SDSS source (" + str(sdssName) + ") detected, Original Type: " + str(sdssClass)

    return sdssObject,rationale

def DetermineFIRSTObjects(firstDictData):
    objectName = []

    firstName = firstDictData[0]
    firstType  = firstDictData[1]

    objectClass = ConvertFIRSTToKnownClass(firstType)
    rationale = "FIRST source (" + str(firstName) + ") detected, Original Type: " + str(firstType)

    objectName.append(objectClass)

    return objectName,rationale

def DetermineATNFObjects(atnfDictData):
    objectName = []


    pulsarName = atnfDictData[0]
    distance   = atnfDictData[1]

    objectName.append(DEFAULT_PULSAR_CLASS)
    rationale = "ATNF source (" + str(pulsarName) + ") detected, Original Type: " + str(DEFAULT_PULSAR_CLASS)

    return objectName,rationale


def ProcessClassificationData(dataRoot,dataSet):

    bValidData = False
    trainingData = []
    sourceDetails = []


    rootData = GetDataRoot(dataRoot,dataSet)

    sourceList = ScanForSources(rootData)

    numberSources = len(sourceList)

    for source in range(numberSources):

        imageLocation, imageList = ScanForImages(rootData, sourceList[source])

        trainingImageData = []

        for imageNo in range(len(imageList)):

            bValidData, sourceData = loadImage_CSVFile(rootData, sourceList[source], imageNo)
            if (bValidData):

                if (bShrinkImages):
                    sourceImageData = np.reshape(sourceData, (1, XSIZE_SMALL_FITS_IMAGE * YSIZE_SMALL_FITS_IMAGE))
                else:
                    sourceImageData = np.reshape(sourceData, (1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE))

                sourceDetails.append(str(sourceList[source]) + UNDERSCORE + str(imageNo))


                trainingImageData.append(sourceImageData)
        trainingData.append(trainingImageData)

    if (len (trainingData) > 0):
        bValidData=True

    sourceTypeName = ConvertTypeToName(dataSet)
    print("No of Sources Loaded For " + sourceTypeName + " = " + str(len(trainingData)))
    for source in range(len(trainingData)):
        imageData = trainingData[source]
        strr = 'no of images for source ' + str(sourceList[source]) + ' = ' + str(len(imageData))
        print(strr)


    return bValidData,trainingData, sourceList

def ProcessLCClassificationData(dataRoot,dataSet):
    bValidData = False
    trainingData = []

    rootData = GetDataRoot(dataRoot,dataSet)
    rootData = rootData+DEFAULT_CSV_DIR+FOLDER_IDENTIFIER
    print("rootData = ",rootData)
    sys.exit()
    fileList = ScanForCSVFiles(rootData)
    print(fileList)
    sys.exit()
    numberFiles = len(fileList)
    if (numberFiles >0):

        for fileEntry in range(numberFiles):

            bValidData, sourceData = loadLC_CSVFile(rootData,fileList[fileEntry])
            if (bValidData):

                trainingData.append(sourceData)


    if (len (trainingData) > 0):
        bValidData=True


    return bValidData,trainingData


def DisplayQQPlot(imageData):
    import statsmodels.api as sm


    fig = sm.qqplot(imageData)





def StackImages(imageData,sourceID):

    numberImagesToStack = len(imageData)

    strr = "no images to stack for source "+str(sourceID)+ "= "+str(numberImagesToStack)

    print(strr)

    stackedImage = imageData[0]

    for image in range(1, len(imageData)):
        stackedImage += imageData[image]

    stackedImage = stackedImage / numberImagesToStack


    return stackedImage

def SetPlotParameters():
    plt.rc('axes', titlesize=SMALL_FONT_SIZE)
    plt.rc('axes', labelsize=SMALL_FONT_SIZE)
    plt.rc('xtick', labelsize=SMALL_FONT_SIZE)
    plt.rc('ytick', labelsize=SMALL_FONT_SIZE)


def DisplayStackedImages(dataSet,stackedData, labelList):

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

 #   fig.savefig(DEFAULT_STACKED_FILENAME_LOC)




def ScaleInputData(X):

    scaler = MinMaxScaler()
    scaler.fit(X)
    normalised = scaler.transform(X)

    return normalised, scaler

def createLabels(labelList):
    labelDict, OHE = createOneHotEncodedSet(labelList)

    return labelDict, OHE

def TransformTrainingData(trainingData):
    dataAsArray = np.asarray(trainingData)
    dataAsArray = np.reshape(dataAsArray, (dataAsArray.shape[0], dataAsArray.shape[2]))

    return dataAsArray


def assignLabelValues(label, numberOfSamples):
    shape = (numberOfSamples, len(label))

    a = np.empty((shape))

    a[:] = label

    return a

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



def CreateTrainingAndTestData(cnnModel,ListOfLabels,completeTrainingData, trainingDataSizes):

    datasetLabels = []
    finalTrainingData = []

    # create labels and scale data

    labelDict, OHE = createLabels(ListOfLabels)

    OHELabelValues = list(labelDict.values())
    labelList = list(labelDict.keys())

    print("labelList = ",labelList)
    print("OHE Values = ", OHELabelValues)


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


def createBinaryTrainingAndTestData(completeTrainingData,completeLabelData,trainingDataSizes):

    datasetLabels = []
    finalTrainingData = []


    for dataset in range(len(completeTrainingData)):
        dataAsArray = TransformTrainingData(completeTrainingData[dataset])

        datasetLabels.append(np.asarray(completeLabelData[dataset]))

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

    # final check for nan's

    XTrain = np.nan_to_num(XTrain)
    XTest = np.nan_to_num(XTest)

    print("Final Training Data = ", XTrain.shape)
    print("Final Test Data = ", XTest.shape)

    print("Final Training Label Data = ", ytrain.shape)
    print("Final Test Label Data = ", ytest.shape)

    return XTrain, XTest, ytrain, ytest, scaler



def StackAllImages(sourceData,sourceList):
    allStackedImages = []
    allLabels = []

    for source in range(len(sourceData)):
        stackedImage = StackImages(sourceData[source],sourceList[source])

        allStackedImages.append(stackedImage)
        allLabels.append(sourceList[source])

    return allLabels,allStackedImages


def CalculateAutocorrelation(image):


    autocorrelation = np.correlate(image,image)

    return autocorrelation


def CalculateEntropy(image):
    import skimage.measure



    entropy= skimage.measure.shannon_entropy(image)

    return entropy


def IsNormaltest(imageData):

    from scipy.stats import normaltest
    from scipy.stats import shapiro

    bNormal=False
    bShapiroNormal = False
    alpha = 0.05

    histData, binEdges = np.histogram(imageData, bins=DEFAULT_NUMBER_HIST_BINS)

    ks_statistic1, p_value1 = normaltest(histData)
    if (p_value1 > alpha):
        bNormal= True

    ks_statistic2, p_value2 = shapiro(histData)
    if (p_value2 > alpha):
        bShapiroNormal=True

    return bNormal,bShapiroNormal

def CalculateSkewnessAndKurtosis(imageNo,histData):
    from scipy.stats import normaltest
    from scipy.stats import shapiro
    from scipy.stats import skewtest
    from scipy.stats import kurtosistest

    stat_skew,pValueSkew = skewtest(histData)
    stat_kur,pValueKurtosis = kurtosistest(histData)
    return pValueSkew,pValueKurtosis



def RandomForestModel(XTrain, ytrain, XTest, ytest):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

  #  rndClf = RandomForestClassifier(n_estimators=30, max_depth=9, min_samples_leaf=15)

    rndClf = RandomForestClassifier()

    models = []
    accuracyScores = []

    MAX_NUMBER_SOAK_TESTS = 10

    if (bSoakModelTest):
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
                print("Random Forest Classifier, Completed Test No (No Greater Accuracy) " + str(testNo))

        highestAccuracy = max(accuracyScores)
        highestIndex = accuracyScores.index(highestAccuracy)

        highestAccuracy = round((highestAccuracy * 100), 2)
        print("Highest Accuracy = ", highestAccuracy)

        bestModel = models[highestIndex]

    else:
        rndClf.fit(XTrain, ytrain)
        y_pred = rndClf.predict(XTest)
        highestAccuracy = accuracy_score(ytest, y_pred)

        print(rndClf.__class__.__name__, accuracy_score(ytest, y_pred))
        bestModel = rndClf

    models.clear()
    accuracyScores.clear()

    return bestModel, highestAccuracy

def ComputeRandomForestModel(XTrain, ytrain, XTest, ytest):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.metrics import precision_score,recall_score,f1_score
 #   global imageClassModelDict

    rndClf = RandomForestClassifier()

    rndClf.fit(XTrain, ytrain)

    y_train_pred = cross_val_predict(rndClf,XTrain,ytrain,cv=10)
    cmTrain = confusion_matrix(ytrain,y_train_pred,labels=rndClf.classes_)

    precisionScoreTrain = precision_score(ytrain,y_train_pred)
    recallScoreTrain = precision_score(ytrain, y_train_pred)
    f1ScoreTrain = f1_score(ytrain, y_train_pred)

    y_test_pred = cross_val_predict(rndClf, XTest, ytest, cv=10)
    precisionScoreTest = precision_score(ytest, y_test_pred)
    recallScoreTest = precision_score(ytest, y_test_pred)
    f1ScoreTest = f1_score(ytest, y_test_pred)

    print("*** PRECISION/RECALL/F1 SCORES - TRAIN :"+str(round(precisionScoreTrain,2))+", "+str(round(recallScoreTrain,2))+", "+str(round(f1ScoreTrain,2)))

    print("*** PRECISION/RECALL/F1 SCORES - TEST :" + str(round(precisionScoreTest, 2))+", " + str(round(recallScoreTest, 2))+", "+str(round(f1ScoreTest,2)))


    cmTest = confusion_matrix(ytest, y_test_pred, labels=rndClf.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cmTrain, display_labels=[DEFAULT_NORMAL_LABEL_NAME,DEFAULT_PQ_LABEL_NAME])
    disp.plot()
    disp = ConfusionMatrixDisplay(confusion_matrix=cmTest,display_labels=[DEFAULT_NORMAL_LABEL_NAME, DEFAULT_PQ_LABEL_NAME])
    disp.plot()
    plt.show()


  #  imageClassModelDict[RF_BINARY_CLASSIFIER] = rndClf


    return rndClf, precisionScoreTest,recallScoreTest,f1ScoreTest


def ReadAllImageListFiles():

    global coefficientThreshold

    resultsVersion = str(round(coefficientThreshold, 2))

    poorQualityFilename = DEFAULT_POOR_QUALITY_SUMMARY_FILE + UNDERSCORE + resultsVersion + DEFAULT_TXT_FILENAME
    normalQualityFilename = DEFAULT_NORMAL_IMAGE_FILE + UNDERSCORE + resultsVersion + DEFAULT_TXT_FILENAME

    poorQualityFileList = ReadImageListFiles(poorQualityFilename)
    normalQualityFileList = ReadImageListFiles(normalQualityFilename)

    return poorQualityFileList,normalQualityFileList


def storeModelPerformance(precision,recall,f1):

 #   versionCoefficient = str(round(coefficientThreshold,2))
    versionCoefficient = CreateVersionNumber(coefficientThreshold)
    f= open(DEFAULT_BINARY_MODEL_FOLDER+DEFAULT_BINARY_MODEL_NAME+UNDERSCORE+'PERF'+UNDERSCORE+versionCoefficient+DEFAULT_TXT_FILENAME,"w")
    if (f):
        f.write("MODEL SUMMARY FOR COEFFICIENT = "+versionCoefficient)
        f.write("\n\n")
        f.write("PRECISION  = "+str(round(precision,2)))
        f.write("\n\n")
        f.write("RECALL  = " + str(round(recall, 2)))
        f.write("\n\n")
        f.write("F1  = " + str(round(f1, 2)))
        f.write("\n\n")

        f.close()
    else:
        print("unable to open model performance file, exiting...")
        sys.exit()


def ProcessImageModels():

    trainingDataSizes = []
    trainingPQImages = []
    trainingPQLabels = []
    trainingNormalImages = []
    trainingNormalLabels = []
    completeTrainingData=  []
    completeLabelData = []


    poorQualityFileList,normalQualityFileList = ReadAllImageListFiles()

    for fileNo in range(len(poorQualityFileList)):

        bValidData, pqImageData = OpenFITSFile(poorQualityFileList[fileNo])
        if (bValidData):
            pqImageData = np.reshape(pqImageData, [1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE])
            trainingPQImages.append(pqImageData)
            trainingPQLabels.append(DEFAULT_PQ_LABEL)


    completeTrainingData.append(trainingPQImages)
    completeLabelData.append(trainingPQLabels)
    numberPQImages = len(trainingPQImages)
    trainingDataSizes.append(len(trainingPQImages))

    for fileNo in range(len(normalQualityFileList)):

        bValidData, normalImageData = OpenFITSFile(normalQualityFileList[fileNo])
        if (bValidData):
            normalImageData = np.reshape(normalImageData, [1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE])
            trainingNormalImages.append(normalImageData)
            trainingNormalLabels.append(DEFAULT_NORMAL_LABEL)

    # truncate normal image list to be same size as poor quality list

    del trainingNormalImages[numberPQImages:]
    del trainingNormalLabels[numberPQImages:]

    completeTrainingData.append(trainingNormalImages)
    trainingDataSizes.append(len(trainingNormalImages))
    completeLabelData.append(trainingNormalLabels)

    print("no of normal quality images = ", len(trainingNormalImages))

    print("size of complete training = ", len(completeTrainingData))
    print("size of  training sizes = ", len(trainingDataSizes))



    XTrain, XTest, ytrain, ytest, scaler = createBinaryTrainingAndTestData(completeTrainingData,completeLabelData,trainingDataSizes)

    XTrain = np.nan_to_num(XTrain)
    XTest = np.nan_to_num(XTest)

    newModel, precision,recall, f1 = ComputeRandomForestModel(XTrain, ytrain, XTest, ytest)

    storeModel(newModel, RF_BINARY_CLASSIFIER,precision, recall, f1,scaler)

    storeModelPerformance(precision,recall, f1)

def TestBinaryClassifier(model,scaler):
    import random
    bGotImageNo = False
    bFinished = False

    while (bFinished==False):
        bGotImageNo = False
        sourceID = input("Enter source ID : ")
        if not sourceID:
            # choose random source
            randomSourceNo = int((random.random()) * len(lightCurveSourceList))

            sourceID = lightCurveSourceList[randomSourceNo]
            print("Selected Random Source No: "+sourceID)
            fitsLocation = DEFAULT_FITS_IMAGE_FOLDER + sourceID
            fitsImageDict = lightCurveFITSImages[sourceID]
            while (bGotImageNo == False):
                imageSelection = input("Select Image No: (0-" + str(len(fitsImageDict) - 1) + ")")
                if not imageSelection:
                    imageNo = int((random.random()) * len(fitsImageDict)-1)
                else:
                    imageNo = int(input("Select Image No: (0-"+str(len(fitsImageDict)-1)+")"))

                if (imageNo <0) or (imageNo > len(fitsImageDict)-1):
                    print("Invalid Selection, choose again")
                else:
                    bGotImageNo = True
                    print("Selected image no "+str(imageNo))

            fitsImageName = fitsImageDict[str(imageNo)]
            fullFitsImageLocation = fitsLocation + FOLDER_IDENTIFIER + fitsImageName

            bValidData, imageData = OpenFITSFile(fullFitsImageLocation)
            if (bValidData):
                imageData = np.reshape(imageData, [1, XSIZE_FITS_IMAGE * YSIZE_FITS_IMAGE])

                prediction = int(TestIndividualImage(model,imageData,scaler))

                if (prediction==DEFAULT_NORMAL_LABEL):
                    resultText= DEFAULT_NORMAL_LABEL_NAME
                else:
                    resultText = DEFAULT_PQ_LABEL_NAME
                print("SourceID "+str(sourceID)+" ,Image No: "+str(imageNo)+" IS: "+resultText)

                coefficient, correlation = CalculateImageCoefficient(imageData[0])
                print("Coefficient = ",coefficient)
                test = input()
                test = test.upper()
                if (test=='X'):
                    bFinished=True


def main():
    bAllClassified = False
    bContinueOperation = True

    global coefficientThreshold

    while (bContinueOperation==True):

        selectedOperation = GetOperationMode()

        if (selectedOperation == OP_PROCESS_FITS_FILES):
            ReadFITSReference()

        elif (selectedOperation == OP_MISMATCH):
            CreateMismatchedDetail()

        elif (selectedOperation == OP_DETERMINE_OBJECT):

            w1 = float(input("Enter w1: "))
            w2 = float(input("Enter w2: "))
            w3 = float(input("Enter w3: "))

            possibleObject,rationale = DetermineLikelyWISEObject(0,w1,w2,w3)

            print("rationale = ",rationale)

        elif (selectedOperation == OP_EXIT):

            sys.exit()

        elif (selectedOperation == OP_FIND_IMAGES):
            if (bAllClassified):
                FindImagesForExamination()

            else:
                print("Need to build classification data first")
                bContinueOperation = False


        elif (selectedOperation == OP_ANALYSE_MAN_CLASSES):

            if (bAllClassified):
                AnalyseManualClasses()

            else:
                print("Need to build classification data first")
                bContinueOperation = False


        elif (selectedOperation == OP_RECLASSIFY):

            if (bAllClassified):
                coefficientThreshold = float(input("Input New Coefficient Value:"))
                bAllClassified = ClassifyAllLightCurves()
                CompareAgainstManualResults()


            else:
                print("Need to build classification data first")
                bContinueOperation = False

        elif (selectedOperation == OP_COMPARE_RESULTS):

            if (bAllClassified):
                coefficientThreshold = float(input("Input New Coefficient Value:"))
                CompareAgainstManualResults()

            else:
                print("Need to build classification data first")
                bContinueOperation = False

        elif (selectedOperation == OP_TEST_MODEL):

            if (bAllClassified):

                bValid, model, modelName, precision, recall, f1,scaler = loadModel(RF_BINARY_CLASSIFIER)

                if (bValid):
                    TestBinaryClassifier(model,scaler)

            else:
                print("Need to build classification data first")
          #      bContinueOperation = False

        elif (selectedOperation == OP_BUILD_CLASSIFY_DATA):

            coefficientThreshold = float(input("Enter Poor Quality Coefficient Value: "))

            ReadManualResults()

            ReadStokesFiles()
            ProcessCompleteLCData()
            QueryNVSSData()
            QueryFIRSTData()
            QuerySDSSData()
            QueryMilliquasData()
            QueryPulsarData()
            QueryWISEData()
            QuerySIMBADData()

            bAllClassified = ClassifyAllLightCurves()
            CompareAgainstManualResults()

            StoreBinaryData()

        #    poorQualityFileList,normalQualityFileList = ReadAllImageListFiles()

        elif (selectedOperation == OP_SOAK_TEST):


            lowerCoefficientThreshold = float(input("Enter Lower Quality Coefficient Value: "))
            upperCoefficientThreshold = float(input("Enter Upper Quality Coefficient Value: "))

            ReadManualResults()
            ReadStokesFiles()
            ProcessCompleteLCData()
            QueryWISEData()
            QuerySIMBADData()

            numberCoefficients = int((upperCoefficientThreshold-lowerCoefficientThreshold)/0.1)
            print("no coefficients = ",numberCoefficients)

            coefficientThreshold= lowerCoefficientThreshold

            for coefficientNo in range(numberCoefficients):

                print("Processing for coefficient threshold of ...",coefficientThreshold)
                bAllClassified = ClassifyAllLightCurves()
                CompareAgainstManualResults()

                StoreBinaryData()

                ProcessImageModels()
                coefficientThreshold = coefficientThreshold+0.1


        elif (selectedOperation == OP_INTERROGATE_DATA):

            if (bAllClassified):

                InterrogateLCData()
            else:
                print("Need to build classification data first")
                bContinueOperation=False

        elif (selectedOperation == OP_ANALYSE_IMAGES):

            if (bAllClassified):
                sourceID = input("Input Source ID:")

                fitsImages = GetAllFITSData(sourceID)

                DisplayAllFITSHistograms(sourceID, fitsImages)

            else:
                print("Need to build classification data first")
              #  bContinueOperation = False

        elif (selectedOperation == OP_BUILD_IMAGE_MODELS):

            coefficientThreshold = float(input("Enter Poor Quality Coefficient Value: "))
            ProcessImageModels()


        elif (selectedOperation == OP_STORE_BINARY_DATA):

            if (bAllClassified):
                StoreBinaryData()
            else:
                print("Need to build classification data first")
                bContinueOperation = False

        elif (selectedOperation == OP_LOAD_BINARY_DATA):

            coefficientThreshold = float(input("Enter Poor Quality Coefficient Value: "))

            LoadBinaryData()
            bAllClassified = True




if __name__ == '__main__':
    main()


