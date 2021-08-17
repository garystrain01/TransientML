# TransientML

# This code uses a 1D Convolutional Neural Network to classify sets of lightcurves from a reference lightcurve dataset.
# Reference: MANTRA: A Machine Learning reference lightcurve dataset for astronomical transient event recognition (Neira et al, 2020)

# The configuration parameters are as follows:

# bDebug = False     #  switch debug code on/off
# bGenerateCSVFiles = False # to re-generate individual transient CSV files
# bDisplayLightCurveSizing = False # used to display number of points per light curve (for reference only)
# bDisplayLightCurvePoints = False # for display of test lightcurves
# bRandom = False # generate random lightcurves for comparison
# bDisplayTransientClasses = False # Display All Transient Classes as processed (only for information)

# The following locations must be set:

# DEFAULT_TRAINING_DATA_LOCATION = '/.../'. # where all input data resides 

# REF_LIGHTCURVE_LOCATION = '/.../' £ where all test light curves reside

# REF_LIGHTCURVE_CSV_LOCATION = '/.../' where all individual CSV files reside 


# Defaults for CNN Model

# DEFAULT_VERBOSE_LEVEL = 2
# DEFAULT_BATCH_SIZE = 32
# DEFAULT_KERNEL_SIZE = 3

# For binary classification 
# DEFAULT_LABEL1 = 0
# DEFAULT_LABEL2 = 1

# change these parameters for the transient types to classify

#BINARY_CLASS1 = 'AGN'
#BINARY_CLASS2 = 'CV'


# TRAIN_TEST_RATIO = 0.70  # 70% of the total data should be training
