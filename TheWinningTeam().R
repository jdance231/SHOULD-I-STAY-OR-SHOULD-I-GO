# PRACTICAL BUSINESS ANALYTICS COURSEWORK


# SHOULD I STAY OR SHOULD I GO? 
# UNDERSTANDING EMPLOYEE ATTRTION IN THE WORKPLACE


# TheWinningTeam()

#       Danylo Kovalenko - 6413526
#       Emily Gould - 6660230
#       Jamie Dance - 6661320
#       Jenna Wilkes - 6662507
#       Prakash Jha - 6659329 
#       William Hemsley - 6414699


# MODULE INFORMATION:

#       MSc Data Science
#       COMM053
#       Prof. Nick F Ryman-Tubb
#       Department of Computer Science
#       Semester 1 2020/21
#       University of Surrey
#       GUILDFORD
#       Surrey GU2 7XH


# VERSION HISTORY:

#       1.0 - Initial version
#       1.1 - Removed underscores and periods from columns' names; 
#             Changed all variables' names into upper case 
#       1.2 - Removed all punctuation from columns' names; 
#             Added a setdiff check for EmployeeIDs to make sure there is 
#             no mismatch between the datasets; 
#             Modified the format of categorical variables so that it matches 
#             the data dictionary in order to simplify the process of 
#             visualising the data; 
#             Added a loop to check for columns with less than 2 unique values
#             and remove them
#       1.3 - Expanded time conversion functions to try different time formats 
#             in case times are recorded in some other way;
#             Added some colour to the console output for the ease of 
#             readability 
#       1.4 - Added a check to confirm that the EmployeeID is a primary key for 
#             each dataset; 
#             Expanded the treatments of missing values. They are now removed 
#             only if they do not exceed a certain threshold. Otherwise, they
#             are substituted by a median or a mode, depending on the type of
#             a variable
#       1.5 - Added a condition to check whether there are any duplicates rows
#             and remove them

#*******************************************************************************
# SETTING UP THE SCRIPT
# PREPARE THE ENVIRONMENT

# Clear all objects in the global environment
rm(list = ls())

# Clear the console area
cat('\014')

# Automatically release memory
gc()

# Start random numbers at the same sequence
set.seed(123)

# Clear plots and other graphics in RStudio output
if(!is.null(dev.list())) dev.off()
graphics.off()

# Clear all warning messages
assign("last.warning", NULL, envir = baseenv())

# Print the current working directory
print(paste("WORKING DIRECTORY: ", getwd()))

#*******************************************************************************
# GLOBAL ENVIRONMENT VARIABLES AND CONSTANTS

STANDARDHOURS <- 8                  # Standard working hours
THRESHOLD <- 1                      # A cut-off point for missing values
CORRELATIONTHRESHOLD <- 0.6         # A cut-off point for collinearity
OUTPUT_FIELD <- "ATTRITION"         # Field name of the output class to predict
MANUALREMOVAL <- "PERFORMANCERATING"# A field we manually remove due to the lack 
                                    # of meaningful values (see data dictionary)

OUTLIER_CONF      <- 0.9            # Confidence p-value for outlier detection
DISCRETE_BINS     <- 7              # Number of empty bins to determine discrete
MAX_LITERALS      <- 55             # Maximum number of hotcoded new fields

TYPE_DISCRETE     <- "DISCRETE"     # Field is discrete (numeric)
TYPE_ORDINAL      <- "ORDINAL"      # Field is continuous numeric
TYPE_SYMBOLIC     <- "SYMBOLIC"     # Field is a string
TYPE_NUMERIC      <- "NUMERIC"      # Field is initially a numeric
TYPE_IGNORE       <- "IGNORE"       # Field is not encoded

DATASET1 <- "employee_survey_data.csv" # A dataset with employees' responses
DATASET2 <- "general_data.csv"         # A dataset with some general information
DATASET3 <- "manager_survey_data.csv"  # A dataset with managers' responses 
DATASET4 <- "in_time.csv"              # A dataset with log-in times
DATASET5 <- "out_time.csv"             # A dataset with log-out times 

PRIMARYKEY <- "EmployeeID"          # The expected primary key for the datasets
PLOTSPDF <- "myplots.pdf"           # The pdf filename where initial 
                                    # visualisations will be saved 

# Random forest
FOREST_SIZE       <- 100                  # Number of trees in the forest

# Deep neural network
DEEP_HIDDEN       <- c(4,6)               # Number of neurons in each layer
DEEP_STOPPING     <- 2                    # Number of times no improvement before stop
DEEP_TOLERANCE    <- 0.01                 # Error threshold
DEEP_ACTIVATION   <- "TanhWithDropout"    # Non-linear activation function
DEEP_REPRODUCABLE <- TRUE                 # Set to TRUE to test training is same for each run
BASICNN_EPOCHS    <- 100                  # Maximum number of training epochs

# Shallow neural network
NEURONS           <- 1                    # Number of neurons for neural network
                                          # The code takes a longer time to run
                                          # with increasing number of neurons 
                                          # (9 neurons~40 minutes)  
                                          # NB We used 9 neurons for the 
                                          # report but have set the value to 1  
                                          # to speed up the execution of the code

SHALLOW_ACT       <- "logistic"           # Non-linear activation function
STEPMAX           <- 10e+5                # Maximum iterations of neural network algorithm

# Stratified Cross Validation
KFOLDS            <- 5                    # Number of folded experiments

# Models - must be written exactly as function names
# Options: "RandomForest", "LogisticRegression", "ShallowNeural", "deepNeural"
MYMODELS <- c("RandomForest",
              "LogisticRegression",
              "ShallowNeural",
              "deepNeural")
#,
#"deepNeural"
#*******************************************************************************
# LIBRARIES USED IN THE PROJECT:

# Library from CRAN     Version
# pacman	               0.5.1
# crayon                 1.3.4
# ggplot2                3.3.2
# gridExtra              2.3
# grid                   4.0.2
# lattice                0.20.41
# plyr                   1.8.6
# dplyr                  1.0.2
# outliers	             0.14
# corrplot	             0.84
# MASS	                 7.3.53
# formattable            0.2.0.1
# stats                  4.0.3
# caret                  6.0.86
# PerformanceAnalytics   2.0.4
# caTools                1.18.0
# car                    3.0.10
# ggpubr                 0.4.0
# stringr                1.4.0
# partykit               1.2.10
# C50                    0.1.3.1
# randomForest           4.6.14
# h2o                    3.32.0.1
# neuralnet              1.44.2

MYLIBRARIES<-c("crayon",
               "ggplot2",
               "gridExtra",
               "grid",
               "lattice",
               "plyr",
               "dplyr",
               "outliers",
               "corrplot",
               "MASS",
               "formattable",
               "stats",
               "caret",
               "caTools",
               "car",
               "PerformanceAnalytics",
               "ggpubr",
               "stringr",
               "partykit",
               "C50",
               "randomForest",
               "h2o",
               "neuralnet")

# Use pacman to install various packages and load the libraries
library(pacman)
pacman::p_load(char=MYLIBRARIES,install=TRUE,character.only=TRUE)

#*******************************************************************************
# USER DEFINED FUNCTIONS

# primaryKeyCheck() :
#
# Check whether the PRIMARYKEY is indeed a primary key of a dataset.
# It is if the number of unique IDs is equivalent to the number
# of records in a dataset
#
# INPUT   :   Dataset        
#
# OUTPUT  :   None               
#
#**************************************

primaryKeyCheck <- function(dataset){
  
  cat(yellow("Checking if"), PRIMARYKEY, yellow("is a primary key for"), deparse(substitute(dataset)), yellow("...\n"))
  if (length(unique(dataset[, PRIMARYKEY])) == nrow(dataset)){
    
    return(cat(PRIMARYKEY, green("is a primary key for"), 
               deparse(substitute(dataset)), "\n\n"))
  } else {
    
    return(cat(PRIMARYKEY, red("is not a primary key for"),
               deparse(substitute(dataset)), "\n\n"))
  }
}
# End of primaryKeyCheck()

#*******************************************************************************
# checkAndMerge() :
#
# Check that the primary keys match for each dataset and 
# merge if they do by the PRIMARYKEY
#
# INPUT   :  Output name, dataset1, dataset2        
#
# OUTPUT  :  Dataset (merged)
#
#**************************************

checkAndMerge <- function (output, dataset1, dataset2){
  
  if(sum(setdiff(dataset1[, PRIMARYKEY], dataset2[, PRIMARYKEY])) == 0){
    cat(green("Primary keys match in"),  deparse(substitute(dataset1)), green("and"), deparse(substitute(dataset2))) 
    cat(green("\nDatasets can be merged"), yellow("\nMerging"), deparse(substitute(dataset1)), 
        yellow("and"), deparse(substitute(dataset2)), yellow("..."))
    merged <- merge(dataset1, dataset2, by = PRIMARYKEY)
    cat(green("\nOutput -"), deparse(substitute(output)), "\n\n")
    
    return(merged)
    
  } else {
    cat(red("There is a mismatch in primary keys. Cannot merge\n"),   fill = TRUE, labels = NULL)
  }
}
# End of checkAndMerge()

#*******************************************************************************
# INITIAL DATA AGGREGATION

# prepareDataset() :
#
# Prepare a single dataset - read the csv files, do the necessary 
# manipulations and calculations with the Time datasets, and merge all datasets
#
# INPUT   :   Datasets
#
# OUTPUT  :   Dataset (merged)                        
#
#**************************************

prepareDataset <- function(DATASET1,DATASET2,DATASET3,DATASET4,DATASET5){
  
  cat(yellow("\nPreparing a combined dataset ...\n\n"))
  
  # Read all csv files
  cat(yellow("Reading"), DATASET1, yellow("...\n"))
  employeeSurvey<-read.csv(DATASET1, stringsAsFactors = FALSE)
  cat(DATASET1, green("has been read as"), "employeeSurvey", "\n\n")
  
  cat(yellow("Reading"), DATASET2, yellow("...\n"))
  managerSurvey<-read.csv(DATASET3, stringsAsFactors = FALSE)
  cat(DATASET2, green("has been read as"), "managerSurvey", "\n\n")
  
  cat(yellow("Reading"), DATASET3, yellow("...\n"))
  generalData<-read.csv(DATASET2, stringsAsFactors = FALSE)
  cat(DATASET3, green("has been read as"), "generalData", "\n\n")
  
  cat(yellow("Reading"), DATASET4, yellow("...\n"))
  inTime<-read.csv(DATASET4, stringsAsFactors = FALSE)
  cat(DATASET4, green("has been read as"), "inTime", "\n\n")
  
  cat(yellow("Reading"), DATASET5, yellow("...\n"))
  outTime<-read.csv(DATASET5, stringsAsFactors = FALSE)
  cat(DATASET5, green("has been read as"), "outTime", "\n\n")

  #***********************************
  # Manipulate dates and times to be in the right format and merge 
  #     into a new data frame
  
  # Take a brief look at the type of data in inTime and outTime datasets
  cat(green("Take a brief look inside the"), "inTime", green("and the"), 
            "outTime", green("datasets:\n"), fill = TRUE, labels = NULL)
  cat("inTime", green("dataset:"), fill = TRUE, labels = NULL)
  cat(str(inTime, list.len=5), "\n")
  cat("outTime", green("dataset:"), fill = TRUE, labels = NULL)
  print(str(outTime, list.len = 5))

  # Dates and times are recorded either as characters or logical. 
  # The index (int) X columnis used to record EmployeeID. 
  # Rename X into EmployeeID
  colnames(inTime)[which(colnames(inTime) == "X")] <- PRIMARYKEY
  colnames(outTime)[which(colnames(outTime) == "X")] <- PRIMARYKEY
  
  # Check that employee IDs match for inTime and outTime. If they do, 
  # record EmployeeID from inTime as a separate variable
  if(sum(setdiff(inTime[, PRIMARYKEY], outTime[, PRIMARYKEY])) == 0){
    cat(yellow("\nChecking whether primary keys match for"), "inTime", yellow("and"), "outTime", yellow("...\n"))
    cat(green("Primary keys match for both datasets\n"), 
                                                     fill = TRUE, labels = NULL)
    ID <- inTime[, PRIMARYKEY]
  } else {
    cat(red("Primary keys do not match\n"), fill = TRUE, labels = NULL)
  }
  
  # Temporarily drop EmployeeID for the ease of calculations
  inTime[, PRIMARYKEY] <- NULL
  outTime[, PRIMARYKEY] <- NULL
  
  # Convert time data into POSIXct format
  inTimeNew <- sapply (inTime, function(x) as.POSIXlt(x, origin="1970-01-01",
                                            tryFormats = c("%Y-%m-%d %H:%M:%S",
                                                           "%Y/%m/%d %H:%M:%S",
                                                           "%Y-%m-%d %H:%M",
                                                           "%Y/%m/%d %H:%M",
                                                           "%Y-%m-%d",
                                                           "%Y/%m/%d")))
  inTimeNew <- as.data.frame(inTimeNew)
  
  outTimeNew <- sapply (outTime, function(x) as.POSIXlt(x, origin="1970-01-01",
                                            tryFormats = c("%Y-%m-%d %H:%M:%S",
                                                           "%Y/%m/%d %H:%M:%S",
                                                           "%Y-%m-%d %H:%M",
                                                           "%Y/%m/%d %H:%M",
                                                           "%Y-%m-%d",
                                                           "%Y/%m/%d")))
  outTimeNew <- as.data.frame(outTimeNew)
  
  # Check that all columns in inTimeNew and outTimeNew are the same
  cat(yellow("Checking whether"), "inTime", yellow("and"), "ouTime", yellow("have the same column names ...\n"))
  if (length(setdiff(colnames(inTimeNew), colnames(outTimeNew))) < 1){
    cat("inTimeNew", green("and"), "outTimeNew", green("have the same columns",
              "so can be used to calculate daily hours worked\n"), 
                                                     fill = TRUE, labels = NULL)
  } else{
    cat(red("Column names are different. May lead to wrong calculations\n"), 
                                                     fill = TRUE, labels = NULL)
  }
  
  # Calculating the hours worked (time difference) and rounding
  cat(yellow("Calculating the time difference between the log-in and log-out times ..."), 
                                                     fill = TRUE, labels = NULL)
  timeDifference <- outTimeNew - inTimeNew
  timeDifference <- as.data.frame(lapply(timeDifference, round, digits=2))
  
  # Convert time difference to Numeric
  timeDifference <- sapply(timeDifference,function(x) as.numeric(x))
  timeDifference <- as.data.frame(timeDifference)
  cat(green("The time difference has been successfully calculated\n"), 
                                                     fill = TRUE, labels = NULL)
  
  # Calculate mean working hours for each employee and round; NAs ignored
  cat(yellow("Calculating the mean working hours for each employee ..."), 
                                                     fill = TRUE, labels = NULL)
  meanWorkingHours <- rowMeans(timeDifference, na.rm = TRUE, dims = 1)
  meanWorkingHours <- round(as.data.frame(meanWorkingHours), digits = 2)
  cat(green("The mean working hours have been successfully calculated\n"), 
                                                     fill = TRUE, labels = NULL)
  
  
  # Sort into early logout / regular / overtime by considering STANDARDHOURS
  cat(yellow("Bucketing the mean working hours into the 'overtime', 'regular'",
             "or 'early logout' depending on the standard hours ..."), 
                                                     fill = TRUE, labels = NULL)
  meanWorkingHours$overtime_cat <- ifelse(meanWorkingHours > (STANDARDHOURS + 1),
                                                                "overtime", 
                                   ifelse(meanWorkingHours > (STANDARDHOURS - 1) 
                                   & meanWorkingHours <= (STANDARDHOURS + 1),
                                                     "regular", "early logout"))
  cat(green("The mean working hours have been successfully bucketed\n"), 
                                                     fill = TRUE, labels = NULL)
  
  # Add back the employee index
  meanWorkingHours <- cbind(ID, meanWorkingHours)
  
  # Make sure that the primary key of meanWorkingHours has the same name as 
  # in other datasets
  colnames(meanWorkingHours)[which(colnames(meanWorkingHours) == "ID")] <- PRIMARYKEY
  
  #***********************************
  # Merging the datasets
  # Check that the PRIMARY KEY is indeed a primary key for each dataset
  primaryKeyCheck(employeeSurvey)
  primaryKeyCheck(managerSurvey)
  primaryKeyCheck(generalData)
  primaryKeyCheck(meanWorkingHours)

  # Merge surveys
  mergedSurvey <- checkAndMerge(mergedSurvey, managerSurvey, employeeSurvey)
  # Merge surveys and general data
  mergedData <- checkAndMerge(mergedData, mergedSurvey, generalData)
  # Add mean working hours
  mergedData <- checkAndMerge(mergedData, mergedData, meanWorkingHours)
  
  # Show a list of the field names in the newly created dataset
  print(formattable::formattable(data.frame(field=names(mergedData))))

  # Export merged dataset as a csv file into the working directory
  write.csv(mergedData, "mergedData.csv", row.names = FALSE)
  cat(green("The combined dataset has been successfully prepared and exported as"),
      "'mergedData.csv'", green("into the working directory"), fill = TRUE, labels = NULL)
  
  return(mergedData)
}
# End of prepareDataset()

#*******************************************************************************
# BASIC DATA EXPLORATION

# basicDataExploration() :
#
# Do basic data explorations - get some summary statistics, 
# check for any missing values and find their percentage
#
# INPUT   :   Dataset      
#
# OUTPUT  :   None                       
#
#**************************************

basicDataExploration <- function(dataset) {
  
  cat(green("\nExplore the"), deparse(substitute(dataset)), green("dataset:\n"), 
                                                     fill = TRUE, labels = NULL)
  
  # Output the names of the columns
  cat(green("These are the names of the columns:"),  
                                                     fill = TRUE, labels = NULL)
  print(names(dataset))

  # Output the dimensionality of the dataset
  cat(green("\nThe dimensions of the dataset:"),     
                                                     fill = TRUE, labels = NULL)
  print(dim(dataset))

  # Look at the dataset using head and tail functions
  cat(green("\nThese are the first 3 rows of the dataset:"), 
                                                     fill = TRUE, labels = NULL)
  print(head(dataset, n=3))
  cat(green("\nThese are the last 3 rows of the dataset:"), 
                                                     fill = TRUE, labels = NULL)
  print(tail(dataset, n=3))

  # Summary statistics for each variable in the dataset
  cat(green("\nHere is some brief summary for each variable:"), 
                                                     fill = TRUE, labels = NULL)
  print(summary(dataset))

  # See some in depth information about the dataset
  cat(green("Here is some more in depth information for each variable:"), 
                                                     fill = TRUE, labels = NULL)
  print(str(dataset))

  # Identify numeric columns and check whether they are as expected
  numericData <- select_if(dataset, is.numeric)
  cat(green("\nThere are"), length(names(numericData)), green("numeric columns:"),
                                                     fill = TRUE, labels = NULL)
  print(names(numericData))

  # Look at quantiles of quantitative variables for boxplot analysis; NAs ignored
  cat(green("\nHere are the quantiles of quantitative variables:"), 
                                                     fill = TRUE, labels = NULL)
  for (i in 1:ncol(numericData)){
    if (colnames(numericData[i]) != PRIMARYKEY){
    cat(cyan(colnames(numericData[i])),  fill = TRUE, labels = NULL)
    print(quantile(numericData[i], na.rm = TRUE))
    }
  }

  # Identify categorical columns
  categoricalData <- select_if(dataset, is.character)
  cat(green("\nThere are"), length(names(categoricalData)), 
                      green("categorical columns:"), fill = TRUE, labels = NULL)
  print(names(categoricalData))

  # Make contingency tables for variables of interest; omit those with too many 
  # unique values to make the output more readable
  cat(green("\nContingency tables for"), OUTPUT_FIELD, 
                      green("VS other variables:"), fill = TRUE, labels = NULL)
  cat(green("NB Variables with more than 60 unique values are not displayed"),
                                                     fill = TRUE, labels = NULL)
  for (i in colnames(dataset)){
  
    if ((colnames(dataset[i]) != PRIMARYKEY) & 
        (colnames(dataset[i]) != OUTPUT_FIELD) & 
        (count(unique(dataset[i])) < 61 )){
    
      cat(cyan(colnames(dataset[i])))
      print(table(dataset$Attrition, dataset[,i], useNA = "ifany", 
                                    dnn = c("Attrition", names(dataset[i]))))
    }
  }

  # Check for missing values
  cat(green("\nTotal number of missing values:"), sum(colSums(is.na(dataset))),
                                          sep = " ", fill = TRUE, labels = NULL)
  print(colSums(is.na(dataset)))

  # Find the percentage of data missing in the dataset
  cat(green("\nThe percentage of data missing in the dataset:"), 
                          (((sum((is.na(dataset)))) / 
                           ((nrow(dataset) * ncol(dataset)))) * 100), "%", 
                                                     fill = TRUE, labels = NULL)
}
# End of basicDataExploration()

#*******************************************************************************
# INITIAL DATA CLEANING (readability, uniqueness and missing values)

# getmode() :
#
# Calculate the mode of a column
#
# INPUT   :   Column      
#
# OUTPUT  :   Mode               - numeric        - character
#
#**************************************

# Create the function to obtain the mode
# Taken from https://www.tutorialspoint.com/r/r_mean_median_mode.htm
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
# End of getmode()

#*******************************************************************************
# initialDataCleaning() :
# 
# Data Cleaning - remove punctuation, duplicates, NAs, redundancy
#
# INPUT   :   Dataset
#
# OUTPUT  :   Dataset                        
# 
#**************************************

initialDataCleaning <- function (dataset){
  
  cat(yellow("\nCommencing intial data cleaning ...\n"), fill = TRUE, labels = NULL)
 
  # Make the field names readable by capitalising and removing punctuation
  # Change all column names to uppercase
  cat(yellow("Changing all column names into uppercase and removing any punctuation ...\n"))
  names(dataset) <- toupper(names(dataset))

  # Remove all punctuation characters in names
  names(dataset) <- gsub("[[:punct:][:blank:]]","", names(dataset))
  
  cat(green("Done\n"))
  
  cat(yellow("\nRemoving any duplicated rows ...\n"))
  # Check for duplicate rows and remove if there are any
  cat(green("There are"), (nrow(dataset) - nrow(unique(dataset))), 
      green("duplicated rows"), sep = " ", 
                                                     fill = TRUE, labels = NULL)
  if ((nrow(dataset) - nrow(unique(dataset))) > 0) {
    dataset <- dataset[!duplicated(dataset),]
    cat(green("All duplicated rows have been removed"),  
                                                     fill = TRUE, labels = NULL)
  }

  # Check for unique values in each column and remove columns which do not have unique values
  cat(yellow("\nRemoving columns that do not have enough unique values ...\n"))
  redundantColumns <- vector(mode = "list")
  n <- 1
  for (i in 1:ncol(dataset)) {
    if (count(distinct((dataset[i]))) < 2) {
      redundantColumns[n] <- names(dataset[i])
      n <- n + 1
    }
  }
  for (n in redundantColumns){
    cat(green("Column"), colnames(dataset[n]), 
        green("removed due to the lack of unique values"), 
                                                     fill = TRUE, labels = NULL)
    dataset[n] <- NULL
  }

  # Remove performance rating due to the lack of unique values of meaning 
  # (see data dictionary)
  cat(yellow("Removing a column specified in"), "MANUALREMOVAL", yellow("...\n"))
  dataset[MANUALREMOVAL] <- NULL
  cat(MANUALREMOVAL, green(" has been removed\n"), fill = TRUE, labels = NULL)

  # Finding the number of missing values in the data
  # Check what columns contain missing value and what their proportion is
  cat(yellow("Dealing with missing values ...\n"))
  for (i in 1:ncol(dataset)){
    if (any(is.na(dataset[i])) == TRUE){
      cat(green("Missing values found in:"), colnames(dataset[i]), sep = " ", 
                                                     fill = TRUE, labels = NULL)
      cat(green("Nulls as percentage of the total:"), 
         (mean(is.na(dataset[i])) * 100), "%", sep = " ", 
                                                     fill = TRUE, labels = NULL)
    }
  }
  
  # Deal with missing values
  # If they account for less that 1% of data entries in the variable - remove. 
  # If they account for more than 1% and are in a
  # numeric column - substitute by the median. Truncate it to avoid potential 
  # conflicts. If they account for more than 1% and are in a categorical 
  # column - substitute by the mode.
  for (i in 1:ncol(dataset)){
    if((any(is.na(dataset[i])) == TRUE) & 
      ((mean(is.na(dataset[i])) * 100) < THRESHOLD) ) {
      cat(green("Missing values removed from"), names(dataset[i]), 
                                                     fill = TRUE, labels = NULL)
      dataset <- dataset[!is.na(dataset[i]),]

    } else if ((any(is.na(dataset[i])) == TRUE) & 
              ((mean(is.na(dataset[i])) * 100) > THRESHOLD) & 
              is.numeric(dataset[,i]) == TRUE) {
      dataset[i][is.na(dataset[i])] <- trunc(median(dataset[,i],
                                                                  na.rm = TRUE))
      cat(green("Missing values replaced by median in"), names(dataset[i]), 
                                                     fill = TRUE, labels = NULL)

    } else if ((any(is.na(dataset[i])) == TRUE) & 
              ((mean(is.na(dataset[i])) * 100) > THRESHOLD) &
              is.numeric(dataset[,i]) == FALSE) {
      cat(green("Missing values replaced by mode in"), names(dataset[i]), 
                                                     fill = TRUE, labels = NULL)
      dataset[i][is.na(dataset[i])] <- c(getmode(dataset[,i]))

    } else if (any(is.na(dataset[i])) == TRUE) {
      cat(red("Missing values could not be removed from"), names(dataset[i]), 
                                                     fill = TRUE, labels = NULL)
    }
  }
  cat(green("All missing values that account for less than"), 
      THRESHOLD, "%", green("of data entries for a column have been removed"), 
                                                    fill = TRUE, labels = NULL)
  cat(green("Those, that account for more than"),
      THRESHOLD, "%", green("and belong to numeric columns have", 
                       "been replaced by the mean value"), 
                                                    fill = TRUE, labels = NULL)
  cat(green("Those, that accound for more than"), 
      THRESHOLD, "%", green("and belong to categorical columns have",
                       "been replaced by the modal value"), 
                                                    fill = TRUE, labels = NULL)

  # Check that all missing values have been dealt with
  if (all(colSums(is.na(dataset))==0) == TRUE){
    cat(green("No missing values left"), 
                                                     fill = TRUE, labels = NULL)
  } else {
    cat(red("There are some missing values left"),  
                                                     fill = TRUE, labels = NULL)
  }
  cat(green("\nInitial data cleaning has been completed\n"))
  
  return(dataset)
}
# End of initialDataCleaning()

#*******************************************************************************
# INITIAL DATA VISUALISATION

# convertFormat() :
#
# Modify the format of categorical variables for graphical visualisations 
# so that it matches the one in the data dictionary
#
# INPUT   :   Dataset       
#
# OUTPUT  :   Dataset                         
#
#**************************************

convertFormat <- function(dataset){

  cat(yellow("\nModifying the format of categorical variables to", 
             "match the data dictionary ..."), fill = TRUE, labels = NULL)
  # Education
  dataset$EDUCATION<-as.factor(dataset$EDUCATION)
  dataset$EDUCATION<-revalue(dataset$EDUCATION,
                             c("1" = "Below College", "2" = "College","3" = "Bachelor", "4" = "Master","5"="Doctor"))
  
  # Environment Satisfaction
  dataset$ENVIRONMENTSATISFACTION<-as.factor(dataset$ENVIRONMENTSATISFACTION)
  dataset$ENVIRONMENTSATISFACTION<-revalue(dataset$ENVIRONMENTSATISFACTION,
                             c("1" = "Low", "2" = "Medium","3" = "High", "4" = "Very High"))
  
  # Job Involvement
  dataset$JOBINVOLVEMENT<-as.factor(dataset$JOBINVOLVEMENT)
  dataset$JOBINVOLVEMENT<-revalue(dataset$JOBINVOLVEMENT,
                             c("1" = "Low", "2" = "Medium","3" = "High", "4" = "Very High"))
  
  # Job Satisfaction
  dataset$JOBSATISFACTION<-as.factor(dataset$JOBSATISFACTION)
  dataset$JOBSATISFACTION<-revalue(dataset$JOBSATISFACTION,
                             c("1" = "Low", "2" = "Medium","3" = "High", "4" = "Very High"))
  
  # Work Life Balance
  dataset$WORKLIFEBALANCE<-as.factor(dataset$WORKLIFEBALANCE)
  dataset$WORKLIFEBALANCE<-revalue(dataset$WORKLIFEBALANCE,
                             c("1" = "Bad", "2" = "Good","3" = "Better", "4" = "Best"))
  cat(green("Done"), fill = TRUE, labels = NULL)
  
  return(dataset)
}
# End of convertFormat()

#*******************************************************************************
# updateNumericData() :
#
# Check for the numerical fields in the dataset and print them
#
# INPUT   :   Dataset
#
# OUTPUT  :   Dataset with numerical fields                       
#
#**************************************

updateNumericData <- function (dataset) {

  cat(yellow("\nRecording numeric data ..."), fill = TRUE, labels = NULL)
  numericData <- select_if(dataset, is.numeric)
  cat(green("There are now"), length(names(numericData)), 
      green("numeric columns:"), fill = TRUE, labels = NULL)
  print(names(numericData))
  
  return(numericData)
}
# End of updateNumericData ()

#*******************************************************************************
# updateCategoricalData() :
# 
# Checks for the categorical fields in the dataset and print them 
#
# INPUT   :   Dataset
#
# OUTPUT  :   Dataset with categorical fields                       
#
#**************************************

updateCategoricalData <- function (dataset) {
  
  cat(yellow("\nRecording categorical data ..."), fill = TRUE, labels = NULL)
  categoricalData <- select_if(dataset, is.character)
  cat(green("There are now"), length(names(categoricalData)), 
      green("categorical columns:"), fill = TRUE, labels = NULL)
  print(names(categoricalData))
  
  return(categoricalData)
}
# End of updateCategoricalData()

#*******************************************************************************
# visualisations() :
#
# Produce a number of plots to explore the dataset
#
# INPUT   :   Dataset
#
# OUTPUT  :   PDF document which is saved in the work space directory                            
#
#**************************************

visualisations <- function(dataset){
  
  cat(yellow("\nProducing initial visualisations ..."), fill = TRUE, labels = NULL)
  
  # 1
  dp1 <- ggplot(dataset, aes(AGE, fill = ATTRITION)) +
    geom_density(alpha = 0.5) +
    ggtitle("Age")
  dp2 <- ggplot(dataset, aes(DISTANCEFROMHOME, fill = ATTRITION)) +
    geom_density(alpha = 0.5) +
    ggtitle("Distance From Home")
  
  dp3<-ggplot(dataset, aes(TOTALWORKINGYEARS, fill = ATTRITION)) +
    geom_density(alpha = 0.5) +
    ggtitle("Total Working Years")
  
  var1<- grid.arrange(dp1, dp2,dp3)
  
  # 2
  # Plots of numerical variables
  num1 <- ggplot(dataset, aes(YEARSATCOMPANY, colour = ATTRITION)) +
    geom_density()
  
  num2 <- ggplot(dataset, aes(PERCENTSALARYHIKE, colour = ATTRITION)) +
    geom_density()
  num3 <- ggplot(dataset, aes(TOTALWORKINGYEARS, colour = ATTRITION)) +
    geom_density()
  num4 <- ggplot(dataset, aes(DISTANCEFROMHOME, colour = ATTRITION)) +
    geom_density()
  var2<-grid.arrange(num1, num2,num3,num4)
  
  # 3
  # Monthlyincome with attrition
  density1 <- ggplot(dataset,color=ATTRITION,
                     aes(x = MONTHLYINCOME,
                         fill = ATTRITION)) +
    geom_density(alpha = 0.7) +
    labs(title = "Salary Vs Attrition")
  
  # Age vs attrition
  density2 <- ggplot(dataset,
                     aes(x = AGE,color=ATTRITION,
                         fill = ATTRITION)) +
    geom_density(alpha = 0.7) +
    labs(title = "Age vs Attrition")
  
  var3<-grid.arrange(density1, density2, nrow = 2)
  
  # 4
  a1<-ggplot(dataset, aes(x = ATTRITION, y = PERCENTSALARYHIKE, fill = ATTRITION)) + geom_boxplot() +
    facet_wrap(~ BUSINESSTRAVEL, ncol = 5)
  a2<-ggplot(dataset, aes(x = ATTRITION, y = PERCENTSALARYHIKE, fill = ATTRITION)) + geom_boxplot() +
    facet_wrap(~ MARITALSTATUS, ncol = 5)
  var4<-grid.arrange(a1,a2,nrow=2)
  
  # 5
  # Categorical
  ab1<-ggplot(data=dataset)+
    geom_bar(mapping=aes(ENVIRONMENTSATISFACTION),fill='#FF6666', width=0.5)+
    labs(title="Environment Satisfaction", subtitle="From 4410 employees", x="Environment Satisfaction Level by Department",
         y="Number of Employees", caption="HR Analytics") + facet_wrap(~ ATTRITION)
  
  ab2<-ggplot(data=dataset)+
    geom_bar(mapping=aes(JOBSATISFACTION),fill='#FF6666', width=0.5)+
    labs(title="Job Satisfaction", subtitle="From 4410 employees", x="Job Satisfaction Level", y="Number of Employees",
         caption="HR Analytics") + facet_wrap(~ ATTRITION)
  var5<-grid.arrange(ab1,ab2)
  
  ab3<-ggplot(data=dataset)+
    geom_bar(mapping=aes(WORKLIFEBALANCE),fill='#FF6666', width=0.5)+
    labs(title="Work life balance", subtitle="From 441O employees", x="Job Satisfaction Level", y="Number of Employees",
         caption="HR Analytics") + facet_wrap(~ ATTRITION)
  
  ab4 <-ggplot(data=dataset)+
    geom_bar(mapping=aes(JOBINVOLVEMENT),fill='#FF6666', width=0.5)+
    labs(title="Job involvement", subtitle="From 4410 employees", x="Job Involvement", y="Number of Employees",
         caption="HR Analytics") + facet_wrap(~ ATTRITION)
  var6<- grid.arrange(ab3,ab4)
  
  # 6
  k1<-ggplot(dataset)+
    geom_bar(position="dodge",mapping=aes(JOBSATISFACTION,fill=ATTRITION), width = 0.7) + 
    labs(title="Job Satisfaction Vs Attrition", x="Job Satisfaction", y="Number of Employees", caption="HR Analytics")
  k2<-ggplot(dataset)+
    geom_bar(position="dodge",mapping=aes(BUSINESSTRAVEL,fill=ATTRITION), width = 0.7) + 
    labs(title="Business Travel Vs Attrition", x="Business Travel", y="Number of Employees", caption="HR Analytics")
  k3<-ggplot(dataset)+
    geom_bar(position="dodge",mapping=aes(MARITALSTATUS,fill=ATTRITION), width = 0.7) + 
    labs(title="Marital Status Vs Attrition", x="Marital Status", y="Number of Employees", caption="HR Analytics")
  k4<-ggplot(dataset)+
    geom_bar(position="dodge",mapping=aes(WORKLIFEBALANCE,fill=ATTRITION), width = 0.7) + 
    labs(title="Work-Life balance Vs Attrition", x="Work Life Balance", y="Number of Employees", caption="HR Analytics")
  var7<- grid.arrange(k1,k2,k3,k4)
  
  # 7 
  dataset %>%
    count(ATTRITION) %>%
    mutate(Percentage = n / nrow(dataset)) -> datasetperc
  
  b1<-ggplot(datasetperc, aes(x = ATTRITION, y = Percentage, fill = ATTRITION)) + 
    geom_bar(stat = "identity", width = 0.5) +
    labs(title = "Attrition",
         caption = "HR Analytics",
         y= "Percentage")
  var8<- grid.arrange(b1)
  remove(datasetperc)
  
  #8
  ATTRITIONYearsManager <- dataset[,c("ATTRITION", "YEARSWITHCURRMANAGER")]
  
  meanYears<- aggregate(.~ATTRITION, ATTRITIONYearsManager,mean)
  
  year1<-ggplot(meanYears, aes(x = ATTRITION, y = YEARSWITHCURRMANAGER, fill = ATTRITION, label = YEARSWITHCURRMANAGER)) + 
    geom_bar(stat = "identity", width = 0.5) +
    geom_text(size = 4, position = position_stack(vjust = 0.5)) +
    labs(title = "Average Amount of years with Current Manager by Attrition",
         caption = "HR Analytics",
         y= "Average Number of Years")
  remove(ATTRITIONYearsManager)
  
  var9<- grid.arrange(year1)
  
  ATTRITIONYearsPromotion <- dataset[,c("ATTRITION", "YEARSSINCELASTPROMOTION")]
  
  meanYears<- aggregate(.~ATTRITION, ATTRITIONYearsPromotion,mean)
  
  year2<-ggplot(meanYears, aes(x = ATTRITION, y = YEARSSINCELASTPROMOTION, fill = ATTRITION, label = YEARSSINCELASTPROMOTION)) + 
    geom_bar(stat = "identity", width = 0.5) +
    geom_text(size = 4, position = position_stack(vjust = 0.5)) +
    labs(title = "Average Amount of years since last Promotion by Attrition",
         caption = "HR Analytics",
         y= "Average Number of Years")
  remove(ATTRITIONYearsPromotion)
  
  var10<- grid.arrange(year2)
  
  MyPlots = list(var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
  multipleplots <- ggarrange(var1, var2,var3,var4,var5,var6,var7,var8,var9,var10, nrow=1, ncol=1) # for one plot per page
  ggexport(multipleplots, filename=PLOTSPDF)
  
  cat(green("Done"), fill = TRUE, labels = NULL)
}
# End of visualisations()

#*******************************************************************************
# PREPROCESSING THE DATA

# convertToNumerical() :
#
# Convert values that have a meaningful order back to 
# numerical to simplify further analysis
#
# INPUT   :   Dataset
#
# OUTPUT  :   Dataset                   
#
#**************************************

# Convert the data that was changed to match the data dictionary back to numerical
convertToNumerical <- function(dataset){
  
  cat(yellow("\nModifying the format of categorical variables back to the", 
             "original to simplify further analysis ..."), 
                                                    fill = TRUE, labels = NULL)
  #Education
  dataset$EDUCATION<-revalue(dataset$EDUCATION,
                                c("Below College" = "1", "College" = "2","Bachelor" = "3", "Master" = "4","Doctor"="5"))
  dataset$EDUCATION <- sapply(dataset$EDUCATION, 
                                                      function(x) as.numeric(x))
  
  #Environment Satisfaction
  dataset$ENVIRONMENTSATISFACTION<-revalue(dataset$ENVIRONMENTSATISFACTION,
                                 c("Low" = "1", "Medium" = "2","High" = "3", "Very High" = "4"))
  
  dataset$ENVIRONMENTSATISFACTION <- sapply(dataset$ENVIRONMENTSATISFACTION,
                                                      function(x) as.numeric(x))
  
  #Job Involvement
  dataset$JOBINVOLVEMENT<-revalue(dataset$JOBINVOLVEMENT,
                                  c("Low" = "1", "Medium" = "2","High" = "3", "Very High" = "4"))
  
  dataset$JOBINVOLVEMENT <- sapply(dataset$JOBINVOLVEMENT, 
                                                      function(x) as.numeric(x))
  
  #Job Satisfaction
  dataset$JOBSATISFACTION<-revalue(dataset$JOBSATISFACTION,
                                   c("Low" = "1", "Medium" = "2","High" = "3", "Very High" = "4"))
  dataset$JOBSATISFACTION <- sapply(dataset$JOBSATISFACTION,
                                       function(x) as.numeric(x))
  
  #Work Life Balance
  dataset$WORKLIFEBALANCE<-revalue(dataset$WORKLIFEBALANCE,
                                   c("Bad" = "1", "Good" = "2","Better" = "3", "Best" = "4"))
  dataset$WORKLIFEBALANCE <- sapply(dataset$WORKLIFEBALANCE,
                                       function(x) as.numeric(x))
  
  # Convert Attrition (dependent variable) into binary (1 or 0 values)
  # This is because the dependent variable has only 2 outcomes
  # Therefore we have a binary classification problem
  dataset$ATTRITION <- ifelse(dataset$ATTRITION == "No" ,0,1)
  
  dataset$ATTRITION <- sapply(dataset$ATTRITION, 
                              function(x) as.numeric(x))
  
  cat(green("Done"), fill = TRUE, labels = NULL)
  
  return(dataset)
}
# End of convertToNumerical() 

# Taken from lab3DataPrep.R
#*******************************************************************************
# NPREPROCESSING_prettyDataset()
# Output simple dataset field analysis results as a table in "Viewer"
#
# INPUT: data frame    - dataset, full dataset used for train/test
#                      - Each row is one record, each column in named
#                      - Values are not scaled or encoded
#        String - OPTIONAL string which is used in table as a header
#
# OUTPUT : none
#
# Requires : PerformanceAnalytics
#            formattable
#********************************************************

NPREPROCESSING_prettyDataset<-function(dataset,...){
  
  params <- list(...)
  
  tidyTable<-data.frame(Field=names(dataset),
                        Catagorical=FALSE,
                        Symbols=0,
                        Name=0,
                        Min=0.0,
                        Mean=0.0,
                        Max=0.0,
                        Skew=0.0,
                        stringsAsFactors = FALSE)
  
  if (length(params)>0){
    names(tidyTable)[1]<-params[1]
  }
  
  for (i in 1:ncol(dataset)){
    isFieldAfactor<-!is.numeric(dataset[,i])
    tidyTable$Catagorical[i]<-isFieldAfactor
    if (isFieldAfactor){
      # Number of symbols in categorical
      tidyTable$Symbols[i]<-length(unique(dataset[,i]))  
      
      # Gets the count of each unique symbol
      symbolTable<-sapply(unique(dataset[,i]),function(x) length(which(dataset[,i]==x)))
      majoritySymbolPC<-round((sort(symbolTable,decreasing = TRUE)[1]/nrow(dataset))*100,digits=0)
      tidyTable$Name[i]<-paste(names(majoritySymbolPC),"(",majoritySymbolPC,"%)",sep="")
    } else
    {
      tidyTable$Max[i]<-round(max(dataset[,i]),2)
      tidyTable$Mean[i]<-round(mean(dataset[,i]),2)
      tidyTable$Min[i]<-round(min(dataset[,i]),2)
      tidyTable$Skew[i]<-round(PerformanceAnalytics::skewness(dataset[,i],
                                                              method="moment"),2)
    }
  }
  
  # Sort table so that all numerics are first
  t<-formattable::formattable(tidyTable[order(tidyTable$Catagorical),],
                              list(Catagorical = formatter("span", style = x ~ style(color = ifelse(x,"green", "red")),
                                                           x ~ icontext(ifelse(x, "ok", "remove"), ifelse(x, "Yes", "No"))),
                                   Symbols = formatter("span",style = x ~ style(color = "black"),x ~ ifelse(x==0,"-",sprintf("%d", x))),
                                   Min = formatter("span",style = x ~ style(color = "black"), ~ ifelse(Catagorical,"-",format(Min, nsmall=2, big.mark=","))),
                                   Mean = formatter("span",style = x ~ style(color = "black"),~ ifelse(Catagorical,"-",format(Mean, nsmall=2, big.mark=","))),
                                   Max = formatter("span",style = x ~ style(color = "black"), ~ ifelse(Catagorical,"-",format(Max, nsmall=2, big.mark=","))),
                                   Skew = formatter("span",style = x ~ style(color = "black"),~ ifelse(Catagorical,"-",sprintf("%.2f", Skew)))
                              ))
  print(t)
}
# End of NPREPROCESSING_prettyDataset()

#*******************************************************************************
# Pre-Processing a Dataset functions

# To manually set a field type
# This will store $name=field name, $type=field type
manualTypes <- data.frame()   

# Taken from lab3DataPrep.R
#*******************************************************************************
# NPREPROCESSING_setInitialFieldType() :
#
# Set  each field for NUMERIC or SYMBOLIC
#
# INPUT:
#        String - name - name of the field to manually set
#        String - type - manual type
#
# OUTPUT : None
#**************************************

NPREPROCESSING_setInitialFieldType<-function(name,type){
  
  # Sets in the global environment
  manualTypes<-rbind(manualTypes,data.frame(name=name,type=type,
                                            stringsAsFactors = FALSE))
}
# End of NPREPROCESSING_setInitialFieldType()

# Taken from lab3DataPrep.R
#*******************************************************************************
# NPREPROCESSING_initialFieldType() :
#
# Test each field for NUMERIC or SYMBOLIC
#
# INPUT: Data Frame - dataset - data
#
# OUTPUT : Vector - Vector of types 

#**************************************

NPREPROCESSING_initialFieldType<-function(dataset){
  
  field_types<-vector()
  for(field in 1:(ncol(dataset))){
    
    entry<-which(manualTypes$name==names(dataset)[field])
    if (length(entry)>0){
      field_types[field]<-manualTypes$type[entry]
      next
    }
    
    if (is.numeric(dataset[,field])) {
      field_types[field]<-TYPE_NUMERIC
    }
    else {
      field_types[field]<-TYPE_SYMBOLIC
    }
  }
  
  return(field_types)
}
# End of NPREPROCESSING_initialFieldType() 

# Taken from lab3DataPrep.R
# ALTERED:
# 1. Added a condition not to test for PRIMARYKEY 
#*******************************************************************************
# NPREPROCESSING_discreteNumeric() :  
#
# Test NUMERIC field if DISCRETE or ORDINAL
#
# INPUT: data frame      - dataset     - input data
#        vector strings  - field_types - Types per field, either {NUMERIC, SYMBOLIC}
#        int             - cutoff      - Number of empty bins needed to determine discrete (1-10)
#
# OUTPUT : vector strings - Updated with types per field {DISCRETE, ORDINAL}

# Uses histogram
# Plots histogram for visualisation
#**************************************

NPREPROCESSING_discreteNumeric<-function(dataset,field_types,cutoff){ 
  
  # For every field in our dataset
  for(field in 1:(ncol(dataset))){
    
    # Only for fields that are all numeric
    if ((field_types[field]==TYPE_NUMERIC) & 
        colnames(dataset[field]) != PRIMARYKEY) {
      
      # Scale the whole field (column) to between 0 and 1
      scaled_column<-Nrescale(dataset[,field])
      
      # Generate the "cutoff" points for each of 10 bins
      # so we will get 0-0.1, 0.1-0.2...0.9-1.0
      cutpoints<-seq(0,1,length=11)
      
      # This creates an empty vector that will hold the counts of the numbers 
      # in the bin range
      bins<-vector()
      
      # Now we count how many numbers fall within the range
      # length(...) is used to count the numbers that fall within the conditional
      for (i in 2:11){
        bins<-append(bins,length(scaled_column[(scaled_column<=cutpoints[i]) & 
                                                 (scaled_column>cutpoints[i-1])]))
      }
      
      # the 10 bins will have a % value of the count (i.e. density)
      bins<-(bins/length(scaled_column))*100.0
      
      graphTitle<-"AUTO:"
      
      # If the number of bins with less than 1% of the values is greater 
      # than the cutoff
      # then the field is determined to be a discrete value
      
      if (length(which(bins<1.0))>cutoff)
        field_types[field]<-TYPE_DISCRETE
      else
        field_types[field]<-TYPE_ORDINAL
      
      # Bar chart helps with visulisation. Type of field is the chart name
      barplot(bins, main=paste(graphTitle,field_types[field]),
              xlab=names(dataset[field]),
              names.arg = 1:10,bty="n")
      
    } # endif numeric types
  } # endof for
  
  return(field_types)
}
# End of NPREPROCESSING_discreteNumeric() 

# Taken from lab3DataPrep.R
#*******************************************************************************
# Nrescale() :
#
# These are the real values, that we scale between 0-1
# i.e. x-min / (max-min)
#
# INPUT:   vector - input - values to scale
#
# OUTPUT : vector - scaled values to [0.0,1.0]
#**************************************

Nrescale<-function(input){
  
  minv<-min(input)
  maxv<-max(input)
  
  return((input-minv)/(maxv-minv))
}
# End of Nrescale() 

# Taken from lab3DataPrep.R
#*******************************************************************************
# Nrescaleentireframe() :
#
# Rescle the entire dataframe to [0.0,1.0]
#
# INPUT:   data frame - dataset - numeric data frame
#
# OUTPUT : data frame - scaled numeric data frame
#**************************************

Nrescaleentireframe<-function(dataset){
  
  scaled<-sapply(as.data.frame(dataset),Nrescale)
  
  return(scaled)
}
# End of Nrescaleentireframe()

# Taken from lab3DataPrep.R
#*******************************************************************************
# NPREPROCESSING_outlier() :
#
# Determine if a value of a record is an outlier for each field
#
# INPUT:   data frame - ordinals   - numeric fields only
#          double     - confidence - Confidence above which is determined an outlier [0,1]
#                                  - Set to negative Confidence if NOT remove outliers
#
# OUTPUT : data frame - ordinals with any outlier values replaced with the median of the field

# ChiSquared method
# Uses   library(outliers)
# https://cran.r-project.org/web/packages/outliers/outliers.pdf
#**************************************

NPREPROCESSING_outlier<-function(ordinals,confidence){
  
  # For every ordinal field in our dataset
  for(field in 1:(ncol(ordinals))){
    
    sorted<-unique(sort(ordinals[,field],decreasing=TRUE))
    outliers<-which(outliers::scores(sorted,type="chisq",prob=abs(confidence)))
    NplotOutliers(sorted,outliers,colnames(ordinals)[field])
    
    # If found records with outlier values
    if ((length(outliers>0))){
      
      # If confidence is positive then replace values with their means,
      # otherwise do nothing
      if (confidence>0){
        outliersGone<-rm.outlier(ordinals[,field],fill=TRUE)
        sorted<-unique(sort(outliersGone,decreasing=TRUE))
        # NplotOutliers(sorted,vector(),colnames(ordinals)[field])
        #Put in the values with the outliers replaced by means
        ordinals[,field]<-outliersGone 
        print(paste("Outlier field =",names(ordinals)[field],"Records =",
                    length(outliers),"Replaced with MEAN"))
      } else {
        print(paste("Outlier field =",names(ordinals)[field],"Records =",
                    length(outliers)))
      }
    }
  }
  
  return(ordinals)
}
# End of NPREPROCESSING_outlier()

# Taken from lab3DataPrep.R
#*******************************************************************************
# NplotOutliers() :
#
# Scatter plot of field values and colours outliers in red
#
# INPUT: Vector - sorted    -  points to plot as literal values
#        Vector - outliers  - list of above points that are considered outliers
#        String - fieldName - name of field to plot
#
# OUTPUT : None
#**************************************

NplotOutliers<-function(sorted,outliers,fieldName){
  
  plot(1:length(sorted),sorted,pch=1,xlab="Unique records", 
       ylab=paste("Sorted values",fieldName),bty="n")
  if (length(outliers)>0)
    points(outliers,sorted[outliers],col="red",pch=19)
}
# End of NplotOutliers ()

# Taken from lab3DataPrep.R
#*******************************************************************************
# NPREPROCESSING_categorical() :
#
# Transform SYMBOLIC or DISCRETE fields using 1-hot-encoding
#
# INPUT: data frame    - dataset      - symbolic fields
#        vector string - field_types  - types per field {ORDINAL, SYMBOLIC, 
#                                                                      DISCRETE}
#
# OUTPUT : data frame    - transformed dataset

# Small number of literals only otherwise too many dimensions
# Uses 1-hot-encoding if more than 2 unique literals in the field
# Otherwise converts the 2 literals into one field of {0,1}
#**************************************

NPREPROCESSING_categorical<-function(dataset,field_types){
  
  # This is a dataframe of the transformed categorical fields
  catagorical<-data.frame(first=rep(NA,nrow(dataset)),stringsAsFactors=FALSE)
  
  # For every field in our dataset
  for(field in 1:(ncol(dataset))){
    
    # Only for fields marked SYMBOLIC or DISCRETE
    if ((field_types[field]==TYPE_SYMBOLIC)||(field_types[field]==TYPE_DISCRETE)) {
      
      # Create a list of unique values in the field (each is a literal)
      literals<-as.vector(unique(dataset[,field]))
      numberLiterals<-length(literals)
      
      # if there are just two literals in the field we can convert to 0 and 1
      if (numberLiterals==2){
        transformed<-ifelse (dataset[,field]==literals[1],0.0,1.0)
        catagorical<-cbind(catagorical,transformed)
        colnames(catagorical)[ncol(catagorical)]<-colnames(dataset)[field]
        
      } else
      {
        # We have now to one-hot encoding FOR SMALL NUMBER of literals
        if (numberLiterals<=MAX_LITERALS){
          for(num in 1:numberLiterals){
            nameOfLiteral<-literals[num]
            hotEncoding<-ifelse (dataset[,field]==nameOfLiteral,1.0,0.0)
            
            # 5/3/2018 - do not convert the field if their are too few literals
            # Use log of number of records as the measure
            literalsActive<-sum(hotEncoding==1)
            if (literalsActive>log(length(hotEncoding))) {
              catagorical<-cbind(catagorical,hotEncoding)
              # Field name has the "_" separator to make easier to read
              colnames(catagorical)[ncol(catagorical)]<-paste(colnames(dataset)[field],
                                                              "_", NPREPROCESSING_removePunctuation(nameOfLiteral), sep="")
            }
            else {
              print(paste("Ignoring in field:",names(dataset)[field],
                          "Literal:",nameOfLiteral,
                          "Too few=",literalsActive))
            }
          }
        } else {
          stop(paste("Error - too many literals in:", names(dataset)[field], 
                     numberLiterals))
        }
        
      }
    }
  }
  
  # Remove that first column that was full of NA due to R
  return(catagorical[,-1]) 
}
# End of NPREPROCESSING_categorical()

# Taken from lab3DataPrep.R
#*******************************************************************************
# NPREPROCESSING_removePunctuation() :
#
#
# INPUT: String - fieldName - name of field
#
# OUTPUT : String - name of field with punctuation removed
#**************************************

NPREPROCESSING_removePunctuation<-function(fieldName){
  
  return(gsub("[[:punct:][:blank:]]+", "", fieldName))
}
# End of NPREPROCESSING_removePunctuation()

# Taken from lab3DataPrep.R
# Changed print to cat
#*******************************************************************************
# NPREPROCESSING_redundantFields() :
#
# Determine if an entire field is redundant
# Uses LINEAR correlation,
# so use with care as information will be lost
#
# INPUT: Data frame - dataset - numeric values only
#        double     - cutoff  - Value above which is determined redundant [0,1]
#
# OUTPUT : Frame - dataset with redundant fields removed
#**************************************

NPREPROCESSING_redundantFields<-function(dataset,cutoff){
  
  cat(green("Number of fields before the redundancy check:"), ncol(dataset), "\n")
  
  # Remove any fields that have a stdev of zero (i.e. they are all the same)
  xx<-which(apply(dataset, 2, function(x) sd(x, na.rm=TRUE))==0)+1
  
  if (length(xx)>0L)
    dataset<-dataset[,-xx]
  
  # Kendall is more robust for data do not necessarily come from a bivariate
  # normal distribution.
  cr<-cor(dataset, use="everything")
  # cr[(which(cr<0))]<-0 #Positive correlation coefficients only
  NPLOT_correlagram(cr)
  
  correlated<-which(abs(cr)>=cutoff,arr.ind = TRUE)
  list_fields_correlated<-correlated[which(correlated[,1]!=correlated[,2]),]
  
  if (nrow(list_fields_correlated)>0){
    
    cat(green("Following fields are correlated:\n"))
    print(list_fields_correlated)
    
    # 240220nrt print list of correlated fields as names
    for (i in 1:nrow(list_fields_correlated)){
      print(paste(names(dataset)[list_fields_correlated[i,1]],"~", 
                  names(dataset)[list_fields_correlated[i,2]]))
    }
    
    # We have to check if one of these fields is correlated with another 
    # as cant remove both!
    v<-vector()
    numc<-nrow(list_fields_correlated)
    for (i in 1:numc){
      if (length(which(list_fields_correlated[i,1]==list_fields_correlated[i:numc,2]))==0) {
        v<-append(v,list_fields_correlated[i,1])
      }
    }
    cat(yellow("\nRemoving the following fields:\n"))
    print(names(dataset)[v])
    
    return(dataset[,-v]) # Remove the first field that is correlated with another
  }

  return(dataset)
}
# End of NPREPROCESSING_redundantFields() :

# Taken from lab3DataPrep.R
#*******************************************************************************
# NPLOT_correlogram () :
#
# INPUT: Data frame - cr - nxn frame of correlation coefficients
#
# OUTPUT : None
# 221019 - plot absolute values only
#**************************************

NPLOT_correlagram<-function(cr){
  
  # Defines the colour range
  col<-colorRampPalette(c("green", "red"))
  
  # To fit on screen, convert field names to a numeric
  rownames(cr)<-1:length(rownames(cr))
  colnames(cr)<-rownames(cr)
  
  corrplot::corrplot(abs(cr),method="square",
                     order="FPC",
                     cl.ratio=0.2,
                     cl.align="r",
                     tl.cex = 0.6,cl.cex = 0.6,
                     cl.lim = c(0, 1),
                     mar=c(1,1,1,1),bty="n")
}
# End of NPLOT_correlogram()

#*******************************************************************************
# multLinearReg () :
# Multiple Linear Regression using the caret package
#
# INPUT: Dataset
#
# OUTPUT : None
#**************************************

multLinearReg <- function (dataset){
  
  # Initilaise the 10 fold cross validation
  control <- trainControl(method = "cv", number = 10)
  
  # Set the dependent variable to be a factor
  #dataset[, OUTPUT_FIELD] <- as.factor(dataset[, OUTPUT_FIELD])
  
  # Create the multiple linear regression model
  multLinModel <- train(ATTRITION ~ ., data = dataset, method='lm',
                          trControl = control, tuneLength = 10)
  print(multLinModel)
  print(summary(multLinModel))
}
# End of multLinearReg()

#******************************************************************************* 
# furtherDimReduction () :
# Further removal of fields based on their significance level
# 
# INPUT: Dataset
#
# OUTPUT : List of required fields
#**************************************

furtherDimReduction <- function(dataset) { 
  
  cat(yellow("\nStarting further dimensionality reduction ...\n\n"))
  
  # Initial model
  # Gives the initial statistics for a linear model based off our variables to predict Attrition
  model_1 <- glm(ATTRITION ~ ., data = dataset, family = "binomial")
  summary(model_1) #AIC: 3209.2 ....Residual deviance: 3137.2
  
  # Stepwise selection
  model_2 <- stepAIC(model_1, direction="both")
  summary(model_2)  #AIC: 939.52...Residual deviance: 871.52
  
  # Remove multicollinearity through VIF check
  vif(model_2)
  
  # Remove JOBINVOLMENT as it is not significant p=0.122935    >0.05  
  model_3 <- glm(formula = ATTRITION ~ ENVIRONMENTSATISFACTION + JOBSATISFACTION + WORKLIFEBALANCE + NUMCOMPANIESWORKED + TRAININGTIMESLASTYEAR + YEARSSINCELASTPROMOTION + YEARSWITHCURRMANAGER + BUSINESSTRAVEL_TravelFrequently + BUSINESSTRAVEL_NonTravel + EDUCATIONFIELD_LifeSciences + EDUCATIONFIELD_Other + EDUCATIONFIELD_Medical + EDUCATIONFIELD_Marketing + EDUCATIONFIELD_TechnicalDegree + JOBROLE_HealthcareRepresentative + JOBROLE_ResearchScientist + JOBROLE_SalesExecutive + JOBROLE_ResearchDirector + JOBROLE_LaboratoryTechnician + MARITALSTATUS_Single + MARITALSTATUS_Divorced + OVERTIMECAT_overtime + OVERTIMECAT_earlylogout, family = "binomial", data = dataset)
  summary(model_3) #AIC: 3199.6  ....Residual deviance:  3151.6
  vif(model_3)
 
  # Remove MARITALSTATUS_Divorced as it is not significant p=0.044186 *
  model_4 <- glm(formula = ATTRITION ~ ENVIRONMENTSATISFACTION + JOBSATISFACTION + WORKLIFEBALANCE + NUMCOMPANIESWORKED + TRAININGTIMESLASTYEAR + YEARSSINCELASTPROMOTION + YEARSWITHCURRMANAGER + BUSINESSTRAVEL_TravelFrequently + BUSINESSTRAVEL_NonTravel + EDUCATIONFIELD_LifeSciences + EDUCATIONFIELD_Other + EDUCATIONFIELD_Medical + EDUCATIONFIELD_Marketing + EDUCATIONFIELD_TechnicalDegree + JOBROLE_HealthcareRepresentative + JOBROLE_ResearchScientist + JOBROLE_SalesExecutive + JOBROLE_ResearchDirector + JOBROLE_LaboratoryTechnician + MARITALSTATUS_Single + OVERTIMECAT_overtime + OVERTIMECAT_earlylogout, family = "binomial", data = dataset)
  summary(model_4) #AIC: 3201.8  ....Residual deviance:  3155.8
  vif(model_4)

  # Remove NUMCOMPANIESWORKED as it is not significant p=0.041432 *
  model_5 <- glm(formula = ATTRITION ~ ENVIRONMENTSATISFACTION + JOBSATISFACTION + WORKLIFEBALANCE + TRAININGTIMESLASTYEAR + YEARSSINCELASTPROMOTION + YEARSWITHCURRMANAGER + BUSINESSTRAVEL_TravelFrequently + BUSINESSTRAVEL_NonTravel + EDUCATIONFIELD_LifeSciences + EDUCATIONFIELD_Other + EDUCATIONFIELD_Medical + EDUCATIONFIELD_Marketing + EDUCATIONFIELD_TechnicalDegree + JOBROLE_HealthcareRepresentative + JOBROLE_ResearchScientist + JOBROLE_SalesExecutive + JOBROLE_ResearchDirector + JOBROLE_LaboratoryTechnician + MARITALSTATUS_Single + OVERTIMECAT_overtime + OVERTIMECAT_earlylogout, family = "binomial", data = dataset)
  summary(model_5) #AIC: 3203.9  ....Residual deviance:  3159.9
  vif(model_5)
  
  # Remove JOBROLE_HealthcareRepresentative as it is not significant p=0.037991 *
  model_6 <- glm(formula = ATTRITION ~ ENVIRONMENTSATISFACTION + JOBSATISFACTION + WORKLIFEBALANCE + TRAININGTIMESLASTYEAR + YEARSSINCELASTPROMOTION + YEARSWITHCURRMANAGER + BUSINESSTRAVEL_TravelFrequently + BUSINESSTRAVEL_NonTravel + EDUCATIONFIELD_LifeSciences + EDUCATIONFIELD_Other + EDUCATIONFIELD_Medical + EDUCATIONFIELD_Marketing + EDUCATIONFIELD_TechnicalDegree + JOBROLE_ResearchScientist + JOBROLE_SalesExecutive + JOBROLE_ResearchDirector + JOBROLE_LaboratoryTechnician + MARITALSTATUS_Single + OVERTIMECAT_overtime + OVERTIMECAT_earlylogout, family = "binomial", data = dataset)
  summary(model_6) #AIC: 3206.1  ....Residual deviance:  3164.1
  vif(model_6)
  
  # Remove JOBROLE_LaboratoryTechnician as it is not significant p=0.018014 *
  model_7 <- glm(formula = ATTRITION ~ ENVIRONMENTSATISFACTION + JOBSATISFACTION + WORKLIFEBALANCE + TRAININGTIMESLASTYEAR + YEARSSINCELASTPROMOTION + YEARSWITHCURRMANAGER + BUSINESSTRAVEL_TravelFrequently + BUSINESSTRAVEL_NonTravel + EDUCATIONFIELD_LifeSciences + EDUCATIONFIELD_Other + EDUCATIONFIELD_Medical + EDUCATIONFIELD_Marketing + EDUCATIONFIELD_TechnicalDegree + JOBROLE_ResearchScientist + JOBROLE_SalesExecutive + JOBROLE_ResearchDirector + MARITALSTATUS_Single + OVERTIMECAT_overtime + OVERTIMECAT_earlylogout, family = "binomial", data = dataset)
  summary(model_7) #AIC: 3209.6  ....Residual deviance:  3169.6
  vif(model_7)
  
  # Remove JOBROLE_ResearchScientist as it is not significant p=0.002045 **
  model_8 <- glm(formula = ATTRITION ~ ENVIRONMENTSATISFACTION + JOBSATISFACTION + WORKLIFEBALANCE + TRAININGTIMESLASTYEAR + YEARSSINCELASTPROMOTION + YEARSWITHCURRMANAGER + BUSINESSTRAVEL_TravelFrequently + BUSINESSTRAVEL_NonTravel + EDUCATIONFIELD_LifeSciences + EDUCATIONFIELD_Other + EDUCATIONFIELD_Medical + EDUCATIONFIELD_Marketing + EDUCATIONFIELD_TechnicalDegree + JOBROLE_SalesExecutive + JOBROLE_ResearchDirector + MARITALSTATUS_Single + OVERTIMECAT_overtime + OVERTIMECAT_earlylogout, family = "binomial", data = dataset)
  summary(model_8) #AIC: 3216.9  ....Residual deviance:  3178.9
  vif(model_8)
  
  # Remove JOBROLE_SalesExecutive as it is not significant p=0.010106 *
  model_9 <- glm(formula = ATTRITION ~ ENVIRONMENTSATISFACTION + JOBSATISFACTION + WORKLIFEBALANCE + TRAININGTIMESLASTYEAR + YEARSSINCELASTPROMOTION + YEARSWITHCURRMANAGER + BUSINESSTRAVEL_TravelFrequently + BUSINESSTRAVEL_NonTravel + EDUCATIONFIELD_LifeSciences + EDUCATIONFIELD_Other + EDUCATIONFIELD_Medical + EDUCATIONFIELD_Marketing + EDUCATIONFIELD_TechnicalDegree + JOBROLE_ResearchDirector + MARITALSTATUS_Single + OVERTIMECAT_overtime + OVERTIMECAT_earlylogout, family = "binomial", data = dataset)
  summary(model_9) #AIC: 3221.4  ....Residual deviance:  3185.4
  vif(model_9)
  
  # Remove JOBROLE_ResearchDirector as it is not significant p=0.006917 **
  model_10 <- glm(formula = ATTRITION ~ ENVIRONMENTSATISFACTION + JOBSATISFACTION + WORKLIFEBALANCE + TRAININGTIMESLASTYEAR + YEARSSINCELASTPROMOTION + YEARSWITHCURRMANAGER + BUSINESSTRAVEL_TravelFrequently + BUSINESSTRAVEL_NonTravel + EDUCATIONFIELD_LifeSciences + EDUCATIONFIELD_Other + EDUCATIONFIELD_Medical + EDUCATIONFIELD_Marketing + EDUCATIONFIELD_TechnicalDegree + MARITALSTATUS_Single + OVERTIMECAT_overtime + OVERTIMECAT_earlylogout, family = "binomial", data = dataset)
  summary(model_10) #AIC: 3226.3  ....Residual deviance:  3192.3
  vif(model_10)
  
  # Remove EDUCATIONFIELD_LifeSciences as it is has a high VIF at 8.117057
  model_11 <- glm(formula = ATTRITION ~ ENVIRONMENTSATISFACTION + JOBSATISFACTION + WORKLIFEBALANCE + TRAININGTIMESLASTYEAR + YEARSSINCELASTPROMOTION + YEARSWITHCURRMANAGER + BUSINESSTRAVEL_TravelFrequently + BUSINESSTRAVEL_NonTravel + EDUCATIONFIELD_Other + EDUCATIONFIELD_Medical + EDUCATIONFIELD_Marketing + EDUCATIONFIELD_TechnicalDegree + MARITALSTATUS_Single + OVERTIMECAT_overtime + OVERTIMECAT_earlylogout, family = "binomial", data = dataset)
  summary(model_11) #AIC: 3201.8  ....Residual deviance:  3155.8
  vif(model_11)
  
  # Remove EDUCATIONFIELD_Marketing as it is not significant p=0.156599     >0.05
  model_12 <- glm(formula = ATTRITION ~ ENVIRONMENTSATISFACTION + JOBSATISFACTION + WORKLIFEBALANCE + TRAININGTIMESLASTYEAR + YEARSSINCELASTPROMOTION + YEARSWITHCURRMANAGER + BUSINESSTRAVEL_TravelFrequently + BUSINESSTRAVEL_NonTravel + EDUCATIONFIELD_Other + EDUCATIONFIELD_Medical + EDUCATIONFIELD_TechnicalDegree + MARITALSTATUS_Single + OVERTIMECAT_overtime + OVERTIMECAT_earlylogout, family = "binomial", data = dataset)
  summary(model_12) #AIC: 3239.7  ....Residual deviance:  3209.7
  vif(model_12)
  
  # Remove EDUCATIONFIELD_Medical as it is not significant p=0.186479    >0.05
  model_13 <- glm(formula = ATTRITION ~ ENVIRONMENTSATISFACTION + JOBSATISFACTION + WORKLIFEBALANCE + TRAININGTIMESLASTYEAR + YEARSSINCELASTPROMOTION + YEARSWITHCURRMANAGER + BUSINESSTRAVEL_TravelFrequently + BUSINESSTRAVEL_NonTravel + EDUCATIONFIELD_Other + EDUCATIONFIELD_TechnicalDegree + MARITALSTATUS_Single + OVERTIMECAT_overtime + OVERTIMECAT_earlylogout, family = "binomial", data = dataset)
  summary(model_13) #AIC: 3239.4  ....Residual deviance:  3211.4
  vif(model_13)
  
  # Remove EDUCATIONFIELD_Other as it is not significant p=0.249282    >0.05
  model_14 <- glm(formula = ATTRITION ~ ENVIRONMENTSATISFACTION + JOBSATISFACTION + WORKLIFEBALANCE + TRAININGTIMESLASTYEAR + YEARSSINCELASTPROMOTION + YEARSWITHCURRMANAGER + BUSINESSTRAVEL_TravelFrequently + BUSINESSTRAVEL_NonTravel + EDUCATIONFIELD_TechnicalDegree + MARITALSTATUS_Single + OVERTIMECAT_overtime + OVERTIMECAT_earlylogout, family = "binomial", data = dataset)
  summary(model_14) #AIC: 3238.8  ....Residual deviance:  3212.8
  vif(model_14)
  
  # Remove EDUCATIONFIELD_TechnicalDegree as it is not significant p=0.249282    >0.05
  model_15 <- glm(formula = ATTRITION ~ ENVIRONMENTSATISFACTION + JOBSATISFACTION + WORKLIFEBALANCE + TRAININGTIMESLASTYEAR + YEARSSINCELASTPROMOTION + YEARSWITHCURRMANAGER + BUSINESSTRAVEL_TravelFrequently + BUSINESSTRAVEL_NonTravel + MARITALSTATUS_Single + OVERTIMECAT_overtime + OVERTIMECAT_earlylogout, family = "binomial", data = dataset)
  summary(model_15) #AIC: 3244.1  ....Residual deviance:  3220.1
  vif(model_15)
  
  cat(green("\nThe following fields have been removed:\n"),
            "JOBINVOLMENT", green("as it is not significant: p=0.122935\n"),
            "MARITALSTATUS_Divorced", green("as it is not significant: p=0.044186\n"),
            "NUMCOMPANIESWORKED", green("as it is not significant: p=0.041432\n"),
            "JOBROLE_HealthcareRepresentative", green("as it is not significant: p=0.037991\n"),
            "JOBROLE_LaboratoryTechnician", green("as it is not significant: p=0.018014\n"),
            "JOBROLE_ResearchScientist", green("as it is not significant: p=0.002045\n"),
            "JOBROLE_SalesExecutive", green("as it is not significant: p=0.010106\n"),
            "JOBROLE_ResearchDirector", green("as it is not significant: p=0.006917\n"),
            "EDUCATIONFIELD_LifeSciences", green("as it is has a high VIF at 8.117057\n"),
            "EDUCATIONFIELD_Marketing", green("as it is not significant: p=0.156599\n"),
            "EDUCATIONFIELD_Medical", green("as it is not significant: p=0.186479\n"),
            "EDUCATIONFIELD_Other", green("as it is not significant: p=0.249282\n"),
            "EDUCATIONFIELD_TechnicalDegree", green("as it is not significant: p=0.249282\n"))
            
  cat(green("\nFurther dimensionality reduction has been completed"), fill = TRUE, labels = NULL)
  
  return(names(model_15$model))
}
# End of furtherDimReduction()

# This function is from PBA module, lab4DataPrepNew.R
# ALTERED:
# 1. Added argument 'title'
# 2. Changed column and row names
#*******************************************************************************
# NplotConfusion()
#
# Plot confusion matrix
#
# INPUT:    list - results - results from NcalcConfusion()
#
# OUTPUT :  NONE
#
# 070819NRT Plots confusion matrix
#**************************************

NplotConfusion<-function(results, title){
  
  aa<-matrix(c(round(results$TP,digits=0),
               round(results$FN,digits=0),
               round(results$FP,digits=0),
               round(results$TN,digits=0)),
             nrow=2)
  row.names(aa)<-c("Attrition","NonAttrition")
  colnames(aa)<-c("Attrition","NonAttrition")
  fourfoldplot(aa,color=c("#cc6666","#99cc99"),
               conf.level=0,
               margin=2,
               main=paste(title, "\nTP  FP / FN   TN"))
} 
# End of NplotConfusion()

# This function is from PBA module, lab4DataPrepNew.R
# ALTERED:
# 1. Removed accuracy since our data is unbalanced
# 2. Added FNR
# 3. Re-ordered the metrics for visual, so TP/FN/TN/FP are in same order as #TPR/FNR/TNR/FPR
# 4. Re-named the precision variables to suit our dataset of Attrition
#*******************************************************************************
# NcalcMeasures() :
#
# Evaluation measures for a confusion matrix
#
# INPUT: numeric  - TP, FN, FP, TN
#
# OUTPUT: A list with the following entries:
#        TP        - double - True Positive records
#        FP        - double - False Positive records
#        TN        - double - True Negative records
#        FN        - double - False Negative records
#        pgood     - double - precision for non-attrition measure
#        pbad      - double - precision for attrition measure
#        FPR       - double - FPR measure
#        TPR       - double - TPR measure
#        TNR       - double - TNR measure
#        FNR       - double - FNR measure
#        MCC       - double - Matthew's Correlation Coeficient
#
# 080819NRT added TNR measure
#**************************************

NcalcMeasures<-function(TP,FN,FP,TN){
  
  retList<-list(   "TP"=TP,
                   "FN"=FN,
                   "TN"=TN,
                   "FP"=FP,
                   "pNA" =   100.0*(TP/(TP+FP)),
                   "pA"=   100.0*(TN/(FN+TN)),
                   "TPR"=   100.0*(TP/(TP+FN)),
                   "FNR"=   100.0*(FN/(TP+FN)),
                   "TNR"=   100.0*(TN/(FP+TN)),
                   "FPR"=   100.0*(FP/(FP+TN)),
                   "MCC"=   ((TP*TN)-(FP*FN))/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
  )
  return(retList)
  
} 
# End of NcalcMeasures

# This function is from PBA module, lab4DataPrepNew.R
#*******************************************************************************
# NcalcConfusion() :
#
# Calculate a confusion matrix for 2-class classifier
# INPUT: vector - expectedClass  - {0,1}, Expected outcome from each row (labels)
#        vector - predictedClass - {0,1}, Predicted outcome from each row (labels)
#
# OUTPUT: A list with the  entries from NcalcMeasures()
#
# 070819NRT convert values to doubles to avoid integers overflowing
# Updated to the following definition of the confusion matrix
#
#                    ACTUAL
#               ------------------
# PREDICTED     GOOD=1   |  BAD=0
#               ------------------
#     GOOD=1      TP     |    FP
#               ==================
#     BAD=0       FN     |    TN
#**************************************

NcalcConfusion<-function(expectedClass,predictedClass){
  
  confusion<-table(factor(predictedClass,levels=0:1),factor(expectedClass,levels=0:1))
  
  # This "converts" the above into our preferred format
  
  TP<-as.double(confusion[2,2])
  FN<-as.double(confusion[1,2])
  FP<-as.double(confusion[2,1])
  TN<-as.double(confusion[1,1])
  
  return(NcalcMeasures(TP,FN,FP,TN))
  
} 
# End of NcalcConfusion()

# Taken from lab sheet 3
#*******************************************************************************
# NConvertClass() :
#
# In original dataset, $ATTRITION is the classification label
# We need to convert this to give the minority class (attrition) a value of 0
#
# INPUT   :
#             Data Frame        - dataset
#
# OUTPUT  :
#             Data Frame        - dataset
#**************************************

NConvertClass<-function(dataset){
  
  positionClassOutput<-which(names(dataset)==OUTPUT_FIELD)
  classes<-sort(table(dataset[,positionClassOutput])) #smallest class will be first
  minority<-names(classes[1])
  indexToStatus2<-which(dataset[positionClassOutput]==minority)
  dataset[positionClassOutput][indexToStatus2,]<-0
  dataset[positionClassOutput][-indexToStatus2,]<-1
  
  return(dataset)
} 
# End of NConvertClass

# This function is from PBA module, 4labFunctions.R
#*******************************************************************************
# NEvaluateClassifier() :
#
# Use dataset to generate predictions from model
# Evaluate as classifier using threshold value
#
# INPUT   :   vector double     - probs        - probability of being class 1
#             Data Frame        - testing_data - Dataset to evaluate
#             double            - threshold     -cutoff (probability) for classification
#
# OUTPUT  :   List       - Named evaluation measures
#                        - Predicted class probability
#**************************************

NEvaluateClassifier<-function(test_predicted,test_expected,threshold) {
  
  predictedClass<-ifelse(test_predicted<threshold,0,1)
  
  results<-NcalcConfusion(expectedClass=test_expected,
                          predictedClass=predictedClass)
  
  return(results)
} 
# End of NEvaluateClassifier()

# This function is from PBA module, 4labFunctions.R
#*******************************************************************************
# auroc() :
#
# Calculate the Area Under Curve (AUC) for ROC
#
# INPUT   :   vector double     - score            - probability of being class 1
#             vector double     - bool             - Expected class of 0 or 1
#
# OUTPUT  :   double   - AUC
#**************************************
# By Miron Kursa https://mbq.me
# See https://stackoverflow.com/questions/4903092/calculate-auc-in-r
#**************************************

auroc <- function(score, bool) {
  n1 <- sum(!bool)
  n2 <- sum(bool)
  U  <- sum(rank(score)[!bool]) - n1 * (n1 + 1) / 2
  return(1 - U / n1 / n2)
}

# This function is from PBA module, 4labFunctions.R
# ALTERED:
# 1. Added model name and kfold number to plot title
#*******************************************************************************
# NdetermineThreshold() :
#
# For the range of threholds [0,1] calculate a confusion matrix
# and classifier metrics.
# Deterime "best" threshold based on either distance or Youdan
# Plot threshold chart and ROC chart
#
# Plot the results
#
# INPUT   :   vector double  - test_predicted   - probability of being class 1
#         :   vector double  - test_expected    - dataset to evaluate
#         :   boolean        - plot             - TRUE=output charts
#         :   string         - title            - chart title
#
# OUTPUT  :   List       - Named evaluation measures from confusion matrix
#                        - Threshold at min Euclidean distance
#                        - AUC - area under the ROC curve
#                        - Predicted class probability
#
# 241019NRT - added plot flag and title for charts
# 311019NRT - added axis bound checks in abline plots
# 191020NRT - Updated to use own ROC plot & calculate AUC
#************************************** 

NdetermineThreshold<-function(test_predicted,
                              test_expected,
                              plot=TRUE,
                              title=""){
  toPlot<-data.frame()
  
  # Vary the threshold
  for(threshold in seq(0,1,by=0.01)){
    results<-NEvaluateClassifier(test_predicted=test_predicted,
                                 test_expected=test_expected,
                                 threshold=threshold)
    toPlot<-rbind(toPlot,data.frame(x=threshold,fpr=results$FPR,tpr=results$TPR))
  }
  
  # the Youden index is the vertical distance between the 45 degree line
  # and the point on the ROC curve.
  # Higher values of the Youden index are better than lower values.
  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5082211/
  # Youdan = sensitivty + specificity -1
  #        = TPR + (1-FPR) -1
  
  toPlot$youdan<-toPlot$tpr+(1-toPlot$fpr)-1
  
  # 121020NRT - max Youdan
  # use which.max() to return a single index to the higest value in the vector
  maxYoudan<-toPlot$x[which.max(toPlot$youdan)]
  
  # Euclidean distance sqrt((1 − sensitivity)^2+ (1 − specificity)^2)
  # To the top left (i.e. perfect classifier)
  toPlot$distance<-sqrt(((100-toPlot$tpr)^2)+((toPlot$fpr)^2))
  
  # 121020NRT - Euclidean distance to "perfect" classifier (smallest the best)
  # use which.min() to return a single index to the lowest value in the vector
  minEuclidean<-toPlot$x[which.min(toPlot$distance)]
  
  #************************************** 
  # Plot threshold graph
  
  if (plot==TRUE){
    # Sensitivity (TPR)
    plot(toPlot$x,toPlot$tpr,
         xlim=c(0, 1), ylim=c(0, 100),
         type="l",lwd=3, col="blue",
         xlab="Threshold",
         ylab="%Rate",
         main=paste("Threshold Perfomance Classifier Model \n",title,"\n Fold:", KFOLDS))
    
    # Plot the specificity (1-FPR)
    lines(toPlot$x,100-toPlot$fpr,type="l",col="red",lwd=3,lty=1)
    
    # The point where specificity and sensitivity are the same
    crosspoint<-toPlot$x[which(toPlot$tpr<(100-toPlot$fpr))[1]]
    
    if (!is.na(crosspoint)){
      if ((crosspoint<1) & (crosspoint>0))
        abline(v=crosspoint,col="red",lty=3,lwd=2)
    }
    
    # Plot the Euclidean distance to "perfect" classifier (smallest the best)
    lines(toPlot$x,toPlot$distance,type="l",col="green",lwd=2,lty=3)
    
    # Plot the min distance, as might be more (311019NRT check it is within range)
    if ((minEuclidean<1) & (minEuclidean>0))
      abline(v=minEuclidean,col="green",lty=3,lwd=2)
    
    # Youdan (Vertical distance between the 45 degree line and the point on the ROC curve )
    lines(toPlot$x,toPlot$youdan,type="l",col="purple",lwd=2,lty=3)
    
    if ((maxYoudan<1) & (maxYoudan>0))
      abline(v=maxYoudan,col="purple",lty=3,lwd=2)
    
    legend("bottom",c("TPR","1-FPR","Distance","Youdan"),col=c("blue","red","green","purple"),lty=1:2,lwd=2)
    text(x=0,y=50, adj = c(-0.2,2),cex=1,col="black",paste("THRESHOLDS:\nEuclidean=",minEuclidean,"\nYoudan=",maxYoudan))
    
    #************************************** 
    # 121020NRT ROC graph
    
    sensitivityROC<-toPlot$tpr[which.min(toPlot$distance)]
    specificityROC<-100-toPlot$fpr[which.min(toPlot$distance)]
    auc<-auroc(score=test_predicted,bool=test_expected) # Estimate the AUC
    
    # Set origin point for plotting
    toPlot<-rbind(toPlot,data.frame(x=0,fpr=0,tpr=0, youdan=0,distance=0))
    
    plot(100-toPlot$fpr,toPlot$tpr,type="l",lwd=3, col="black",
         main=paste("ROC:",title, "\n Fold:", KFOLDS ),
         xlab="Specificity (1-FPR) %",
         ylab="Sensitivity (TPR) %",
         xlim=c(100,0),
         ylim=c(0,100)
    )
    
    axis(1, seq(0.0,100,10))
    axis(2, seq(0.0,100,10))
    
    #Add crosshairs to the graph
    abline(h=sensitivityROC,col="red",lty=3,lwd=2)
    abline(v=specificityROC,col="red",lty=3,lwd=2)
    
    annotate<-paste("Threshold: ",round(minEuclidean,digits=4L),
                    "\nTPR: ",round(sensitivityROC,digits=2L),
                    "%\n1-FPR: ",round(specificityROC,digits=2L),
                    "%\nAUC: ",round(auc,digits=2L),sep="")
    
    text(x=specificityROC, y=sensitivityROC, adj = c(-0.2,1.2),cex=1, col="red",annotate)
    
  } # endof if plotting
  
  # Select the threshold - I have choosen distance
  
  myThreshold<-minEuclidean      # Min Distance should be the same as analysis["threshold"]
  
  #Use the "best" distance threshold to evaluate classifier
  results<-NEvaluateClassifier(test_predicted=test_predicted,
                               test_expected=test_expected,
                               threshold=myThreshold)
  
  results$threshold<-myThreshold
  results$AUC<-auroc(score=test_predicted,bool=test_expected) # Estimate the AUC
  
  return(results)
} 
# End of Ndeterminethreshold()

# This function is from PBA module, 4labFunctions.R
#*******************************************************************************
# N_DEEP_Initialise()
# Initialise the H2O server
#
# INPUT:
#         Bool       - reproducible       - TRUE if model must be reproducable each run
#
# OUTPUT : none
#************************************** 

N_DEEP_Initialise<-function(reproducible=TRUE){
  
  # print("Initialise the H2O server")
  # Initialise the external h20 deep learning local server if needed
  # 130517NRT - set nthreads to -1 to use maximum so fast, but set to 1 to get reproducable results
  # 080819NRT - use reproducible parameter
  # 111120NRT - remove "max_mem_size=" that prevents H2O from using more than that amount of memory
  if (reproducible==TRUE)
    nthreads<-1
  else
    nthreads<- -1
  
  h2o.init(nthreads = nthreads)
  
  h2o.removeAll() # 261019NRT clean slate - just in case the cluster was already running
  
  #h2o.no_progress()
} 
# End of N_DEEP_Initialise()

# This function is from PBA module, 4labFunctions.R
#*******************************************************************************
# N_DEEP_TrainClassifier()
#
# h2O NEURAL NETWORK : DEEP LEARNING CLASSIFIER TRAIN
#
# INPUT:  Frame      - train              - scaled [0.0,1.0], fields & rows
#         String     - fieldNameOutput    - Name of the field to classify
#         Int Vector - hidden             - Number of hidden layer neurons for each layer
#         int        - stopping_rounds    - Number of times no improvement before stop
#         double     - stopping_tolerance - Error threshold
#         String     - activation         - Name of activation function
#         Bool       - reproducible       - TRUE if model must be reproducable each run
#
# OUTPUT: object     - trained neural network
#************************************** 

N_DEEP_TrainClassifier<- function(train,
                                  fieldNameOutput,
                                  hidden,
                                  stopping_rounds,
                                  stopping_tolerance,
                                  activation,
                                  reproducible){
  
  # positionOutput<-which(names(train)==fieldNameOutput)
  
  # Creates the h2o training dataset
  train[fieldNameOutput] <- lapply(train[fieldNameOutput] , factor) #Output class has to be a R "factor"
  
  train_h2o <- as.h2o(train, destination_frame = "traindata")
  
  # Create validation dataset for early stopping
  splits <- h2o.splitFrame(train_h2o, 0.9, seed=1234)
  nntrain  <- h2o.assign(splits[[1]], "nntrain.hex") # 90%
  nnvalid  <- h2o.assign(splits[[2]], "nnvalid.hex") # 10%
  
  # This lists all the input field names ignoring the fieldNameOutput
  predictors <- setdiff(names(train_h2o), fieldNameOutput)
  
  # Deep training neural network
  # Updated 13/5/17 - set reproducible = TRUE so that the same random numbers are used to initalise
  # 281019NRT - added validation dataset for early stopping
  
  deep<-h2o::h2o.deeplearning(x=predictors,
                              y=fieldNameOutput,
                              training_frame = nntrain,
                              validation_frame=nnvalid,
                              epochs=BASICNN_EPOCHS,
                              hidden=hidden,
                              adaptive_rate=TRUE,
                              stopping_rounds=stopping_rounds,
                              stopping_tolerance=stopping_tolerance,
                              stopping_metric = "misclassification",
                              fast_mode=FALSE,
                              activation=activation,
                              seed=1234,
                              l1 = 1e-2,
                              l2 = 1e-2,
                              variable_importances = TRUE,
                              reproducible = TRUE)
  
  return(deep)
} 
# End of N_DEEP_TrainClassifier()

# This function is from PBA module, 4labFunctions.R
#*******************************************************************************
# N_EVALUATE_DeepNeural() :
#
# Evaluate Deep Neural Network classifier
# Generates probabilities from the classifier
#
# INPUT: Data Frame    -  test             - scaled [0.0,1.0], fields & rows
#        String        -  fieldNameOutput  - Name of the field that we are training on (i.e.ATTRITION)
#        Object        - deep             - trained NN including the learn weights, etc.
#        boolean       - plot              - TRUE = output charts/results
#
# OUTPUT :
#         list - metrics from confusion matrix
#************************************** 

# Uses   library(h2o)
N_EVALUATE_DeepNeural<-function(test,fieldNameOutput, deep,plot, myTitle){
  
  # 201020NRT train data: expedcted class output as a numeric vector 0 or 1
  test_expected<-test[,fieldNameOutput]
  
  # Creates the h2o test dataset
  test[fieldNameOutput] <- lapply(test[fieldNameOutput] , factor) # Output class has to be a R "factor"
  test_h2o <- as.h2o(test, destination_frame = "testdata")
  
  pred <- h2o::h2o.predict(deep, test_h2o)
  
  test_predicted<-as.vector(pred$p1)  # Returns the probabilities of class 1
  
  measures<-NdetermineThreshold(test_expected=test_expected,
                                test_predicted=test_predicted,
                                plot=plot,
                                title=myTitle)
  
  
  return(measures)
} 
# End of N_EVALUATE_DeepNeural

# This function is from PBA module, Lab 4
# ALTERED:
# 1. Removed TRUE from arguement plot to only plot on last kfold
#*******************************************************************************
# getTreeClassifications() :
#
# Put in test dataset and get out class predictions of the decision tree
# Determine the threshold, plot the results and calculate metrics
#
# INPUT   :   object         - myTree        - tree
#         :   Data Frame     - testDataset - dataset to evaluate
#         :   string         - title        - string to plot as the chart title
#         :   int            - classLabel   - lable given to the positive (TRUE) class
#         :   boolean        - plot         - TRUE to output results/charts
#
# OUTPUT  :   List       - Named evaluation measures
#************************************** 

getTreeClassifications<-function(myTree,
                                 testDataset,title,
                                 classLabel=1,
                                 plot){
  
  positionClassOutput=which(names(testDataset)==OUTPUT_FIELD)
  
  #test data: dataframe with with just input fields
  test_inputs<-testDataset[-positionClassOutput]
  
  # Generate class membership probabilities
  # Column 1 is for class 0 (ATTRITION) and column 2 is for class 1 (No ATTRITION)
  
  testPredictedClassProbs<-predict(myTree,test_inputs, type="prob")
  # print(head(testPredictedClassProbs))
  
  
  # Get the column index with the class label
  classIndex<-which(as.numeric(colnames(testPredictedClassProbs))==classLabel)
  
  # Get the probabilities for classifying ATTRITION
  test_predictedProbs<-testPredictedClassProbs[,classIndex]
  
  #test data: vector with just the expected output class
  test_expected<-testDataset[,positionClassOutput]
  
  measures<-NdetermineThreshold(test_expected=test_expected,
                                test_predicted=test_predictedProbs,
                                plot=plot,
                                title=title)
  
  return(measures)
} 
# End of getTreeClassifications()

# This function is from PBA module, Lab 4
# ALTERED:
# 1. Removed TRUE from arguement plot, will now only plot for last kfold
# 2. Removed 'myTitle', used new titles for plots
# 3. Removed if plot statement / barplot of importance
# 4. Now returns a list of two items
#*******************************************************************************
# RandomForest() :
#
# Create Random Forest on pre-processed dataset
#
# INPUT   :
#         :   Data Frame     - train       - train dataset
#             Data Frame     - test        - test dataset
#             boolean        - plot        - TRUE = output charts/results
#
# OUTPUT  :
#         :   List           - dflist      - list of performance metrics
#                                          - dataframe of importance variables
#************************************** 

RandomForest<-function(train,test,plot){
  
  positionClassOutput<-which(names(train)==OUTPUT_FIELD)
  
  # train data: dataframe with the input fields
  train_inputs<-train[-positionClassOutput]
  
  # train data: vector with the expedcted output
  train_expected<-train[,positionClassOutput]
  
  rf<-randomForest::randomForest(train_inputs,
                                 factor(train_expected),
                                 ntree=FOREST_SIZE ,
                                 importance=TRUE,
                                 mtry=sqrt(ncol(train_inputs)))
  
  # get metrics from model using test data
  measures<-getTreeClassifications(myTree = rf,
                                   testDataset = test,
                                   plot=plot, title="RandomForest")
  
  # Calculate importance of variables from model (train data)
  importance<-randomForest::importance(rf,scale=TRUE,type=1)
  
  # create list to contain two dataframes to be used in funtions
  dflist <- list(measures, importance)
  
  return(dflist)
} 
# End of RandomForest()

# This function is from PBA module, Lab 4
# ALTERED:
# 1. Removed TRUE from argument plot, will now only plot for last kfold
# 2. Removed 'myTitle', gave new titles for plots
# 3. Removed if plot statement / barplot of importance
# 4. Now returns a list of two items
#*******************************************************************************
# deepNeural() :
#
# DEEP LEARNING EXAMPLE USING H2O library
#
# INPUT   :
#         :   Data Frame     - train       - train dataset
#         ;   Data Frame     - test        - test dataset
#         :   boolean        - plot        - TRUE = output charts/results
#
# OUTPUT  :
#         :   List           - dflist      - list of performance metrics
#                                          - dataframe of importance variables
#************************************** 

deepNeural<-function(train,test,plot){
  
  #h2o
  N_DEEP_Initialise()
  
  deep_classifier<-N_DEEP_TrainClassifier(train=train,
                                          fieldNameOutput=OUTPUT_FIELD,
                                          hidden=DEEP_HIDDEN,
                                          stopping_rounds=DEEP_STOPPING,
                                          stopping_tolerance=DEEP_TOLERANCE,
                                          activation=DEEP_ACTIVATION,
                                          reproducible=DEEP_REPRODUCABLE)
  
  # Evaluate the deep NN
  measures<-N_EVALUATE_DeepNeural(test=test,
                                  fieldNameOutput=OUTPUT_FIELD,
                                  deep=deep_classifier,
                                  plot=plot,
                                  myTitle = "DeepNeural")
  
  # variable importance from the deep neural network
  importance = as.data.frame(h2o::h2o.varimp(deep_classifier))
  
  row.names(importance)<-importance$variable
  importanceScaled<-subset(importance, select=scaled_importance)*100
  
  dflist <- list(measures, importanceScaled)
  
  return(dflist)
}  

# End of deepNeural()

# NEW FUNTION
# Neuralnet function found here: https://cran.r-project.org/web/packages/neuralnet/neuralnet.pdf
#*******************************************************************************
# ShallowNeural() :
#
#
# INPUT   :
#         :   Data Frame     - train       - train dataset
#         :   Data Frame     - test        - test dataset
#         :   boolean        - plot        - TRUE = output charts/results
#
# OUTPUT  :
#         :   List           - dflist      - list of performance metrics
#                                          - dataframe of importance variables
#************************************** 

ShallowNeural<-function(train,test, plot){
  
  #formula for model using all predictor variables
  formula<-paste(OUTPUT_FIELD,"~.")
  
  #Fit the neural network to the train data
  model=neuralnet(formula,data=train, hidden=NEURONS,act.fct = SHALLOW_ACT,
                  linear.output = FALSE, stepmax = STEPMAX)
  
  #Use neural network, "model", on test data
  predict <- compute(model,test)
  
  #Get probabilities from the predicitons
  probabilities <- predict$net.result
  
  #Find metrics based on test data
  measures <- NdetermineThreshold(probabilities,test[, OUTPUT_FIELD], title="ShallowNeural",plot=plot)
  
  #Create return variable for measures
  #Since we cannot use importance function on neuralnet to rank the varaibles of importance
  #the 2nd element has been set to FALSE - there is no dataframe
  dflist <- list(measures, FALSE)
  
  return(dflist)
} 
# End of ShallowNeural

# This function is from PBA module, Lab 3
# ALTERED:
# 1. Set decreasing = TRUE in the barplot for consistency with Random Forest bar plot
# 2. Set title for plot
# 3. Added values (1 dec place) to bar plot
# 4. Replaced the 'mymodelforumla' with one line of code and then removed the function
# 5. Removed if plot statement / barplot of importance
# 6. Now returns a list of two items
# 7. Renamed function from mymodelling to logisticregression
#*******************************************************************************
# LogisticRegression() :
# Create a logistic regression classifier and evaluate
#
# INPUT   :
#         :   Data Frame     - train       - train dataset
#         :   Data Frame     - test        - test dataset
#         :   boolean        - plot        - TRUE = output charts/results
#
# OUTPUT  :
#         :   List           - dflist      - list of performance metrics
#                                          - dataframe of importance variables
#**************************************

LogisticRegression<-function(train,test, plot){
  
  #formula for model using all predictor variables
  formula<-paste(OUTPUT_FIELD,"~.")
  
  #Build a logistic regression classifier on training dataset
  logisticModel<-stats::glm(formula,data=train,family=quasibinomial)
  #Use logistic model on test data
  probabilities<-predict(logisticModel, test,type="response")
  
  #Retreive metrics on model using probabilites from test data
  measures <-  NdetermineThreshold(probabilities,test$ATTRITION, title="LogisticRegression",plot=plot)
  
  #calculate importance variables
  importance<-as.data.frame(caret::varImp(logisticModel, scale = TRUE))
  row.names(importance)<-gsub("[[:punct:][:blank:]]+", "", row.names(importance))
  
  #Create return variable for measures (performance metrics) and importance variables
  dflist <- list(measures, importance)
  return(dflist)
}
# End of LogisticRegression

#NEW FUNCTION
#*******************************************************************************
# importanceBar() :
# Plots bar chart from most important to least important variable depending on model used
#
# INPUT   :
#         :   Data Frame     - importance  - mean importance values for variables
#         :   string         - title       - name of model
#
# OUTPUT  :
#         :   None
#**************************************

importanceBar <- function(importance, title){
  
  #Set graph dimensions
  #Used these numbers so that we could read the whole variable name (was cutting them off previously)
  par(mar=c(10,3,3,1))
  
  #Assigns the bar plot to x
  x <- barplot(t(importance[order(importance[,1], decreasing = TRUE),,drop=FALSE]),las=2, border = 0,
               cex.names =0.7,main=paste("Ranking Variables by (mean) importance:",title))
  
  #Assigns
  y<-as.matrix(t(importance[order(importance[,1], decreasing = TRUE),,drop=FALSE]))
  
  #plots bar plot with values to 1 decimal place
  text(x,y-1,labels=sprintf("%.1f", y))
  
  #logic was inspired by: https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781783988785/6/ch06lvl1sec69/displaying-values-on-top-of-or-next-to-the-bars
  
}
# End of importanceBar()

# This function is from PBA module, Lab sheet 4
# ALTERED:
# 1. Prints the model has started and finished
# 2. Now also calculates mean of importance variables for each model
#*******************************************************************************
# runExperiment() :
#
#
# INPUT   :   data frame         - dataset        - dataset
#             object function    - FUN            - name of function
#             ...                - optional       - parameters are passed on
#
# OUTPUT  :
#         :   List           - dflist      - list of mean performance metrics
#                                          - dataframe of mean importance variables
#**************************************

runExperiment<-function(dataset,title,FUN, ...){
  
  # create empty dataframes to append to
  metrics<-data.frame()
  ImpMeans <-data.frame()
  
  # run for each k fold in stratified split
  for (k in 1:KFOLDS){
    
    # tell user what model has began
    if (k==1){
      print(paste(title, "started"))
    }
    
    # get train and test data for fold k
    splitData<-stratifiedSplit(newDataset=dataset,fold=k)
    
    # get performance metrics and importance values from models
    measures <- FUN(train=splitData$train,
                    test=splitData$test,
                    plot=(k==KFOLDS),...)
    
    # assign performance metrics from measures to metrics
    metrics<-rbind(metrics,data.frame(measures[[1]]))
    
    # check to see if measures contains dataframe for importance
    if (length(measures[[2]]) != 0){
      
      # assign importance dataframe from measures to importance
      importance <- data.frame(measures[[2]])
      # order the dataframe alphabetically by variable names
      # this is very important step (each model gives initial dataframe in random order)
      importance <- importance[order(rownames(importance)),, drop=FALSE]
      # transpose dataframe
      importance <- t(importance)
      # append data to ImpMeans where all data for that model is kept
      # this will be used to calculate the mean over all folds
      ImpMeans <- rbind(ImpMeans,importance)
    }
    
    # tell user that model is complete
    if (k==KFOLDS){
      print(paste(title, "completed"))
    }
    
    
  } # end of for()
  
  
  # Return the means of the metrics from all the experiments back as a list
  getMeans<-colMeans(metrics)
  getMeans[1:4]<-as.integer(getMeans[1:4])  # TP, FN, TN, FP are rounded to ints
  
  # check to see if there is an importance variables dataframe
  if (length(ImpMeans)!=0){
    ImpMeans <-colMeans(ImpMeans)
    ImpMeans<- data.frame(ImpMeans)
    
  }
  
  # convert metrics dataframe to list
  getMeans<- as.list(getMeans)
  
  # create list for function so that it can return both dataframes
  means<-list(getMeans,ImpMeans)
  
  return(means)
} 
# End of runExperiment()

#This function is from PBA module, Lab sheet 4
#*******************************************************************************
# allocateFoldID() :
#
# Append a column called "foldID" that indicates the fold number
#
# INPUT   :   data frame         - dataset        - dataset
#
# OUTPUT  :   data frame         - dataset        - dataset with foldID added
#
#**************************************

allocateFoldID<-function(dataset){
  recordsPerFold<-ceiling(nrow(dataset)/KFOLDS)
  
  foldIds<-rep(seq(1:KFOLDS),recordsPerFold)
  
  foldIds<-foldIds[1:nrow(dataset)]
  
  dataset$foldId<-foldIds
  
  return(dataset)
} 
# End of allocateFoldID()

#This function is from PBA module, Lab sheet 4
#*******************************************************************************
# stratifiedDataset() :
#
# Split dataset by the class (assume 2-class)
# Calculate the number of records that will appear in each fold
# Give each of these blocks a unique foldID
# combine the datasets & randomise
# The dataset now has a foldID from which we can select the data
# for the experiments
#
# INPUT   :   data frame         - dataset        - dataset
#
# OUTPUT  :   data frame         - dataset        - dataset with foldID added
#**************************************

stratifiedDataset<-function(originalDataset){
  
  positionClassOutput<-which(names(originalDataset)==OUTPUT_FIELD)
  
  # Get the unique class values
  classes<-unique(originalDataset[,positionClassOutput])
  
  # Split dataset into the two classes (so as to keep the class balance the same in the datasets)
  indexClass1<-which(originalDataset[,positionClassOutput]==classes[1])
  split1<-originalDataset[indexClass1,]
  split2<-originalDataset[-indexClass1,]
  
  # Append a column that indicates the fold number for each class
  split1<-allocateFoldID(split1)
  split2<-allocateFoldID(split2)
  
  # Combine the two datasets
  
  newDataset<-rbind(split1,split2)
  
  #Randomise the classes
  newDataset<-newDataset[order(runif(nrow(newDataset))),]
  
  return(newDataset)
} 
# End of stratifiedDataset()


#This function is from PBA module, Lab sheet 4
#*******************************************************************************
# stratifiedSplit() :
#
# Generate the TRAIN and TEST dataset based on the current fold
#
# INPUT   :   data frame         - dataset        - dataset
#
# OUTPUT  :   list               - train & test datasets
#**************************************

stratifiedSplit<-function(newDataset,fold){
  
  test<-subset(newDataset, subset= foldId==fold, select=-foldId)
  train<-subset(newDataset, subset= foldId!=fold,select=-foldId)
  
  return(list(
    train=train,
    test=test))
} 
# End of stratifiedSplit

# NEW FUNCTION
#*******************************************************************************
# AllResults() :
#
# Run each model and obtain corresponding data in a single dataframe
# Prints results to viewer
# Plots importance bar plot if there are values for model
#
# INPUT   :   list               - models         -list of function names
#         :   data frame         - dataset        - dataset
#
# OUTPUT  :   None
#**************************************

AllResults <- function(models, dataset){
  
  # Results dataframe to append results to
  # The row names have been labelled so there is no error in size when appending columns
  allResults <- data.frame(row.names=c("TP","FN","TN","FP","pNA","pA","TPR","FNR","TNR","FPR","MCC","Threshold","AUC"))
  
  # runs for each model in MYMODEL
  for (model in models){
    # run each model to obtain mean measures
    means<-runExperiment(dataset = dataset,title = model, FUN = get(model))
    # append (mean) metric performance obtained from model to all results
    allResults <- cbind(allResults,data.frame(model=unlist(means[[1]])))
    
    # plot (mean) confusion matrix for model
    NplotConfusion(means[[1]],title = model)
    
    # Check there are mean importance results
    # if so plot the bar chart
    if (means[[2]]>1){
      importanceBar(means[[2]],title =model)
    }
    
  } # End of for loop over models
  
  # transpose the data
  allResults<-data.frame(t(allResults))
  
  # rename the rows to the corresponding model
  row.names(allResults) <- MYMODELS
  
  # order results by MCC value
  allResults<-allResults[order(allResults$MCC,decreasing = TRUE),]
  
  # output results to compare all classifiers
  allResults[,1:4]<-sapply(allResults[,1:4], as.integer)
  # Set values to 2 decimal places
  allResults[,5:13]<-round(allResults[,5:13],digits=2)
  
  # print results to viewer
  print(formattable::formattable(allResults))
} 
# End of AllResults

# Some of the functions were taken from the labs as mentioned previously
#*******************************************************************************
# main() :
#
# The main function
#
# INPUT   :    none      
#
# OUTPUT  :    none         
#************************************************

main <- function(){
  
  # Combine all datasets into one file
  mergedData <- prepareDataset(DATASET1,DATASET2,DATASET3,DATASET4,DATASET5)
  
  # Do some data exploration
  basicDataExploration(mergedData)
  
  # Clean the data - remove missing values, redundancies 
  mergedData <- initialDataCleaning(mergedData)
  
  # Convert the format of categorical variables into the one in the 
  # data dictionary to facilitate further visualisations
  mergedData <- convertFormat(mergedData)
  
  # Update the record of Numeric columns
  numericData <- updateNumericData(mergedData)
  
  # Update the record of Categorical columns
  categoricData <- updateCategoricalData(mergedData)
  
  # Visualise the results
  visualresults<- visualisations(mergedData)

  # Convert the format of some categorical variables back into numeric in order
  # to facilitate further analysis
  mergedData <- convertToNumerical(mergedData)

  # Update the record of Numeric columns
  numericData <- updateNumericData(mergedData)

  # Update the record of Categorical columns
  categoricData <- updateCategoricalData(mergedData)

  # Output simple dataset field analysis results as a table in "Viewer"
  NPREPROCESSING_prettyDataset(mergedData)

  # Set each field for NUMERIC or SYMBOLIC
  cat(yellow("\nTesting each field if Numeric or Symbolic ...\n"))
  field_types <- NPREPROCESSING_initialFieldType(mergedData)
  cat(green("Done\n"))

  # Test NUMERIC fields for DISCRETE or ORDINAL
  cat(yellow("\nTesting numeric fields if Discrete or Ordinal ...\n"))
  field_types1 <- NPREPROCESSING_discreteNumeric(dataset=mergedData,
                                               field_types=field_types,
                                               cutoff=DISCRETE_BINS)
  cat(green("Done\n"))

  # Print in the terminal to check the variables have been named
  cat(green("\nThese are the resulting field types:\n"))
  print(field_types1)

  # Summarise these changes in a table
  results <- data.frame(field=names(mergedData),initial=field_types,types1=field_types1)
  print(formattable::formattable(results))

  # Create a sub-set frame of ordinal fields
  ordinals <- mergedData[,which(field_types1==TYPE_ORDINAL)]

  # Test if any ordinals are outliers and replace with mean values
  # Null hypothesis is there are no outliers
  # We reject this if the p-value<significance (i.e. 0.05), confidence=95%
  # Calculate the outliers and plot graphically
  cat(yellow("\nDealing with outliers ...\n"))
  ordinals <- NPREPROCESSING_outlier(ordinals=ordinals,confidence=OUTLIER_CONF)
  cat(green("The outliers have been dealt with\n"))

  # z-scale - normalise the entire data frame to be in interval [0,1]
  cat(yellow("\nNormalising the data frame to be in [0,1] interval ...\n"))
  zscaled <- as.data.frame(scale(ordinals,center=TRUE, scale=TRUE))
  cat(green("Normalisation has been completed\n"))

  # In the chosen classifier, the input values need to be scaled to [0.0,1.0]
  ordinalReadyforML <- Nrescaleentireframe(zscaled)

  # Categorical Pre-Processing
  cat(yellow("\nPreprocessing categorical fields ...\n"))
  catagoricalReadyforML <- NPREPROCESSING_categorical(dataset=mergedData,
                                                         field_types=field_types1)
  cat(green("Done\n"))
  print(formattable::formattable(data.frame(fields=names(catagoricalReadyforML))))

  # Explain how many categoric fields have been transformed
  # number of non-numeric fields before transformation
  # 241019 which fields are either SYMBOLIC or DISCRETE
  nonNumericbefore<-length(which(field_types1!=TYPE_ORDINAL))
  
  # How many fields have be generated through the 1-hot-encoding process
  nonNumerictranformed<-ncol(catagoricalReadyforML)
  cat(green("\nSymbolic fields before encoding:"), nonNumericbefore,
              green("After:"), nonNumerictranformed, "\n")

  # Combine the transformed numeric and categoric fields that are now pre-processed
  # Combine the two sets of data that are ready for ML
  cat(yellow("\nCombining the transformed numeric and categorical fields ...\n"))
  combinedML <- cbind(ordinalReadyforML,catagoricalReadyforML)
  cat(green("Done\n\n"))

  # Check for redundant fields
  combinedML<- NPREPROCESSING_redundantFields(dataset=combinedML,cutoff=CORRELATIONTHRESHOLD)
  
  # The dataset for ML information
  cat(green("\nNumber of fields after the redundancy check:"), ncol(combinedML), "\n")

  # Randomise the entire dataset
  cat(yellow("\nRandomising the entire dataset ...\n"))
  combinedML<-combinedML[order(runif(nrow(combinedML))),]
  cat(green("Done\n"))

  # Inspect the final fields
  cat(green("\nDataset:\n"))
  print(names(combinedML))

  # Multiple Linear Regression - commented out
  # This does not work well as we have a binary classification problem
  # R Squared value is so low that the results yielded are not trustworthy
  # multLinearReg(dataset)
  
  # Further reduce the dimensionality using multiple logistic regression
  finalfields <- furtherDimReduction(combinedML)
  finalDataset <- combinedML[, c(finalfields)]

  # Print the final fields and output the final dataset, which is ready for 
  # modelling into a csv file
  cat(green("\nThis is the final dataset:\n"))
  print(names(finalDataset))
  write.csv(finalDataset, "finaldataset.csv", row.names = FALSE)
  cat(green("The final dataset has been successfully exported as 'finaldataset.csv'"),
      green("into the working directory"), fill = TRUE, labels = NULL)

  # Converth the field of interest into the minority class (ATTRITION) awith a value of 0
  original <- NConvertClass(finalDataset)

  # Stratification
  cat(yellow("\nPreparing a stratified dataset ...\n"))
  dataset <- stratifiedDataset(original)
  cat(green("\nStratification has been successful\n"))

  # Modelling
  cat(green("\nStarting modelling ...\n"))
  AllResults(models=MYMODELS, dataset=dataset)
  cat(green("Modelling has finsished. See viewer for results.\n"))
}
# End of main()

#*******************************************************************************

main()
print("The End")
