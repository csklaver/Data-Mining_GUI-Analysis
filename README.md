# Final-Project-Group9
Introduction to Data Mining - DATS6103
Caroline Sklaver & Sayra Moore

The purpose of this project is to perform EDA and machine learning algorithms to predict 2015 London bike share activity. This project includes python script to preprocess and export the cleaned data, as well as create visualizations for our final presentation. We also include a GUI portion for users to run through EDA, Linear Regressions, and Multiple Linear Regressions with this dataset. 

The structure of the dataset is as follows:

**Independant Variable --**
Bike Share Count

**Dependant Variables(Features) --**
* Temperature
* Temperature "feels"
* Humidity
* Wind_speed
* Weather_code
* Is_holiday
* Is_weekend
* Season

**Machine Leaning algorithms**
* Linear Regression
* Multiple Linear Regression

# Description of files 

**read_clean_lm.py** is the python file that reads, cleans, and exports the new dataset. Sample multiple linear regression is run. Plots and visuals are made. 

**Main_GUI.py** is the python file that contains all the code for the GUI demo.

**London_raw.csv** is the initial datasets, it is raw data that is used in the read_clean_lm.py code file.

**London_exported.csv** is the processed dataset that is used the EDA and in the ML algorithms of the GUI.

All the files need to be located in the same directory.

# Description of the application 

The purpose of the application is to present a basic EDA analysis of the processed dataset and two dashboards with the results of the algoritms. The algorithms used are Linear Regression and Multiple Linear Regression. The Linear Regression algorithm is used mostly for visualization, while the Multiple Linear Regression builds a complete model. 

The structure of the application is as follows:
  
1. EDA Analysis
    >* **Correlation Plot :** This option presents a correlation plot for all the features in the dataset. The features can be add o deleted from the plot. Each time that a modification is made the button **Create Plot** should be pressed.
    
3. ML Models
    >* **Linear Regression :** 
    >>* This option presents the plot of user-chosen continous feature variable with dependent variable(count). 
    >>* The features to be included in the plot can be manipulated by using the dropdown on the screen. 
    >>* The application also presents a check-box option to show the regression line. 
    >>* The plot and regression line automatically populate as the user changes the variable or check-box options. 
   
    
    >* **Multiple Linear Regression :** 
    >>* This option creates a dashboard with results from the Multiple Linear Regression algorithm results given by the Sklearn library. 
    >>* The features to be included in the algorithm are chosen. However, you can manipulate how many feature by manipulating the checkboxes in the screen. 
    >>* The application presents the model intercept, R2 value, Mean Squared Error, RMSE value, and Coefficients by feature (Parameters).
    >>* Once you are comfortable with the chosen features and parameters you can click **Run Regression** to populate the dashboard with the data. 
    >>* You can execute the algorithms as many times as you wish. The dashboard will erase the previous results and present the new ones. 
    
