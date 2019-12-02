import sys
import PyQt5

#from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel, QGridLayout, QCheckBox, QGroupBox
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt

from scipy import interp
from itertools import cycle


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split

import random
import seaborn as sns
import statsmodels.regression.linear_model as sm
from statsmodels.tools.tools import add_constant

#%%-----------------------------------------------------------------------
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'
#%%-----------------------------------------------------------------------

def main():
    #::-------------------------------------------------
    # Initiates the application
    #::-------------------------------------------------
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = App()
    ex.show()
    sys.exit(app.exec_())


def data_bike():
    #::--------------------------------------------------
    # Loads the dataset London_exported.csv (preprocessed)
    #::--------------------------------------------------
    global features_list
    global london_bikes
    london_bikes = pd.read_csv('london_exported.csv')
    X= london_bikes["count"]
    features_list = ["temp", "tempf", "humidity", "wind_speed","weather_code",
         "is_holiday", "is_weekend", "season", "month","day","hour"]


#::--------------------------------
# Default font size for all the windows
#::--------------------------------
font_size_window = 'font-size:15px'

class App(QMainWindow):
    #::-------------------------------------------------------
    # This class creates all the elements of the application
    #::-------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.Title = 'London Bike Share Analysis'
        self.width = 500
        self.height = 300
        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the menu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #::-----------------------------
        # Create the menu bar
        # and three items for the menu, File, EDA Analysis and ML Models
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet('background-color: lightblue')

        fileMenu = mainMenu.addMenu('File')
        EDAMenu = mainMenu.addMenu('EDA Analysis')
        MLModelMenu = mainMenu.addMenu('ML Model')


        # Create Dropdown For EDA
        EDA1Button = QAction(QIcon('analysis.png'),'Correlation Plot', self)
        EDA1Button.setStatusTip('Features Correlation Plot')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)

        self.dialogs = list()

        #::--------------------------------------------------
        # ML Models for prediction
        #::--------------------------------------------------
        # Linear regression Model
        #::--------------------------------------------------
        # dropdown for Linear Regression
        MLModel1Button =  QAction(QIcon(), 'Linear Regression', self)
        MLModel1Button.setStatusTip('ML algorithm ')
        MLModel1Button.triggered.connect(self.MLLR)

        MLModelMenu.addAction(MLModel1Button)

        #::--------------------------------------------------
        # Multiple Linear regression Model
        #::--------------------------------------------------
        # Dropdown for multiple linear regression
        MLModel2Button =  QAction(QIcon(), 'Multiple Linear Regression', self)
        MLModel2Button.setStatusTip('ML algorithm')
        MLModel2Button.triggered.connect(self.MLMLR)

        MLModelMenu.addAction(MLModel2Button)

        #::--------------------------------------------------
        # Decision Tree Model
        #::--------------------------------------------------
        MLModel3Button =  QAction(QIcon(), 'Decision Tree Entropy', self)
        MLModel3Button.setStatusTip('ML algorithm with Entropy ')
        MLModel3Button.triggered.connect(self.MLDT)

        MLModelMenu.addAction(MLModel3Button)

        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

    def EDA1(self):
        #::----------------------------------------------------------
        # This function creates an instance of the CorrelationPlot class
        #::----------------------------------------------------------
        dialog = CorrelationPlot()
        self.dialogs.append(dialog)
        dialog.show()

    def MLLR(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the Linear Regression
        #::-----------------------------------------------------------
        dialog = LinearRegression()
        self.dialogs.append(dialog)
        dialog.show()

    def MLMLR(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the Multiple Linear Regression
        #::-----------------------------------------------------------
        dialog = MultipleLinearRegression()
        self.dialogs.append(dialog)
        dialog.show()

    def MLDT(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the Multiple Linear Regression
        #::-----------------------------------------------------------
        dialog = DecisionTree()
        self.dialogs.append(dialog)
        dialog.show()

class DecisionTree(QMainWindow):
    #::----------------------
    # Implementation of Decision Tree Algorithm using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #       view_tree : shows the tree in a pdf form
    #::----------------------

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(DecisionTree, self).__init__()

        self.Title ="Decision Tree Classifier"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid layout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML Decision Tree Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)
        self.feature6 = QCheckBox(features_list[6], self)
        self.feature7 = QCheckBox(features_list[7], self)
        self.feature8 = QCheckBox(features_list[8], self)
        self.feature9 = QCheckBox(features_list[9], self)
        self.feature10 = QCheckBox(features_list[10], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)
        self.feature7.setChecked(True)
        self.feature8.setChecked(True)
        self.feature9.setChecked(True)
        self.feature10.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel('Maximun Depth :')
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("3")

        self.btnExecute = QPushButton("Execute DT")
        self.btnExecute.clicked.connect(self.update)

        self.btnDTFigure = QPushButton("View Tree")
        self.btnDTFigure.clicked.connect(self.view_tree)

        # We create a checkbox for each feature

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0)
        self.groupBox1Layout.addWidget(self.feature3,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0)
        self.groupBox1Layout.addWidget(self.feature5,2,1)
        self.groupBox1Layout.addWidget(self.feature6,3,0)
        self.groupBox1Layout.addWidget(self.feature7,3,1)
        self.groupBox1Layout.addWidget(self.feature8,4,0)
        self.groupBox1Layout.addWidget(self.feature9,4,1)
        self.groupBox1Layout.addWidget(self.feature10, 4, 2)
        self.groupBox1Layout.addWidget(self.lblPercentTest,5,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,5,1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth,6,0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth,6,1)
        self.groupBox1Layout.addWidget(self.btnExecute,7,0)
        self.groupBox1Layout.addWidget(self.btnDTFigure,7,1)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        #::-------------------------------------
        # Graphic 1 : Confusion Matrix
        #::-------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::--------------------------------------------
        ## End Graph1
        #::--------------------------------------------

        #::---------------------------------------------
        # Graphic 2 : ROC Curve
        #::---------------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::---------------------------------------------------
        # Graphic 3 : ROC Curve by Class
        #::---------------------------------------------------


        ## End of elements on the dashboard

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,0,2)
        self.layout.addWidget(self.groupBoxG2,1,1)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()


    def update(self):
        '''
        Decision Tree Algorithm
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Decision Tree algorithm
          then the results are presented in graphics and reports in the canvas
        :return: None
        '''

        # We process the parameters
        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = london_bikes[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, london_bikes[features_list[0]]],axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = london_bikes[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, london_bikes[features_list[1]]],axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = london_bikes[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, london_bikes[features_list[2]]],axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = london_bikes[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, london_bikes[features_list[3]]],axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = london_bikes[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, london_bikes[features_list[4]]],axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = london_bikes[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, london_bikes[features_list[5]]],axis=1)

        if self.feature6.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = london_bikes[features_list[6]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, london_bikes[features_list[6]]],axis=1)

        if self.feature7.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = london_bikes[features_list[7]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, london_bikes[features_list[7]]],axis=1)

        if self.feature8.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = london_bikes[features_list[8]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, london_bikes[features_list[8]]],axis=1)

        if self.feature9.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = london_bikes[features_list[9]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, london_bikes[features_list[9]]],axis=1)

        if self.feature10.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = london_bikes[features_list[10]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, london_bikes[features_list[10]]],axis=1)



        vtest_per = float(self.txtPercentTest.text())
        vmax_depth = float(self.txtMaxDepth.text())

        self.ax1.clear()
        self.ax2.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100


        # We assign the values to X and y to run the algorithm

        X_dt =  self.list_corr_features
        y_dt = london_bikes["count"]

        class_le = LabelEncoder()

        # fit and transform the class

        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100)
        # perform training with entropy.
        # Decision tree with entropy
        self.clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=vmax_depth, min_samples_leaf=5)

        # Performing training
        self.clf_entropy.fit(X_train, y_train)

        # predicton on test using entropy
        y_pred_entropy = self.clf_entropy.predict(X_test)

        # confusion matrix for entropy model

        conf_matrix = confusion_matrix(y_test, y_pred_entropy)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred_entropy)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred_entropy) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))


        #::----------------------------------------------------------------
        # Graph1 -- Confusion Matrix
        #::-----------------------------------------------------------------

        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))


        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::-----------------------------------------------------
        # End Graph 1 -- Confusioin Matrix
        #::-----------------------------------------------------

        #::-----------------------------------------------------
        # Graph 2 -- ROC Cure
        #::-----------------------------------------------------

        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
        n_classes = y_test_bin.shape[1]


        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

    def view_tree(self):
        '''
        Executes the graphviz to create a tree view of the information
         then it presents the graphic in a pdf formt using webbrowser
        :return:None
        '''
        dot_data = export_graphviz(self.clf_entropy, filled=True, rounded=True,
                                   feature_names=self.list_corr_features.columns, out_file=None)


        graph = graph_from_dot_data(dot_data)
        graph.write_pdf("decision_tree_entropy.gui.pdf")
        webbrowser.open_new(r'decision_tree_entropy.gui.pdf')


class MultipleLinearRegression(QMainWindow):
    #::----------------------
    # Implementation of Multiple Linear regression using the bike share dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #::----------------------

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(MultipleLinearRegression, self).__init__()

        self.Title ="Multiple Linear Regression"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid layout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Independent Variables')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        # Create check boxes for each feature
        self.feature0 = QCheckBox(features_list[0], self)
        self.feature1 = QCheckBox(features_list[1], self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4], self)
        self.feature5 = QCheckBox(features_list[5], self)
        self.feature6 = QCheckBox(features_list[6], self)
        self.feature7 = QCheckBox(features_list[7], self)
        self.feature8 = QCheckBox(features_list[8], self)
        self.feature9 = QCheckBox(features_list[9], self)
        self.feature10 = QCheckBox(features_list[10], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(False)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)
        self.feature7.setChecked(True)
        self.feature8.setChecked(True)
        self.feature9.setChecked(True)
        self.feature10.setChecked(True)

        self.btnExecute = QPushButton("Run Regression")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 0, 2)
        self.groupBox1Layout.addWidget(self.feature3, 0, 3)
        self.groupBox1Layout.addWidget(self.feature4, 1, 0)
        self.groupBox1Layout.addWidget(self.feature5, 1, 1)
        self.groupBox1Layout.addWidget(self.feature6, 1, 2)
        self.groupBox1Layout.addWidget(self.feature7, 1, 3)
        self.groupBox1Layout.addWidget(self.feature8, 2, 0)
        self.groupBox1Layout.addWidget(self.feature9, 2, 1)
        self.groupBox1Layout.addWidget(self.feature10, 3, 0)
        self.groupBox1Layout.addWidget(self.btnExecute, 3, 1)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblR2 = QLabel('R2:')
        self.txtR2 = QLineEdit()


        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblR2)
        self.groupBox2Layout.addWidget(self.txtR2)

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBox2, 1, 0)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
    #::------------------------------------------------------------
    # Populates the elements in the canvas using the values
    # chosen as parameters for the multiple linear regression
    #::------------------------------------------------------------

        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        self.Y = london_bikes['count']
        self.X = pd.DataFrame()
        if self.feature0.isChecked():
            self.X[features_list[0]] = london_bikes[features_list[0]]
        if self.feature1.isChecked():
            self.X[features_list[1]] = london_bikes[features_list[1]]
        if self.feature2.isChecked():
            self.X[features_list[2]] = london_bikes[features_list[2]]
        if self.feature3.isChecked():
            self.X[features_list[3]] = london_bikes[features_list[3]]
        if self.feature4.isChecked():
            self.X[features_list[4]] = london_bikes[features_list[4]]
        if self.feature5.isChecked():
            self.X[features_list[5]] = london_bikes[features_list[5]]
        if self.feature6.isChecked():
            self.X[features_list[6]] = london_bikes[features_list[6]]
        if self.feature7.isChecked():
            self.X[features_list[7]] = london_bikes[features_list[7]]
        if self.feature8.isChecked():
            self.X[features_list[8]] = london_bikes[features_list[8]]
        if self.feature9.isChecked():
            self.X[features_list[9]] = london_bikes[features_list[9]]
        if self.feature10.isChecked():
            self.X[features_list[9]] = london_bikes[features_list[10]]

        # splitting X and y into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y,
                                                                                test_size=0.4,
                                                                                random_state=1)

        # create linear regression object
        self.reg = linear_model.LinearRegression()

        # train the model using the training sets
        self.results = self.reg.fit(self.X_train, self.y_train)
        self.y_pred = self.reg.predict(self.X_test)

        # regression coefficients
        self.coef =  self.reg.coef_

        # variance score: 1 means perfect prediction
        self.variance = self.reg.score(self.X_test, self.y_test)

        # R2 and Mean Squared Error
        self.R2 = r2_score(self.y_test, self.y_pred)  # Priniting R2 Score
        self.MSE = mean_squared_error(self.y_test, self.y_pred)

        #self.txtResults.appendPlainText(self.coef)

        self.txtR2.setText(str(self.R2))


class LinearRegression(QMainWindow):
    #::----------------------
    # Implementation of Linear regression using the bike share dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parameters
    #               chosen by the user
    #::----------------------

    send_fig = pyqtSignal(str)

    def __init__(self):

    #::--------------------------------------------------------
    # Create a canvas with the layout to draw a dotplot
    # The layout sets all the elements and manage the changes
    # made on the canvas
    #::--------------------------------------------------------
        super(LinearRegression, self).__init__()

        self.Title = "Continuous Features vs. Bike Count"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                              QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(["temp","tempf","humidity","wind_speed"])

        self.dropdown1.currentIndexChanged.connect(self.update)
        self.label = QLabel("A plot:")

        self.checkbox1 = QCheckBox('Show Regression Line', self)
        self.checkbox1.stateChanged.connect(self.update)

        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(QLabel("Select Feature for Linear Regression Subplot"))
        self.layout.addWidget(self.dropdown1)
        self.layout.addWidget(self.checkbox1)
        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)
        self.show()
        self.update()


    def update(self):
        #::--------------------------------------------------------
        # This method executes each time a change is made on the canvas
        # containing the elements of the graph
        # The purpose of the method es to draw a scatterplot using the
        # count of bikes and the feature chosen the canvas
        #::--------------------------------------------------------
        colors = ["b", "r", "g", "y", "k", "c"]
        self.ax1.clear()
        cat1 = self.dropdown1.currentText()

        X_1 = london_bikes["count"]
        y_1 = london_bikes[cat1]

        self.ax1.scatter(X_1,y_1)

        if self.checkbox1.isChecked():
            b, m = polyfit(X_1, y_1, 1)
            self.ax1.plot(X_1, b + m * X_1, '-', color="orange")

        vtitle = "Bike Share Count vs. " + cat1 + " 2017"
        self.ax1.set_title(vtitle)
        self.ax1.set_xlabel("Bike Count")
        self.ax1.set_ylabel(cat1)
        self.ax1.grid(True)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


class CorrelationPlot(QMainWindow):
    #;:-----------------------------------------------------------------------
    # This class creates a canvas to draw a correlation plot
    # It presents all the features plus the happiness score
    # the methods for this class are:
    #   _init_
    #   initUi
    #   update
    #::-----------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        #::--------------------------------------------------------
        super(CorrelationPlot, self).__init__()

        self.Title = 'Correlation Plot'
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  Creates the canvas and elements of the canvas
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Correlation Plot Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)


        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)
        self.feature6 = QCheckBox(features_list[6], self)
        self.feature7 = QCheckBox(features_list[7], self)
        self.feature8 = QCheckBox(features_list[8], self)
        self.feature9 = QCheckBox(features_list[9], self)
        self.feature10 = QCheckBox(features_list[10], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)
        self.feature7.setChecked(True)
        self.feature8.setChecked(True)
        self.feature9.setChecked(False)
        self.feature10.setChecked(False)

        self.btnExecute = QPushButton("Create Plot")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,0,2)
        self.groupBox1Layout.addWidget(self.feature3,0,3)
        self.groupBox1Layout.addWidget(self.feature4,1,0)
        self.groupBox1Layout.addWidget(self.feature5,1,1)
        self.groupBox1Layout.addWidget(self.feature6,1,2)
        self.groupBox1Layout.addWidget(self.feature7,1,3)
        self.groupBox1Layout.addWidget(self.feature8, 2, 0)
        self.groupBox1Layout.addWidget(self.feature9, 2, 1)
        self.groupBox1Layout.addWidget(self.feature10, 3, 0)
        self.groupBox1Layout.addWidget(self.btnExecute,3,1)


        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()


        self.groupBox2 = QGroupBox('Correlation Plot')
        self.groupBox2Layout= QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.groupBox2Layout.addWidget(self.canvas)


        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)

        self.setCentralWidget(self.main_widget)
        self.resize(900, 900)
        self.show()
        self.update()

    def update(self):

        #::------------------------------------------------------------
        # Populates the elements in the canvas using the values
        # chosen as parameters for the correlation plot
        #::------------------------------------------------------------
        self.ax1.clear()

        list_corr_features = pd.DataFrame()
        if self.feature0.isChecked():
            list_corr_features = pd.concat([list_corr_features, london_bikes[features_list[0]]],axis=1)

        if self.feature1.isChecked():
            list_corr_features = pd.concat([list_corr_features,london_bikes[features_list[1]]],axis=1)

        if self.feature2.isChecked():
            list_corr_features = pd.concat([list_corr_features, london_bikes[features_list[2]]],axis=1)

        if self.feature3.isChecked():
            list_corr_features = pd.concat([list_corr_features, london_bikes[features_list[3]]],axis=1)
        if self.feature4.isChecked():
            list_corr_features = pd.concat([list_corr_features, london_bikes[features_list[4]]],axis=1)

        if self.feature5.isChecked():
            list_corr_features = pd.concat([list_corr_features, london_bikes[features_list[5]]],axis=1)

        if self.feature6.isChecked():
            list_corr_features = pd.concat([list_corr_features, london_bikes[features_list[6]]],axis=1)

        if self.feature7.isChecked():
            list_corr_features = pd.concat([list_corr_features, london_bikes[features_list[7]]],axis=1)

        if self.feature8.isChecked():
            list_corr_features = pd.concat([list_corr_features, london_bikes[features_list[8]]],axis=1)

        if self.feature9.isChecked():
            list_corr_features = pd.concat([list_corr_features, london_bikes[features_list[9]]],axis=1)

        if self.feature10.isChecked():
            list_corr_features = pd.concat([list_corr_features, london_bikes[features_list[10]]],axis=1)



        vsticks = ["dummy"]
        vsticks1 = list(list_corr_features.columns)
        vsticks1 = vsticks + vsticks1
        res_corr = list_corr_features.corr()
        self.ax1.matshow(res_corr, cmap=plt.cm.Reds)
        #self.ax1.set_xticks(np.arange(len(list_corr_features)))
        #self.ax1.set_yticks(np.arange(len(list_corr_features)))
        self.ax1.set_yticklabels(vsticks1)
        self.ax1.set_xticklabels(vsticks1,rotation = 90)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()



if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    data_bike()
    main()