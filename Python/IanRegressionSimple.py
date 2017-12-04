'''
Created on Oct 12, 2017
'''

import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
import main
import scatterPlot
from sklearn import preprocessing
from sklearn.preprocessing.data import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

if __name__ == '__main__':
    pass
    dataFile = Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/Alldata_CH3CN.csv')
    CoHOMO = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(68,70))
    therm = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(8,9,10))
#    lc = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(16,17,18))
#    therm = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(8,9,10))
#    bv = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(21,24))
#    tau = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(11,12))

#### Hydricity prediction: Examples using the lowdin, tau and thermodynamic proporties ###
    predictors = np.column_stack((CoHOMO[:,1])).reshape((-1,1))


#chosing training targets  ###
    hydricities = therm[:,1]

   #simple feature analysis, linear regression
    regressor = make_pipeline(StandardScaler(), LinearRegression())
#    regressor = LinearRegression()

####Make the output, print statements with detailed analysis of each step ##### 
    regressor.fit(predictors, hydricities)
    predictions = regressor.predict(predictors)
    print('R^2: ', regressor.score(predictors, hydricities))
    print('coef: ', regressor.steps[1][1].coef_)
    print('intercept: ', regressor.steps[1][1].intercept_)



#### Graphing the model versus prediction section ####
    if hydricities[0]==therm[0,0]:
        scatterPlot.pka(hydricities, predictions, Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/PkaPrediction'))
    if hydricities[0]==therm[0,1]: 
        scatterPlot.Hyd(hydricities, predictions, Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/HydricityPrediction'))
    if hydricities[0]==therm[0,2]:
        scatterPlot.h2binding(hydricities, predictions, Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/H2Binding'))
    pass 