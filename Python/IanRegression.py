'''
Created on Oct 12, 2017
'''

import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
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
    dataFile = Path.home() / 'Downloads' / 'DataForJosh.csv'




    lc = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(16,17,18))
    therm = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(8,9,10))
    bv = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(21,24))
    tau = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(11,12))



#    predictors = np.column_stack((lc[:,0],lc[:,1],lc[:,2],tau[:,0],tau[:,1]))
#    predictors = np.column_stack((lc[:,0],lc[:,1],lc[:,2],tau[:,0],tau[:,1]))

#### Hydricity prediction: Examples using the lowdin, tau and thermodynamic proporties ###
#    predictors = np.column_stack((lc[:,0]-lc[:,2],tau[:,0],tau[:,1],therm[:,0]))
#    predictors = np.column_stack((lc[:,0],lc[:,2],tau[:,0],tau[:,1],therm[:,0]))
    predictors = np.column_stack((lc[:,0],lc[:,2],bv[:,0],bv[:,1],therm[:,0]))


#    predictors = np.column_stack((lc[:,2]-lc[:,1],lc[:,0]-lc[:,2],bv[:,0],bv[:,1]))


#    predictors = np.column_stack((lc[:,0]-lc[:,2],bv[:,0],bv[:,1],therm[:,0]))

    hydricities = therm[:,1]

    # compound features
    polyFeatures = PolynomialFeatures(degree=2,interaction_only=True)
#    regressor = make_pipeline(polyFeatures, StandardScaler(), LassoCV(max_iter=2000))
    regressor = make_pipeline(polyFeatures, LassoCV(max_iter=2000))
#    regressor = make_pipeline(polyFeatures, StandardScaler(), LinearRegression())
#    regressor = make_pipeline(polyFeatures, LinearRegression())
#    regressor = LinearRegression()

    regressor.fit(predictors, hydricities)
    print(regressor.steps[1][1].coef_)
    print(regressor.steps[1][1].intercept_)
    print(regressor.steps[1][1].n_iter_)
    print(polyFeatures.get_feature_names())
    cross_val_score(, iris.data, iris.target, cv=cv)
#    print('R^2: ', regressor.score(predictors, hydricities))
    predictions = regressor.predict(predictors)
    scatterPlot.Hyd(hydricities, predictions, (Path.home() / 'Desktop' / 'ML_Figures' / 'ianPredictions'))
    pass 
