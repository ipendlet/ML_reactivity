'''
Created on Oct 12, 2017
'''

import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
import main
import scatterPlot
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
#    print('R^2: ', regressor.score(predictors, hydricities))
    predictions = regressor.predict(predictors)
    scatterPlot.Hyd(hydricities, predictions, (Path.home() / 'Desktop' / 'ML_Figures' / 'ianPredictions'))
   
















 
    pass















#    data2 = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(0))
#    predictors = np.column_stack((data[:,0]-data[:,1],(data[:,2]+data[:,3])/2,data[:,4],data[:,5],data[:,6]))
#    predictors = np.column_stack((data[:,0]-data[:,1],data[:,2],data[:,3],data[:,4]))

#    predictors = np.column_stack((data[:,0],data[:,2],data[:,3],data[:,4])) #LowdwinH2, Buried, VBuried, pka, r2=0.70951(hyd), r2(h2)=0.2226

#    predictors = np.column_stack((data[:,-4],data[:,-3],data[:,4])) #tau,tau,pka r2=0.7004262


#startpka predict#   predictors = np.column_stack((data[:,-1]-data[:,2],data[:,2],data[:,3],data[:,6],data[:,7]))

#    predictors = np.column_stack((data[:,-1]-data[:,0],data[:,5],data[:,6],(data[:,0]-data[:,1]),(data[:,1]-data[:,-1])))
#    X = np.matrix([0,1,2,3,4,5,6,7,8,9,10]).reshape((11,1))

#    predictors = np.column_stack((lc[:,0],lc[:,1],lc[:,2],bv[:,0],bv[:,1]))#,tau[:,0],tau[:,1]))
#    print(regressor.fit_transform(X))
