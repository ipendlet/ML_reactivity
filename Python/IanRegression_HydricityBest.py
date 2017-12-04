'''
Created on Oct 12, 2017
'''

import numpy as np
from pathlib import Path
from sklearn.linear_model.base import LinearRegression
import main
import scatterPlot
from sklearn.preprocessing.data import PolynomialFeatures
from sklearn.pipeline import make_pipeline

if __name__ == '__main__':
    pass
    dataFile = Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/AllData_CH3CN.csv')
    data = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(16,18,21,24,8,11,12,9,10))
    CoHOMO = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(68,70))
    therm = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(8,9,10))



    predictors = np.column_stack((CoHOMO[:,1]))


#    predictors = np.column_stack((data[:,0],data[:,2],data[:,3],data[:,4])) #LowdwinH2, Buried, VBuried, pka, r2=0.70951(hyd), r2(h2)=0.2226


#    predictors = np.column_stack((data[:,-4],data[:,-3],data[:,4])) #tau,tau,pka r2=0.7004262



#    hydricities = data[:,-1] #actually for H2binding
    hydricities = therm[:,1]

    # compound features
    polyFeatures = PolynomialFeatures(degree=1,interaction_only=True)
    regressor = make_pipeline(polyFeatures, LinearRegression())
#    regressor = LinearRegression()

    regressor.fit(predictors, hydricities)
    predictions = regressor.predict(predictors)
    print('R^2: ', regressor.score(predictors, hydricities))
    scatterPlot.plotScatterPlot(hydricities, predictions, (Path.home() / 'Desktop' / 'ianPredictions'))
#    print(regressor.coef_,regressor.intercept_)
    pass
