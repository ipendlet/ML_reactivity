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
    dataFile = Path.home() / 'Downloads' / 'DataForJosh.csv'
    data = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(16,18,21,24,8,11,12,9,10))
#    predictors = np.column_stack((data[:,0]-data[:,1],(data[:,2]+data[:,3])/2,data[:,4],data[:,5],data[:,6]))
    predictors = np.column_stack((data[:,0]-data[:,1],data[:,2],data[:,3],data[:,4]))
#    predictors = np.column_stack((data[:,4]))


#    predictors = np.column_stack((data[:,0],data[:,2],data[:,3],data[:,4])) #LowdwinH2, Buried, VBuried, pka, r2=0.70951(hyd), r2(h2)=0.2226


#    predictors = np.column_stack((data[:,-4],data[:,-3],data[:,4])) #tau,tau,pka r2=0.7004262



#    hydricities = data[:,-1] #actually for H2binding
    hydricities = data[:,-2]

    # compound features
    polyFeatures = PolynomialFeatures(degree=2,interaction_only=True)
    regressor = make_pipeline(polyFeatures, LinearRegression())
#    regressor = LinearRegression()

    regressor.fit(predictors, hydricities)
    predictions = regressor.predict(predictors)
    print('R^2: ', regressor.score(predictors, hydricities))
    scatterPlot.plotScatterPlot(hydricities, predictions, (Path.home() / 'Desktop' / 'ianPredictions'))
#    print(regressor.coef_,regressor.intercept_)
    pass
