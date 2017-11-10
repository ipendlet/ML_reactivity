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
#    predictors = np.column_stack((lc[:,0],lc[:,1],lc[:,2],bv[:,0],bv[:,1],tau[:,0],tau[:,1]))
    predictors = np.column_stack((lc[:,0],lc[:,2],bv[:,0],bv[:,1],therm[:,0]))
#    predictors = np.column_stack((bv[:,0],bv[:,1],therm[:,0]))
#    predictors = np.column_stack((therm[:,0])).reshape((-1,1))



#chosing training targets  ###
    hydricities = therm[:,1]

    # compound features
    polyFeatures = PolynomialFeatures(degree=2,interaction_only=True)
    regressor = make_pipeline(polyFeatures, StandardScaler(), LinearRegression())
#    regressor = make_pipeline(polyFeatures, StandardScaler(), LassoCV(max_iter=2000))
#    regressor = make_pipeline(polyFeatures, StandardScaler(), LassoCV(max_iter=6000, cv=5))
#    regressor = make_pipeline(polyFeatures, StandardScaler(), Lasso(alpha=0.0494993771664, max_iter=6000))
#    regressor = make_pipeline(polyFeatures, StandardScaler(), Lasso(alpha=1.0, max_iter=6000))


####Make the output, print statements with detailed analysis of each step ##### 
    regressor.fit(predictors, hydricities)
    count=0
    print('R^2: ', regressor.score(predictors, hydricities))
#    print('Alpha',regressor.steps[2][1].alpha_)
#    print('Intercept:',regressor.steps[2][1].intercept_)
#    print('LassoOptIter:',regressor.steps[2][1].n_iter_)
    fn=np.asarray(polyFeatures.get_feature_names())
    for item in regressor.steps[2][1].coef_:
        print(fn[count], item)#,polyFeatures.get_feature_names())
        count+=1
#    print(regressor.steps[2][1])
#    print(cross_val_score(predictors, hydricities))
    predictions = regressor.predict(predictors)


#### Graphing the model versus prediction section ####
    if hydricities[0]==therm[0,0]:
        scatterPlot.pka(hydricities, predictions, Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/PkaPrediction'))
        print('pKa')
    if hydricities[0]==therm[0,1]: 
        scatterPlot.Hyd(hydricities, predictions, Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/HydricityPrediction'))
        print('Hydricity')
    if hydricities[0]==therm[0,2]:
        scatterPlot.h2binding(hydricities, predictions, Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/H2Binding'))
        print('H2Binding')
    pass 
#    print(Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/ianPredictions'))