'''
Created on Oct 12, 2017
'''

import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import main
import scatterPlot
from sklearn import preprocessing
from sklearn.preprocessing.data import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import scale
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from scipy import stats


if __name__ == '__main__':
    pass
    dataFile = Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/AllData_CH3CN.csv')

########### Features ###############
    name = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(0))
#Cobalt Charges
    NBO = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(29,30,31))
    lc = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(16,17,18))
    mc = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(13,14,15))
#Phosphorous Charges
    lcPH2 = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(56,57,58,59))
    lcPH = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(60,61,62,63))
    lcPnoH = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(64,65,66,67))
    mcPH2 = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(44,45,46,47))
    mcPH = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(48,49,50,51))
    mcPnoH = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(52,53,54,55))
    NBOPH2 = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(32,33,34,35))
    NBOPH = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(36,37,38,39))
    NBOPnoH = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(40,41,42,43))
#Steric Properties
   #Buriedvolume
    bv = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(23,26))
   #tau
    tau = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(11,12))
   #surface area 
    sa = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(19,21))
    sv = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(20,22))
   #surface area 
   #surface area 
#Key Thermodynamic Properties
    therm = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(8,9,10))
    hydricities = therm[:,1]
#PKA stuffs
    CoHOMO = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(68,70))
    CoH2Len = np.loadtxt(dataFile,delimiter=',',skiprows=2,usecols=(74,75,76,77)) ##76 is CoH and 77 is angles
 


#Main Figures if wanting to reproduce
    predictors = np.column_stack((NBO[:,0],NBO[:,1],NBO[:,2],lc[:,2],lc[:,1],lc[:,0],mc[:,0],mc[:,2],tau[:,0],tau[:,1],bv[:,0],bv[:,1],CoHOMO[:,0],CoH2Len[:,0],CoH2Len[:,1],CoH2Len[:,2],CoH2Len[:,3])) #kitchensink for all
#    predictors = np.column_stack((CoHOMO[:,1],therm[:,0]))  #Main Figures for text slide 1
#   predictors = np.column_stack((lc[:,1],bv[:,0],bv[:,1],tau[:,0],CoHOMO[:,0],CoH2Len[:,0]))  #Main Figures for text slide 1
#    predictors = np.column_stack((lc[:,0],lc[:,2],lc[:,1],tau[:,0],CoHOMO[:,0],CoH2Len[:,0],sa[:,0],sa[:,1],sv[:,1],sv[:,0],bv[:,0],bv[:,1]))  #Main Figures for text slide 1
#    predictors = np.column_stack((lc[:,0],lc[:,2],lc[:,1],tau[:,0],CoHOMO[:,0]))  #Main Figures for text slide 1







###Paper walkthrough ###





#######Training targets  ###
#    hydricities = CoHOMO[:,1]
#    hyduns = np.column_stack((therm[:,1])).reshape((-1,1))
#    hyduns = np.column_stack((therm[:,1])).reshape((-1,1))
#    scaler = StandardScaler()
#    hydricities2 = hydricities1.reshape((-1,1))
#    hydricities=scale(hydricities2)
#    print(hyd1)


    # compound features
    polyFeatures = PolynomialFeatures(degree=3,interaction_only=False)
    regressor = make_pipeline(polyFeatures, StandardScaler(), LassoCV(max_iter=60000, cv=KFold(n_splits=3, shuffle=True)))
#    regressor = make_pipeline(polyFeatures, StandardScaler(), Lasso(alpha=0, max_iter=70000))#, fit_intercept=True))
#    regressor = make_pipeline(polyFeatures, StandardScaler(), LinearRegression())
#    regressor = RandomForestRegressor(oob_score=True,n_estimators=2000)
    





####Make the output, print statements with detailed analysis of each step ##### 
    regressor.fit(predictors, hydricities)
    predictions = regressor.predict(predictors)
    print('R^2: ', regressor.score(predictors, hydricities))
    print('Intercept:',regressor.steps[2][1].intercept_)
 #   print('Intercept:',regressor.steps[2][1].intercept_)
 #   print('Intercept:',regressor.steps[2][1].intercept_)
#    print((regressor.steps[2][1].coef_).reshape((-1,1)))


### Print Functions
#    cv = ShuffleSplit(n_splits=9, test_size=0.1, random_state=None) 
#    scores=cross_val_score(regressor,predictors,hydricities,cv=cv)
#    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
#    print(scores)
    count=0
#    count2=0
#    for item in regressor.predict(predictors):
#        print(item, name[count2])
#        count2+=1
#    print('Intercept:',regressor.steps[2][1].intercept_)
#    print('LassoOptIter:',regressor.steps[2][1].n_iter_)
    fn=np.asarray(polyFeatures.get_feature_names())
    print(regressor.steps[2][1].sparse_coef_)
#    scaler=regressor.steps[1][1].scale_
#    for item in regressor.steps[2][1].coef_:
#        print(fn[count], item, scaler[count])#,polyFeatures.get_feature_names())
#        print(fn[count], item)#,polyFeatures.get_feature_names())
#        count+=1
#        count+=1

#    print(regressor.steps[2][1])
#    print(cross_val_score(predictors, hydricities))
#    predictions2 = predictions3
#    print(len(predictions2), len(hydricities))

#    print('Alpha',regressor.steps[2][1].alpha_)

 #   predictions3 =(regressor.predict(predictors))
 #   predictions2 = predictions3.reshape((-1,1)) 



#    regressor2 = LinearRegression()
#    regressor2.fit(predictors, hydricities)
#    predictions = regressor2.predict(predictors)
#    predictions2 = predictions.reshape((-1,1))
#    print('R^2(2): ', regressor2.score(predictions2, hydricities))
#    print("scalar=",regressor[2][1].coef_, "intercept=", regressor[2][1].intercept_)
#    predictions = regressor2.predict(predictions2)
 


#    print('R^2: ', regressor.score(predictors, hydricities))
#    print(regressor2.predict(predictors))
#    print(predictions)

#    slope, intercept, r_value, p_value, std_err = stats.linregress(hydricities,predictions)
#    print("linear Regression", r_value**2)


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
#    elif:
#        scatterPlot.h2binding(hydricities, predictions, Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/H2Binding'))
#        print('none-looks like H2')

#    print(Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/ianPredictions')) 
#    regressor = make_pipeline(polyFeatures, StandardScaler(), LassoCV(eps=1e-3, max_iter=60000, cv=KFold(n_splits=3, shuffle=True),tol=1e-10,selection='random'))