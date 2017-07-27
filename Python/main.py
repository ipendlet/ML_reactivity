import sys
import fileinput
from DrivingCoordinate import DrivingCoordinate, DriveCoordType
from Reaction import Reaction
import re
import numpy as np
from sklearn import linear_model as lm, svm
from sklearn import preprocessing as pre
import matplotlib as mpl
from sklearn.metrics.regression import r2_score
from sklearn.model_selection._split import KFold
from sklearn.model_selection._search import GridSearchCV
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

def getNumOfLines(fName):
    'return the number of lines in the file named fName'
    with open(fName) as fh:
        for i, l in enumerate(fh):
            pass
    return (i + 1)

def readFromFile(fName):
    'parse the type of reaction info file that Paul Zimmerman provided into Reaction objects'
    # TODO: add error checking
    counter = 0
    typeAdd = []
    addAtom1 = []
    addAtom2 = []
    typeBrk = []
    brkAtom1 = []
    brkAtom2 = []
    reactions = []
    for line in fileinput.input(str(fName)):
        # collapse multiple whitespaces into one
        lineSpaceCollapsed = re.sub( '\s+', ' ', line).strip()
        # split string based on spaces
        lineSpaceSplit = lineSpaceCollapsed.split()
        if len(line.strip()) != 0: # ignore empty lines
            # 1st line of data point
            if fileinput.lineno() == (1+counter*5):
                pntNum = int(lineSpaceSplit[1])
                idNum = int(lineSpaceSplit[3])
                activationEnergy = float(lineSpaceSplit[5])
                EofRxn = float(lineSpaceSplit[7])
                reactions.append(Reaction(idNum,activationEnergy,EofRxn))
            # 2nd line of data point
            if fileinput.lineno() == (2+counter*5):
                numOfAddMoves = len(lineSpaceSplit) - 1
                for i in range(numOfAddMoves):
                    typeAdd.append(lineSpaceSplit[0])
                    addAtom1.append(lineSpaceSplit[i+1].split('-')[0])
                    addAtom2.append(lineSpaceSplit[i+1].split('-')[1])
            # 3rd line of data point
            if fileinput.lineno() == (3+counter*5):
                numOfBrkMoves = len(lineSpaceSplit) - 1
                for i in range(numOfBrkMoves):
                    typeBrk.append(lineSpaceSplit[0])
                    brkAtom1.append(lineSpaceSplit[i+1].split('-')[0])
                    brkAtom2.append(lineSpaceSplit[i+1].split('-')[1])
            # 4th line of data point
            if fileinput.lineno() == (4+counter*5):
                addNBO = [float(elem) for elem in lineSpaceSplit[0:10]]
                addHybrid = [float(elem) for elem in lineSpaceSplit[10:20]]
                brkNBO = [float(elem) for elem in lineSpaceSplit[20:30]]
                brkHybrid = [float(elem) for elem in lineSpaceSplit[30:40]]
                for i in range(len(typeAdd)):
                    reactions[-1].addDrivingCoordinate(DrivingCoordinate(Type=DriveCoordType.ADD,
                            Atoms=[addAtom1[i],addAtom2[i]], NBO=addNBO[2*i:2*i+2],
                            Hybrid=addHybrid[2*i:2*i+2]))
                    #print ("add", reactions[-1]._drivingCoordinates[-1].__dict__)
                for i in range(len(typeBrk)):
                    reactions[-1].addDrivingCoordinate(DrivingCoordinate(Type=DriveCoordType.BREAK,
                            Atoms=[brkAtom1[i], brkAtom2[i]], NBO=brkNBO[2*i:2*i+2],
                            Hybrid=brkHybrid[2*i:2*i+2]))
                    #print ("brk", reactions[-1]._drivingCoordinates[-1].__dict__)
                maxMovesOfType = 5 # only keeping reactions with <= 5 add moves and <= 5 break moves
                if max(len(reactions[-1].movesOfType('add')), len(reactions[-1].movesOfType('break'))) > maxMovesOfType:
                    reactions.remove(reactions[-1])
        # 5 comes from the number of lines that compose a data point
        if (fileinput.lineno()%5 == 0):
            counter += 1
            # each data point has 5 lines, so reset for each data point
            typeAdd = []
            addAtom1 = []
            addAtom2 = []
            typeBrk = []
            brkAtom1 = []
            brkAtom2 = []

    return reactions

def logisticRegression(data,targets):
    logistic = lm.LogisticRegression()
    logistic.fit(data,targets)
    pass #TODO: MAYBE FINISH THIS
    
def linearRegression(data, targets):
    numOfDataPnts = data.shape[0]
    scores = []
    sixFoldCrossValid = KFold(n=numOfDataPnts, n_folds=6, shuffle=True, random_state=None)
    i = 0
    for train_index, test_index in sixFoldCrossValid:
        # split the data into training and testing sets
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = targets[train_index], targets[test_index]
        
        # execute the underlying linear regression
        linRegress = lm.LinearRegression()
        linRegress.fit(data_train, target_train)
        targetPredicted = linRegress.predict(data_test)
        scores.append(linRegress.score(data_test, target_test))
        
        # make plot of true vs predicted reactivity
        plt.scatter(targetPredicted, target_test) #, color='b', s=121/2, alpha=.4)
        axes = plt.gca()
        slope, intercept, r_value, p_value, std_err = stats.linregress(targetPredicted,target_test)
        rSquared = r_value**2
        plt.annotate(str(rSquared), xy=(1,4), xytext=(1, 4), textcoords='figure points')
        m, b = np.polyfit(targetPredicted, target_test, 1)
        X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
        plt.plot(X_plot, m*X_plot + b, '-')
        plt.xlabel('Predicted values')
        plt.ylabel('True values')
        plt.title('Scatter plot of true vs. predcited values')
        plt.savefig('linReg%i.png' % i)
        plt.clf()
        i += 1
    meanScore = np.mean(scores)
    print ("prediction score: ", meanScore)
    return meanScore

def ridgeRegression(data, targets, alphaVals):
    ridge = lm.RidgeCV(alphas=alphaVals,store_cv_values=True)
    ridge.fit(data, targets)
    print('alpha: ' + str(ridge.alpha_))
    optimalScore = ridge.score(data, targets)
    targetPredicted = ridge.predict(data)
    print ("prediction optimalScore: " , optimalScore)
    scores = np.sqrt(np.mean(ridge.cv_values_,axis=0)) 
    return optimalScore, scores, targetPredicted
    
def lassoRegression(data, targets):
    lasso = lm.LassoCV(max_iter=10000)
    lasso.fit(data, targets)
    print(lasso.coef_)
    score = lasso.score(data, targets)
    targetPredicted = lasso.predict(data)
    print ("prediction score: " , score)
    return score, targetPredicted
    
def supportVectorRegression(data, targets):
    data = pre.scale(data)
    svr = svm.SVR()
#     svr.fit(data, targets)
#     param_grid = [{'C': np.logspace(0,4,num=4), }]
    param_grid = [
        {'C': [1, 10, 100, 1000, 10000], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']},
     ]
#         {'C': [100, 1000], 'gamma': [0.01], 'kernel': ['rbf']},
#         {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    svrGrid = GridSearchCV(svr, param_grid=param_grid, n_jobs=-1)
    svrGrid.fit(data, targets)
    targetPredicted = svrGrid.predict(data)
    score = svrGrid.score(data, targets)
    print ("best parameters: ", svrGrid.best_params_)
    print ("prediction score: " , score)
    return score, targetPredicted
#     print ("prediction score: " , cv.cross_val_score(svr, data, targets).mean())

def plotBarGraph(inputX, inputY, labels):
    '''make a bar graph comparing the accuracy of the various machine learning algorithms
    and feature sets'''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(inputX))
    width = 0.2
    #numOfBars = len(labels)
    barsOrig = ax.bar(ind, inputY[0:4], width, color='blue')
    barsCharge = ax.bar(ind+width, inputY[4:8], width, color='red')
    barsChargeMove = ax.bar(ind+2*width, inputY[8:12], width, color='yellow')
    barsOrdered = ax.bar(ind+3*width, inputY[12:16], width, color='green')
	
    ax.set_xlim(-width,len(ind)+width)
    ax.set_ylim(-0.2,1)
    ax.set_ylabel('Scores')
    ax.set_title('Scores by regression method and feature matrix')
    xTickMarks = [name for name in inputX]
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=10, fontsize=10)

    ax.legend( (barsOrig[0], barsCharge[0], barsChargeMove[0], barsOrdered[0]), (labels[0], labels[1], labels[2], labels[3]), loc=4)
    plt.savefig("figure.pdf")

def plotRegularizationGraph(alphaVals, regScores):
    '''make a plot of training and test accuracy over a range of regularization
    parameters (alphaVals) used in ridge regression'''
    for name, currentScores in sorted(regScores.items()):
        plt.semilogx(alphaVals, currentScores, label=name)
    plt.title('Optimization of ridge regularization parameter\nby cross validation')
    plt.xlabel('Regularization parameter')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.savefig('Regularization curve.png')

def plotScatterPlot(actual, predicted, outFileName):
    'Make a scatter plot showing the predicted vs actual activation energy for each reaction'
    plt.scatter(actual, predicted, s=8)
    axes = plt.gca()
    
    # make plot square with equal x and y axes
    bounds = [min(list(actual) + list(predicted) + [0])-1, max(list(actual) + list(predicted))+1]
    plt.axis(bounds * 2)
    axes.set_aspect('equal', adjustable='box')
    
    # plot the identity for visual reference (10% darker than data)
    plt.plot([bounds[0], bounds[1]], [bounds[0], bounds[1]], color='#065E9B')

    rSquared = r2_score(actual, predicted)
    print(rSquared)
    plt.figtext(0.1,0.01,'$R^2 = $'+format(rSquared,'.4f'))
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('Scatter plot of true vs. predicted values')
    plt.tight_layout()
    plt.savefig(str(outFileName) + '.png')
    plt.clf()

def main():
    # TODO: add error checking
    fileName = sys.argv[1]
    
    reactions = readFromFile(fileName)
    #numFeatures = 40
    targets = np.asarray([reaction._activationEnergy for reaction in reactions])
    
    # create and initialize data matrices for each variant on the feature set
    data = {}
    data['hybrid & charge'] = np.asarray([reaction.buildFeatureVector() for reaction in reactions])
    data['hybrid & charge w/ products'] = np.asarray([reaction.buildFeatureVector(includeChargeMult=True) for reaction in reactions])
    data['hybrid, add/break, &\ncharge w/ products'] = np.asarray([reaction.buildFeatureVector(includeChargeMult=True, includeAddBreak=True) for reaction in reactions])
    data['ordered features'] = np.asarray([reaction.buildOrderedFeatureVector() for reaction in reactions])
    '''
    data['only original feature set'] = np.asarray([reaction.buildFeatureVector() for reaction in reactions])
    data['include charge multiplication information'] = np.asarray([reaction.buildFeatureVector(includeChargeMult=True) for reaction in reactions])
    data['include charge multiplication and add / break move information'] = np.asarray([reaction.buildFeatureVector(includeChargeMult=True, includeAddBreak=True) for reaction in reactions])
    '''

    labels = []
    inputX = ['Linear regression', 'Ridge regression', 'Lasso regression', 'Support vector regression' ]
    inputY = []
    alphaScores = {}
    alphaVals = np.logspace(-20,2)
    
    # execute each of the machine learning algorithms and plot 
    for name, matrix in sorted(data.items()):
        labels.append(name)
        print(name)
        print('Linear regression: ')
        score = linearRegression(matrix, targets)
        inputY.append(score)
        print('Ridge regression: ')
        score, alphaScores[name], targetPredicted = ridgeRegression(matrix, targets, alphaVals=alphaVals)
        plotScatterPlot(targets, targetPredicted, 'RidgeRegression-'+name.replace('/', ''))
        inputY.append(score)
        print('Lasso regression: ')
        score, targetPredicted = lassoRegression(matrix, targets)
        plotScatterPlot(targets, targetPredicted, 'LassoRegression-'+name.replace('/', ''))
        inputY.append(score)
        print('Support vector regression: ')
        score, targetPredicted = supportVectorRegression(matrix, targets)
        plotScatterPlot(targets, targetPredicted, 'SupportVectorRegression-'+name.replace('/', ''))
        inputY.append(score)
    plotRegularizationGraph(alphaVals, alphaScores)
    plotBarGraph(inputX, inputY, labels)

if __name__ == "__main__":
    main()
