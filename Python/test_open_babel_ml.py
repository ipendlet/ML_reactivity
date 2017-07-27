from builtins import range
import json
import os
from pathlib import Path
from subprocess import run, call
from textwrap import dedent

from sklearn.metrics.regression import mean_absolute_error, r2_score
from sklearn.model_selection._split import KFold, train_test_split
from sklearn.model_selection._validation import cross_val_score, \
    cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.svm.classes import SVR

from autoen import Autoencoder
from bp_setup import archive_path
import main
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pybel as pb
from reactivity_data_loader import ReactivityDataLoader
from reactivity_data_explorer import ReactivityDataExplorer
from DrivingCoordinate import DriveCoordType

mpl.use('Agg')

class MLParameterTuner():
    dims = list(range(2,51,4))
    
    @classmethod
    def run(cls, executionStage='submit_training'):
        if (executionStage == 'submit_training'):
            'submit PBS jobs for individual autoencoder dimensions'
            cls().submit_training_jobs()
        elif (executionStage == 'compile_results'):
            'execute training and testing for a particular autoencoder dimension'
            cls().compile_results()
        else: raise ValueError('Invalid value for executionStage')    
    
    def get_path(self, dim=None):
        'returns the appropriate directory on athena for these jobs or a particular job'
        path = (Path.home() / 'Molecules' / 'ReactivityMLTestData' / 'MLPipelineTesting'
                / 'LatestRun')
        if(dim != None): path /= 'Dim' + str(dim)
        return path

    def submit_training_jobs(self):
        archive_path(self.get_path())
        self.get_path().mkdir()
        
        for dim in MLParameterTuner.dims:
            # create a folder and submit a job for each value of the autoencoder dimension
            self.get_path(dim).mkdir()
            os.chdir(str(self.get_path(dim)))
        
            print("Current working directory:", Path.cwd())
            qshFilename = 'submit.qsh'
            with (Path.cwd() / qshFilename).open(mode='x') as qshFile:
                pbsDirectives = dedent('''\
                #PBS -l nodes=1:ppn=1 -l walltime=1:00:00
                #PBS -q zimmerman -N TensorFlow
                ''')
                print(pbsDirectives, file=qshFile)
                print('cd $PBS_O_WORKDIR', file=qshFile)
                print("python -c 'import test_open_babel_ml; test_open_babel_ml.MLParameterTuner().individual_training_executor("
                      + str(dim) + ")'",
                      file=qshFile)
                
            # execute qsub command to submit job to the queue
            run(['qsub', qshFilename])

    def individual_training_executor(self, dim):
        # make a pipeline with preprocessing, autoencoder, regression
        scaler = MinMaxScaler(feature_range=(-0.5,0.5))
        autoencoder = Autoencoder(logPath=self.get_path(dim), hiddenDims=[50,dim],beta=0.1)
        mlPipeline = make_pipeline(scaler, autoencoder)
        
        # read in the data and train the autoencoder
        data, targets = self.read_mopac_reactivity_data()
        mlPipeline.fit(data, targets)
        
        # test the accuracy of an SVM on the transformed data using cross validation
        latent = mlPipeline.transform(data)
        regressor = SVR(C=10000)
        cross_validator = KFold(n_splits=5, shuffle=True, random_state=40)
        predictions = cross_val_predict(regressor, latent, targets, cv=cross_validator)
        
        # make a cross_val_predict-ed vs actual graph
        main.plotScatterPlot(targets, predictions, 'predictedVsActual')
        
        # print the cross validation actual and predicted targets to file
        actualThenPredicted = np.array([targets, predictions])
        np.savetxt('actualThenPredicted.txt', actualThenPredicted)
                
    def compile_results(self):
        # compute the average absolute error and R^2 at each autoencoder dimension
        avgAbsErrors = []
        r2Values = []
        for dim in MLParameterTuner.dims:
            actualThenPredicted = np.loadtxt(str(self.get_path(dim) / 'actualThenPredicted.txt'))
            avgAbsErrors.append(mean_absolute_error(actualThenPredicted[0], actualThenPredicted[1]))
            r2Values.append(r2_score(actualThenPredicted[0], actualThenPredicted[1]))
            
        # make plots with matplotlib
        plt.plot(MLParameterTuner.dims, avgAbsErrors)
        plt.ylim(0,20)
        plt.title('ML pipeline tuning: scan over autoencoder dimension')
        plt.xlabel('Latent representation dimension')
        plt.ylabel('SVM cross validation avg absolute error')
        plt.savefig(str(Path.home() / 'Desktop' / 'AvgAbsErrorVsDim.png'))
        plt.gcf().clear()
        
        plt.plot(MLParameterTuner.dims, r2Values)
        plt.ylim(0,1)
        plt.title('ML pipeline tuning: scan over autoencoder dimension')
        plt.xlabel('Latent representation dimension')
        plt.ylabel('SVM cross validation R^2')
        plt.savefig(str(Path.home() / 'Desktop' / 'R2VsDim.png'))
        
    @classmethod
    def plot_regularization_curve(cls):
        'take errors from a collection of TensorFlow runs and produce test vs train error graph'
        betaVals = []
        trainScores = []
        testScores = []
        for dir in (cls().get_path() / 'LatestRun').iterdir():
            os.chdir(str(dir))
            with open('scores', 'r') as scoresFile:
                scores = json.load(scoresFile)
            betaVals.append(scores[0])
            trainScores.append(scores[1])
            testScores.append(scores[2])
        betaVals, trainScores, testScores = zip(*sorted(zip(betaVals,trainScores,testScores)))
        plt.loglog(betaVals, trainScores, label="Training error")
        plt.loglog(betaVals, testScores, label="Test error")
        plt.title('Regularization curve for autoencoder')
        plt.xlabel('Regularization coefficient')
        plt.ylabel('RMSE')
        plt.legend(loc='best')
        plt.savefig(str(Path.home() / 'Desktop' / 'RegularizationCurve.png'))

def test_spectrophores():
    # compute sample spectrophores of test molecules using the pybel and open babel APIs
    testDataPath = Path.home() / 'Molecules' / 'TestMLData'
    os.chdir(str(testDataPath))
    spectrophoreCalculator = pb.ob.OBSpectrophore()
    spectrophores = []
    for currentMolecule in get_test_molecules():
        spectrophores.append(spectrophoreCalculator.GetSpectrophore(currentMolecule.OBMol))
        # print('Spectrophore:', spectrophoreCalculator.GetSpectrophore(molecule.OBMol))
    return spectrophores

def test_png_create_command_line():
    "Try using open babel command line interface to create png's"
    print('Path:', os.environ['PATH'])
    call('/Users/joshkamm/miniconda3/bin/obabel react*.xyz -O testMolecules.png -xd -xp 1000', shell=True)
    run(['open', 'testMolecules.png'])

def test_output_molecules(molecules):
    # compute some sample spectrophores and determine the nearest neighbors within them
    # molecule.write('svg','testMolecules.svg',overwrite=True)
    conv = pb.ob.OBConversion()
    conv.SetInAndOutFormats('smi','svg')
    conv.OpenInAndOutFiles('test.smi','test.svg')
#     conv.Convert()
    molecule = pb.ob.OBMol()
    molecule2 = pb.ob.OBMol()
    conv.ReadString(molecule,'C1=CC=CS1')
    conv.ReadString(molecule2,'CC')
    
    outFormat = conv.GetOutFormat()
    
    conv.AddChemObject(molecule)
    conv.SetOneObjectOnly(False)
#     outFormat.WriteChemObject(conv)
    
    conv.AddChemObject(molecule2)
    conv.SetOneObjectOnly(True)
    outFormat.WriteChemObject(conv)
    
def get_test_molecules():
    os.chdir(str(Path.home() / 'Molecules' / 'TestMLData'))
    molecules = []
    for i in range(1,7):
        currentMolecule = next(pb.readfile('xyz', 'react' + str(i) + '.xyz', None))
        molecules.append(currentMolecule)
    return molecules

def test_nearest_neighbors():
    'trying out computing nearest neighbors on spectrophore data for some sample molecules'
    nearNeighbors = NearestNeighbors(1).fit(test_spectrophores())
    print(nearNeighbors.kneighbors())

def test_molecular_fingerprints(molecule):
    'trying out stuff with using molecular fingerprints'
    print('Available fingerprints:')
    print(pb.fps)
    print('\nMolecule:')
    print(molecule.write('smi'))
    print(molecule.calcfp(fptype='maccs'))

def autoencoder_dim_tuning_graph():
    '''run the autoencoder with a variety of hidden layer dimensionalities and plot the cross
    validation errors for each
    '''

    data = read_atoms_data()
    scaledData = data / 10 - 0.5
    kFold = KFold(n_splits=5, shuffle=True)
    errors = []
    
#     for layer1Dim in range(6,16):
    for layer1Dim in range(4,5):
        print('LAYER 1 DIMENSIONALITY: ', layer1Dim)
        errors.append([])
        latentLayerDims = range(4,layer1Dim+1)
        for latentLayerDim in latentLayerDims:
            auto = Autoencoder(hiddenDims=[layer1Dim,latentLayerDim])
            errors[-1].append(-10.0 * np.mean(cross_val_score(auto, scaledData, cv=kFold)))
        plt.semilogy(latentLayerDims,errors[-1],label=layer1Dim)
    print(errors)
    # create plot of errors and write them to a file in case I need to tweak the plot
#     plt.title('Searching for intrinsic dimensionality of sample data')
#     plt.xlabel('Dimensionality of latent representation')
#     plt.ylabel('Scaled data reconstruction error')
#     plt.legend(title='Hidden layer dimensionality')
#     plt.savefig('IntrinsicDimensionality.png')
#     with open('autoencoder_scores.txt', 'w') as file:
#         json.dump(errors, file)

def read_atoms_data(filename='ATOMS'):
    with open(str(filename), 'r') as file:
        atomsData = json.load(file)
    return np.array(atomsData)

def test_ml_pipeline():
    'load a test data set, run SVM on it, and plot the predictions vs the actual values'
    data, targets = ReactivityDataLoader().load_mopac_learning()
    regressor = SVR(C=1000)
    trainData, testData, trainTargets, testTargets = train_test_split(data, targets)
    regressor.fit(trainData, trainTargets)
    os.chdir(str(Path.home() / 'Desktop'))
    main.plotScatterPlot(testTargets, regressor.predict(testData), 'predictedVsActual')

def test_reactivity_data_analysis():
    'testing function for exploring and visualizing reactivity data'
    explorer = ReactivityDataExplorer(ReactivityDataLoader().load_mopac_learning(genFeatures=False))
    explorer.plot_coord_num_dist_for_element_and_move_type(7, DriveCoordType.ADD)
    
if __name__ == '__main__':
    print('Test ML pipeline version 0.1.4')
#     MLParameterTuner().compile_results()
    test_ml_pipeline()
#     test_reactivity_data_analysis()
#     autoencoder_dim_tuning_graph()
#     print(sys.path)
#     MLParameterTuner().run(executionStage='compile_results')
#     test_output_molecules(get_test_molecules())
    print('DONE WITHOUT ERROR')