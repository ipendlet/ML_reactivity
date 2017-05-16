import glob
import json
import os
from pathlib import Path
from subprocess import run, call
import sys
from textwrap import dedent

from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.ridge import RidgeCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection._split import KFold
from sklearn.model_selection._validation import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.svm.classes import SVR
from sklearn.utils import shuffle
from sklearn.utils.estimator_checks import check_estimator
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

from autoen import Autoencoder
from bp_setup import archive_path
import main
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pybel as pb
import tensorflow as tf

class AutoencoderRegularizer():
    @classmethod
    def run(cls):
        if (len(sys.argv) == 2):
            'execute the autoencoder for the appropriate beta value'
            cls().run_autoencoder(float(sys.argv[1]))
        elif (len(sys.argv) == 1):
            'setup the jobs for the autoencoder'
            cls().setup_jobs()
    
    def get_path(self):
        return (Path.home() / 'Molecules' / 'ReactivityMLTestData' / 'AutoencoderRegularization')
    
    def setup_jobs(self):
        betaVals = np.logspace(-4,0,10)
        runPath = self.get_path() / 'LatestRun'
        archive_path(runPath)
        runPath.mkdir()
        
        for i, beta in enumerate(betaVals):
            # create a folder for each value of the regularization parameter
            path = (runPath / str(i))
            path.mkdir()
            os.chdir(str(path))
            self.submit_job(beta)
    
    def submit_job(self, beta):
        print("Current working directory:", Path.cwd())
        
        qshFilename = 'submit.qsh'
        with (Path.cwd() / qshFilename).open(mode='x') as qshFile:
            pbsDirectives = dedent('''\
            #PBS -l nodes=1:ppn=1 -l walltime=12:00:00
            #PBS -q zimmerman -N TensorFlow
            ''')
            print(pbsDirectives, file=qshFile)
            print('cd $PBS_O_WORKDIR', file=qshFile)
            print('python ~/ReactivityMachineLearning/Python/test_open_babel_ml.py',
                  beta, file=qshFile)
            
        # execute qsub command to submit job to the queue
        run(['qsub', qshFilename])
    
    def run_autoencoder(self, beta):
        'beta is l2 regularization parameter'
        data = shuffle(read_atoms_data(self.get_path() / 'ATOMS'), random_state=40)
        scaledData = data / 10 - 0.5
        
        with tf.Session() as sess:
            Autoencoder.tf_session = sess
            auto = Autoencoder()
            # not actually scanning over beta values within this function but validation curve utility is still useful
            train_scores, test_scores = validation_curve(auto, scaledData, y=None, param_name='beta', param_range=[beta], cv=5)
            with open('scores', 'x') as scoresFile:
                scores = [beta,-10.0*np.mean(train_scores),-10.0*np.mean(test_scores)]
                json.dump(scores, scoresFile)

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
    
#     svg = conv.WriteString(molecule)
#     print(svg)
#     emptyList = ['']
#     a = pb.ob.vectorString(['test.smi'])
#     b = pb.ob.vectorString(['b'])
#     c = pb.ob.stringbuf()
#     conv.FullConvert(['abc','def'],'testMolecules.png',['react' + str(i) + '.xyz' for i in range(1,7)])
#     conv.FullConvert(a,'',b)
#     print(b)
#     outPbFile = pb.Outputfile('svg','testMolecules.svg',overwrite=True)
#     for currentMolecule in molecules:
#         outPbFile.write(currentMolecule)
#     outPbFile.close()
    # print(spectrophores)

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
    
    with tf.Session() as sess:
        Autoencoder.tf_session = sess
#         for layer1Dim in range(6,16):
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
    
    # read in Paul's data
#     testDataFile = (Path.home() / 'Molecules' / 'ReactivityMLTestData' / 'MLPipelineTesting'
#                     / 'xydata_set1')
    testDataFile = (Path.home() / 'Box Sync' / 'Eecs545FinalProject' / 'data' / 'xydata_set1')
    reactions = main.readFromFile(testDataFile)
    
    # generate base features
    reactionData = np.array([reaction.buildFeatureVector(includeChargeMult=True) for reaction in reactions])
    targets = np.asarray([reaction._activationEnergy for reaction in reactions])
    
    # train / test split
    trainData, testData, trainTargets, testTargets = train_test_split(reactionData, targets)
    
    # make a pipeline with preprocessing, autoencoder, regression
    scaler = MinMaxScaler(feature_range=(-0.5,0.5))
    with tf.Session() as sess:
        Autoencoder.tf_session = sess
        autoencoder = Autoencoder(hiddenDims=[50,40],beta=0.1)
        regressor = SVR(C=10000)
        
        mlPipeline = make_pipeline(scaler, autoencoder, regressor)
#         mlPipeline = make_pipeline(autoencoder, regressor)
        
        mlPipeline.fit(trainData, trainTargets)
        trainPredictions = mlPipeline.predict(trainData)
        testPredictions = mlPipeline.predict(testData)
        
    # make plot of predicted values vs actual values
    main.plotScatterPlot(trainTargets, trainPredictions, Path.home() / 'Desktop' / 'PredictedVsActualTrain')
    main.plotScatterPlot(testTargets, testPredictions, Path.home() / 'Desktop' / 'PredictedVsActualTest')

if __name__ == '__main__':
    print('STARTED!')
#     autoencoder_dim_tuning_graph()
#     print(sys.path)
#     AutoencoderRegularizer().run()
#     AutoencoderRegularizer.plot_regularization_curve()
#     test_output_molecules(get_test_molecules())
    test_ml_pipeline()
    print('DONE WITHOUT ERROR')