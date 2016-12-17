from subprocess import run, call
import numpy as np
import pybel as pb
from pathlib import Path
import os
import glob
from sklearn.neighbors import NearestNeighbors

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
    conv.SetInAndOutFormats('xyz','png')
    emptyList = ['']
    # conv.FullConvert(['abc','def'],'testMolecules.png',['react' + str(i) + '.xyz' for i in range(1,7)])
    outPbFile = pb.Outputfile('svg','testMolecules.svg',overwrite=True)
    for currentMolecule in molecules:
        pass
    #     outPbFile.write(currentMolecule)
    outPbFile.close()
    # print(spectrophores)

def get_test_molecules():
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

if __name__ == '__main__':
    test_nearest_neighbors()
    print('DONE WITHOUT ERROR')