from subprocess import run, call
import numpy as np
import pybel as pb
from pathlib import Path
import os
import glob

# try calculating a couple spectrophores using the command line utility
# run(['obspectrophore','-i','react1.xyz','-i','react2.xyz'])

# compute a sample spectrophore of a molecule using the pybel and open babel APIs
testDataPath = Path.home() / 'Molecules' / 'TestMLData'
os.chdir(str(testDataPath))
molecule = next(pb.readfile('xyz', 'react3.xyz', None))
spectrophoreCalculator = pb.ob.OBSpectrophore()
# print('Spectrophore:', spectrophoreCalculator.GetSpectrophore(molecule.OBMol))

# Try using open babel command line interface to create png's
print('Path:', os.environ['PATH'])
call('/Users/joshkamm/miniconda3/bin/obabel react*.xyz -O testMolecules.png -xd -xp 1000', shell=True)
run(['open', 'testMolecules.png'])

# compute some sample spectrophores and determine the nearest neighbors within them
molecules = []
spectrophores = []
# molecule.write('svg','testMolecules.svg',overwrite=True)
conv = pb.ob.OBConversion()
conv.SetInAndOutFormats('xyz','png')
emptyList = ['']
# conv.FullConvert(['abc','def'],'testMolecules.png',['react' + str(i) + '.xyz' for i in range(1,7)])
outPbFile = pb.Outputfile('svg','testMolecules.svg',overwrite=True)
for i in range(1,7):
    currentMolecule = next(pb.readfile('xyz', 'react' + str(i) + '.xyz', None))
    molecules.append(currentMolecule)
#     outPbFile.write(currentMolecule)
    spectrophores.append(spectrophoreCalculator.GetSpectrophore(currentMolecule.OBMol))
outPbFile.close()
# print(spectrophores)

# trying out stuff with using molecular fingerprints
# print('Available fingerprints:')
# print(pybel.fps)
# print('\nMolecule:')
# print(molecule.write('smi'))
# print(molecule.calcfp(fptype='maccs'))

# trying out SVD
# u,s,v = np.linalg.svd([[0.690715, 0.685874],[0.503314,0.792463]])
# print(u,'\n',s,'\n',v)

print('DONE WITHOUT ERROR')