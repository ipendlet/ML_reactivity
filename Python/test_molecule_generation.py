'''
Created on Jan 17, 2017

@author: joshkamm
'''
import urllib.request
import pybel as pb
import openbabel as ob
from urllib.error import HTTPError

if __name__ == '__main__':
    # read in file with molecule names
    with open('moleculeNames.txt','r') as file:
        moleculeNames = file.read().splitlines()
        
    for i, moleculeName in enumerate(moleculeNames):
        # use online database to convert molecule name into a smiles string
        urlName = urllib.request.pathname2url(moleculeName)
        try:
            smiles = urllib.request.urlopen('http://cactus.nci.nih.gov/chemical/structure/' + urlName
                                            + '/smiles').read().decode('utf-8')
            print(smiles)
            
            # use open babel to convert the smiles string into an XYZ file
            molecule = pb.readstring('smi', smiles)
            molecule.make3D(steps=50)
            molecule.write('xyz', 'react' + str(i) + '.xyz', overwrite=True)
        except HTTPError:
            print("Molecule: " + moleculeName + " failed")
    