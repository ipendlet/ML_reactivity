'''
Created on Jan 17, 2017

@author: joshkamm
'''
import urllib.request
import pybel as pb
import openbabel as ob
import ob1
import json
from urllib.error import HTTPError

def generate_atom_feature_vectors(molecules):
    '''output atom environment feature vectors to ATOMS file for all unique atom environments
    from a list of pybel molecules
    '''
    atomVectors = []
    atomIndex = 0
    for currentMolecule in molecules:
        atomIndex = ob1.addallatoms(atomIndex,currentMolecule,atomVectors)
        
    # remove duplicates and sort
    atomVectors = map(list,list(set(map(tuple,atomVectors))))
    atomVectors = sorted(atomVectors)
    
    with open('ATOMS','w') as atomsFile:
        json.dump(atomVectors, atomsFile)

def generate_molecules_from_names():
    'take all of the atom names in moleculeNames.txt and generate pybel molecule objects from them'
    # read in file with molecule names
    with open('moleculeNames.txt','r') as file:
        moleculeNames = file.read().splitlines()
        
    molecules = []
    for moleculeName in moleculeNames:
        # use online database to convert molecule name into a smiles string
        urlName = urllib.request.pathname2url(moleculeName)
        try:
            smiles = urllib.request.urlopen('http://cactus.nci.nih.gov/chemical/structure/' + urlName
                                            + '/smiles').read().decode('utf-8')
            print(smiles)
            
            # create a pybel molecule based on the smiles string
            molecule = pb.readstring('smi', smiles)
            molecule.addh() # add hydrogens
            molecules.append(molecule)
        except HTTPError:
            print("Molecule: " + moleculeName + " failed")
    return molecules

if __name__ == '__main__':
    molecules = generate_molecules_from_names()
    generate_atom_feature_vectors(molecules)