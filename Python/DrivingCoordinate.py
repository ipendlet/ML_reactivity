import numpy as np
from ml_atom import MLAtom
from enum import Enum

class DriveCoordType(Enum):
    ADD = 0
    BREAK = 1
    
class DrivingCoordinate:
    def __init__(self, Type=None, Atoms=None, NBO=None, Hybrid=None):
        '''
        Type = Add, Brk, None
        Atoms = pair of atoms being added or broken (expects MLAtom atoms)
        NBO = 2 element list of charges on the atoms involved in this driving coordinate
        Hybrid = 2 element list of p to s orbital occupancy ratio on each atom in this coordinate
        '''

        # Add error checking. Check for correct size and type of the input
        # variables
        if NBO is None or Hybrid is None:
            NBO = []
            Hybrid = []
        self._Type = Type
        self._Atoms = Atoms
        self._NBO = NBO
        self._Hybrid = Hybrid
        
    def chargeProduct(self):
        return self._NBO[0] * self._NBO[1]

    def sortByCharge(self):
        '''
        Order the atoms with the one with the lower charge first and maintain ordering consistency
        so that the first charge (_NBO[0]) and hybridization (_Hybrid[0]) values correspond to the
        first atom
        '''
        self._NBO, self._Atoms, self._Hybrid = zip(*sorted(zip(self._NBO,self._Atoms,
                                                               self._Hybrid),key=lambda x:x[0]))
    
    def build_atom_rep_feature_vec(self):
        'combine the features of the two atoms that make up this driving coordinate'
        featureVec = np.array([atom.build_atom_rep_feature_vec() for atom in self._Atoms])
        featureVec.sort(axis=0)
        return np.flip(featureVec, axis=0).reshape(-1)

    @staticmethod
    def atom_rep_feature_vec_size():
        return 2 * MLAtom.atom_rep_feature_vec_size()

        