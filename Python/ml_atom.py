'''
Created on Jun 8, 2017

@author: joshkamm
'''
import pybel as pb
import numpy as np

class MLAtom():
    '''
    Extension of pybel's atom type to return a representation of ML features
    '''
    def __init__(self, atom):
        self.atom = atom
        self.atom_rep_feature_vec = self.build_atom_rep_feature_vec()
    
    def build_atom_rep_feature_vec(self):
        '''
        build a feature vector consisting of atomic num, valence, and hybridization of this
        atom followed by the same info of neighboring atoms 
        '''
        try:
            return self.atom_rep_feature_vec
        except AttributeError:
            featureVec = []
            featureVec += self.basic_info(self.atom)
            atomFeatureVecs = []
            for connectedOBAtom in pb.ob.OBAtomAtomIter(self.atom.OBAtom):
                atomFeatureVecs.append(self.basic_info(pb.Atom(connectedOBAtom)))
                bond = connectedOBAtom.GetBond(self.atom.OBAtom)
                atomFeatureVecs[-1].append(bond.GetBondOrder())
            
            # sort by descending atomic num, then other basic info
            atomFeatureVecs.sort(reverse=True)
            for atomFeatureVec in atomFeatureVecs:
                featureVec += atomFeatureVec
            array = np.array(featureVec)
            array.resize(3+4*4, refcheck=False) # fixed size of feature vector
            self.atom_rep_feature_vec = array
            return self.atom_rep_feature_vec
            
    def basic_info(self, atom):
        features = []
        features.append(atom.atomicnum)
        features.append(atom.valence)
        features.append(atom.hyb)
        return features

        