import numpy as np
from itertools import combinations_with_replacement
from DrivingCoordinate import DriveCoordType, DrivingCoordinate

class Reaction():
    '''
    A class representing a chemical reaction
    '''

    def __init__(self, id=None, activationEnergy=None, heatOfRxn=None, reactants=None):
        '''
        Constructor
        '''
        self._possibleAtoms = 'BCHNO'
        self._id = id
        self._activationEnergy = activationEnergy
        self._heatOfRxn = heatOfRxn
        self._drivingCoordinates = []
        self._reactants = reactants # currently only expecting a single pybel molecule
        # TODO: add miscellaneous values (the ones in the dataset not associated with an
        # add or break move)

    def addDrivingCoordinate(self, drivingCoordinate):
        self._drivingCoordinates.append(drivingCoordinate)

    def sortDrivingCoordinates(self):
        '''
        Sort the atoms within each driving coordinate and then sort the driving coordinates
        by the lower of the 2 charges associated with each
        '''
        addMoves = self.movesOfType(DriveCoordType.ADD)
        breakMoves = self.movesOfType(DriveCoordType.BREAK)
        for addMove in addMoves:
            addMove.sortByCharge()
        for breakMove in breakMoves:
            breakMove.sortByCharge()
        addMoves = sorted(addMoves, key=lambda x : x._NBO[0])
        breakMoves = sorted(breakMoves, key=lambda x : x._NBO[0])
        
        return addMoves, breakMoves

    def movesOfType(self, type):
        '''
        return all driving coordinates of argument type
        '''
        return list(filter(lambda x : x._Type == type,self._drivingCoordinates))
    
    def buildFeatureVector(self,includeChargeMult=False,includeAddBreak=False,isSorted=True):
        '''
        Builds a feature vector containing data associated with this Reaction
        includeChargeMult: whether to include the product of the 2 charges in each driving
            coordinate as features (5 add + 5 break driving coordinates = 10 features)
        inculdeAddBreak: whether to include the existance of each possible pair of elements
            as a one-hot (binary) feature (num elements choose 2 features). If the pair of
            elements appears in an add or break move, the feature value is 1 (otherwise 0)
        isSorted: whether to sort the driving coordinates before constructing the feature vector
        '''
        # up to 40+10=50 features because of hard limit on add and break moves of 5 and 4 features per move
        if isSorted:
            addMoves, breakMoves = self.sortDrivingCoordinates()
        else:
            addMoves = self.movesOfType(DriveCoordType.ADD)
            breakMoves = self.movesOfType(DriveCoordType.BREAK)
        
        featureVector = np.zeros((40))
        featureVecChargeMult = np.zeros((10))
        for i, addMove in enumerate(addMoves):
            featureVector[2*i:2*(i+1)] = addMove._NBO
            featureVector[10+2*i:10+2*(i+1)] = addMove._Hybrid
            featureVecChargeMult[i] = addMove._NBO[0] * addMove._NBO[1]
        
        for i, breakMove in enumerate(breakMoves):
            #print(breakMove._NBO)
            featureVector[20+2*i:20+2*(i+1)] = breakMove._NBO
            featureVector[30+2*i:30+2*(i+1)] = breakMove._Hybrid
            featureVecChargeMult[5+i] = breakMove._NBO[0] * breakMove._NBO[1]
        #print ("feature vector", featureVector)
        #print ("charge", featureVecChargeMult)
        if includeChargeMult:
            featureVector = np.concatenate((featureVector, featureVecChargeMult))
        if includeAddBreak:
            featureVector = np.concatenate((featureVector, self.buildAddBrkFeatureVector()))
        return featureVector

    def buildAddBrkFeatureVector(self):
        possibleBonds = list(combinations_with_replacement(self._possibleAtoms, 2))
        featureVector = np.zeros((len(possibleBonds)))
        for i, bond in enumerate(possibleBonds):
            for coordinate in self._drivingCoordinates:
                if (coordinate._Atoms[0] == bond[0] and coordinate._Atoms[1] == bond[1]) or\
                   (coordinate._Atoms[0] == bond[1] and coordinate._Atoms[1] == bond[0]):
                    featureVector[i] = 1
        return featureVector

    def buildOrderedFeatureVector(self):
        '''
        Builds an alterantive feature vector to buildFeatureVector. An ordering of possible
        pairs of elements is chosen and two binary features (one for add and one for break) are
        created for each pair of elements corresponding to whether the reaction contains a move
        of the given type between the pair of elements. For each of these features a corresponding
        feature is created to contain the charge product of the elements in the add or break move.
        
        Example: if the 5th feature corresponds to whether there is an add move between carbon
        and hydrogen and there are 15 of this type of feature, the 20th feature correpsonds to the
        charge product of the carbon and hydrogen involved in the add move
        '''
        possibleBonds = list(combinations_with_replacement(sorted(self._possibleAtoms), 2))
        existenceFeatures = np.zeros((len(possibleBonds)*2))
        chargeMultFeatures = np.zeros((len(possibleBonds)*2))
        for coordinate in self._drivingCoordinates:
            index = possibleBonds.index(tuple(sorted(coordinate._Atoms)))
            if coordinate._Type == DriveCoordType.ADD:
                if existenceFeatures[index] == 1:
                    chargeMultFeatures[index] = max(chargeMultFeatures[index], coordinate.chargeProduct())
                else:
                    chargeMultFeatures[index] = coordinate.chargeProduct()
                    existenceFeatures[index] = 1
            elif coordinate._Type == DriveCoordType.BREAK:
                index += len(possibleBonds) # break move indices start where add move indices end
                if existenceFeatures[index] == 1:
                    chargeMultFeatures[index] = min(chargeMultFeatures[index], coordinate.chargeProduct())
                else:
                    chargeMultFeatures[index] = coordinate.chargeProduct()
                    existenceFeatures[index] = 1
            else: raise Exception('Invalid coordinate type!')
        return np.concatenate((existenceFeatures, chargeMultFeatures))

    def build_atom_rep_feature_vec(self):
        '''
        Builds feature vector representing this reaction using representations of the atoms
        involved in the GSM driving coordinates
        '''
        featureVec = []
        for type in DriveCoordType:
            coordReps = np.array([coord.build_atom_rep_feature_vec() for coord in self.movesOfType(type)])
            if coordReps.size:
                featureVec += list(np.max(coordReps, axis=0))
                featureVec += list(np.min(coordReps, axis=0))
                featureVec += list(np.mean(coordReps, axis=0))
                featureVec.append(coordReps.shape[0])
            else:
                featureVec += [0] * (DrivingCoordinate.atom_rep_feature_vec_size() * 3 + 1)  
        return np.array(featureVec)
    