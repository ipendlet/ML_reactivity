import numpy as np
from itertools import combinations, combinations_with_replacement

class Reaction():
    '''
    A class representing a chemical reaction
    '''

    def __init__(self, id=None, activationEnergy=None, heatOfRxn=None):
        '''
        Constructor
        '''
        self._possibleAtoms = 'BCHNO'
        self._id = id
        self._activationEnergy = activationEnergy
        self._heatOfRxn = heatOfRxn
        self._drivingCoordinates = []
        # TODO: add miscellaneous values

    def addDrivingCoordinate(self, drivingCoordinate):
        self._drivingCoordinates.append(drivingCoordinate)

    def sortDrivingCoordinates(self):
        addMoves = self.movesOfType('add')
        breakMoves = self.movesOfType('break')
        for addMove in addMoves:
            addMove.sortByCharge()
        for breakMove in breakMoves:
            breakMove.sortByCharge()
        addMoves = sorted(addMoves, key=lambda x : x._NBO[0])
        breakMoves = sorted(breakMoves, key=lambda x : x._NBO[0])
        
#         addCharges = [i._NBO for i in sorted(addMoves, lambda x : x._NBO[0] + x._NBO[1])]
#         breakCharges = [i._NBO for i in sorted(breakMoves, lambda x : x._NBO[0] + x._NBO[1])]
        #Josh - RESUME HERE
        return addMoves, breakMoves

    def movesOfType(self, type):
        '''
        return all driving coordinates of argument type
        '''
        return list(filter(lambda x : x._Type == type,self._drivingCoordinates))
    
    def buildFeatureVector(self,includeChargeMult=False,includeAddBreak=False,isSorted=True):
        # up to 40+10=50 features because of hard limit on add and break moves of 5 and 4 features per move
        if isSorted:
            addMoves, breakMoves = self.sortDrivingCoordinates()
        else:
            addMoves = self.movesOfType('add')
            breakMoves = self.movesOfType('break')
        
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
        possibleBonds = list(combinations_with_replacement(sorted(self._possibleAtoms), 2))
        existenceFeatures = np.zeros((len(possibleBonds)*2))
        chargeMultFeatures = np.zeros((len(possibleBonds)*2))
        for coordinate in self._drivingCoordinates:
            index = possibleBonds.index(tuple(sorted(coordinate._Atoms)))
            if coordinate._Type == 'add':
                if existenceFeatures[index] == 1:
                    chargeMultFeatures[index] = max(chargeMultFeatures[index], coordinate.chargeProduct())
                else:
                    chargeMultFeatures[index] = coordinate.chargeProduct()
                    existenceFeatures[index] = 1
            elif coordinate._Type == 'break':
                index += len(possibleBonds) # break move indices start where add move indices end
                if existenceFeatures[index] == 1:
                    chargeMultFeatures[index] = min(chargeMultFeatures[index], coordinate.chargeProduct())
                else:
                    chargeMultFeatures[index] = coordinate.chargeProduct()
                    existenceFeatures[index] = 1
            else: raise Exception('Invalid coordinate type!')
        return np.concatenate((existenceFeatures, chargeMultFeatures))
