'''
Created on Jun 13, 2017

@author: joshkamm
'''
from pathlib import Path
from Reaction import Reaction
from DrivingCoordinate import DriveCoordType, DrivingCoordinate
from builtins import range
import pybel as pb
import numpy as np

class ReactivityDataLoader():
    '''
    Loads and generates features from chemical reactivity data. Does not currently utilize NBO
    or mopac hybridization data.
    '''

    def __init__(self):
        pass
    
    def load_mopac_learning(self):
        'Load a small molecule reactivity data set that Paul produced with GSM / Zstruct / mopac'
        self.reactions = []
        
        # open the appropriate file running on athena or locally
        self.dataFilePath = Path('~paulzim').expanduser() / 'zstruct2' / 'test' / 'mopac_learning' / 'xydata'
        if not self.dataFilePath.exists():
            self.dataFilePath = Path.home() / 'Desktop' / 'xydata'
            
        with self.dataFilePath.open() as dataFile:
            dataFileLines = dataFile.readlines()
            i = 0
            while i + 5 <= len(dataFileLines):
                self.read_reaction(dataFileLines[i:i+5])
                i += 6
        
        reactionData = np.array([reaction.build_atom_rep_feature_vec() for reaction in self.reactions])
        targets = np.asarray([reaction._activationEnergy for reaction in self.reactions])
        return reactionData, targets
        
    def read_reaction(self, reactionLines):
        "read the lines of Paul's file representing an individual reaction"
        # read reaction header line
        headerLine = reactionLines[0].split()
        pntNum = int(headerLine[1])
        idNum = int(headerLine[3])
        activationEnergy = float(headerLine[5])
        EofRxn = float(headerLine[7])
        self.reactions.append(Reaction(idNum,activationEnergy,EofRxn))
        
        stringFilePath = (self.dataFilePath.parent / 'savestrings'
                          / ('stringfile.xyz' + str(idNum).zfill(4)))
        reactants = next(pb.readfile('xyz', str(stringFilePath)))
        
        # read add and break lines
        for i, driveCoordType in enumerate(DriveCoordType):
            driveCoordLine = reactionLines[i+1].split()
            for j in range(2,len(driveCoordLine),2):
                driveCoordIndices = driveCoordLine[j].strip('()').split('-')
                driveCoordAtoms = [reactants.atoms[int(k)-1] for k in driveCoordIndices]
                self.reactions[-1].addDrivingCoordinate(DrivingCoordinate(Type=driveCoordType,
                                                                          Atoms=driveCoordAtoms))
