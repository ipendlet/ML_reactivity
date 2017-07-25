'''
Created on Jun 28, 2017

@author: joshkamm
'''
from pathlib import Path

from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame
from seaborn.categorical import countplot

class ReactivityDataExplorer():
    '''
    generates statistics and visualizations for the purpose of exploring and better understanding
    reactivity data
    '''

    def __init__(self, reactions):
        self.reactions = reactions
    
    def plot_coord_num_dist_for_element_and_move_type(self, atomicNum):
        '''plot the distribution of coordination number for all of a given element involved in
        driving coordinates grouped by driving coordinate type
        '''

        coordNumsDataFrame = DataFrame()
        for reaction in self.reactions:
            for driveCoordinate in reaction._drivingCoordinates:
                for mlAtom in driveCoordinate._Atoms:
                    if mlAtom._atom.atomicnum == atomicNum:
                        coordNumsDataFrame = coordNumsDataFrame.append(DataFrame(
                                {'move type':[driveCoordinate._Type],
                                 'coordination number':[mlAtom._atom.valence]}),
                                ignore_index=True)

        countplot(x='coordination number', hue='move type', data=coordNumsDataFrame)
        plt.savefig(str(Path.home() / 'Desktop' / 'testPlot.png'))
