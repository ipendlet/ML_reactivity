import pybel as pb

class ChemDraw:
    def __init__(self):
        self.moleculesToPositions = {}
        
    def add_molecule(self, molecule, position):
        self.moleculesToPositions[molecule] = position
        
    def write_to_file(self):
        outFile = pb.Outputfile('cdxml','testCdxml')
        for molecule in self.moleculesToPositions:
            outFile.write(molecule)
    
if __name__ == '__main__':
    ethane = pb.readstring('smi', 'CC')
    propane = pb.readstring('smi', 'CCC')
    butane = pb.readstring('smi', 'CCCC')
    chemDraw = ChemDraw()
    chemDraw.add_molecule(ethane, [1,1])
    chemDraw.add_molecule(propane, [1,2])
    chemDraw.add_molecule(butane, [1,3])
    chemDraw.write_to_file()