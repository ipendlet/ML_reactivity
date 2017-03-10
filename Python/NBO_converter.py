#import openbabel as ob

class Atom:
    def __init__(self, atomicNum, naturalCharge):
        self.atomicNum = atomicNum
        self.naturalCharge = naturalCharge

class NaturalBondOrbital:
    def __init__(self, nboLines, atomsList):
        '''parse the lines of an NBO output file specific to an individual NBO and extract the
        information we want about the atoms in the NBO
        '''
        self.atomicNum2 = 0
        self.atom2Charge = 0
        self.orbitalRatioAtom1 = 0
        
        firstLine = nboLines[0].split()
        
        if firstLine[2].startswith('BD*'):
            # if the type is 3 characters long there isn't a space after it
            firstLine[2:3] = ['BD*','(']
        self.type = firstLine[2]
        
        # extract the atom indices from the first line of the NBO output
        if self.type != 'LP':
            atom1Index = int(firstLine[6][:-1]) - 1
            atom2Index = int(firstLine[8]) - 1
        elif self.type == 'LP':
            atom1Index = int(firstLine[6]) - 1
        
        self.atomicNum1 = atomsList[atom1Index].atomicNum
        self.atom1Charge = atomsList[atom1Index].naturalCharge
        
        # look for the p/s ratios contained in the NBO output
        self.hybrids = []
        for line in nboLines:
            if 's(' in line:
                splitLine = line.split('%)p ')
                if len(splitLine) > 1:
                    self.hybrids.append(float(splitLine[1][:3]))
                else:
                    self.hybrids.append(0.0)
        # get the length of the list up to 2 padding with zeros
        self.hybrids += [0.0] * (2 - len(self.hybrids))
           
        if self.type != 'LP':
            self.atomicNum2 = atomsList[atom2Index].atomicNum
            self.atom2Charge = atomsList[atom2Index].naturalCharge
            self.orbitalRatioAtom1 = float(nboLines[1].split()[1][:-2]) / 100.0
        
    def get_feature_vector(self):
           return [self.type, self.atomicNum1, self.atom1Charge, self.atomicNum2, self.atom2Charge,
                   self.orbitalRatioAtom1] + self.hybrids

def create_atom_list(atomLines):
    '''parse the lines of an NBO output file containing the identities and charges of each atom
    '''
    element_table=["X","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No"];
    atomsList = []
    for line in atomLines:
        elementSymbol = line.split()[0]
        #atomicNum = ob.OBElementTable().GetAtomicNum(elementSymbol)
        atomicNum = element_table.index(elementSymbol)
        naturalCharge = float(line.split()[2])
        atomsList.append(Atom(atomicNum, naturalCharge))
    return atomsList

def extract_NBOs_from_qchem_output(file):
    '''find the NBO information in a QChem output file and parse it into a single feature vector
    '''
    lines = file.readlines()
    
    # find the lines where the natural charge information for all of the atoms is printed
    for i, line in enumerate(lines):
        if 'Summary of Natural Population Analysis:' in line:
            startIndex = i + 6
            break
    for i, line in enumerate(lines[startIndex:], start=startIndex):
        if '===' in line:
            endIndex = i
            break
    atomsList = create_atom_list(lines[startIndex:endIndex])
    
    # find the lines where the NBO data is printed
    for i, line in enumerate(lines):
        if '(Occupancy)   Bond orbital/ Coefficients/ Hybrids' in line:
            startIndex = i + 2
            break
    for i, line in enumerate(lines[startIndex:], start=startIndex):
        if 'SVNBO: NDIM' in line:
            endIndex = i + 1
            break
    
    # split the NBO data lines into individual NBOs and create NBO objects for each of them
    featureVector = []        
    allNBOLines = lines[startIndex:endIndex]
    lastIndex = 0
    for i, line in enumerate(allNBOLines[1:], start=1):
        # the line is a the start of an NBO if the first 5 characters are not all whitespace
        if line[:5].strip() or i + 1 == len(allNBOLines):
            # use the NBO if it is of the type BD(*) or LP
            if any(type in allNBOLines[lastIndex] for type in ['BD', 'LP']):
                featureVector += NaturalBondOrbital(allNBOLines[lastIndex:i],atomsList).get_feature_vector()
            lastIndex = i
    print(featureVector)

if __name__ == '__main__':
#     with open('/export/zimmerman/ericwalk/QChem/nbo_example.out') as file:
    with open('/export/zimmerman/ericwalk/QChem/nbo_example.out') as file:
        extract_NBOs_from_qchem_output(file)

