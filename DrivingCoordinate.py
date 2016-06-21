

class DrivingCoordinate:
    def __init__(self, Type=None, Atoms=None, NBO=None, Hybrid=None):
        '''
        Type = Add, Brk, None
        Atoms = pair of atoms being added or broken
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
        self._NBO, self._Atoms, self._Hybrid = zip(*sorted(zip(self._NBO,self._Atoms,self._Hybrid), key=lambda x:x[0]))
        pass