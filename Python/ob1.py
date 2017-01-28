import openbabel as ob
import pybel
import glob
import json

obconv = ob.OBConversion()
earray = [ 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
           'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar' ]

###################################################
def reorderatoms( atlist ):
   natt = 4
   numats = len(atlist)//natt
   #print 'testing reorder atoms. natoms: ',numats
   for n in range(0,numats-2):
      swapit = 0
      if atlist[natt+natt*n] < atlist[2*natt+natt*n]:
         swapit += 1
      if atlist[natt+natt*n] == atlist[2*natt+natt*n]:
         if atlist[natt+natt*n+1] < atlist[2*natt+natt*n+1]:
            swapit += 1
      if atlist[natt+natt*n] == atlist[2*natt+natt*n] and atlist[natt+natt*n+1] == atlist[2*natt+natt*n+1]:
         if atlist[natt+natt*n+3] < atlist[2*natt+natt*n+3]:
            swapit += 1
      if swapit:
         for i in range(0,natt):
            tmp = atlist[natt+natt*n+i]
            atlist[natt+natt*n+i] = atlist[2*natt+natt*n+i]
            atlist[2*natt+natt*n+i] = tmp
         return reorderatoms( atlist )
###################################################
def addallatoms ( i1, mol1, allats ):
   print('now moving through pybel atoms, via converting to OBAtom')
   print('')
   for atom in mol1:
      i1 += 1
      mol_ob_atoms = ob.OBMol()

      print('  working on atom ',i1)
      atpr = []
      elem = earray[atom.atomicnum-1]
      atmin = [ elem, atom.coords[0], atom.coords[1], atom.coords[2] ]
      print('atomic coords: ',atmin)

      obatom = atom.OBAtom
      mol_ob_atoms.AddAtom(obatom)
      print('this atom is: ',atom.atomicnum,' ',atom.type,' index: ',atom.idx)

      atpr.append(atom.atomicnum)
      atpr.append(atom.valence)
      atpr.append(atom.hyb)
      atpr.append(0)
      for nat in ob.OBAtomAtomIter(obatom):
         print('  atomic#: ',nat.GetAtomicNum(),' hybridization: ',nat.GetHyb())
         atpr.append(nat.GetAtomicNum())
         atpr.append(nat.GetValence())
         atpr.append(nat.GetHyb())
         bond = mol_ob_atoms.GetBond(obatom,nat)
         bondorder = bond.GetBondOrder()
         atpr.append(bondorder)
         elem2 = earray[nat.GetAtomicNum()-1]
         mol_ob_atoms.AddAtom(nat)
      reorderatoms(atpr)
      if (atom.valence<2):
         atpr.extend([0,0,0,0])
      if (atom.valence<3):
         atpr.extend([0,0,0,0])
      if (atom.valence<4):
         atpr.extend([0,0,0,0])
      #atpr.append(i)
      print('  fragment atoms: ',mol_ob_atoms.NumAtoms())
      obconv.WriteFile(mol_ob_atoms,"atoms_all/w"+str(i1).rjust(2,'0')+".xyz")
      print('  atpr: ',atpr)
#      print 'atmin: ',string.join(atmin," ")
      print('')

      allats.append(atpr)
   return i1
##########################################################

if __name__ == '__main__':
    
    filelist = glob.glob("react*.xyz")
    
    print('adding all react.xyz files')
    molpr = []
    i = 0
    for file in filelist:
       print(' ',file)
       mol1 = next(pybel.readfile("xyz", file))
       i = addallatoms(i,mol1,molpr)
       
    
    #mol_ob = ob.OBMol()
    # png write doesn't work. can use babel -O
    #obconv.SetOutFormat("png")
    #obconv.WriteFile(mol_ob,'bz1.xyz')
    
    
    
    print('')
    #print ' molpr:',molpr
    #print ' molpr[1,5]: ',molpr[1][5]   
    
    
    #create unique list of atoms
    listpr = set()
    for thisat in molpr:
    #   print 'thisat: ',thisat
       listpr.add(repr(thisat))
    
    #turn set back into int list
    b2list = []
    for thisat in sorted(listpr):
       print('sortat: ',thisat)
       l1 = list(thisat)
       nlist = []
       for item in l1:
          if item.isdigit():
             nlist.append(item)
    #   print 'l1: ',l1
    #   print 'nlist: ',nlist
       b2list.append(nlist)
    #print ' listpr:',sorted(listpr)
    print(' number of unique atoms: ',len(listpr))
    
    f = open('ATOMS', 'w')
    json.dump(b2list,f)
    f.close()
