import sys

""" This module, specifically the read_write_natural_charge method, reads the natural charge from QChem output with the NBO calculation turned on.  The read_write_natural_charge method also reads the atom positions from the QChem output.  It writes the atom positions to an .xyz format file except with the atoms and their natural charge placed at the end of the modified .xyz file """

def read_write_natural_charge(qchem_output, xyz_file):
  line_was = 0
  start_reading = False
  opened_xyz_file = open(xyz_file, 'w+')
  opened_xyz_file.write('\n\n') # The first line of xyz_file will be filled with the number of atoms written as a string.
  opened_xyz_file.close()
  with open('qchem_output') as content:
    for i, line in enumerate(content):
      if '$molecule' in line:
        line_was = i
        start_reading = True
      if i >= line_was + 2 & start_reading == True & '$end' not in line:
        xyz_file.write(line + '\n')
      if '$end' in line:
        start_reading = False
        xyz_file.write('\n\n')
        number_of_atoms = i - line_was
      if 'Summary of Natural Population Analysis:' in line:
        line_was = i
        start_reading = True
      if i >= line_was + 4 & start_reading == True & '=' not in line:
        fields = line.strip().split()
        xyz_file.write(fields[0:2] + '\n')
      if '=' in line:
        start_reading = False
  put_atom_number = xyz_file.open()
  put_atom_number.write(str(number_of_atoms))


##########################################################

if __name__ == '__main__':
  read_write_natural_charge('nbo_example.out','methanol.xyz')
