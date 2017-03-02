import sys #Make sure we have access to core python methods.

""" This module, specifically the read_write_natural_charge method, reads the natural charge from QChem output with the natural bond ordital (NBO) calculation turned on.  The read_write_natural_charge method also reads the atom positions from the QChem output.  It writes the atom positions to an .xyz format file except with the atoms and their natural charge placed at the end of the modified .xyz file.  Note this was used on a single-point example.  Perhaps a geometry optimization has different xyz coordinates versus the starting  """

def read_write_natural_charge(qchem_output, xyz_file): #qchem_output is passed a string which is the name of a QChem output file, .out, with NBO information printed out.  The xyz_file is also passed a string, and will be the name of the file.  The .xyz need not be created already.  
  line_was = 0 # Initiate the variable which will be tested in if conditional statements.  Lines will be passed before this variable is reassigned, so we put it to zero to prevent 'not defined' errors. 
  start_reading = False # This boolean will also be passed to conditional statements before it is switched to True and back and forth and so forth.
  opened_xyz_file = open(xyz_file, 'w+') # create a file object, opened_xyz_file, in order to perform file methods on it.  Here the xyz_file is the string passed in to this read_write_natural_charge method.  'w+' means make the file object writeable and other permissions from the plus sign which may not be necessary.
  opened_xyz_file.write('\n\n') # The first line of xyz_file will be filled with the number of atoms written as a string. In .xyz files the atoms and their positions begin on the third line and the first line is the total number of atoms.  \n in a string creates a new line.
  content = open(qchem_output, 'r+') # Create a file object which we will read.
  content.readlines() # On the file object, make a sequence which is separated by line.  We will be able to loop over the lines of the file this way.
  for i, line in enumerate(content): # enumate will give us the index, i for the sequence/list of lines in the file object called content.  We want the index number in order to certain lines.  This loop faced the challenge that the lines to read apppear a number of lines after a string
    if '$molecule' in line: # This string '$molecule' appears two lines before the first atom.  The atom contains the element, followed by a space, and the x,y,z coordinates.  First there is a space, then the element, then whitespace until the x coordinate, followed by whitespace, followed by the y coordinate, followed by whitespace, followed by the z coordinate.
      line_was = i # Grab the index.  We want to start reading two lines later.
      start_reading = True # Turn on this boolean.  It is not enough to start reading when the line index is greater than line_was becuase we initiated line_was as zero.
    if i >= line_was + 2 & start_reading == True & '$end' not in line: # Read the lines, except if we have reached '$end'.  '$end' appears on the last directly after the last atom.
      xyz_file.write(line + '\n') # 
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
  opened_xyz_file.write(str(number_of_atoms)) # 
  opened_xyz_file.close() # Close up the file object for the sake of style.  Might not be necessary. 

##########################################################

if __name__ == '__main__':
  read_write_natural_charge('nbo_example.out','methanol.xyz')
