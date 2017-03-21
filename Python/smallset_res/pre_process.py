# Eric Walker Paul Zimmerman 3/16/2017
import sys                  
#from sets import Set
import csv
import os.path
sys.path.append('/export/zimmerman/ericwalk/reactivity-machine-learning/Python')
import NBO_converter_gamma as converter

def pre_process():
  dir_path = '/export/zimmerman/paulzim/zstruct2/test/qm_learning/smallset_res'

  # Go through xydata
  xydata = open(dir_path + '/xydata')

  # Extract string id
  lines = xydata.readlines() #Grab the lines of the file as a sequence of strings assigned to the name 'lines'.
  matrix = [] # We will append to this.
  for line in lines: # This will iterate over the lines with the individual line assigned to 'line' at each iteration.
    if 'id:' in line:
      fields = line.strip().split()[2:] # Do not take the first two fields.             
      # Save the string id and also save the activation barrier and the reaction energy
      if os.path.exists(dir_path + '/scratch/startdft' + str(fields[1].zfill(4)) + '.out'):
        matrix.append(fields)
  xydata.close() # We have what we need from this file.

  matrix_NBO = []#This will contain NBO data.

  # For id's- gather raw NBO's
  react_pairs = set() # This will be the bond indice pairs
  prod_pairs = set()
  for m in range(0,len(matrix)):
    reactant = open(dir_path + '/scratch/startdft' + str(matrix[m][1].zfill(4)) + '.out')
    product = open(dir_path + '/scratch/proddft' + str(matrix[m][1].zfill(4)) + '.out')
  # Vector for each reactant/product pair will be [string_name, E_A, delta_E, delta_NBO].  A matrix will be created.
    start_NBO, start_pair = converter.extract_NBOs_from_qchem_output(reactant)
    prod_NBO, prod_pair = converter.extract_NBOs_from_qchem_output(product)
#    matrix_NBO.append(matrix[m][:] + start_NBO + prod_NBO)    
    react_pairs.add(tuple(start_pair)) # There was a concern that the order of the tuple would matter, that 7,3 would be considered unequal to 3,7.  However, NBO starts with the smallest index atom as the first of the pair and lists in order all the atoms in the second place with with there is the bond.  Therefore, the first index is always less than the second index.  The exception is lone pairs (LP) for which the second index is zero.
    prod_pairs.add(tuple(prod_pair))
    NBO_data = []
    for i,pairs in enumerate(start_pair): # Here is the key loop to check does this pair exist in the product set of pairs.  If so, add it to NBO_data.
      if pairs not in prod_pair:
        NBO_data = NBO_data + start_NBO[i][:]

    for j,pairs_beta in enumerate(prod_pair): # Now do the same for any product pairs which are not in the reactant set.
      if pairs_beta not in start_pair:
        NBO_data = NBO_data + prod_NBO[j][:]

    matrix_NBO.append(matrix[m][:] + NBO_data)

  tensor_file = open('tensor_data', 'w+')
  tfw = csv.writer(tensor_file)
  tfw.writerows(matrix_NBO)
#  print start_NBO      
  print matrix_NBO
#  for m in range(0,len(matrix_NBO)):
#    print len(matrix_NBO[m])
#  print matrix_NBO[2][:]
#  print react_pairs
#  print prod_pairs
  reactant.close()
  product.close()
 
if __name__ == '__main__':
  pre_process()
