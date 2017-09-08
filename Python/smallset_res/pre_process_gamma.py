# Eric Walker Paul Zimmerman 3/16/2017
# Preprocess QChem reactant/product pair NBO output to feature vectors where each feature vector represents one reactant/product pair.  The file 'tensor_data' is written one feature vector per line.
import sys                  
#from sets import Set
import csv
import os.path
sys.path.append('/export/zimmerman/ericwalk/reactivity-machine-learning/Python')
import NBO_converter_delta as converter

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
        NBO_data.append(start_NBO[i][:4])

    occ_change_react = [] # This is created because we will sort the NBO vectors by their type.
    for i,NBO_alpha in enumerate(start_NBO): # if the same NBO exists in reactant and product but the occupancy changed
      for j,NBO_beta in enumerate(prod_NBO):
        if NBO_alpha[1:3] == NBO_beta[1:3] and len(NBO_alpha[0]) == len(NBO_beta[0]) and abs(float(NBO_alpha[3]) - float(NBO_beta[3])) > 1:
          occ_change_react.append(start_NBO[i][:4])
    occ_change_react.sort()
    for i in range(0,len(occ_change_react)):
      NBO_data.append(occ_change_react[i][:4])

    start_similar = []  # We now start checking if there is a mismatch in the number of similar type NBOs in reactant and product, i.e. a similar NBO could be added or removed, changing the bond order of between the two same atoms or number of lone pairs on an atom.
    for a in range(0,len(start_NBO)):
      start_similar.append(start_NBO[a][:3])

    prod_similar = []  # analogy of start_similar for product
    for a in range(0,len(prod_NBO)):
      prod_similar.append(prod_NBO[a][:3])

    for b in range(0,len(start_similar)):
      count_alpha = start_similar.count(start_similar[b])
      count_beta = prod_similar.count(start_similar[b])
      if count_alpha - count_beta != 0:
        NBO_data.append(start_NBO[b][:4])

    NBO_data_no_duplicates = []
    for i in NBO_data:
      if i not in NBO_data_no_duplicates:
        NBO_data_no_duplicates.append(i)
    NBO_data_no_duplicates_row = []
    for i in NBO_data_no_duplicates:
      NBO_data_no_duplicates_row = NBO_data_no_duplicates_row + i

    #NBO_data.append('separator')
    NBO_data_no_duplicates_row.append('separator')

    NBO_data_prod = []
    occ_change_prod = [] # This is created because we will sort the NBO vectors by their type.
    for j,pairs_beta in enumerate(prod_pair): # Now do the same for any product pairs which are not in the reactant set.
      if pairs_beta not in start_pair:
        NBO_data_prod.append(prod_NBO[j][:4])

    for k,NBO_gamma in enumerate(prod_NBO): # if the same NBO exists in product and reactant but the occupancy changed 
      for l,NBO_delta in enumerate(start_NBO):
        if NBO_gamma[:3] == NBO_delta[:3] and len(NBO_gamma[0]) == len(NBO_delta[0]) and abs(float(NBO_gamma[3]) - float(NBO_delta[3])) > 1:
          occ_change_prod.append(prod_NBO[k][:4])
    occ_change_prod.sort()
    for i in range(0,len(occ_change_prod)):
      NBO_data_prod.append(occ_change_prod[i][:4])

    for b in range(0,len(prod_similar)):
      count_alpha = prod_similar.count(prod_similar[b])
      count_beta = start_similar.count(prod_similar[b])
      if count_alpha - count_beta != 0:
        NBO_data_prod.append(prod_NBO[b][:4])

    NBO_data_prod_no_duplicates = []
    for i in NBO_data_prod:
      if i not in NBO_data_prod_no_duplicates:
        NBO_data_prod_no_duplicates.append(i)
    NBO_data_prod_no_duplicates_row = []
    for i in NBO_data_prod_no_duplicates:
      NBO_data_prod_no_duplicates_row = NBO_data_prod_no_duplicates_row + i

    matrix_NBO.append(matrix[m][:] + NBO_data_no_duplicates_row + NBO_data_prod_no_duplicates_row)

#    print len(NBO_data_prod_no_duplicates) == len(NBO_data_no_duplicates)

  tensor_file = open('tensor_data', 'w+')
  tfw = csv.writer(tensor_file)
  tfw.writerows(matrix_NBO)
  
#  print i
#  print NBO_data
#  print NBO_data_no_duplicates
#  print start_NBO      
#  print matrix_NBO
#  for m in range(0,len(matrix_NBO)):
#    print matrix_NBO[m][9]
#    print len(matrix_NBO[m])
#  print matrix_NBO[4][:]
#  print NBO_gamma[:3]
#  print NBO_delta[:3]
#  print len(NBO_gamma[0])
#  print len(NBO_delta[0])
#  print react_pairs
#  print prod_pairs
  reactant.close()
  product.close()
 
if __name__ == '__main__':
  pre_process()
