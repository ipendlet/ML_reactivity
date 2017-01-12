# %% Imports
import tensorflow as tf
import numpy as np
import math
import json
import openbabel as ob
import pybel
from sklearn.neighbors import NearestNeighbors
import os.path
from subprocess import run

nsteps = 25000

obconv = ob.OBConversion()
obconv.SetInAndOutFormats("smi", "smi")
#obconv2 = ob.OBConversion()
#obconv2.SetInAndOutFormats("smi", "svg")
elist = [ 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
           'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar' ]
elist_s = [ '[H]', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
           'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar' ]

######################################################
#def getdist(z1,z2):
#    return tf.reduce_sum(tf.abs(tf.add(z1, tf.neg(z2))), reduction_indices=1)


######################################################
def autoencoder(dimensions=[15, 100, 100, 15], bias1=[]):
    """Build a deep autoencoder w/ tied weights.
    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # %% bias on the latent rep
    bias = bias1

    # %% input to the network
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    current_input = x

    # %% Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    # %% latent representation
    z = current_input
    encoder.reverse()

    # %% Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    n_input = int(current_input.get_shape()[1])

#    Ws = tf.Variable(
#         tf.random_uniform([n_input, n_output],
#                            -1.0 / math.sqrt(n_input),
#                            1.0 / math.sqrt(n_input)))

    # %% now have the reconstruction through the network
    y = current_input

    # %% cost function measures pixel-wise difference
    dcost = 0.
    for i in range(0,len(bias)):
      b12 = bias[i]
      b1 = b12[0]
      b2 = b12[1]
      print ("in dcost. b's: ",b1," ",b2)

      distb = tf.reduce_sum(tf.square(z[b1]-z[b2]))
      normb = tf.multiply(tf.reduce_sum(tf.multiply(z[b1],z[b1])),tf.reduce_sum(tf.multiply(z[b2],z[b2]))) 
      dcost += 100.*distb/normb
    cost = tf.reduce_sum(tf.square(y - x)) + dcost
    return {'x': x, 'z': z, 'y': y, 'cost': cost}
######################################################

def tozero(val):
   if (val<0):
      val = 0.
   return val


######################################################
def test_ob(knn):
   print (' ')
   print (' testing autoencoder!')

   with open('ATOMS', 'r') as f:
      xdata = json.load(f)
   print('read-in data:')
   for at in xdata:
      print (' atom: ',at)
   xsize = len(xdata[0])
   print('dim: ',len(xdata),',',xsize)
   for i in range(0,len(xdata)):
      for j in range(0,xsize):
         xdata[i][j] = int(xdata[i][j]) / 8. - 0.5
   for at in xdata:
      print (' atom: ',at)

   bdata = []
   if os.path.isfile('BIAS'):
      with open('BIAS') as f:
         for line in f:
            pline = line.rstrip('\n')
            nline = map(int, pline.split(' '))
            for i in range(0,len(nline)):
               nline[i] -= 1
            bdata.append(nline)
   print ('bdata (ref to 0):',bdata)

   ae = autoencoder(dimensions=[xsize, 6],bias1=bdata)
   learning_rate = 0.005
   optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

   # We create a session to use the graph
   sess = tf.Session()
   sess.run(tf.initialize_all_variables())
#    sess.run(tf.global_variables_initializer())

   n_epochs = nsteps
   for epoch_i in range(n_epochs):
      train = xdata
      sess.run(optimizer, feed_dict={ae['x']: train})
      if (epoch_i%100==0):
         cost1 = sess.run(ae['cost'], feed_dict={ae['x']: train})
         print(epoch_i, cost1)
      if (cost1<0.001):
         break

   recon = sess.run(ae['y'], feed_dict={ae['x']: xdata})
   latent = sess.run(ae['z'], feed_dict={ae['x']: xdata})

   print("\n errors: ")
   err = []
   for i in range(0,len(recon)):
      err.append(0.)
      for j in range(0,len(recon[0])):
         err[i] += pow(recon[i][j] - xdata[i][j],2)
      print('error: %0.5f' % err[i])

   print("\n reconstruction: ")
   for i in range(0,len(recon)):
      for j in range(0,len(recon[0])):
         recon[i][j] = (recon[i][j]+0.5) * 8.
   for item in recon:
      r1 = ["%0.1f" % tozero(i) for i in item]
      print(r1)

   print("\n latent representation: ")
   for item in latent:
      l1 = ["%0.2f" % i for i in item]
      print(l1)

   nbrs = NearestNeighbors(n_neighbors=knn).fit(latent)
   distances, indices = nbrs.kneighbors()
   print (' indices: ')
   print (indices)
   print (' distances: ')
   print (distances)

   return indices, distances


def s2i(ws):
   return int(ws)-1

def s2e(ws):
   return elist_s[s2i(ws)]

def add_bonded(str1,wa,wb,nv):
   if int(wa)==0:
      return str1

   bc = int(wb)-1
   str1 += "("
   if (int(wb)==3):
      str1 += "#"+s2e(wa)
   if (int(wb)==2):
      str1 += "="+s2e(wa)
   else:
      str1 += s2e(wa)
   if (int(wa)==6):
      if (int(nv)==4):
         for i in range(0,3-bc):
            str1 += "([*:"+str(i+1)+"])"
      if (int(nv)==3):
         if (bc==0):
            str1 += "(=[*:1])"
         else:
            str1 += "([*:1])"
         str1 += "([*:2])"
      if (int(nv)==2):
         if (bc==0):
            str1 += "(#[*:1])"
         else:
            str1 += "(=[*:1])"
   elif (int(wa)==7):
      if (int(nv)==4):
         for i in range(0,3-bc):
            str1 += "([*:"+str(i+1)+"])"
      if (int(nv)==3):
         str1 += "([*:1])([*:2])"
      if (int(nv)==2):
         if (bc==0):
            str1 += "(=[*:1])"
         else:
            str1 += "([*:1])"
   elif (int(wa)>1):
      for i in range(0,int(nv)-1-bc):
         str1 += "([*:"+str(i+1)+"])"
   str1 += ")"

   return str1


##############################################
def test_smiles():
   print('\n testing smiles')
   with open('ATOMS', 'r') as f:
      xdata = json.load(f)
   print('read-in data:')
   for at in xdata:
      print (' atom: ',at)
   print('dim: ',len(xdata),',',len(xdata[0]))

#   gen2d = ob.OBOp.FindType("Gen2D")

#   mol = pybel.readstring("smi", "C/C=C(O)\C")
#   gen2d.Do(mol.OBMol)
 
   n = 1
   for at in xdata:
      str1 = ""
      str1 = s2e(at[0])
      str1 = add_bonded(str1,at[4],at[7],at[5])
      str1 = add_bonded(str1,at[8],at[11],at[9])
      str1 = add_bonded(str1,at[12],at[15],at[13])
#      for i in range(0,len(at)):
#         print(" this one:",at[i])
      print(" str1: ",str1)

      mol = ob.OBMol()
      obconv.ReadString(mol, str1)
      mol.AddHydrogens()
#      pmol = pybel.Molecule(mol)
#      output = pybel.Outputfile("xyz", "at"+str(n)+".xyz")
#      output.write(pmol)
#      output.close()
      obconv.WriteFile(mol,"at"+str(n).zfill(2)+".smi")
## openbabel can't read the R's
#      obconv2.WriteFile(mol,"at"+str(n).zfill(2)+".svg")
      n += 1

#########################################################
def do_nn(knn):
   with open('ATOMS', 'r') as f:
      xdata = json.load(f)

   nnat, dists = test_ob(knn)
 
   for n in range(0,len(xdata)):
      at = xdata[n]
      str1 = ""
      str1 = s2e(at[0])
      str1 = add_bonded(str1,at[4],at[7],at[5])
      str1 = add_bonded(str1,at[8],at[11],at[9])
      str1 = add_bonded(str1,at[12],at[15],at[13])
      mol = ob.OBMol()
      obconv.ReadString(mol, str1)
      mol.AddHydrogens()

      fname = "at"+str(n+1).zfill(2)+".00.smi"
      #obconv.WriteFile(mol,fname)
      molpy = pybel.Molecule(mol)
      molpy = pybel.readstring("smi",str1)
      molpy.title = "S"+str(n+1)
      molpy.write("smi",fname,overwrite=True)

      for m in range(0,knn):
         at2 = xdata[nnat[n,m]]
         str2 = ""
         str2 = s2e(at2[0])
         str2 = add_bonded(str2,at2[4],at2[7],at2[5])
         str2 = add_bonded(str2,at2[8],at2[11],at2[9])
         str2 = add_bonded(str2,at2[12],at2[15],at2[13])
         mol2 = ob.OBMol()
         obconv.ReadString(mol2, str2)
         mol2.AddHydrogens()

         fname2 = "at"+str(n+1).zfill(2)+".nn"+str(m+1)+".smi"
         #obconv.WriteFile(mol2,fname2)
         molpy2 = pybel.Molecule(mol2)
         molpy2 = pybel.readstring("smi",str2)
         molpy2.title = "NN"+str(n+1)+"-"+str(nnat[n,m]+1)+": "+str("%0.2f" % dists[n,m])
         molpy2.write("smi",fname2,overwrite=True)
#    run('. ~/.bash_profile; ./do_svg',shell=True)
            

########################
#test_ob(4)
#test_smiles()
do_nn(3)