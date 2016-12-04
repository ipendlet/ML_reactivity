from subprocess import run
import numpy as np

run(['obspectrophore','-i','react1.xyz','-i','react2.xyz'])
print()

u,s,v = np.linalg.svd([[0.690715, 0.685874],[0.503314,0.792463]])

print(u,'\n',s,'\n',v)

print('DONE!')