import numpy as np
from os import listdir
from os.path import isfile, join
from os import getcwd

mypath = './Autoencoder/results/IR_score/'
#mypath = './VariationalAutoencoder-Modified/results/IR_score/'
print 'loading all .npy and .npz files from ',mypath
files = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
files.sort()


for f in files:  
    if f[-4:] in ('.npy','.npz'):
        name = f[:-4]+'_'+f[-3:]
        print 'loading', f
        var = np.load(mypath+f)

        print var['AUC_score']

        print '\n'*3