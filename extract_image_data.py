import nibabel as nib
from skimage.transform import resize
import pickle
import numpy as np

with open('pickles/mri_filenames.pickle', 'rb') as f:
	mri_files = pickle.load(f)

input_size = (240, 256, 160)
x = []

for i,file in enumerate(mri_files[150:]):
    print(i)
    nimg = nib.load(file).get_fdata()
    nimg = resize(nimg, input_size)
    x.append(nimg)

x = np.array(x)
x = x.reshape((x.shape[0],1,240, 256, 160))

block_size = 50

for i in range(int(x.shape[0]/block_size)+1):
    print(i)
	with open('pickles/x%d.pickle' % i, 'wb') as f:
	    pickle.dump(x, f)
