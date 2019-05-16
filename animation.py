import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

path = "ADNI/941_S_1311/MPR__GradWarp__B1_Correction__N3__Scaled/2007-03-02_15_49_04.0/S27408/ADNI_941_S_1311_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080313130949784_S27408_I97327.nii"
nimg = nib.load(path)
nimg_data = nimg.get_fdata()

# Setting up plots
fig, ax = plt.subplots()

im = plt.imshow(nimg_data[0,:,:])
ax.set_title("Frame = 0")

def animate(i):
	ax.set_title("Frame = " + str(i))
	im = plt.imshow(nimg_data[i,:,:])
	return [im]

ani = animation.FuncAnimation(fig, animate, frames=192, interval=1, blit=True)

ani.save("test2.mp4",fps=5)

# plt.show()