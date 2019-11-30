#
#

from mat4py import loadmat
import os

for f in os.listdir():
	if f[-2:] != 'py':
		data = loadmat(f)
		with open(f[:-8]+'.pts', 'w') as ptsf:
			ar = data['pts_3d']
			ptsf.write("version: 1\n")
			ptsf.write("n_points:  68\n")
			ptsf.write("{\n")
			for x, y in ar:
				ptsf.write("{} {}\n".format(x,y))
			ptsf.write("{\n")
