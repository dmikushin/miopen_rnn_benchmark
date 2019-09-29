import numpy as np

lines = open("npys").readlines()
for line in lines:
	a = np.load(line.replace("\n",""))
	np.save(line.replace("\n",""),a.transpose())
