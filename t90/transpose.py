import numpy as np

files = [i.replace("\n","") for i in open("npys").readlines()]

for file in files:
	np.save(file,np.load(file).transpose())
