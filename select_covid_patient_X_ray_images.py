'''
This code finds all images of patients of a specified VIRUS and X-Ray view and stores selected image to an OUTPUT directory
+ It uses metadata.csv for searching and retrieving images name
+ Using ./images folder it selects the retrieved images and copies them in output folder
Code can be modified for any combination of selection of images
'''

import numpy as np
import pandas as pd
import shutil
import os
from COVIDdata import COVIDdataset
import torch
import torchvision
import torchxrayvision as xrv
from tqdm import tqdm
import sys

import matplotlib.pyplot as plt


# Selecting all combination of 'COVID-19' patients with 'PA' X-Ray view
class DataLoader:

	def __init__(self,x_ray_view):
		# virus =  # Virus to look for
		# x_ray_view = "PA" # View of X-Ray
		self.x_ray_view = x_ray_view
		self.metadata = "../metadata.csv"  # Meta info
		self.imageDir = "../images"  # Directory of images
		self.outputDir = '../output'  # Output directory to store selected images
		self.covidData = COVIDdataset()

		#metadata_csv = pd.read_csv(metadata)
		# loop over the rows of the COVID-19 data frame
		# for (i, row) in metadata_csv.iterrows():
		#	if row["finding"] != virus or row["view"] != x_ray_view:
		#		continue
		#
		#	filename = row["filename"].split(os.path.sep)[-1]
		#	filePath = os.path.sep.join([imageDir, filename])
		#	shutil.copy2(filePath, outputDir)


	def loadDataSet(self):
		d_covid19 = xrv.datasets.COVID19_Dataset(views=self.x_ray_view,
												 imgpath=self.imageDir,
												 csvpath=self.metadata)

		for i in tqdm(range(len(d_covid19))):

		#for i in tqdm(range(30)):
			try:
				# start from the least recent
				sample = d_covid19[i]
				self.covidData.add(sample)

			except KeyboardInterrupt:
				break;
			#except:
			#	print("Error with {}".format(i) + d_covid19.csv.iloc[i].filename)
			#	print(sys.exc_info()[1])

		self.covidData.normalize()
		self.covidData.vectorize()
		self.covidData.generateMatrices()

		return self.covidData




if __name__ == "__main__":
	dataLoader = DataLoader(['PA'])
	print(40 * "=")
	print("Loading dataset from file")
	print(40 * "=")
	covidset = dataLoader.loadDataSet()
	print()
	print(40 * "=")
	print("Completed loading dataset from file")
	f = covidset.X[0, :]
	f = f.reshape((covidset.minsize, covidset.minsize))
	plt.imshow(f, cmap=plt.cm.gray)
	plt.show()