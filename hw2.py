import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from sklearn.manifold import MDS
from operator import itemgetter
from collections import Counter
import csv
import matplotlib.pyplot as plt
import random

## Data source: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/dengue.html
## Ref: http://ithelp.ithome.com.tw/articles/10187314
## Ref: http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_3d.html
## Ref: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
## Ref: http://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html
## Ref: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

##########################################
######### TASK 0 Pre-proccessing #########
##########################################

# Parameters
sample_size = 1000
nGroup = 3

# import csv file
with open('dengue.csv', 'r') as f:
	reader = csv.reader(f)
	inFile = list(reader)
	rowName = inFile[0][:]
	inFile = inFile[1:][:]

# All row names
rowName = rowName[1:]
rowName = np.asarray(rowName).reshape(1,len(rowName))

# All input data
# data pre-proccessing: remove "NA" items by replacing average values
for i in range(len(inFile[0])):
	avg = 0.0
	cnt = 0.0
	for j in range(len(inFile)):
		if inFile[j][i] != 'NA':
			cnt += 1
			avg += float(inFile[j][i])
	avg = avg / cnt
	for j in range(1, len(inFile)):
		if inFile[j][i] == 'NA':
			inFile[j][i] = avg
inputData = [inFile[i][1:] for i in range(1, len(inFile))]

##########################################
############ TASK 1 Sampling  ############
##########################################

# Task (1a) Random-sampling: 2000 -> 1000
randInputData = [inputData[i] for i in random.sample(range(len(inputData)), sample_size)]

# Task (1b) K-means clustering + Adeptive Sampling
kmLabels = KMeans(n_clusters = nGroup).fit(inputData).labels_

# Seperate input data into n groups
kmInputData = np.append(np.asarray(inputData), np.asarray(kmLabels).reshape((len(kmLabels), 1)), 1)
kmInputData = kmInputData.tolist()

# seperate input data into n groups
inputDataGroup = [[] for i in range(nGroup)]
for i in range(len(kmInputData)):
	inputDataGroup[kmLabels[i]].append(kmInputData[i])

# Adeptive Sampling: 2000 -> 1000
kmInputData = []
for i in range(nGroup):
	kmInputData.extend([inputDataGroup[i][j] for j in random.sample(range(len(inputDataGroup[i])), 
		int(len(inputDataGroup[i]) * sample_size / len(inputData)))])

##########################################
############### TASK 2 PCA ###############
##########################################

randpca = PCA().fit(randInputData)
pcaRandComponents = randpca.components_[0:3]
pcaRandEigenvalue = randpca.explained_variance_ratio_

with open('Random_PCA_dengue_intrinsic_dimensionality.csv', 'w', newline = '') as f:
	writer = csv.writer(f)
	writer.writerows([['feature','PC1','PC2','PC3', 'ev']])
	pcaRandComponents = np.append(rowName, pcaRandComponents, 0)
	pcaRandComponents = np.append(pcaRandComponents, pcaRandEigenvalue.reshape(1,len(pcaRandEigenvalue)), 0)
	writer.writerows(np.transpose(pcaRandComponents))

pcaRandOutputData = PCA(n_components=2).fit_transform(randInputData)
pcaRandOutputData = np.append(pcaRandOutputData, randInputData, 1)
with open('Random_PCA_dengue.csv', 'w', newline = '') as f:
	writer = csv.writer(f)
	writer.writerows(np.append([['x','y']], rowName, 1))
	writer.writerows(pcaRandOutputData)


kmInputDataGroup = np.asarray(kmInputData)[:, 13]
kmInputDataGroup = kmInputDataGroup.reshape(len(kmInputDataGroup), 1)

kmInputData = np.delete(kmInputData, 13, 1)
kmpca = PCA().fit(kmInputData)
pcaKmComponents = kmpca.components_[0:3]
pcaKmEigenvalue = kmpca.explained_variance_ratio_
with open('KMean_PCA_dengue_intrinsic_dimensionality.csv', 'w', newline = '') as f:
	writer = csv.writer(f)
	writer.writerows([['feature','PC1','PC2','PC3', 'ev']])
	pcaKmComponents = np.append(rowName, pcaKmComponents, 0)
	pcaKmComponents = np.append(pcaKmComponents, pcaKmEigenvalue.reshape(1,len(pcaKmEigenvalue)), 0)
	writer.writerows(np.transpose(pcaKmComponents))
	
pcaKmOutputData = PCA(n_components=2).fit_transform(kmInputData)
pcaKmOutputData = np.append(pcaKmOutputData, kmInputData, 1)
pcaKmOutputData = np.append(pcaKmOutputData, kmInputDataGroup, 1)

with open('KMean_PCA_dengue.csv', 'w', newline = '') as f:
	writer = csv.writer(f)
	writer.writerows(np.append(np.append([['x','y']], rowName, 1), [['group']], 1))
	writer.writerows(pcaKmOutputData)

##########################################
############### TASK 3 MDS ###############
##########################################

## Task 3(a) euclidean_distances

randInputData = np.asarray(randInputData)
mdsRandEucOutputData = MDS(n_components=2).fit(randInputData).embedding_
mdsRandEucOutputData = np.append(mdsRandEucOutputData, randInputData, 1)

with open('Random_MDS_Euc_dengue.csv', 'w', newline = '') as f:
	writer = csv.writer(f)
	writer.writerows(np.append([['x','y']], rowName, 1))
	writer.writerows(mdsRandEucOutputData)


kmInputData = np.asarray(kmInputData)
mdsKmEucOutputData = MDS(n_components=2, max_iter=300).fit(kmInputData).embedding_
mdsKmEucOutputData = np.append(mdsKmEucOutputData, kmInputData, 1)
mdsKmEucOutputData = np.append(mdsKmEucOutputData, kmInputDataGroup, 1)

with open('KMean_MDS_Euc_dengue.csv', 'w', newline = '') as f:
	writer = csv.writer(f)
	writer.writerows(np.append(np.append([['x','y']], rowName, 1), [['group']], 1))
	writer.writerows(mdsKmEucOutputData)

## Task 3(b) correlation_distances

similarities = euclidean_distances(randInputData)
mdsRandCorrOutputData = MDS(n_components=2, dissimilarity="precomputed").fit(similarities).embedding_
mdsRandCorrOutputData = np.append(mdsRandCorrOutputData, randInputData, 1)

with open('Random_MDS_Corr_dengue.csv', 'w', newline = '') as f:
	writer = csv.writer(f)
	writer.writerows(np.append([['x','y']], rowName, 1))
	writer.writerows(mdsRandCorrOutputData)


mdsKmCorrOutputData = MDS(n_components=2, max_iter=300).fit(kmInputData).embedding_
mdsKmCorrOutputData = np.append(mdsKmCorrOutputData, kmInputData, 1)
mdsKmCorrOutputData = np.append(mdsKmCorrOutputData, kmInputDataGroup, 1)

with open('KMean_MDS_Corr_dengue.csv', 'w', newline = '') as f:
	writer = csv.writer(f)
	writer.writerows(np.append(np.append([['x','y']], rowName, 1), [['group']], 1))
	writer.writerows(mdsKmCorrOutputData)
