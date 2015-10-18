import csv
import random
import math
import operator

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def split(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separateByClass(dataset):
	result = {}
	for vector in dataset:
		if (vector[-1] not in result):
			result[vector[-1]] = []
		result[vector[-1]].append(vector)
	return result

def mean(numbers):
    if len(numbers)!=0:
	return float(sum(numbers)/(len(numbers)))

def standarddeviation(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
    blah = []
    for x in range(len(dataset)):
        blah.append([])
    for x in range(len(dataset)):
        for l in range(len(dataset[x])-1):
            blah[x].append(dataset[x][l])
    result = []
    for k in blah:
        result.append((mean(k),standarddeviation(k)))
    del result[-1]
    return result

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries


def gaussianProbability(x, mean, stdev):
    if stdev!=0:
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= gaussianProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
	
def main(trainfile, testfile):
	filename = trainfile
	splitRatio = 1
	dataset = loadCsv(filename)
	dataset2 = loadCsv (testfile)
	#trainingSet, la = split(dataset, splitRatio)
	#testSet, la = split(dataset2, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(dataset), len(dataset2))
	# prepare model
	summaries = summarizeByClass(dataset)
	# test model
	predictions = getPredictions(summaries, dataset2)
	return predictions

#print str(main('C:/Users/Harshit Agarwal/Desktop/py scripts/1.csv', 'C:/Users/Harshit Agarwal/Desktop/py scripts/2.csv'))