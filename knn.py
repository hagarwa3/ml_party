import csv
import random
import math
import operator

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    csvfile = open(filename, 'rb')
    lines = csv.reader(csvfile)
    dataset = list(lines)
    for x in range(len(dataset)):
        for y in range(len(dataset[x])):
            dataset[x][y] = float(dataset[x][y])
        if random.random() < split:
            trainingSet.append(dataset[x])
        else:
            testSet.append(dataset[x])
 

def distance(a, b, length):
    distance = 0
    for x in range(length):
        distance += pow((a[x] - b[x]), 2)
    return math.sqrt(distance)

def neighbors(train, test_elem, k):
    dists = []
    for x in train:
        dists.append((x,distance(test_elem, x, len(test_elem)-1)))
    dists.sort(key=operator.itemgetter(1))
    neigh = []
    for x in range(k):
        neigh.append(dists[x][0])
    return neigh

def solve(neighbors):
    appearances = {}
    for x in neighbors:
        ans = x[-1]
        if ans in appearances:
            appearances[ans]+=1
        else:
            appearances[ans]=1
    sortedVotes = sorted(appearances.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def main(filename_train, filename_test, k):
    train = []
    test = []
    split = 1
    loadDataset(filename_train, split, train, train)
    loadDataset(filename_test, split, test, test)

    print 'Train set: ' + repr(len(train))
    print 'Test set: ' + repr(len(test))
	# generate predictions
    predictions=[]
    #k = 3
    for x in test:
		neigh = neighbors(train, x, k)
		result = solve(neigh)
		predictions.append(result)
    return predictions

#print str(main('C:/Users/Harshit Agarwal/Desktop/py scripts/1.csv', 'C:/Users/Harshit Agarwal/Desktop/py scripts/2.csv', 3))
    