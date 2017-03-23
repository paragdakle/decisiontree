#!/usr/bin/python
import csv
import sys
import math
import random
import copy
from collections import Counter

INTERNAL_NODE = 1
LEAF_NODE = 2
numberOfArguments = 7
dataClasses = []

def extractData(reader):
	data = []
	headers = []
	rownum = 0
	for row in reader:
		if rownum == 0:
			headers = row
			rownum = 1
		else:
			itemClass = row[len(row) - 1]
			if itemClass not in dataClasses:
				dataClasses.append(itemClass)
			data.append(row)
	return headers, data

def createInternalNode(decidingAttr):
	node = {}
	node['nodeType'] = INTERNAL_NODE
	node['attr'] = decidingAttr
	node['cl'] = Counter()
	for dataClass in dataClasses:
		node[dataClass] = {}
	return node

def createLeafNode(leafClass):
	node = {}
	node['nodeType'] = LEAF_NODE
	node['class'] = leafClass
	return node

"""
Functions to calculate entropy heuristic
"""
def getDataEntropy(data, headers):
	if len(data) > 0:
		entropy = 1
		attributeIndex = len(headers) - 1
		classCount = Counter() 
		for dataItem in data:
			classCount[dataItem[attributeIndex]] += 1.0
		entropy = 0
		for item in ['0', '1']:
			if classCount[item] > 0:
				entropy += (-1) * (classCount[item]/len(data)) * math.log(classCount[item]/len(data))
		return entropy
	return 0

def getEntropyOfAttribute(data, headers, attribute, value):
	if attribute in headers:
		attributeIndex = headers.index(attribute)
		classIndex = len(headers) - 1
		classCount = Counter()
		for dataItem in data:
			if dataItem[attributeIndex] == value:
				classCount[dataItem[classIndex]] += 1.0
				classCount['size'] += 1.0
		entropy = 0
		for item in ['0', '1']:
			if classCount[item] > 0:
				entropy += (-1) * (classCount[item]/classCount['size']) * math.log(classCount[item]/classCount['size'])
		return entropy, classCount['size']
	return 0

def getEGainOnAttribute(data, headers, attribute):
	if attribute in headers:
		positiveClassEntropy, positiveClassSize = getEntropyOfAttribute(data, headers, attribute, '1')
		negativeClassEntropy, negativeClassSize = getEntropyOfAttribute(data, headers, attribute, '0')
		dataEntrophy = getDataEntropy(data, headers)
		gain = dataEntrophy - (((positiveClassSize / len(data)) * positiveClassEntropy) + ((negativeClassSize / len(data)) * negativeClassEntropy))
		return gain
	return 0

def getBestClassifierAttributeEntropyHeuristic(data, headers):
	if len(data) > 0 and len(headers) > 1:
		gainList = []
		for counter in range(0, len(headers) - 1):
			gainList.append(getEGainOnAttribute(data, headers, headers[counter]))
		return headers[gainList.index(max(gainList))]
	return headers[0]

"""
Functions to calculate variance impurity heuristic
"""
def getDataVarianceImpurity(data, headers):
	if len(data) > 0:
		attributeIndex = len(headers) - 1
		classCount = Counter() 
		for dataItem in data:
			classCount[dataItem[attributeIndex]] += 1.0
		varianceImpurity = 1
		for item in ['0', '1']:
			varianceImpurity *= classCount[item]/len(data)
		return varianceImpurity
	return 0

def getVarianceImpurityOfAttribute(data, headers, attribute, value):
	if attribute in headers:
		attributeIndex = headers.index(attribute)
		classIndex = len(headers) - 1
		classCount = Counter()
		for dataItem in data:
			if dataItem[attributeIndex] == value:
				classCount[dataItem[classIndex]] += 1.0
				classCount['size'] += 1.0
		varianceImpurity = 1
		for item in ['0', '1']:
			if classCount[item] > 0:
				varianceImpurity *= classCount[item]/classCount['size']
			else:
				return 0, classCount['size']
		return varianceImpurity, classCount['size']
	return 0

def getVIGainOnAttribute(data, headers, attribute):
	if attribute in headers:
		positiveClassVI, positiveClassSize = getVarianceImpurityOfAttribute(data, headers, attribute, '1')
		negativeClassVI, negativeClassSize = getVarianceImpurityOfAttribute(data, headers, attribute, '0')
		dataVI = getDataVarianceImpurity(data, headers)
		gain = dataVI - (((positiveClassSize / len(data)) * positiveClassVI) + ((negativeClassSize / len(data)) * negativeClassVI))
		return gain
	return 0

def getBestClassifierAttributeVIHeuristic(data, headers):
	if len(data) > 0 and len(headers) > 1:
		gainList = []
		for counter in range(0, len(headers) - 1):
			gainList.append(getVIGainOnAttribute(data, headers, headers[counter]))
		return headers[gainList.index(max(gainList))]
	return headers[0]

def partitionDataSet(data, headers, attr):
	attrPosition = headers.index(attr)
	newDataSet = {}
	for dataClass in dataClasses:
		newDataSet[dataClass] = []
	for dataItem in data:
		newItem = copy.deepcopy(dataItem)
		del newItem[attrPosition]
		newDataSet[dataItem[attrPosition]].append(newItem)
	newHeaders = copy.deepcopy(headers)
	del newHeaders[attrPosition]
	return newDataSet, newHeaders

def getClassDivision(data, headers):
	classCounter = Counter()
	if data and headers:
		classIndex = len(headers) - 1
		for dataItem in data:
			classCounter[dataItem[classIndex]] += 1
	return classCounter

def trainDecisionTree(data, headers, heuristic):
	root = {}
	if len(headers) == 1:
		dataClassDir = {}
		for dataClass in dataClasses:
			dataClassDir[dataClass] = 0
		for item in data:
			dataClassDir[item[len(item) - 1]] += 1
		root = createLeafNode(dataClassDir.keys()[max(dataClassDir.values())])
	else:
		classCounter = {}
		for item in data:
			if classCounter.has_key(item[len(item) - 1]):
				classCounter[item[len(item) - 1]] += 1
			else:
				classCounter[item[len(item) - 1]] = 1
		for cl in classCounter.keys():
			if classCounter[cl] == len(data):
				return createLeafNode(cl)
	attr = heuristic(data, headers)
	root = createInternalNode(attr)
	root['cl'] = getClassDivision(data, headers)
	newDataSet, headers = partitionDataSet(data, headers, attr)
	childClassCount = 0
	nextClass = ''
	for dataClass in dataClasses:
		if len(newDataSet[dataClass]) > 0 and len(headers) > 1:
			childClassCount += 1
			nextClass = dataClass
			root[dataClass] = trainDecisionTree(newDataSet[dataClass], headers, heuristic)
	if childClassCount == 1:
		return createLeafNode(nextClass)
	return root

def testDecisionTree(data, headers, decisionTree):
	results = Counter()
	for dataItem in data:
		currentNode = decisionTree
		while currentNode.get('nodeType') != LEAF_NODE:
			attr = currentNode.get('attr')
			attrValue = dataItem[headers.index(attr)]
			currentNode = currentNode.get('' + attrValue)
		if currentNode.get('class') == dataItem[len(dataItem) - 1]:
			results['1'] += 1.0
		else:
			results['0'] += 1.0
	return (results['1'] * 100.0)/ len(data)

def train(trainingDataFile, heuristic):
	
	ifile  = open(trainingDataFile, "rb")
	reader = csv.reader(ifile)
	headers, data = extractData(reader)
	ifile.close()

	tree = trainDecisionTree(data, headers, heuristic)

	return tree

def test(testDataFile, decisionTree):

	ifile  = open(testDataFile, "rb")
	reader = csv.reader(ifile)
	headers, data = extractData(reader)
	ifile.close()
	
	return testDecisionTree(data, headers, decisionTree)

def getInternalNodeCount(tree):
	if(tree.get('nodeType') == LEAF_NODE):
		return 0
	return 1 + getInternalNodeCount(tree.get('0')) + getInternalNodeCount(tree.get('1'))

def pruneTree(tree, nodeToPrune):
	queue = []
	counter = 0
	queue.append(tree)
	pruningNode = tree
	while counter < nodeToPrune:
		pruningNode = queue[0]
		queue.remove(pruningNode)
		if pruningNode.get('nodeType') == INTERNAL_NODE:
			counter += 1
			queue.append(pruningNode.get('0'))
			queue.append(pruningNode.get('1'))
	classCounter = pruningNode.get('cl')
	dominantClassList = classCounter.most_common(1)
	dominantClass = dominantClassList[0]
	pruningNode['nodeType'] = LEAF_NODE
	pruningNode['class'] = dominantClass[0]
	del pruningNode['attr']
	del pruningNode['cl']
	for dataClass in dataClasses:
		del pruningNode[dataClass]
	return tree

def postPruneDecisionTree(tree, L, K, validationFilePath):
	bestDecisionTree = copy.deepcopy(tree)
	originalAccuracy = test(validationFilePath, tree)
	for i in range(L):
		newTree = copy.deepcopy(tree)
		M = random.randint(1, K)
		for j in range(M):
			N = getInternalNodeCount(newTree)
			if(N > 1):
				P = random.randint(1, N)
				newTree = pruneTree(newTree, P)
		accuracy = test(validationFilePath, newTree)
		if accuracy > originalAccuracy:
			bestDecisionTree = copy.deepcopy(newTree)
			originalAccuracy = accuracy
	return bestDecisionTree

def printTree(tree, level):
	prefix = "| " * level
	if level == 0:
		print tree.get('attr'), "=",
	else:
		print prefix, tree.get('attr'), "=",
	node = tree.get('0')
	if node.get('nodeType') == LEAF_NODE:
		print '0 :', node.get('class')
	else:
		print '0 :'
		printTree(node, level + 1)
	if level == 0:
		print tree.get('attr'), "=",
	else:
		print prefix, tree.get('attr'), "=",
	node = tree.get('1')
	if node.get('nodeType') == LEAF_NODE:
		print '1 :', node.get('class')
	else:
		print '1 :'
		printTree(node, level + 1)

def main(args):
	entropyTree = train(args[3], getBestClassifierAttributeEntropyHeuristic)
	results = test(args[5], entropyTree)
	print "Entropy Heuristic Test Results: %.2f percent accuracy." % (results)

	varianceImpurityTree = train(args[3], getBestClassifierAttributeVIHeuristic)
	results = test(args[5], varianceImpurityTree)
	print "Variance Impurity Heuristic Test Results: %.2f percent accuracy." % (results)

	# treeToPrune = copy.deepcopy(gainTree)
	entropyTree = postPruneDecisionTree(entropyTree, int(args[1]), int(args[2]), args[4])
	results = test(args[5], entropyTree)
	print "Entropy Heuristic Post Pruning Test Results: %.2f percent accuracy." % (results)

	# treeToPrune = copy.deepcopy(viTree)
	varianceImpurityTree = postPruneDecisionTree(varianceImpurityTree, int(args[1]), int(args[2]), args[4])
	results = test(args[5], varianceImpurityTree)
	print "Variance Impurity Heuristic Post Pruning Test Results: %.2f percent accuracy." % (results)

	if args[6].lower() == 'yes':
		print "Printing Decision Tree Constructed Using Entropy Heuristic"
		printTree(entropyTree, 0)
		print "Printing Decision Tree Constructed Using Variable Impurity Heuristic"
		printTree(varianceImpurityTree, 0)

if len(sys.argv) == numberOfArguments:
	main(sys.argv)
else:
	print "Invalid number of arguments found!"
	print "Expected:"
	print "python decisionTree.py <L> <K> <training-set> <validation-set> <test-set> <to-print>"
	print "L: integer (used in the post-pruning algorithm)"
	print "K: integer (used in the post-pruning algorithm)"
	print "to-print:{yes,no}"
