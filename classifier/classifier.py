#!/usr/bin/env python3
# coding: utf-8
import os
import csv
import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from functools import reduce
import pickle

def distance(p1x,p1y,p2x,p2y):
    return (p2x-p1x)**2 + (p2y-p1y)**2

def getClassifier(pathToPkl):
    with open(pathToPkl, 'rb') as fid:
        classifier = pickle.load(fid)
        return classifier

# def predict(classifier, data):
# 	distanceData = []
# 	distanceData.append(distance(data[2],data[3],data[10],data[11]))
# 	distanceData.append(distance(data[4],data[5],data[8],data[9]))
# 	distanceData.append(distance(data[2+12],data[3+12],data[10+12],data[11+12]))
# 	distanceData.append(distance(data[4+12],data[5+12],data[8+12],data[9+12]))
# 	return classifier.predict([distanceData])[0]

# def createPredictor(classifier):
# 	def predict(data):
# 		distanceData = []
# 		distanceData.append(distance(data[2],data[3],data[10],data[11]))
# 		distanceData.append(distance(data[4],data[5],data[8],data[9]))
# 		distanceData.append(distance(data[2+12],data[3+12],data[10+12],data[11+12]))
# 		distanceData.append(distance(data[4+12],data[5+12],data[8+12],data[9+12]))
# 		return classifier.predict([distanceData])[0]

# 	return predict

def predict(classifier, data):
    distanceData = [
        distance(data[2],data[3],data[10],data[11]),
        distance(data[4],data[5],data[8],data[9]),
        distance(data[0],data[1],data[6],data[7]),
        distance(data[2+12],data[3+12],data[10+12],data[11+12]),
        distance(data[4+12],data[5+12],data[8+12],data[9+12]),
        distance(data[0+12],data[1+12],data[6+12],data[7+12])]
    return classifier.predict([distanceData])[0]

def createPredictor(classifier):
    def predict(data):
        distanceData = [
            distance(data[2],data[3],data[10],data[11]),
            distance(data[4],data[5],data[8],data[9]),
            distance(data[0],data[1],data[6],data[7]),
            distance(data[2+12],data[3+12],data[10+12],data[11+12]),
            distance(data[4+12],data[5+12],data[8+12],data[9+12]),
            distance(data[0+12],data[1+12],data[6+12],data[7+12])]
        return classifier.predict([distanceData])[0]
    
    
    return predict