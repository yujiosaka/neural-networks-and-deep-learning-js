#!/usr/bin/env coffee
'use strict'

mnistLoader = require 'mnist_loader'
Network = require 'network2'
CrossEntropyCost = require 'costs/cross_entropy_cost'

[trainingData, validationData, testData] = mnistLoader.loadDataWrapper()
net = new Network [784, 30, 10],
  cost: CrossEntropyCost
net.SGD trainingData, 30, 10, 0.5,
  evaluationData: testData
  lmbda: 0.1
  # monitorEvaluationCost: true
  monitorEvaluationAccuracy: true
  # monitorTrainingCost: true
  monitorTrainingAccuracy: true
