#!/usr/bin/env coffee
'use strict'

mnistLoader = require 'mnist_loader'
Network = require 'network'

[trainingData, validationData, testData] = mnistLoader.loadDataWrapper()

net = new Network([784, 30, 10])
net.SGD(trainingData, 30, 10, 3, testData: testData)
