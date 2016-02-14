#!/usr/bin/env coffee
'use strict'

mnistLoader = require 'mnist_loader'
Network = require 'network3'
FullyConnectedLayer = require 'layers/fully_connected_layer'
# FIXME!
# There is a bug in cost gradient function
# so it's learning speed is very very slow
SoftmaxLayer = require 'layers/softmax_layer'

[trainingData, validationData, testData] = mnistLoader.loadDataWrapper()
net = new Network [
  new FullyConnectedLayer(784, 100, pDropout: 0.5)
  new FullyConnectedLayer(100, 100, pDropout: 0.5)
  new FullyConnectedLayer(100, 10, pDropout: 0.5)
]
net.SGD trainingData, 60, 10, 0.1,
  validationData: validationData
  testData: testData
  lmbda: 0.1
