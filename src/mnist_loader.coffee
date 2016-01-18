'use strict'

_ = require 'lodash'
linearAlgebra = require('linear-algebra')()
Matrix = linearAlgebra.Matrix

lib = require './lib'

loadData = ->
  trainingInput = require '../data/training_input'
  trainingOutput = require '../data/training_output'
  validationInput = require '../data/validation_input'
  validationOutput = require '../data/validation_output'
  testInput = require '../data/test_input'
  testOutput = require '../data/test_output'
  [
    [trainingInput, trainingOutput]
    [validationInput, validationOutput]
    [testInput, testOutput]
  ]

exports.loadDataWrapper = ->
  [trD, vaD, teD] = loadData()
  trainingInputs = (Matrix.reshape(x, 784, 1) for x in trD[0])
  trainingResults = (lib.vectorizedResult(y) for y in trD[1])
  trainingData = _.zip(trainingInputs, trainingResults)
  validationInputs = (Matrix.reshape(x, 784, 1) for x in vaD[0])
  validationData = _.zip(validationInputs, vaD[1])
  testInputs = (Matrix.reshape(x, 784, 1) for x in teD[0])
  testData = _.zip(testInputs, teD[1])
  [trainingData, validationData, testData]
