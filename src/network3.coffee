'use strict'

_ = require 'lodash'
linearAlgebra = require('linear-algebra')()
Matrix = linearAlgebra.Matrix

lib = require './lib'

class Network

  constructor: (layers) ->
    @layers = layers

  SGD: (trainingData, epochs, miniBatchSize, eta, opts = {}) ->
    opts.validationData or= null
    opts.testData or= null
    opts.lmbda or= 0

    bestValidationAccuracy = 0
    testAccuracy = null
    for j in [0...epochs]
      trainingData = _.shuffle trainingData
      miniBatches = @createMiniBatches trainingData, miniBatchSize
      for miniBatch, i in miniBatches
        iteration = trainingData.length / miniBatchSize * j + i
        if iteration % 1000 is 0
          console.log "Training mini-batch number #{iteration}"
        @updateMiniBatch miniBatch, eta, opts.lmbda, trainingData.length
      if opts.validationData
        validationAccuracy = @accuracy opts.validationData
        console.log "Epoch #{j}: validation accuracy #{validationAccuracy}"
        if validationAccuracy >= bestValidationAccuracy
          console.log 'This is the best validation accuracy to date.'
          bestValidationAccuracy = validationAccuracy
          if opts.testData
            testAccuracy = @accuracy opts.testData
            console.log "The corresponding test accuracy #{testAccuracy}"
      else if opts.testData
        testAccuracy = @accuracy opts.testData
        console.log "Epoch #{j}: test accuracy #{testAccuracy}"
    console.log 'Finished training network.'
    if opts.validationData
      console.log "Best validation accuracy #{bestValidationAccuracy}"
      if opts.testData
        console.log "Corresponding test accuracy #{testAccuracy}"

  createMiniBatches: (trainingData, miniBatchSize) ->
    (trainingData[k...(k + miniBatchSize)] for k in [0...trainingData.length] by miniBatchSize)

  updateMiniBatch: (miniBatch, eta, lmbda, n) ->
    x = new Matrix(_x.ravel() for [_x, _y] in miniBatch).trans()
    y = new Matrix(_y.ravel() for [_x, _y] in miniBatch).trans()
    @train x, miniBatch.length
    @backprop y
    for layer in @layers
      layer.w = layer.w.mulEach(1 - eta * (lmbda / n)).minus (layer.nw.mulEach(eta / miniBatch.length))
      layer.b = layer.b.minus layer.nb.mulEach(eta / miniBatch.length)

  train: (x, miniBatchSize) ->
    initLayer = @layers[0]
    initLayer.setInput x, x, miniBatchSize
    for j in [1...@layers.length]
      prevLayer = @layers[j - 1]
      layer = @layers[j]
      layer.setInput prevLayer.output, prevLayer.outputDropout, miniBatchSize

  backprop: (y) ->
    lastLayer = @layers[@layers.length - 1]
    delta = lastLayer.costDelta y
    lastLayer.update delta
    for l in [2...(@layers.length + 1)]
      followinglayer = @layers[@layers.length - l + 1]
      layer = @layers[@layers.length - l]
      delta = followinglayer.w.trans().dot(delta).mul lib.sigmoidPrime(layer.z)
      layer.update(delta)

  accuracy: (data) ->
    _.mean (@feedforward(x).accuracy(y) for [x, y] in data)

  feedforward: (a) ->
    @train a, 1
    @layers[@layers.length - 1]

  test: (data) ->
    @accuracy data

  predict: (inputs) ->
    (@feedforward(x).yOut for x in inputs)

module.exports = Network
