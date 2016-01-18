'use strict'

linearAlgebra = require('linear-algebra')()
Matrix = linearAlgebra.Matrix

exports.sigmoidPrime = (z) ->
  ###
  Derivative of the sigmoid function.
  ###
  z.sigmoid().mul(z.sigmoid().mulEach(-1).plusEach(1))

exports.vectorizedResult = (j) ->
  ###
  Return a 10-dimensional unit vector with a 1.0 in the j'th position
  and zeroes elsewhere.  This is used to convert a digit (0...9)
  into a corresponding desired output from the neural network.
  ###
  e = ([0] for i in [0...10])
  e[j] = [1]
  new Matrix(e)

exports.dropoutLayer = (layer, pDropout) ->
  layer.eleMap (elem) ->
    if Math.random() < pDropout then 0 else elem
