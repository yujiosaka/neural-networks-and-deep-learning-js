###
network.coffee
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
###

'use strict'

_ = require 'lodash'
jStat = require('jStat').jStat
linearAlgebra = require('linear-algebra')()
Matrix = linearAlgebra.Matrix

lib = require './lib'

class Network

  constructor: (sizes) ->
    ###
    The list ``sizes`` contains the number of neurons in the
    respective layers of the network.  For example, if the list
    was [2, 3, 1] then it would be a three-layer network, with the
    first layer containing 2 neurons, the second layer 3 neurons,
    and the third layer 1 neuron.  The biases and weights for the
    network are initialized randomly, using a Gaussian
    distribution with mean 0, and variance 1.  Note that the first
    layer is assumed to be an input layer, and by convention we
    won't set any biases for those neurons, since biases are only
    ever used in computing the outputs from later layers.
    ###
    @numLayers = sizes.length
    @sizes = sizes
    @biases = (new Matrix(jStat.randn(y, 1)) for y in sizes.slice(1))
    @weights = (new Matrix(jStat.randn(y, x)) for [x, y] in _.zip(sizes.slice(0, -1), sizes.slice(1)))

  feedforward: (a) ->
    ###
    Return the output of the network if ``a`` is input.
    ###
    for [b, w] in _.zip(@biases, @weights)
      a = w.dot(a).plus(b).sigmoid()
    a

  SGD: (trainingData, epochs, miniBatchSize, eta, opts = {}) ->
    ###
    Train the neural network using mini-batch stochastic
    gradient descent.  The ``trainingData`` is a list of arrays
    ``[x, y]`` representing the training inputs and the desired
    outputs.  The other non-optional parameters are
    self-explanatory.  If ``testData`` is provided then the
    network will be evaluated against the test data after each
    epoch, and partial progress printed out.  This is useful for
    tracking progress, but slows things down substantially.
    ###

    opts.testData or= null

    nTest = opts.testData.length if opts.testData
    n = trainingData.length
    for j in [0...epochs]
      trainingData = _.shuffle(trainingData)
      miniBatches = for k in [0...n] by miniBatchSize
        trainingData[k...(k + miniBatchSize)]
      for miniBatch in miniBatches
        @updateMiniBatch(miniBatch, eta)
      if opts.testData
        console.log "Epoch #{j}: #{@evaluate(opts.testData)} / #{nTest}"
      else
        console.log "Epoch #{j} complete"

  updateMiniBatch: (miniBatch, eta) ->
    ###
    Update the network's weights and biases by applying
    gradient descent using backpropagation to a single mini batch.
    The ``miniBatch`` is a list of arrays ``[x, y]``, and ``eta``
    is the learning rate.
    ###
    nablaB = (Matrix.zeros(b.rows, b.cols) for b in @biases)
    nablaW = (Matrix.zeros(w.rows, w.cols) for w in @weights)
    for [x, y] in miniBatch
      [deltaNablaB, deltaNablaW] = @backprop(x, y)
      nablaB = (nb.plus(dnb) for [nb, dnb] in _.zip(nablaB, deltaNablaB))
      nablaW = (nw.plus(dnw) for [nw, dnw] in _.zip(nablaW, deltaNablaW))
    @weights = (w.minus(nw.mulEach(eta / miniBatch.length)) for [w, nw] in _.zip(@weights, nablaW))
    @biases = (b.minus(nb.mulEach(eta / miniBatch.length)) for [b, nb] in _.zip(@biases, nablaB))

  backprop: (x, y) ->
    ###
    Return a array ``[nablaB, nablaW]`` representing the
    gradient for the cost function C_x.  ``nablaB`` and
    ``nablaW`` are layer-by-layer lists of numpy arrays, similar
    to ``this.biases`` and ``this.weights``.
    ###
    nablaB = (Matrix.zeros(b.rows, b.cols) for b in @biases)
    nablaW = (Matrix.zeros(w.rows, w.cols) for w in @weights)
    activation = x
    activations = [x]
    zs = []
    for [b, w] in _.zip(@biases, @weights)
      z = w.dot(activation).plus(b)
      zs.push(z)
      activation = z.sigmoid()
      activations.push(activation)
    delta = @costDerivative(activations[activations.length - 1], y).mul(lib.sigmoidPrime(zs[zs.length - 1]))
    nablaB[nablaB.length - 1] = delta
    nablaW[nablaW.length - 1] = delta.dot(activations[activations.length - 2].trans())
    for l in [2...@numLayers]
      z = zs[zs.length - l]
      sp = lib.sigmoidPrime(z)
      delta = @weights[@weights.length - l + 1].trans().dot(delta).mul(sp)
      nablaB[nablaB.length - l] = delta
      nablaW[nablaW.length - l] = delta.dot(activations[activations.length - l - 1].trans())
    [nablaB, nablaW]

  evaluate: (testData) ->
    ###
    Return the number of test inputs for which the neural
    network outputs the correct result. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation.
    ###
    testResults = for [x, y] in testData
      [@feedforward(x).getArgMax(), y]
    _.sum (+(x is y) for [x, y] in testResults)

  costDerivative: (outputActivations, y) ->
    ###
    Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations.
    ###
    outputActivations.minus(y)

module.exports = Network
