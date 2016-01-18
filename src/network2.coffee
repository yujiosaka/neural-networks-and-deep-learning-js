###
network2.coffee
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

###

'use strict'

fs = require 'fs'
_ = require 'lodash'
jStat = require('jStat').jStat
linearAlgebra = require('linear-algebra')()
Matrix = linearAlgebra.Matrix

lib = require './lib'
CrossEntropyCost = require './costs/cross_entropy_cost'

class Network

  constructor: (sizes, opts = {}) ->
    ###
    The list ``sizes`` contains the number of neurons in the respective
    layers of the network.  For example, if the list was [2, 3, 1]
    then it would be a three-layer network, with the first layer
    containing 2 neurons, the second layer 3 neurons, and the
    third layer 1 neuron.  The biases and weights for the network
    are initialized randomly, using
    ``this.defaultWeightInitializer`` (see docstring for that
    method).
    ###

    opts.cost or= CrossEntropyCost

    @numLayers = sizes.length
    @sizes = sizes
    @defaultWeightInitializer()
    @cost = opts.cost

  defaultWeightInitializer: ->
    ###
    Initialize each weight using a Gaussian distribution with mean 0
    and standard deviation 1 over the square root of the number of
    weights connecting to the same neuron.  Initialize the biases
    using a Gaussian distribution with mean 0 and standard
    deviation 1.

    Note that the first layer is assumed to be an input layer, and
    by convention we won't set any biases for those neurons, since
    biases are only ever used in computing the outputs from later
    layers.
    ###
    @biases = (new Matrix(jStat.randn(y, 1)) for y in @sizes.slice(1))
    @weights = (new Matrix(jStat.randn(y, x)).mulEach(1 / Math.sqrt(x)) for [x, y] in _.zip(@sizes.slice(0, -1), @sizes.slice(1)))

  largeWeightInitializer: ->
    ###
    Initialize the weights using a Gaussian distribution with mean 0
    and standard deviation 1.  Initialize the biases using a
    Gaussian distribution with mean 0 and standard deviation 1.

    Note that the first layer is assumed to be an input layer, and
    by convention we won't set any biases for those neurons, since
    biases are only ever used in computing the outputs from later
    layers.

    This weight and bias initializer uses the same approach as in
    Chapter 1, and is included for purposes of comparison.  It
    will usually be better to use the default weight initializer
    instead.
    ###
    @biases = (new Matrix(jStat.randn(y, 1)) for y in @sizes.slice(1))
    @weights = (new Matrix(jStat.randn(y, x)) for [x, y] in _.zip(@sizes.slice(0, -1), @sizes.slice(1)))

  feedforward: (a) ->
    ###Return the output of the network if ``a`` is input.###
    for [b, w] in _.zip(@biases, @weights)
      a = w.dot(a).plus(b).sigmoid()
    a

  SGD: (trainingData, epochs, miniBatchSize, eta, opts = {}) ->
    ###
    Train the neural network using mini-batch stochastic gradient
    descent.  The ``trainingData`` is a list of arrays ``[x, y]``
    representing the training inputs and the desired outputs.  The
    other non-optional parameters are self-explanatory, as is the
    regularization parameter ``lmbda``.  The method also accepts
    ``evaluationData``, usually either the validation or test
    data.  We can monitor the cost and accuracy on either the
    evaluation data or the training data, by setting the
    appropriate flags.  The method returns a array containing four
    lists: the (per-epoch) costs on the evaluation data, the
    accuracies on the evaluation data, the costs on the training
    data, and the accuracies on the training data.  All values are
    evaluated at the end of each training epoch.  So, for example,
    if we train for 30 epochs, then the first element of the array
    will be a 30-element list containing the cost on the
    evaluation data at the end of each epoch. Note that the lists
    are empty if the corresponding flag is not set.
    ###
    opts.lmbda or= 0
    opts.evaluationData or= null
    opts.monitorEvaluationCost or= false
    opts.monitorEvaluationAccuracy or= false
    opts.monitorTrainingCost or= false
    opts.monitorTrainingAccuracy or= false

    nData = opts.evaluationData.length if opts.evaluationData
    n = trainingData.length
    evaluationCost = []
    evaluationAccuracy = []
    trainingCost = []
    trainingAccuracy = []
    for j in [0...epochs]
      trainingData = _.shuffle(trainingData)
      miniBatches = for k in [0...n] by miniBatchSize
        trainingData[k...(k + miniBatchSize)]
      for miniBatch in miniBatches
        @updateMiniBatch(miniBatch, eta, opts.lmbda, trainingData.length)
      console.log "Epoch #{j} training complete"
      if opts.monitorTrainingCost
        cost = @totalCost(trainingData, opts.lmbda)
        trainingCost.push(cost)
        console.log "Cost on training data: #{cost}"
      if opts.monitorTrainingAccuracy
        accuracy = @accuracy(trainingData, convert: true)
        trainingAccuracy.push(accuracy)
        console.log "Accuracy on training data: #{accuracy} / #{n}"
      if opts.monitorEvaluationCost
        cost = @totalCost(opts.evaluationData, opts.lmbda, convert: true)
        evaluationCost.push(cost)
        console.log "Cost on evaluation data: #{cost}"
      if opts.monitorEvaluationAccuracy
        accuracy = @accuracy(opts.evaluationData)
        evaluationAccuracy.push(accuracy)
        console.log "Accuracy on evaluation data: #{@accuracy(opts.evaluationData)} / #{nData}"
    [evaluationCost, evaluationAccuracy, trainingCost, trainingAccuracy]

  updateMiniBatch: (miniBatch, eta, lmbda, n) ->
    ###
    Update the network's weights and biases by applying gradient
    descent using backpropagation to a single mini batch.  The
    ``miniBatch`` is a list of arrays ``[x, y]``, ``eta`` is the
    learning rate, ``lmbda`` is the regularization parameter, and
    ``n`` is the total size of the training data set.
    ###
    nablaB = (Matrix.zeros(b.rows, b.cols) for b in @biases)
    nablaW = (Matrix.zeros(w.rows, w.cols) for w in @weights)
    for [x, y] in miniBatch
      [deltaNablaB, deltaNablaW] = @backprop(x, y)
      nablaB = (nb.plus(dnb) for [nb, dnb] in _.zip(nablaB, deltaNablaB))
      nablaW = (nw.plus(dnw) for [nw, dnw] in _.zip(nablaW, deltaNablaW))
    @weights = (w.mulEach(1 - eta * (lmbda / n)).minus(nw.mulEach(eta / miniBatch.length)) for [w, nw] in _.zip(@weights, nablaW))
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
    delta = @cost.delta(zs[zs.length - 1], activations[activations.length - 1], y)
    nablaB[nablaB.length - 1] = delta
    nablaW[nablaW.length - 1] = delta.dot(activations[activations.length - 2].trans())
    for l in [2...@numLayers]
      z = zs[zs.length - l]
      sp = lib.sigmoidPrime(z)
      delta = @weights[@weights.length - l + 1].trans().dot(delta).mul(sp)
      nablaB[nablaB.length - l] = delta
      nablaW[nablaW.length - l] = delta.dot(activations[activations.length - l - 1].trans())
    [nablaB, nablaW]

  accuracy: (data, opts = {}) ->
    ###
    Return the number of inputs in ``data`` for which the neural
    network outputs the correct result. The neural network's
    output is assumed to be the index of whichever neuron in the
    final layer has the highest activation.

    The flag ``convert`` should be set to false if the data set is
    validation or test data (the usual case), and to true if the
    data set is the training data. The need for this flag arises
    due to differences in the way the results ``y`` are
    represented in the different data sets.  In particular, it
    flags whether we need to convert between the different
    representations.  It may seem strange to use different
    representations for the different data sets.  Why not use the
    same representation for all three data sets?  It's done for
    efficiency reasons -- the program usually evaluates the cost
    on the training data and the accuracy on other data sets.
    These are different types of computations, and using different
    representations speeds things up.  More details on the
    representations can be found in
    mnistLoader.loadDataWrapper.
    ###
    opts.convert or= false

    if opts.convert
      results = ([@feedforward(x).getArgMax(), y.getArgMax()] for [x, y] in data)
    else
      results = ([@feedforward(x).getArgMax(), y] for [x, y] in data)
    _.sum (+(x is y) for [x, y] in results)

  totalCost: (data, lmbda, opts = {}) ->
    ###
    Return the total cost for the data set ``data``.  The flag
    ``convert`` should be set to false if the data set is the
    training data (the usual case), and to true if the data set is
    the validation or test data.  See comments on the similar (but
    reversed) convention for the ``accuracy`` method, above.
    ###
    opts.convert or= false

    cost = 0
    for [x, y] in data
      a = @feedforward(x)
      y = lib.vectorizedResult(y) if opts.convert
      cost += @cost.fn(a, y) / data.length
    cost += 0.5 * (lmbda / data.length) * _.sum(w.getNorm() ** 2 for w in @weights)
    cost

  save: (filename) ->
    ###
    Save the neural network to the file ``filename``.
    ###
    data =
      sizes: @sizes
      weights: (w.toArray() for w in @weights)
      biases: (b.toArray() for b in @biases)
      cost: str(@cost.className)
    fs.writeFileSync JSON.stringify(data)

  @load: (filename) ->
    ###
    Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    ###
    data = JSON.parse fs.readFileSync(filename)
    cost = eval(data.cost)
    net = new Network(data.sizes, cost)
    net.weights = (new Matrix(w) for w in data.weights)
    net.biases = (new Matrix(b) for b in data.biases)
    net

module.exports = Network
