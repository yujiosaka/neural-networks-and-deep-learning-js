'use strict'

jStat = require('jStat').jStat
linearAlgebra = require('linear-algebra')()
Matrix = linearAlgebra.Matrix

lib = require '../lib'

class FullyConnectedLayer

  constructor: (nIn, nOut, opts = {}) ->
    @nIn = nIn
    @nOut = nOut
    @pDropout = opts.pDropout or= 0
    @w = new Matrix(jStat.randn(@nOut, @nIn)).mulEach(1 / Math.sqrt(@nIn))
    @b = new Matrix(jStat.randn(@nOut, 1))

  setInput: (input, inputDropout, miniBatchSize) ->
    bMask = new Matrix(@b.ravel().map (v) -> (v for i in [0...miniBatchSize]))
    @input = input
    @z = @w.dot(input).mulEach(1 - @pDropout).plus(bMask)
    @output = @z.sigmoid()
    @yOut = @output.getArgMax()
    @inputDropout = lib.dropoutLayer inputDropout, @pDropout
    @outputDropout = @w.dot(@inputDropout).plus(bMask).sigmoid()

  accuracy: (y) ->
    @yOut is y

  costDelta: (y) ->
    @outputDropout.minus(y).mul lib.sigmoidPrime(@z)

  update: (delta) ->
    @nb = new Matrix(delta.getSum(1)).trans()
    @nw = delta.dot @input.trans()

module.exports = FullyConnectedLayer
