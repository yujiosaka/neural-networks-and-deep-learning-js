'use strict'

jStat = require('jStat').jStat
linearAlgebra = require('linear-algebra')()
Matrix = linearAlgebra.Matrix

lib = require '../lib'

class SoftmaxLayer

  constructor: (nIn, nOut, opts = {}) ->
    @nIn = nIn
    @nOut = nOut
    @pDropout = opts.pDropout or= 0
    @w = Matrix.zeros(@nOut, @nIn)
    @b = Matrix.zeros(@nOut, 1)

  setInput: (input, inputDropout, miniBatchSize) ->
    bMask = new Matrix(@b.ravel().map (v) -> (v for i in [0...miniBatchSize]))
    @input = input
    @z = @w.dot(input).mulEach(1 - @pDropout).plus(bMask)
    @output = @z.softmax(0)
    @yOut = @output.getArgMax()
    @inputDropout = lib.dropoutLayer inputDropout, @pDropout
    @outputDropout = @w.dot(@inputDropout).plus(bMask).softmax(0)

  accuracy: (y) ->
    @yOut is y

  costDelta: (y) ->
    @outputDropout.minus(y)

  update: (delta) ->
    @nb = new Matrix(delta.getSum(1)).trans()
    @nw = delta.dot @input.trans()

module.exports = SoftmaxLayer
