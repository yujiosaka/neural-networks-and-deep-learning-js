'use strict'

jStat = require('jStat').jStat

lib = require '../lib'

class QuadraticCost

  @className: 'QuadraticCost'

  @fn: (a, y) ->
    ###
    Return the cost associated with an output ``a`` and desired output
    ``y``.
    ###
    0.5 * jStat(a.minus(y).toArray()).norm() ** 2

  @delta: (z, a, y) ->
    ###
    Return the error delta from the output layer.
    ###
    a.minus(y).mul(lib.sigmoidPrime(z))

module.exports = QuadraticCost
