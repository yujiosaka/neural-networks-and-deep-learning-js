'use strict'

class CrossEntropyCost

  @className: 'CrossEntropyCost'

  @fn: (a, y) ->
    ###
    Return the cost associated with an output ``a`` and desired output
    ``y``.  Note that np.nan_to_num is used to ensure numerical
    stability.  In particular, if both ``a`` and ``y`` have a 1.0
    in the same slot, then the expression (1-y)*np.log(1-a)
    returns nan.  The np.nan_to_num ensures that that is converted
    to the correct value (0.0).
    ###
    y
    .mul(a.log())
    .mulEach(-1)
    .minus(y.mulEach(-1)
           .plusEach(1)
           .mul(a.mulEach(-1)
                .plusEach(1)
                .log()
                )
    )
    .nanToNum()
    .getSum()

  @delta: (z, a, y) ->
    ###
    Return the error delta from the output layer.  Note that the
    parameter ``z`` is not used by the method.  It is included in
    the method's parameters in order to make the interface
    consistent with the delta method for other cost classes.
    ###
    a.minus(y)

module.exports = CrossEntropyCost
