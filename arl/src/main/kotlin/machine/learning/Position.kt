package machine.learning

/**
 * @author Romain Mencattini
 *
 * @param lastPositionPrice the last price on which we take position
 * @param currentPrice the current price
 * @param lastPosition is the last non null position we got
 * @param closePosition true if we want to close position
 * @param doNothing true if the threshold of layer 2 is triggered
 */

internal class Position(
        var lastPositionPrice: Double, var currentPrice: Double, var lastPosition : Double, var doNothing : Boolean,
        var closePosition : Boolean)