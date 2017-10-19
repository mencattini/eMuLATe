package machine.learning

/**
 * @author Romain Mencattini
 *
 * @param lastPositionPrice the last price on which we take position
 * @param currentPrice the current price
 * @param lastPosition is the last non null position we got
 * @param holdPosition says if we are holding a position or not
 */

internal class Position(
        var lastPositionPrice: Double, var currentPrice: Double, var lastPosition : Double, var holdPosition : Boolean)