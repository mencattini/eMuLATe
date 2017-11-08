package machine.learning

/**
 * @author Romain Mencattini
 *
 * @param maxPnl the maximum p&l reach during the position life
 * @param currentPnl the current p&l
 * @param lastPosition the current position {long, short, nothing}
 */

internal class Position(var maxPnl: Double, var currentPnl: Double, var lastPosition: Double)