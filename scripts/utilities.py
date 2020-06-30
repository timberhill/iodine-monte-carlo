import numpy as np


def phases(tdata, start, period):
    """
    Get phases from time data, given start of the periods and the period.
    """
    return [ (t-start)/period - int((t-start)/period)  for t in tdata]


def percentiles(sigma):
    assert sigma in [1,2,3,4,5], "<sigma> must be one of 1, 2, 3, 4, 5."
    fractions = [68.2689492137086, 95.4499736103642, 99.7300203936740, 99.9936657516334, 99.9999426696856]
    return (50-fractions[sigma-1]/2, 50+fractions[sigma-1]/2 )


def getParabolaVertex(x, y):
    """
    Finds a parabola vertex coordinates from 3 points.

    x, list<3>: X coordinates of the 3 points,

    y, list<3>: Y coordinates of the 3 points.

    Returns:
        X and Y coordinates of the parabola peak.
    """
    denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
    A     = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
    B     = (x[2]*x[2] * (y[0] - y[1]) + x[1]*x[1] * (y[2] - y[0]) + x[0]*x[0] * (y[1] - y[2])) / denom
    C     = (x[1] * x[2] * (x[1] - x[2]) * y[0] + x[2] * x[0] * (x[2] - x[0]) * y[1] + x[0] * x[1] * (x[0] - x[1]) * y[2]) / denom
    x_top = -B / (2*A)
    y_top = C - B*B / (4*A)
    
    return x_top, y_top


def normalize_bfp(periods, logbf, n, period_range=[1, 1e100]):
    """
    Correct the baseline model of the periodogram.

    logbf, list: log BF value of the BFP,

    n, int: number of observations.

    Returns:
        logbf, list: corrected log BF values.
    """
    mask = (periods >= period_range[0]) & (periods <= period_range[1])
    offset = np.min(logbf[mask]) + np.log(n)

    return logbf - offset
