import numpy as np
import os
from multiprocessing import Pool

from .utilities import getParabolaVertex, normalize_bfp


def find_peak(x, y, xmin, xmax, fit=None):
    # find a maximum of a range
    mask = (x > xmin) & (x < xmax)
    xbit, ybit = x[mask], y[mask]
    i_max  = np.argmax(ybit)

    # if the peak is at the edge, remove the edge pixels
    while i_max == 0 or i_max == len(ybit)-1:
        xbit = xbit[1:-1]
        ybit = ybit[1:-1]
        i_max  = np.argmax(ybit)

    if fit == "parabola": # fit a parabola to a tallest peak, find the peak
        return getParabolaVertex(xbit[i_max-1:i_max+2], ybit[i_max-1:i_max+2])
    else: # just return the tallest pixel
        return xbit[i_max], ybit[i_max]


def measure_peaks(periods, logbf, peaks=[], fit_peak="parabola", ignore_errors=True):
    """
    Measure peak heights from a periodogram.

    periods, list: list of periodogram periods,

    logbf, list: list of periodogram values,

    peaks, list: list of tuples for each peak to measure. Tuple contains peak period and a search radius around that value.

    fit_peak, str: periodogram peak fit type. "parabola" fits parabola to top 3 points, otherwise just return the highest pixel value.

    ignore_errors, bool: raise the errors or ignore them (print only)
    """
    measurements = []

    # first find the highest peak
    highest_i = np.argmax(logbf)
    highest_period, highest_logbf = find_peak(
        x    = periods, 
        y    = logbf,
        xmin = periods[highest_i] / 1.1,
        xmax = periods[highest_i] * 1.1,
        fit  = fit_peak
    )
    measurements.append((highest_period, highest_logbf))

    for peak in peaks:
        try:
            peak_period, peak_logbf = find_peak(
                x    = periods, 
                y    = logbf,
                xmin = peak[0] - peak[1],
                xmax = peak[0] + peak[1],
                fit  = fit_peak
            )
            measurements.append((peak_period, peak_logbf))
        except Exception as e:
            if ignore_errors:
                print(f"Failed to measure peak {peak[0]} +/- {peak[1]}. Error:\n{e}")
            else:
                raise

    return measurements


def measure_bfp(path, peaks, nobs, fit_peak="parabola"):
    bfp    = np.genfromtxt(path, skip_header=1, delimiter=",").T
    bfp[1] = normalize_bfp(periods=bfp[0], logbf=bfp[1], n=nobs)
    return measure_peaks(bfp[0], bfp[1], peaks, fit_peak)


def measure_bfps(folder, saveto=None, prefix="", indices=np.arange(0, 25000), nobs=None, peaks1=[], peaks2=[], ncores=1):
    """
    Measure all   monte carlo bfps in a folder.

    folder, str:  folder containing "rv/" and "bfp/" subfolders containing the monte carlo output files.

    saveto, str:  path to the output csv file.

    prefix, str:  filename prefix, using format f"{prefix}{index}.txt" for RV file, for instance.

    indices, int: monte carlo indices (default 0 to 25000).

    nobs, int:    number of RV observations, used for BFP normalization (reads the RV files by default, slower).

    peaks1, list: peaks to measure from *-bfp1.txt files (BFPs of the original data set, see measure_peaks()).

    peaks2, list: peaks to measure from *-bfp2.txt files (BFPs of the BFP 1 residuals, see measure_peaks()).
    """
    if saveto is None:
        saveto = f"{prefix}-measurements.txt"

    # construct a header
    header1 = "\t".join([f"bfp1_P{int(x[0])}_period\tbfp1_P{int(x[0])}_logbf" for x in peaks1])
    header2 = "\t".join([f"bfp2_P{int(x[0])}_period\tbfp2_P{int(x[0])}_logbf" for x in peaks2])
    header = f"index\tbfp1_highest_period\tbfp1_highest_logbf\t{header1}\tbfp2_highest_period\tbfp2_highest_logbf\t{header2}\n"

    with open(saveto, "w+") as csv:
        csv.write(header)

    if ncores > 1:
        Pool(ncores).starmap(
            measure_and_write_bfp,
            [(saveto, nobs, folder, prefix, index, peaks1, peaks2) for index in indices]
        )
    else:
        for index in indices:
            measure_and_write_bfp(saveto, nobs, folder, prefix, index, peaks1, peaks2)
    
    print("Done.                       ")


def measure_and_write_bfp(saveto, nobs, folder, prefix, index, peaks1, peaks2):
    print(f"Processing index #{index}   ", end='\r')

    if nobs is None:
        nobs = len(np.genfromtxt(os.path.join(folder, "rv", f"{prefix}-{index}.txt"), skip_header=1, delimiter="\t").T[0])

    line = f"{index}\t"

    measured_peaks1 = measure_bfp(
        path  = os.path.join(folder, "bfp", f"{prefix}-{index}-bfp1.txt"),
        peaks = peaks1,
        nobs  = nobs
    )
    line += "\t".join([f"{x[0]}\t{x[1]}" for x in measured_peaks1])
    line += "\t"

    measured_peaks2 = measure_bfp(
        path  = os.path.join(folder, "bfp", f"{prefix}-{index}-bfp2.txt"),
        peaks = peaks2,
        nobs  = nobs
    )
    line += "\t".join([f"{x[0]}\t{x[1]}" for x in measured_peaks2])
    line += "\n"

    if measured_peaks2[1][1] < 2:
        print(measured_peaks2[1][1])
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(2, 1)

        path   = os.path.join(folder, "bfp", f"{prefix}-{index}-bfp1.txt")
        bfp    = np.genfromtxt(path, skip_header=1, delimiter=",").T
        bfp[1] = normalize_bfp(periods=bfp[0], logbf=bfp[1], n=nobs)
        ax[0].plot(bfp[0], bfp[1], "k-")

        path   = os.path.join(folder, "bfp", f"{prefix}-{index}-bfp2.txt")
        bfp    = np.genfromtxt(path, skip_header=1, delimiter=",").T
        bfp[1] = normalize_bfp(periods=bfp[0], logbf=bfp[1], n=nobs)
        ax[1].plot(bfp[0], bfp[1], "k-")

        ax[0].set_xscale("log")
        ax[1].set_xscale("log")

        plt.show()

    with open(saveto, "a") as csv:
        csv.write(line)
