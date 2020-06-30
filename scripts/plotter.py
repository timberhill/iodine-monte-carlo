import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from .utilities import normalize_bfp, getParabolaVertex

colour_whole = "#9ba5cc"
colour_partial  = "#111d4a"
colour_mc     = "#cc3363"


def poisson_uncertainty(arr):
    return [np.sqrt(x) if x >= 0 else 0 for x in arr]


def tick_log_formatter(x, pos):
    """
    Tick formatter for the log axes.

    Example: https://matplotlib.org/3.2.1/gallery/ticks_and_spines/custom_ticker1.html#sphx-glr-gallery-ticks-and-spines-custom-ticker1-py
    """
    if str(x)[0] in ["1","2","3","6"]:
        return f"{x:.0f}"
    else:
        return ""


def add_periodogram_panel(ax, path, period_range=[0, 1e100], nobs=None, rotation=None, color="k", label="", peak_label_ha="right"):
    if nobs is None:
        rv_path = path.replace("/bfp", "/rv").replace("-bfp1", "").replace("-bfp2", "")
        nobs = len(np.genfromtxt(rv_path, skip_header=1, delimiter="\t").T[0])

    ax.set_xscale("log")

    # read the files
    bfp     = np.genfromtxt(path, skip_header=1, delimiter=",").T
    periods = bfp[0]
    logbf   = normalize_bfp(bfp[0], bfp[1], n=nobs, period_range=[0,500])

    # make full periodogram measurements
    highest_index = np.argmax(logbf)
    highest_peak  = (periods[highest_index], logbf[highest_index])

    # cut out the period range
    mask = (periods >= period_range[0]) & (periods <= period_range[1])
    periods = periods[mask]
    logbf   =   logbf[mask]

    # plot rotation markers
    if rotation is not None and rotation >= period_range[0] and rotation <= period_range[1]:
        ax.axvline(rotation/1, color="#EB98CD", lw=1, ls="-",  label=r"$P_{rot}$ = " + str(rotation) + r" $d$")
        ax.axvline(rotation/2, color="#EB98CD", lw=1, ls="--", label=r"$P_{rot} / 2$")
        ax.axvline(rotation/3, color="#EB98CD", lw=1, ls=":",  label=r"$P_{rot} / 3$")
        ax.axvline(rotation*2, color="#EB98CD", lw=1, ls="--", label=r"$P_{rot} \times 2$")
        ax.axvline(rotation*3, color="#EB98CD", lw=1, ls=":",  label=r"$P_{rot} \times 3$")
    # plot logbf = 5
    ax.axhline(5, color="k", lw=1, ls="--", alpha=0.4)

    # handle reverse
    if periods[0] < periods[-1]:
        periods = list(reversed(periods))
        logbf   = list(reversed(logbf))
    
    # PLOT
    ax.plot(periods, logbf, c=color, lw=2, label=label)

    # mark the peak if within range
    if highest_peak[0] >= period_range[0] and highest_peak[0] <= period_range[1]:
        label = f"{highest_peak[0]:.1f} d →" if peak_label_ha=="right" else f"← {highest_peak[0]:.1f} d"
        ax.text(highest_peak[0], highest_peak[1], label, color=color, va="center", ha=peak_label_ha)
    
    ax.set_xlim(periods[-1], periods[0])

    # change tick label format - number instead of 10^n
    ax.xaxis.set_major_formatter(FuncFormatter(tick_log_formatter))
    ax.xaxis.set_minor_formatter(FuncFormatter(tick_log_formatter))

    return ax


def plot_periodograms(hd, rotation=None, nobs=None, range1=[0, 1e100], range2=[0, 1e100], ylim=None):
    f  = plt.figure("BFPs", figsize=(17, 8))
    ax = f.subplots(2, 4, sharex=False, sharey=False)

    # plot original data set bfps on every panel
    for col in range(4):
        for row in range(2):
            ax[row, col] = add_periodogram_panel(ax[row, col],
                path=f"../data/HD{hd}/selection/bfp/hd{hd}-bfp{row+1}.txt",
                period_range=range1 if row == 0 else range2,
                nobs=nobs,
                rotation=rotation,
                color=colour_whole,
                label="Original RVs" if row == 0 else "Original RVs (residuals)",
                peak_label_ha="right"
            )

        
    tags   = ["", "_noact", "_notell", "_noact_notell"]
    labels = ["Whole spectrum", "Active lines removed", "Telluric lines removed", "Active & telluric lines removed"]
        
    for col in [1,2,3]:
        for row in range(2):
            ax[row, col] = add_periodogram_panel(ax[row, col],
                path=f"../data/HD{hd}/selection/bfp/hd{hd}{tags[col]}-bfp{row+1}.txt",
                period_range=range1 if row == 0 else range2,
                nobs=nobs,
                rotation=rotation,
                color=colour_partial,
                label=labels[col] if row == 0 else labels[col] + " (residuals)",
                peak_label_ha="left"
            )

    title_fontsize = 10
    for col in range(4):
        ax[0, col].set_title(labels[col], fontsize=title_fontsize)
    
    # get the y limits, same for a whole row

    if ylim is None:
        ymax0 = max([ax[0, i].get_ylim()[1] for i in range(4)])
        ymax1 = max([ax[1, i].get_ylim()[1] for i in range(4)])
        for i in range(4):
            ax[0, i].set_ylim(-5, ymax0)
            ax[1, i].set_ylim(-5, ymax1)
    else:
        for col in range(4):
            for row in range(2):
                ax[row, col].set_ylim(*ylim)
    
    ax[1, 0].set_title("residuals:", loc="left")
    
    return f, ax


def plot_monte_carlo(subsets, labels, bins, whole_logbf=None, partial_logbf=None, mask=None, ylim=None):
    n = len(subsets)
    percentile = 99.7

    f, ax = plt.subplots(1, n, figsize=(12, 5), sharey=True)

    for i in range(n):
        subset = subsets[i] if mask is None else subsets[i][mask]

        y, x = np.histogram(subset, bins=bins[i])
        x = 0.5 * (x[1:] + x[:-1])
        y[y == 0] = -1e5

        ax[i].hist(subset, bins=bins[i], color=colour_mc, alpha=0.2)
        ax[i].step(x, y, where="mid", label="Monte Carlo sample", color=colour_mc, alpha=0.8, lw=2)
        ax[i].errorbar(x, y, yerr=poisson_uncertainty(y), color=colour_mc, alpha=0.8, fmt="none")

        ax[i].axvline(np.median(subset), lw=1, color=colour_mc, ls="--", label="sample median")
        ax[i].axvline(np.percentile(subset, percentile), lw=1, color=colour_mc, ls=":",  label=f"{percentile} percentile")
        ax[i].axvline(np.percentile(subset, percentile), lw=1, color=colour_mc, ls=":")

        if whole_logbf is not None:
            ax[i].axvline(whole_logbf[i], color=colour_whole,   ls="-", lw=2, label="whole spectrum")
        
        if partial_logbf is not None:
            ax[i].axvline(partial_logbf[i], color=colour_partial, ls="-", lw=2, label="partial spectrum")

        ax[i].set_xlabel(labels[i])

    ax[0].set_ylabel("N")
    if ylim is None:
        ax[0].set_ylim(0, ax[0].get_ylim()[1])
    else:
        ax[0].set_ylim(*ylim)

    plt.legend()
