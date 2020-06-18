import numpy as np
import matplotlib.pyplot as plt
from .utilities import normalize_bfp

colour_before = "#9ba5cc"
colour_after  = "#111d4a"
colour_mc     = "#cc3363"


def add_periodogram_panel(ax, path, period_range=[0, 1e100], nobs=None, rotation=None, color="k", label="", peak_label_ha="right"):
    if nobs is None:
        rv_path = path.replace("/bfp", "/rv").replace("-bfp1", "").replace("-bfp2", "")
        nobs = len(np.genfromtxt(rv_path, skip_header=1, delimiter="\t").T[0])

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

    # plot
    if rotation is not None and rotation >= period_range[0] and rotation <= period_range[1]:
        ax.axvline(rotation/1, color="#e574bc", lw=1, ls="-",  label=r"$P_{rot}$ = " + str(rotation) + r" $d$")
        ax.axvline(rotation/2, color="#e574bc", lw=1, ls="--", label=r"$P_{rot} / 2$")
        ax.axvline(rotation/3, color="#e574bc", lw=1, ls=":",  label=r"$P_{rot} / 3$")
        ax.axvline(rotation*2, color="#e574bc", lw=1, ls="--", label=r"$P_{rot} \times 2$")
        ax.axvline(rotation*3, color="#e574bc", lw=1, ls=":",  label=r"$P_{rot} \times 3$")

    ax.plot(periods, logbf, c=color, lw=2, label=label)
    ax.set_xscale("log")

    # mark the peak if within range
    if highest_peak[0] >= period_range[0] and highest_peak[0] <= period_range[1]:
        label = f"{highest_peak[0]:.1f} d →" if peak_label_ha=="right" else f"← {highest_peak[0]:.1f} d"
        ax.text(highest_peak[0], highest_peak[1], label, color=color, va="center", ha=peak_label_ha)
    
    ax.set_xlim(periods[-1], periods[0])

    return ax


def plot_periodograms(hd, rotation=None, nobs=None, range1=[0, 1e100], range2=[0, 1e100]):    
    f  = plt.figure("BFPs", figsize=(17, 9))
    ax = f.subplots(2, 4, sharex=False, sharey=True)

    # plot original data set bfps on every panel
    for col in range(4):
        for row in range(2):
            ax[row, col] = add_periodogram_panel(ax[row, col],
                path=f"../data/HD{hd}/selection/bfp/hd{hd}-bfp{row+1}.txt",
                period_range=range1 if row == 0 else range2,
                nobs=nobs,
                rotation=rotation,
                color=colour_before,
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
                color=colour_after,
                label=labels[col] if row == 0 else labels[col] + " (residuals)",
                peak_label_ha="left"
            )

    title_fontsize = 10
    for col in range(4):
        ax[0, col].set_title(labels[col], fontsize=title_fontsize)
        
    return f, ax
