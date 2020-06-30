# import sys, os
# sys.path.append(os.path.join(os.path.abspath(''), ".."))
import numpy as np
from measure_peaks import measure_bfps, measure_bfp
from plotter import plot_periodograms, plot_monte_carlo
import matplotlib.pyplot as plt

# from the original BFP
peaks1 = [(2600, 1000)]

# from the BFP of the residuals
peaks2 = [(11.8, 1)]

### Number of RV mesurements in the data set
nobs = 104

target = ".measurements-test.txt"
source = "../data/HD22049/monte-carlo/"

# measure_bfps(
#     folder = source,
#     saveto = target,
#     prefix = "hd22049",
#     nobs   = nobs,
#     peaks1 = peaks1,
#     peaks2 = peaks2
# )
# exit()

data      = np.genfromtxt(target, names=True, delimiter="\t")

main_mask = data["bfp2_P11_logbf"] < 2
d1 = data[main_mask]
d2 = data[~main_mask]

bins = np.arange(-1, 35, 0.1)
print(d1["bfp1_highest_period"])
plt.hist(d1["bfp1_P2600_logbf"], bins=bins)
plt.hist(d2["bfp1_P2600_logbf"], bins=bins)
plt.show()

# whole_p   = measure_bfp("../data/HD22049/selection/bfp/hd22049-bfp1.txt", peaks1, nobs)
# partial_p = measure_bfp("../data/HD22049/selection/bfp/hd22049_noact_notell-bfp1.txt", peaks1, nobs)
# whole_r   = measure_bfp("../data/HD22049/selection/bfp/hd22049-bfp2.txt", peaks2, nobs)
# partial_r = measure_bfp("../data/HD22049/selection/bfp/hd22049_noact_notell-bfp2.txt", peaks2, nobs)

# plot_monte_carlo(
#     subsets = [data["bfp1_P2600_logbf"], data["bfp2_P11_logbf"]],
#     labels  = ["Eps Eri b", "Eps Eri rotation"],
#     bins    = [np.arange(9, 15, 0.2), np.arange(-10, 10, 0.2)],
#     whole_logbf   = [  whole_p[1][1],   whole_r[1][1]],
#     partial_logbf = [partial_p[1][1], partial_r[1][1]],
# )
# plt.show()