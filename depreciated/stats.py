import numpy as np
from scipy.stats import stats


def perm_test_1samp(data1, data2, nperm=5000):
    # Step 1: Calculate the observed t-statistic
    observed_diff = data1 - data2
    observed_t_stat, pval = stats.ttest_rel(data1, data2)

    # Step 2: Generate permutation samples
    perm_t_stats = np.zeros(nperm)
    for i in range(nperm):
        # Randomly flip signs of the differences
        signs = np.random.choice([-1, 1], size=len(observed_diff))
        perm_diff = signs * observed_diff
        perm_t_stat, _ = stats.ttest_1samp(perm_diff, 0)
        perm_t_stats[i] = perm_t_stat

    # Step 3: Calculate the p-value
    pval_perm = np.sum(np.abs(perm_t_stats) >= np.abs(observed_t_stat)) / nperm

    return pval, pval_perm