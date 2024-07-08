import numpy as np
import core
import scipy
import line_profiler

@line_profiler.profile
def main():
    NN = 100_000

    # NB 1 -> 100.
    ns = 100 + np.arange(NN)
    ps = 0.5 * np.ones_like(ns)
    ks = 10. * np.ones_like(ns, dtype=float)

    exp = scipy.stats.nbinom.pmf(ks, ns, ps)
    result = core.nb(ns, ps)

    # print(exp[:5])
    # print(result[:5])
    
    assert np.allclose(exp, result)

    # NB see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
    exp = scipy.stats.betabinom.logpmf(ks, ns, 1., 1.)

    alphas = np.ones_like(ks)
    betas = np.ones_like(ks)
    ks = ks.astype(int)

    result = core.beta_binomial(ns, ks, alphas, betas)

    # print(exp[:5])
    # print(result[:5])

    assert np.allclose(exp, result)

if __name__ == "__main__":
    main()
