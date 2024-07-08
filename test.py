import numpy as np
import core
import scipy

@profile
def main():
    NN = 1_000_000

    ns = 1 + np.arange(NN)
    ps = 0.5 * np.ones_like(ns)
    ks = 10. * np.ones_like(ns, dtype=float)

    exp = scipy.stats.nbinom.pmf(ks, ns, ps)
    result = core.nb(ns, ps)

    print(exp[:5])
    print(result[:5])
    
    assert np.allclose(exp, result)

    
if __name__ == "__main__":
    main()
