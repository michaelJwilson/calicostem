import pytest
import numpy as np
import core
import scipy
import line_profiler
from concurrent.futures import ThreadPoolExecutor

NUM_THREADS = 4

def compute_nbinom_pmf_chunk(args):
    data_chunk, n, p = args
    return scipy.stats.nbinom.logpmf(data_chunk, n, p)

def thread_nbinom(data, n, p, num_threads=NUM_THREADS):
    executor = ThreadPoolExecutor(max_workers=NUM_THREADS)
    
    data_chunks = np.array_split(data, num_threads)
    n_chunks = np.array_split(n, num_threads)
    p_chunks = np.array_split(p, num_threads)

    args = [xx for xx in zip(data_chunks, n_chunks, p_chunks)]
    results = executor.map(compute_nbinom_pmf_chunk, args)

    return np.concatenate(list(results))

def get_mock_data():
    # NN = 10
    NN = 500_000
    # NN = 213_157_056
    
    # NB 1 -> 100.                                                                                                                                                                  
    ns = 100 + np.arange(NN)
    ps = 0.5 * np.ones_like(ns)
    ks = 10 * np.ones_like(ns)

    exp = scipy.stats.nbinom.logpmf(ks, ns, ps)
    
    return ks, ns, ps, exp

@pytest.fixture
def mock_data():
    return get_mock_data()

def test_exp(mock_data, benchmark):
    ks, ns, ps, exp = mock_data

    def wrap_exp():
        return scipy.stats.nbinom.logpmf(ks, ns, ps)

    exp2 = benchmark(wrap_exp)

    assert np.allclose(exp, exp2)
    
def test_rust(mock_data, benchmark):
    ks, ns, ps, exp = mock_data

    def wrap_rust():
        return core.nb(ks, ns, ps)
        
    rust_result = benchmark(wrap_rust)

    # print(exp[:5])
    # print(rust_result[:5])
    
    assert np.allclose(exp, rust_result)

def test_thread(mock_data, benchmark):
    ks, ns, ps, exp = mock_data

    def wrap_thread():
        return thread_nbinom(ks, ns, ps)
    
    thread_result = benchmark(wrap_thread)

    assert np.allclose(exp, thread_result)
    
@line_profiler.profile
def profile(ks, ns, ps, exp, iterations=100):

    for _ in range(iterations):
        exp = scipy.stats.nbinom.logpmf(ks, ns, ps)
        rust_result = core.nb(ks, ns, ps)
        thread_result = thread_nbinom(ks, ns, ps)

    assert np.allclose(exp, thread_result)
    assert np.allclose(exp, rust_result)

    print("Profiling complete.")
    
    """
    # NB see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
    exp = scipy.stats.betabinom.logpmf(ks, ns, 10., 20.)

    alphas = 10. * np.ones_like(ks, dtype=float)
    betas = 20. * np.ones_like(ks, dtype=float)

    result = core.bb(ks, ns, alphas, betas)

    print(exp[:5])
    print(result[:5])

    # assert np.allclose(exp, result)
    """

    
if __name__ == "__main__":
    mock_data = get_mock_data()
    
    profile(*mock_data)
