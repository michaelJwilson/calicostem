import pytest
import numpy as np
import core
import scipy
import line_profiler
from concurrent.futures import ThreadPoolExecutor

NUM_THREADS = 8

def compute_nbinom_pmf_chunk(args):
    data_chunk, n, p = args
    return scipy.stats.nbinom.logpmf(data_chunk, n, p)

def compute_betabinom_pmf_chunk(args):
    data_chunk, n, a, b = args
    return scipy.stats.betabinom.logpmf(data_chunk, n, a, b)

def thread_nbinom(k, n, p, num_threads=NUM_THREADS, executor=None):
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=NUM_THREADS)
    
    k_chunks = np.array_split(k, num_threads)
    n_chunks = np.array_split(n, num_threads)
    p_chunks = np.array_split(p, num_threads)

    args = (xx for xx in zip(k_chunks, n_chunks, p_chunks))
    results = executor.map(compute_nbinom_pmf_chunk, args)

    return np.concatenate(list(results))

def thread_betabinom(k, n, a, b, num_threads=NUM_THREADS, executor=None):
    if executor	is None:
        executor = ThreadPoolExecutor(max_workers=NUM_THREADS)
    
    k_chunks = np.array_split(k, num_threads)
    n_chunks = np.array_split(n, num_threads)
    a_chunks = np.array_split(a, num_threads)
    b_chunks = np.array_split(b, num_threads)

    args = (xx for xx in zip(k_chunks, n_chunks, a_chunks, b_chunks))
    results = executor.map(compute_betabinom_pmf_chunk, args)

    return np.concatenate(list(results))

def get_mock_data():
    NN = 1_000_000
    
    ns = 100 + np.arange(NN)
    ps = 0.5 * np.ones_like(ns)
    ks = 10 * np.ones_like(ns)

    aa = np.ones_like(ns, dtype=float)
    bb = np.ones_like(ns, dtype=float)
    
    exp = scipy.stats.nbinom.logpmf(ks, ns, ps)
    exp_bb = scipy.stats.betabinom.logpmf(ks, ns, aa, bb)
    
    return ks, ns, ps, aa, bb, exp, exp_bb

@pytest.fixture
def mock_data():
    return get_mock_data()

def test_exp(mock_data, benchmark):
    ks, ns, ps, aa, bb, exp, exp_bb = mock_data

    def wrap_exp():
        return scipy.stats.nbinom.logpmf(ks, ns, ps)

    benchmark.group = "nb"
    result = benchmark(wrap_exp)

    assert np.allclose(exp, result)

def test_exp_bb(mock_data, benchmark):
    ks, ns, ps, aa, bb, exp, exp_bb = mock_data

    def wrap_exp():
        return scipy.stats.betabinom.logpmf(ks, ns, aa, bb)

    benchmark.group = "bb"
    result = benchmark(wrap_exp)

    assert np.allclose(exp_bb, result)
    
def test_rust(mock_data, benchmark):
    ks, ns, ps, aa, bb, exp, exp_bb = mock_data
    ks = ks.astype(float)
    ns = ns.astype(float)
    
    def wrap_rust():
        return core.nb(ks, ns, ps)

    benchmark.group = "nb"
    rust_result = benchmark(wrap_rust)
    
    assert np.allclose(exp, rust_result)

def test_rust_bb(mock_data, benchmark):
    ks, ns, ps, aa, bb, exp, exp_bb = mock_data
    ks = ks.astype(float)
    ns = ns.astype(float)

    def wrap_rust():
        return core.bb(ks, ns, aa, bb)

    benchmark.group = "bb"
    rust_result = benchmark(wrap_rust)

    assert np.allclose(exp_bb, rust_result)
    
def test_thread(mock_data, benchmark):
    ks, ns, ps, aa, bb, exp, exp_bb = mock_data

    executor = ThreadPoolExecutor(max_workers=NUM_THREADS)
    
    def wrap_thread():
        return thread_nbinom(ks, ns, ps, executor=executor)

    benchmark.group = "nb"
    thread_result = benchmark(wrap_thread)

    assert np.allclose(exp, thread_result)

def test_thread_bb(mock_data, benchmark):
    ks, ns, ps, aa, bb, exp, exp_bb = mock_data

    executor = ThreadPoolExecutor(max_workers=NUM_THREADS)
    
    def wrap_thread():
        return thread_betabinom(ks, ns, aa, bb, executor=executor)

    benchmark.group = "bb"
    result = benchmark(wrap_thread)

    assert np.allclose(exp_bb, result)
    
@line_profiler.profile
def profile(ks, ns, ps, aa, bb, exp, exp_bb, iterations=100):
    ks = ks.astype(float)
    ns = ns.astype(float)
    
    for _ in range(iterations):
        exp = scipy.stats.nbinom.logpmf(ks, ns, ps)
        rust_result = core.nb(ks, ns, ps)
        thread_result = thread_nbinom(ks, ns, ps)

        exp_bb = scipy.stats.betabinom.logpmf(ks, ns, aa, bb)
        rust_bb = core.bb(ks, ns, aa, bb)
        
    assert np.allclose(exp, thread_result)
    assert np.allclose(exp, rust_result)

    assert np.allclose(exp_bb, rust_bb)

    print(exp[:5])
    print(rust_result[:5])
    
    print(exp_bb[:5])
    print(rust_bb[:5])
    
    print("Profiling complete.")

    
if __name__ == "__main__":
    mock_data = get_mock_data()
    ks, ns, ps, aa, bb, exp, exp_bb = mock_data
    
    profile(*mock_data)
    
