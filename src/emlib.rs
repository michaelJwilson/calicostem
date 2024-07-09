extern crate statrs;

use ndarray::parallel::par_azip;
use ndarray::Array1;
use numpy::{PyReadonlyArray1, ToPyArray, PyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use statrs::function::gamma::ln_gamma;

// NB export RAYON_NUM_THREADS=12

#[pyfunction]
fn nb(
    k: PyReadonlyArray1<'_, f64>,
    n: PyReadonlyArray1<'_, f64>,
    p: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {

    let k = k.to_vec()?;
    let n = n.to_vec()?;
    let p = p.to_vec()?;

    let probs: Vec<f64> = n
        .par_iter()
        .zip(k.par_iter())
        .zip(p.par_iter())
        .map(|((&n_i, &k_i), &p_i)| {
            ln_gamma(k_i + n_i) - ln_gamma(k_i + 1.) - ln_gamma(n_i)
                + k_i * (1. - p_i).ln()
                + n_i * p_i.ln()
        })
        .collect();

    Ok(probs)
}

#[pyfunction]
fn bb(
    k: PyReadonlyArray1<'_, f64>,
    n: PyReadonlyArray1<'_, f64>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    let k = k.to_vec()?;
    let n = n.to_vec()?;
    let a = a.to_vec()?;
    let b = b.to_vec()?;

    let probs: Vec<f64> = n
        .par_iter()
        .zip(k.par_iter())
        .zip(a.par_iter())
        .zip(b.par_iter())
        .map(|(((&n_i, &k_i), &a_i), &b_i)| {
            let result = ln_gamma(n_i + 1.) - ln_gamma(k_i + 1.) - ln_gamma(n_i - k_i + 1.)
                + ln_gamma(k_i + a_i)
                + ln_gamma(n_i - k_i + b_i)
                - ln_gamma(n_i + a_i + b_i)
                + ln_gamma(a_i + b_i)
                - ln_gamma(a_i)
                - ln_gamma(b_i);
            result
        })
        .collect();

    Ok(probs)
}

#[pymodule]
#[pyo3(name = "core")]
fn core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nb, m)?)?;
    m.add_function(wrap_pyfunction!(bb, m)?)?;
    Ok(())
}
