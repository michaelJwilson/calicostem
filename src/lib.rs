extern crate statrs;

use statrs::{distribution::{Beta, Discrete, NegativeBinomial}, function::beta};
use crate::statrs::distribution::Continuous;
use ndarray::Array;
use numpy::{PyArray2,IntoPyArray};
use pyo3::prelude::*;
use numpy::{PyReadonlyArray1};
use rayon::prelude::*;
use statrs::function::gamma::ln_gamma;

#[pyfunction]
fn nb(n: PyReadonlyArray1<'_, i64>, p: PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    let n = n.to_vec()?;
    let p = p.to_vec()?;

    let probs: Vec<f64> = n.par_iter().zip(p.par_iter())
        .map(|(&n_i, &p_i)| {
            let neg_binom: NegativeBinomial = NegativeBinomial::new(n_i as f64, p_i).unwrap();
            let k: u64 = 10; // Example usage, adjust as needed
            neg_binom.pmf(k)
        })
        .collect();

    Ok(probs)
}

#[pyfunction]
fn beta_binomial(n: PyReadonlyArray1<'_, i64>, k: PyReadonlyArray1<'_, i64>, alphas: PyReadonlyArray1<'_, f64>, betas: PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    let n: Vec<i64> = n.to_vec()?;
    let k: Vec<i64> = k.to_vec()?;
    let alphas: Vec<f64> = alphas.to_vec()?;
    let betas: Vec<f64> = betas.to_vec()?;

    let probs: Vec<f64> = n.par_iter().zip(k.par_iter()).zip(alphas.par_iter()).zip(betas.par_iter())
        .map(|(((&n_i, &k_i), &a_i), &b_i)| {
            let ln_binom_coeff = ln_gamma((n_i + 1) as f64) - ln_gamma((k_i + 1) as f64) - ln_gamma((n_i - k_i + 1) as f64);
            let ln_beta_k_alpha = ln_gamma(k_i as f64 + a_i) + ln_gamma((n_i - k_i) as f64 + b_i) - ln_gamma(n_i as f64 + a_i + b_i);
            let ln_beta_alpha_beta = ln_gamma(a_i + b_i) - ln_gamma(a_i) - ln_gamma(b_i);

            let ln_pmf = ln_binom_coeff + ln_beta_k_alpha - ln_beta_alpha_beta;
                   
            ln_binom_coeff + ln_beta_k_alpha - ln_beta_alpha_beta
        })
        .collect();

    Ok(probs)
}

#[pymodule]
#[pyo3(name="core")]
fn core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nb, m)?)?;
    m.add_function(wrap_pyfunction!(beta_binomial, m)?)?;
    Ok(())
}