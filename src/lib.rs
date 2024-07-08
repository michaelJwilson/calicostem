extern crate statrs;

use statrs::{distribution::{Discrete, NegativeBinomial}};
use pyo3::prelude::*;
use numpy::{PyReadonlyArray1};
use rayon::prelude::*;
use statrs::function::gamma::ln_gamma;

#[pyfunction]
fn nb(k: PyReadonlyArray1<'_, i64>, n: PyReadonlyArray1<'_, i64>, p: PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    let k = k.to_vec()?;
    let n = n.to_vec()?;
    let p = p.to_vec()?;

    // let mut probs: Vec<f64> = Vec::new();
    /*
    (&k, &n, &p).into_par_iter()
        .map(|(k_i, n_i, p_i)| {
            ln_gamma((k_i + n_i) as f64) - ln_gamma((*n_i) as f64) - ln_gamma((*k_i) as f64);
        }).collect_into_vec(&mut probs);
    */
    
    let probs: Vec<f64> = n.par_iter().zip(k.par_iter()).zip(p.par_iter())
        .map(|((&n_i, &k_i), &p_i)| {
            let neg_binom: NegativeBinomial = NegativeBinomial::new(n_i as f64, p_i).unwrap();
            neg_binom.ln_pmf(k_i.try_into().unwrap())

            // let interim = ln_gamma((k_i + n_i) as f64) - ln_gamma((n_i) as f64) - ln_gamma((k_i) as f64);
            // let result = interim + (n_i as f64) * (p_i.ln()) + (k_i as f64) * (1.0 - p_i).ln();
            // result
        })
        .collect();
    
    Ok(probs)
}
/*
#[pyfunction]
fn bb(k: PyReadonlyArray1<'_, i64>, n: PyReadonlyArray1<'_, i64>, alphas: PyReadonlyArray1<'_, f64>, betas: PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    let n = n.to_vec()?;
    let k = k.to_vec()?;
    let alphas = alphas.to_vec()?;
    let betas = betas.to_vec()?;
    
    let probs: Vec<f64> = n.par_iter().zip(k.par_iter()).zip(alphas.par_iter()).zip(betas.par_iter())
        .map(|(((&n_i, &k_i), &a_i), &b_i)| {
            let ln_binom_coeff = ln_gamma((n_i + 1) as f64) - ln_gamma((k_i + 1) as f64) - ln_gamma((n_i - k_i + 1) as f64);
            let ln_beta_k_alpha = ln_gamma(k_i as f64 + a_i) + ln_gamma((n_i - k_i) as f64 + b_i) - ln_gamma(n_i as f64 + a_i + b_i);
            let ln_beta_alpha_beta = ln_gamma(a_i + b_i) - ln_gamma(a_i) - ln_gamma(b_i);
            
            ln_binom_coeff + ln_beta_k_alpha - ln_beta_alpha_beta
        })
        .collect();

    Ok(probs)
}
*/

#[pymodule]
#[pyo3(name="core")]
fn core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nb, m)?)?;
    // m.add_function(wrap_pyfunction!(bb, m)?)?;
    Ok(())
}