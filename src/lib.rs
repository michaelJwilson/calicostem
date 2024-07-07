extern crate statrs;

use statrs::distribution::{NegativeBinomial, Discrete};
use ndarray::Array;
use numpy::{PyArray2,IntoPyArray};
use pyo3::prelude::*;
use numpy::{PyReadonlyArray1};
use rayon::prelude::*;

#[pyfunction]
fn nb(n: PyReadonlyArray1<'_, i64>, p: PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    let n = n.to_vec()?;
    let p = p.to_vec()?;

    /*
    for (&n_i, &p_i) in n.iter().zip(p.iter()) {
        let neg_binom = NegativeBinomial::new(n_i as f64, p_i).unwrap();
        let k = 10;
        let prob = neg_binom.pmf(k);
        probs.push(prob);
    }
    */

    let probs: Vec<f64> = n.par_iter().zip(p.par_iter())
        .map(|(&n_i, &p_i)| {
            let neg_binom = NegativeBinomial::new(n_i as f64, p_i).unwrap();
            let k = 10; // Example usage, adjust as needed
            neg_binom.pmf(k)
        })
        .collect();

    Ok(probs)
}


#[pymodule]
#[pyo3(name="core")]
fn core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nb, m)?)?;
    Ok(())
}