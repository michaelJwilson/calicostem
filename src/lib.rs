use ndarray;
use ndarray::{Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArrayDyn,
};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

// See https://itnext.io/how-to-bind-python-numpy-with-rust-ndarray-2efa5717ed21
#[pymodule]
fn core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn max_min<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<f64>) -> &'py PyArray1<f64> {
        let array = x.as_array();
        let result_array = rust_fn::max_min(&array);

        result_array.into_pyarray(py)
    }

    #[pyfn(m)]
    fn nb<'py>(
        py: Python<'py>,
        ink: PyReadonlyArray1<f64>,
        inn: PyReadonlyArray1<f64>,
        inp: PyReadonlyArray1<f64>,
    ) -> &'py PyArray1<f64> {
        let k = ink.as_array();
        let n = inn.as_array();
        let p = inp.as_array();

        let shape = k.shape();
        let mut result = Array1::<f64>::zeros(shape[0]);
        let mut vresult = result.view_mut();

        rust_fn::nb(&mut vresult, &k, &n, &p);
        result.into_pyarray(py)
    }

    #[pyfn(m)]
    fn nb_moments<'py>(
        py: Python<'py>,
        k: PyReadonlyArray3<f64>,
        mu: PyReadonlyArray3<f64>,
        var: PyReadonlyArray3<f64>,
    ) -> &'py PyArray3<f64> {
        let k = k.as_array();
        let mu = mu.as_array();
        let var = var.as_array();

        let shape = k.shape();
        let mut result = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
        let mut view_result = result.view_mut();

        rust_fn::nb_moments(&mut view_result, &k, &mu, &var);
        result.into_pyarray(py)
    }

    #[pyfn(m)]
    fn compute_emission_probability_nb_mix<'py>(
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
        base_nb_mean: PyReadonlyArray2<f64>,
        tumor_prop: PyReadonlyArray2<f64>,
        log_mu: PyReadonlyArray2<f64>,
        alphas: PyReadonlyArray2<f64>,
    ) -> &'py PyArray3<f64> {
        let X = X.as_array();
        let base_nb_mean = base_nb_mean.as_array();
        let tumor_prop = tumor_prop.as_array();
        let log_mu = log_mu.as_array();
        let alphas = alphas.as_array();

        let n_obs = X.shape()[0];
        let n_spots = X.shape()[1];
        let n_states = log_mu.shape()[0];

        let mut result = Array3::<f64>::zeros((n_states, n_obs, n_spots));
        let mut view_result = result.view_mut();

        rust_fn::compute_emission_probability_nb_betabinom_mix(
            &mut view_result,
            &X,
            &base_nb_mean,
            &tumor_prop,
            &log_mu,
            &alphas,
        );

        result.into_pyarray(py)
    }

    #[pyfn(m)]
    fn compute_emission_probability_bb_mix<'py>(
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
        base_nb_mean: PyReadonlyArray2<f64>,
        total_bb_RD: PyReadonlyArray2<f64>,
        pbinom: PyReadonlyArray2<f64>,
        taus: PyReadonlyArray2<f64>,
        tumor_prop: PyReadonlyArray2<f64>,
    ) -> &'py PyArray3<f64> {
        let X = X.as_array();
        let base_nb_mean = base_nb_mean.as_array();
        let total_bb_RD = total_bb_RD.as_array();
        let pbinom = pbinom.as_array();
        let taus = taus.as_array();
        let tumor_prop = tumor_prop.as_array();

        let n_obs = X.shape()[0];
        let n_spots = X.shape()[1];
        let n_states = pbinom.shape()[0];

        let mut result = Array3::<f64>::zeros((n_states, n_obs, n_spots));
        let mut view_result = result.view_mut();

        rust_fn::compute_emission_probability_bb_betabinom_mix(
            &mut view_result,
            &X,
            &base_nb_mean,
            &total_bb_RD,
            &pbinom,
            &taus,
            &tumor_prop,
        );

        result.into_pyarray(py)
    }

    #[pyfn(m)]
    fn compute_emission_probability_bb_mix_weighted<'py>(
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
        base_nb_mean: PyReadonlyArray2<f64>,
        total_bb_RD: PyReadonlyArray2<f64>,
        pbinom: PyReadonlyArray2<f64>,
        taus: PyReadonlyArray2<f64>,
        tumor_prop: PyReadonlyArray2<f64>,
        sample_lengths: PyReadonlyArray1<f64>,
        log_mu: PyReadonlyArray2<f64>,
        log_mu_shift: PyReadonlyArray2<f64>,
    ) -> &'py PyArray3<f64> {
        let X = X.as_array();
        let base_nb_mean = base_nb_mean.as_array();
        let total_bb_RD = total_bb_RD.as_array();
        let pbinom = pbinom.as_array();
        let taus = taus.as_array();
        let tumor_prop = tumor_prop.as_array();
        let sample_lengths = sample_lengths.as_array();
        let log_mu = log_mu.as_array();
        let log_mu_shift = log_mu_shift.as_array();

        let n_obs = X.shape()[0];
        let n_spots = X.shape()[1];
        let n_states = pbinom.shape()[0];

        let mut result = Array3::<f64>::zeros((n_states, n_obs, n_spots));
        let mut view_result = result.view_mut();

        rust_fn::compute_emission_probability_bb_betabinom_mix_weighted(
            &mut view_result,
            &X,
            &base_nb_mean,
            &total_bb_RD,
            &pbinom,
            &taus,
            &tumor_prop,
            &sample_lengths,
            &log_mu,
            &log_mu_shift,
        );

        result.into_pyarray(py)
    }

    #[pyfn(m)]
    fn bb<'py>(
        py: Python<'py>,
        ink: PyReadonlyArray1<f64>,
        inn: PyReadonlyArray1<f64>,
        ina: PyReadonlyArray1<f64>,
        inb: PyReadonlyArray1<f64>,
    ) -> &'py PyArray1<f64> {
        let k = ink.as_array();
        let n = inn.as_array();
        let a = ina.as_array();
        let b = inb.as_array();

        let shape = k.shape();
        let mut result = Array1::<f64>::zeros(shape[0]);
        let mut vresult = result.view_mut();

        rust_fn::bb(&mut vresult, &k, &n, &a, &b);
        result.into_pyarray(py)
    }

    #[pyfn(m)]
    fn bbab<'py>(
        py: Python<'py>,
        ink: PyReadonlyArray1<f64>,
        inn: PyReadonlyArray1<f64>,
        ina: PyReadonlyArray1<f64>,
        inb: PyReadonlyArray1<f64>,
    ) -> &'py PyArray1<f64> {
        let k = ink.as_array();
        let n = inn.as_array();
        let a = ina.as_array();
        let b = inb.as_array();

        let shape = k.shape();
        let mut result = Array1::<f64>::zeros(shape[0]);
        let mut vresult = result.view_mut();

        rust_fn::bbab(&mut vresult, &k, &n, &a, &b);
        result.into_pyarray(py)
    }

    #[pyfn(m)]
    fn double_and_random_perturbation(
        _py: Python<'_>,
        x: &PyArrayDyn<f64>,
        perturbation_scaling: f64,
    ) {
        let mut array = unsafe { x.as_array_mut() };

        rust_fn::double_and_random_perturbation(&mut array, perturbation_scaling);
    }

    #[pyfn(m)]
    fn eye<'py>(py: Python<'py>, size: usize) -> &PyArray2<f64> {
        let array = ndarray::Array::eye(size);

        array.into_pyarray(py)
    }

    Ok(())
}

mod rust_fn {
    extern crate rayon;
    use rayon::prelude::*;

    use ndarray::{arr1, Array1, Array2, Array3, Zip};
    use numpy::ndarray::{
        ArrayView1, ArrayView2, ArrayView3, ArrayViewD, ArrayViewMut1, ArrayViewMut2,
        ArrayViewMut3, ArrayViewMutD,
    };
    use ordered_float::OrderedFloat;
    use rand::Rng;
    use rgsl::gamma_beta::beta::lnbeta;
    use rgsl::gamma_beta::factorials::{lnchoose, lnfact};
    use rgsl::gamma_beta::gamma::lngamma;
    use statrs::distribution::{Discrete, NegativeBinomial};
    use statrs::function::beta::ln_beta;
    use statrs::function::gamma::ln_gamma;

    pub fn double_and_random_perturbation(x: &mut ArrayViewMutD<'_, f64>, scaling: f64) {
        let mut rng = rand::thread_rng();

        x.iter_mut()
            .for_each(|x| *x = *x * 2. + (rng.gen::<f64>() - 0.5) * scaling);
    }

    pub fn nb(
        r: &mut ArrayViewMut1<'_, f64>,
        k: &ArrayView1<'_, f64>,
        n: &ArrayView1<'_, f64>,
        p: &ArrayView1<'_, f64>,
    ) {
        /*
        See:
            https://docs.rs/GSL/latest/rgsl/gamma_beta/factorials/fn.lnchoose.html
            https://mathworld.wolfram.com/NegativeBinomialDistribution.html
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
        */

        Zip::from(r)
            .and(k)
            .and(n)
            .and(p)
            .par_for_each(|r, &k, &n, &p| {
                /*
                See:
                  https://docs.rs/GSL/latest/rgsl/gamma_beta/factorials/fn.lnchoose.html
                  https://mathworld.wolfram.com/NegativeBinomialDistribution.html
                  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
                */

                let n32 = n as u32;
                let k32 = k as u32;

                *r = lnfact(k32 + n32 - 1) - lnfact(n32 - 1) - lnfact(k32)
                    + k * (1. - p).ln()
                    + n * p.ln();

                // *r = lngamma(k + n) - lngamma(k + 1.) - lngamma(n) + k * (1. - p).ln() + n * p.ln();

                // let dist = NegativeBinomial::new(n, p).unwrap();
                // *r = dist.ln_pmf(k as u64);
            });
    }

    pub fn nb_moments(
        r: &mut ArrayViewMut3<'_, f64>,
        k: &ArrayView3<'_, f64>,
        mu: &ArrayView3<'_, f64>,
        var: &ArrayView3<'_, f64>,
    ) {
        /*
        See:
            https://docs.rs/GSL/latest/rgsl/gamma_beta/factorials/fn.lnchoose.html
            https://mathworld.wolfram.com/NegativeBinomialDistribution.html
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
        */
        Zip::from(r)
            .and(k)
            .and(mu)
            .and(var)
            .par_for_each(|r, &k, &mu, &var| {
                /*
                See:
                  https://docs.rs/GSL/latest/rgsl/gamma_beta/factorials/fn.lnchoose.html
                  https://mathworld.wolfram.com/NegativeBinomialDistribution.html
                  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
                */

                let p = mu / var;
                let n = mu * p / (1. - p);

                let n32 = n as u32;
                let k32 = k as u32;

                *r = lnfact(k32 + n32 - 1) - lnfact(n32 - 1) - lnfact(k32)
                    + k * (1. - p).ln()
                    + n * p.ln();
            });
    }

    pub fn compute_emission_probability_nb_betabinom_mix(
        r: &mut ArrayViewMut3<'_, f64>,
        X: &ArrayView2<'_, f64>,
        base_nb_mean: &ArrayView2<'_, f64>,
        tumor_prop: &ArrayView2<'_, f64>,
        log_mu: &ArrayView2<'_, f64>,
        alphas: &ArrayView2<'_, f64>,
    ) {
        /*
        See:
            https://docs.rs/GSL/latest/rgsl/gamma_beta/factorials/fn.lnchoose.html
            https://mathworld.wolfram.com/NegativeBinomialDistribution.html
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
        */

        let n_obs = X.shape()[0];
        let n_spots = X.shape()[1];
        let n_states = log_mu.shape()[0];

        for segment in 0..n_obs {
            for spot in 0..n_spots {
                let base = base_nb_mean[[segment, spot]];

                if tumor_prop[[segment, spot]].is_nan() {
                    for state in 0..n_states {
                        r[[state, segment, spot]] = std::f64::NAN;
                    }

                    continue;
                }

                if base > 0. {
                    let shift = base * (1. - tumor_prop[[segment, spot]]);
                    let x = X[[segment, spot]];

                    // ln_gamma(x + 1.0)
                    let lnGx = lnfact(x as u32);

                    for state in 0..n_states {
                        let mu = log_mu[[state, spot]].exp();

                        let mean = shift + (mu * base * tumor_prop[[segment, spot]]);
                        let var = mean + alphas[[state, spot]] * mean.powf(2.);

                        let p = mean / var;
                        let n = mean * p / (1. - p);

                        r[[state, segment, spot]] = ln_gamma(n + x) - ln_gamma(n) - lnGx
                            + (n * p.ln())
                            + (x * (-p).ln_1p());
                    }
                }
            }
        }
    }

    pub fn compute_emission_probability_bb_betabinom_mix(
        r: &mut ArrayViewMut3<'_, f64>,
        X: &ArrayView2<'_, f64>,
        base_nb_mean: &ArrayView2<'_, f64>,
        total_bb_RD: &ArrayView2<'_, f64>,
        pbinom: &ArrayView2<'_, f64>,
        taus: &ArrayView2<'_, f64>,
        tumor_prop: &ArrayView2<'_, f64>,
    ) {
        let n_obs = X.shape()[0];
        let n_spots = X.shape()[1];
        let n_states = pbinom.shape()[0];

        for segment in 0..n_obs {
            for spot in 0..n_spots {
                if tumor_prop[[segment, spot]].is_nan() {
                    for state in 0..n_states {
                        r[[state, segment, spot]] = std::f64::NAN;
                    }

                    continue;
                }

                let rd = total_bb_RD[[segment, spot]];

                if rd > 0. {
                    let kk = X[[segment, spot]];
                    let nn = total_bb_RD[[segment, spot]];

                    let term_logn = -(nn + 1.).ln();
                    let term_beta = -lnbeta(nn - kk + 1., kk + 1.);

                    let shift = 0.5 * (1. - tumor_prop[[segment, spot]]);

                    for state in 0..n_states {
                        let tau = taus[[state, spot]];

                        let mix_pa = pbinom[[state, spot]] * tumor_prop[[segment, spot]] + shift;
                        let mix_pb = (1. - pbinom[[state, spot]]) * tumor_prop[[segment, spot]] + shift;
    
                        let aa = mix_pa * tau;
                        let bb = mix_pb * tau;

                        r[[state, segment, spot]] = term_logn + term_beta + lnbeta(kk + aa, nn - kk + bb) - lnbeta(aa, bb);
                    }
                }
            }
        }
    }

    fn get_segment_chunks(repeat_counts: &ArrayView1<'_, f64>) -> Vec<usize> {
        repeat_counts
            .iter()
            .enumerate() // Get both the index and the value
            .flat_map(|(index, &count)| {
                std::iter::repeat(index) // Repeat the index
                    .take(count.round() as usize) // Number of times to repeat, rounded to nearest whole number
            })
            .collect()
    }

    pub fn compute_emission_probability_bb_betabinom_mix_weighted(
        r: &mut ArrayViewMut3<'_, f64>,
        X: &ArrayView2<'_, f64>,
        base_nb_mean: &ArrayView2<'_, f64>,
        total_bb_RD: &ArrayView2<'_, f64>,
        pbinom: &ArrayView2<'_, f64>,
        taus: &ArrayView2<'_, f64>,
        tumor_prop: &ArrayView2<'_, f64>,
        sample_lengths: &ArrayView1<'_, f64>,
        log_mu: &ArrayView2<'_, f64>,
        log_mu_shift: &ArrayView2<'_, f64>,
    ) {
        let n_obs = X.shape()[0];
        let n_spots = X.shape()[1];
        let n_states = pbinom.shape()[0];

        let segment_chunks = get_segment_chunks(sample_lengths);

        assert segment_chunks.len() == n_obs;

        for segment in 0..n_obs {
            let segment_chunk = segment_chunks[segment];

            for spot in 0..n_spots {
                if tumor_prop[[segment, spot]].is_nan() {
                    for state in 0..n_states {
                        r[[state, segment, spot]] = std::f64::NAN;
                    }

                    continue;
                }

                let rd = total_bb_RD[[segment, spot]];

                if rd > 0. {
                    let kk = X[[segment, spot]];
                    let nn = total_bb_RD[[segment, spot]];

                    let term_logn = -(nn + 1.).ln();
                    let term_beta = -lnbeta(nn - kk + 1., kk + 1.);

                    let mu_norm = log_mu_shift[[segment_chunk, spot]].exp();

                    for state in 0..n_states {
                        let mu = log_mu[[state, spot]].exp();
                        let tau = taus[[state, spot]];

                        let weighted_tumor_prop = (tumor_prop[[segment, spot]] * mu) / (tumor_prop[[segment, spot]] * mu + 1. - tumor_prop[[segment, spot]]); 
                        let shift = 0.5 * (1. - weighted_tumor_prop);

                        let mix_pa = pbinom[[state, spot]] * weighted_tumor_prop + shift;
                        let mix_pb = (1. - pbinom[[state, spot]]) * weighted_tumor_prop + shift;
    
                        let aa = mix_pa * tau;
                        let bb = mix_pb * tau;

                        r[[state, segment, spot]] = term_logn + term_beta + lnbeta(kk + aa, nn - kk + bb) - lnbeta(aa, bb);
                    }
                }
            }
        }
    }

    pub fn bb(
        r: &mut ArrayViewMut1<'_, f64>,
        k: &ArrayView1<'_, f64>,
        n: &ArrayView1<'_, f64>,
        a: &ArrayView1<'_, f64>,
        b: &ArrayView1<'_, f64>,
    ) {
        //  See https://docs.rs/ndarray/latest/ndarray/struct.Zip.html#method.par_for_each
        Zip::from(r)
            .and(k)
            .and(n)
            .and(a)
            .and(b)
            .par_for_each(|r, &k, &n, &a, &b| {
                // https://github.com/scipy/scipy/blob/87c46641a8b3b5b47b81de44c07b840468f7ebe7/scipy/stats/_discrete_distns.py#L238
                *r = -(n + 1.).ln() - lnbeta(n - k + 1., k + 1.) + lnbeta(k + a, n - k + b)
                    - lnbeta(a, b)
            });
    }

    pub fn max_min(x: &ArrayViewD<'_, f64>) -> Array1<f64> {
        if x.len() == 0 {
            return arr1(&[]); // If the array has no elements, return empty array
        }

        let max_val = x
            .iter()
            .map(|a| OrderedFloat(*a))
            .max()
            .expect("Error calculating max value.")
            .0;

        let min_val = x
            .iter()
            .map(|a| OrderedFloat(*a))
            .min()
            .expect("Error calculating min value.")
            .0;

        let result_array = arr1(&[max_val, min_val]);

        result_array
    }
}
