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
    fn compute_emission_probability_nb_betabinom_mix<'py>(
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

        for i in 0..n_states {
            for j in 0..n_obs {
                let mu = log_mu[[i, j]].exp();
                let alpha = alphas[[i, j]];

                for k in 0..n_spots {
                    let base = base_nb_mean[[j, k]];

                    if base == 0. {
                        continue;
                    }

                    let x = X[[j, k]];
                    let shift = base * (1. - tumor_prop[[j, k]]);

                    let mean = shift + base * tumor_prop[[j, k]] * mu;
                    let var = mean + alpha * mean.powf(2.);

                    let p = mean / var;
                    let n = mean * p / (1. - p);

                    let n32 = n as u32;
                    let x32 = x as u32;

                    // println!("x: {}, mu: {}, var: {}, p: {}, n: {}", x, mu, var, p, n);

                    r[[i, j, k]] = lnfact(x32 + n32 - 1) - lnfact(n32 - 1) - lnfact(x32)
                        + x * (1. - p).ln()
                        + n * p.ln();
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

    pub fn bbab(
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
                *r = lnbeta(k + a, n - k + b) - lnbeta(a, b)
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
