use ndarray;
use ndarray::{Array, Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArrayDyn,
};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

// See https://itnext.io/how-to-bind-python-numpy-with-rust-ndarray-2efa5717ed21
#[pymodule]
fn calicostem(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
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
    fn compute_emission_probability_nb<'py>(
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

        let mut interim = vec![0.0; n_states * n_obs * n_spots];
        
        rust_fn::compute_emission_probability_nb(
            &mut interim,
            &X,
            &base_nb_mean,
            &tumor_prop,
            &log_mu,
            &alphas,
        );

        let shape = (n_states, n_obs, n_spots);
        let result = Array::from_shape_vec(shape, interim).unwrap();

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

        let mut interim = vec![0.0; n_states * n_obs * n_spots];

        rust_fn::compute_emission_probability_bb_mix(
            &mut interim,
            &X,
            &base_nb_mean,
            &total_bb_RD,
            &pbinom,
            &taus,
            &tumor_prop,
        );

        let shape = (n_states, n_obs, n_spots);
        let result = Array::from_shape_vec(shape, interim).unwrap();

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

        let mut interim = vec![0.0; n_states * n_obs * n_spots];

        rust_fn::compute_emission_probability_bb_mix_weighted(
            &mut interim,
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

        let shape = (n_states, n_obs, n_spots);
        let result = Array::from_shape_vec(shape, interim).unwrap();

        result.into_pyarray(py)
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

    fn get_model_param_with_broadcast(param_array: &ArrayView2<'_, f64>, state: usize, spot: usize, n_spots: usize) -> f64 {
        if param_array.shape()[1] == n_spots {
            return param_array[[state, spot]]
        }
        else {
            return param_array[[state, 0]]
        }
    }

    pub fn compute_emission_probability_nb(
        r: &mut Vec<f64>,
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

	// TODO Toggle threading on n_spots
        let n_obs = X.shape()[0];
        let n_spots = X.shape()[1];
        let n_states = log_mu.shape()[0];

        let state_chunks: Vec<(usize, &mut [f64])> = r
        .chunks_mut(n_obs * n_spots)
        .enumerate()
        .collect();

        state_chunks.into_par_iter()
        .for_each(|(state, state_chunk)| {
            for segment in 0..n_obs {
                for spot in 0..n_spots {
                    let idx = segment * n_spots + spot;
		    let base = base_nb_mean[[segment, spot]];

                    if base > 0. {
		        if tumor_prop[[segment, spot]].is_nan() {
                     	   state_chunk[idx] = std::f64::NAN;
                           continue;
                    	}	   
		       
                        let shift = base * (1. - tumor_prop[[segment, spot]]);
                        let x = X[[segment, spot]];
    
                        let lnGx = lnfact(x as u32);

			//  log_mu[[state, spot]].exp();
                        let mu = get_model_param_with_broadcast(log_mu, state, spot, n_spots).exp();
    
                        let mean = shift + (mu * base * tumor_prop[[segment, spot]]);

			let alpha = get_model_param_with_broadcast(alphas, state, spot, n_spots);
                        let var = mean + alpha * mean.powf(2.);
    
                        let p = mean / var;
                        let n = mean * p / (1. - p);
        
                        state_chunk[idx] = ln_gamma(n + x) - ln_gamma(n) - lnGx
                                + (n * p.ln())
                                + (x * (-p).ln_1p());
                    }
                }
            }
        });
    }

    pub fn compute_emission_probability_bb_mix(
        r: &mut Vec<f64>,
        X: &ArrayView2<'_, f64>,
        base_nb_mean: &ArrayView2<'_, f64>,
        total_bb_RD: &ArrayView2<'_, f64>,
        pbinom: &ArrayView2<'_, f64>,
        taus: &ArrayView2<'_, f64>,
        tumor_prop: &ArrayView2<'_, f64>,
    ) {
        // TODO Toggle threading on n_spots
        let n_obs = X.shape()[0];
        let n_spots = X.shape()[1];
        let n_states = pbinom.shape()[0];

        let state_chunks: Vec<(usize, &mut [f64])> = r
        .chunks_mut(n_obs * n_spots)
        .enumerate()
        .collect();

        state_chunks.into_par_iter()
        .for_each(|(state, state_chunk)| {
            for segment in 0..n_obs {
                for spot in 0..n_spots {
                    let idx = segment * n_spots + spot;
                
                    if total_bb_RD[[segment, spot]] > 0. {
		        if tumor_prop[[segment, spot]].is_nan() {
                     	   state_chunk[idx] = std::f64::NAN;
                           continue;
                        }	   
		    
                        let kk = X[[segment, spot]];
                        let nn = total_bb_RD[[segment, spot]];

                        let term_logn = -(nn + 1.).ln();
                        let term_beta = -lnbeta(nn - kk + 1., kk + 1.);

                        let shift = 0.5 * (1. - tumor_prop[[segment, spot]]);

			let tau = get_model_param_with_broadcast(taus, state, spot, n_spots);
			let pbin = get_model_param_with_broadcast(pbinom, state, spot, n_spots);

                        let mix_pa = pbin * tumor_prop[[segment, spot]] + shift;
                        let mix_pb = (1. - pbin) * tumor_prop[[segment, spot]] + shift;
    
                        let aa = mix_pa * tau;
                        let bb = mix_pb * tau;

                        state_chunk[idx] = term_logn + term_beta + lnbeta(kk + aa, nn - kk + bb) - lnbeta(aa, bb);
                    }
                }
            }
        });
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

    pub fn compute_emission_probability_bb_mix_weighted(
        r: &mut Vec<f64>,
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
        // TODO Toggle threading on n_spots
        let n_obs = X.shape()[0];
        let n_spots = X.shape()[1];
        let n_states = pbinom.shape()[0];

        let segment_chunks = get_segment_chunks(sample_lengths);

        let state_chunks: Vec<(usize, &mut [f64])> = r
        .chunks_mut(n_obs * n_spots)
        .enumerate()
        .collect();

        state_chunks.into_par_iter()
        .for_each(|(state, state_chunk)| {
            for segment in 0..n_obs {
                for spot in 0..n_spots {
                    let idx = segment * n_spots + spot;  

                    if total_bb_RD[[segment, spot]] > 0. {
		        if tumor_prop[[segment, spot]].is_nan() {
                            state_chunk[idx] = std::f64::NAN;
                            continue;
                        }

                        let kk = X[[segment, spot]];
                        let nn = total_bb_RD[[segment, spot]];
    
                        let segment_chunk = segment_chunks[segment];
    
                        let term_logn = -(nn + 1.).ln();
                        let term_beta = -lnbeta(nn - kk + 1., kk + 1.);

			//  (log_mu[[state, spot]] - log_mu_shift[[segment_chunk, spot]]).exp();
			let mu = (
			    get_model_param_with_broadcast(log_mu, state, spot, n_spots)
			  - get_model_param_with_broadcast(log_mu_shift, segment_chunk, spot, n_spots)
			).exp();

                        // let tau = taus[[state, spot]];
			let tau = get_model_param_with_broadcast(taus, state, spot, n_spots);
			let pbin = get_model_param_with_broadcast(pbinom, state, spot, n_spots);

                        let weighted_tumor_prop = (tumor_prop[[segment, spot]] * mu) / (tumor_prop[[segment, spot]] * mu + 1. - tumor_prop[[segment, spot]]); 
                        let shift = 0.5 * (1. - weighted_tumor_prop);
    
                        let mix_pa = pbin * weighted_tumor_prop + shift;
                        let mix_pb = (1. - pbin) * weighted_tumor_prop + shift;
        
                        let aa = mix_pa * tau;
                        let bb = mix_pb * tau;
    
                        state_chunk[idx] = term_logn + term_beta + lnbeta(kk + aa, nn - kk + bb) - lnbeta(aa, bb);
                    }
                }
            }
        });
    }

    fn get_lnbeta(a: f64, b: f64) -> f64 {
        // TODO check equality.
        if a <= 0. || b <= 0. {
            return std::f64::NAN;
        }
        else {
            return lnbeta(a, b)
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
                *r = -(n + 1.).ln() - get_lnbeta(n - k + 1., k + 1.) - get_lnbeta(a, b) + get_lnbeta(k + a, n - k + b)
            });
    }
}
