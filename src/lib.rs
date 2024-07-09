use ndarray;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn};
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
    use ndarray::{arr1, Array1, Zip};
    use numpy::ndarray::{ArrayView1, ArrayViewD, ArrayViewMut1, ArrayViewMutD};
    use ordered_float::OrderedFloat;
    use rand::Rng;
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
        Zip::from(r)
            .and(k)
            .and(n)
            .and(p)
            .par_for_each(|r, &k, &n, &p| {
                *r = ln_gamma(k + n) - ln_gamma(k + 1.) - ln_gamma(n)
                    + k * (1. - p).ln()
                    + n * p.ln()
            });
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
                *r = ln_gamma(n + 1.) - ln_gamma(k + 1.) - ln_gamma(n - k + 1.)
                + ln_gamma(k + a)
                + ln_gamma(n - k + b)
                - ln_gamma(n + a + b)
                + ln_gamma(a + b)
                - ln_gamma(a)
                - ln_gamma(b);
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
