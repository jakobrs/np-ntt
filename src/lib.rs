use numpy::{PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::{
    exceptions::PyAssertionError, pyclass, pymethods, pymodule, types::PyModule, Py, PyResult,
    Python,
};

pub(crate) mod classical {
    pub(crate) fn add(a: i32, b: i32, modulus: i32) -> i32 {
        let res = a + b;
        if res >= modulus {
            res - modulus
        } else {
            res
        }
    }

    pub(crate) fn sub(a: i32, b: i32, modulus: i32) -> i32 {
        let res = a - b;
        if res < 0 {
            res + modulus
        } else {
            res
        }
    }

    pub(crate) fn mul(a: i32, b: i32, modulus: i32) -> i32 {
        (a as i64 * b as i64 % modulus as i64) as i32
    }

    pub(crate) fn pow(mut base: i32, mut exp: i32, modulus: i32) -> i32 {
        let mut res = 1;

        while exp != 0 {
            if (exp & 1) != 0 {
                res = mul(res, base, modulus);
            }

            base = mul(base, base, modulus);
            exp >>= 1;
        }

        res
    }
}

/// A structure that stores precomputed information that will be used during the NTT
#[pyclass]
#[derive(Clone)]
struct Plan {
    /// The modulus
    #[pyo3(get)]
    modulus: i32,
    /// The length of the array that will be transformed
    #[pyo3(get)]
    length: usize,
    /// Whether this plan is for an inverse transform
    #[pyo3(get)]
    inv: bool,
    /// The basic twiddle factors that will be used
    #[pyo3(get)]
    twiddles: Vec<i32>,
}

#[pymethods]
impl Plan {
    #[new]
    #[pyo3(signature = (length, modulus, inv=false))]
    fn new(length: usize, modulus: i32, inv: bool) -> PyResult<Self> {
        if !length.is_power_of_two() {
            return Err(PyAssertionError::new_err(
                "Only power-of-two NTT lengths are implemented",
            ));
        }
        if !modulus.is_positive() {
            return Err(PyAssertionError::new_err("Modulus must be positive"));
        }
        if (modulus as usize - 1) % length != 0 {
            return Err(PyAssertionError::new_err(
                "Length must be a divisor of modulus - 1",
            ));
        }
        let mut g = 'g: {
            for base in 2.. {
                let attempt = classical::pow(base, (modulus - 1) / length as i32, modulus);
                let mut d = 0;
                if loop {
                    d += 1;
                    if d * d > length {
                        break true;
                    }
                    if length % d != 0 {
                        continue;
                    }
                    if classical::pow(attempt, d as i32, modulus) == 1 {
                        break false;
                    }
                    if d != 1
                        && d * d != length
                        && classical::pow(attempt, (length / d) as i32, modulus) == 1
                    {
                        break false;
                    }
                } {
                    break 'g attempt;
                }
            }
            unreachable!();
        };

        if inv {
            g = classical::pow(g, modulus - 2, modulus);
        }

        Ok(Plan {
            modulus,
            length,
            inv,
            twiddles: std::iter::successors(Some(g), |&n| {
                if n == 1 {
                    None
                } else {
                    Some(classical::mul(n, n, modulus))
                }
            })
            .collect(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Plan(length={length}, modulus={modulus}, inv={inv})",
            length = self.length,
            modulus = self.modulus,
            inv = self.inv,
        )
    }

    /// Performs the NTT in-place
    #[pyo3(signature = (arr, normalize=true))]
    fn perform_inplace(&self, mut arr: PyReadwriteArray1<i32>, normalize: bool) -> PyResult<()> {
        if arr.len() != self.length {
            return Err(PyAssertionError::new_err(format!(
                "Incorrect length: extected {}, got {}",
                self.length,
                arr.len()
            )));
        }

        if normalize {
            for x in arr.as_slice_mut()? {
                *x = x.rem_euclid(self.modulus);
            }
        }

        self.perform_impl(arr.as_slice_mut()?);

        Ok(())
    }

    /// Performs the NTT out-of-place, returning a new array
    #[pyo3(signature = (arr, normalize=true))]
    fn perform(
        &self,
        arr: PyReadonlyArray1<i32>,
        normalize: bool,
        py: Python<'_>,
    ) -> PyResult<Py<PyArray1<i32>>> {
        if arr.len() != self.length {
            return Err(PyAssertionError::new_err(format!(
                "Incorrect length: extected {}, got {}",
                self.length,
                arr.len()
            )));
        }

        let buffer: &PyArray1<i32> = unsafe { PyArray1::new(py, [self.length], false) };
        arr.copy_to(&buffer)?;

        if normalize {
            for x in unsafe { buffer.as_slice_mut() }? {
                *x = x.rem_euclid(self.modulus);
            }
        }

        self.perform_impl(unsafe { buffer.as_slice_mut() }?);

        Ok(buffer.into())
    }
}

/// A tuple of a forward and a reverse NTT plan
#[pyclass]
#[derive(Clone)]
struct Biplan {
    #[pyo3(get)]
    forward: Plan,
    #[pyo3(get)]
    reverse: Plan,
}

#[pymethods]
impl Biplan {
    #[new]
    #[pyo3(signature = (length, modulus))]
    fn new(length: usize, modulus: i32) -> PyResult<Self> {
        Ok(Self {
            forward: Plan::new(length, modulus, false)?,
            reverse: Plan::new(length, modulus, true)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Biplan(length={length}, modulus={modulus})",
            length = self.forward.length,
            modulus = self.forward.modulus,
        )
    }

    /// Performs the NTT in-place
    #[pyo3(signature = (arr, inv=false, normalize=true))]
    fn perform_inplace(
        &self,
        arr: PyReadwriteArray1<i32>,
        inv: bool,
        normalize: bool,
    ) -> PyResult<()> {
        self.get_plan(inv).perform_inplace(arr, normalize)
    }

    /// Performs the NTT out-of-place, returning a new array
    #[pyo3(signature = (arr, inv=false, normalize=true))]
    fn perform(
        &self,
        arr: PyReadonlyArray1<i32>,
        inv: bool,
        normalize: bool,
        py: Python<'_>,
    ) -> PyResult<Py<PyArray1<i32>>> {
        self.get_plan(inv).perform(arr, normalize, py)
    }

    /// Calculates the convolution in-place. b will be overwritten.
    #[pyo3(signature = (a, b, normalize=true))]
    fn convolve_inplace(
        &self,
        mut a: PyReadwriteArray1<i32>,
        mut b: PyReadwriteArray1<i32>,
        normalize: bool,
    ) -> PyResult<()> {
        if a.len() != self.forward.length {
            return Err(PyAssertionError::new_err(format!(
                "Incorrect length: extected {}, got {}",
                self.forward.length,
                a.len()
            )));
        }
        if b.len() != self.forward.length {
            return Err(PyAssertionError::new_err(format!(
                "Incorrect length: extected {}, got {}",
                self.forward.length,
                b.len()
            )));
        }

        let a_data = a.as_slice_mut()?;
        let b_data = b.as_slice_mut()?;

        if normalize {
            for x in a_data.iter_mut() {
                *x = x.rem_euclid(self.forward.modulus);
            }
            for x in b_data.iter_mut() {
                *x = x.rem_euclid(self.forward.modulus);
            }
        }

        self.forward.perform_impl(a_data);
        self.forward.perform_impl(b_data);

        for i in 0..a_data.len() {
            a_data[i] = classical::mul(a_data[i], b_data[i], self.forward.modulus);
        }

        self.reverse.perform_impl(a_data);

        Ok(())
    }

    /// Calculates the convolution out-of-place, returning a new array
    #[pyo3(signature = (a, b, normalize=true))]
    fn convolve(
        &self,
        a: PyReadonlyArray1<i32>,
        b: PyReadonlyArray1<i32>,
        normalize: bool,
        py: Python<'_>,
    ) -> PyResult<Py<PyArray1<i32>>> {
        if a.len() != self.forward.length {
            return Err(PyAssertionError::new_err(format!(
                "Incorrect length: extected {}, got {}",
                self.forward.length,
                a.len()
            )));
        }
        if b.len() != self.forward.length {
            return Err(PyAssertionError::new_err(format!(
                "Incorrect length: extected {}, got {}",
                self.forward.length,
                b.len()
            )));
        }

        let buffer: &PyArray1<i32> = unsafe { PyArray1::new(py, [self.forward.length], false) };
        a.copy_to(&buffer)?;

        let a_data = unsafe { buffer.as_slice_mut()? };
        let mut b_data = b.as_slice()?.to_vec();

        if normalize {
            for x in a_data.iter_mut() {
                *x = x.rem_euclid(self.forward.modulus);
            }
            for x in b_data.iter_mut() {
                *x = x.rem_euclid(self.forward.modulus);
            }
        }

        self.forward.perform_impl(a_data);
        self.forward.perform_impl(b_data.as_mut_slice());

        for i in 0..a_data.len() {
            a_data[i] = classical::mul(a_data[i], b_data[i], self.forward.modulus);
        }

        self.reverse.perform_impl(a_data);

        Ok(buffer.into())
    }
}

impl Biplan {
    fn get_plan(&self, inv: bool) -> &Plan {
        if inv {
            &self.reverse
        } else {
            &self.forward
        }
    }
}

/// Performs the bit-reversal permutation on `vec`
pub fn bit_reversal<T>(vec: &mut [T]) {
    let n = vec.len();

    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;

        if i < j {
            vec.swap(i, j);
        }
    }
}

impl Plan {
    fn perform_impl(&self, arr: &mut [i32]) {
        bit_reversal(arr);

        let n = arr.len();

        let mut twiddles = self.twiddles.iter().rev();
        let mut width = 2;
        while width <= n {
            let &w_d = twiddles.next().unwrap();

            for i in (0..n).step_by(width) {
                let mut w = 1;
                for j in 0..width / 2 {
                    let l = i + j;
                    let r = i + j + width / 2;

                    let arrl = arr[l];
                    let warrr = classical::mul(w, arr[r], self.modulus);

                    arr[l] = classical::add(arrl, warrr, self.modulus);
                    arr[r] = classical::sub(arrl, warrr, self.modulus);

                    w = classical::mul(w, w_d, self.modulus);
                }
            }

            width *= 2;
        }

        if self.inv {
            let n_i = classical::pow(n as i32, self.modulus - 2, self.modulus);
            for x in arr {
                *x = classical::mul(*x, n_i, self.modulus);
            }
        }
    }
}

#[pymodule]
fn np_ntt(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Plan>()?;
    m.add_class::<Biplan>()?;
    Ok(())
}
