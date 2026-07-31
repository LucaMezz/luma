//! Small helpers for elementwise operations on fixed-size arrays.
//!
//! `Matrix` and `Vector` both need to combine two same-length arrays of a
//! `Field` element-by-element (for `Add`/`Sub`); `Matrix` does this per row,
//! `Vector` does it over its whole (flat) storage. `zip_map` is the one
//! piece of logic that's actually shared between the two, so it lives here
//! instead of being written out twice. Scalar multiplication doesn't need a
//! helper of its own since `[F; N]` already has `.map()` built in.

use core::array;

/// Combines two arrays elementwise using `f`
pub(crate) fn zip_map<F: Copy, const N: usize>(
    a: [F; N],
    b: [F; N],
    mut f: impl FnMut(F, F) -> F,
) -> [F; N] {
    array::from_fn(|i| f(a[i], b[i]))
}
