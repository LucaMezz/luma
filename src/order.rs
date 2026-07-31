//! Defines a `min`/`max` capability that works for both integers and
//! floats.
//!
//! `core::cmp::Ord` can't be the bound for this: floats only implement
//! `PartialOrd` (`NaN` breaks the total ordering `Ord` requires), and a
//! blanket `impl<T: Ord> MinMax for T` plus manual `f32`/`f64` impls doesn't
//! work either - the compiler rejects it as a potential conflict, since
//! `std` could in principle add `Ord` for floats in a future version.
//! Instead, each type gets its own impl, same pattern as `Zero`/`One`.

/// A quantity that can be compared and picked between, without requiring a
/// total order (`Ord`) - so it works for floats too.
pub trait MinMax {
    /// Returns the smaller of the two values
    fn min(self, other: Self) -> Self;

    /// Returns the larger of the two values
    fn max(self, other: Self) -> Self;
}

macro_rules! impl_minmax_ord {
    ($($t:ty),* $(,)?) => {
        $(
            impl MinMax for $t {
                fn min(self, other: Self) -> Self {
                    Ord::min(self, other)
                }

                fn max(self, other: Self) -> Self {
                    Ord::max(self, other)
                }
            }
        )*
    };
}

impl_minmax_ord!(i8, i16, i32, i64, i128, isize);

// Floats follow IEEE 754 minNum/maxNum semantics: if exactly one operand is
// NaN, the other is returned; only if both are NaN is the result NaN.
impl MinMax for f32 {
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
}

impl MinMax for f64 {
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
}
