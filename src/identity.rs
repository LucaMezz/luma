//! Defines additive and multiplicative identities

use core::ops::{Add, Mul};

/// A quantity that has an additive identity
pub trait Zero: Sized + Add {
    /// The additive identity
    fn zero() -> Self;
}

/// A quantity that has a multiplicative identity
pub trait One: Sized + Mul {
    /// The multiplicative identity
    fn one() -> Self;
}
