//! Defines additive and multiplicative identities
//!
//! These are their own traits, rather than methods on `Field` directly,
//! because the identity values can't be derived from the arithmetic
//! operators alone - each numeric type has to state what its zero and one
//! actually are. `field.rs`'s `impl_zero_one!` macro does that once per
//! built-in type, and `Field`'s blanket impl picks both up as supertraits.

use core::ops::{Add, Mul};

/// A quantity that has an additive identity: `x + Self::zero() == x`
pub trait Zero: Sized + Add {
    /// The additive identity
    fn zero() -> Self;
}

/// A quantity that has a multiplicative identity: `x * Self::one() == x`
pub trait One: Sized + Mul {
    /// The multiplicative identity
    fn one() -> Self;
}
