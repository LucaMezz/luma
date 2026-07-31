//! Defines operations only meaningful for real (floating-point) numbers.
//!
//! Not every `Field` has a square root (or, later, sine/cosine) - these
//! aren't field axioms, only true for reals - so they live in their own
//! trait(s) here rather than being pulled into `Field` itself.

/// A quantity that has a square root
pub trait Sqrt {
    /// Returns the square root
    fn sqrt(self) -> Self;
}

impl Sqrt for f32 {
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
}

impl Sqrt for f64 {
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
}

/// A quantity that has an inverse cosine (arccosine)
pub trait Acos {
    /// Returns the arccosine, in radians
    fn acos(self) -> Self;
}

impl Acos for f32 {
    fn acos(self) -> Self {
        libm::acosf(self)
    }
}

impl Acos for f64 {
    fn acos(self) -> Self {
        libm::acos(self)
    }
}

/// A quantity that can be rounded to a whole number
pub trait Round {
    /// Rounds down to the nearest whole number
    fn floor(self) -> Self;

    /// Rounds up to the nearest whole number
    fn ceil(self) -> Self;

    /// Rounds to the nearest whole number, away from zero on a tie
    fn round(self) -> Self;
}

impl Round for f32 {
    fn floor(self) -> Self {
        libm::floorf(self)
    }

    fn ceil(self) -> Self {
        libm::ceilf(self)
    }

    fn round(self) -> Self {
        libm::roundf(self)
    }
}

impl Round for f64 {
    fn floor(self) -> Self {
        libm::floor(self)
    }

    fn ceil(self) -> Self {
        libm::ceil(self)
    }

    fn round(self) -> Self {
        libm::round(self)
    }
}
