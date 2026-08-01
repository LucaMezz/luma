//! Defines a Quaternion
//!
//! A `Quaternion<F>` extends complex numbers with three imaginary units
//! (`i`, `j`, `k`) instead of one, giving `w + xi + yj + zk`. Unit
//! quaternions represent 3D rotations: composing rotations by multiplying
//! quaternions can only drift away from unit length, corrected by a cheap
//! `normalize`, unlike composing rotation matrices which can drift away
//! from orthogonality and needs a full re-orthonormalization to fix.

use core::ops::Mul;

use crate::field::Field;
use crate::real::Sqrt;

/// A quaternion `w + xi + yj + zk` over a `Field` `F`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion<F>
where
    F: Field,
{
    w: F,
    x: F,
    y: F,
    z: F,
}

impl<F> Quaternion<F>
where
    F: Field,
{
    /// Create a new quaternion from its scalar part `w` and imaginary parts
    /// `x`, `y`, `z`
    pub fn new(w: F, x: F, y: F, z: F) -> Self {
        Self { w, x, y, z }
    }

    /// The squared norm `w^2 + x^2 + y^2 + z^2`, cheaper than the norm
    /// itself since it avoids a square root
    fn norm_squared(&self) -> F {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Returns a unit quaternion (norm 1) representing the same rotation as
    /// `self`, correcting for any drift accumulated from repeated
    /// multiplication
    pub fn normalize(&self) -> Self
    where
        F: Sqrt,
    {
        let len = self.norm_squared().sqrt();
        Self {
            w: self.w / len,
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }
}

/// Hamilton product: composes two rotations (when both are unit
/// quaternions). Not commutative - `self * rhs` applies `rhs` first, then
/// `self`, matching the convention `Matrix::mul` uses for composing
/// transforms.
impl<F> Mul for Quaternion<F>
where
    F: Field,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::quaternion::Quaternion;

    #[test]
    fn test_mul_identity() {
        let identity = Quaternion::new(1, 0, 0, 0);
        let q = Quaternion::new(1, 2, 3, 4);

        assert_eq!(identity * q, q);
        assert_eq!(q * identity, q);
    }

    #[test]
    fn test_mul_imaginary_units() {
        // The defining property of the quaternion imaginary units.
        let i = Quaternion::new(0, 1, 0, 0);
        let j = Quaternion::new(0, 0, 1, 0);
        let k = Quaternion::new(0, 0, 0, 1);
        let neg_one = Quaternion::new(-1, 0, 0, 0);

        assert_eq!(i * i, neg_one);
        assert_eq!(j * j, neg_one);
        assert_eq!(k * k, neg_one);

        // i * j = k, but j * i = -k: multiplication isn't commutative.
        assert_eq!(i * j, k);
        assert_eq!(j * i, Quaternion::new(0, 0, 0, -1));
    }

    #[test]
    fn test_normalize() {
        let q = Quaternion::new(0.0, 3.0, 4.0, 0.0);

        assert_eq!(q.normalize(), Quaternion::new(0.0, 0.6, 0.8, 0.0));
    }
}
