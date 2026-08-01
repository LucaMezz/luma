//! Defines a Quaternion
//!
//! A `Quaternion<F>` extends complex numbers with three imaginary units
//! (`i`, `j`, `k`) instead of one, giving `w + xi + yj + zk`. Unit
//! quaternions represent 3D rotations: composing rotations by multiplying
//! quaternions can only drift away from unit length, corrected by a cheap
//! `normalize`, unlike composing rotation matrices which can drift away
//! from orthogonality and needs a full re-orthonormalization to fix.

use core::array;
use core::ops::Mul;

use crate::field::Field;
use crate::matrix::Matrix;
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

    /// Returns the identity quaternion, representing no rotation - the
    /// multiplicative identity for the Hamilton product (`id() * q == q ==
    /// q * id()`)
    pub fn id() -> Self {
        Self {
            w: F::one(),
            x: F::zero(),
            y: F::zero(),
            z: F::zero(),
        }
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

/// Converts a quaternion into the 3x3 rotation matrix it represents.
///
/// Uses the general formula (scaling by `2 / norm_squared` rather than
/// assuming `norm_squared == 1`), so this gives the correct rotation for any
/// nonzero quaternion, not just unit ones - there's no need to `normalize`
/// first.
impl<F> From<Quaternion<F>> for Matrix<F, 3, 3>
where
    F: Field,
{
    fn from(q: Quaternion<F>) -> Self {
        let Quaternion { w, x, y, z } = q;
        let two = F::one() + F::one();
        let s = two / q.norm_squared();

        Matrix::new([
            [
                F::one() - s * (y * y + z * z),
                s * (x * y - w * z),
                s * (x * z + w * y),
            ],
            [
                s * (x * y + w * z),
                F::one() - s * (x * x + z * z),
                s * (y * z - w * x),
            ],
            [
                s * (x * z - w * y),
                s * (y * z + w * x),
                F::one() - s * (x * x + y * y),
            ],
        ])
    }
}

/// Converts a quaternion into the 4x4 rotation matrix it represents, for
/// composing with other 4x4 transforms (see `Matrix::translation`). The
/// rotation occupies the top-left 3x3 block; the rest matches the identity.
impl<F> From<Quaternion<F>> for Matrix<F, 4, 4>
where
    F: Field,
{
    fn from(q: Quaternion<F>) -> Self {
        let r: Matrix<F, 3, 3> = q.into();
        Matrix::new(array::from_fn(|i| {
            array::from_fn(|j| {
                if i < 3 && j < 3 {
                    r[i][j]
                } else if i == j {
                    F::one()
                } else {
                    F::zero()
                }
            })
        }))
    }
}

#[cfg(test)]
mod test {
    use crate::matrix::Matrix;
    use crate::quaternion::Quaternion;

    #[test]
    fn test_mul_identity() {
        let identity = Quaternion::id();
        let q = Quaternion::new(1, 2, 3, 4);

        assert_eq!(identity * q, q);
        assert_eq!(q * identity, q);
    }

    #[test]
    fn test_id() {
        assert_eq!(Quaternion::id(), Quaternion::new(1, 0, 0, 0));
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

    #[test]
    fn test_into_mat3_identity() {
        let identity = Quaternion::id();

        assert_eq!(Matrix::from(identity), Matrix::<i32, 3, 3>::id());
    }

    #[test]
    fn test_into_mat3_spot_check() {
        // A 180-degree rotation is a "nice" angle: cos(90 deg) = 0 and
        // sin(90 deg) = 1 exactly (it's a unit quaternion's *half*-angle
        // that appears in w/x/y/z), so this can use exact integers and
        // assert_eq! rather than a float tolerance. 180 degrees around Z
        // should send X to -X and Y to -Y, leaving Z fixed.
        let q = Quaternion::new(0, 0, 0, 1);

        assert_eq!(
            Matrix::from(q),
            Matrix::new([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        );
    }

    #[test]
    fn test_into_mat3_scale_invariant() {
        // A non-unit quaternion representing the same rotation as the spot
        // check above should convert to the same matrix - no `normalize`
        // needed first. `s = 2 / norm_squared` here is `2 / 4 = 0.5`, exact
        // in IEEE 754, so this can still use assert_eq!.
        let q = Quaternion::new(0.0, 0.0, 0.0, 2.0);

        assert_eq!(
            Matrix::from(q),
            Matrix::new([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        );
    }

    #[test]
    fn test_into_mat4_embeds_rotation() {
        let q = Quaternion::new(0, 0, 0, 1);

        assert_eq!(
            Matrix::from(q),
            Matrix::new([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],])
        );
    }
}
