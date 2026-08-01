//! Defines a Vector
//!
//! A `Vector<F, N>` is a flat array of `N` components, distinct from a
//! `Matrix<F, N, 1>`/`Matrix<F, 1, N>` (column/row matrix) even though they
//! hold the same data - the `From` impls below convert between the two so a
//! vector can be multiplied against a matrix.

use core::array;
use core::ops::{Add, Index, Mul, Neg, Sub};

use crate::array::zip_map;
use crate::field::Field;
use crate::matrix::Matrix;
use crate::order::MinMax;
use crate::real::{Acos, Approx, Round, Sqrt};

/// A vector with `N` components over a `Field` `F`, used for coordinates,
/// directions, and offsets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector<F, const N: usize>
where
    F: Field,
{
    data: [F; N],
}

/// A 2-component vector
pub type Vec2<F> = Vector<F, 2>;

/// A 3-component vector
pub type Vec3<F> = Vector<F, 3>;

/// A 4-component vector
pub type Vec4<F> = Vector<F, 4>;

impl<F, const N: usize> Vector<F, N>
where
    F: Field,
{
    /// Create a new vector from its components
    pub fn new(data: [F; N]) -> Self {
        Self { data }
    }

    /// Returns the zero vector
    pub fn zero() -> Self {
        Self {
            data: [F::zero(); N],
        }
    }

    /// Returns the vector with every component equal to one
    pub fn one() -> Self {
        Self {
            data: [F::one(); N],
        }
    }

    /// Returns this vector as a 1xN row matrix, so it can be multiplied
    /// against a matrix on the left (`vector.to_row_matrix() * matrix`).
    pub fn to_row_matrix(&self) -> Matrix<F, 1, N> {
        Matrix::new([self.data])
    }

    /// Returns the negation of the vector. It negates each component of
    /// the original vector so it faces in the opposite direction.
    pub fn negate(&self) -> Self {
        Self {
            data: array::from_fn(|i| -self.data[i]),
        }
    }

    /// Returns the dot product of two vectors
    pub fn dot(&self, rhs: Self) -> F {
        (0..N)
            .map(|i| self.data[i] * rhs.data[i])
            .fold(F::zero(), |acc, x| acc + x)
    }

    /// Computes the length of the vector as the square root of its inner
    /// product with itself, as it's defined for any inner product space
    pub fn length(&self) -> F
    where
        F: Sqrt,
    {
        self.dot(*self).sqrt()
    }

    /// Returns a unit vector (length 1) pointing in the same direction
    pub fn unit(&self) -> Self
    where
        F: Mul<Self, Output = Self> + Sqrt,
    {
        (F::one() / self.length()) * *self
    }

    /// Returns the distance between two vectors
    pub fn distance(&self, other: Self) -> F
    where
        F: Sqrt,
    {
        (*self - other).length()
    }

    /// Returns whether `self` and `other` are componentwise within
    /// `epsilon` of each other. See `Approx` for choosing an epsilon.
    pub fn is_close(&self, other: Self, epsilon: F) -> bool
    where
        F: Approx,
    {
        (0..N).all(|i| self.data[i].is_close(other.data[i], epsilon))
    }

    /// Like `is_close`, but using `F`'s default epsilon
    pub fn is_close_default(&self, other: Self) -> bool
    where
        F: Approx,
    {
        self.is_close(other, F::default_epsilon())
    }

    /// Linearly interpolates between two vectors: `t = 0` gives `self`,
    /// `t = 1` gives `other`. `t` outside `[0, 1]` extrapolates rather than
    /// being clamped.
    pub fn lerp(&self, other: Self, t: F) -> Self {
        *self + (other - *self) * t
    }

    /// Projects this vector onto `other`
    pub fn project(&self, other: Self) -> Self
    where
        F: Mul<Self, Output = Self>,
    {
        (self.dot(other) / other.dot(other)) * other
    }

    /// Computes the angle between a vector and another vector
    ///
    /// Uses inverse cosine so can be expensive
    pub fn angle(&self, other: Self) -> F
    where
        F: Sqrt + Acos,
    {
        (self.dot(other) / (self.length() * other.length())).acos()
    }

    /// Computes a new vector by taking the component-wise minimum
    /// of this vector and another vector
    pub fn min(&self, other: Self) -> Self
    where
        F: MinMax,
    {
        Self {
            data: array::from_fn(|i| self.data[i].min(other.data[i])),
        }
    }

    /// Computes a new vector by taking the component-wise maximum
    /// of this vector and another vector
    pub fn max(&self, other: Self) -> Self
    where
        F: MinMax,
    {
        Self {
            data: array::from_fn(|i| self.data[i].max(other.data[i])),
        }
    }

    /// Rounds each component down to the nearest whole number
    pub fn floor(&self) -> Self
    where
        F: Round,
    {
        Self {
            data: self.data.map(|c| c.floor()),
        }
    }

    /// Rounds each component up to the nearest whole number
    pub fn ceil(&self) -> Self
    where
        F: Round,
    {
        Self {
            data: self.data.map(|c| c.ceil()),
        }
    }

    /// Rounds each component to the nearest whole number, away from zero on
    /// a tie
    pub fn round(&self) -> Self
    where
        F: Round,
    {
        Self {
            data: self.data.map(|c| c.round()),
        }
    }
}

impl<F> Vector<F, 3>
where
    F: Field,
{
    /// Returns the cross product of two 3D vectors: a vector perpendicular
    /// to both, following the right-hand rule (`x.cross(&y) == z` for the
    /// standard basis vectors), matching the rotation convention used by
    /// `Matrix::rotation`
    pub fn cross(&self, rhs: &Self) -> Self {
        Self::new([
            self[1] * rhs[2] - self[2] * rhs[1],
            self[2] * rhs[0] - self[0] * rhs[2],
            self[0] * rhs[1] - self[1] * rhs[0],
        ])
    }
}

/// Converts a 3D vector into a 4D homogeneous vector with `w = 1`, e.g. so a
/// position can be multiplied against a 4x4 matrix. `w = 1` marks the vector
/// as a point rather than a direction, so `Matrix::translation`'s offset
/// carries through to it - a pure direction would instead use `w = 0` so
/// translation leaves it unaffected, but there's no need for that case here
/// yet.
impl<F> From<Vector<F, 3>> for Vector<F, 4>
where
    F: Field,
{
    fn from(v: Vector<F, 3>) -> Self {
        Self::new([v[0], v[1], v[2], F::one()])
    }
}

impl<F> Vector<F, 4>
where
    F: Field,
{
    /// Converts a 4D homogeneous vector back to 3D by perspective dividing:
    /// dividing `x`, `y`, and `z` by `w`. Deliberately not a `From` impl -
    /// `w` is only guaranteed to be `1` (making this equivalent to just
    /// dropping it) right after `Vector<F, 3>::into`; after a perspective
    /// projection matrix `w` generally isn't `1` anymore, and a `From`
    /// impl's unconditional truncation would silently produce the wrong
    /// point in that case rather than the actual division needed.
    pub fn perspective_divide(&self) -> Vector<F, 3> {
        Vector::new([self[0] / self[3], self[1] / self[3], self[2] / self[3]])
    }
}

/// Indexes into the vector's components, e.g. `v[0]` for the first
impl<F, const N: usize> Index<usize> for Vector<F, N>
where
    F: Field,
{
    type Output = F;

    fn index(&self, index: usize) -> &F {
        &self.data[index]
    }
}

/// Elementwise vector addition
impl<F, const N: usize> Add for Vector<F, N>
where
    F: Field,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            data: zip_map(self.data, rhs.data, |a, b| a + b),
        }
    }
}

/// Elementwise vector subtraction
impl<F, const N: usize> Sub for Vector<F, N>
where
    F: Field,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            data: zip_map(self.data, rhs.data, |a, b| a - b),
        }
    }
}

/// Componentwise vector negation, e.g. `-vector`. Equivalent to `negate`.
impl<F, const N: usize> Neg for Vector<F, N>
where
    F: Field,
{
    type Output = Self;

    fn neg(self) -> Self {
        self.negate()
    }
}

/// Scalar multiplication: multiplies every component by `rhs`
impl<F, const N: usize> Mul<F> for Vector<F, N>
where
    F: Field,
{
    type Output = Self;

    fn mul(self, rhs: F) -> Self {
        Self {
            data: self.data.map(|c| rhs * c),
        }
    }
}

// `impl<F: Field, const N: usize> Mul<Vector<F, N>> for F` can't be written
// generically - F is an unconstrained/foreign type parameter here, so Rust's
// orphan rule rejects a blanket impl of the foreign `Mul` trait for it (the
// same restriction that stops a generic `scalar * matrix` impl). Instead,
// implement it concretely per field type, same pattern as `Zero`/`One`.
macro_rules! impl_scalar_mul {
    ($($t:ty),* $(,)?) => {
        $(
            impl<const N: usize> Mul<Vector<$t, N>> for $t {
                type Output = Vector<$t, N>;

                fn mul(self, rhs: Vector<$t, N>) -> Vector<$t, N> {
                    rhs * self
                }
            }
        )*
    };
}

impl_scalar_mul!(i8, i16, i32, i64, i128, isize, f32, f64);

/// Converts a vector into an Nx1 column matrix, so it can be multiplied
/// against a matrix on the right (`matrix * Matrix::from(vector)`).
///
/// There's deliberately no equivalent `From<Vector<F, N>> for Matrix<F, 1,
/// N>` (row) impl alongside this one: for `N = 1` both would apply to the
/// same `Matrix<F, 1, 1>`, which is exactly the coherence conflict a
/// `Mul`-for-dot-product impl hit earlier. `to_row_matrix` above covers that
/// direction as a plain method instead, sidestepping the conflict entirely.
impl<F, const N: usize> From<Vector<F, N>> for Matrix<F, N, 1>
where
    F: Field,
{
    fn from(v: Vector<F, N>) -> Self {
        Matrix::new(v.data.map(|c| [c]))
    }
}

/// Converts an Nx1 column matrix back into a vector. The inverse of
/// `From<Vector<F, N>> for Matrix<F, N, 1>` above; see that impl for why
/// there's no equivalent for row matrices (`Matrix::to_vector` covers that
/// direction instead).
impl<F, const N: usize> From<Matrix<F, N, 1>> for Vector<F, N>
where
    F: Field,
{
    fn from(m: Matrix<F, N, 1>) -> Self {
        Vector::new(m.rows().map(|row| row[0]))
    }
}

/// Converts a vector into its underlying `[F; N]` array of components.
impl<F, const N: usize> From<Vector<F, N>> for [F; N]
where
    F: Field,
{
    fn from(v: Vector<F, N>) -> Self {
        v.data
    }
}

#[cfg(test)]
mod test {
    use crate::matrix::Matrix;
    use crate::real::Approx;
    use crate::vector::Vector;

    #[test]
    fn test_add() {
        let a = Vector::new([1, 2, 3]);
        let b = Vector::new([4, 5, 6]);

        assert_eq!(a + b, Vector::new([5, 7, 9]));
    }

    #[test]
    fn test_sub() {
        let a = Vector::new([1, 2, 3]);
        let b = Vector::new([4, 5, 6]);

        assert_eq!(a - b, Vector::new([-3, -3, -3]));
    }

    #[test]
    fn test_neg() {
        let a = Vector::new([1, -2, 3]);

        assert_eq!(-a, Vector::new([-1, 2, -3]));
    }

    #[test]
    fn test_mul_scalar() {
        let a = Vector::new([1, 2, 3]);

        assert_eq!(a * 2, Vector::new([2, 4, 6]));
        assert_eq!(2 * a, Vector::new([2, 4, 6]));
    }

    #[test]
    fn test_index() {
        let a = Vector::new([1, 2, 3]);

        assert_eq!(a[0], 1);
        assert_eq!(a[1], 2);
        assert_eq!(a[2], 3);
    }

    #[test]
    fn test_dot() {
        let a = Vector::new([1, 2, 3]);
        let b = Vector::new([4, 5, 6]);

        assert_eq!(a.dot(b), 4 + 10 + 18);
    }

    #[test]
    fn test_cross() {
        let x = Vector::new([1, 0, 0]);
        let y = Vector::new([0, 1, 0]);

        assert_eq!(x.cross(&y), Vector::new([0, 0, 1]));
    }

    #[test]
    fn test_from_vec3_for_vec4() {
        let v = Vector::new([1, 2, 3]);
        let h: Vector<i32, 4> = v.into();

        assert_eq!(h, Vector::new([1, 2, 3, 1]));
    }

    #[test]
    fn test_perspective_divide() {
        let affine = Vector::new([1.0, 2.0, 3.0, 1.0]);
        assert_eq!(affine.perspective_divide(), Vector::new([1.0, 2.0, 3.0]));

        let projected = Vector::new([2.0, 4.0, 6.0, 2.0]);
        assert_eq!(projected.perspective_divide(), Vector::new([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_zero() {
        assert_eq!(Vector::<i32, 3>::zero(), Vector::new([0, 0, 0]));
    }

    #[test]
    fn test_one() {
        assert_eq!(Vector::<i32, 3>::one(), Vector::new([1, 1, 1]));
    }

    #[test]
    fn test_into_array() {
        let v = Vector::new([1, 2, 3]);
        let a: [i32; 3] = v.into();

        assert_eq!(a, [1, 2, 3]);
    }

    #[test]
    fn test_into_column_matrix() {
        let v = Vector::new([1, 2, 3]);
        let m: Matrix<i32, 3, 1> = v.into();

        assert_eq!(m, Matrix::new([[1], [2], [3]]));
    }

    #[test]
    fn test_to_row_matrix() {
        let v = Vector::new([1, 2, 3]);

        assert_eq!(v.to_row_matrix(), Matrix::new([[1, 2, 3]]));
    }

    #[test]
    fn test_from_column_matrix() {
        let m = Matrix::new([[1], [2], [3]]);
        let v: Vector<i32, 3> = m.into();

        assert_eq!(v, Vector::new([1, 2, 3]));
    }

    #[test]
    fn test_matrix_to_vector() {
        let m = Matrix::new([[1, 2, 3]]);

        assert_eq!(m.to_vector(), Vector::new([1, 2, 3]));
    }

    #[test]
    fn test_matrix_times_vector() {
        let m = Matrix::new([[1, 2, 3], [4, 5, 6]]);
        let v = Vector::new([1, 0, 1]);

        let result = m * Matrix::from(v);

        assert_eq!(result, Matrix::new([[4], [10]]));
    }

    #[test]
    fn test_length() {
        let v = Vector::new([3.0, 4.0]);

        assert_eq!(v.length(), 5.0);
    }

    #[test]
    fn test_unit() {
        let v: Vector<f64, 2> = Vector::new([3.0, 4.0]);
        let u = v.unit();

        // Exact float equality is unreliable after arithmetic, so check
        // within a tolerance rather than with assert_eq!.
        assert!(u.is_close_default(Vector::new([0.6, 0.8])));
        assert!(u.length().is_close_default(1.0));
    }

    #[test]
    fn test_is_close() {
        let a = Vector::new([1.0, 2.0]);
        let b = Vector::new([1.0 + 1e-11, 2.0 - 1e-11]);
        let c = Vector::new([1.1, 2.0]);

        assert!(a.is_close_default(b));
        assert!(!a.is_close_default(c));
        assert!(a.is_close(c, 0.2));
    }

    #[test]
    fn test_project() {
        let a = Vector::new([3.0, 4.0]);
        let b = Vector::new([1.0, 0.0]);

        assert_eq!(a.project(b), Vector::new([3.0, 0.0]));
    }

    #[test]
    fn test_min_max() {
        let a = Vector::new([1.0, 5.0, f64::NAN]);
        let b = Vector::new([3.0, 2.0, 7.0]);

        assert_eq!(a.min(b), Vector::new([1.0, 2.0, 7.0]));
        assert_eq!(a.max(b), Vector::new([3.0, 5.0, 7.0]));
    }

    #[test]
    fn test_floor_ceil_round() {
        let v = Vector::new([1.2, -1.2, 1.5]);

        assert_eq!(v.floor(), Vector::new([1.0, -2.0, 1.0]));
        assert_eq!(v.ceil(), Vector::new([2.0, -1.0, 2.0]));
        assert_eq!(v.round(), Vector::new([1.0, -1.0, 2.0]));
    }
}
