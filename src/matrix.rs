//! Defines a Matrix
//!
//! Storage is row-major (`data[row][column]`) and dimensions are compile-time
//! `const` generics, so the type system catches mismatched-dimension errors
//! (e.g. multiplying a 3x2 by a 4x2) instead of them surfacing as runtime
//! panics.

use core::array;
use core::ops::{Add, Index, IndexMut, Mul, Sub};

use crate::array::zip_map;
use crate::field::Field;
use crate::real::{Cos, Sin};
use crate::vector::Vector;

/// An NxM matrix (N rows, M columns) over a `Field` `F`. Size must be known
/// at compile time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix<F, const N: usize, const M: usize>
where
    F: Field,
{
    data: [[F; M]; N],
}

/// A 2x2 matrix
pub type Mat2<F> = Matrix<F, 2, 2>;

/// A 3x3 matrix
pub type Mat3<F> = Matrix<F, 3, 3>;

/// A 4x4 matrix
pub type Mat4<F> = Matrix<F, 4, 4>;

impl<F, const N: usize, const M: usize> Matrix<F, N, M>
where
    F: Field,
{
    /// Create a new matrix from its rows, e.g. `Matrix::new([[1, 2], [3,
    /// 4]])` is the matrix with first row `[1, 2]` and second row `[3, 4]`
    pub fn new(data: [[F; M]; N]) -> Self {
        Self { data }
    }

    /// Create a new diagonal matrix from diagonal entries, with every
    /// off-diagonal entry set to zero
    pub fn diagonal(data: [F; N]) -> Self {
        Self {
            data: array::from_fn(|i| array::from_fn(|j| if i == j { data[i] } else { F::zero() })),
        }
    }

    /// Constructs a matrix from an array of row vectors. Inverse of `rows`.
    pub fn from_rows(rows: [Vector<F, M>; N]) -> Self {
        Self {
            data: rows.map(|row| row.into()),
        }
    }

    /// Constructs a matrix from an array of column vectors. Inverse of
    /// `columns`.
    pub fn from_columns(columns: [Vector<F, N>; M]) -> Self {
        Self {
            data: array::from_fn(|i| array::from_fn(|j| columns[j][i])),
        }
    }

    /// Returns the zero matrix
    pub fn zero() -> Self {
        Self {
            data: [[F::zero(); M]; N],
        }
    }

    /// Returns the identity matrix
    pub fn id() -> Self {
        Self {
            data: array::from_fn(|i| array::from_fn(|j| if i == j { F::one() } else { F::zero() })),
        }
    }

    /// Returns the transpose of the matrix
    pub fn transpose(&self) -> Matrix<F, M, N> {
        Matrix::new(array::from_fn(|i| array::from_fn(|j| self.data[j][i])))
    }

    /// Returns the rows of the matrix as an array of vectors. Inverse of
    /// `from_rows`.
    pub fn rows(&self) -> [Vector<F, M>; N] {
        array::from_fn(|i| Vector::new(self.data[i]))
    }

    /// Returns the columns of the matrix as an array of vectors. Inverse of
    /// `from_columns`.
    pub fn columns(&self) -> [Vector<F, N>; M] {
        array::from_fn(|j| Vector::new(array::from_fn(|i| self.data[i][j])))
    }

    /// Checks if the matrix is square
    pub fn is_square(&self) -> bool {
        N == M
    }

    /// Checks if the matrix is the identity matrix
    pub fn is_identity(&self) -> bool {
        *self == Matrix::id()
    }

    /// Checks if all off-diagonal entries are zero
    pub fn is_diagonal(&self) -> bool {
        (0..N).all(|i| (0..M).all(|j| i == j || self.data[i][j] == F::zero()))
    }
}

impl<F, const N: usize> Matrix<F, N, N>
where
    F: Field,
{
    /// Checks if the matrix equals its own transpose
    pub fn is_symmetric(&self) -> bool {
        self.transpose() == *self
    }

    /// Checks if the matrix's transpose is also its inverse
    /// (`self * self.transpose() == identity`)
    pub fn is_orthogonal(&self) -> bool {
        (*self * self.transpose()).is_identity()
    }

    /// Returns the trace of the matrix (the sum of its diagonal entries)
    pub fn trace(&self) -> F {
        (0..N)
            .map(|x| self.data[x][x])
            .fold(F::zero(), |acc, x| acc + x)
    }

    /// Constructs the matrix for a scale by `s` along each axis: a diagonal
    /// matrix with `s`'s components on the diagonal
    pub fn scale(s: Vector<F, N>) -> Self {
        Self::diagonal(s.into())
    }
}

impl<F> Matrix<F, 2, 2>
where
    F: Field + Sin + Cos,
{
    /// Constructs the matrix for a rotation by an angle in radians,
    /// counterclockwise (from the x-axis towards the y-axis)
    ///
    /// Unlike the 3x3/4x4 rotations, there's only one plane to rotate in, so
    /// this doesn't take an `Axis`.
    pub fn rotation(angle: F) -> Self {
        let s = angle.sin();
        let c = angle.cos();
        Matrix::new([[c, -s], [s, c]])
    }
}

impl<F> Matrix<F, 4, 4>
where
    F: Field,
{
    /// Constructs the matrix for a translation in three dimensions
    ///
    /// Assumes points are represented as column vectors and transformed by
    /// multiplying this matrix on the left (`translation * point`), which is
    /// why the offsets end up in the last column rather than the last row.
    pub fn translation(tx: F, ty: F, tz: F) -> Self {
        let mut t = Matrix::id();
        t[0][3] = tx;
        t[1][3] = ty;
        t[2][3] = tz;
        t
    }
}

impl<F> Matrix<F, 4, 4>
where
    F: Field + Sin + Cos,
{
    /// Constructs the matrix for a rotation around an axis, by an angle in
    /// radians
    ///
    /// Follows the right-hand rule: looking from the positive end of `axis`
    /// towards the origin, a positive angle rotates counterclockwise (e.g.
    /// rotating around `Axis::Z` sends the x-axis towards the y-axis).
    pub fn rotation(axis: Axis, angle: F) -> Self {
        let s = angle.sin();
        let c = angle.cos();
        let mut r = Matrix::id();
        match axis {
            Axis::X => {
                r[1][1] = c;
                r[1][2] = -s;
                r[2][1] = s;
                r[2][2] = c;
            }
            Axis::Y => {
                r[0][0] = c;
                r[0][2] = s;
                r[2][0] = -s;
                r[2][2] = c;
            }
            Axis::Z => {
                r[0][0] = c;
                r[0][1] = -s;
                r[1][0] = s;
                r[1][1] = c;
            }
        }
        r
    }
}

impl<F> Matrix<F, 3, 3>
where
    F: Field + Sin + Cos,
{
    /// Constructs the matrix for a rotation around an axis, by an angle in
    /// radians
    ///
    /// Follows the right-hand rule: looking from the positive end of `axis`
    /// towards the origin, a positive angle rotates counterclockwise (e.g.
    /// rotating around `Axis::Z` sends the x-axis towards the y-axis).
    pub fn rotation(axis: Axis, angle: F) -> Self {
        let s = angle.sin();
        let c = angle.cos();
        let mut r = Matrix::id();
        match axis {
            Axis::X => {
                r[1][1] = c;
                r[1][2] = -s;
                r[2][1] = s;
                r[2][2] = c;
            }
            Axis::Y => {
                r[0][0] = c;
                r[0][2] = s;
                r[2][0] = -s;
                r[2][2] = c;
            }
            Axis::Z => {
                r[0][0] = c;
                r[0][1] = -s;
                r[1][0] = s;
                r[1][1] = c;
            }
        }
        r
    }
}

/// A coordinate axis in three dimensions, used to pick which axis a 3x3 or
/// 4x4 rotation happens around
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// The x-axis
    X,
    /// The y-axis
    Y,
    /// The z-axis
    Z,
}

/// Indexes into a row, so `matrix[i][j]` reads the entry at row `i`, column
/// `j`
impl<F, const N: usize, const M: usize> Index<usize> for Matrix<F, N, M>
where
    F: Field,
{
    type Output = [F; M];

    fn index(&self, index: usize) -> &[F; M] {
        &self.data[index]
    }
}

/// Mutably indexes into a row, so `matrix[i][j] = x` sets the entry at row
/// `i`, column `j`
impl<F, const N: usize, const M: usize> IndexMut<usize> for Matrix<F, N, M>
where
    F: Field,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

/// Elementwise matrix addition
impl<F, const N: usize, const M: usize> Add for Matrix<F, N, M>
where
    F: Field,
{
    type Output = Matrix<F, N, M>;

    fn add(self, rhs: Self) -> Self {
        Self {
            data: array::from_fn(|i| zip_map(self.data[i], rhs.data[i], |a, b| a + b)),
        }
    }
}

/// Elementwise matrix subtraction
impl<F, const N: usize, const M: usize> Sub for Matrix<F, N, M>
where
    F: Field,
{
    type Output = Matrix<F, N, M>;

    fn sub(self, rhs: Self) -> Self {
        Self {
            data: array::from_fn(|i| zip_map(self.data[i], rhs.data[i], |a, b| a - b)),
        }
    }
}

/// Scalar multiplication: multiplies every entry by `rhs`
impl<F, const N: usize, const M: usize> Mul<F> for Matrix<F, N, M>
where
    F: Field,
{
    type Output = Matrix<F, N, M>;

    fn mul(self, rhs: F) -> Self::Output {
        Self {
            data: self.data.map(|row| row.map(|c| rhs * c)),
        }
    }
}

/// Matrix multiplication: an NxM matrix times an MxP matrix gives an NxP
/// matrix - the shared dimension `M` is enforced at compile time
impl<F, const N: usize, const M: usize, const P: usize> Mul<Matrix<F, M, P>> for Matrix<F, N, M>
where
    F: Field,
{
    type Output = Matrix<F, N, P>;

    fn mul(self, rhs: Matrix<F, M, P>) -> Self::Output {
        Self::Output {
            data: array::from_fn(|i| {
                array::from_fn(|j| {
                    (0..M)
                        .map(|k| self.data[i][k] * rhs.data[k][j])
                        .fold(F::zero(), |acc, x| acc + x)
                })
            }),
        }
    }
}

#[cfg(test)]
mod test {
    use core::f64::consts::FRAC_PI_2;

    use crate::matrix::{Axis, Mat2, Matrix};
    use crate::vector::Vector;

    #[test]
    fn test_add() {
        let a = Matrix::new([[1, 2], [3, 4]]);
        let b = Matrix::new([[5, 3], [7, 7]]);

        let c = a + b;

        assert_eq!(c, Matrix::new([[6, 5], [10, 11]]));
    }

    #[test]
    fn test_rows_columns() {
        // Non-square on purpose, same reasoning as test_mul_matrix: it
        // catches a rows/columns mixup that a square matrix would hide.
        let m = Matrix::new([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(m.rows(), [Vector::new([1, 2, 3]), Vector::new([4, 5, 6])]);
        assert_eq!(
            m.columns(),
            [
                Vector::new([1, 4]),
                Vector::new([2, 5]),
                Vector::new([3, 6])
            ]
        );
    }

    #[test]
    fn test_from_rows_columns() {
        let rows = [Vector::new([1, 2, 3]), Vector::new([4, 5, 6])];
        let columns = [
            Vector::new([1, 4]),
            Vector::new([2, 5]),
            Vector::new([3, 6]),
        ];

        let expected = Matrix::new([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(Matrix::from_rows(rows), expected);
        assert_eq!(Matrix::from_columns(columns), expected);
    }

    #[test]
    fn test_id() {
        assert_eq!(Matrix::id(), Matrix::new([[1, 0], [0, 1]]));
        assert_eq!(Matrix::id(), Matrix::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]));
        assert_ne!(Matrix::id(), Matrix::new([[1, 0], [0, 0]]));
        assert_ne!(Matrix::<i32, 2, 2>::id(), Matrix::zero());
    }

    #[test]
    fn test_is_identity() {
        assert!(Matrix::<i32, 2, 2>::id().is_identity());
        assert!(!Matrix::<i32, 2, 2>::zero().is_identity());
    }

    #[test]
    fn test_is_diagonal() {
        assert!(Matrix::new([[1, 0], [0, 2]]).is_diagonal());
        assert!(!Matrix::new([[1, 1], [0, 2]]).is_diagonal());
    }

    #[test]
    fn test_is_symmetric() {
        assert!(Matrix::new([[1, 2], [2, 1]]).is_symmetric());
        assert!(!Matrix::new([[1, 2], [3, 1]]).is_symmetric());
    }

    #[test]
    fn test_is_orthogonal() {
        // A 90-degree rotation matrix is orthogonal
        assert!(Matrix::new([[0, -1], [1, 0]]).is_orthogonal());
        assert!(!Matrix::new([[1, 1], [0, 1]]).is_orthogonal());
    }

    #[test]
    fn test_trace() {
        assert_eq!(Matrix::new([[1, 2], [3, 4]]).trace(), 5);
    }

    #[test]
    fn test_mul_scalar() {
        let a = Matrix::new([[1, 2], [3, 4]]);

        assert_eq!(a * 2, Matrix::new([[2, 4], [6, 8]]));
    }

    #[test]
    fn test_mul_matrix() {
        // Non-square on purpose: N, M, P are all different (2, 3, 4), so a
        // bug that mixes up which index ranges over which dimension either
        // panics (out-of-bounds) or silently drops terms, instead of
        // happening to still work like it could with square matrices.
        let a = Matrix::new([[1, 2, 3], [4, 5, 6]]);
        let b = Matrix::new([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]);

        let c = a * b;

        assert_eq!(c, Matrix::new([[1, 2, 3, 6], [4, 5, 6, 15]]));
    }

    #[test]
    fn test_translation() {
        let t = Matrix::translation(34, 17, 82);

        assert_eq!(
            t,
            Matrix::new([[1, 0, 0, 34], [0, 1, 0, 17], [0, 0, 1, 82], [0, 0, 0, 1],])
        )
    }

    #[test]
    fn test_rotation_2d() {
        // sin(0)/cos(0) are exact in IEEE 754, so this can use assert_eq!
        // rather than a tolerance.
        assert_eq!(Matrix::<f64, 2, 2>::rotation(0.0), Matrix::id());

        // A 90-degree rotation should send X to Y.
        let close = |a: f64, b: f64| (a - b).abs() < 1e-10;
        let r = Mat2::<f64>::rotation(FRAC_PI_2);
        let m = r * Matrix::from(Vector::new([1.0, 0.0]));

        assert!(close(m[0][0], 0.0));
        assert!(close(m[1][0], 1.0));
    }

    #[test]
    fn test_rotation_zero_is_identity() {
        // sin(0)/cos(0) are exact in IEEE 754, so this can use assert_eq!
        // rather than a tolerance.
        assert_eq!(Matrix::<f64, 3, 3>::rotation(Axis::X, 0.0), Matrix::id());
        assert_eq!(Matrix::<f64, 3, 3>::rotation(Axis::Y, 0.0), Matrix::id());
        assert_eq!(Matrix::<f64, 3, 3>::rotation(Axis::Z, 0.0), Matrix::id());
    }

    #[test]
    fn test_rotation_spot_check() {
        // 90-degree rotations have exact expected results, which pins down
        // both the axis and the sign/direction of the rotation - properties
        // that an orthogonality check alone can't catch.
        let close = |a: f64, b: f64| (a - b).abs() < 1e-10;
        let close_vec = |a: Vector<f64, 3>, b: Vector<f64, 3>| {
            close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
        };
        let rotate = |axis: Axis, v: Vector<f64, 3>| {
            let r = Matrix::<f64, 3, 3>::rotation(axis, FRAC_PI_2);
            let m = r * Matrix::from(v);
            Vector::new([m[0][0], m[1][0], m[2][0]])
        };

        let x = Vector::new([1.0, 0.0, 0.0]);
        let y = Vector::new([0.0, 1.0, 0.0]);
        let z = Vector::new([0.0, 0.0, 1.0]);

        assert!(close_vec(rotate(Axis::Z, x), y));
        assert!(close_vec(rotate(Axis::X, y), z));
        assert!(close_vec(rotate(Axis::Y, z), x));
    }

    #[test]
    fn test_rotation_is_orthogonal() {
        // Property check: a rotation matrix's transpose should be its
        // inverse for any angle, not just the "nice" ones above.
        let close = |a: f64, b: f64| (a - b).abs() < 1e-10;

        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let r = Matrix::<f64, 3, 3>::rotation(axis, 0.73);
            let product = r * r.transpose();

            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(close(product[i][j], expected));
                }
            }
        }
    }
}
