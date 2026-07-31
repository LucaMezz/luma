//! Defines a Matrix

use core::array;
use core::ops::{Add, Index, Mul, Sub};

use crate::array::zip_map;
use crate::field::Field;

/// An NxM matrix (N rows, M columns). Size must be known at compile time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix<F, const N: usize, const M: usize>
where
    F: Field,
{
    data: [[F; M]; N],
}

impl<F, const N: usize, const M: usize> Matrix<F, N, M>
where
    F: Field,
{
    /// Create a new matrix
    pub fn new(data: [[F; M]; N]) -> Self {
        Self { data }
    }

    /// Create a new diagonal matrix from diagonal entries
    pub fn diagonal(data: [F; N]) -> Self {
        Self {
            data: array::from_fn(|i| array::from_fn(|j| if i == j { data[i] } else { F::zero() })),
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
}

impl<F, const N: usize, const M: usize> Index<usize> for Matrix<F, N, M>
where
    F: Field,
{
    type Output = [F; M];

    fn index(&self, index: usize) -> &[F; M] {
        &self.data[index]
    }
}

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
    use crate::matrix::Matrix;

    #[test]
    fn test_add() {
        let a = Matrix::new([[1, 2], [3, 4]]);
        let b = Matrix::new([[5, 3], [7, 7]]);

        let c = a + b;

        assert_eq!(c, Matrix::new([[6, 5], [10, 11]]));
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
}
