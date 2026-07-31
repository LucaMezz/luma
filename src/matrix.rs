//! Defines a Matrix

use core::array;
use core::ops::{Add, Mul};

use crate::field::Field;

/// An NxM matrix (N rows, M columns). Size must be known at compile time.
#[derive(Debug, PartialEq)]
pub struct Matrix<T, const N: usize, const M: usize>
where
    T: Field,
{
    data: [[T; M]; N],
}

impl<T, const N: usize, const M: usize> Matrix<T, N, M>
where
    T: Field,
{
    /// Create a new matrix
    pub fn new(data: [[T; M]; N]) -> Self {
        Self { data }
    }

    /// Create a new diagonal matrix from diagonal entries
    pub fn diagonal(data: [T; N]) -> Self {
        Self {
            data: array::from_fn(|i| array::from_fn(|j| if i == j { data[i] } else { T::zero() })),
        }
    }

    /// Returns the zero matrix
    pub fn zero() -> Self {
        Self {
            data: [[T::zero(); M]; N],
        }
    }

    /// Returns the identity matrix
    pub fn id() -> Self {
        Self {
            data: array::from_fn(|i| array::from_fn(|j| if i == j { T::one() } else { T::zero() })),
        }
    }
}

impl<T, const N: usize, const M: usize> Add for Matrix<T, N, M>
where
    T: Field,
{
    type Output = Matrix<T, N, M>;

    fn add(self, other: Self) -> Self {
        Self {
            data: array::from_fn(|i| array::from_fn(|j| self.data[i][j] + other.data[i][j])),
        }
    }
}

impl<T, const N: usize, const M: usize> Mul<T> for Matrix<T, N, M>
where
    T: Field,
{
    type Output = Matrix<T, N, M>;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            data: array::from_fn(|i| array::from_fn(|j| rhs * self.data[i][j])),
        }
    }
}

impl<T, const N: usize, const M: usize, const P: usize> Mul<Matrix<T, M, P>> for Matrix<T, N, M>
where
    T: Field,
{
    type Output = Matrix<T, N, P>;

    fn mul(self, rhs: Matrix<T, M, P>) -> Self::Output {
        Self::Output {
            data: array::from_fn(|i| {
                array::from_fn(|j| {
                    (0..M)
                        .map(|k| self.data[i][k] * rhs.data[k][j])
                        .fold(T::zero(), |acc, x| acc + x)
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
