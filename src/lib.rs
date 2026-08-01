#![no_std]

//! `luma` is a linear algebra library, built for graphics applications but
//! usable generally.
//!
//! The two main types are [`Matrix`] and [`Vector`], both generic over any
//! [`Field`] (not just floats) — these are re-exported at the crate root
//! since `matrix::Matrix`, `vector::Vector`, and `field::Field` would just
//! repeat themselves. [`real`] adds operations (square root, trigonometry,
//! rounding) that only make sense for real numbers, and [`identity`] and
//! [`order`] provide the small supporting traits `Field` and the rest are
//! built from.
//!
//! This crate is under active development.

mod array;
mod field;
pub mod identity;
mod matrix;
pub mod order;
pub mod real;
mod vector;

pub use field::Field;
pub use matrix::{Axis, Mat2, Mat3, Mat4, Matrix};
pub use vector::{Vec2, Vec3, Vec4, Vector};
