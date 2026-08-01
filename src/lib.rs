#![no_std]

//! `luma` is a linear algebra library, built for graphics applications but
//! usable generally.
//!
//! The two main types are [`matrix::Matrix`] and [`vector::Vector`], both
//! generic over any [`field::Field`] (not just floats). [`real`] adds
//! operations (square root, trigonometry, rounding) that only make sense
//! for real numbers, and [`identity`] and [`order`] provide the small
//! supporting traits `Field` and the rest are built from.
//!
//! This crate is under active development.

mod array;
pub mod field;
pub mod identity;
pub mod matrix;
pub mod order;
pub mod real;
pub mod vector;
