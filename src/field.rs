//! Defines a Field

use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::identity::{One, Zero};

/// A field
pub trait Field:
    Sized
    + Copy
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Zero
    + One
{
}

impl<
    T: Sized
        + Copy
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Zero
        + One,
> Field for T
{
}

macro_rules! impl_zero_one {
    ($($t:ty => $zero:expr, $one:expr);* $(;)?) => {
        $(
            impl Zero for $t {
                fn zero() -> Self {
                    $zero
                }
            }

            impl One for $t {
                fn one() -> Self {
                    $one
                }
            }
        )*
    };
}

impl_zero_one! {
    i8 => 0, 1;
    i16 => 0, 1;
    i32 => 0, 1;
    i64 => 0, 1;
    i128 => 0, 1;
    isize => 0, 1;
    u8 => 0, 1;
    u16 => 0, 1;
    u32 => 0, 1;
    u64 => 0, 1;
    u128 => 0, 1;
    usize => 0, 1;
    f32 => 0.0, 1.0;
    f64 => 0.0, 1.0;
}
