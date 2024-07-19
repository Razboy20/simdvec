#![feature(portable_simd)]

use core::hash::{Hash, Hasher};
use core::slice;
use std::{
  mem::MaybeUninit,
  ops::*,
  simd::{num::SimdInt, LaneCount, Simd, SimdElement, SupportedLaneCount},
};

// non-resizable SIMD processing vectors
// TODO: rewrite impls to use SimdInt and SimdUint traits
#[derive(Default, Clone)]
pub struct SimdVec<T, const LANES: usize>
where
  LaneCount<LANES>: SupportedLaneCount,
  T: SimdElement,
{
  buf: Vec<Simd<T, LANES>>,
  // len: usize,
}

impl<T, const N: usize> SimdVec<T, N>
where
  LaneCount<N>: SupportedLaneCount,
  T: SimdElement,
{
  pub const LANES: usize = N;

  pub fn iter(&self) -> slice::Iter<'_, Simd<T, N>> {
    self.buf.iter()
  }

  pub fn iter_mut(&mut self) -> slice::IterMut<'_, Simd<T, N>> {
    self.buf.iter_mut()
  }

  /// Provides upper bound on the number of elements that can be stored in the vector
  pub fn capacity(&self) -> usize {
    self.buf.len() * Self::LANES
  }
}

impl<T, const N: usize> SimdVec<T, N>
where
  LaneCount<N>: SupportedLaneCount,
  T: SimdElement + Default,
{
  pub fn with_capacity(capacity: usize) -> Self {
    let size = (capacity + Self::LANES - 1) / Self::LANES;

    Self {
      buf: vec![Simd::default(); size],
      // len: capacity,
    }
  }

  pub fn with_capacity_value(capacity: usize, value: T) -> Self {
    let size = (capacity + Self::LANES - 1) / Self::LANES;

    Self {
      buf: vec![Simd::splat(value); size],
    }
  }

  pub fn into_vec(self) -> Vec<T> {
    self.into()
  }
}

impl<T, const N: usize> From<&[T]> for SimdVec<T, N>
where
  LaneCount<N>: SupportedLaneCount,
  T: SimdElement + Default,
{
  fn from(slice: &[T]) -> Self {
    let capacity = (slice.len() + Self::LANES - 1) / Self::LANES;

    let mut buf = vec![MaybeUninit::uninit(); capacity];
    // let len = slice.len();

    let mut slice_iter = buf.iter_mut().zip(slice.chunks(Self::LANES)).peekable();

    while let Some((buf, slice)) = slice_iter.next() {
      let el = if slice_iter.peek().is_some() {
        Simd::from_slice(slice)
      } else {
        Simd::load_or_default(slice)
      };

      buf.write(el);
    }

    let buf = unsafe { core::mem::transmute(buf) };

    Self { buf }
  }
}

impl<T, const N: usize, const U: usize> From<[T; U]> for SimdVec<T, N>
where
  LaneCount<N>: SupportedLaneCount,
  T: SimdElement + Default,
{
  fn from(slice: [T; U]) -> Self {
    Self::from(&slice as &[T])
  }
}

impl<T, const N: usize> From<Vec<T>> for SimdVec<T, N>
where
  LaneCount<N>: SupportedLaneCount,
  T: SimdElement + Default,
{
  fn from(vec: Vec<T>) -> Self {
    Self::from(vec.as_slice())
  }
}

impl<T, const N: usize> From<SimdVec<T, N>> for Vec<T>
where
  LaneCount<N>: SupportedLaneCount,
  T: SimdElement + Default,
{
  fn from(vec: SimdVec<T, N>) -> Self {
    let capacity = vec.capacity();
    // let slice: &[T] = unsafe { core::mem::transmute(vec.buf.as_slice()) };
    // let mut vec = Vec::from(slice);
    // unsafe { vec.set_len(capacity) };
    // assert_eq!(capacity, vec.len());
    // vec
    let mut vec: Vec<T> = unsafe { core::mem::transmute(vec.buf) };
    unsafe { vec.set_len(capacity) };
    vec
  }
}

impl<T, const N: usize> FromIterator<Simd<T, N>> for SimdVec<T, N>
where
  LaneCount<N>: SupportedLaneCount,
  T: SimdElement,
{
  fn from_iter<I: IntoIterator<Item = Simd<T, N>>>(iter: I) -> Self {
    Self {
      buf: Vec::from_iter(iter),
    }
  }
}

impl<'a, T, const N: usize> FromIterator<&'a Simd<T, N>> for SimdVec<T, N>
where
  LaneCount<N>: SupportedLaneCount,
  T: SimdElement,
{
  fn from_iter<I: IntoIterator<Item = &'a Simd<T, N>>>(iter: I) -> Self {
    Self {
      buf: Vec::from_iter(iter.into_iter().map(|m| m.clone())),
    }
  }
}

impl<T, const N: usize, I: slice::SliceIndex<[T]>> Index<I> for SimdVec<T, N>
where
  LaneCount<N>: SupportedLaneCount,
  T: SimdElement,
{
  type Output = I::Output;

  fn index(&self, index: I) -> &Self::Output {
    let slice = unsafe {
      slice::from_raw_parts::<T>(core::mem::transmute(self.buf.as_ptr()), self.capacity())
    };
    slice.index(index)
  }
}

impl<T, const N: usize, I: slice::SliceIndex<[T]>> IndexMut<I> for SimdVec<T, N>
where
  LaneCount<N>: SupportedLaneCount,
  T: SimdElement,
{
  fn index_mut(&mut self, index: I) -> &mut Self::Output {
    let slice = unsafe {
      slice::from_raw_parts_mut::<T>(core::mem::transmute(self.buf.as_mut_ptr()), self.capacity())
    };
    slice.index_mut(index)
  }
}

impl<'a, T, const N: usize> PartialEq for SimdVec<T, N>
where
  LaneCount<N>: SupportedLaneCount,
  T: SimdElement + PartialEq,
{
  fn eq(&self, other: &Self) -> bool {
    self.iter().zip(other.iter()).all(|(a, b)| a == b)
  }
}

impl<'a, T, const N: usize> Eq for SimdVec<T, N>
where
  LaneCount<N>: SupportedLaneCount,
  T: SimdElement + Eq,
{
}

impl<'a, T, const N: usize> Hash for SimdVec<T, N>
where
  LaneCount<N>: SupportedLaneCount,
  T: SimdElement + PartialEq,
  Vec<Simd<T, N>>: Hash,
  [T]: Hash,
{
  #[inline]
  fn hash<H: Hasher>(&self, state: &mut H) {
    // self.buf.hash(state);
    // let vec: &[T] = unsafe { core::mem::transmute(self.buf.as_slice()) };
    let vec: &[T] =
      unsafe { slice::from_raw_parts(core::mem::transmute(self.buf.as_ptr()), self.capacity()) };
    // unsafe { vec.set_len(self.capacity()) };
    vec.hash(state);
  }
}

macro_rules! deref_ops {
  ($($trait:ident: fn $call:ident),*) => {
    $(
      // deref left hand side
      impl<T, const N: usize> $trait<SimdVec<T, N>> for &SimdVec<T, N>
      where
        LaneCount<N>: SupportedLaneCount,
        T: SimdElement + $trait,
        Simd<T, N>: $trait<Output = Simd<T, N>>,
      {
        type Output = SimdVec<T, N>;

        fn $call(self, rhs: SimdVec<T, N>) -> Self::Output {
          (*self)
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| a.$call(b))
            .collect()
        }
      }

      // deref right hand ride
      impl<T, const N: usize> $trait<&SimdVec<T, N>> for SimdVec<T, N>
      where
        LaneCount<N>: SupportedLaneCount,
        T: SimdElement + $trait,
        Simd<T, N>: $trait<Output = Simd<T, N>>,
      {
        type Output = SimdVec<T, N>;

        fn $call(self, rhs: &SimdVec<T, N>) -> Self::Output {
          self
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| a.$call(b))
            .collect()
        }
      }

      // deref both sides
      impl<'lhs, 'rhs, T, const N: usize> $trait<&'rhs SimdVec<T, N>> for &'lhs SimdVec<T, N>
      where
        LaneCount<N>: SupportedLaneCount,
        T: SimdElement + $trait,
        Simd<T, N>: $trait<Output = Simd<T, N>>,
      {
        type Output = SimdVec<T, N>;

        fn $call(self, rhs: &'rhs SimdVec<T, N>) -> Self::Output {
          (*self)
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| a.$call(b))
            .collect()
        }
      }

      // both sides are owned
      impl<T, const N: usize> $trait<SimdVec<T, N>> for SimdVec<T, N>
      where
        LaneCount<N>: SupportedLaneCount,
        T: SimdElement + $trait,
        Simd<T, N>: $trait<Output = Simd<T, N>>,
      {
        type Output = SimdVec<T, N>;

        fn $call(self, rhs: SimdVec<T, N>) -> Self::Output {
          self
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| a.$call(b))
            .collect()
        }
      }
    )*
  };
}

macro_rules! unary_ops {
  ($($trait:ident: fn $call:ident),*) => {
    $(
      impl<T, const N: usize> $trait for SimdVec<T, N>
      where
        LaneCount<N>: SupportedLaneCount,
        T: SimdElement + $trait,
        Simd<T, N>: $trait<Output = Simd<T, N>>,
      {
        type Output = SimdVec<T, N>;

        fn $call(self) -> Self::Output {
          self
            .iter()
            .map(|a| a.$call())
            .collect()
        }
      }
    )*
  };
}

macro_rules! propagate_ops {
  ($(fn $call:ident),*) => {
    $(
      impl<T, const N: usize> SimdVec<T, N>
      where
        LaneCount<N>: SupportedLaneCount,
        T: SimdElement,
        Simd<T, N>: SimdInt
      {
        pub fn $call(&self) -> Self {
          self
            .iter()
            .map(|a| a.$call())
            .collect()
        }
      }
    )*
  };
}

macro_rules! scalar_ops {
  ($($trait:ident: fn $call:ident => $op:tt),*) => {
    $(
      impl<T, const N: usize> SimdVec<T, N>
      where
        LaneCount<N>: SupportedLaneCount,
        T: SimdElement + $trait<Output = T> + Default,
      {
        pub fn $call(&self) -> T
        where
          Simd<T, N>: SimdInt<Scalar = T>,
        {
          self
            .buf
            .iter()
            .fold(Default::default(), |acc, x| acc $op x.$call())
        }
      }
    )*
  };
}

deref_ops! {
  Add: fn add,
  Mul: fn mul,
  Sub: fn sub,
  Div: fn div,
  Rem: fn rem,
  BitAnd: fn bitand,
  BitOr: fn bitor,
  BitXor: fn bitxor,
  Shl: fn shl,
  Shr: fn shr
}

unary_ops! {
  Not: fn not,
  Neg: fn neg
}

propagate_ops! {
  fn abs,
  fn saturating_abs,
  fn saturating_neg,
  // fn is_positive,
  // fn is_negative,
  fn signum,
  // fn reduce_max,
  // fn reduce_min,
  fn swap_bytes,
  fn reverse_bits
  // fn leading_zeros,
  // fn trailing_zeros
  // fn leading_ones,
  // fn trailing_ones
}

scalar_ops! {
  Add: fn reduce_sum => +,
  Mul: fn reduce_product => *,
  // Max: fn reduce_max => max,
  // Min: fn reduce_min => min,
  BitAnd: fn reduce_and => &,
  BitOr: fn reduce_or => |,
  BitXor: fn reduce_xor => ^
}
