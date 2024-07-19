# SIMD Vec

An easy way to make/work with arbitrary length Vecs, with the benefits of SIMD.

Very much a WIP, a lot of methods still need implementing, and documentation is currently nonexistant. Contributions are welcome!

## Examples

```rust
let vec = vec![1, 2, 3]; // or any other Vec/slice of Simd-compatible types
let simd_vec: SimdVec<_, 16> = vec.into(); // or SimdVec::from([1, 2, 3]), ...

// do some operations
assert_eq!(&simd_vec + SimdVec::from([5, 6, 7, 6, 5, 4]), SimdVec::from([6, 8, 10, 6, 5, 4]));
assert_eq!(simd_vec.reduce_sum(), 6);

// ... and more, mostly replicated from methods that Simd types have.
```
