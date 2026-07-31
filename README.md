# luma

A linear algebra library written from scratch in Rust, built with graphics
applications in mind but usable as a general-purpose math library.

This is a learning project: the goal is to implement the core building blocks
of linear algebra (vectors, matrices, and the operations built on top of
them) rather than depending on an existing crate. Expect the API to change
as it grows.

## Status

Early scaffolding — no public API yet. See [CHANGELOG.md](CHANGELOG.md) for
progress.

## Usage

Once published, add it to your `Cargo.toml`:

```toml
[dependencies]
luma = "0.1"
```

Until then, depend on it directly from git or a local path:

```toml
[dependencies]
luma = { git = "https://github.com/<your-username>/luma" }
```

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request, and note that this project follows a
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

Licensed under the [MIT license](LICENSE).
