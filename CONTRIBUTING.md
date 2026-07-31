# Contributing to luma

Thanks for your interest in contributing! This project is primarily a
personal learning exercise, but suggestions, bug reports, and pull requests
are welcome.

By participating in this project, you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Getting started

1. Fork the repository and clone your fork.
2. Install a recent stable Rust toolchain (see `rust-version` in
   `Cargo.toml` for the minimum supported version). [rustup](https://rustup.rs/)
   is the easiest way to manage this.
3. Build the project and run the test suite:

   ```sh
   cargo build
   cargo test
   ```

## Before opening a pull request

Please make sure the following pass locally:

```sh
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

CI runs the same checks on every pull request.

## Making changes

- Keep pull requests focused on a single change where possible; it makes
  review much easier.
- Add or update tests for any behavior you add or change.
- Add doc comments (`///`) for new public items.
- Update `CHANGELOG.md` under the `[Unreleased]` section for any
  user-facing change.

## Reporting bugs and requesting features

Please use the GitHub issue templates. Include as much detail as you can:
what you expected, what happened instead, and steps to reproduce for bugs.

## Commit messages

Write clear, descriptive commit messages that explain *why* a change was
made, not just what changed.

## Questions

If anything here is unclear, feel free to open an issue to ask.
