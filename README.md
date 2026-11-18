# QTrans

QTrans is a scientific-computing package for electron-phonon coupling,
electronic and phonon spectra, and related transport properties.

The source tree is currently undergoing a package-architecture refactor. The
execution plan and acceptance criteria are recorded in
`QTRANS_REFACTORING_PLAN.md`. The documentation index is available in
`docs/INDEX.md`.

## Development installation

QTrans requires Python 3.9 or newer and a working C compiler with OpenMP
support.

```bash
python -m pip install -e .
```

The supported command-line entry points are:

```text
QTrans_run --config calculator.yaml
QTrans_prep --config preparer.yaml
```

## Tests

```bash
python -m pytest
```

Scientific regression fixtures are required before refactoring numerical
kernels. See the Phase 1 implementation report for the currently available
coverage and outstanding validation work.

## License

This project is licensed under the MIT License.

Copyright (c) 2026 Yang's Group.

You are free to use, modify, and distribute this software for academic,
research, and commercial purposes, subject to the terms and conditions
of the MIT License.

See the [LICENSE](LICENSE) file for the full license text.
