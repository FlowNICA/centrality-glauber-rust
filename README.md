# centrality-glauber-rust

A Rust port of the MC-Glauber multiplicity fitter from
[CentralityFramework](https://github.com/FlowNICA/CentralityFramework/tree/parallel-params-fitter/Framework/McGlauber/centrality-master/glauber).
It reads a data multiplicity histogram and an MC-Glauber tree, fits the
multiplicity with a Glauber-based model, and writes the fit results and QA
histograms to ROOT files.

**NOTE**: This is a very early version, developed with AI-assisted tools. It may not yet be ready for use in a formal analysis.

ROOT files are read and written with [oxiroot](https://github.com/mathieuouillon/oxiroot),
a pure-Rust implementation of the ROOT format, so no ROOT installation is needed.

## Model

Each Glauber event has a number of ancestors (independent particle sources),
`Na(f; Npart, Ncoll)`. Each ancestor produces a Gamma-distributed number of
particles with mean `mu` and NBD-like width parameter `k`. With probability `p`
a second (pile-up) event is added. The functional form of `Na` is chosen by `mode`:

| mode        | Na                         |
|-------------|----------------------------|
| `Default`   | f·Npart + (1−f)·Ncoll      |
| `PSD`       | f − Npart                  |
| `Npart`     | Npart^f                    |
| `Ncoll`     | Ncoll^f                    |
| `STAR`      | (1−f)·Npart/2 + f·Ncoll    |
| `HADES`     | (1 − f·Npart²)·Npart       |

The fitter scans a grid of `(f, k, p)`. For each grid point it finds the best
`mu` with a golden-section search in `[0, max_multiplicity / Na_max(f)]`,
minimizing χ²/ndf between the data and the model normalized in the fit range.
All grid points are fitted together, and each iteration runs in parallel on all
cores.

## Requirements

- Rust 1.95 or newer (edition 2024)

## Building

```sh
cargo build --release
```

## Usage

Each step of the analysis is a separate executable in `src/bin/`. All of them
read one shared configuration file, [`config.ron`](config.ron), with a section
per executable:

```sh
cargo run --release --bin fit -- config.ron
# or
target/release/fit config.ron
```

### Configuration

The `fit` section is a port of the original `config.c`:

```ron
(
    fit: (
        glauber_file: "~/input_glauber_file.root",    // MC-Glauber input (with TTree)
        glauber_tree: "glauber",                      // name of the MC-Glauber tree
        data_file: "~/input_data_file.root",          // data input
        data_hist: "hRefMult",                        // name of the data histogram
        out_dir: ".",                                 // output directory
        n_iter: 20,                                   // golden-section iterations for mu
        f: (min: 0.1, max: 0.1, step: 0.01),          // parameter scans
        k: (min: 0.5, max: 1.0, step: 0.01),
        p: (min: 0.001, max: 0.05, step: 0.001),
        mult_min: 10,                                 // fit range, data histogram bins
        mult_max: 110,
        bin_size: 1.0,                                // bin width of the Npart/Ncoll histograms
        mode: "STAR",                                 // Number of ancestors parametrization
        // n_threads: 8,                              // default: all cores
        distribution: Gamma,                          // Gamma or Nbd; only sets the output histogram name
        // seed: 42,                                  // fixed seed for reproducible results
    ),
)
```

- Only the input files and the tree and histogram names are required. Any other
  field you leave out uses the value shown above.
- A misspelled field or an invalid value stops the program with the line and
  column of the problem.
- A leading `~/` in a path is expanded to `$HOME`.
- Optional values are written directly, without `Some(...)`.
- A parameter is fixed when `min == max`.

The Glauber tree must have the branches `B`, `Npart` and `Ncoll`. The branches
`Ecc1..5` and `Psi1..5` are optional; if they are present, the matching
`*_VS_Multiplicity` histograms are also written.

The model uses the first 10 × (data integral in the fit range) Glauber events.
If the tree has fewer events, the fit stops with an error.

### Output

Both files are written to `out_dir`:

- `fit_<f_min>_<k_min>_<k_max>_<p_min>_<mult_min>.root`: tree `test_tree` with
  one entry per `(f, k, p)` grid point. Its branches are `f`, `mu`, `k`, `p`,
  `chi2`, `chi2_error` and `sigma`, where `mu` is the best value for that point.
- `glauber_qa.root`, which contains:
  - the input data histogram (under its original name);
  - the best-fit model: `glaub_fit_histo` (total), `glaub_plp_histo` (pile-up
    events) and `glaub_sng_histo` (single events), all normalized to the data in
    the fit range;
  - `glaub_plp_ev1ev2`: the main-event versus pile-up-event multiplicity;
  - `B_VS_Multiplicity`, `Npart_VS_Multiplicity`, `Ncoll_VS_Multiplicity` and
    `Ecc*/Psi*_VS_Multiplicity`;
  - `fNpartHisto` and `fNcollHisto`;
  - `gamma` (or `nbd`): the multiplicity from a single ancestor;
  - tree `BestResult`: `mu`, `f`, `k`, `p`, `chi2` and `chi2_error` of the best fit.

## Library use

The fitter can also be used from Rust, with the configuration built in code:

```rust
use centrality_glauber_rust::{FitConfig, Mode};

let config = FitConfig::builder()
    .glauber("glauber.root", "nt_Au3_Au3")
    .data("data.root", "hRefMult")
    .k_range(0.5, 1.0, 0.01)
    .fit_range(10, 110)
    .mode(Mode::Star)
    .seed(42)
    .build()?;
let result = centrality_glauber_rust::run(config)?;   // fits and writes both output files
println!("mu = {}, chi2/ndf = {}", result.best.mu, result.chi2);
```

For finer control, use `Fitter::fit`, `Fitter::model_histograms` and the
functions in `output`. To add a section to the shared configuration for a new
executable, read it with `config_file::read_section::<YourConfig>(path, "name")`.

## Differences from the C++ version

- Random numbers come from a different generator, so results agree with the C++
  version only statistically. With `seed` set and the same `n_threads`, results
  are reproducible.
- Nothing is drawn: there is no `glauber.pdf` or canvas. `glauber_qa.root` is
  written to `out_dir` instead of the current directory.
- `GetModelHisto` is not ported.
- These C++ behaviors are kept on purpose:
  - The model is normalized over bins `mult_min+1..=mult_max`, while χ² uses
    bins `mult_min..=mult_max`.
  - `UseNbd` (`distribution: Nbd`) only renames the output histogram; the
    sampling is always Gamma.
  - The upper limit of the `mu` search comes from the last filled data bin. If
    the data histogram's range cuts off the multiplicity tail, the true `mu` can
    lie outside the search range.

## Development

```sh
cargo test --release                        # unit + end-to-end tests
cargo clippy --all-targets -- -D warnings
cargo fmt
```

The end-to-end tests make a toy Glauber tree and data generated with known
parameters, run the full fit, and check that the parameters are recovered and
that the output files can be read back.
