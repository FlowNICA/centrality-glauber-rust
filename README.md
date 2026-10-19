# centrality-glauber-rust

A Rust port of the MC-Glauber multiplicity fitter from
[CentralityFramework](https://github.com/FlowNICA/CentralityFramework/tree/parallel-params-fitter/Framework/McGlauber/centrality-master/glauber).
It reads a data multiplicity histogram and an MC-Glauber tree, fits the
multiplicity with a Glauber-based model, and writes the fit results and QA
histograms to ROOT files (`fit`). From the fit results, it then defines
centrality classes: multiplicity borders and the mean impact parameter,
`Npart` and `Ncoll` of each class (`define-centrality`).

**NOTE**: This is a very early version, developed with AI-assisted tools. It may not yet be ready for use in a formal analysis.

ROOT files are read and written with [oxiroot](https://github.com/mathieuouillon/oxiroot),
a pure-Rust implementation of the ROOT format, so no ROOT installation is needed.

## Model

Each Glauber event has a number of ancestors (independent particle sources),
`Na(f; Npart, Ncoll)`, truncated to an integer. Each ancestor produces a
random number of particles with mean `mu` and variance `mu·(1 + mu/k)`. With
`distribution: Gamma` (the default) that number follows a Gamma distribution.
With `distribution: Nbd` it follows a negative binomial distribution, so
multiplicities are integers. Either way, the total for `Na` ancestors is drawn
in a single step: the sum of `Na` NBD(mu, k) draws is exactly NBD(Na·mu, Na·k),
and similarly for Gamma. With probability `p` a second (pile-up) event is
added.

The functional form of `Na` is chosen by `mode` (case-insensitive):

| mode        | Na                         |
|-------------|----------------------------|
| `Default`   | f·Npart + (1−f)·Ncoll      |
| `PSD`       | f − Npart                  |
| `Npart`     | Npart^f                    |
| `Ncoll`     | Ncoll^f                    |
| `NpartFast` | Npart^f / 10^f             |
| `NcollFast` | Ncoll^f / 100^f            |
| `STAR`      | (1−f)·Npart/2 + f·Ncoll    |
| `HADES`     | (1 − f·Npart²)·Npart       |
| `TOF`       | f + Npart                  |

The fitter scans a grid of `(f, k, p)`. For each grid point it finds the best
`mu` with a golden-section search in `[0, x_max / Na_max(f)]`, where `x_max`
is the upper edge of the last filled data bin, minimizing a fit statistic
between the data and the model normalized in the fit range. All grid points
are fitted together, and each iteration runs in parallel on all cores.

The statistic is chosen by `fit_method`:

- `Chi2` (default, as in the C++ version): χ²/ndf with the statistical errors
  of both the data and the model. Bins with empty data are skipped.
- `Likelihood`: maximum Poisson likelihood. The fitter minimizes the
  likelihood-ratio χ² (Baker–Cousins),
  `−2 ln(L / L_saturated) = 2 Σ [m − d + d·ln(d/m)]`, divided by the number of
  bins in the fit range. This differs from −2 ln L only by a constant, so its
  minimum is the maximum-likelihood estimate, and like χ²/ndf it is ≈ 1 for a
  good fit. All bins in the fit range are used, including those with empty
  data. Model bins with no simulated events are counted as 0.1 events, so the
  logarithm stays finite. The model's own statistical fluctuations are not in
  the likelihood; with 10× more simulated events than data, they raise the
  value by roughly 10%.

### Continuous observables

The data histogram does not have to hold an integer multiplicity: it can be
any observable, e.g. a deposited energy, with any uniform bin width and lower
edge. For such data:

- Use `distribution: GammaScaleFree` in `fit`: a Gamma distribution with mean
  `mu` and variance `mu²/k` (shape `k`, scale `mu/k`; the sum over `Na`
  ancestors is Gamma(Na·k, mu/k)). The `mu·(1 + mu/k)` variance of the other
  two distributions has a Poisson term `mu`, so it depends on the units of the
  observable: it can never be narrower than `mu`. With the energy in small
  units (e.g. GeV, where `mu` is far below 1), the fit then converges to a
  wrong `mu` with a much wider per-event distribution and loses the
  correlation with `Npart`, even when its χ² looks fine. With
  `GammaScaleFree`, `k` is dimensionless and the fit gives the same result in
  any units, with `mu` in the units of the data.
- The fit range can be given in data units with `fit_min`/`fit_max` instead of
  the bin numbers `mult_min`/`mult_max` (see [Configuration](#configuration)).
- Set `min_value: 0` in `centrality` (see [Centrality classes](#centrality-classes)),
  so that the low end of the observable is classified too. The class borders
  are then written as floating point numbers.

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
cargo run --release --bin define-centrality -- config.ron
# or
target/release/fit config.ron
target/release/define-centrality config.ron
# or both steps in one go
target/release/run-all config.ron
```

`run-all` runs `fit` and then `define-centrality` with the same file, which
needs both sections. Both are read and checked before the fit starts, so an
error in the `centrality` section does not waste a fit. The centrality step
uses the QA file just written (`<out_dir>/glauber_qa.root` of the `fit`
section): a different `qa_file` is replaced, with a warning, and so is
`data_file` if it defaults to `qa_file`.

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
        // fit_min: 9.0,                              // or the fit range [fit_min, fit_max)
        // fit_max: 110.0,                            // in data units (instead of mult_*)
        fit_method: Chi2,                             // Chi2 or Likelihood
        mode: "STAR",                                 // number of ancestors parametrization
        // n_threads: 8,                              // default: all cores
        distribution: Gamma,                          // per-ancestor multiplicity: Gamma, Nbd or GammaScaleFree
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
- The data histogram must have uniform bins; any bin width and lower edge
  work, and the model histograms use the same binning.
- The fit range is either `mult_min..=mult_max`, in bins of the data histogram
  (as `fFitMinBin`/`fFitMaxBin` in the C++ version), or `[fit_min, fit_max)` in
  the units of the data, e.g. multiplicity or GeV. Setting both kinds is an
  error. Values are turned into whole bins: from the bin containing `fit_min` to
  the bin containing `fit_max` (the bin before it if `fit_max` is on a bin edge),
  within the histogram. For unit bins from 0, `fit_min: 10, fit_max: 110` gives
  bins 11..=110. The bins and their edges are printed at the start of the fit.

The Glauber tree must have the branches `B`, `Npart` and `Ncoll`. The branches
`Ecc1..5` and `Psi1..5` are optional; if they are present, the matching
`*_VS_Multiplicity` histograms are also written.

The model uses the first 10 × (data integral in the fit range) Glauber events.
If the tree has fewer events, the fit stops with an error.

### Output

Both files are written to `out_dir`:

- `fit_<f_min>_<k_min>_<k_max>_<p_min>_<first fit bin>.root`: tree `test_tree` with
  one entry per `(f, k, p)` grid point. Its branches are `f`, `mu`, `k`, `p`,
  `chi2`, `chi2_error` and `sigma`, where `mu` is the best value for that point
  and `sigma` is the per-ancestor variance of the chosen `distribution`.
  With `fit_method: Likelihood`, `chi2` holds −2 ln(L/L_sat)/ndf, and
  `chi2_error` is its error from the model's statistical fluctuations. The
  branch names are the same for both methods.
- `glauber_qa.root`, which contains:
  - the input data histogram (under its original name);
  - the best-fit model: `glaub_fit_histo` (total), `glaub_plp_histo` (pile-up
    events) and `glaub_sng_histo` (single events), all normalized to the data in
    the fit range;
  - `glaub_plp_ev1ev2`: the main-event versus pile-up-event multiplicity;
  - `B_VS_Multiplicity`, `Npart_VS_Multiplicity`, `Ncoll_VS_Multiplicity` and
    `Ecc*/Psi*_VS_Multiplicity`;
  - `fNpartHisto` and `fNcollHisto`;
  - `gamma` (or `nbd`, `gamma_scale_free`): the multiplicity from a single
    ancestor, in 100 bins up to 6 standard deviations above `mu` (unit bins for
    `nbd`);
  - tree `BestResult`: `mu`, `f`, `k`, `p`, `chi2` and `chi2_error` of the best
    fit, and the fit settings `mode`, `distribution` and `fit_method` as
    strings (used for the header of the `define-centrality` picture).

### Centrality classes

`define-centrality` combines `HistoCut.C`, `CentralityClasses.C` and
`printFinal.C`. It reads the QA file of the fit, using the `centrality` section
of the same configuration file:

```sh
cargo run --release --bin define-centrality -- config.ron
```

```ron
    centrality: (
        qa_file: "./out/glauber_qa.root",   // QA file written by fit
        data_hist: "hRefMult",              // data histogram (the QA file has a copy)
        // data_file: "~/data.root",        // default: qa_file
        out_dir: ".",
        final_file: "FINAL.root",
        n_classes: 10,                      // classes of equal width in percent, or
        // classes: [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100],   // class edges in percent
        min_value: 1.0,                     // bins below are not classified; 0 for energy
        table_formats: [Csv],               // any of Tex, Csv, Cpp
        table_name: "centrality_table",     // <table_name>.tex/.csv/.C in out_dir
        plot_file: "centrality_result.png", // PNG picture in out_dir; None: no picture
        // plot_title: "BM@N Run8, Xe+CsI", // first line of the picture's header
    ),
```

Only `qa_file` and `data_hist` are required. The classes are either
`n_classes` classes of equal width (default 10), or set by their edges in
percent with `classes`, as in
[centrality-inverse-bayes-rust](https://github.com/FlowNICA/centrality-inverse-bayes-rust).
The edges must increase within [0, 100]; setting both `classes` and `n_classes`
is an error. They do not have to cover 0–100%: with `classes: [10, 20, 40, 60]`
the 10% highest and the 40% lowest multiplicities are in no class (and the
plot draws the outer borders), while the percents still refer to all
classified events from `min_value` on.

How it works:

1. **Class borders.** The classes are defined with the single-event model
   `glaub_sng_histo`. Starting from the highest multiplicity and going down,
   each bin goes to the class that its cumulative fraction (including the bin)
   falls in, so each class holds its percent range of the single events. Bins
   that lie entirely below `min_value` are not used. The default 1 skips
   events without hits, e.g. bin 1 for unit bins from 0. For a continuous
   observable such as energy, set `min_value: 0` to use all bins, or set it to
   a threshold. The fraction of skipped events is printed. Any bin width
   works: with wider bins starting at 0, bin 1 also holds events with hits, so
   it is used, and its events without hits count toward the most peripheral
   class (a warning is printed). The pile-up dominated tail is not assigned to
   any class; it starts at the first bin where `glaub_plp_histo` exceeds
   `glaub_sng_histo`. Its single events still count toward the most central
   class. A class covers multiplicities `[MinBorder, MaxBorder)`, and each
   class starts exactly where the previous one ends.
2. **Class averages.** `<b>`, `<Npart>` and `<Ncoll>` and their RMS are
   computed from the projections of `B/Npart/Ncoll_VS_Multiplicity` over the
   multiplicity bins of each class.
3. **Ranges.** The min/max columns give the range that a sharp cut on the
   observable itself would select for the class's percents. They are computed
   over the classified events (from `min_value` on), the same events the
   percents refer to. For `b`, centrality `c` corresponds to the `c`-quantile
   of the `b` distribution. For `Npart` and `Ncoll`, which decrease with
   centrality, it is the `(1 − c)`-quantile. The values are interpolated
   linearly within bins. So the 0–10% class starts at `b` = 0 fm and at the
   largest `Npart`/`Ncoll`, the 100% edge is at the largest `b` and the
   smallest `Npart`/`Ncoll`, and neighboring classes share their edges.
   Because multiplicity fluctuates at a given `b`, a class's mean can lie
   slightly outside this range, mostly in peripheral classes.

The table is always printed to stdout. `FINAL.root` in `out_dir` contains:

- tree `Result`, one entry per class: `Ncc`, `MinPercent`, `MaxPercent`,
  `MinBorder`, `MaxBorder`, and for `B`, `Npart` and `Ncoll` the branches
  `<X>Average`, `<X>Width` (RMS), `<X>Min` and `<X>Max`. The borders are
  `Int_t` when all of them are integers (as `printFinal.C` expects); otherwise,
  e.g. for an energy, they are `Double_t`, and the tables print them unrounded
  (`minMult`/`maxMult` and `GetCentMult` in the `.C` table then use `Double_t`);
- `B/Npart/Ncoll_average_VS_Centrality`: the averages versus centrality, with
  the RMS as bin errors (with variable bins at the `classes` edges). The original `printFinal.C` can read this file;
- `B/Npart/Ncoll_VS_CentralityClass <min>%-<max>%`: the distributions in each
  class, plus `... 0%-100%` for all events;
- `CentralityClass_Fit <min>%-<max>%` and `CentralityClass <min>%-<max>%`: the
  model and data multiplicity in each class;
- `Centrality_vs_Multiplicity`: the upper percent of the class of each
  multiplicity bin.

`plot_file` (in `out_dir`) is a PNG picture of the result, in the style of the
ROOT canvases of the C++ framework (and of
[centrality-inverse-bayes-rust](https://github.com/FlowNICA/centrality-inverse-bayes-rust)):
the data and the best fit (`glaub_fit_histo`) on a logarithmic scale, with the
class borders and labels, and the ratio data / fit below. The header shows
`plot_title`, `Na` and the per-ancestor distribution as formulas, and the fitted
parameters with χ²/ndf (or −2lnλ/ndf). QA files from older versions have no
fit settings, so only the parameters are shown. Under the legend, `Eff` shows
the efficiency: the integral of the data divided by the integral of the fit,
both over the multiplicities that the class percents refer to (from
`min_value` on). The fit is normalized to the data in the fit range and
extrapolated below it, so `Eff` is the fraction of all events that were
recorded, as with `normalization: Free` in centrality-inverse-bayes-rust. Its
accuracy depends on how well the model describes the data below the fit range.
`define-centrality` also prints it. The picture is drawn in pure
Rust ([plotters](https://crates.io/crates/plotters)) with the embedded
Liberation Sans font (`data/fonts`, SIL Open Font License), so no system
libraries are needed.

## Library use

The fitter can also be used from Rust, with the configuration built in code:

```rust
use centrality_glauber_rust::{FitConfig, Mode};

let config = FitConfig::builder()
    .glauber("glauber.root", "nt_Au3_Au3")
    .data("data.root", "hRefMult")
    .k_range(0.5, 1.0, 0.01)
    .fit_range(10, 110)              // bins; or .fit_value_range(9., 110.) in data units
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
- There is no `bin_size` setting. In the C++ version it only set the bin width
  of the `fNpartHisto` and `fNcollHisto` QA histograms, never the fit, and the
  range `[0, int(max))` was split into `int(max / bin_size)` bins, so the width
  was not `bin_size` unless it divided the maximum evenly, and the events at the
  maximum ended up in the overflow bin. Here, these histograms have unit bins
  from 0 for integer `Npart`/`Ncoll` (1/1000 of the maximum otherwise), with
  enough bins to include the maximum.
- `distribution: Nbd` samples NBD multiplicities in the fit and in the `nbd`
  histogram. In the C++ version, `UseNbd()` never changes the fit, which always
  samples Gamma. On `master`, it only switches the per-ancestor histogram
  (`SetNBDhist`) to NBD, and on `parallel-params-fitter` it only renames that
  histogram. Also, the C++ histogram's `std::negative_binomial_distribution`
  truncates `k` to an integer, so `k < 1` gives a degenerate distribution. This
  port uses the real value of `k`. With `distribution: Gamma` (the default),
  the results are the same as before.
- `distribution: GammaScaleFree` is new: the C++ fit only has the
  `mu·(1 + mu/k)` Gamma, whose result depends on the units of a continuous
  observable (see [Model](#model)).
- `define-centrality` fixes several bugs in the macros it replaces. Instead of
  ROOT canvases and PDFs, it draws one PNG picture (`plot_file`):
  - `HistoCut.C` assigns bins with a running sum that it resets to 0 at each
    class boundary. The boundary bin's events are therefore never counted, and
    the bin at the start of the most peripheral class belongs to no class. Its
    `MinBorder`/`MaxBorder` are bin centers on an axis that doesn't quite match
    the model's, and the data-per-class loop mishandles bins without model
    events. Here, each bin goes to the class its cumulative fraction falls in,
    and the classes are contiguous.
  - `HistoCut.C` always skips bin 1, which is multiplicity 0 only for unit
    bins: with wider bins it drops events with hits from all class percents,
    and with narrower bins it keeps some events without hits. Here, the bins
    below `min_value` are skipped, which also works for continuous observables
    (`min_value: 0`).
  - The C++ macros store the class borders as integers. Here, non-integer
    borders (e.g. of an energy) are kept as floating point numbers.
  - `CentralityClasses.C` passes the multiplicity borders to `ProjectionY` as
    bin numbers, which is off by one bin, and it includes the `MaxBorder` bin in
    both neighboring classes. Here, the projections cover exactly the bins of
    each class.
  - `printFinal.C` takes the min/max columns from a degree-5 polynomial fitted
    to the class averages versus centrality. At the edges that is an
    extrapolation: `b_min` of the most central class is not 0 (1.86 fm instead
    of 0 on the 3 GeV Au+Au data), and `Npart_max` stays well below the largest
    `Npart`. Here, they come from sharp-cut quantiles (see above). Running
    `printFinal.C` on `FINAL.root` still gives its own polynomial values in
    these columns.
  - `printFinal.C` drops classes with a zero average from the averages, but not
    from the percents and borders, which misaligns the table rows. Here, empty
    classes are removed consistently, with a warning. Its C++ output labels the
    Npart/Ncoll values at the lower and upper percent as `min`/`max`, which is
    reversed for quantities that decrease with centrality. Here, `min`/`max`
    are the actual minimum and maximum. The CSV has no trailing commas, and its
    first line is a `#` comment. `Centrality_vs_Multiplisity` is renamed
    `Centrality_vs_Multiplicity`.
- These C++ behaviors are kept on purpose:
  - The model is normalized over the fit range without its first bin
    (`mult_min+1..=mult_max`), while χ² uses the whole fit range
    (`mult_min..=mult_max`). The same holds for a range given by
    `fit_min`/`fit_max`.
  - The upper limit of the `mu` search comes from the last filled data bin. If
    the data histogram's range cuts off the multiplicity tail, the true `mu` can
    lie outside the search range.

## Development

```sh
cargo test --release                        # unit + end-to-end tests
cargo test --release --test end_to_end wide_data_bins   # a single test
cargo clippy --all-targets -- -D warnings
cargo fmt
```

The end-to-end tests make a toy Glauber tree and data generated with known
parameters, run the full fit and `define-centrality`, and check that the
parameters are recovered and that the output files can be read back. They
also cover data with wider bins and a continuous observable fitted in GeV and
MeV, which must give the same result.
