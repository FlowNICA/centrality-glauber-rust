//! Centrality classes from the fit results: port of `HistoCut.C`,
//! `CentralityClasses.C` and `printFinal.C` of the CentralityFramework.
//!
//! The classes are defined with the fitted model of single (non pile-up)
//! events, `glaub_sng_histo`: counting from the highest multiplicity down,
//! each class holds `1 / n_classes` of the single-event integral. For each
//! class the Glauber observables `B`, `Npart` and `Ncoll` are averaged over
//! the model events in its multiplicity range (`<name>_VS_Multiplicity`), and
//! their range is estimated from a polynomial fit of the averages versus
//! centrality.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use oxiroot::prelude::*;
use serde::Deserialize;

use crate::config::expand_home;
use crate::config_file;
use crate::error::{Error, Result};

/// Maximum number of centrality classes.
const MAX_CLASSES: usize = 1000;
/// Degree of the polynomial fitted to the class averages versus centrality
/// (`pol5` as in `printFinal.C`); lower if there are fewer classes.
const POLY_DEGREE: usize = 5;
/// Glauber observables averaged per class: (name, axis label).
const OBSERVABLES: [(&str, &str); 3] =
    [("B", "B, fm"), ("Npart", "N_{part}"), ("Ncoll", "N_{coll}")];

/// Output format of the centrality table, in addition to the plain table
/// printed to stdout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum TableFormat {
    /// LaTeX document with the table (`.tex`).
    Tex,
    /// Comma separated values (`.csv`).
    Csv,
    /// ROOT macro with arrays of the class borders and averages and a
    /// `GetCentMult(mult)` lookup function (`.C`).
    Cpp,
}

impl TableFormat {
    pub fn extension(self) -> &'static str {
        match self {
            TableFormat::Tex => "tex",
            TableFormat::Csv => "csv",
            TableFormat::Cpp => "C",
        }
    }
}

/// Configuration of the centrality determination. Create it with
/// [`CentralityConfig::builder`] or read it from a RON file.
#[derive(Debug, Clone, PartialEq)]
pub struct CentralityConfig {
    /// QA file written by the fit (`glauber_qa.root`).
    pub qa_file: PathBuf,
    /// ROOT file with the data multiplicity histogram.
    pub data_file: PathBuf,
    /// Name of the data multiplicity histogram.
    pub data_hist: String,
    /// Directory for the output files.
    pub out_dir: PathBuf,
    /// Name of the output ROOT file in `out_dir`.
    pub final_file: String,
    /// Number of centrality classes of equal width in percent.
    pub n_classes: usize,
    /// Formats of the table files written in addition to stdout.
    pub table_formats: Vec<TableFormat>,
    /// File name, without extension, of the table files in `out_dir`.
    pub table_name: String,
}

impl CentralityConfig {
    /// Name of the section of the shared RON file read by `bin/define-centrality`.
    pub const RON_SECTION: &'static str = "centrality";

    pub fn builder() -> CentralityConfigBuilder {
        CentralityConfigBuilder::default()
    }

    /// Reads and validates the [`RON_SECTION`](Self::RON_SECTION) section of
    /// a RON configuration file, see `config.ron`.
    pub fn from_ron_file(path: impl AsRef<Path>) -> Result<Self> {
        let builder: CentralityConfigBuilder =
            config_file::read_section(path.as_ref(), Self::RON_SECTION)?;
        builder.build()
    }

    /// Path of the output ROOT file.
    pub fn final_path(&self) -> PathBuf {
        self.out_dir.join(&self.final_file)
    }

    /// Path of the table file in the given format.
    pub fn table_path(&self, format: TableFormat) -> PathBuf {
        self.out_dir
            .join(format!("{}.{}", self.table_name, format.extension()))
    }
}

/// Builder for [`CentralityConfig`]; the QA file and the data histogram name
/// are required. It can also be deserialized from the `centrality` section of
/// a RON file; omitted fields keep their defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CentralityConfigBuilder {
    qa_file: Option<PathBuf>,
    data_file: Option<PathBuf>,
    data_hist: Option<String>,
    out_dir: PathBuf,
    final_file: String,
    n_classes: usize,
    table_formats: Vec<TableFormat>,
    table_name: String,
}

impl Default for CentralityConfigBuilder {
    fn default() -> Self {
        Self {
            qa_file: None,
            data_file: None,
            data_hist: None,
            out_dir: PathBuf::from("."),
            final_file: "FINAL.root".into(),
            n_classes: 10,
            table_formats: Vec::new(),
            table_name: "centrality_table".into(),
        }
    }
}

impl CentralityConfigBuilder {
    /// Parses the `centrality` section of a RON configuration (other
    /// sections are ignored).
    pub fn from_ron_str(text: &str) -> Result<Self> {
        config_file::parse_section(text, CentralityConfig::RON_SECTION)
            .map_err(|e| Error::Config(e.to_string()))
    }

    /// QA file of the fit (`glauber_qa.root`).
    pub fn qa_file(mut self, file: impl Into<PathBuf>) -> Self {
        self.qa_file = Some(file.into());
        self
    }

    /// Data multiplicity histogram `hist` in `file`. Without a file, the
    /// histogram is read from the QA file, which contains a copy of it.
    pub fn data(mut self, file: Option<PathBuf>, hist: impl Into<String>) -> Self {
        self.data_file = file;
        self.data_hist = Some(hist.into());
        self
    }

    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = dir.into();
        self
    }

    pub fn final_file(mut self, name: impl Into<String>) -> Self {
        self.final_file = name.into();
        self
    }

    pub fn n_classes(mut self, n_classes: usize) -> Self {
        self.n_classes = n_classes;
        self
    }

    pub fn table_formats(mut self, formats: Vec<TableFormat>) -> Self {
        self.table_formats = formats;
        self
    }

    pub fn table_name(mut self, name: impl Into<String>) -> Self {
        self.table_name = name.into();
        self
    }

    pub fn build(self) -> Result<CentralityConfig> {
        let missing = |what: &str| Error::Config(format!("{what} is not set"));
        let qa_file = expand_home(&self.qa_file.ok_or_else(|| missing("QA file"))?);
        let data_hist = self.data_hist.ok_or_else(|| missing("data histogram"))?;
        let data_file = self
            .data_file
            .map_or_else(|| qa_file.clone(), |f| expand_home(&f));
        if !(1..=MAX_CLASSES).contains(&self.n_classes) {
            return Err(Error::Config(format!(
                "n_classes must be in [1, {MAX_CLASSES}], got {}",
                self.n_classes
            )));
        }
        if self.final_file.is_empty() || self.table_name.is_empty() {
            return Err(Error::Config(
                "final_file and table_name must not be empty".into(),
            ));
        }
        Ok(CentralityConfig {
            qa_file,
            data_file,
            data_hist,
            out_dir: expand_home(&self.out_dir),
            final_file: self.final_file,
            n_classes: self.n_classes,
            table_formats: self.table_formats,
            table_name: self.table_name,
        })
    }
}

/// Multiplicity bins of the centrality classes.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassBins {
    /// Inclusive range of histogram bins of each class, from the most
    /// central; `None` if the class holds no bin (a single bin holds more
    /// than a class).
    pub ranges: Vec<Option<(usize, usize)>>,
    /// First bin of the pile-up dominated tail, which belongs to no class.
    pub pile_up_start: Option<usize>,
}

/// Splits the multiplicity bins into `n_classes` classes holding equal
/// fractions of the single-event model.
///
/// `single` and `pile_up` are the bin contents (with under- and overflow) of
/// the single and pile-up event models. As in `HistoCut.C`, bin 1
/// (multiplicity 0) is not used, and the bins from the first one where the
/// pile-up events exceed the single events are excluded from the classes;
/// their single events still count for the most central class. Counting from
/// the highest multiplicity down, a bin belongs to the class in which its
/// cumulative fraction (including the bin) falls.
pub fn class_bins(single: &[f64], pile_up: &[f64], n_classes: usize) -> Result<ClassBins> {
    if single.len() < 3 || n_classes == 0 {
        return Err(Error::Input("empty single-event model histogram".into()));
    }
    let n_bins = single.len() - 2;
    let pile_up_start = (2..=n_bins).find(|&b| single[b] < pile_up.get(b).copied().unwrap_or(0.));
    let integral: f64 = single[2..=n_bins].iter().sum();
    if !(integral > 0.) {
        return Err(Error::Input("single-event model histogram is empty".into()));
    }

    let mut ranges: Vec<Option<(usize, usize)>> = vec![None; n_classes];
    let mut cumulative = 0.;
    for bin in (2..=n_bins).rev() {
        cumulative += single[bin];
        if pile_up_start.is_some_and(|start| bin >= start) {
            continue;
        }
        let class = ((n_classes as f64 * cumulative / integral).ceil() as usize)
            .saturating_sub(1)
            .min(n_classes - 1);
        let range = ranges[class].get_or_insert((bin, bin));
        range.0 = bin;
    }
    Ok(ClassBins {
        ranges,
        pile_up_start,
    })
}

/// Mean and RMS of an observable in a class, and its range in the class from
/// the polynomial fit of the means versus centrality (NaN if the fit failed).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservableStats {
    pub mean: f64,
    pub rms: f64,
    pub min: f64,
    pub max: f64,
}

/// One centrality class.
#[derive(Debug, Clone, PartialEq)]
pub struct CentralityClass {
    /// Class number, 1 for the most central (`Ncc`).
    pub number: usize,
    pub min_percent: f64,
    pub max_percent: f64,
    /// Multiplicity range `[min_border, max_border)`: the edges of the bins.
    pub min_border: f64,
    pub max_border: f64,
    /// Inclusive range of multiplicity histogram bins.
    pub bins: (usize, usize),
    pub b: ObservableStats,
    pub npart: ObservableStats,
    pub ncoll: ObservableStats,
}

impl CentralityClass {
    /// Name suffix of the per-class histograms, e.g. `0.0%-10.0%`.
    pub fn label(&self) -> String {
        class_label(self.min_percent, self.max_percent)
    }

    /// Statistics of observable `k` of [`OBSERVABLES`] (B, Npart, Ncoll).
    fn stats(&self, k: usize) -> ObservableStats {
        [self.b, self.npart, self.ncoll][k]
    }

    fn stats_mut(&mut self, k: usize) -> &mut ObservableStats {
        match k {
            0 => &mut self.b,
            1 => &mut self.npart,
            _ => &mut self.ncoll,
        }
    }
}

/// Result of the centrality determination.
#[derive(Debug, Clone, PartialEq)]
pub struct CentralityResult {
    /// The classes holding at least one multiplicity bin, from the most central.
    pub classes: Vec<CentralityClass>,
    /// Lower multiplicity edge of the pile-up dominated tail, if any.
    pub pile_up_border: Option<f64>,
    /// QA file the classes were determined from.
    pub source: PathBuf,
}

/// Determines the centrality classes, writes the ROOT file and the tables,
/// and returns the classes. Progress and warnings are printed.
pub fn run(config: &CentralityConfig) -> Result<CentralityResult> {
    let open = |path: &Path| {
        FileReader::open(path)
            .map_err(|e| Error::Input(format!("cannot open {}: {e}", path.display())))
    };
    let qa = open(&config.qa_file)?;
    let read_th1 = |file: &FileReader, name: &str| {
        TH1::read_root(file, name)
            .map_err(|e| Error::Input(format!("cannot read histogram {name}: {e}")))
    };
    let single = read_th1(&qa, "glaub_sng_histo")?;
    let pile_up = read_th1(&qa, "glaub_plp_histo")?;
    let fit = read_th1(&qa, "glaub_fit_histo")?;
    let data = if config.data_file == config.qa_file {
        read_th1(&qa, &config.data_hist)?
    } else {
        read_th1(&open(&config.data_file)?, &config.data_hist)?
    };
    let vs_multiplicity = OBSERVABLES
        .iter()
        .map(|(name, _)| {
            let hist = format!("{name}_VS_Multiplicity");
            TH2::read_root(&qa, &hist)
                .map_err(|e| Error::Input(format!("cannot read histogram {hist}: {e}")))
        })
        .collect::<Result<Vec<_>>>()?;

    check_axes(&single, &pile_up, &fit, &data, &vs_multiplicity)?;
    let class_bins = class_bins(&single.contents, &pile_up.contents, config.n_classes)?;
    let axis = &single.xaxis;
    let width = (axis.xmax - axis.xmin) / axis.nbins as f64;
    let edge = |bin: usize| axis.xmin + (bin - 1) as f64 * width;
    let pile_up_border = class_bins.pile_up_start.map(edge);
    if let Some(border) = pile_up_border {
        println!("Pile-up dominated multiplicities >= {border} are not classified");
    }

    /* per-class averages of the observables */
    let n = config.n_classes;
    let mut classes = Vec::new();
    for (i, range) in class_bins.ranges.iter().enumerate() {
        let (min_percent, max_percent) = class_percents(i, n);
        let Some((lo, hi)) = *range else {
            eprintln!(
                "Warning: class {} is empty: a single multiplicity bin holds more than {}% \
                 of the events; use fewer classes",
                class_label(min_percent, max_percent),
                100. / n as f64
            );
            continue;
        };
        let stats: Vec<ObservableStats> = vs_multiplicity
            .iter()
            .map(|h| {
                let (mean, rms) = mean_rms(&project_y(h, "", Some((lo, hi))));
                ObservableStats {
                    mean,
                    rms,
                    min: f64::NAN,
                    max: f64::NAN,
                }
            })
            .collect();
        classes.push(CentralityClass {
            number: i + 1,
            min_percent,
            max_percent,
            min_border: edge(lo),
            max_border: edge(hi + 1),
            bins: (lo, hi),
            b: stats[0],
            npart: stats[1],
            ncoll: stats[2],
        });
    }

    /* ranges of the observables from polynomial fits of the averages versus centrality */
    for (k, (name, _)) in OBSERVABLES.iter().enumerate() {
        let points: Vec<(f64, f64, f64)> = classes
            .iter()
            .map(|c| {
                let s = c.stats(k);
                ((c.min_percent + c.max_percent) / 2., s.mean, s.rms)
            })
            .collect();
        let poly = fit_polynomial(&points);
        if poly.is_none() {
            eprintln!("Warning: polynomial fit of <{name}> versus centrality failed");
        }
        for c in &mut classes {
            let (lo, hi) = (c.min_percent, c.max_percent);
            let s = c.stats_mut(k);
            if let Some(p) = &poly {
                let (a, b) = (eval_polynomial(p, lo), eval_polynomial(p, hi));
                (s.min, s.max) = (a.min(b), a.max(b));
            }
        }
    }
    for c in &classes {
        for border in [c.min_border, c.max_border] {
            if (border - border.round()).abs() > 1e-6 {
                eprintln!(
                    "Warning: multiplicity border {border} is not an integer; \
                     it is rounded in the Result tree and the tables"
                );
            }
        }
    }

    let result = CentralityResult {
        classes,
        pile_up_border,
        source: config.qa_file.clone(),
    };
    std::fs::create_dir_all(&config.out_dir)?;
    write_final(
        &config.final_path(),
        &result,
        &FinalInputs {
            class_bins: &class_bins,
            n_classes: n,
            fit: &fit,
            data: &data,
            vs_multiplicity: &vs_multiplicity,
        },
    )?;
    println!(
        "Centrality classes written to {}",
        config.final_path().display()
    );
    for &format in &config.table_formats {
        let path = config.table_path(format);
        let text = match format {
            TableFormat::Tex => result.tex_table(),
            TableFormat::Csv => result.csv_table(),
            TableFormat::Cpp => result.cpp_table(),
        };
        std::fs::write(&path, text)?;
        println!("Table written to {}", path.display());
    }
    Ok(result)
}

/// The model histograms must share the binning, and the data histogram must
/// have the same bin width and lower edge (the fit's model bin i is data bin i).
fn check_axes(single: &TH1, pile_up: &TH1, fit: &TH1, data: &TH1, vs: &[TH2]) -> Result<()> {
    let uniform = |a: &TAxis| a.xbins.is_empty() && a.nbins > 0;
    let same = |a: &TAxis, b: &TAxis| {
        a.nbins == b.nbins && (a.xmin - b.xmin).abs() < 1e-9 && (a.xmax - b.xmax).abs() < 1e-9
    };
    let model = &single.xaxis;
    if !uniform(model) || !uniform(&data.xaxis) {
        return Err(Error::Input(
            "model and data histograms must have uniform binning".into(),
        ));
    }
    if !same(model, &pile_up.xaxis)
        || !same(model, &fit.xaxis)
        || vs.iter().any(|h| !same(model, &h.xaxis))
    {
        return Err(Error::Input(
            "the model histograms of the QA file have different binnings".into(),
        ));
    }
    let width = |a: &TAxis| (a.xmax - a.xmin) / a.nbins as f64;
    if (model.xmin - data.xaxis.xmin).abs() > 1e-9 * width(model).max(1.)
        || (width(model) / width(&data.xaxis) - 1.).abs() > 1e-9
    {
        return Err(Error::Input(format!(
            "data histogram binning ({} bins in [{}, {}]) does not match the model \
             ({} bins in [{}, {}]): use the data histogram of the fit",
            data.xaxis.nbins, data.xaxis.xmin, data.xaxis.xmax, model.nbins, model.xmin, model.xmax
        )));
    }
    Ok(())
}

/// Nominal percent range of class `i` of `n`.
fn class_percents(i: usize, n: usize) -> (f64, f64) {
    (100. * i as f64 / n as f64, 100. * (i + 1) as f64 / n as f64)
}

/// `%.1f%%-%.1f%%` as in the names of the original histograms.
fn class_label(min_percent: f64, max_percent: f64) -> String {
    format!("{min_percent:.1}%-{max_percent:.1}%")
}

/// Projection on the y axis of the x bins `lo..=hi` (all bins, with under-
/// and overflow, if `None`), like `TH2::ProjectionY`.
fn project_y(h: &TH2, name: &str, bins: Option<(usize, usize)>) -> TH1 {
    let (nx, ny) = (h.nx(), h.ny());
    let stride = nx + 2;
    let (lo, hi) = bins.unwrap_or((0, nx + 1));
    let mut p = if h.yaxis.xbins.is_empty() {
        Hist::reg(ny as i32, h.yaxis.xmin, h.yaxis.xmax)
    } else {
        Hist::var(&h.yaxis.xbins)
    }
    .name(name)
    .double();
    p.xaxis.title = h.yaxis.title.clone();
    p.yaxis.title = "counts".into();
    let sum = |cells: &[f64], iy: usize| (lo..=hi).map(|ix| cells[ix + stride * iy]).sum();
    p.contents = (0..ny + 2).map(|iy| sum(&h.contents, iy)).collect();
    if !h.sumw2.is_empty() {
        p.sumw2 = (0..ny + 2).map(|iy| sum(&h.sumw2, iy)).collect();
    }
    reset_stats(&mut p);
    p
}

/// Makes ROOT recompute the statistics from the bin contents
/// (`TH1::GetStats` does so if `fTsumw == 0` and `fEntries > 0`).
fn reset_stats(h: &mut TH1) {
    h.entries = h.contents.iter().sum();
    h.tsumw = 0.;
    h.tsumw2 = 0.;
    h.tsumwx = 0.;
    h.tsumwx2 = 0.;
}

/// Center of bin `bin` (1-based) of an axis.
fn bin_center(axis: &TAxis, bin: usize) -> f64 {
    if axis.xbins.is_empty() {
        let width = (axis.xmax - axis.xmin) / axis.nbins as f64;
        axis.xmin + (bin as f64 - 0.5) * width
    } else {
        (axis.xbins[bin - 1] + axis.xbins[bin]) / 2.
    }
}

/// Mean and RMS of a histogram from its bin centers, without under- and
/// overflow, as `TH1::GetMean` and `TH1::GetRMS`; NaN if it is empty.
fn mean_rms(h: &TH1) -> (f64, f64) {
    let (mut s0, mut s1, mut s2) = (0., 0., 0.);
    for bin in 1..=h.xaxis.nbins as usize {
        let (x, w) = (bin_center(&h.xaxis, bin), h.contents[bin]);
        s0 += w;
        s1 += w * x;
        s2 += w * x * x;
    }
    if !(s0 > 0.) {
        return (f64::NAN, f64::NAN);
    }
    let mean = s1 / s0;
    (mean, (s2 / s0 - mean * mean).max(0.).sqrt())
}

/// Weighted least-squares fit of the polynomial `sum c_i (x / 100)^i` to the
/// points `(x, y, error)`, of degree [`POLY_DEGREE`] or lower if there are
/// fewer points. Points with a zero or invalid error are skipped, as in ROOT's
/// chi2 fits. `None` if no point is left or the system is singular.
fn fit_polynomial(points: &[(f64, f64, f64)]) -> Option<Vec<f64>> {
    let points: Vec<_> = points
        .iter()
        .filter(|(x, y, e)| *e > 0. && e.is_finite() && x.is_finite() && y.is_finite())
        .collect();
    let n = (POLY_DEGREE + 1).min(points.len());
    if n == 0 {
        return None;
    }
    /* normal equations a c = r */
    let mut a = vec![vec![0.; n]; n];
    let mut r = vec![0.; n];
    for &&(x, y, e) in &points {
        let w = 1. / (e * e);
        let powers: Vec<f64> = (0..n).map(|i| (x / 100.).powi(i as i32)).collect();
        for i in 0..n {
            r[i] += w * powers[i] * y;
            for k in 0..n {
                a[i][k] += w * powers[i] * powers[k];
            }
        }
    }
    solve(a, r)
}

/// Solves `a x = r` by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut r: Vec<f64>) -> Option<Vec<f64>> {
    let n = r.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &k| a[i][col].abs().total_cmp(&a[k][col].abs()))?;
        if !(a[pivot][col].abs() > 1e-300) {
            return None;
        }
        a.swap(col, pivot);
        r.swap(col, pivot);
        let (upper, lower) = a.split_at_mut(col + 1);
        let pivot_row = &upper[col];
        for (row, a_row) in lower.iter_mut().enumerate() {
            let factor = a_row[col] / pivot_row[col];
            for (x, p) in a_row[col..].iter_mut().zip(&pivot_row[col..]) {
                *x -= factor * p;
            }
            r[col + 1 + row] -= factor * r[col];
        }
    }
    let mut x = vec![0.; n];
    for row in (0..n).rev() {
        let s: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (r[row] - s) / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

fn eval_polynomial(coefficients: &[f64], x: f64) -> f64 {
    let u = x / 100.;
    coefficients.iter().rev().fold(0., |acc, c| acc * u + c)
}

/// Inputs of the output ROOT file besides the result.
struct FinalInputs<'a> {
    class_bins: &'a ClassBins,
    n_classes: usize,
    fit: &'a TH1,
    data: &'a TH1,
    vs_multiplicity: &'a [TH2],
}

/// Copy of `h` named `name` with only the bins `lo..=hi`.
fn restricted(h: &TH1, name: &str, (lo, hi): (usize, usize)) -> TH1 {
    let mut r = h.clone();
    r.name = name.into();
    r.title = String::new();
    r.xaxis.title = "tracks".into();
    r.yaxis.title = "counts".into();
    let outside = |bin: usize| bin < lo || bin > hi;
    for (bin, c) in r.contents.iter_mut().enumerate() {
        if outside(bin) {
            *c = 0.;
        }
    }
    for (bin, c) in r.sumw2.iter_mut().enumerate() {
        if outside(bin) {
            *c = 0.;
        }
    }
    reset_stats(&mut r);
    r
}

/// Writes the output ROOT file: the tree `Result`, the averages versus
/// centrality, the per-class distributions of the observables, and the model
/// and data multiplicity of each class.
fn write_final(path: &Path, result: &CentralityResult, inputs: &FinalInputs) -> Result<()> {
    let n = inputs.n_classes;
    let classes = &result.classes;

    let column_f64 = |get: &dyn Fn(&CentralityClass) -> f64| classes.iter().map(get).collect();
    let mut branches = vec![
        Branch::i32("Ncc", classes.iter().map(|c| c.number as i32).collect()),
        Branch::f32(
            "MinPercent",
            classes.iter().map(|c| c.min_percent as f32).collect(),
        ),
        Branch::f32(
            "MaxPercent",
            classes.iter().map(|c| c.max_percent as f32).collect(),
        ),
        Branch::i32(
            "MinBorder",
            classes
                .iter()
                .map(|c| c.min_border.round() as i32)
                .collect(),
        ),
        Branch::i32(
            "MaxBorder",
            classes
                .iter()
                .map(|c| c.max_border.round() as i32)
                .collect(),
        ),
    ];
    for (k, (name, _)) in OBSERVABLES.iter().enumerate() {
        branches.extend([
            Branch::f64(format!("{name}Average"), column_f64(&|c| c.stats(k).mean)),
            Branch::f64(format!("{name}Width"), column_f64(&|c| c.stats(k).rms)),
            Branch::f64(format!("{name}Min"), column_f64(&|c| c.stats(k).min)),
            Branch::f64(format!("{name}Max"), column_f64(&|c| c.stats(k).max)),
        ]);
    }
    let tree = Tree::new("Result", branches);

    /* <observable> versus centrality, with the RMS as error */
    let mut averages = Vec::new();
    for (k, (name, label)) in OBSERVABLES.iter().enumerate() {
        let mut h = Hist::reg(n as i32, 0., 100.)
            .name(format!("{name}_average_VS_Centrality"))
            .double();
        h.xaxis.title = "Centrality, %".into();
        h.yaxis.title = (*label).into();
        h.sumw2 = vec![0.; n + 2];
        for c in classes {
            let s = c.stats(k);
            h.contents[c.number] = s.mean;
            h.sumw2[c.number] = s.rms * s.rms;
        }
        reset_stats(&mut h);
        averages.push(h);
    }

    /* distributions of the observables per class, and for all events */
    let mut projections = Vec::new();
    for ((name, _), h2) in OBSERVABLES.iter().zip(inputs.vs_multiplicity) {
        for c in classes {
            let hist_name = format!("{name}_VS_CentralityClass {}", c.label());
            projections.push(project_y(h2, &hist_name, Some(c.bins)));
        }
        let all = format!("{name}_VS_CentralityClass 0%-100%");
        projections.push(project_y(h2, &all, None));
    }

    /* model and data multiplicity per class, and centrality versus multiplicity */
    let mut per_class = Vec::new();
    for c in classes {
        per_class.push(restricted(
            inputs.fit,
            &format!("CentralityClass_Fit {}", c.label()),
            c.bins,
        ));
    }
    for c in classes {
        per_class.push(restricted(
            inputs.data,
            &format!("CentralityClass {}", c.label()),
            c.bins,
        ));
    }
    let mut vs_mult = inputs.fit.clone();
    vs_mult.name = "Centrality_vs_Multiplicity".into();
    vs_mult.title = String::new();
    vs_mult.xaxis.title = "tracks".into();
    vs_mult.yaxis.title = "Centrality, %".into();
    vs_mult.contents.iter_mut().for_each(|c| *c = 0.);
    vs_mult.sumw2.clear();
    for (i, range) in inputs.class_bins.ranges.iter().enumerate() {
        if let Some((lo, hi)) = *range {
            let (_, max_percent) = class_percents(i, n);
            vs_mult.contents[lo..=hi]
                .iter_mut()
                .for_each(|c| *c = max_percent);
        }
    }
    reset_stats(&mut vs_mult);

    let mut writer = FileWriter::create(path);
    for h in averages.iter().chain(&projections).chain(&per_class) {
        writer = writer.add(h);
    }
    writer
        .add(&vs_mult)
        .put(tree)
        .write(Compression::default())?;
    Ok(())
}

/// Percent with the decimals only when needed, e.g. `10` or `12.5`.
fn percent(p: f64) -> String {
    if (p - p.round()).abs() < 1e-9 {
        format!("{p:.0}")
    } else {
        format!("{p:.1}")
    }
}

impl CentralityResult {
    fn centrality(c: &CentralityClass) -> String {
        format!("{} - {}", percent(c.min_percent), percent(c.max_percent))
    }

    /// Plain text table, as printed by `printFinal.C` without an output file.
    pub fn plain_table(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "File: {}.", self.source.display());
        let _ = writeln!(
            s,
            "  Cent, %   | Mult_min | Mult_max | <b>, fm |   RMS   | bmin, fm | bmax, fm | \
             <Npart> |   RMS   | Npart_min | Npart_max | <Ncoll> |   RMS   | Ncoll_min | Ncoll_max |"
        );
        let _ = writeln!(
            s,
            "------------|----------|----------|---------|---------|----------|----------|\
             ---------|---------|-----------|-----------|---------|---------|-----------|-----------|"
        );
        for c in &self.classes {
            let _ = writeln!(
                s,
                "{:>11} | {:8} | {:8} | {:7.2} | {:7.2} | {:8.2} | {:8.2} | {:7.2} | {:7.2} | \
                 {:9.2} | {:9.2} | {:7.2} | {:7.2} | {:9.2} | {:9.2} |",
                Self::centrality(c),
                c.min_border.round(),
                c.max_border.round(),
                c.b.mean,
                c.b.rms,
                c.b.min,
                c.b.max,
                c.npart.mean,
                c.npart.rms,
                c.npart.min,
                c.npart.max,
                c.ncoll.mean,
                c.ncoll.rms,
                c.ncoll.min,
                c.ncoll.max
            );
        }
        s
    }

    /// LaTeX document with the table, with the columns of `printFinal.C`.
    pub fn tex_table(&self) -> String {
        let mut s = String::new();
        s.push_str(
            "\\documentclass[11pt]{article}\n\
             \\usepackage[utf8]{inputenc}\n\
             \\usepackage{geometry}\n\
             \\geometry{legalpaper, landscape, margin=2in}\n\n\
             \\begin{document}\n\n\
             Generated from a file:\n\
             \\begin{verbatim*}\n",
        );
        let _ = writeln!(s, "{}", self.source.display());
        s.push_str(
            "\\end{verbatim*}\n\
             \\begin{center}\n\
             \\begin{tabular}{ |c|c|c|c|c|c|c|c|c| }\n\
             \t\\hline\n\
             \tCentrality, \\% & $N_{ch}^{min}$ & $N_{ch}^{max}$ & $\\langle b \\rangle$, fm & RMS & \
             $\\langle N_{part} \\rangle$ & RMS & $\\langle N_{coll} \\rangle$ & RMS \\\\\n",
        );
        for c in &self.classes {
            let _ = writeln!(
                s,
                "\t\\hline\n\t{} & {} & {} & {:.2} & {:.2} & {:.2} & {:.2} & {:.2} & {:.2} \\\\",
                Self::centrality(c),
                c.min_border.round(),
                c.max_border.round(),
                c.b.mean,
                c.b.rms,
                c.npart.mean,
                c.npart.rms,
                c.ncoll.mean,
                c.ncoll.rms
            );
        }
        s.push_str("\t\\hline\n\\end{tabular}\n\\end{center}\n\\end{document}\n");
        s
    }

    /// CSV table with the columns of `printFinal.C`, after a `#` comment line
    /// naming the source file.
    pub fn csv_table(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# Generated from a file: {}", self.source.display());
        s.push_str(
            "Centrality class,Mult_min,Mult_max,Mean b,RMS,b_min,b_max,Mean Npart,RMS,\
             Npart_min,Npart_max,Mean Ncoll,RMS,Ncoll_min,Ncoll_max\n",
        );
        for c in &self.classes {
            let _ = writeln!(
                s,
                "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
                Self::centrality(c),
                c.min_border.round(),
                c.max_border.round(),
                c.b.mean,
                c.b.rms,
                c.b.min,
                c.b.max,
                c.npart.mean,
                c.npart.rms,
                c.npart.min,
                c.npart.max,
                c.ncoll.mean,
                c.ncoll.rms,
                c.ncoll.min,
                c.ncoll.max
            );
        }
        s
    }

    /// ROOT macro with arrays of the class borders and averages, and the
    /// function `GetCentMult(mult)` returning the centrality (middle of the
    /// class, in percent) of a multiplicity, or -1.
    pub fn cpp_table(&self) -> String {
        let n = self.classes.len();
        let mut s = String::new();
        let _ = writeln!(s, "// Generated from a file: {}", self.source.display());
        let mut array = |ty: &str, name: &str, values: Vec<String>| {
            let _ = writeln!(s, "{ty} {name} [{n}] = {{ {} }};", values.join(", "));
        };
        let float = |v: f64| {
            if v.is_finite() {
                format!("{}", v as f32)
            } else {
                "NAN".into()
            }
        };
        let column = |get: &dyn Fn(&CentralityClass) -> f64| {
            self.classes.iter().map(|c| float(get(c))).collect()
        };
        let int_column = |get: &dyn Fn(&CentralityClass) -> f64| {
            self.classes
                .iter()
                .map(|c| format!("{}", get(c).round() as i64))
                .collect()
        };
        array("Float_t", "minCentPercent", column(&|c| c.min_percent));
        array("Float_t", "maxCentPercent", column(&|c| c.max_percent));
        array("Int_t", "minMult", int_column(&|c| c.min_border));
        array("Int_t", "maxMult", int_column(&|c| c.max_border));
        for (k, (name, _)) in OBSERVABLES.iter().enumerate() {
            array(
                "Float_t",
                &format!("mean{name}"),
                column(&|c| c.stats(k).mean),
            );
            array(
                "Float_t",
                &format!("rms{name}"),
                column(&|c| c.stats(k).rms),
            );
            array(
                "Float_t",
                &format!("min{name}"),
                column(&|c| c.stats(k).min),
            );
            array(
                "Float_t",
                &format!("max{name}"),
                column(&|c| c.stats(k).max),
            );
        }
        let _ = write!(
            s,
            "\nFloat_t GetCentMult(Int_t mult)\n\
             // Returns the centrality (middle of the class, in percent) for a given multiplicity, or -1\n\
             {{\n\
             \tInt_t centBin = -1;\n\
             \tfor (Int_t i = 0; i < {n}; i++)\n\
             \t{{\n\
             \t\tif (mult >= minMult[i] && mult < maxMult[i])\n\
             \t\t\tcentBin = i;\n\
             \t}}\n\
             \tif (centBin == -1) return -1.;\n\
             \treturn (maxCentPercent[centBin] - minCentPercent[centBin]) / 2. + minCentPercent[centBin];\n\
             }}\n"
        );
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Contents with under/overflow from the in-range values.
    fn contents(values: &[f64]) -> Vec<f64> {
        let mut c = vec![0.];
        c.extend_from_slice(values);
        c.push(0.);
        c
    }

    #[test]
    fn equal_classes_of_a_flat_distribution() {
        /* bin 1 (multiplicity 0) is ignored; 100 bins of 1 event */
        let mut values = vec![1000.];
        values.extend([1.; 100]);
        let bins = class_bins(&contents(&values), &contents(&[0.; 101]), 10).unwrap();
        assert_eq!(bins.pile_up_start, None);
        /* class 0 holds the 10 highest bins 92..=101, the last class 2..=11 */
        for (i, range) in bins.ranges.iter().enumerate() {
            let hi = 101 - 10 * i;
            assert_eq!(*range, Some((hi - 9, hi)), "class {i}");
        }
    }

    #[test]
    fn pile_up_tail_is_excluded_but_counted() {
        let single = contents(&[5., 4., 3., 2., 1., 0.5, 0.5]);
        let pile_up = contents(&[0., 0., 0., 0.1, 0.1, 1., 0.2]);
        let bins = class_bins(&single, &pile_up, 2).unwrap();
        /* from bin 6 on the pile-up exceeds the single events (also bin 7 is excluded) */
        assert_eq!(bins.pile_up_start, Some(6));
        /* integral of bins 2..=7 is 11: class 0 is <= 5.5 from the top: 0.5 + 0.5 + 1 + 2 */
        assert_eq!(bins.ranges, vec![Some((4, 5)), Some((2, 3))]);
    }

    #[test]
    fn a_large_bin_leaves_classes_empty() {
        let single = contents(&[0., 1., 8., 1.]);
        let bins = class_bins(&single, &contents(&[0.; 4]), 5).unwrap();
        /* 10%: bin 4, bin 3 reaches 90% (class 4), classes 1..=3 are empty */
        assert_eq!(
            bins.ranges,
            vec![Some((4, 4)), None, None, None, Some((2, 3))]
        );
        assert!(class_bins(&contents(&[3., 0.]), &contents(&[0., 0.]), 2).is_err());
    }

    #[test]
    fn polynomial_fit() {
        let f = |x: f64| 3. - 0.2 * x + 1e-3 * x * x;
        let points: Vec<_> = (0..10)
            .map(|i| (5. + 10. * i as f64, f(5. + 10. * i as f64), 0.1))
            .collect();
        let p = fit_polynomial(&points).unwrap();
        for x in [0., 37., 100.] {
            assert!((eval_polynomial(&p, x) - f(x)).abs() < 1e-8, "x = {x}");
        }
        /* two points: a line; zero errors are skipped */
        let p = fit_polynomial(&[(10., 1., 0.1), (30., 3., 0.2), (50., 99., 0.)]).unwrap();
        assert_eq!(p.len(), 2);
        assert!((eval_polynomial(&p, 20.) - 2.).abs() < 1e-12);
        assert!(fit_polynomial(&[(10., 1., 0.)]).is_none());
    }

    fn class(number: usize, borders: (f64, f64)) -> CentralityClass {
        let stats = |m: f64| ObservableStats {
            mean: m,
            rms: 0.5,
            min: m - 1.,
            max: m + 1.,
        };
        CentralityClass {
            number,
            min_percent: 50. * (number - 1) as f64,
            max_percent: 50. * number as f64,
            min_border: borders.0,
            max_border: borders.1,
            bins: (1, 1),
            b: stats(3.),
            npart: stats(200.),
            ncoll: stats(500.),
        }
    }

    #[test]
    fn tables() {
        let result = CentralityResult {
            classes: vec![class(1, (40., 120.)), class(2, (1., 40.))],
            pile_up_border: None,
            source: "qa.root".into(),
        };
        let csv = result.csv_table();
        let lines: Vec<_> = csv.lines().collect();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with('#'));
        assert_eq!(lines[1].split(',').count(), 15);
        assert_eq!(
            lines[2],
            "0 - 50,40,120,3.00,0.50,2.00,4.00,200.00,0.50,199.00,201.00,500.00,0.50,499.00,501.00"
        );
        let cpp = result.cpp_table();
        assert!(cpp.contains("Int_t minMult [2] = { 40, 1 };"), "{cpp}");
        assert!(
            cpp.contains("Float_t maxNpart [2] = { 201, 201 };"),
            "{cpp}"
        );
        assert!(cpp.contains("Float_t GetCentMult(Int_t mult)"));
        let tex = result.tex_table();
        assert!(
            tex.contains("\t50 - 100 & 1 & 40 & 3.00 & 0.50 & 200.00 & 0.50 & 500.00 & 0.50 \\\\"),
            "{tex}"
        );
        assert!(tex.trim_end().ends_with("\\end{document}"));
        assert_eq!(result.plain_table().lines().count(), 5);
    }

    #[test]
    fn config() {
        let c = CentralityConfigBuilder::from_ron_str(
            r#"(
                fit: (glauber_file: "g.root"),
                centrality: (qa_file: "qa.root", data_hist: "h", n_classes: 5,
                             table_formats: [Csv, Tex, Cpp]),
            )"#,
        )
        .unwrap()
        .build()
        .unwrap();
        assert_eq!(c.data_file, PathBuf::from("qa.root"));
        assert_eq!((c.n_classes, c.final_file.as_str()), (5, "FINAL.root"));
        assert_eq!(
            c.table_path(TableFormat::Cpp),
            PathBuf::from("./centrality_table.C")
        );
        assert_eq!(c.table_formats.len(), 3);

        let parse = |s: &str| CentralityConfigBuilder::from_ron_str(s).and_then(|b| b.build());
        assert!(parse(r#"(centrality: (data_hist: "h"))"#).is_err());
        assert!(parse(r#"(centrality: (qa_file: "q", data_hist: "h", n_classes: 0))"#).is_err());
        assert!(
            parse(r#"(centrality: (qa_file: "q", data_hist: "h", table_formats: [Pdf]))"#).is_err()
        );
        assert!(parse(r#"(centrality: (qa_file: "q", data_hist: "h", typo: 1))"#).is_err());
    }
}
