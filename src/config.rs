use std::path::{Path, PathBuf};

use oxiroot::prelude::TAxis;
use serde::Deserialize;

use crate::config_file;
use crate::error::{Error, Result};
use crate::mode::Mode;

/// Maximum number of scan points per parameter.
const MAX_SCAN_POINTS: usize = 100_000;
/// Maximum number of centrality classes.
const MAX_CLASSES: usize = 1000;

/// Scan of one fit parameter: `min, min + step, ... <= max`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScanRange {
    pub min: f32,
    pub max: f32,
    pub step: f32,
}

impl ScanRange {
    pub const fn new(min: f32, max: f32, step: f32) -> Self {
        Self { min, max, step }
    }

    /// A single fixed value.
    pub const fn fixed(value: f32) -> Self {
        Self::new(value, value, 0.)
    }

    /// Scan points, computed by index rather than by accumulation.
    /// `max <= min` gives the single point `min`.
    pub fn points(&self) -> Vec<f32> {
        if self.max <= self.min || !(self.step > 0.) {
            return vec![self.min];
        }
        let n = ((self.max - self.min) / self.step + 1e-4).floor() as usize + 1;
        (0..n).map(|i| self.min + i as f32 * self.step).collect()
    }

    fn validate(&self, name: &str) -> Result<()> {
        if !(self.min.is_finite() && self.max.is_finite()) {
            return Err(Error::Config(format!("{name} range must be finite")));
        }
        if self.max < self.min {
            return Err(Error::Config(format!(
                "{name} range: max ({}) < min ({})",
                self.max, self.min
            )));
        }
        if self.max > self.min {
            if !(self.step > 0.) {
                return Err(Error::Config(format!(
                    "{name} range: step must be positive, got {}",
                    self.step
                )));
            }
            let n = ((self.max - self.min) / self.step).floor() as f64 + 1.;
            if n > MAX_SCAN_POINTS as f64 {
                return Err(Error::Config(format!(
                    "{name} range: {n} points (step {}), maximum is {MAX_SCAN_POINTS}",
                    self.step
                )));
            }
        }
        Ok(())
    }
}

/// Distribution of the multiplicity produced by one ancestor, with mean `mu`
/// and variance `mu (1 + mu / k)` ([`Distribution::GammaScaleFree`]: `mu^2 / k`).
/// It is used both in the fit and for the per-ancestor output histogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
pub enum Distribution {
    /// Continuous Gamma distribution, as in the original framework's fit.
    #[default]
    Gamma,
    /// Negative binomial distribution (integer multiplicities).
    Nbd,
    /// Gamma distribution with shape `k` and scale `mu / k`: the variance
    /// `mu^2 / k` has no Poisson term, so `k` and the fit do not depend on
    /// the units of a continuous observable such as energy.
    GammaScaleFree,
}

impl Distribution {
    pub fn histogram_name(self) -> &'static str {
        match self {
            Distribution::Gamma => "gamma",
            Distribution::Nbd => "nbd",
            Distribution::GammaScaleFree => "gamma_scale_free",
        }
    }

    /// Variance of the value produced by one ancestor.
    pub fn variance(self, mu: f64, k: f64) -> f64 {
        match self {
            Distribution::Gamma | Distribution::Nbd => mu * (1. + mu / k),
            Distribution::GammaScaleFree => mu * mu / k,
        }
    }
}

/// Statistic minimized to find the optimal fit parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
pub enum FitMethod {
    /// Minimum chi2 (Neyman chi2 with the data and model statistical errors),
    /// as in the original framework.
    #[default]
    Chi2,
    /// Maximum Poisson likelihood: minimizes the likelihood ratio chi2
    /// `-2 ln(L / L_saturated) = 2 sum(m - d + d ln(d / m))` (Baker-Cousins).
    Likelihood,
}

impl FitMethod {
    /// Label of the minimized statistic per degree of freedom.
    pub fn statistic_name(self) -> &'static str {
        match self {
            FitMethod::Chi2 => "chi2/ndf",
            FitMethod::Likelihood => "-2lnL/ndf",
        }
    }
}

/// Full configuration of a Glauber fit. Create it with [`FitConfig::builder`].
#[derive(Debug, Clone, PartialEq)]
pub struct FitConfig {
    /// ROOT file with the MC-Glauber tree.
    pub glauber_file: PathBuf,
    /// Name of the MC-Glauber tree (branches `B`, `Npart`, `Ncoll`, `Ecc1..5`, `Psi1..5`).
    pub glauber_tree: String,
    /// ROOT file with the data multiplicity histogram.
    pub data_file: PathBuf,
    /// Name of the data multiplicity histogram.
    pub data_hist: String,
    /// Directory for the output files.
    pub out_dir: PathBuf,
    /// Number of golden section iterations used to find the optimal `mu`.
    pub n_iter: u32,
    /// Scan of the `f` parameter of the number of ancestors.
    pub f: ScanRange,
    /// Scan of the NBD/Gamma `k` parameter.
    pub k: ScanRange,
    /// Scan of the pile-up probability `p`.
    pub p: ScanRange,
    /// First bin of the chi2 range.
    pub fit_min_bin: usize,
    /// Last bin of the chi2 range.
    pub fit_max_bin: usize,
    /// Fit range `[min, max)` in units of the data (`fit_min`, `fit_max`)
    /// instead of bins. The bins are then set from the data histogram by
    /// [`FitConfig::resolve_fit_bins`] (done by the fitter); until then
    /// `fit_min_bin` and `fit_max_bin` are 0.
    pub fit_values: Option<(f64, f64)>,
    /// Minimum chi2 or maximum likelihood.
    pub fit_method: FitMethod,
    /// Functional form of the number of ancestors.
    pub mode: Mode,
    /// Number of worker threads.
    pub n_threads: usize,
    pub distribution: Distribution,
    /// Seed of the random generator; `None` gives a different seed every run.
    pub seed: Option<u64>,
}

impl FitConfig {
    pub fn builder() -> FitConfigBuilder {
        FitConfigBuilder::default()
    }

    /// Name of the section of the shared RON file read by `bin/fit`.
    pub const RON_SECTION: &'static str = "fit";

    /// With [`fit_values`](Self::fit_values), sets `fit_min_bin` and
    /// `fit_max_bin` to the bins of the uniform data axis covering
    /// `[min, max)`: from the bin containing `min` to the bin containing `max`,
    /// or the bin before if `max` is on a bin edge, within the axis.
    pub fn resolve_fit_bins(&mut self, axis: &TAxis) -> Result<()> {
        let Some((min, max)) = self.fit_values else {
            return Ok(());
        };
        let nbins = axis.nbins.max(0) as usize;
        if !axis.xbins.is_empty() || nbins == 0 {
            return Err(Error::Input(
                "data histogram must have uniform binning".into(),
            ));
        }
        if !(min < axis.xmax && max > axis.xmin) {
            return Err(Error::Config(format!(
                "fit range [{min}, {max}) is outside the data histogram [{}, {})",
                axis.xmin, axis.xmax
            )));
        }
        let width = (axis.xmax - axis.xmin) / nbins as f64;
        /* bin edges count as on the edge within a relative 1e-9 of a bin */
        let position = |x: f64| (x - axis.xmin) / width;
        let lo = (position(min) + 1e-9).floor().max(0.) as usize + 1;
        let hi = ((position(max) - 1e-9).ceil() as usize).min(nbins);
        if lo > hi {
            return Err(Error::Config(format!(
                "fit range [{min}, {max}) holds no data histogram bin"
            )));
        }
        self.fit_min_bin = lo;
        self.fit_max_bin = hi;
        Ok(())
    }

    /// Reads and validates the [`RON_SECTION`](Self::RON_SECTION) section of
    /// a RON configuration file, see `config.ron`.
    pub fn from_ron_file(path: impl AsRef<Path>) -> Result<Self> {
        let builder: FitConfigBuilder =
            config_file::read_section(path.as_ref(), Self::RON_SECTION)?;
        builder.expand_home().build()
    }
}

/// Builder for [`FitConfig`]. The input files and object names are required,
/// everything else defaults to the values of the original `config.c`.
///
/// It can also be deserialized from the `fit` section of a RON file
/// ([`FitConfig::from_ron_file`], [`FitConfigBuilder::from_ron_str`]);
/// omitted fields keep their defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FitConfigBuilder {
    glauber_file: Option<PathBuf>,
    glauber_tree: Option<String>,
    data_file: Option<PathBuf>,
    data_hist: Option<String>,
    out_dir: PathBuf,
    n_iter: u32,
    f: ScanRange,
    k: ScanRange,
    p: ScanRange,
    #[serde(rename = "mult_min")]
    fit_min_bin: Option<usize>,
    #[serde(rename = "mult_max")]
    fit_max_bin: Option<usize>,
    fit_min: Option<f64>,
    fit_max: Option<f64>,
    fit_method: FitMethod,
    mode: Mode,
    n_threads: Option<usize>,
    distribution: Distribution,
    seed: Option<u64>,
}

impl Default for FitConfigBuilder {
    fn default() -> Self {
        Self {
            glauber_file: None,
            glauber_tree: None,
            data_file: None,
            data_hist: None,
            out_dir: PathBuf::from("."),
            n_iter: 20,
            f: ScanRange::new(0.1, 0.1, 0.01),
            k: ScanRange::new(0.5, 1.0, 0.01),
            p: ScanRange::new(0.001, 0.05, 0.001),
            fit_min_bin: None,
            fit_max_bin: None,
            fit_min: None,
            fit_max: None,
            fit_method: FitMethod::Chi2,
            mode: Mode::Star,
            n_threads: None,
            distribution: Distribution::Gamma,
            seed: None,
        }
    }
}

impl FitConfigBuilder {
    /// Parses the `fit` section of a RON configuration (other sections are
    /// ignored). `Option` fields may be written without `Some(...)`, and a
    /// leading `~/` in the file paths is expanded to `$HOME`.
    pub fn from_ron_str(text: &str) -> Result<Self> {
        let builder: Self = config_file::parse_section(text, FitConfig::RON_SECTION)
            .map_err(|e| Error::Config(e.to_string()))?;
        Ok(builder.expand_home())
    }

    fn expand_home(mut self) -> Self {
        for path in [&mut self.glauber_file, &mut self.data_file]
            .into_iter()
            .flatten()
            .chain([&mut self.out_dir])
        {
            *path = expand_home(path);
        }
        self
    }

    /// MC-Glauber input: ROOT file and tree name.
    pub fn glauber(mut self, file: impl Into<PathBuf>, tree: impl Into<String>) -> Self {
        self.glauber_file = Some(file.into());
        self.glauber_tree = Some(tree.into());
        self
    }

    /// Data input: ROOT file and multiplicity histogram name.
    pub fn data(mut self, file: impl Into<PathBuf>, hist: impl Into<String>) -> Self {
        self.data_file = Some(file.into());
        self.data_hist = Some(hist.into());
        self
    }

    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = dir.into();
        self
    }

    pub fn n_iter(mut self, n_iter: u32) -> Self {
        self.n_iter = n_iter;
        self
    }

    pub fn f_range(mut self, min: f32, max: f32, step: f32) -> Self {
        self.f = ScanRange::new(min, max, step);
        self
    }

    pub fn k_range(mut self, min: f32, max: f32, step: f32) -> Self {
        self.k = ScanRange::new(min, max, step);
        self
    }

    pub fn p_range(mut self, min: f32, max: f32, step: f32) -> Self {
        self.p = ScanRange::new(min, max, step);
        self
    }

    /// Bin range `[min, max]` of the data histogram used for chi2
    /// (default 10..=110).
    pub fn fit_range(mut self, min: usize, max: usize) -> Self {
        self.fit_min_bin = Some(min);
        self.fit_max_bin = Some(max);
        self
    }

    pub fn fit_min_bin(mut self, min: usize) -> Self {
        self.fit_min_bin = Some(min);
        self
    }

    pub fn fit_max_bin(mut self, max: usize) -> Self {
        self.fit_max_bin = Some(max);
        self
    }

    /// Fit range `[min, max)` in units of the data (e.g. multiplicity or
    /// energy) instead of bins, see [`FitConfig::resolve_fit_bins`].
    pub fn fit_value_range(mut self, min: f64, max: f64) -> Self {
        self.fit_min = Some(min);
        self.fit_max = Some(max);
        self
    }

    pub fn fit_method(mut self, fit_method: FitMethod) -> Self {
        self.fit_method = fit_method;
        self
    }

    pub fn mode(mut self, mode: Mode) -> Self {
        self.mode = mode;
        self
    }

    /// Number of worker threads; defaults to the available parallelism.
    pub fn n_threads(mut self, n_threads: usize) -> Self {
        self.n_threads = Some(n_threads);
        self
    }

    pub fn distribution(mut self, distribution: Distribution) -> Self {
        self.distribution = distribution;
        self
    }

    /// Fixed random seed for reproducible fits.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn build(self) -> Result<FitConfig> {
        let missing = |what: &str| Error::Config(format!("{what} is not set"));
        let glauber_file = self.glauber_file.ok_or_else(|| missing("Glauber file"))?;
        let glauber_tree = self.glauber_tree.ok_or_else(|| missing("Glauber tree"))?;
        let data_file = self.data_file.ok_or_else(|| missing("data file"))?;
        let data_hist = self.data_hist.ok_or_else(|| missing("data histogram"))?;

        if self.n_iter == 0 {
            return Err(Error::Config("n_iter must be at least 1".into()));
        }
        self.f.validate("f")?;
        self.k.validate("k")?;
        self.p.validate("p")?;
        if !(self.k.min > 0.) {
            return Err(Error::Config(format!(
                "k must be positive, got {}",
                self.k.min
            )));
        }
        if !(self.p.min >= 0. && self.p.max < 1.) {
            return Err(Error::Config(format!(
                "p must be in [0, 1), got [{}, {}]",
                self.p.min, self.p.max
            )));
        }
        let bins_set = self.fit_min_bin.is_some() || self.fit_max_bin.is_some();
        let (fit_min_bin, fit_max_bin, fit_values) = match (self.fit_min, self.fit_max) {
            (None, None) => {
                let (min, max) = (
                    self.fit_min_bin.unwrap_or(10),
                    self.fit_max_bin.unwrap_or(110),
                );
                if min >= max {
                    return Err(Error::Config(format!(
                        "fit range: min bin ({min}) must be below max bin ({max})"
                    )));
                }
                (min, max, None)
            }
            (Some(min), Some(max)) => {
                if bins_set {
                    return Err(Error::Config(
                        "fit range: set either mult_min/mult_max (bins) or fit_min/fit_max \
                         (values), not both"
                            .into(),
                    ));
                }
                if !(min.is_finite() && max.is_finite() && min < max) {
                    return Err(Error::Config(format!(
                        "fit range: fit_min ({min}) must be below fit_max ({max})"
                    )));
                }
                (0, 0, Some((min, max)))
            }
            _ => {
                return Err(Error::Config(
                    "fit range: fit_min and fit_max must be set together".into(),
                ));
            }
        };
        let n_threads = self
            .n_threads
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        if n_threads == 0 {
            return Err(Error::Config("n_threads must be at least 1".into()));
        }

        Ok(FitConfig {
            glauber_file,
            glauber_tree,
            data_file,
            data_hist,
            out_dir: self.out_dir,
            n_iter: self.n_iter,
            f: self.f,
            k: self.k,
            p: self.p,
            fit_min_bin,
            fit_max_bin,
            fit_values,
            fit_method: self.fit_method,
            mode: self.mode,
            n_threads,
            distribution: self.distribution,
            seed: self.seed,
        })
    }
}

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
    /// Multiplicity bins entirely below this value (events without hits) are
    /// not classified; 0 for a continuous observable such as energy.
    pub min_value: f64,
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
    min_value: f64,
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
            min_value: 1.,
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

    /// Bins entirely below `min_value` are not classified (default 1: events
    /// without hits).
    pub fn min_value(mut self, min_value: f64) -> Self {
        self.min_value = min_value;
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
        if !self.min_value.is_finite() {
            return Err(Error::Config(format!(
                "min_value must be finite, got {}",
                self.min_value
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
            min_value: self.min_value,
            table_formats: self.table_formats,
            table_name: self.table_name,
        })
    }
}

/// Replaces a leading `~` with `$HOME`.
pub(crate) fn expand_home(path: &Path) -> PathBuf {
    match (path.strip_prefix("~"), std::env::var_os("HOME")) {
        (Ok(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> FitConfigBuilder {
        FitConfig::builder()
            .glauber("glauber.root", "nt")
            .data("data.root", "h")
    }

    #[test]
    fn defaults_follow_config_c() {
        let c = base().build().unwrap();
        assert_eq!(c.n_iter, 20);
        assert_eq!(c.mode, Mode::Star);
        assert_eq!((c.fit_min_bin, c.fit_max_bin), (10, 110));
        assert_eq!(c.k, ScanRange::new(0.5, 1.0, 0.01));
        assert_eq!(c.fit_method, FitMethod::Chi2);
        assert!(c.n_threads >= 1);
    }

    #[test]
    fn requires_inputs() {
        assert!(FitConfig::builder().build().is_err());
        assert!(FitConfig::builder().glauber("g", "t").build().is_err());
    }

    #[test]
    fn rejects_bad_values() {
        assert!(base().fit_range(20, 10).build().is_err());
        assert!(base().p_range(0.1, 1.0, 0.1).build().is_err());
        assert!(base().k_range(0., 1., 0.1).build().is_err());
        assert!(base().f_range(0., 1., 0.).build().is_err());
        assert!(base().n_iter(0).build().is_err());
    }

    #[test]
    fn fit_range_in_values() {
        let axis = |nbins, min, max| oxiroot::prelude::Hist::reg(nbins, min, max).float().xaxis;
        let bins = |builder: FitConfigBuilder, axis: &TAxis| {
            let mut c = builder.build().unwrap();
            c.resolve_fit_bins(axis)
                .map(|_| (c.fit_min_bin, c.fit_max_bin))
        };
        let values = |min, max| base().fit_value_range(min, max);
        /* unit bins from 0: [10, 110) is bins 11..=110 */
        let unit = axis(1000, 0., 1000.);
        assert_eq!(bins(values(10., 110.), &unit).unwrap(), (11, 110));
        assert_eq!(bins(values(10.5, 110.5), &unit).unwrap(), (11, 111));
        /* bins of 0.04 GeV: edges and values inside bins */
        let energy = axis(1000, 0., 40.);
        assert_eq!(bins(values(0.4, 28.), &energy).unwrap(), (11, 700));
        assert_eq!(bins(values(0.41, 27.99), &energy).unwrap(), (11, 700));
        /* clipped to the axis; outside is an error */
        assert_eq!(bins(values(-5., 2000.), &unit).unwrap(), (1, 1000));
        assert!(bins(values(1000., 2000.), &unit).is_err());
        /* bin ranges are unchanged */
        assert_eq!(bins(base().fit_range(20, 300), &unit).unwrap(), (20, 300));

        assert!(values(5., 5.).build().is_err());
        assert!(values(5., f64::NAN).build().is_err());
        assert!(values(1., 5.).fit_min_bin(3).build().is_err());
        let parse = |s: &str| FitConfigBuilder::from_ron_str(s).and_then(|b| b.build());
        let c = parse(
            r#"(fit: (glauber_file: "g", glauber_tree: "t", data_file: "d", data_hist: "h",
                      fit_min: 0.4, fit_max: 28))"#,
        )
        .unwrap();
        assert_eq!(c.fit_values, Some((0.4, 28.)));
        assert!(
            parse(r#"(fit: (glauber_file: "g", glauber_tree: "t", data_file: "d", data_hist: "h", fit_min: 0.4))"#)
                .is_err()
        );
    }

    #[test]
    fn shipped_config_ron_matches_config_c() {
        let text = include_str!("../config.ron");
        let c = FitConfigBuilder::from_ron_str(text)
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(c.glauber_tree, "nt_Au3_Au3");
        assert_eq!(c.data_hist, "hRefMult");
        assert_eq!(c.n_iter, 20);
        assert_eq!(c.f, ScanRange::new(0.1, 0.1, 0.01));
        assert_eq!(c.k, ScanRange::new(0.5, 1.0, 0.01));
        assert_eq!(c.p, ScanRange::new(0.001, 0.05, 0.001));
        assert_eq!((c.fit_min_bin, c.fit_max_bin), (10, 110));
        assert_eq!(c.mode, Mode::Star);
        assert_eq!(c.distribution, Distribution::Gamma);
        assert_eq!(c.fit_method, FitMethod::Chi2);
        assert!(!c.glauber_file.starts_with("~"));
    }

    #[test]
    fn ron_partial_and_errors() {
        let c = FitConfigBuilder::from_ron_str(
            r#"(
                other_step: (whatever: [1, 2], mode: 3),
                fit: (glauber_file: "g.root", glauber_tree: "t", data_file: "d.root",
                      data_hist: "h", mode: "hades", n_threads: 3, seed: 5,
                      fit_method: Likelihood),
            )"#,
        )
        .unwrap()
        .build()
        .unwrap();
        assert_eq!(c.mode, Mode::Hades);
        assert_eq!(c.fit_method, FitMethod::Likelihood);
        assert_eq!((c.n_threads, c.seed), (3, Some(5)));
        assert_eq!(c.n_iter, 20);

        assert!(FitConfigBuilder::from_ron_str("(fit: (typo_field: 1))").is_err());
        assert!(FitConfigBuilder::from_ron_str(r#"(fit: (mode: "nope"))"#).is_err());
        assert!(FitConfigBuilder::from_ron_str("(fit: (fit_method: Nope))").is_err());
        assert!(FitConfigBuilder::from_ron_str(r#"(glauber_file: "g.root")"#).is_err());
    }

    #[test]
    fn scan_points() {
        assert_eq!(ScanRange::fixed(0.3).points(), vec![0.3]);
        let pts = ScanRange::new(0.5, 1.0, 0.01).points();
        assert_eq!(pts.len(), 51);
        assert!((pts[50] - 1.0).abs() < 1e-5);
    }
    #[test]
    fn centrality_config() {
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
        assert_eq!(c.min_value, 1.);

        let parse = |s: &str| CentralityConfigBuilder::from_ron_str(s).and_then(|b| b.build());
        assert!(parse(r#"(centrality: (data_hist: "h"))"#).is_err());
        assert!(parse(r#"(centrality: (qa_file: "q", data_hist: "h", n_classes: 0))"#).is_err());
        assert!(
            parse(r#"(centrality: (qa_file: "q", data_hist: "h", table_formats: [Pdf]))"#).is_err()
        );
        assert!(parse(r#"(centrality: (qa_file: "q", data_hist: "h", typo: 1))"#).is_err());
    }
}
