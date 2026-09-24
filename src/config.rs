use std::path::PathBuf;

use crate::error::{Error, Result};
use crate::mode::Mode;

/// Maximum number of scan points per parameter.
const MAX_SCAN_POINTS: usize = 100_000;

/// Scan of one fit parameter: `min, min + step, ... <= max`.
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// Which distribution is reported as the per-ancestor multiplicity histogram.
///
/// As in the original framework the sampling always uses a Gamma distribution
/// with mean `mu` and NBD-like parameter `k`; this only names the output
/// histogram (`gamma` or `nbd`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Distribution {
    #[default]
    Gamma,
    Nbd,
}

impl Distribution {
    pub fn histogram_name(self) -> &'static str {
        match self {
            Distribution::Gamma => "gamma",
            Distribution::Nbd => "nbd",
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
    /// Bin width of the `Npart` and `Ncoll` histograms.
    pub bin_size: f32,
    /// Functional form of the number of ancestors.
    pub mode: Mode,
    /// Number of worker threads.
    pub n_threads: usize,
    /// Number of Glauber events used to build the model multiplicity.
    /// `None` means 10 times the data integral in the fit range.
    pub n_events: Option<usize>,
    pub distribution: Distribution,
    /// Seed of the random generator; `None` gives a different seed every run.
    pub seed: Option<u64>,
}

impl FitConfig {
    pub fn builder() -> FitConfigBuilder {
        FitConfigBuilder::default()
    }
}

/// Builder for [`FitConfig`]. The input files and object names are required,
/// everything else defaults to the values of the original `config.c`.
#[derive(Debug, Clone)]
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
    fit_min_bin: usize,
    fit_max_bin: usize,
    bin_size: f32,
    mode: Mode,
    n_threads: Option<usize>,
    n_events: Option<usize>,
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
            fit_min_bin: 10,
            fit_max_bin: 110,
            bin_size: 1.,
            mode: Mode::Star,
            n_threads: None,
            n_events: None,
            distribution: Distribution::Gamma,
            seed: None,
        }
    }
}

impl FitConfigBuilder {
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

    /// Bin range `[min, max]` of the data histogram used for chi2.
    pub fn fit_range(mut self, min: usize, max: usize) -> Self {
        self.fit_min_bin = min;
        self.fit_max_bin = max;
        self
    }

    pub fn fit_min_bin(mut self, min: usize) -> Self {
        self.fit_min_bin = min;
        self
    }

    pub fn fit_max_bin(mut self, max: usize) -> Self {
        self.fit_max_bin = max;
        self
    }

    pub fn bin_size(mut self, bin_size: f32) -> Self {
        self.bin_size = bin_size;
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

    /// Number of Glauber events used to build the model multiplicity;
    /// defaults to 10 times the data integral in the fit range.
    pub fn n_events(mut self, n_events: usize) -> Self {
        self.n_events = Some(n_events);
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
        if self.fit_min_bin >= self.fit_max_bin {
            return Err(Error::Config(format!(
                "fit range: min bin ({}) must be below max bin ({})",
                self.fit_min_bin, self.fit_max_bin
            )));
        }
        if !(self.bin_size > 0.) {
            return Err(Error::Config(format!(
                "bin size must be positive, got {}",
                self.bin_size
            )));
        }
        let n_threads = self
            .n_threads
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        if n_threads == 0 {
            return Err(Error::Config("n_threads must be at least 1".into()));
        }
        if self.n_events == Some(0) {
            return Err(Error::Config("n_events must be at least 1".into()));
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
            fit_min_bin: self.fit_min_bin,
            fit_max_bin: self.fit_max_bin,
            bin_size: self.bin_size,
            mode: self.mode,
            n_threads,
            n_events: self.n_events,
            distribution: self.distribution,
            seed: self.seed,
        })
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
    fn scan_points() {
        assert_eq!(ScanRange::fixed(0.3).points(), vec![0.3]);
        let pts = ScanRange::new(0.5, 1.0, 0.01).points();
        assert_eq!(pts.len(), 51);
        assert!((pts[50] - 1.0).abs() < 1e-5);
    }
}
