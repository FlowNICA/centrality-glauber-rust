use std::ops::Range;
use std::time::{Duration, Instant};

use oxiroot::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution as _, Gamma};
use rayon::prelude::*;

use crate::config::{FitConfig, FitMethod};
use crate::error::{Error, Result};
use crate::glauber::GlauberEvents;
use crate::mode::Mode;

/// chi2/ndf assigned to points which cannot be evaluated.
const CHI2_INVALID: f64 = 1e10;
/// Number of random draws in the per-ancestor multiplicity histogram.
const NBD_SAMPLES: usize = 100_000;
/// Simulated count used for an empty model bin in the likelihood, so that an
/// empty model bin with data gives a large but finite penalty instead of `ln 0`.
const MIN_MODEL_COUNT: f64 = 0.1;
/// Maximum number of bins of the `Npart` and `Ncoll` histograms.
const MAX_RANGE_BINS: usize = 10_000_000;
/// Parameters of the multiplicity model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitParams {
    /// Parameter of the number of ancestors, see [`Mode`].
    pub f: f32,
    /// Mean multiplicity per ancestor.
    pub mu: f32,
    /// NBD-like width parameter of the per-ancestor multiplicity.
    pub k: f32,
    /// Pile-up probability.
    pub p: f32,
}

impl FitParams {
    /// Variance of the per-ancestor multiplicity.
    pub fn sigma(&self) -> f32 {
        (self.mu / self.k + 1.) * self.mu
    }
}

/// Result of the golden section search for one (f, k, p) grid point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanPoint {
    pub params: FitParams,
    /// Fit statistic per degree of freedom: chi2/ndf, or -2lnL/ndf with
    /// [`FitMethod::Likelihood`].
    pub chi2: f32,
    pub chi2_error: f32,
    /// `false` if the point was skipped (e.g. non-positive `mu` range).
    pub valid: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    pub best: FitParams,
    /// Best fit statistic per degree of freedom (see [`ScanPoint::chi2`]).
    pub chi2: f32,
    pub chi2_error: f32,
    /// All scanned (f, k, p) points with their optimal `mu`.
    pub scan: Vec<ScanPoint>,
}

/// Progress of [`Fitter::fit_with_progress`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FitProgress {
    /// The scan starts; `total` work units will be done.
    Start { total: u64 },
    /// One work unit (a chunk of the events for one `mu` evaluation) is done.
    /// Reported from the worker threads.
    Step,
    /// The first two golden section points of all grid points are evaluated.
    Initialized { elapsed: Duration },
    /// Golden section iteration `iter` (1-based) is done.
    Iteration {
        iter: u32,
        n_iter: u32,
        best: FitParams,
        /// Best fit statistic per degree of freedom of `method`.
        chi2: f64,
        method: FitMethod,
        elapsed: Duration,
    },
    /// The scan is done.
    Finish,
}

impl FitProgress {
    /// Status line for the initialization and iteration events.
    pub fn status_line(&self) -> Option<String> {
        match self {
            Self::Initialized { elapsed } => {
                Some(format!("FitGlauber: initialization done ({elapsed:.1?})"))
            }
            Self::Iteration {
                iter,
                n_iter,
                best,
                chi2,
                method,
                elapsed,
            } => Some(format!(
                "FitGlauber: iteration [{iter}/{n_iter}] best {} = {chi2:.4} \
                 (f = {} mu = {:.4} k = {} p = {}) ({elapsed:.1?})",
                method.statistic_name(),
                best.f,
                best.mu,
                best.k,
                best.p
            )),
            _ => None,
        }
    }

    /// Prints the status lines to stdout.
    pub fn print(self) {
        if let Some(line) = self.status_line() {
            println!("{line}");
        }
    }
}

/// Model histograms for one set of parameters, normalized to the data in the
/// fit range.
#[derive(Debug, Clone)]
pub struct ModelHistograms {
    /// Total model multiplicity (`glaub_fit_histo`).
    pub fit: TH1,
    /// Events with pile-up (`glaub_plp_histo`).
    pub pile_up: TH1,
    /// Events without pile-up (`glaub_sng_histo`).
    pub single: TH1,
    /// Multiplicity of the main vs. the pile-up event (`glaub_plp_ev1ev2`).
    pub pile_up_ev1_ev2: TH2,
    /// Glauber observables vs. multiplicity (`<name>_VS_Multiplicity`).
    pub vs_multiplicity: Vec<TH2>,
}

/// Fits the data multiplicity with the MC-Glauber based model: the number of
/// ancestors (from `Npart`, `Ncoll`) times Gamma distributed multiplicities per
/// ancestor, with optional pile-up.
pub struct Fitter {
    config: FitConfig,
    data: TH1,
    events: GlauberEvents,
    n_events: usize,
    /// Last filled data bin + 1.
    n_bins: usize,
    /// Upper edge of `n_bins` data bins.
    max_value: f64,
    axis: ModelAxis,
    npart_histo: TH1,
    ncoll_histo: TH1,
    pool: rayon::ThreadPool,
    seed: u64,
}

impl Fitter {
    /// Reads the data histogram and the Glauber tree given in `config`.
    pub fn new(config: FitConfig) -> Result<Self> {
        let data_file = FileReader::open(&config.data_file).map_err(|e| {
            Error::Input(format!("cannot open {}: {e}", config.data_file.display()))
        })?;
        let data = TH1::read_root(&data_file, &config.data_hist)?;
        let n_events = required_n_events(&data, &config)?;
        let events =
            GlauberEvents::load(&config.glauber_file, &config.glauber_tree, Some(n_events))?;
        Self::with_inputs(config, data, events)
    }

    /// Creates the fitter from already loaded inputs.
    pub fn with_inputs(config: FitConfig, data: TH1, events: GlauberEvents) -> Result<Self> {
        if !data.xaxis.xbins.is_empty() {
            return Err(Error::Input(
                "data histogram must have uniform binning".into(),
            ));
        }
        let n_events = required_n_events(&data, &config)?;
        if events.len() < n_events {
            return Err(Error::Input(format!(
                "not enough Glauber events: {} available, at least {n_events} \
                 (10 x data integral in the fit range) needed",
                events.len()
            )));
        }

        /* last non-empty data bin */
        let nbins_x = data.xaxis.nbins.max(0) as usize;
        let content = |bin: usize| data.contents.get(bin).copied().unwrap_or(0.);
        let mut n_bins = nbins_x;
        while n_bins > 1 && content(n_bins - 1) == 0. {
            n_bins -= 1;
        }
        if n_bins <= 1 {
            return Err(Error::Input("data histogram is empty".into()));
        }
        n_bins += 1;

        let (min, max) = (data.xaxis.xmin, data.xaxis.xmax);
        let width = (max - min) / nbins_x as f64;
        let max_value = min + width * n_bins as f64;

        /*
         * Model multiplicity histograms: same bin width and lower edge as the
         * data histogram (so model bin i matches data bin i) plus extra bins
         * for the tail beyond the last filled data bin
         */
        let model_nbins = n_bins + 50.max((0.3 * n_bins as f64).ceil() as usize);
        let axis = ModelAxis::new(model_nbins, min, min + width * model_nbins as f64);

        let npart_histo = range_histo(
            "fNpartHisto",
            "Npart",
            events.npart_max,
            config.bin_size,
            &events.npart[..n_events],
        )?;
        let ncoll_histo = range_histo(
            "fNcollHisto",
            "Ncoll",
            events.ncoll_max,
            config.bin_size,
            &events.ncoll[..n_events],
        )?;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.n_threads)
            .build()?;
        let seed = config.seed.unwrap_or_else(rand::random);

        println!("Glauber events used: {n_events}");
        println!("Last data bin (fNbins): {n_bins}");
        println!("Maximum multiplicity (fMaxValue): {max_value}");
        println!("Threads: {}", config.n_threads);
        println!("Fit method: {:?}", config.fit_method);

        Ok(Self {
            config,
            data,
            events,
            n_events,
            n_bins,
            max_value,
            axis,
            npart_histo,
            ncoll_histo,
            pool,
            seed,
        })
    }

    pub fn config(&self) -> &FitConfig {
        &self.config
    }

    pub fn data(&self) -> &TH1 {
        &self.data
    }

    pub fn npart_histo(&self) -> &TH1 {
        &self.npart_histo
    }

    pub fn ncoll_histo(&self) -> &TH1 {
        &self.ncoll_histo
    }

    pub fn n_events(&self) -> usize {
        self.n_events
    }

    /// Scans all (f, k, p) grid points and finds the best `mu` for each with a
    /// golden section search.
    ///
    /// All grid points are fitted simultaneously: in each iteration the
    /// multiplicity distributions for all grid points are built in parallel.
    /// The progress is printed to stdout.
    pub fn fit(&self) -> Result<FitResult> {
        self.fit_with_progress(&FitProgress::print)
    }

    /// Same as [`Fitter::fit`], reporting the progress to `progress`.
    pub fn fit_with_progress(&self, progress: &(dyn Fn(FitProgress) + Sync)) -> Result<FitResult> {
        let mode = self.config.mode;
        /* int(TTree::GetMaximum) as in the original; independent of bin_size */
        let npart_max = self.events.npart_max.trunc() as f64;
        let ncoll_max = self.events.ncoll_max.trunc() as f64;

        let mut grid = Vec::new();
        for &f in &self.config.f.points() {
            let mu_max = self.max_value / mode.n_ancestors_max(f as f64, npart_max, ncoll_max);
            for &k in &self.config.k.points() {
                for &p in &self.config.p.points() {
                    let valid =
                        mu_max.is_finite() && mu_max > 0. && k > 0. && (0. ..1.).contains(&p);
                    if !valid {
                        eprintln!("Warning: skipping f = {f} k = {k} p = {p} (mu_max = {mu_max})");
                    }
                    grid.push(GridPoint::new(f, k, p, mu_max, valid));
                }
            }
        }
        println!("FitGlauber: {} (f, k, p) points", grid.len());

        let phi = (1. + 5f64.sqrt()) / 2.;
        let mut evals = Vec::new();
        for (g, gp) in grid.iter_mut().enumerate().filter(|(_, gp)| gp.valid) {
            gp.mu_1 = gp.mu_max - (gp.mu_max - gp.mu_min) / phi;
            gp.mu_2 = gp.mu_min + (gp.mu_max - gp.mu_min) / phi;
            evals.push(Evaluation {
                g,
                mu: gp.mu_1,
                is_mu2: false,
            });
            evals.push(Evaluation {
                g,
                mu: gp.mu_2,
                is_mu2: true,
            });
        }
        if evals.is_empty() {
            return Err(Error::Input("no valid (f, k, p) points to fit".into()));
        }

        /* 2 evaluations per valid grid point at initialization, then 1 per iteration */
        let n_iter = self.config.n_iter;
        let n_valid = evals.len() / 2;
        let total = self.n_units(2 * n_valid) + n_iter as u64 * self.n_units(n_valid);
        progress(FitProgress::Start { total });

        let start = Instant::now();
        let counts = self.build_counts(&grid, &evals, 0, progress);
        self.update_chi2(&mut grid, &evals, &counts);
        progress(FitProgress::Initialized {
            elapsed: start.elapsed(),
        });

        for j in 0..n_iter {
            evals.clear();
            for (g, gp) in grid.iter_mut().enumerate().filter(|(_, gp)| gp.valid) {
                if gp.chi2_mu1 >= gp.chi2_mu2 {
                    gp.mu_min = gp.mu_1;
                    gp.mu_1 = gp.mu_2;
                    gp.mu_2 = gp.mu_min + (gp.mu_max - gp.mu_min) / phi;
                    gp.chi2_mu1 = gp.chi2_mu2;
                    gp.chi2_mu1_error = gp.chi2_mu2_error;
                    evals.push(Evaluation {
                        g,
                        mu: gp.mu_2,
                        is_mu2: true,
                    });
                } else {
                    gp.mu_max = gp.mu_2;
                    gp.mu_2 = gp.mu_1;
                    gp.mu_1 = gp.mu_max - (gp.mu_max - gp.mu_min) / phi;
                    gp.chi2_mu2 = gp.chi2_mu1;
                    gp.chi2_mu2_error = gp.chi2_mu1_error;
                    evals.push(Evaluation {
                        g,
                        mu: gp.mu_1,
                        is_mu2: false,
                    });
                }
            }
            let counts = self.build_counts(&grid, &evals, j as u64 + 1, progress);
            self.update_chi2(&mut grid, &evals, &counts);

            if let Some(best) = best_point(&grid) {
                let (mu, chi2, _) = best.optimum();
                progress(FitProgress::Iteration {
                    iter: j + 1,
                    n_iter,
                    best: FitParams {
                        f: best.f,
                        mu: mu as f32,
                        k: best.k,
                        p: best.p,
                    },
                    chi2,
                    method: self.config.fit_method,
                    elapsed: start.elapsed(),
                });
            }
        }
        progress(FitProgress::Finish);

        let scan: Vec<ScanPoint> = grid
            .iter()
            .map(|gp| {
                let (mu, chi2, chi2_error) = gp.optimum();
                ScanPoint {
                    params: FitParams {
                        f: gp.f,
                        mu: mu as f32,
                        k: gp.k,
                        p: gp.p,
                    },
                    chi2: chi2 as f32,
                    chi2_error: chi2_error as f32,
                    valid: gp.valid,
                }
            })
            .collect();
        let best = scan
            .iter()
            .filter(|s| s.valid)
            .min_by(|a, b| a.chi2.total_cmp(&b.chi2))
            .expect("at least one valid grid point");

        Ok(FitResult {
            best: best.params,
            chi2: best.chi2,
            chi2_error: best.chi2_error,
            scan,
        })
    }

    /// Builds the full set of model histograms for the given parameters,
    /// normalized to the data in the fit range.
    pub fn model_histograms(&self, params: &FitParams) -> ModelHistograms {
        let n = self.n_events;
        let n_workers = self.config.n_threads;
        let sampler = Sampler::new(
            self.config.mode,
            params.f,
            params.mu as f64,
            params.k as f64,
        );

        /* each worker takes a contiguous block: main events first, then its pile-up pool */
        let sim: Vec<Vec<SimEvent>> = self.pool.install(|| {
            (0..n_workers)
                .into_par_iter()
                .map(|w| {
                    let i_start = w * n / n_workers;
                    let plp_stop = (w + 1) * n / n_workers;
                    let i_stop =
                        i_start + ((plp_stop - i_start) as f64 * (1. - params.p as f64)) as usize;
                    let mut rng = self.rng(u64::MAX, w, 0);
                    let mut out = Vec::with_capacity(i_stop - i_start);
                    self.simulate(
                        &sampler,
                        params.p,
                        i_start..i_stop,
                        i_stop..plp_stop,
                        &mut rng,
                        |i, n_hits, n_plp| {
                            out.push(SimEvent { i, n_hits, n_plp });
                        },
                    );
                    out
                })
                .collect()
        });

        let axis = &self.axis;
        let model_th1 = |name: &str| {
            Hist::reg(axis.nbins as i32, axis.min, axis.max)
                .label("nHits")
                .name(name)
                .float()
        };
        let mut fit = model_th1("glaub_fit_histo");
        let mut pile_up = model_th1("glaub_plp_histo");
        let mut single = model_th1("glaub_sng_histo");
        let mut pile_up_ev1_ev2 = Hist::reg(axis.nbins as i32, axis.min, axis.max)
            .label("nHits 1")
            .reg(axis.nbins as i32, axis.min, axis.max)
            .label("nHits 2")
            .name("glaub_plp_ev1ev2")
            .title("Multiplicity ev1 vs Multiplicity ev2")
            .float();
        let mut vs_multiplicity: Vec<TH2> = self
            .events
            .observables
            .iter()
            .map(|obs| {
                let (nbins, lo, hi) = obs.binning;
                Hist::reg(axis.nbins as i32, axis.min, axis.max)
                    .label("nHits")
                    .reg(nbins, lo, hi)
                    .label(obs.label)
                    .name(format!("{}_VS_Multiplicity", obs.name))
                    .title(format!("{} VS Multiplicity", obs.label))
                    .float()
            })
            .collect();

        for ev in sim.iter().flatten() {
            match ev.n_plp {
                Some(n_plp) => {
                    pile_up.fill(ev.n_hits);
                    pile_up_ev1_ev2.fill(ev.n_hits - n_plp, n_plp);
                }
                None => single.fill(ev.n_hits),
            }
            fit.fill(ev.n_hits);
            for (h, obs) in vs_multiplicity.iter_mut().zip(&self.events.observables) {
                h.fill(ev.n_hits, obs.values[ev.i] as f64);
            }
        }

        /* normalize to the data in the fit range */
        let (low, high) = self.chi2_bins();
        let model_int: f64 = (low + 1..=high).map(|b| fit.contents[b]).sum();
        let data_int: f64 = (low + 1..=high).map(|b| self.data_content(b)).sum();
        if model_int > 0. {
            let scale = data_int / model_int;
            fit.scale(scale);
            pile_up.scale(scale);
            single.scale(scale);
            pile_up_ev1_ev2.scale(scale);
        } else {
            eprintln!("Warning: empty model histogram in the fit range, not normalized");
        }

        ModelHistograms {
            fit,
            pile_up,
            single,
            pile_up_ev1_ev2,
            vs_multiplicity,
        }
    }

    /// Distribution of the multiplicity from a single ancestor.
    pub fn nbd_histogram(&self, params: &FitParams) -> TH1 {
        let nbins = ((params.mu as f64 + 1.) * 3.).max(10.) as i32;
        let mut h = Hist::reg(nbins, 0., nbins as f64)
            .name(self.config.distribution.histogram_name())
            .float();
        let sampler = Sampler::new(
            self.config.mode,
            params.f,
            params.mu as f64,
            params.k as f64,
        );
        let mut rng = self.rng(u64::MAX - 1, 0, 0);
        for _ in 0..NBD_SAMPLES {
            h.fill(sampler.sum_of_gammas(1, &mut rng));
        }
        h
    }

    /// Simulates the events `main`; with probability `p` a pile-up partner
    /// from `pool` (taken cyclically) is added. Calls `emit(i, n_hits, n_plp)`.
    fn simulate(
        &self,
        sampler: &Sampler,
        p: f32,
        main: Range<usize>,
        pool: Range<usize>,
        rng: &mut SmallRng,
        mut emit: impl FnMut(usize, f64, Option<f64>),
    ) {
        let npart = &self.events.npart;
        let ncoll = &self.events.ncoll;
        let mut next_plp = pool.start;
        for i in main {
            let mut n_hits = sampler.n_hits(npart[i], ncoll[i], rng);
            let mut n_plp = None;
            if p > 1e-10 && rng.random::<f32>() <= p && !pool.is_empty() {
                if !pool.contains(&next_plp) {
                    next_plp = pool.start;
                }
                let j = next_plp;
                next_plp += 1;
                let n = sampler.n_hits(npart[j], ncoll[j], rng);
                n_hits += n;
                n_plp = Some(n);
            }
            emit(i, n_hits, n_plp);
        }
    }

    /// Model multiplicity counts for all evaluations. Work units are
    /// (evaluation, chunk of events); the events are split into chunks only if
    /// there are fewer evaluations than threads.
    fn build_counts(
        &self,
        grid: &[GridPoint],
        evals: &[Evaluation],
        stream: u64,
        progress: &(dyn Fn(FitProgress) + Sync),
    ) -> Vec<Vec<f64>> {
        let n_events = self.n_events;
        let n_chunks = self.n_chunks(evals.len());
        let units: Vec<(usize, usize)> = (0..n_chunks)
            .flat_map(|c| (0..evals.len()).map(move |e| (e, c)))
            .collect();

        let partial: Vec<(usize, Vec<f64>)> = self.pool.install(|| {
            units
                .par_iter()
                .map(|&(e, c)| {
                    let eval = &evals[e];
                    let gp = &grid[eval.g];
                    let n_main = (n_events as f64 * (1. - gp.p as f64)) as usize;
                    let n_plp = n_events - n_main;
                    let main = c * n_main / n_chunks..(c + 1) * n_main / n_chunks;
                    let pool = n_main + c * n_plp / n_chunks..n_main + (c + 1) * n_plp / n_chunks;

                    let sampler = Sampler::new(self.config.mode, gp.f, eval.mu, gp.k as f64);
                    let mut rng = self.rng(stream, e, c);
                    let mut counts = vec![0.; self.axis.n_cells()];
                    self.simulate(&sampler, gp.p, main, pool, &mut rng, |_, n_hits, _| {
                        counts[self.axis.find_bin(n_hits)] += 1.;
                    });
                    progress(FitProgress::Step);
                    (e, counts)
                })
                .collect()
        });

        let mut counts = vec![vec![0.; self.axis.n_cells()]; evals.len()];
        for (e, c) in partial {
            for (total, x) in counts[e].iter_mut().zip(c) {
                *total += x;
            }
        }
        counts
    }

    /// Number of event chunks per evaluation in [`Fitter::build_counts`].
    fn n_chunks(&self, n_evals: usize) -> usize {
        self.config.n_threads.div_ceil(n_evals).max(1)
    }

    /// Number of work units of [`Fitter::build_counts`] for `n_evals` evaluations.
    fn n_units(&self, n_evals: usize) -> u64 {
        (n_evals * self.n_chunks(n_evals)) as u64
    }

    fn update_chi2(&self, grid: &mut [GridPoint], evals: &[Evaluation], counts: &[Vec<f64>]) {
        for (eval, c) in evals.iter().zip(counts) {
            let (chi2, error) = self.statistic(c);
            let gp = &mut grid[eval.g];
            if eval.is_mu2 {
                gp.chi2_mu2 = chi2;
                gp.chi2_mu2_error = error;
            } else {
                gp.chi2_mu1 = chi2;
                gp.chi2_mu1_error = error;
            }
        }
    }

    /// First and last bin of the chi2 range.
    fn chi2_bins(&self) -> (usize, usize) {
        (
            self.config.fit_min_bin,
            self.config.fit_max_bin.min(self.n_bins),
        )
    }

    fn data_content(&self, bin: usize) -> f64 {
        self.data.contents.get(bin).copied().unwrap_or(0.)
    }

    /// chi2/ndf and its error of the model counts normalized to the data.
    /// Bins with empty data are not used, empty model bins with non-empty data
    /// are; ndf is the number of used bins.
    fn chi2(&self, counts: &[f64]) -> (f64, f64) {
        let (low, high) = self.chi2_bins();
        let Some(scale) = self.model_scale(counts) else {
            return (CHI2_INVALID, 0.);
        };

        let mut sum_chi2 = 0.;
        let mut sum_error = 0.;
        let mut ndf = 0;
        for (bin, &count) in counts.iter().enumerate().take(high + 1).skip(low) {
            let data = self.data_content(bin);
            if data < 1. {
                continue;
            }
            let data_error = self.data.bin_error(bin);
            let model = count * scale;
            let model_error = count.sqrt() * scale;
            let error2 = data_error.powi(2) + model_error.powi(2);
            if !(error2 > 0.) {
                continue;
            }
            let diff = model - data;
            sum_chi2 += diff.powi(2) / error2;
            sum_error += (diff * (model_error - data_error) / error2).powi(2);
            ndf += 1;
        }
        if ndf == 0 {
            return (CHI2_INVALID, 0.);
        }
        (sum_chi2 / ndf as f64, 2. * sum_error.sqrt() / ndf as f64)
    }

    /// Poisson likelihood ratio chi2 `-2 ln(L / L_saturated) / ndf` of the model
    /// counts normalized to the data, and its error from the model statistics.
    /// All bins of the chi2 range are used (also those with empty data), so
    /// ndf does not depend on the model and minimizing this maximizes ln L.
    fn likelihood_chi2(&self, counts: &[f64]) -> (f64, f64) {
        let (low, high) = self.chi2_bins();
        let Some(scale) = self.model_scale(counts) else {
            return (CHI2_INVALID, 0.);
        };

        if high < low {
            return (CHI2_INVALID, 0.);
        }

        let mut sum = 0.;
        let mut sum_error = 0.;
        for (bin, &count) in counts.iter().enumerate().take(high + 1).skip(low) {
            let data = self.data_content(bin);
            let model = count.max(MIN_MODEL_COUNT) * scale;
            sum += model - data;
            if data > 0. {
                sum += data * (data / model).ln();
            }
            /* d(-2 ln L)/dm times the statistical error of the model */
            let model_error = count.sqrt() * scale;
            sum_error += (2. * (1. - data / model) * model_error).powi(2);
        }
        let ndf = (high - low + 1) as f64;
        (2. * sum / ndf, sum_error.sqrt() / ndf)
    }

    /// Fit statistic per degree of freedom of the configured method, and its error.
    fn statistic(&self, counts: &[f64]) -> (f64, f64) {
        match self.config.fit_method {
            FitMethod::Chi2 => self.chi2(counts),
            FitMethod::Likelihood => self.likelihood_chi2(counts),
        }
    }

    /// Factor normalizing the model counts to the data in the fit range, or
    /// `None` if the model is empty there.
    fn model_scale(&self, counts: &[f64]) -> Option<f64> {
        let (low, high) = self.chi2_bins();
        let model_int: f64 = (low + 1..=high).map(|b| counts[b]).sum();
        let data_int: f64 = (low + 1..=high).map(|b| self.data_content(b)).sum();
        (model_int != 0.).then(|| data_int / model_int)
    }

    /// Independent random stream for (stream, a, b).
    fn rng(&self, stream: u64, a: usize, b: usize) -> SmallRng {
        let mut s = splitmix64(self.seed ^ splitmix64(stream));
        s = splitmix64(s ^ a as u64);
        s = splitmix64(s ^ (b as u64).rotate_left(32));
        SmallRng::seed_from_u64(s)
    }
}

/// Number of Glauber events used to build the model multiplicity: 10 times the
/// data integral in the fit range.
fn required_n_events(data: &TH1, config: &FitConfig) -> Result<usize> {
    let last = config
        .fit_max_bin
        .min(data.contents.len().saturating_sub(1));
    let integral: f64 = data
        .contents
        .get(config.fit_min_bin..=last)
        .map_or(0., |c| c.iter().sum());
    let n = (10. * integral.trunc()) as usize;
    if n == 0 {
        return Err(Error::Input(
            "data histogram is empty in the fit range".into(),
        ));
    }
    Ok(n)
}

/// Histogram with bins of exactly `width` starting at 0, with as many bins as
/// needed to include `max` (the upper edge is exclusive, so a value equal to
/// `max` must not fall into the overflow).
fn range_histo(name: &str, title: &str, max: f32, width: f64, values: &[f32]) -> Result<TH1> {
    /* the tolerance keeps e.g. max = 0.9, width = 0.3 at 3 full bins + 1 */
    let n = (max.max(0.) as f64 / width + 1e-6).floor() + 1.;
    if n > MAX_RANGE_BINS as f64 {
        return Err(Error::Config(format!(
            "bin_size {width} gives {n} bins for {title} up to {max}, maximum is {MAX_RANGE_BINS}"
        )));
    }
    let nbins = n as usize;
    let mut h = Hist::reg(nbins as i32, 0., nbins as f64 * width)
        .name(name)
        .title(title)
        .float();
    for &v in values {
        h.fill(v as f64);
    }
    Ok(h)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Valid grid point with the lowest chi2/ndf.
fn best_point(grid: &[GridPoint]) -> Option<&GridPoint> {
    grid.iter()
        .filter(|gp| gp.valid)
        .min_by(|a, b| a.optimum().1.total_cmp(&b.optimum().1))
}

/// Binning of the model multiplicity histograms.
#[derive(Debug, Clone, Copy)]
struct ModelAxis {
    nbins: usize,
    min: f64,
    max: f64,
    bins_per_unit: f64,
}

impl ModelAxis {
    fn new(nbins: usize, min: f64, max: f64) -> Self {
        Self {
            nbins,
            min,
            max,
            bins_per_unit: nbins as f64 / (max - min),
        }
    }

    /// Number of cells including under- and overflow.
    fn n_cells(&self) -> usize {
        self.nbins + 2
    }

    /// Same as `TAxis::FindFixBin`.
    fn find_bin(&self, x: f64) -> usize {
        if !(x >= self.min) {
            0
        } else if !(x < self.max) {
            self.nbins + 1
        } else {
            (1 + ((x - self.min) * self.bins_per_unit) as usize).min(self.nbins)
        }
    }
}

/// Gamma distributed multiplicity per ancestor, with mean `mu` and NBD-like `k`.
#[derive(Debug, Clone, Copy)]
struct Sampler {
    mode: Mode,
    f: f64,
    alpha: f64,
    theta: f64,
}

impl Sampler {
    fn new(mode: Mode, f: f32, mu: f64, k: f64) -> Self {
        Self {
            mode,
            f: f as f64,
            alpha: mu * k / (mu + k),
            theta: (k + mu) / k,
        }
    }

    fn n_hits(&self, npart: f32, ncoll: f32, rng: &mut SmallRng) -> f64 {
        let na = self.mode.n_ancestors(self.f, npart as f64, ncoll as f64);
        self.sum_of_gammas(na as i64, rng)
    }

    /// The sum of `n` i.i.d. Gamma(alpha, theta) draws is one Gamma(n*alpha, theta) draw.
    fn sum_of_gammas(&self, n: i64, rng: &mut SmallRng) -> f64 {
        if n <= 0 || !(self.alpha > 0.) || !(self.theta > 0.) {
            return 0.;
        }
        Gamma::new(n as f64 * self.alpha, self.theta).map_or(0., |g| g.sample(rng))
    }
}

/// Golden section state of a single (f, k, p) grid point.
#[derive(Debug, Clone)]
struct GridPoint {
    f: f32,
    k: f32,
    p: f32,
    mu_min: f64,
    mu_max: f64,
    mu_1: f64,
    mu_2: f64,
    chi2_mu1: f64,
    chi2_mu2: f64,
    chi2_mu1_error: f64,
    chi2_mu2_error: f64,
    valid: bool,
}

impl GridPoint {
    fn new(f: f32, k: f32, p: f32, mu_max: f64, valid: bool) -> Self {
        Self {
            f,
            k,
            p,
            mu_min: 0.,
            mu_max,
            mu_1: 0.,
            mu_2: 0.,
            chi2_mu1: CHI2_INVALID,
            chi2_mu2: CHI2_INVALID,
            chi2_mu1_error: 0.,
            chi2_mu2_error: 0.,
            valid,
        }
    }

    /// (mu, chi2, chi2_error) of the better golden section point.
    fn optimum(&self) -> (f64, f64, f64) {
        if self.chi2_mu1 < self.chi2_mu2 {
            (self.mu_1, self.chi2_mu1, self.chi2_mu1_error)
        } else {
            (self.mu_2, self.chi2_mu2, self.chi2_mu2_error)
        }
    }
}

/// Multiplicity to be built with a given `mu` for grid point `g`.
#[derive(Debug, Clone, Copy)]
struct Evaluation {
    g: usize,
    mu: f64,
    is_mu2: bool,
}

/// Multiplicity of a single simulated event.
#[derive(Debug, Clone, Copy)]
struct SimEvent {
    /// Index of the Glauber event.
    i: usize,
    /// Total multiplicity (with pile-up).
    n_hits: f64,
    /// Multiplicity of the pile-up partner, if pile-up happened.
    n_plp: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_histo_bins_have_exactly_bin_size() {
        /* integer values 0..=394 as in a Glauber tree */
        let values: Vec<f32> = (0..=394).map(|v| v as f32).collect();
        for tenths in [1u32, 3, 5, 7, 10, 20, 30, 70, 10_000] {
            let width = tenths as f64 / 10.;
            let h = range_histo("h", "t", 394., width, &values).unwrap();
            let nbins = h.xaxis.nbins as usize;
            assert_eq!(h.xaxis.xmin, 0.);
            assert!(
                ((h.xaxis.xmax / nbins as f64) - width).abs() < 1e-9,
                "width {width}: bin width {}",
                h.xaxis.xmax / nbins as f64
            );
            /* no under/overflow: the maximum is in the last bin */
            assert_eq!(
                (h.contents[0], h.contents[nbins + 1]),
                (0., 0.),
                "width {width}"
            );
            /* every integer is in its exact bin floor(10 v / tenths) + 1 */
            let mut expected = vec![0.; nbins + 2];
            for v in 0..=394u32 {
                expected[(10 * v / tenths) as usize + 1] += 1.;
            }
            assert_eq!(h.contents, expected, "width {width}");
        }
    }

    #[test]
    fn range_histo_edge_cases() {
        /* max on a bin edge despite rounding (33 / 1.1 = 29.999999999999996) */
        let h = range_histo("h", "t", 33., 1.1, &[33.]).unwrap();
        assert_eq!(h.xaxis.nbins, 31);
        assert_eq!(h.contents[32], 0., "maximum in the overflow");
        /* degenerate trees still give one bin of the requested width */
        let h = range_histo("h", "t", 0., 2., &[0.]).unwrap();
        assert_eq!((h.xaxis.nbins, h.xaxis.xmax), (1, 2.));
        assert!(range_histo("h", "t", 400., 1e-6, &[]).is_err());
    }
}
