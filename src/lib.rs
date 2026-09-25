//! Rust port of the MC-Glauber multiplicity fitter of the
//! [CentralityFramework](https://github.com/FlowNICA/CentralityFramework).
//!
//! The data multiplicity is fitted with `Na(f; Npart, Ncoll)` ancestors from an
//! MC-Glauber simulation, each producing a Gamma (NBD-like) distributed number
//! of particles with mean `mu` and width parameter `k`, plus pile-up with
//! probability `p`. `f`, `k` and `p` are scanned on a grid, `mu` is found by a
//! golden section search minimizing chi2/ndf.

// `!(x > 0.)` is used on purpose: it also rejects NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod centrality;
pub mod config;
pub mod config_file;
pub mod error;
pub mod fitter;
pub mod glauber;
pub mod mode;
pub mod output;

pub use centrality::{CentralityConfig, CentralityResult, TableFormat};
pub use config::{Distribution, FitConfig, FitConfigBuilder, FitMethod, ScanRange};
pub use error::{Error, Result};
pub use fitter::{FitParams, FitProgress, FitResult, Fitter, ModelHistograms, ScanPoint};
pub use glauber::GlauberEvents;
pub use mode::Mode;

/// Runs the whole fit: reads the inputs, fits, and writes the scan file and
/// the QA file into `config.out_dir`. The fit progress is printed to stdout.
pub fn run(config: FitConfig) -> Result<FitResult> {
    run_with_progress(config, &FitProgress::print)
}

/// Same as [`run`], reporting the fit progress to `progress`.
pub fn run_with_progress(
    config: FitConfig,
    progress: &(dyn Fn(FitProgress) + Sync),
) -> Result<FitResult> {
    std::fs::create_dir_all(&config.out_dir)?;
    let fitter = Fitter::new(config)?;
    let result = fitter.fit_with_progress(progress)?;

    let scan_path = output::scan_file_path(fitter.config());
    output::write_scan(&scan_path, &result.scan)?;

    let model = fitter.model_histograms(&result.best);
    let nbd = fitter.nbd_histogram(&result.best);
    let qa_path = fitter.config().out_dir.join(output::QA_FILE_NAME);
    output::write_qa(
        &qa_path,
        &output::QaOutput {
            npart: fitter.npart_histo(),
            ncoll: fitter.ncoll_histo(),
            data: fitter.data(),
            model: &model,
            nbd: &nbd,
            result: &result,
        },
    )?;
    println!("Scan written to {}", scan_path.display());
    println!("QA histograms written to {}", qa_path.display());
    Ok(result)
}
