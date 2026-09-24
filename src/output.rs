use std::path::{Path, PathBuf};

use oxiroot::prelude::*;

use crate::config::FitConfig;
use crate::error::Result;
use crate::fitter::{FitResult, ModelHistograms, ScanPoint};

/// Name of the QA file with the data, the best fit and the result tree.
pub const QA_FILE_NAME: &str = "glauber_qa.root";

/// `fit_<f0>_<k0>_<k1>_<p0>_<fit min bin>.root`, as in the original framework.
pub fn scan_file_path(config: &FitConfig) -> PathBuf {
    config.out_dir.join(format!(
        "fit_{:4.2}_{:4.2}_{:4.2}_{:4.2}_{}.root",
        config.f.min, config.k.min, config.k.max, config.p.min, config.fit_min_bin
    ))
}

/// Writes all scanned (f, k, p) points with their optimal `mu` into the tree
/// `test_tree` (branches `f`, `mu`, `k`, `p`, `chi2`, `chi2_error`, `sigma`).
pub fn write_scan(path: &Path, scan: &[ScanPoint]) -> Result<()> {
    let column = |get: fn(&ScanPoint) -> f32| scan.iter().map(get).collect::<Vec<_>>();
    let tree = Tree::new(
        "test_tree",
        vec![
            Branch::f32("f", column(|s| s.params.f)),
            Branch::f32("mu", column(|s| s.params.mu)),
            Branch::f32("k", column(|s| s.params.k)),
            Branch::f32("p", column(|s| s.params.p)),
            Branch::f32("chi2", column(|s| s.chi2)),
            Branch::f32("chi2_error", column(|s| s.chi2_error)),
            Branch::f32("sigma", column(|s| s.params.sigma())),
        ],
    );
    FileWriter::create(path)
        .put(tree)
        .write(Compression::default())?;
    Ok(())
}

/// Everything written to the QA file.
pub struct QaOutput<'a> {
    pub npart: &'a TH1,
    pub ncoll: &'a TH1,
    pub data: &'a TH1,
    pub model: &'a ModelHistograms,
    pub nbd: &'a TH1,
    pub result: &'a FitResult,
}

/// Writes the Glauber distributions, the data, the best fit histograms and
/// the tree `BestResult` with the optimal parameters.
pub fn write_qa(path: &Path, qa: &QaOutput) -> Result<()> {
    let best = &qa.result.best;
    let best_result = Tree::new(
        "BestResult",
        vec![
            Branch::f32("mu", vec![best.mu]),
            Branch::f32("f", vec![best.f]),
            Branch::f32("k", vec![best.k]),
            Branch::f32("p", vec![best.p]),
            Branch::f32("chi2", vec![qa.result.chi2]),
            Branch::f32("chi2_error", vec![qa.result.chi2_error]),
        ],
    );

    let model = qa.model;
    let mut writer = FileWriter::create(path)
        .add(qa.ncoll)
        .add(qa.npart)
        .add(qa.data)
        .add(&model.fit)
        .add(&model.pile_up)
        .add(&model.single)
        .add(&model.pile_up_ev1_ev2);
    for h in &model.vs_multiplicity {
        writer = writer.add(h);
    }
    writer
        .add(qa.nbd)
        .put(best_result)
        .write(Compression::default())?;
    Ok(())
}
