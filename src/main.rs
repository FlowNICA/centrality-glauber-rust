use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use centrality_rust::{Distribution, FitConfig, Mode};
use clap::Parser;

/// Fit a data multiplicity distribution with an MC-Glauber based model.
///
/// Optional parameters default to the values of the original config.c.
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// ROOT file with the MC-Glauber tree
    #[arg(long)]
    glauber_file: PathBuf,
    /// Name of the MC-Glauber tree
    #[arg(long)]
    glauber_tree: String,
    /// ROOT file with the data multiplicity histogram
    #[arg(long)]
    data_file: PathBuf,
    /// Name of the data multiplicity histogram
    #[arg(long)]
    data_hist: String,
    /// Output directory [default: .]
    #[arg(long)]
    out_dir: Option<PathBuf>,
    /// Number of golden section iterations for mu [default: 20]
    #[arg(long)]
    n_iter: Option<u32>,
    /// f scan as MIN:MAX:STEP [default: 0.1:0.1:0.01]
    #[arg(long, value_parser = parse_range)]
    f: Option<(f32, f32, f32)>,
    /// k scan as MIN:MAX:STEP [default: 0.5:1.0:0.01]
    #[arg(long, value_parser = parse_range)]
    k: Option<(f32, f32, f32)>,
    /// Pile-up probability scan as MIN:MAX:STEP [default: 0.001:0.05:0.001]
    #[arg(long, value_parser = parse_range)]
    p: Option<(f32, f32, f32)>,
    /// First bin of the fit range [default: 10]
    #[arg(long)]
    mult_min: Option<usize>,
    /// Last bin of the fit range [default: 110]
    #[arg(long)]
    mult_max: Option<usize>,
    /// Bin width of the Npart/Ncoll histograms [default: 1]
    #[arg(long)]
    bin_size: Option<f32>,
    /// Na mode: Default, PSD, Npart, Ncoll, NpartFast, NcollFast, STAR, HADES [default: STAR]
    #[arg(long)]
    mode: Option<Mode>,
    /// Number of threads [default: all cores]
    #[arg(long)]
    threads: Option<usize>,
    /// Number of Glauber events [default: 10 x data integral in the fit range]
    #[arg(long)]
    n_events: Option<usize>,
    /// Name the per-ancestor multiplicity histogram "nbd" instead of "gamma"
    #[arg(long)]
    nbd: bool,
    /// Random seed for reproducible results
    #[arg(long)]
    seed: Option<u64>,
}

fn parse_range(s: &str) -> Result<(f32, f32, f32), String> {
    let parts: Vec<&str> = s.split(':').collect();
    let [min, max, step] = parts[..] else {
        return Err(format!("expected MIN:MAX:STEP, got '{s}'"));
    };
    let num = |x: &str| x.trim().parse::<f32>().map_err(|e| format!("'{x}': {e}"));
    Ok((num(min)?, num(max)?, num(step)?))
}

fn build_config(args: Args) -> centrality_rust::Result<FitConfig> {
    let mut b = FitConfig::builder()
        .glauber(args.glauber_file, args.glauber_tree)
        .data(args.data_file, args.data_hist);
    if let Some(dir) = args.out_dir {
        b = b.out_dir(dir);
    }
    if let Some(n) = args.n_iter {
        b = b.n_iter(n);
    }
    if let Some((min, max, step)) = args.f {
        b = b.f_range(min, max, step);
    }
    if let Some((min, max, step)) = args.k {
        b = b.k_range(min, max, step);
    }
    if let Some((min, max, step)) = args.p {
        b = b.p_range(min, max, step);
    }
    if let Some(min) = args.mult_min {
        b = b.fit_min_bin(min);
    }
    if let Some(max) = args.mult_max {
        b = b.fit_max_bin(max);
    }
    if let Some(size) = args.bin_size {
        b = b.bin_size(size);
    }
    if let Some(mode) = args.mode {
        b = b.mode(mode);
    }
    if let Some(n) = args.threads {
        b = b.n_threads(n);
    }
    if let Some(n) = args.n_events {
        b = b.n_events(n);
    }
    if args.nbd {
        b = b.distribution(Distribution::Nbd);
    }
    if let Some(seed) = args.seed {
        b = b.seed(seed);
    }
    b.build()
}

fn main() -> ExitCode {
    let start = Instant::now();
    let result = build_config(Args::parse()).and_then(centrality_rust::run);
    match result {
        Ok(r) => {
            println!();
            println!("Results of the fit:");
            println!(
                "f = {}    mu = {}    k = {}    p = {}    chi2 = {}    chi2_error = {}",
                r.best.f, r.best.mu, r.best.k, r.best.p, r.chi2, r.chi2_error
            );
            println!("Total time: {:.1?}", start.elapsed());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
