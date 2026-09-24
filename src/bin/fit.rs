use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use centrality_rust::FitConfig;
use clap::Parser;

/// Fit a data multiplicity distribution with an MC-Glauber based model.
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// RON configuration file with all fit parameters (see config.ron)
    config: PathBuf,
}

fn main() -> ExitCode {
    let start = Instant::now();
    let args = Args::parse();
    println!("fit: using configuration {}", args.config.display());
    let result = FitConfig::from_ron_file(&args.config).and_then(centrality_rust::run);
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
