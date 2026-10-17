use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use centrality_glauber_rust::FitConfig;
use centrality_glauber_rust::cli::{print_fit_summary, run_with_progress_bar};
use clap::Parser;

/// Fit a data multiplicity distribution with an MC-Glauber based model.
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// RON configuration file; the `fit` section is used (see config.ron)
    config: PathBuf,
}

fn main() -> ExitCode {
    let start = Instant::now();
    let args = Args::parse();
    println!("fit: using configuration {}", args.config.display());
    let result = FitConfig::from_ron_file(&args.config).and_then(|config| {
        let method = config.fit_method;
        run_with_progress_bar(config).map(|r| (r, method))
    });
    match result {
        Ok((r, method)) => {
            print_fit_summary(&r, method);
            println!("Total time: {:.1?}", start.elapsed());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
