use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use centrality_glauber_rust::cli::{print_fit_summary, run_with_progress_bar};
use centrality_glauber_rust::output::QA_FILE_NAME;
use centrality_glauber_rust::{CentralityConfig, FitConfig, Result};
use clap::Parser;

/// Run `fit` and then `define-centrality` with one configuration file that
/// has both the `fit` and the `centrality` sections. The centrality step uses
/// the QA file just written by the fit.
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// RON configuration file with `fit` and `centrality` sections (see config.ron)
    config: PathBuf,
}

/// Whether two paths name the same file, comparing them as absolute paths
/// without `.` components (the QA file need not exist yet).
fn same_path(a: &Path, b: &Path) -> bool {
    let normal = |p: &Path| -> PathBuf {
        p.components()
            .filter(|c| !matches!(c, Component::CurDir))
            .collect()
    };
    match (std::path::absolute(a), std::path::absolute(b)) {
        (Ok(a), Ok(b)) => normal(&a) == normal(&b),
        _ => normal(a) == normal(b),
    }
}

fn run(args: &Args, start: Instant) -> Result<()> {
    /* both sections are read and checked before the fit starts */
    let fit = FitConfig::from_ron_file(&args.config)?;
    let mut centrality = CentralityConfig::from_ron_file(&args.config)?;
    let qa_file = fit.out_dir.join(QA_FILE_NAME);
    if !same_path(&centrality.qa_file, &qa_file) {
        eprintln!(
            "Warning: qa_file of the centrality section ({}) is not the file written by the \
             fit; using {}",
            centrality.qa_file.display(),
            qa_file.display()
        );
        /* data_file defaults to qa_file: the copy of the data in it */
        if centrality.data_file == centrality.qa_file {
            centrality.data_file = qa_file.clone();
        }
        centrality.qa_file = qa_file;
    }

    println!("== fit");
    let method = fit.fit_method;
    let result = run_with_progress_bar(fit)?;
    print_fit_summary(&result, method);
    println!("Fit time: {:.1?}", start.elapsed());

    println!();
    println!("== define-centrality");
    let classes = centrality_glauber_rust::centrality::run(&centrality)?;
    println!();
    print!("{}", classes.plain_table());
    Ok(())
}

fn main() -> ExitCode {
    let start = Instant::now();
    let args = Args::parse();
    println!("run-all: using configuration {}", args.config.display());
    match run(&args, start) {
        Ok(()) => {
            println!("Total time: {:.1?}", start.elapsed());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
