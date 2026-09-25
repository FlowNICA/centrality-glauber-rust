use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use centrality_glauber_rust::{FitConfig, FitProgress};
use clap::Parser;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

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
    let bar = ProgressBar::hidden();
    let progress = |p: FitProgress| match p {
        FitProgress::Start { total } => {
            bar.set_style(
                ProgressStyle::with_template(
                    "{spinner} [{elapsed_precise}] [{wide_bar}] {percent}% {msg} (ETA {eta})",
                )
                .expect("valid progress bar template")
                .progress_chars("=> "),
            );
            bar.set_length(total);
            bar.set_message("initialization");
            bar.reset();
            bar.set_draw_target(ProgressDrawTarget::stderr());
        }
        FitProgress::Step => bar.inc(1),
        FitProgress::Initialized { .. } => {
            bar.set_message("initialized");
            println_above(&bar, p);
        }
        FitProgress::Iteration {
            iter,
            n_iter,
            chi2,
            method,
            ..
        } => {
            bar.set_message(format!(
                "iteration {iter}/{n_iter}, {} = {chi2:.4}",
                method.statistic_name()
            ));
            println_above(&bar, p);
        }
        FitProgress::Finish => bar.finish_and_clear(),
    };
    let result = FitConfig::from_ron_file(&args.config).and_then(|config| {
        let method = config.fit_method;
        centrality_glauber_rust::run_with_progress(config, &progress).map(|r| (r, method))
    });
    bar.finish_and_clear();
    match result {
        Ok((r, method)) => {
            println!();
            println!(
                "Results of the fit ({:?}, chi2 = {}):",
                method,
                method.statistic_name()
            );
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

/// Prints the status line of `p` above the progress bar, or plainly if the bar
/// is not drawn (e.g. stderr is not a terminal).
fn println_above(bar: &ProgressBar, p: FitProgress) {
    let Some(line) = p.status_line() else { return };
    if bar.is_hidden() {
        println!("{line}");
    } else {
        bar.println(line);
    }
}
