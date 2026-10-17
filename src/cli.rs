//! Console output shared by the executables in `src/bin`.

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::config::{FitConfig, FitMethod};
use crate::error::Result;
use crate::fitter::{FitProgress, FitResult};

/// [`crate::run`] with a progress bar on stderr (if it is a terminal) and
/// the status lines on stdout.
pub fn run_with_progress_bar(config: FitConfig) -> Result<FitResult> {
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
    let result = crate::run_with_progress(config, &progress);
    bar.finish_and_clear();
    result
}

/// Prints the best fit parameters.
pub fn print_fit_summary(result: &FitResult, method: FitMethod) {
    let r = result;
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
