use std::path::PathBuf;
use std::process::ExitCode;

use centrality_glauber_rust::CentralityConfig;
use clap::Parser;

/// Define centrality classes from the results of `fit`: class borders in
/// multiplicity and <b>, <Npart>, <Ncoll> per class (port of HistoCut.C,
/// CentralityClasses.C and printFinal.C).
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// RON configuration file; the `centrality` section is used (see config.ron)
    config: PathBuf,
}

fn main() -> ExitCode {
    let args = Args::parse();
    println!(
        "define-centrality: using configuration {}",
        args.config.display()
    );
    let result = CentralityConfig::from_ron_file(&args.config)
        .and_then(|config| centrality_glauber_rust::centrality::run(&config));
    match result {
        Ok(r) => {
            println!();
            print!("{}", r.plain_table());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
