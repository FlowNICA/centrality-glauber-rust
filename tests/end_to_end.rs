use std::path::PathBuf;

use centrality_glauber_rust::{FitConfig, FitMethod, Mode};
use oxiroot::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Gamma};

const TRUE_F: f32 = 0.5;
const TRUE_MU: f64 = 0.8;
const TRUE_K: f64 = 1.5;

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "centrality-glauber-rust-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Toy Glauber tree with 600k events and a data histogram with `n_data`
/// entries generated with known parameters.
fn write_inputs(dir: &std::path::Path, n_data: usize) -> (PathBuf, PathBuf) {
    let mut rng = SmallRng::seed_from_u64(1);
    let n = 600_000;
    let mut b = Vec::with_capacity(n);
    let mut npart = Vec::with_capacity(n);
    let mut ncoll = Vec::with_capacity(n);
    for _ in 0..n {
        let bi: f32 = 14. * rng.random::<f32>().sqrt();
        let np = (2. + 198. * (1. - bi / 14.).powi(2) * rng.random_range(0.8..1.2f32)).round();
        b.push(bi);
        npart.push(np);
        ncoll.push((np * (1. + np / 60.)).round());
    }

    let glauber = dir.join("glauber.root");
    Tree::new(
        "nt_toy",
        vec![
            Branch::f32("B", b),
            Branch::f32("Npart", npart.clone()),
            Branch::f32("Ncoll", ncoll.clone()),
            Branch::f32("Ecc2", vec![0.3; n]),
        ],
    )
    .write_root(&glauber, Compression::None)
    .unwrap();

    /* data from the Default mode with independent random numbers */
    let alpha = TRUE_MU * TRUE_K / (TRUE_MU + TRUE_K);
    let theta = (TRUE_K + TRUE_MU) / TRUE_K;
    let mut data = Hist::reg(1000, 0., 1000.).name("hMult").float();
    let mut rng = SmallRng::seed_from_u64(2);
    for i in 0..n_data {
        let j = (i * 7919) % n;
        let na = Mode::Default.n_ancestors(TRUE_F as f64, npart[j] as f64, ncoll[j] as f64) as i64;
        if na > 0 {
            data.fill(
                Gamma::new(na as f64 * alpha, theta)
                    .unwrap()
                    .sample(&mut rng),
            );
        }
    }
    let data_path = dir.join("data.root");
    data.write_root(&data_path, Compression::None).unwrap();
    (glauber, data_path)
}

#[test]
fn fit_recovers_generated_parameters() {
    let dir = temp_dir("e2e");
    let (glauber, data) = write_inputs(&dir, 50_000);
    let out_dir = dir.join("out");

    let config = FitConfig::builder()
        .glauber(&glauber, "nt_toy")
        .data(&data, "hMult")
        .out_dir(&out_dir)
        .mode(Mode::Default)
        .f_range(TRUE_F, TRUE_F, 0.)
        .k_range(1.0, 2.0, 0.25)
        .p_range(0., 0., 0.)
        .fit_range(20, 300)
        .n_iter(15)
        .seed(42)
        .build()
        .unwrap();

    let result = centrality_glauber_rust::run(config.clone()).unwrap();
    println!("{result:?}");
    assert_eq!(result.scan.len(), 5);
    assert!(
        (result.best.mu as f64 - TRUE_MU).abs() < 0.05,
        "mu = {}",
        result.best.mu
    );
    assert!(
        (result.best.k as f64 - TRUE_K).abs() <= 0.25,
        "k = {}",
        result.best.k
    );
    assert!(result.chi2 < 3., "chi2/ndf = {}", result.chi2);

    /* outputs are readable */
    let scan = FileReader::open(centrality_glauber_rust::output::scan_file_path(&config)).unwrap();
    let tree = TreeReader::open(&scan, "test_tree").unwrap();
    assert_eq!(tree.num_entries(), 5);

    let qa = FileReader::open(out_dir.join(centrality_glauber_rust::output::QA_FILE_NAME)).unwrap();
    let fit = TH1::read_root(&qa, "glaub_fit_histo").unwrap();
    let data = TH1::read_root(&qa, "hMult").unwrap();
    let fit_int: f64 = fit.contents[21..=300].iter().sum();
    let data_int: f64 = data.contents[21..=300].iter().sum();
    assert!((fit_int / data_int - 1.).abs() < 1e-3);
    for name in [
        "glaub_plp_histo",
        "glaub_sng_histo",
        "gamma",
        "fNpartHisto",
        "fNcollHisto",
    ] {
        TH1::read_root(&qa, name).unwrap();
    }
    for name in [
        "glaub_plp_ev1ev2",
        "B_VS_Multiplicity",
        "Npart_VS_Multiplicity",
        "Ecc2_VS_Multiplicity",
    ] {
        TH2::read_root(&qa, name).unwrap();
    }
    assert!(TH2::read_root(&qa, "Psi2_VS_Multiplicity").is_err());
    let best = TreeReader::open(&qa, "BestResult").unwrap();
    let mu = best.read_branch(&qa, "mu").unwrap();
    assert_eq!(mu.as_f32().unwrap(), &[result.best.mu]);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn likelihood_fit_recovers_generated_parameters() {
    let dir = temp_dir("likelihood");
    let (glauber, data) = write_inputs(&dir, 50_000);

    let config = FitConfig::builder()
        .glauber(&glauber, "nt_toy")
        .data(&data, "hMult")
        .out_dir(dir.join("out"))
        .mode(Mode::Default)
        .fit_method(FitMethod::Likelihood)
        .f_range(TRUE_F, TRUE_F, 0.)
        .k_range(1.0, 2.0, 0.25)
        .p_range(0., 0., 0.)
        .fit_range(20, 300)
        .n_iter(15)
        .seed(42)
        .build()
        .unwrap();

    let result = centrality_glauber_rust::Fitter::new(config)
        .unwrap()
        .fit()
        .unwrap();
    println!("{result:?}");
    assert!(
        (result.best.mu as f64 - TRUE_MU).abs() < 0.05,
        "mu = {}",
        result.best.mu
    );
    assert!(
        (result.best.k as f64 - TRUE_K).abs() <= 0.25,
        "k = {}",
        result.best.k
    );
    assert!(
        result.chi2 > 0. && result.chi2 < 3.,
        "-2lnL/ndf = {}",
        result.chi2
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn bin_size_does_not_change_the_fit() {
    let dir = temp_dir("bin-size");
    let (glauber, data) = write_inputs(&dir, 50_000);
    let fit = |bin_size| {
        let config = FitConfig::builder()
            .glauber(&glauber, "nt_toy")
            .data(&data, "hMult")
            .mode(Mode::Default)
            .f_range(TRUE_F, TRUE_F, 0.)
            .k_range(1.0, 2.0, 0.5)
            .p_range(0., 0.02, 0.01)
            .fit_range(20, 300)
            .n_iter(5)
            .n_threads(4)
            .bin_size(bin_size)
            .seed(3)
            .build()
            .unwrap();
        let fitter = centrality_glauber_rust::Fitter::new(config).unwrap();
        let h = fitter.npart_histo();
        let nbins = h.xaxis.nbins as usize;
        assert!(((h.xaxis.xmax / nbins as f64) - bin_size).abs() < 1e-9);
        assert_eq!(h.contents[nbins + 1], 0., "Npart maximum in the overflow");
        fitter.fit().unwrap()
    };
    let reference = fit(1.);
    for bin_size in [0.1, 0.3, 2., 3., 7.] {
        assert_eq!(fit(bin_size), reference, "bin_size {bin_size}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn same_seed_gives_same_result() {
    let dir = temp_dir("seed");
    let (glauber, data) = write_inputs(&dir, 50_000);
    let config = |threads| {
        FitConfig::builder()
            .glauber(&glauber, "nt_toy")
            .data(&data, "hMult")
            .out_dir(dir.join("out"))
            .mode(Mode::Default)
            .f_range(TRUE_F, TRUE_F, 0.)
            .k_range(1.0, 2.0, 0.5)
            .p_range(0., 0.02, 0.01)
            .fit_range(20, 300)
            .n_iter(5)
            .n_threads(threads)
            .seed(7)
            .build()
            .unwrap()
    };
    let a = centrality_glauber_rust::Fitter::new(config(4))
        .unwrap()
        .fit()
        .unwrap();
    let b = centrality_glauber_rust::Fitter::new(config(4))
        .unwrap()
        .fit()
        .unwrap();
    assert_eq!(a, b);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn too_few_glauber_events_is_an_error() {
    let dir = temp_dir("few");
    /* ~half of the entries are in the fit range: ~1M Glauber events needed, the tree has 600k */
    let (glauber, data) = write_inputs(&dir, 200_000);
    let config = FitConfig::builder()
        .glauber(&glauber, "nt_toy")
        .data(&data, "hMult")
        .out_dir(dir.join("out"))
        .mode(Mode::Default)
        .fit_range(20, 300)
        .build()
        .unwrap();
    let err = centrality_glauber_rust::Fitter::new(config)
        .err()
        .expect("fit must fail");
    assert!(
        err.to_string().contains("not enough Glauber events"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
