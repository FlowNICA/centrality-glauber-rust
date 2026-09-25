use std::path::PathBuf;

use centrality_glauber_rust::Distribution as MultDistribution;
use centrality_glauber_rust::{FitConfig, FitMethod, Mode};
use oxiroot::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Gamma, Poisson};

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
/// entries generated with known parameters, with Gamma or (`nbd`) NBD
/// distributed multiplicity per ancestor.
fn write_inputs(dir: &std::path::Path, n_data: usize, nbd: bool) -> (PathBuf, PathBuf) {
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
        if na > 0 && nbd {
            /* each ancestor separately: NBD(mu, k) as a Gamma-Poisson mixture */
            let gamma = Gamma::new(TRUE_K, TRUE_MU / TRUE_K).unwrap();
            let n_hits: f64 = (0..na)
                .map(|_| Poisson::new(gamma.sample(&mut rng)).map_or(0., |d| d.sample(&mut rng)))
                .sum();
            data.fill(n_hits);
        } else if na > 0 {
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
    let (glauber, data) = write_inputs(&dir, 50_000, false);
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
    let (glauber, data) = write_inputs(&dir, 50_000, false);

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
fn nbd_fit_uses_nbd_multiplicities() {
    let dir = temp_dir("nbd");
    let (glauber, data) = write_inputs(&dir, 50_000, true);
    let out_dir = dir.join("out");

    let config = FitConfig::builder()
        .glauber(&glauber, "nt_toy")
        .data(&data, "hMult")
        .out_dir(&out_dir)
        .mode(Mode::Default)
        .distribution(MultDistribution::Nbd)
        .f_range(TRUE_F, TRUE_F, 0.)
        .k_range(1.0, 2.0, 0.25)
        .p_range(0., 0., 0.)
        .fit_range(20, 300)
        .n_iter(15)
        .seed(42)
        .build()
        .unwrap();

    let result = centrality_glauber_rust::run(config).unwrap();
    println!("{:?} chi2/ndf = {}", result.best, result.chi2);
    /*
     * Loose: the golden section search on the noisy simulated chi2 pulls mu
     * low by a few 0.01 for Gamma and NBD alike, and k is barely constrained
     * by this toy; the NBD sampling itself is checked below and in the
     * fitter unit tests
     */
    assert!(
        (result.best.mu as f64 - TRUE_MU).abs() < 0.1,
        "mu = {}",
        result.best.mu
    );
    assert!(result.chi2 < 1.5, "chi2/ndf = {}", result.chi2);

    /*
     * The per-ancestor histogram is NBD(mu, k): integer values, so bin
     * [0, 1) holds exactly P(0) = (k / (k + mu))^k (a Gamma sample would put
     * ~0.66 there instead of ~0.53), and the mean is mu
     */
    let qa = FileReader::open(out_dir.join(centrality_glauber_rust::output::QA_FILE_NAME)).unwrap();
    let nbd = TH1::read_root(&qa, "nbd").unwrap();
    let total: f64 = nbd.contents.iter().sum();
    let (mu, k) = (result.best.mu as f64, result.best.k as f64);
    let p0 = (k / (k + mu)).powf(k);
    assert!(
        (nbd.contents[1] / total - p0).abs() < 0.01,
        "P(0) = {}, expected {p0}",
        nbd.contents[1] / total
    );
    let mean = (1..=nbd.xaxis.nbins as usize)
        .map(|b| (b - 1) as f64 * nbd.contents[b])
        .sum::<f64>()
        / total;
    assert!((mean - mu).abs() < 0.02, "mean {mean}, mu {mu}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Mean and standard deviation of a multiplicity histogram with unit bins,
/// taking the lower bin edge (the integer value for NBD) for each bin.
fn mean_std(h: &TH1) -> (f64, f64) {
    let (mut s0, mut s1, mut s2) = (0., 0., 0.);
    for b in 1..=h.xaxis.nbins as usize {
        let (x, c) = ((b - 1) as f64, h.contents[b]);
        s0 += c;
        s1 += c * x;
        s2 += c * x * x;
    }
    let mean = s1 / s0;
    (mean, (s2 / s0 - mean * mean).sqrt())
}

/// Sanity check: the per-ancestor Gamma has the same mean and variance as
/// NBD(mu, k), and a sum of many ancestors is nearly the same for both, so
/// they must give nearly the same model and fit. The only systematic
/// difference is the discreteness: NBD values are integers at the lower edge
/// of the unit bins, Gamma values are spread over the bins (+0.5 on average).
#[test]
fn gamma_and_nbd_agree_for_integer_k() {
    let dir = temp_dir("gamma-nbd");
    let (glauber, data) = write_inputs(&dir, 50_000, false);
    let config = |distribution, k: f32, seed| {
        FitConfig::builder()
            .glauber(&glauber, "nt_toy")
            .data(&data, "hMult")
            .mode(Mode::Default)
            .distribution(distribution)
            .f_range(TRUE_F, TRUE_F, 0.)
            .k_range(k, k, 0.)
            .p_range(0., 0., 0.)
            .fit_range(20, 300)
            .n_iter(20)
            .seed(seed)
            .build()
            .unwrap()
    };

    for k in [1., 2., 5.] {
        /* model: same width, NBD higher by the discreteness shift of ~0.5 */
        let params = centrality_glauber_rust::FitParams {
            f: TRUE_F,
            mu: TRUE_MU as f32,
            k,
            p: 0.,
        };
        let model = |distribution| {
            let fitter = centrality_glauber_rust::Fitter::new(config(distribution, k, 1)).unwrap();
            mean_std(&fitter.model_histograms(&params).fit)
        };
        let (gamma, nbd) = (model(MultDistribution::Gamma), model(MultDistribution::Nbd));
        println!("k = {k}: model (mean, std) Gamma {gamma:?}, NBD {nbd:?}");
        assert!(
            (nbd.0 - gamma.0 - 0.5).abs() < 0.1,
            "k = {k}: mean NBD - Gamma = {}",
            nbd.0 - gamma.0
        );
        assert!(
            (nbd.1 / gamma.1 - 1.).abs() < 0.01,
            "k = {k}: std NBD / Gamma = {}",
            nbd.1 / gamma.1
        );

        /*
         * Fit with k fixed: a single fit scatters by ~0.02 in mu (simulation
         * noise in the golden section search), so compare averages over seeds
         * (uncertainty of the difference ~0.008)
         */
        let n_seeds = 12;
        let mean_mu = |distribution| {
            (0..n_seeds)
                .map(|seed| {
                    centrality_glauber_rust::Fitter::new(config(distribution, k, 7 * seed + 3))
                        .unwrap()
                        .fit_with_progress(&|_| {})
                        .unwrap()
                        .best
                        .mu as f64
                })
                .sum::<f64>()
                / n_seeds as f64
        };
        let (gamma_mu, nbd_mu) = (
            mean_mu(MultDistribution::Gamma),
            mean_mu(MultDistribution::Nbd),
        );
        println!("k = {k}: mean fitted mu Gamma {gamma_mu:.4}, NBD {nbd_mu:.4}");
        assert!(
            (nbd_mu - gamma_mu).abs() < 0.03,
            "k = {k}: mu Gamma {gamma_mu}, NBD {nbd_mu}"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn bin_size_does_not_change_the_fit() {
    let dir = temp_dir("bin-size");
    let (glauber, data) = write_inputs(&dir, 50_000, false);
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
    let (glauber, data) = write_inputs(&dir, 50_000, false);
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
    let (glauber, data) = write_inputs(&dir, 200_000, false);
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

#[test]
fn centrality_classes_from_the_fit() {
    use centrality_glauber_rust::centrality;
    use centrality_glauber_rust::{CentralityConfig, TableFormat};

    let dir = temp_dir("centrality");
    let (glauber, data) = write_inputs(&dir, 50_000, false);
    let out_dir = dir.join("out");
    let fit_config = FitConfig::builder()
        .glauber(&glauber, "nt_toy")
        .data(&data, "hMult")
        .out_dir(&out_dir)
        .mode(Mode::Default)
        .f_range(TRUE_F, TRUE_F, 0.)
        .k_range(TRUE_K as f32, TRUE_K as f32, 0.)
        .p_range(0.01, 0.01, 0.)
        .fit_range(20, 300)
        .n_iter(10)
        .seed(5)
        .build()
        .unwrap();
    centrality_glauber_rust::run(fit_config).unwrap();

    let qa_file = out_dir.join(centrality_glauber_rust::output::QA_FILE_NAME);
    let config = CentralityConfig::builder()
        .qa_file(&qa_file)
        .data(None, "hMult")
        .out_dir(&out_dir)
        .n_classes(10)
        .table_formats(vec![TableFormat::Csv, TableFormat::Tex, TableFormat::Cpp])
        .build()
        .unwrap();
    let result = centrality::run(&config).unwrap();
    print!("{}", result.plain_table());
    let classes = &result.classes;
    assert_eq!(classes.len(), 10);

    /* each class holds 10% of the single events, up to one boundary bin */
    let qa = FileReader::open(&qa_file).unwrap();
    let single = TH1::read_root(&qa, "glaub_sng_histo").unwrap();
    let n_bins = single.xaxis.nbins as usize;
    let integral: f64 = single.contents[2..=n_bins].iter().sum();
    let max_bin = single.contents[2..=n_bins]
        .iter()
        .cloned()
        .fold(0., f64::max)
        / integral;
    let pile_up_start = result
        .pile_up_border
        .map_or(n_bins + 1, |border| border as usize + 1);
    for (i, c) in classes.iter().enumerate() {
        /* the pile-up tail counts for the most central class */
        let hi = if i == 0 { n_bins } else { c.bins.1 };
        let fraction: f64 = single.contents[c.bins.0..=hi].iter().sum::<f64>() / integral;
        assert!(
            (fraction - 0.1).abs() <= max_bin + 1e-9,
            "class {}: fraction {fraction}",
            c.label()
        );
        /* contiguous classes, most central at the highest multiplicity */
        if i > 0 {
            assert_eq!(
                c.max_border,
                classes[i - 1].min_border,
                "class {}",
                c.label()
            );
            assert!(c.npart.mean < classes[i - 1].npart.mean);
            assert!(c.ncoll.mean < classes[i - 1].ncoll.mean);
            assert!(c.b.mean > classes[i - 1].b.mean);
        }
        assert!(c.b.min <= c.b.max && c.npart.min <= c.npart.max);
    }
    assert!(classes[0].bins.1 < pile_up_start);
    assert_eq!(classes[9].min_border, 1.);

    /* FINAL.root and the tables */
    let fin = FileReader::open(config.final_path()).unwrap();
    let tree = TreeReader::open(&fin, "Result").unwrap();
    assert_eq!(tree.num_entries(), 10);
    let npart = tree.read_branch(&fin, "NpartAverage").unwrap();
    assert_eq!(npart.as_f64().unwrap()[0], classes[0].npart.mean);
    for name in [
        "B_average_VS_Centrality",
        "Npart_average_VS_Centrality",
        "Ncoll_average_VS_Centrality",
        "Npart_VS_CentralityClass 0.0%-10.0%",
        "B_VS_CentralityClass 0%-100%",
        "CentralityClass_Fit 90.0%-100.0%",
        "CentralityClass 0.0%-10.0%",
        "Centrality_vs_Multiplicity",
    ] {
        TH1::read_root(&fin, name).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
    let csv = std::fs::read_to_string(config.table_path(TableFormat::Csv)).unwrap();
    assert_eq!(csv.lines().count(), 12);
    let cpp = std::fs::read_to_string(config.table_path(TableFormat::Cpp)).unwrap();
    assert!(cpp.contains("Int_t minMult [10]") && cpp.contains("GetCentMult"));
    assert!(config.table_path(TableFormat::Tex).exists());
    std::fs::remove_dir_all(&dir).unwrap();
}
