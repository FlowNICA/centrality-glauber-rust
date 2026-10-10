use std::path::PathBuf;

use centrality_glauber_rust::Distribution as MultDistribution;
use centrality_glauber_rust::{FitConfig, FitMethod, FitResult, Mode};
use oxiroot::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Gamma, Poisson};

const TRUE_F: f32 = 0.5;
const TRUE_MU: f64 = 0.8;
const TRUE_K: f64 = 1.5;
/// Mean energy (GeV) and k of one ancestor in [`write_energy_data`].
const TRUE_E_MU: f64 = 0.05;
const TRUE_E_K: f64 = 4.;

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
fn npart_ncoll_histograms_have_unit_bins() {
    let dir = temp_dir("range-histos");
    let (glauber, data) = write_inputs(&dir, 50_000, false);
    let config = FitConfig::builder()
        .glauber(&glauber, "nt_toy")
        .data(&data, "hMult")
        .fit_range(20, 300)
        .build()
        .unwrap();
    let fitter = centrality_glauber_rust::Fitter::new(config).unwrap();
    /* integer Npart and Ncoll: unit bins from 0, the maximum not in the overflow */
    for h in [fitter.npart_histo(), fitter.ncoll_histo()] {
        let nbins = h.xaxis.nbins as usize;
        assert_eq!(
            (h.xaxis.xmin, h.xaxis.xmax),
            (0., nbins as f64),
            "{}",
            h.name
        );
        assert_eq!(
            h.contents[nbins + 1],
            0.,
            "{}: maximum in the overflow",
            h.name
        );
        assert_eq!(h.contents[0], 0., "{}", h.name);
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

    /*
     * Ranges from a sharp cut in each observable: the most central class
     * starts at b = 0 (and ends at the largest Npart), and neighboring
     * classes share their edges
     */
    assert_eq!(classes[0].b.min, 0.);
    assert!(classes[0].npart.max > classes[0].npart.mean + classes[0].npart.rms);
    for w in classes.windows(2) {
        assert_eq!(w[1].b.min, w[0].b.max);
        assert_eq!(w[1].npart.max, w[0].npart.min);
        assert_eq!(w[1].ncoll.max, w[0].ncoll.min);
    }

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

/// Copy of `h` with `w` neighboring bins merged.
fn rebin(h: &TH1, w: usize) -> TH1 {
    let nbins = h.xaxis.nbins as usize / w;
    let width = (h.xaxis.xmax - h.xaxis.xmin) / h.xaxis.nbins as f64 * w as f64;
    let mut r = Hist::reg(
        nbins as i32,
        h.xaxis.xmin,
        h.xaxis.xmin + width * nbins as f64,
    )
    .name(h.name.clone())
    .float();
    r.sumw2 = vec![0.; nbins + 2];
    for bin in 1..=nbins * w {
        let merged = (bin - 1) / w + 1;
        r.contents[merged] += h.contents[bin];
        r.sumw2[merged] += h.contents[bin];
    }
    r
}

#[test]
fn wide_data_bins() {
    use centrality_glauber_rust::CentralityConfig;
    use centrality_glauber_rust::centrality;

    let dir = temp_dir("wide-bins");
    let (glauber, data) = write_inputs(&dir, 50_000, false);
    let unit = TH1::read_root(&FileReader::open(&data).unwrap(), "hMult").unwrap();
    let data = dir.join("data2.root");
    rebin(&unit, 2)
        .write_root(&data, Compression::None)
        .unwrap();
    let out_dir = dir.join("out");

    /* multiplicities 20..300 as with unit bins */
    let config = FitConfig::builder()
        .glauber(&glauber, "nt_toy")
        .data(&data, "hMult")
        .out_dir(&out_dir)
        .mode(Mode::Default)
        .f_range(TRUE_F, TRUE_F, 0.)
        .k_range(TRUE_K as f32, TRUE_K as f32, 0.)
        .p_range(0., 0., 0.)
        .fit_range(10, 150)
        .n_iter(15)
        .seed(42)
        .build()
        .unwrap();
    let result = centrality_glauber_rust::run(config).unwrap();
    assert!(
        (result.best.mu as f64 - TRUE_MU).abs() < 0.05,
        "mu = {}",
        result.best.mu
    );
    assert!(result.chi2 < 3., "chi2/ndf = {}", result.chi2);

    let qa_file = out_dir.join(centrality_glauber_rust::output::QA_FILE_NAME);
    let qa = FileReader::open(&qa_file).unwrap();
    let single = TH1::read_root(&qa, "glaub_sng_histo").unwrap();
    assert_eq!(
        (
            single.xaxis.xmin,
            single.xaxis.xmax / single.xaxis.nbins as f64
        ),
        (0., 2.)
    );

    /* bin 1, multiplicities [0, 2), holds hits: it is in the most peripheral class */
    let config = CentralityConfig::builder()
        .qa_file(&qa_file)
        .data(None, "hMult")
        .out_dir(&out_dir)
        .n_classes(5)
        .table_formats(vec![])
        .build()
        .unwrap();
    let classes = centrality::run(&config).unwrap().classes;
    assert_eq!(classes.len(), 5);
    assert_eq!((classes[4].bins.0, classes[4].min_border), (1, 0.));
    let n_bins = single.xaxis.nbins as usize;
    let integral: f64 = single.contents[1..=n_bins].iter().sum();
    let max_bin = single.contents[1..=n_bins]
        .iter()
        .cloned()
        .fold(0., f64::max)
        / integral;
    for (i, c) in classes.iter().enumerate() {
        let hi = if i == 0 { n_bins } else { c.bins.1 };
        let fraction: f64 = single.contents[c.bins.0..=hi].iter().sum::<f64>() / integral;
        assert!(
            (fraction - 0.2).abs() <= max_bin + 1e-9,
            "class {}: fraction {fraction}",
            c.label()
        );
        assert_eq!(c.min_border % 2., 0.);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Data histogram `hEnergy` of a continuous observable ("energy") from the
/// Glauber tree written by [`write_inputs`]: each ancestor deposits a
/// Gamma distributed energy with mean `TRUE_E_MU` GeV and variance
/// `TRUE_E_MU^2 / TRUE_E_K`, in units of `unit` GeV (1000 for MeV).
fn write_energy_data(dir: &std::path::Path, glauber: &std::path::Path, unit: f64) -> PathBuf {
    let file = FileReader::open(glauber).unwrap();
    let tree = TreeReader::open(&file, "nt_toy").unwrap();
    let npart = tree.read_branch(&file, "Npart").unwrap();
    let ncoll = tree.read_branch(&file, "Ncoll").unwrap();
    let (npart, ncoll) = (npart.as_f32().unwrap(), ncoll.as_f32().unwrap());

    let gamma = Gamma::new(TRUE_E_K, TRUE_E_MU / TRUE_E_K).unwrap();
    let mut data = Hist::reg(1000, 0., 40. * unit).name("hEnergy").float();
    let mut rng = SmallRng::seed_from_u64(3);
    for i in 0..50_000 {
        let j = (i * 7919) % npart.len();
        let na = Mode::Default.n_ancestors(TRUE_F as f64, npart[j] as f64, ncoll[j] as f64) as i64;
        let energy: f64 = (0..na).map(|_| gamma.sample(&mut rng)).sum();
        data.fill(energy * unit);
    }
    let path = dir.join(format!("energy_{unit}.root"));
    data.write_root(&path, Compression::None).unwrap();
    path
}

#[test]
fn energy_fit_does_not_depend_on_units() {
    use centrality_glauber_rust::centrality;
    use centrality_glauber_rust::{CentralityConfig, CentralityResult};

    let dir = temp_dir("energy");
    let (glauber, _) = write_inputs(&dir, 1, false);
    let fit = |unit: f64| -> (FitResult, CentralityResult) {
        let data = write_energy_data(&dir, &glauber, unit);
        let out_dir = dir.join(format!("out_{unit}"));
        let config = FitConfig::builder()
            .glauber(&glauber, "nt_toy")
            .data(&data, "hEnergy")
            .out_dir(&out_dir)
            .mode(Mode::Default)
            .distribution(MultDistribution::GammaScaleFree)
            .f_range(TRUE_F, TRUE_F, 0.)
            .k_range(2., 6., 1.)
            .p_range(0., 0., 0.)
            /* 0.4..28 GeV: bins 11..=700 */
            .fit_value_range(0.4 * unit, 28. * unit)
            .n_iter(15)
            .seed(42)
            .build()
            .unwrap();
        let result = centrality_glauber_rust::run(config).unwrap();
        let config = CentralityConfig::builder()
            .qa_file(out_dir.join(centrality_glauber_rust::output::QA_FILE_NAME))
            .data(None, "hEnergy")
            .out_dir(&out_dir)
            .n_classes(10)
            .min_value(0.)
            .table_formats(vec![])
            .build()
            .unwrap();
        (result, centrality::run(&config).unwrap())
    };
    let fit_file = |unit: f64| {
        dir.join(format!("out_{unit}"))
            .join("fit_0.50_2.00_6.00_0.00_11.root")
    };
    let (gev, gev_classes) = fit(1.);
    let (mev, mev_classes) = fit(1000.);
    println!("GeV: {gev:?}\nMeV: {mev:?}");
    print!("{}{}", gev_classes.plain_table(), mev_classes.plain_table());

    /* the toy constrains mu to a few percent; the chi2 is flat in k within errors */
    assert!(
        (gev.best.mu as f64 / TRUE_E_MU - 1.).abs() < 0.08,
        "mu = {}",
        gev.best.mu
    );
    assert!(gev.chi2 < 3., "chi2/ndf = {}", gev.chi2);
    /* the same fit in MeV: mu scales with the unit, k does not */
    assert!(
        (mev.best.mu as f64 / (1000. * gev.best.mu as f64) - 1.).abs() < 1e-3,
        "mu = {} MeV vs {} GeV",
        mev.best.mu,
        gev.best.mu
    );
    assert_eq!(mev.best.k, gev.best.k);
    /* the fit range in values gives the same bins in both units */
    assert!(fit_file(1.).exists() && fit_file(1000.).exists());
    assert!((mev.chi2 / gev.chi2 - 1.).abs() < 1e-2);

    /* the same classes: borders scale with the unit, not rounded to integers */
    assert_eq!(gev_classes.classes.len(), 10);
    assert_eq!(mev_classes.classes.len(), 10);
    assert!(!gev_classes.integer_borders());
    assert_eq!(gev_classes.classes[9].min_border, 0.);
    for (g, m) in gev_classes.classes.iter().zip(&mev_classes.classes) {
        assert!((m.min_border - 1000. * g.min_border).abs() < 1e-6 * m.max_border);
        assert!((m.npart.mean - g.npart.mean).abs() < 0.01 * g.npart.mean);
    }
    /* most central class: large Npart, narrow compared with all events */
    let central = &gev_classes.classes[0];
    assert!(central.npart.mean > 100. && central.npart.rms < 0.3 * central.npart.mean);

    let fin = FileReader::open(dir.join("out_1").join("FINAL.root")).unwrap();
    let tree = TreeReader::open(&fin, "Result").unwrap();
    let borders = tree.read_branch(&fin, "MinBorder").unwrap();
    let borders = borders.as_f64().unwrap();
    assert_eq!(borders[0], gev_classes.classes[0].min_border);
    std::fs::remove_dir_all(&dir).unwrap();
}
