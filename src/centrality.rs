//! Centrality classes from the fit results: port of `HistoCut.C`,
//! `CentralityClasses.C` and `printFinal.C` of the CentralityFramework.
//!
//! The classes are defined with the fitted model of single (non pile-up)
//! events, `glaub_sng_histo`: counting from the highest multiplicity down,
//! each class holds its percent range (`classes`, or `1 / n_classes` each)
//! of the single-event integral. For each
//! class the Glauber observables `B`, `Npart` and `Ncoll` are averaged over
//! the model events in its multiplicity range (`<name>_VS_Multiplicity`). The
//! range of each observable in a class is the one of a sharp cut in that
//! observable at the class percents, e.g. `b` from 0 fm in the most central
//! class.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use oxiroot::prelude::*;

use crate::config::{CentralityConfig, TableFormat};
use crate::error::{Error, Result};

/// Glauber observables averaged per class: (name, axis label).
const OBSERVABLES: [(&str, &str); 3] =
    [("B", "B, fm"), ("Npart", "N_{part}"), ("Ncoll", "N_{coll}")];

/// Multiplicity bins of the centrality classes.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassBins {
    /// Inclusive range of histogram bins of each class, from the most
    /// central; `None` if the class holds no bin (a single bin holds more
    /// than a class).
    pub ranges: Vec<Option<(usize, usize)>>,
    /// First bin of the pile-up dominated tail, which belongs to no class.
    pub pile_up_start: Option<usize>,
}

/// Splits the multiplicity bins into the classes between the percent `edges`
/// (increasing within [0, 100]) of the single-event model.
///
/// `single` and `pile_up` are the bin contents (with under- and overflow) of
/// the single and pile-up event models. The bins before `first_bin` are not
/// used (see [`first_class_bin`]), and the bins from the first one where the
/// pile-up events exceed the single events are excluded from the classes;
/// their single events still count for the most central class. Counting from
/// the highest multiplicity down, a bin belongs to the class in which its
/// cumulative fraction (including the bin) falls; bins outside the edges
/// (e.g. above 80% with edges up to 80) belong to no class.
pub fn class_bins(
    single: &[f64],
    pile_up: &[f64],
    first_bin: usize,
    edges: &[f64],
) -> Result<ClassBins> {
    let first_bin = first_bin.max(1);
    let n_classes = edges.len().saturating_sub(1);
    if single.len() < first_bin + 2 || n_classes == 0 {
        return Err(Error::Input("empty single-event model histogram".into()));
    }
    let n_bins = single.len() - 2;
    let pile_up_start =
        (first_bin..=n_bins).find(|&b| single[b] < pile_up.get(b).copied().unwrap_or(0.));
    let integral: f64 = single[first_bin..=n_bins].iter().sum();
    if !(integral > 0.) {
        return Err(Error::Input("single-event model histogram is empty".into()));
    }

    let mut ranges: Vec<Option<(usize, usize)>> = vec![None; n_classes];
    let mut cumulative = 0.;
    for bin in (first_bin..=n_bins).rev() {
        cumulative += single[bin];
        if pile_up_start.is_some_and(|start| bin >= start) {
            continue;
        }
        let Some(class) = class_of(edges, 100. * cumulative / integral) else {
            continue;
        };
        let range = ranges[class].get_or_insert((bin, bin));
        range.0 = bin;
    }
    Ok(ClassBins {
        ranges,
        pile_up_start,
    })
}

/// First multiplicity bin used for the classes: the first one that is not
/// entirely below `min_value`. With `min_value` 1, events without hits are
/// not classified: as in `HistoCut.C`, this is bin 2 for unit bins starting at
/// 0 or -0.5, while wider bins starting at 0 also hold hits in bin 1, which is
/// used. With a continuous observable, `min_value` 0 uses all bins from 0.
pub fn first_class_bin(axis: &TAxis, min_value: f64) -> usize {
    let n = axis.nbins.max(0) as usize;
    let width = (axis.xmax - axis.xmin) / n.max(1) as f64;
    (1..=n)
        .find(|&bin| bin_edges(axis, bin).1 > min_value + 1e-9 * width)
        .unwrap_or(n + 1)
}

/// Shortest decimal form of a class border: an integer, or the value with
/// the precision of an `f32` (which hides rounding errors of the bin edges).
fn border(v: f64) -> String {
    format!("{}", v as f32)
}

/// Mean and RMS of an observable in a class, and its range for a sharp cut in
/// the observable at the class percents (NaN if there are no events).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservableStats {
    pub mean: f64,
    pub rms: f64,
    pub min: f64,
    pub max: f64,
}

/// One centrality class.
#[derive(Debug, Clone, PartialEq)]
pub struct CentralityClass {
    /// Class number, 1 for the most central (`Ncc`).
    pub number: usize,
    pub min_percent: f64,
    pub max_percent: f64,
    /// Multiplicity range `[min_border, max_border)`: the edges of the bins.
    pub min_border: f64,
    pub max_border: f64,
    /// Inclusive range of multiplicity histogram bins.
    pub bins: (usize, usize),
    pub b: ObservableStats,
    pub npart: ObservableStats,
    pub ncoll: ObservableStats,
}

impl CentralityClass {
    /// Name suffix of the per-class histograms, e.g. `0.0%-10.0%`.
    pub fn label(&self) -> String {
        class_label(self.min_percent, self.max_percent)
    }

    /// Statistics of observable `k` of [`OBSERVABLES`] (B, Npart, Ncoll).
    fn stats(&self, k: usize) -> ObservableStats {
        [self.b, self.npart, self.ncoll][k]
    }

    fn stats_mut(&mut self, k: usize) -> &mut ObservableStats {
        match k {
            0 => &mut self.b,
            1 => &mut self.npart,
            _ => &mut self.ncoll,
        }
    }
}

/// Result of the centrality determination.
#[derive(Debug, Clone, PartialEq)]
pub struct CentralityResult {
    /// The classes holding at least one multiplicity bin, from the most central.
    pub classes: Vec<CentralityClass>,
    /// Lower multiplicity edge of the pile-up dominated tail, if any.
    pub pile_up_border: Option<f64>,
    /// Efficiency: integral of the data over integral of the fit
    /// (`glaub_fit_histo`), over the multiplicity bins the class percents
    /// refer to (from `min_value` on). The fit is normalized to the data in
    /// the fit range and extrapolated below it, so this is the fraction of
    /// the events that were recorded. NaN if the fit is empty there.
    pub efficiency: f64,
    /// QA file the classes were determined from.
    pub source: PathBuf,
}

/// Determines the centrality classes, writes the ROOT file and the tables,
/// and returns the classes. Progress and warnings are printed.
pub fn run(config: &CentralityConfig) -> Result<CentralityResult> {
    let open = |path: &Path| {
        FileReader::open(path)
            .map_err(|e| Error::Input(format!("cannot open {}: {e}", path.display())))
    };
    let qa = open(&config.qa_file)?;
    let read_th1 = |file: &FileReader, name: &str| {
        TH1::read_root(file, name)
            .map_err(|e| Error::Input(format!("cannot read histogram {name}: {e}")))
    };
    let single = read_th1(&qa, "glaub_sng_histo")?;
    let pile_up = read_th1(&qa, "glaub_plp_histo")?;
    let fit = read_th1(&qa, "glaub_fit_histo")?;
    let data = if config.data_file == config.qa_file {
        read_th1(&qa, &config.data_hist)?
    } else {
        read_th1(&open(&config.data_file)?, &config.data_hist)?
    };
    let vs_multiplicity = OBSERVABLES
        .iter()
        .map(|(name, _)| {
            let hist = format!("{name}_VS_Multiplicity");
            TH2::read_root(&qa, &hist)
                .map_err(|e| Error::Input(format!("cannot read histogram {hist}: {e}")))
        })
        .collect::<Result<Vec<_>>>()?;

    check_axes(&single, &pile_up, &fit, &data, &vs_multiplicity)?;
    let axis = &single.xaxis;
    let first_bin = first_class_bin(axis, config.min_value);
    let (first_lo, first_hi) = bin_edges(axis, first_bin.clamp(1, axis.nbins.max(1) as usize));
    if config.min_value > 0. && first_lo <= 0. && first_hi > config.min_value {
        eprintln!(
            "Warning: multiplicity bin [{}, {}) holds values both below and above \
             min_value = {}; all its events are counted in the most peripheral class",
            border(first_lo),
            border(first_hi),
            config.min_value
        );
    }
    let n_bins = axis.nbins.max(0) as usize;
    let total: f64 = single.contents[1..=n_bins].iter().sum();
    let skipped: f64 = single.contents[1..first_bin.min(n_bins + 1)].iter().sum();
    if skipped > 0. && total > 0. {
        println!(
            "Not classified: multiplicities below {} ({:.2}% of the single events, min_value = {})",
            border(bin_edges(axis, first_bin - 1).1),
            100. * skipped / total,
            config.min_value
        );
    }
    let class_bins = class_bins(
        &single.contents,
        &pile_up.contents,
        first_bin,
        &config.class_edges,
    )?;
    let width = (axis.xmax - axis.xmin) / axis.nbins as f64;
    let edge = |bin: usize| axis.xmin + (bin - 1) as f64 * width;
    let pile_up_border = class_bins.pile_up_start.map(edge);
    if let Some(border) = pile_up_border {
        println!("Pile-up dominated multiplicities >= {border} are not classified");
    }

    /* per-class averages of the observables */
    let edges = &config.class_edges;
    let mut classes = Vec::new();
    for (i, range) in class_bins.ranges.iter().enumerate() {
        let (min_percent, max_percent) = class_percents(edges, i);
        let Some((lo, hi)) = *range else {
            eprintln!(
                "Warning: class {} is empty: a single multiplicity bin holds more than {}% \
                 of the events; use wider classes",
                class_label(min_percent, max_percent),
                percent(max_percent - min_percent)
            );
            continue;
        };
        let stats: Vec<ObservableStats> = vs_multiplicity
            .iter()
            .map(|h| {
                let (mean, rms) = mean_rms(&project_y(h, "", Some((lo, hi))));
                ObservableStats {
                    mean,
                    rms,
                    min: f64::NAN,
                    max: f64::NAN,
                }
            })
            .collect();
        classes.push(CentralityClass {
            number: i + 1,
            min_percent,
            max_percent,
            min_border: edge(lo),
            max_border: edge(hi + 1),
            bins: (lo, hi),
            b: stats[0],
            npart: stats[1],
            ncoll: stats[2],
        });
    }

    /*
     * Ranges of the observables for a sharp cut in each of them, over the
     * events the class percents refer to (multiplicity bins >= first_bin): centrality
     * c is the c-quantile of b, and the (1 - c)-quantile of Npart and Ncoll,
     * which decrease with centrality. So b starts at 0 in the most central class.
     */
    for (k, h2) in vs_multiplicity.iter().enumerate() {
        let all = project_y(h2, "", Some((first_bin, h2.nx() + 1)));
        let decreasing = k > 0;
        let at = |percent: f64| {
            let q = percent / 100.;
            quantile(&all, if decreasing { 1. - q } else { q })
        };
        for c in &mut classes {
            let (a, b) = (at(c.min_percent), at(c.max_percent));
            let s = c.stats_mut(k);
            (s.min, s.max) = (a.min(b), a.max(b));
        }
    }
    /* data / fit over the bins the percents refer to (the data and model bins match) */
    let integral = |h: &TH1| -> f64 {
        let n = h.xaxis.nbins.max(0) as usize;
        h.contents.get(first_bin..=n).map_or(0., |c| c.iter().sum())
    };
    let fit_integral = integral(&fit);
    let efficiency = if fit_integral > 0. {
        integral(&data) / fit_integral
    } else {
        f64::NAN
    };
    println!("Efficiency (data / fit): {:.2}%", 100. * efficiency);
    let result = CentralityResult {
        classes,
        pile_up_border,
        efficiency,
        source: config.qa_file.clone(),
    };
    std::fs::create_dir_all(&config.out_dir)?;
    write_final(
        &config.final_path(),
        &result,
        &FinalInputs {
            class_bins: &class_bins,
            edges,
            fit: &fit,
            data: &data,
            vs_multiplicity: &vs_multiplicity,
        },
    )?;
    println!(
        "Centrality classes written to {}",
        config.final_path().display()
    );
    for &format in &config.table_formats {
        let path = config.table_path(format);
        let text = match format {
            TableFormat::Tex => result.tex_table(),
            TableFormat::Csv => result.csv_table(),
            TableFormat::Cpp => result.cpp_table(),
        };
        std::fs::write(&path, text)?;
        println!("Table written to {}", path.display());
    }
    if let Some(path) = config.plot_path() {
        /* the model and parameters in the header, if the QA file has them */
        let summary = crate::plot::FitSummary::read(&qa)
            .map_err(|e| eprintln!("Warning: no fit parameters in the plot: {e}"))
            .ok();
        crate::plot::write_plot(
            &path,
            &data,
            &fit,
            &result,
            summary.as_ref(),
            config.plot_title.as_deref(),
        )?;
        println!("Plot written to {}", path.display());
    }
    Ok(result)
}

/// The model histograms must share the binning, and the data histogram must
/// have the same bin width and lower edge (the fit's model bin i is data bin i).
fn check_axes(single: &TH1, pile_up: &TH1, fit: &TH1, data: &TH1, vs: &[TH2]) -> Result<()> {
    let uniform = |a: &TAxis| a.xbins.is_empty() && a.nbins > 0;
    let same = |a: &TAxis, b: &TAxis| {
        a.nbins == b.nbins && (a.xmin - b.xmin).abs() < 1e-9 && (a.xmax - b.xmax).abs() < 1e-9
    };
    let model = &single.xaxis;
    if !uniform(model) || !uniform(&data.xaxis) {
        return Err(Error::Input(
            "model and data histograms must have uniform binning".into(),
        ));
    }
    if !same(model, &pile_up.xaxis)
        || !same(model, &fit.xaxis)
        || vs.iter().any(|h| !same(model, &h.xaxis))
    {
        return Err(Error::Input(
            "the model histograms of the QA file have different binnings".into(),
        ));
    }
    let width = |a: &TAxis| (a.xmax - a.xmin) / a.nbins as f64;
    if (model.xmin - data.xaxis.xmin).abs() > 1e-9 * width(model).max(1.)
        || (width(model) / width(&data.xaxis) - 1.).abs() > 1e-9
    {
        return Err(Error::Input(format!(
            "data histogram binning ({} bins in [{}, {}]) does not match the model \
             ({} bins in [{}, {}]): use the data histogram of the fit",
            data.xaxis.nbins, data.xaxis.xmin, data.xaxis.xmax, model.nbins, model.xmin, model.xmax
        )));
    }
    Ok(())
}

/// Class `i` of the percent `edges` with `edges[i] < percent <= edges[i + 1]`
/// (within rounding), or `None` outside the edges. 0% is in the first class
/// if it starts at 0, and 100% (up to rounding) in the last if it ends at 100.
fn class_of(edges: &[f64], percent: f64) -> Option<usize> {
    const EPS: f64 = 1e-9;
    let n = edges.len() - 1;
    match edges.partition_point(|&e| e < percent - EPS) {
        0 => (edges[0] <= 0.).then_some(0),
        i if i <= n => Some(i - 1),
        _ => (edges[n] >= 100.).then_some(n - 1),
    }
}

/// Nominal percent range of class `i`.
fn class_percents(edges: &[f64], i: usize) -> (f64, f64) {
    (edges[i], edges[i + 1])
}

/// `%.1f%%-%.1f%%` as in the names of the original histograms.
fn class_label(min_percent: f64, max_percent: f64) -> String {
    format!("{min_percent:.1}%-{max_percent:.1}%")
}

/// Projection on the y axis of the x bins `lo..=hi` (all bins, with under-
/// and overflow, if `None`), like `TH2::ProjectionY`.
fn project_y(h: &TH2, name: &str, bins: Option<(usize, usize)>) -> TH1 {
    let (nx, ny) = (h.nx(), h.ny());
    let stride = nx + 2;
    let (lo, hi) = bins.unwrap_or((0, nx + 1));
    let mut p = if h.yaxis.xbins.is_empty() {
        Hist::reg(ny as i32, h.yaxis.xmin, h.yaxis.xmax)
    } else {
        Hist::var(&h.yaxis.xbins)
    }
    .name(name)
    .double();
    p.xaxis.title = h.yaxis.title.clone();
    p.yaxis.title = "counts".into();
    let sum = |cells: &[f64], iy: usize| (lo..=hi).map(|ix| cells[ix + stride * iy]).sum();
    p.contents = (0..ny + 2).map(|iy| sum(&h.contents, iy)).collect();
    if !h.sumw2.is_empty() {
        p.sumw2 = (0..ny + 2).map(|iy| sum(&h.sumw2, iy)).collect();
    }
    reset_stats(&mut p);
    p
}

/// Makes ROOT recompute the statistics from the bin contents
/// (`TH1::GetStats` does so if `fTsumw == 0` and `fEntries > 0`).
fn reset_stats(h: &mut TH1) {
    h.entries = h.contents.iter().sum();
    h.tsumw = 0.;
    h.tsumw2 = 0.;
    h.tsumwx = 0.;
    h.tsumwx2 = 0.;
}

/// Lower and upper edge of bin `bin` (1-based) of an axis.
fn bin_edges(axis: &TAxis, bin: usize) -> (f64, f64) {
    if axis.xbins.is_empty() {
        let width = (axis.xmax - axis.xmin) / axis.nbins as f64;
        let lo = axis.xmin + (bin - 1) as f64 * width;
        (lo, lo + width)
    } else {
        (axis.xbins[bin - 1], axis.xbins[bin])
    }
}

/// Center of bin `bin` (1-based) of an axis.
fn bin_center(axis: &TAxis, bin: usize) -> f64 {
    let (lo, hi) = bin_edges(axis, bin);
    (lo + hi) / 2.
}

/// `q`-quantile (`0 <= q <= 1`) of a histogram without under- and overflow,
/// interpolated linearly within the bin: 0 gives the lower edge of the first
/// filled bin, 1 the upper edge of the last one. NaN if it is empty.
fn quantile(h: &TH1, q: f64) -> f64 {
    let n = h.xaxis.nbins as usize;
    let total: f64 = h.contents[1..=n].iter().sum();
    if !(total > 0.) {
        return f64::NAN;
    }
    let target = q.clamp(0., 1.) * total;
    let mut cumulative = 0.;
    let mut last_edge = f64::NAN;
    for bin in 1..=n {
        let c = h.contents[bin];
        if !(c > 0.) {
            continue;
        }
        let (lo, hi) = bin_edges(&h.xaxis, bin);
        if cumulative + c >= target {
            return lo + (target - cumulative) / c * (hi - lo);
        }
        cumulative += c;
        last_edge = hi;
    }
    /* q = 1 with rounding */
    last_edge
}

/// Mean and RMS of a histogram from its bin centers, without under- and
/// overflow, as `TH1::GetMean` and `TH1::GetRMS`; NaN if it is empty.
fn mean_rms(h: &TH1) -> (f64, f64) {
    let (mut s0, mut s1, mut s2) = (0., 0., 0.);
    for bin in 1..=h.xaxis.nbins as usize {
        let (x, w) = (bin_center(&h.xaxis, bin), h.contents[bin]);
        s0 += w;
        s1 += w * x;
        s2 += w * x * x;
    }
    if !(s0 > 0.) {
        return (f64::NAN, f64::NAN);
    }
    let mean = s1 / s0;
    (mean, (s2 / s0 - mean * mean).max(0.).sqrt())
}

/// Inputs of the output ROOT file besides the result.
struct FinalInputs<'a> {
    class_bins: &'a ClassBins,
    /// Class edges in percent.
    edges: &'a [f64],
    fit: &'a TH1,
    data: &'a TH1,
    vs_multiplicity: &'a [TH2],
}

/// Copy of `h` named `name` with only the bins `lo..=hi`.
fn restricted(h: &TH1, name: &str, (lo, hi): (usize, usize)) -> TH1 {
    let mut r = h.clone();
    r.name = name.into();
    r.title = String::new();
    r.xaxis.title = "tracks".into();
    r.yaxis.title = "counts".into();
    let outside = |bin: usize| bin < lo || bin > hi;
    for (bin, c) in r.contents.iter_mut().enumerate() {
        if outside(bin) {
            *c = 0.;
        }
    }
    for (bin, c) in r.sumw2.iter_mut().enumerate() {
        if outside(bin) {
            *c = 0.;
        }
    }
    reset_stats(&mut r);
    r
}

/// Writes the output ROOT file: the tree `Result`, the averages versus
/// centrality, the per-class distributions of the observables, and the model
/// and data multiplicity of each class.
fn write_final(path: &Path, result: &CentralityResult, inputs: &FinalInputs) -> Result<()> {
    let edges = inputs.edges;
    let n = edges.len() - 1;
    let classes = &result.classes;

    let column_f64 = |get: &dyn Fn(&CentralityClass) -> f64| classes.iter().map(get).collect();
    /* integer borders as Int_t, as read by printFinal.C; otherwise Double_t */
    let border_branch = |name: &str, get: &dyn Fn(&CentralityClass) -> f64| {
        if result.integer_borders() {
            Branch::i32(
                name,
                classes.iter().map(|c| get(c).round() as i32).collect(),
            )
        } else {
            Branch::f64(name, column_f64(get))
        }
    };
    let mut branches = vec![
        Branch::i32("Ncc", classes.iter().map(|c| c.number as i32).collect()),
        Branch::f32(
            "MinPercent",
            classes.iter().map(|c| c.min_percent as f32).collect(),
        ),
        Branch::f32(
            "MaxPercent",
            classes.iter().map(|c| c.max_percent as f32).collect(),
        ),
        border_branch("MinBorder", &|c| c.min_border),
        border_branch("MaxBorder", &|c| c.max_border),
    ];
    for (k, (name, _)) in OBSERVABLES.iter().enumerate() {
        branches.extend([
            Branch::f64(format!("{name}Average"), column_f64(&|c| c.stats(k).mean)),
            Branch::f64(format!("{name}Width"), column_f64(&|c| c.stats(k).rms)),
            Branch::f64(format!("{name}Min"), column_f64(&|c| c.stats(k).min)),
            Branch::f64(format!("{name}Max"), column_f64(&|c| c.stats(k).max)),
        ]);
    }
    let tree = Tree::new("Result", branches);

    /* <observable> versus centrality, with the RMS as error */
    let mut averages = Vec::new();
    for (k, (name, label)) in OBSERVABLES.iter().enumerate() {
        /* uniform bins for equal classes over 0-100%, as in the original */
        let equal = edges
            .iter()
            .enumerate()
            .all(|(i, &e)| (e - 100. * i as f64 / n as f64).abs() < 1e-9);
        let hist = if equal {
            Hist::reg(n as i32, 0., 100.)
        } else {
            Hist::var(edges)
        };
        let mut h = hist.name(format!("{name}_average_VS_Centrality")).double();
        h.xaxis.title = "Centrality, %".into();
        h.yaxis.title = (*label).into();
        h.sumw2 = vec![0.; n + 2];
        for c in classes {
            let s = c.stats(k);
            h.contents[c.number] = s.mean;
            h.sumw2[c.number] = s.rms * s.rms;
        }
        reset_stats(&mut h);
        averages.push(h);
    }

    /* distributions of the observables per class, and for all events */
    let mut projections = Vec::new();
    for ((name, _), h2) in OBSERVABLES.iter().zip(inputs.vs_multiplicity) {
        for c in classes {
            let hist_name = format!("{name}_VS_CentralityClass {}", c.label());
            projections.push(project_y(h2, &hist_name, Some(c.bins)));
        }
        let all = format!("{name}_VS_CentralityClass 0%-100%");
        projections.push(project_y(h2, &all, None));
    }

    /* model and data multiplicity per class, and centrality versus multiplicity */
    let mut per_class = Vec::new();
    for c in classes {
        per_class.push(restricted(
            inputs.fit,
            &format!("CentralityClass_Fit {}", c.label()),
            c.bins,
        ));
    }
    for c in classes {
        per_class.push(restricted(
            inputs.data,
            &format!("CentralityClass {}", c.label()),
            c.bins,
        ));
    }
    let mut vs_mult = inputs.fit.clone();
    vs_mult.name = "Centrality_vs_Multiplicity".into();
    vs_mult.title = String::new();
    vs_mult.xaxis.title = "tracks".into();
    vs_mult.yaxis.title = "Centrality, %".into();
    vs_mult.contents.iter_mut().for_each(|c| *c = 0.);
    vs_mult.sumw2.clear();
    for (i, range) in inputs.class_bins.ranges.iter().enumerate() {
        if let Some((lo, hi)) = *range {
            let (_, max_percent) = class_percents(edges, i);
            vs_mult.contents[lo..=hi]
                .iter_mut()
                .for_each(|c| *c = max_percent);
        }
    }
    reset_stats(&mut vs_mult);

    let mut writer = FileWriter::create(path);
    for h in averages.iter().chain(&projections).chain(&per_class) {
        writer = writer.add(h);
    }
    writer
        .add(&vs_mult)
        .put(tree)
        .write(Compression::default())?;
    Ok(())
}

/// Percent with the decimals only when needed, e.g. `10` or `12.5`.
pub(crate) fn percent(p: f64) -> String {
    if (p - p.round()).abs() < 1e-9 {
        format!("{p:.0}")
    } else {
        format!("{p:.1}")
    }
}

impl CentralityResult {
    /// Whether all class borders are integers, as for a multiplicity with
    /// unit or integer bin widths. Otherwise (e.g. an energy) the borders are
    /// written as floating point numbers.
    pub fn integer_borders(&self) -> bool {
        self.classes.iter().all(|c| {
            [c.min_border, c.max_border]
                .iter()
                .all(|&b| (b - b.round()).abs() <= 1e-6 * b.abs().max(1.))
        })
    }

    fn centrality(c: &CentralityClass) -> String {
        format!("{} - {}", percent(c.min_percent), percent(c.max_percent))
    }

    /// Plain text table, as printed by `printFinal.C` without an output file.
    pub fn plain_table(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "File: {}.", self.source.display());
        let _ = writeln!(
            s,
            "  Cent, %   | Mult_min | Mult_max | <b>, fm |   RMS   | bmin, fm | bmax, fm | \
             <Npart> |   RMS   | Npart_min | Npart_max | <Ncoll> |   RMS   | Ncoll_min | Ncoll_max |"
        );
        let _ = writeln!(
            s,
            "------------|----------|----------|---------|---------|----------|----------|\
             ---------|---------|-----------|-----------|---------|---------|-----------|-----------|"
        );
        for c in &self.classes {
            let _ = writeln!(
                s,
                "{:>11} | {:>8} | {:>8} | {:7.2} | {:7.2} | {:8.2} | {:8.2} | {:7.2} | {:7.2} | \
                 {:9.2} | {:9.2} | {:7.2} | {:7.2} | {:9.2} | {:9.2} |",
                Self::centrality(c),
                border(c.min_border),
                border(c.max_border),
                c.b.mean,
                c.b.rms,
                c.b.min,
                c.b.max,
                c.npart.mean,
                c.npart.rms,
                c.npart.min,
                c.npart.max,
                c.ncoll.mean,
                c.ncoll.rms,
                c.ncoll.min,
                c.ncoll.max
            );
        }
        s
    }

    /// LaTeX document with the table, with the columns of `printFinal.C`.
    pub fn tex_table(&self) -> String {
        let mut s = String::new();
        s.push_str(
            "\\documentclass[11pt]{article}\n\
             \\usepackage[utf8]{inputenc}\n\
             \\usepackage{geometry}\n\
             \\geometry{legalpaper, landscape, margin=2in}\n\n\
             \\begin{document}\n\n\
             Generated from a file:\n\
             \\begin{verbatim*}\n",
        );
        let _ = writeln!(s, "{}", self.source.display());
        s.push_str(
            "\\end{verbatim*}\n\
             \\begin{center}\n\
             \\begin{tabular}{ |c|c|c|c|c|c|c|c|c| }\n\
             \t\\hline\n\
             \tCentrality, \\% & $N_{ch}^{min}$ & $N_{ch}^{max}$ & $\\langle b \\rangle$, fm & RMS & \
             $\\langle N_{part} \\rangle$ & RMS & $\\langle N_{coll} \\rangle$ & RMS \\\\\n",
        );
        for c in &self.classes {
            let _ = writeln!(
                s,
                "\t\\hline\n\t{} & {} & {} & {:.2} & {:.2} & {:.2} & {:.2} & {:.2} & {:.2} \\\\",
                Self::centrality(c),
                border(c.min_border),
                border(c.max_border),
                c.b.mean,
                c.b.rms,
                c.npart.mean,
                c.npart.rms,
                c.ncoll.mean,
                c.ncoll.rms
            );
        }
        s.push_str("\t\\hline\n\\end{tabular}\n\\end{center}\n\\end{document}\n");
        s
    }

    /// CSV table with the columns of `printFinal.C`, after a `#` comment line
    /// naming the source file.
    pub fn csv_table(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# Generated from a file: {}", self.source.display());
        s.push_str(
            "Centrality class,Mult_min,Mult_max,Mean b,RMS,b_min,b_max,Mean Npart,RMS,\
             Npart_min,Npart_max,Mean Ncoll,RMS,Ncoll_min,Ncoll_max\n",
        );
        for c in &self.classes {
            let _ = writeln!(
                s,
                "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
                Self::centrality(c),
                border(c.min_border),
                border(c.max_border),
                c.b.mean,
                c.b.rms,
                c.b.min,
                c.b.max,
                c.npart.mean,
                c.npart.rms,
                c.npart.min,
                c.npart.max,
                c.ncoll.mean,
                c.ncoll.rms,
                c.ncoll.min,
                c.ncoll.max
            );
        }
        s
    }

    /// ROOT macro with arrays of the class borders and averages, and the
    /// function `GetCentMult(mult)` returning the centrality (middle of the
    /// class, in percent) of a multiplicity, or -1.
    pub fn cpp_table(&self) -> String {
        let n = self.classes.len();
        let mut s = String::new();
        let _ = writeln!(s, "// Generated from a file: {}", self.source.display());
        let mut array = |ty: &str, name: &str, values: Vec<String>| {
            let _ = writeln!(s, "{ty} {name} [{n}] = {{ {} }};", values.join(", "));
        };
        let float = |v: f64| {
            if v.is_finite() {
                format!("{}", v as f32)
            } else {
                "NAN".into()
            }
        };
        let column = |get: &dyn Fn(&CentralityClass) -> f64| {
            self.classes.iter().map(|c| float(get(c))).collect()
        };
        let border_type = if self.integer_borders() {
            "Int_t"
        } else {
            "Double_t"
        };
        let border_column = |get: &dyn Fn(&CentralityClass) -> f64| {
            self.classes.iter().map(|c| border(get(c))).collect()
        };
        array("Float_t", "minCentPercent", column(&|c| c.min_percent));
        array("Float_t", "maxCentPercent", column(&|c| c.max_percent));
        array(border_type, "minMult", border_column(&|c| c.min_border));
        array(border_type, "maxMult", border_column(&|c| c.max_border));
        for (k, (name, _)) in OBSERVABLES.iter().enumerate() {
            array(
                "Float_t",
                &format!("mean{name}"),
                column(&|c| c.stats(k).mean),
            );
            array(
                "Float_t",
                &format!("rms{name}"),
                column(&|c| c.stats(k).rms),
            );
            array(
                "Float_t",
                &format!("min{name}"),
                column(&|c| c.stats(k).min),
            );
            array(
                "Float_t",
                &format!("max{name}"),
                column(&|c| c.stats(k).max),
            );
        }
        let _ = write!(
            s,
            "\nFloat_t GetCentMult({border_type} mult)\n\
             // Returns the centrality (middle of the class, in percent) for a given multiplicity, or -1\n\
             {{\n\
             \tInt_t centBin = -1;\n\
             \tfor (Int_t i = 0; i < {n}; i++)\n\
             \t{{\n\
             \t\tif (mult >= minMult[i] && mult < maxMult[i])\n\
             \t\t\tcentBin = i;\n\
             \t}}\n\
             \tif (centBin == -1) return -1.;\n\
             \treturn (maxCentPercent[centBin] - minCentPercent[centBin]) / 2. + minCentPercent[centBin];\n\
             }}\n"
        );
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Edges of `n` classes of equal width.
    fn equal(n: usize) -> Vec<f64> {
        (0..=n).map(|i| 100. * i as f64 / n as f64).collect()
    }

    #[test]
    fn class_of_percent() {
        let edges = [0., 5., 10., 30., 100.];
        assert_eq!(class_of(&edges, 0.), Some(0));
        assert_eq!(class_of(&edges, 5.), Some(0));
        assert_eq!(class_of(&edges, 5.000001), Some(1));
        assert_eq!(class_of(&edges, 30.), Some(2));
        assert_eq!(class_of(&edges, 100. + 1e-12), Some(3));
        /* partial coverage: outside the edges no class */
        let edges = [10., 20., 80.];
        assert_eq!(class_of(&edges, 5.), None);
        assert_eq!(class_of(&edges, 10.), None);
        assert_eq!(class_of(&edges, 15.), Some(0));
        assert_eq!(class_of(&edges, 80.), Some(1));
        assert_eq!(class_of(&edges, 90.), None);
    }

    #[test]
    fn classes_of_unequal_width() {
        /* 100 bins of 1 event from bin 2: 0-5% is the top 5 bins, 5-10% the next 5, ... */
        let mut values = vec![1000.];
        values.extend([1.; 100]);
        let (single, pile_up) = (contents(&values), contents(&[0.; 101]));
        let bins = class_bins(&single, &pile_up, 2, &[0., 5., 10., 30., 100.]).unwrap();
        assert_eq!(
            bins.ranges,
            vec![
                Some((97, 101)),
                Some((92, 96)),
                Some((72, 91)),
                Some((2, 71))
            ]
        );
        /* only 10-30% and 30-80%: the top 10 and the bottom 20 bins are in no class */
        let bins = class_bins(&single, &pile_up, 2, &[10., 30., 80.]).unwrap();
        assert_eq!(bins.ranges, vec![Some((72, 91)), Some((22, 71))]);
    }

    /// Contents with under/overflow from the in-range values.
    fn contents(values: &[f64]) -> Vec<f64> {
        let mut c = vec![0.];
        c.extend_from_slice(values);
        c.push(0.);
        c
    }

    #[test]
    fn equal_classes_of_a_flat_distribution() {
        /* bin 1 (multiplicity 0) is ignored; 100 bins of 1 event */
        let mut values = vec![1000.];
        values.extend([1.; 100]);
        let bins = class_bins(&contents(&values), &contents(&[0.; 101]), 2, &equal(10)).unwrap();
        assert_eq!(bins.pile_up_start, None);
        /* class 0 holds the 10 highest bins 92..=101, the last class 2..=11 */
        for (i, range) in bins.ranges.iter().enumerate() {
            let hi = 101 - 10 * i;
            assert_eq!(*range, Some((hi - 9, hi)), "class {i}");
        }
    }

    #[test]
    fn pile_up_tail_is_excluded_but_counted() {
        let single = contents(&[5., 4., 3., 2., 1., 0.5, 0.5]);
        let pile_up = contents(&[0., 0., 0., 0.1, 0.1, 1., 0.2]);
        let bins = class_bins(&single, &pile_up, 2, &equal(2)).unwrap();
        /* from bin 6 on the pile-up exceeds the single events (also bin 7 is excluded) */
        assert_eq!(bins.pile_up_start, Some(6));
        /* integral of bins 2..=7 is 11: class 0 is <= 5.5 from the top: 0.5 + 0.5 + 1 + 2 */
        assert_eq!(bins.ranges, vec![Some((4, 5)), Some((2, 3))]);
    }

    #[test]
    fn a_large_bin_leaves_classes_empty() {
        let single = contents(&[0., 1., 8., 1.]);
        let bins = class_bins(&single, &contents(&[0.; 4]), 2, &equal(5)).unwrap();
        /* 10%: bin 4, bin 3 reaches 90% (class 4), classes 1..=3 are empty */
        assert_eq!(
            bins.ranges,
            vec![Some((4, 4)), None, None, None, Some((2, 3))]
        );
        assert!(class_bins(&contents(&[3., 0.]), &contents(&[0., 0.]), 2, &equal(2)).is_err());
    }

    #[test]
    fn first_class_bin_skips_only_bins_below_one_hit() {
        let first = |nbins, lo, hi| first_class_bin(&Hist::reg(nbins, lo, hi).double().xaxis, 1.);
        /* unit bins: bin 1 is multiplicity 0 */
        assert_eq!(first(100, 0., 100.), 2);
        assert_eq!(first(100, -0.5, 99.5), 2);
        /* wide bins starting at 0: bin 1 also holds hits */
        assert_eq!(first(20, 0., 100.), 1);
        assert_eq!(first(20, -0.5, 99.5), 1);
        /* narrow bins: all bins below 1 */
        assert_eq!(first(200, 0., 100.), 3);
        assert_eq!(first(400, 0., 100.), 5);
        /* histogram above multiplicity 1 */
        assert_eq!(first(10, 10., 20.), 1);
        /* continuous observable: all bins from 0, or from a threshold */
        let axis = Hist::reg(300, 0., 3.).double().xaxis;
        assert_eq!(first_class_bin(&axis, 0.), 1);
        assert_eq!(first_class_bin(&axis, 0.25), 26);
        /* the first bin of the classes is used */
        let bins = class_bins(
            &contents(&[1., 1., 1., 1.]),
            &contents(&[0.; 4]),
            1,
            &equal(4),
        )
        .unwrap();
        assert_eq!(
            bins.ranges,
            vec![Some((4, 4)), Some((3, 3)), Some((2, 2)), Some((1, 1))]
        );
    }

    #[test]
    fn quantiles() {
        /* one entry per unit bin over [0, 10) */
        let mut h = Hist::reg(10, 0., 10.).double();
        for x in 0..10 {
            h.fill(x as f64 + 0.5);
        }
        assert_eq!(quantile(&h, 0.), 0.);
        assert!((quantile(&h, 0.25) - 2.5).abs() < 1e-12);
        assert!((quantile(&h, 1.) - 10.).abs() < 1e-12);
        /* empty bins at the edges: from the first to the last filled bin */
        let mut h = Hist::reg(10, 0., 10.).double();
        h.fill(3.5);
        h.fill(6.5);
        assert_eq!(quantile(&h, 0.), 3.);
        assert!((quantile(&h, 0.5) - 4.).abs() < 1e-12);
        assert!((quantile(&h, 1.) - 7.).abs() < 1e-12);
        assert!(quantile(&Hist::reg(10, 0., 10.).double(), 0.5).is_nan());
    }

    fn class(number: usize, borders: (f64, f64)) -> CentralityClass {
        let stats = |m: f64| ObservableStats {
            mean: m,
            rms: 0.5,
            min: m - 1.,
            max: m + 1.,
        };
        CentralityClass {
            number,
            min_percent: 50. * (number - 1) as f64,
            max_percent: 50. * number as f64,
            min_border: borders.0,
            max_border: borders.1,
            bins: (1, 1),
            b: stats(3.),
            npart: stats(200.),
            ncoll: stats(500.),
        }
    }

    #[test]
    fn tables() {
        let result = CentralityResult {
            classes: vec![class(1, (40., 120.)), class(2, (1., 40.))],
            pile_up_border: None,
            efficiency: 1.,
            source: "qa.root".into(),
        };
        let csv = result.csv_table();
        let lines: Vec<_> = csv.lines().collect();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with('#'));
        assert_eq!(lines[1].split(',').count(), 15);
        assert_eq!(
            lines[2],
            "0 - 50,40,120,3.00,0.50,2.00,4.00,200.00,0.50,199.00,201.00,500.00,0.50,499.00,501.00"
        );
        let cpp = result.cpp_table();
        assert!(cpp.contains("Int_t minMult [2] = { 40, 1 };"), "{cpp}");
        assert!(
            cpp.contains("Float_t maxNpart [2] = { 201, 201 };"),
            "{cpp}"
        );
        assert!(cpp.contains("Float_t GetCentMult(Int_t mult)"));
        let tex = result.tex_table();
        assert!(
            tex.contains("\t50 - 100 & 1 & 40 & 3.00 & 0.50 & 200.00 & 0.50 & 500.00 & 0.50 \\\\"),
            "{tex}"
        );
        assert!(tex.trim_end().ends_with("\\end{document}"));
        assert_eq!(result.plain_table().lines().count(), 5);
        assert!(result.integer_borders());
    }

    #[test]
    fn non_integer_borders_are_not_rounded() {
        let result = CentralityResult {
            classes: vec![class(1, (2.49, 7.640000000000001)), class(2, (0., 2.49))],
            pile_up_border: None,
            efficiency: 1.,
            source: "qa.root".into(),
        };
        assert!(!result.integer_borders());
        assert!(
            result
                .csv_table()
                .lines()
                .nth(2)
                .unwrap()
                .starts_with("0 - 50,2.49,7.64,")
        );
        let cpp = result.cpp_table();
        assert!(cpp.contains("Double_t minMult [2] = { 2.49, 0 };"), "{cpp}");
        assert!(cpp.contains("Float_t GetCentMult(Double_t mult)"), "{cpp}");
        assert!(result.tex_table().contains("\t0 - 50 & 2.49 & 7.64 &"));
        assert!(result.plain_table().contains("|     2.49 |     7.64 |"));
    }
}
