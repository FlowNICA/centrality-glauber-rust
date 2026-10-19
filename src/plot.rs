//! PNG picture of the result of `define-centrality`, in the style of the
//! ROOT canvases of the CentralityFramework: the data and the fit on a
//! logarithmic scale with the class borders and the fit parameters, and the
//! ratio data / fit below.
//!
//! Drawn with `plotters` (pure Rust, no system libraries) and the embedded
//! Liberation Sans font (`data/fonts`, SIL Open Font License). Ported from
//! `centrality-inverse-bayes-rust`.

use std::path::Path;
use std::sync::Once;

use oxiroot::prelude::*;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use crate::centrality::{CentralityResult, percent};
use crate::error::{Error, Result};
use crate::mode::Mode;

const WIDTH: u32 = 1200;
const HEIGHT: u32 = 1000;
/// Frame of the upper pad (data and fit) and of the lower pad (ratio), in
/// pixels.
const LEFT: i32 = 140;
const RIGHT: i32 = 1160;
const TOP: i32 = 20;
const SPLIT: i32 = 690;
const BOTTOM: i32 = 890;
/// Lowest value of the logarithmic axis, so that bins with one count show.
const Y_MIN: f64 = 0.4;
/// Top of the first header line below the frame, clear of the ticks.
const HEADER_OFFSET: i32 = 28;
/// Text sizes in pixels.
const LABEL_SIZE: u32 = 34;
const HEADER_SIZE: u32 = 30;
const LEGEND_SIZE: u32 = 40;
const CLASS_SIZE: u32 = 30;

const FONT: &str = "sans-serif";
const DATA_COLOR: RGBColor = RGBColor(0, 0, 255);
const FIT_COLOR: RGBColor = RGBColor(255, 0, 0);

/// Best fit parameters and settings of the fit, from the `BestResult` tree
/// of the QA file. The settings are missing in QA files of older versions.
#[derive(Debug, Clone, PartialEq)]
pub struct FitSummary {
    pub f: f64,
    pub mu: f64,
    pub k: f64,
    pub p: f64,
    /// Fit statistic per degree of freedom.
    pub chi2: f64,
    /// [`Mode`] name, e.g. `STAR`.
    pub mode: Option<String>,
    /// [`crate::Distribution`] name, e.g. `Gamma`.
    pub distribution: Option<String>,
    /// [`crate::FitMethod`] name, e.g. `Chi2`.
    pub fit_method: Option<String>,
}

impl FitSummary {
    /// Reads the `BestResult` tree of the QA file.
    pub fn read(qa: &FileReader) -> Result<Self> {
        let tree = TreeReader::open(qa, "BestResult")?;
        let number = |name: &str| -> Result<f64> {
            let values = tree.read_branch(qa, name)?;
            values
                .as_f32()
                .and_then(|v| v.first())
                .map(|&v| v as f64)
                .ok_or_else(|| Error::Input(format!("BestResult: no value of {name}")))
        };
        let text = |name: &str| match tree.read_branch(qa, name) {
            Ok(oxiroot::tree::BranchValues::Str(v)) => v.into_iter().next(),
            _ => None,
        };
        Ok(FitSummary {
            f: number("f")?,
            mu: number("mu")?,
            k: number("k")?,
            p: number("p")?,
            chi2: number("chi2")?,
            mode: text("mode"),
            distribution: text("distribution"),
            fit_method: text("fit_method"),
        })
    }
}

/// Data and fit in the data bins.
struct Bins {
    /// `n + 1` bin edges.
    edges: Vec<f64>,
    counts: Vec<f64>,
    errors: Vec<f64>,
    model: Vec<f64>,
}

impl Bins {
    fn new(data: &TH1, fit: &TH1) -> Self {
        let axis = &data.xaxis;
        let n = axis.nbins.max(0) as usize;
        let width = (axis.xmax - axis.xmin) / n.max(1) as f64;
        Bins {
            edges: (0..=n).map(|i| axis.xmin + i as f64 * width).collect(),
            counts: (1..=n)
                .map(|b| data.contents.get(b).copied().unwrap_or(0.))
                .collect(),
            errors: (1..=n).map(|b| data.bin_error(b)).collect(),
            model: (1..=n)
                .map(|b| fit.contents.get(b).copied().unwrap_or(0.))
                .collect(),
        }
    }
    fn n_bins(&self) -> usize {
        self.counts.len()
    }
    fn center(&self, i: usize) -> f64 {
        0.5 * (self.edges[i] + self.edges[i + 1])
    }
}

static FONTS: Once = Once::new();

/// Registers the embedded fonts with `plotters` (once).
fn register_fonts() {
    FONTS.call_once(|| {
        let fonts: [(FontStyle, &'static [u8]); 2] = [
            (
                FontStyle::Normal,
                include_bytes!("../data/fonts/LiberationSans-Regular.ttf"),
            ),
            (
                FontStyle::Bold,
                include_bytes!("../data/fonts/LiberationSans-Bold.ttf"),
            ),
        ];
        for (style, bytes) in fonts {
            /* the embedded fonts are valid */
            let _ = plotters::style::register_font(FONT, style, bytes);
        }
    });
}

fn plot_error(e: impl std::fmt::Display) -> Error {
    Error::Plot(e.to_string())
}

/// Drawing in pixels on the whole picture.
struct Canvas<'a, 'b> {
    area: &'a DrawingArea<BitMapBackend<'b>, plotters::coord::Shift>,
}

/// How a text is placed and drawn.
#[derive(Clone, Copy)]
struct Font {
    size: u32,
    bold: bool,
    color: RGBColor,
    h: HPos,
    v: VPos,
    vertical: bool,
}

impl Font {
    fn new(size: u32) -> Self {
        Font {
            size,
            bold: false,
            color: BLACK,
            h: HPos::Left,
            v: VPos::Top,
            vertical: false,
        }
    }
    fn bold(self) -> Self {
        Font { bold: true, ..self }
    }
    fn at(self, h: HPos, v: VPos) -> Self {
        Font { h, v, ..self }
    }
    fn color(self, color: RGBColor) -> Self {
        Font { color, ..self }
    }
    /// Rotated by 90°, read from bottom to top.
    fn vertical(self) -> Self {
        Font {
            vertical: true,
            ..self
        }
    }
    fn style(&self) -> TextStyle<'static> {
        let style = if self.bold {
            FontStyle::Bold
        } else {
            FontStyle::Normal
        };
        let mut font = (FONT, self.size as f64, style).into_font();
        if self.vertical {
            font = font.transform(FontTransform::Rotate270);
        }
        TextStyle {
            font,
            color: self.color.to_backend_color(),
            pos: Pos::new(self.h, self.v),
        }
    }
}

impl Canvas<'_, '_> {
    fn text(&self, s: &str, (x, y): (i32, i32), font: Font) -> Result<()> {
        self.area
            .draw(&Text::new(s.to_string(), (x, y), font.style()))
            .map_err(plot_error)
    }

    /// Width and height of a text in pixels.
    fn text_size(&self, s: &str, font: Font) -> Result<(i32, i32)> {
        let (w, h) = self
            .area
            .estimate_text_size(
                s,
                &Font {
                    vertical: false,
                    ..font
                }
                .style(),
            )
            .map_err(plot_error)?;
        Ok((w as i32, h as i32))
    }

    fn line(&self, a: (i32, i32), b: (i32, i32), color: RGBColor, width: u32) -> Result<()> {
        self.area
            .draw(&PathElement::new(vec![a, b], color.stroke_width(width)))
            .map_err(plot_error)
    }

    /// Dotted horizontal line.
    fn dotted(&self, x0: i32, x1: i32, y: i32) -> Result<()> {
        let mut x = x0;
        while x < x1 {
            self.line((x, y), ((x + 4).min(x1), y), BLACK, 2)?;
            x += 9;
        }
        Ok(())
    }

    fn frame(&self, top: i32, bottom: i32) -> Result<()> {
        self.area
            .draw(&Rectangle::new(
                [(LEFT, top), (RIGHT, bottom)],
                BLACK.stroke_width(2),
            ))
            .map_err(plot_error)
    }

    /// Marker of half-size `r` pixels.
    fn marker(&self, (x, y): (i32, i32), r: i32, color: RGBColor, kind: Marker) -> Result<()> {
        match kind {
            Marker::Dot => self
                .area
                .draw(&Circle::new((x, y), r, color.filled()))
                .map_err(plot_error),
            Marker::Circle => self
                .area
                .draw(&Circle::new((x, y), r, color.stroke_width(2)))
                .map_err(plot_error),
            Marker::Square => self
                .area
                .draw(&Rectangle::new(
                    [(x - r, y - r), (x + r, y + r)],
                    color.stroke_width(2),
                ))
                .map_err(plot_error),
        }
    }
}

#[derive(Clone, Copy)]
enum Marker {
    Dot,
    Circle,
    Square,
}

/// Linear map of `[lo, hi]` onto the pixels `[p_lo, p_hi]`.
#[derive(Clone, Copy)]
struct Axis {
    lo: f64,
    hi: f64,
    p_lo: i32,
    p_hi: i32,
    log: bool,
}

impl Axis {
    fn value(&self, v: f64) -> f64 {
        if self.log { v.log10() } else { v }
    }
    fn pixel(&self, v: f64) -> i32 {
        let (lo, hi) = (self.value(self.lo), self.value(self.hi));
        let t = (self.value(v) - lo) / (hi - lo);
        (self.p_lo as f64 + t * (self.p_hi - self.p_lo) as f64).round() as i32
    }
    /// The value at pixel `p`.
    fn value_at(&self, p: i32) -> f64 {
        let (lo, hi) = (self.value(self.lo), self.value(self.hi));
        let t = (p - self.p_lo) as f64 / (self.p_hi - self.p_lo) as f64;
        let v = lo + t * (hi - lo);
        if self.log { 10f64.powf(v) } else { v }
    }
    fn contains(&self, v: f64) -> bool {
        v >= self.lo && v <= self.hi
    }
    fn clamp(&self, v: f64) -> f64 {
        v.clamp(self.lo, self.hi)
    }
}

/// Major and minor tick values of a linear axis, about `n` major ones.
fn linear_ticks(lo: f64, hi: f64, n: f64) -> (Vec<f64>, Vec<f64>) {
    let raw = (hi - lo) / n;
    let mag = 10f64.powf(raw.log10().floor());
    let (step, minor) = match raw / mag {
        r if r < 1.5 => (mag, mag / 5.),
        r if r < 3.5 => (2. * mag, mag / 2.),
        r if r < 7.5 => (5. * mag, mag),
        _ => (10. * mag, 2. * mag),
    };
    let values = |s: f64| -> Vec<f64> {
        let first = (lo / s).ceil() as i64;
        let last = (hi / s).floor() as i64;
        (first..=last).map(|k| k as f64 * s).collect()
    };
    (values(step), values(minor))
}

/// A number with 4 significant digits, without trailing zeros.
fn number(v: f64) -> String {
    if v == 0. || !v.is_finite() {
        return format!("{v}");
    }
    let digits = (3 - v.abs().log10().floor() as i32).clamp(0, 10) as usize;
    let s = format!("{v:.digits$}");
    let s = if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    };
    s.replace('-', "−")
}

/// Tick label of a linear axis.
fn tick_label(v: f64, step: f64) -> String {
    let digits = (-(step.log10().floor()) as i32).max(0) as usize;
    let s = format!("{v:.digits$}");
    if s.starts_with('-') && s.trim_start_matches(['-', '0', '.']).is_empty() {
        s[1..].to_string()
    } else {
        s
    }
}

/// Writes the picture of the data `data`, the fit `fit` (`glaub_fit_histo`)
/// and the classes `result` to `path`; `title` (e.g. experiment and colliding
/// system) starts the header, followed by the model and the fit parameters.
pub fn write_plot(
    path: &Path,
    data: &TH1,
    fit: &TH1,
    result: &CentralityResult,
    summary: Option<&FitSummary>,
    title: Option<&str>,
) -> Result<()> {
    if data.xaxis.nbins < 1 {
        return Err(Error::Input("empty data histogram".into()));
    }
    register_fonts();
    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&WHITE).map_err(plot_error)?;
    let canvas = Canvas { area: &root };
    draw(&canvas, data, &Bins::new(data, fit), result, summary, title)?;
    root.present().map_err(plot_error)?;
    Ok(())
}

/// The model as formulas: number of ancestors and per-ancestor distribution.
fn model_text(summary: &FitSummary) -> Vec<String> {
    let mut items = Vec::new();
    if let Some(name) = &summary.mode {
        items.push(match name.parse::<Mode>() {
            Ok(mode) => format!("N_a = {} ({name})", mode.formula()),
            Err(_) => format!("mode {name}"),
        });
    }
    if let Some(name) = &summary.distribution {
        items.push(
            match name.as_str() {
                "Gamma" => "Gamma: σ² = μ(1 + μ/k)",
                "Nbd" => "NBD: σ² = μ(1 + μ/k)",
                "GammaScaleFree" => "Gamma: σ² = μ²/k",
                other => other,
            }
            .to_string(),
        );
    }
    items
}

/// The fitted parameters and the fit quality.
fn parameter_text(summary: &FitSummary) -> Vec<String> {
    let statistic = match summary.fit_method.as_deref() {
        Some("Likelihood") => "−2lnλ/ndf",
        _ => "χ²/ndf",
    };
    vec![
        format!("f = {}", number(summary.f)),
        format!("μ = {}", number(summary.mu)),
        format!("k = {}", number(summary.k)),
        format!("p = {}%", number(100. * summary.p)),
        format!("{statistic} = {:.2}", summary.chi2),
    ]
}

/// Joins the items with ", " into lines no wider than `width` pixels.
fn wrap(canvas: &Canvas, items: &[String], width: i32, font: Font) -> Result<Vec<String>> {
    let mut lines: Vec<String> = Vec::new();
    for item in items {
        if let Some(last) = lines.last_mut() {
            let joined = format!("{last}, {item}");
            if canvas.text_size(&joined, font)?.0 <= width {
                *last = joined;
                continue;
            }
            last.push(',');
        }
        lines.push(item.clone());
    }
    Ok(lines)
}

fn draw(
    canvas: &Canvas,
    hist: &TH1,
    data: &Bins,
    result: &CentralityResult,
    summary: Option<&FitSummary>,
    title: Option<&str>,
) -> Result<()> {
    let model = &data.model;
    let n_bins = data.n_bins();
    let center = |i: usize| data.center(i);

    /* x range: from the first edge to beyond the last bin with data or fit */
    let last = (0..n_bins)
        .rev()
        .find(|&i| data.counts[i] > 0. || model[i] >= 1.)
        .unwrap_or(n_bins - 1);
    let x_lo = data.edges[0];
    let x_hi = data.edges[last + 1] + 0.05 * (data.edges[last + 1] - x_lo);

    /* header lines, then the y range with room for them above the data */
    let header_font = Font::new(HEADER_SIZE);
    let mut items = Vec::new();
    if let Some(t) = title.filter(|t| !t.is_empty()) {
        items.push(t.to_string());
    }
    if let Some(summary) = summary {
        items.extend(model_text(summary));
    }
    let text_width = RIGHT - LEFT - 40;
    let mut header = wrap(canvas, &items, text_width, header_font)?;
    if let Some(summary) = summary {
        header.extend(wrap(
            canvas,
            &parameter_text(summary),
            text_width,
            header_font,
        )?);
    }
    let line_height = (HEADER_SIZE as f64 * 1.3) as i32;
    let header_bottom = TOP + HEADER_OFFSET + line_height * header.len() as i32;
    let y_max = (0..n_bins)
        .map(|i| (data.counts[i] + data.errors[i]).max(model[i]))
        .fold(1., f64::max);
    /* the data maximum sits 30 pixels below the header */
    let free = (SPLIT - header_bottom - 30).max(100) as f64 / (SPLIT - TOP) as f64;
    let decades = (y_max / Y_MIN).log10() / free;
    let y_hi = Y_MIN * 10f64.powf(decades);

    let x_axis = |p_lo, p_hi| Axis {
        lo: x_lo,
        hi: x_hi,
        p_lo,
        p_hi,
        log: false,
    };
    let x = x_axis(LEFT, RIGHT);
    let y = Axis {
        lo: Y_MIN,
        hi: y_hi,
        p_lo: SPLIT,
        p_hi: TOP,
        log: true,
    };

    /* the ratio data / fit and its range */
    let ratios: Vec<Option<(f64, f64)>> = (0..n_bins)
        .map(|i| {
            (data.counts[i] > 0. && model[i] > 0.)
                .then(|| (data.counts[i] / model[i], data.errors[i] / model[i]))
        })
        .collect();
    let lowest = ratios
        .iter()
        .zip(&data.counts)
        .filter_map(|(r, &c)| r.filter(|_| c >= 10.).map(|(v, _)| v))
        .fold(f64::INFINITY, f64::min);
    let r_lo = if lowest < 0.5 {
        ((lowest - 0.05) * 10.).floor().max(0.) / 10.
    } else {
        0.45
    };
    let r = Axis {
        lo: r_lo,
        hi: 1.55,
        p_lo: BOTTOM,
        p_hi: SPLIT,
        log: false,
    };

    /* class borders and labels */
    let classes = &result.classes;
    let class_font = Font::new(CLASS_SIZE)
        .vertical()
        .at(HPos::Center, VPos::Center);
    let label_y = y.pixel(Y_MIN * 10f64.powf(0.3 * decades));
    let bin_of = |v: f64| {
        data.edges
            .partition_point(|&e| e <= v)
            .saturating_sub(1)
            .min(n_bins - 1)
    };
    for (k, c) in classes.iter().enumerate() {
        /*
         * The most peripheral class extends to the lowest multiplicity and the
         * most central one is open above, if the classes cover 0-100%. The
         * borders are drawn elsewhere: the lower one of every class, and the
         * upper one where it is not the lower one of the class above.
         */
        let open_below = k + 1 == classes.len() && c.max_percent >= 100.;
        let open_above = k == 0 && c.min_percent <= 0.;
        let shared = k > 0 && classes[k - 1].min_border == c.max_border;
        let mut borders = Vec::new();
        if !open_below {
            borders.push(c.min_border);
        }
        if !open_above && !shared {
            borders.push(c.max_border);
        }
        for border in borders {
            if x.contains(border) {
                let px = x.pixel(border);
                let top = y.clamp(model[bin_of(border)].max(data.counts[bin_of(border)]));
                canvas.line((px, y.pixel(Y_MIN)), (px, y.pixel(top)), BLACK, 2)?;
            }
        }
        let lo = if open_below {
            x_lo
        } else {
            c.min_border.max(x_lo)
        };
        let hi = if open_above {
            x_hi
        } else {
            c.max_border.min(x_hi)
        };
        if !(hi > lo) {
            continue;
        }
        let label = format!("{}-{}%", percent(c.min_percent), percent(c.max_percent));
        let (w, h) = canvas.text_size(&label, class_font)?;
        let py = label_y.max(header_bottom + w / 2 + 10);
        /* centered where the fit is above the label, e.g. not on the falling
         * tail of the most central class */
        let top = y.value_at(py - w / 2 - 5);
        let hi = (0..n_bins)
            .filter(|&i| center(i) >= lo && center(i) <= hi && model[i] >= top)
            .map(|i| data.edges[i + 1].min(hi))
            .next_back()
            .unwrap_or(hi);
        let (p_lo, p_hi) = (x.pixel(lo), x.pixel(hi));
        if p_hi - p_lo >= h + 4 {
            canvas.text(&label, ((p_lo + p_hi) / 2, py), class_font)?;
        }
    }

    /* the fit (bins with at least one count) and the data, with markers
     * smaller for narrow bins so that both stay visible */
    let bin_pixels =
        (RIGHT - LEFT) as f64 * (data.edges[last + 1] - x_lo) / (x_hi - x_lo) / (last + 1) as f64;
    let marker_size = (0.4 * bin_pixels).clamp(2.5, 6.).round() as i32;
    for (i, &m) in model.iter().enumerate() {
        if m >= 1. && y.contains(m) && x.contains(center(i)) {
            let p = (x.pixel(center(i)), y.pixel(m));
            canvas.marker(p, marker_size, FIT_COLOR, Marker::Square)?;
        }
    }
    for i in 0..n_bins {
        let (d, e) = (data.counts[i], data.errors[i]);
        if !(d > 0.) || !y.contains(d) || !x.contains(center(i)) {
            continue;
        }
        let px = x.pixel(center(i));
        let (lo, hi) = (y.clamp(d - e), y.clamp(d + e));
        canvas.line((px, y.pixel(lo)), (px, y.pixel(hi)), DATA_COLOR, 2)?;
        canvas.marker((px, y.pixel(d)), marker_size, DATA_COLOR, Marker::Dot)?;
    }

    /* the ratio */
    let one = r.pixel(1.);
    canvas.line((LEFT, one), (RIGHT, one), BLACK, 2)?;
    canvas.dotted(LEFT, RIGHT, r.pixel(0.9))?;
    canvas.dotted(LEFT, RIGHT, r.pixel(1.1))?;
    for (i, ratio) in ratios.iter().enumerate() {
        let Some((v, e)) = *ratio else { continue };
        if !x.contains(center(i)) {
            continue;
        }
        let px = x.pixel(center(i));
        let (lo, hi) = (r.clamp(v - e), r.clamp(v + e));
        if hi > lo {
            canvas.line((px, r.pixel(lo)), (px, r.pixel(hi)), DATA_COLOR, 2)?;
        }
        if r.contains(v) {
            canvas.marker((px, r.pixel(v)), marker_size, DATA_COLOR, Marker::Circle)?;
        }
    }

    /* frames and axes */
    canvas.frame(TOP, SPLIT)?;
    canvas.frame(SPLIT, BOTTOM)?;
    let label_font = Font::new(LABEL_SIZE).bold();
    let (major, minor) = linear_ticks(x_lo, x_hi, 8.);
    for (values, len) in [(&major, 20), (&minor, 10)] {
        for &v in values.iter() {
            let px = x.pixel(v);
            for (base, dir) in [(SPLIT, -1), (TOP, 1), (BOTTOM, -1), (SPLIT, 1)] {
                canvas.line((px, base), (px, base + dir * len), BLACK, 2)?;
            }
        }
    }
    let step = major.get(1).zip(major.first()).map_or(1., |(b, a)| b - a);
    for &v in &major {
        canvas.text(
            &tick_label(v, step),
            (x.pixel(v), BOTTOM + 8),
            label_font.at(HPos::Center, VPos::Top),
        )?;
    }
    let x_title = if hist.xaxis.title.is_empty() {
        hist.name.as_str()
    } else {
        hist.xaxis.title.as_str()
    };
    canvas.text(
        x_title,
        (RIGHT, BOTTOM + 52),
        Font::new(LABEL_SIZE - 4).at(HPos::Right, VPos::Top),
    )?;

    /* logarithmic y axis: decades with labels, minor ticks between */
    let exponent_font = Font::new(LABEL_SIZE * 3 / 5).bold();
    let (k_lo, k_hi) = (Y_MIN.log10().floor() as i32, y_hi.log10().floor() as i32);
    for k in k_lo..=k_hi {
        for m in 1..10 {
            let v = m as f64 * 10f64.powi(k);
            if !y.contains(v) {
                continue;
            }
            let py = y.pixel(v);
            let len = if m == 1 { 20 } else { 10 };
            canvas.line((LEFT, py), (LEFT + len, py), BLACK, 2)?;
            canvas.line((RIGHT, py), (RIGHT - len, py), BLACK, 2)?;
            if m != 1 {
                continue;
            }
            let at = label_font.at(HPos::Right, VPos::Center);
            match k {
                0 => canvas.text("1", (LEFT - 12, py), at)?,
                1 => canvas.text("10", (LEFT - 12, py), at)?,
                _ => {
                    let e = k.to_string();
                    let (ew, _) = canvas.text_size(&e, exponent_font)?;
                    canvas.text("10", (LEFT - 14 - ew, py), at)?;
                    canvas.text(
                        &e,
                        (LEFT - 12 - ew, py - LABEL_SIZE as i32 / 2),
                        exponent_font.at(HPos::Left, VPos::Center),
                    )?;
                }
            }
        }
    }
    canvas.text(
        "dN",
        (24, TOP + 10),
        Font::new(LABEL_SIZE + 6)
            .vertical()
            .at(HPos::Right, VPos::Top),
    )?;

    /* linear axis of the ratio */
    let (r_major, r_minor) = linear_ticks(r.lo, r.hi, 5.);
    for (values, len) in [(&r_major, 20), (&r_minor, 10)] {
        for &v in values.iter() {
            let py = r.pixel(v);
            canvas.line((LEFT, py), (LEFT + len, py), BLACK, 2)?;
            canvas.line((RIGHT, py), (RIGHT - len, py), BLACK, 2)?;
        }
    }
    let r_step = r_major
        .get(1)
        .zip(r_major.first())
        .map_or(0.2, |(b, a)| b - a);
    for &v in &r_major {
        canvas.text(
            &tick_label(v, r_step),
            (LEFT - 12, r.pixel(v)),
            label_font.at(HPos::Right, VPos::Center),
        )?;
    }
    canvas.text(
        "data/fit",
        (24, (SPLIT + BOTTOM) / 2),
        Font::new(LABEL_SIZE - 2)
            .vertical()
            .at(HPos::Center, VPos::Center),
    )?;

    /* header, legend and efficiency */
    for (i, line) in header.iter().enumerate() {
        canvas.text(
            line,
            (
                (LEFT + RIGHT) / 2,
                TOP + HEADER_OFFSET + line_height * i as i32,
            ),
            header_font.at(HPos::Center, VPos::Top),
        )?;
    }
    let legend_font = Font::new(LEGEND_SIZE).at(HPos::Left, VPos::Center);
    let legend_y = header_bottom + 50;
    let (fit_w, _) = canvas.text_size("fit", legend_font)?;
    let fit_x = RIGHT - 30 - fit_w;
    let (data_w, _) = canvas.text_size("data", legend_font)?;
    let data_x = fit_x - 70 - data_w;
    canvas.marker((data_x - 25, legend_y), 7, DATA_COLOR, Marker::Dot)?;
    canvas.text("data", (data_x, legend_y), legend_font)?;
    canvas.marker((fit_x - 25, legend_y), 7, FIT_COLOR, Marker::Square)?;
    canvas.text("fit", (fit_x, legend_y), legend_font)?;
    if result.efficiency.is_finite() {
        canvas.text(
            &format!("Eff = {:.2} %", 100. * result.efficiency),
            (RIGHT - 30, legend_y + 70),
            Font::new(HEADER_SIZE)
                .bold()
                .color(FIT_COLOR)
                .at(HPos::Right, VPos::Center),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_and_ticks() {
        assert_eq!(number(1.0982392), "1.098");
        assert_eq!(number(219.87408), "219.9");
        assert_eq!(number(-0.919787), "−0.9198");
        assert_eq!(number(2000.), "2000");
        let (major, minor) = linear_ticks(0., 147., 8.);
        assert_eq!(major, [0., 20., 40., 60., 80., 100., 120., 140.]);
        assert_eq!(minor.len(), 30);
        assert_eq!(tick_label(1.2000000000000002, 0.2), "1.2");
        assert_eq!(tick_label(-0., 0.2), "0.0");
    }
}
