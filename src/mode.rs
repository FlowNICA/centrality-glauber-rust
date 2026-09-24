use std::fmt;
use std::str::FromStr;

use crate::error::Error;

/// Functional form of the number of ancestors (independent particle sources)
/// as a function of `Npart` and `Ncoll`.
///
/// | mode        | Na                         |
/// |-------------|----------------------------|
/// | `Default`   | f*Npart + (1-f)*Ncoll      |
/// | `PSD`       | f - Npart                  |
/// | `Npart`     | Npart^f                    |
/// | `Ncoll`     | Ncoll^f                    |
/// | `NpartFast` | Npart^f / 10^f             |
/// | `NcollFast` | Ncoll^f / 100^f            |
/// | `STAR`      | (1-f)*Npart/2 + f*Ncoll    |
/// | `HADES`     | (1 - f*Npart^2)*Npart      |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Default,
    Psd,
    Npart,
    Ncoll,
    NpartFast,
    NcollFast,
    Star,
    Hades,
}

impl Mode {
    pub const ALL: [Mode; 8] = [
        Mode::Default,
        Mode::Psd,
        Mode::Npart,
        Mode::Ncoll,
        Mode::NpartFast,
        Mode::NcollFast,
        Mode::Star,
        Mode::Hades,
    ];

    /// Number of ancestors for an event with the given `Npart` and `Ncoll`.
    pub fn n_ancestors(self, f: f64, npart: f64, ncoll: f64) -> f64 {
        match self {
            Mode::Default => f * npart + (1. - f) * ncoll,
            Mode::Psd => f - npart,
            Mode::Npart => npart.powf(f),
            Mode::Ncoll => ncoll.powf(f),
            Mode::NpartFast => npart.powf(f) / 10f64.powf(f),
            Mode::NcollFast => ncoll.powf(f) / 100f64.powf(f),
            Mode::Star => (1. - f) * npart / 2. + f * ncoll,
            Mode::Hades => (1. - f * npart * npart) * npart,
        }
    }

    /// Upper estimate of the number of ancestors, used to bound the golden
    /// section search for `mu`.
    pub fn n_ancestors_max(self, f: f64, npart_max: f64, ncoll_max: f64) -> f64 {
        match self {
            Mode::Psd => f,
            _ => self.n_ancestors(f, npart_max, ncoll_max),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Default => "Default",
            Mode::Psd => "PSD",
            Mode::Npart => "Npart",
            Mode::Ncoll => "Ncoll",
            Mode::NpartFast => "NpartFast",
            Mode::NcollFast => "NcollFast",
            Mode::Star => "STAR",
            Mode::Hades => "HADES",
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Mode {
    type Err = Error;

    /// Parses the mode names of the original framework, case-insensitively.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Mode::ALL
            .into_iter()
            .find(|m| m.name().eq_ignore_ascii_case(s))
            .ok_or_else(|| {
                let names: Vec<_> = Mode::ALL.iter().map(|m| m.name()).collect();
                Error::Config(format!(
                    "unknown mode '{s}', expected one of: {}",
                    names.join(", ")
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_names() {
        for mode in Mode::ALL {
            assert_eq!(mode.name().parse::<Mode>().unwrap(), mode);
            assert_eq!(mode.name().to_lowercase().parse::<Mode>().unwrap(), mode);
        }
        assert!("nope".parse::<Mode>().is_err());
    }

    #[test]
    fn ancestors() {
        assert_eq!(Mode::Default.n_ancestors(0.5, 10., 20.), 15.);
        assert_eq!(Mode::Star.n_ancestors(0.5, 10., 20.), 12.5);
        assert_eq!(Mode::Psd.n_ancestors_max(3., 100., 200.), 3.);
        assert_eq!(Mode::Npart.n_ancestors(2., 3., 0.), 9.);
    }
}
