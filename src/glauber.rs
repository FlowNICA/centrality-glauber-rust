use std::path::Path;

use oxiroot::prelude::*;

use crate::error::{Error, Result};

/// A per-event MC-Glauber quantity correlated with the model multiplicity in
/// the `<name>_VS_Multiplicity` output histograms.
#[derive(Debug, Clone)]
pub struct Observable {
    /// Branch name.
    pub name: &'static str,
    /// Axis label.
    pub label: &'static str,
    /// Binning of the observable axis: (bins, min, max).
    pub binning: Binning,
    pub values: Vec<f32>,
}

/// Binning of an axis: (bins, min, max).
type Binning = (i32, f64, f64);

/// Branches correlated with multiplicity: (name, label, binning, required).
const OBSERVABLES: [(&str, &str, Binning, bool); 13] = [
    ("B", "B, fm", (200, 0., 20.), true),
    ("Npart", "N_{part}", (10000, 0., 10000.), true),
    ("Ncoll", "N_{coll}", (10000, 0., 10000.), true),
    ("Ecc1", "#epsilon1", ECC_BINNING, false),
    ("Psi1", "#psi1", PSI_BINNING, false),
    ("Ecc2", "#epsilon2", ECC_BINNING, false),
    ("Psi2", "#psi2", PSI_BINNING, false),
    ("Ecc3", "#epsilon3", ECC_BINNING, false),
    ("Psi3", "#psi3", PSI_BINNING, false),
    ("Ecc4", "#epsilon4", ECC_BINNING, false),
    ("Psi4", "#psi4", PSI_BINNING, false),
    ("Ecc5", "#epsilon5", ECC_BINNING, false),
    ("Psi5", "#psi5", PSI_BINNING, false),
];
const ECC_BINNING: Binning = (100, 0., 1.);
/// 0.01 wide bins up to 2 * 3.14, as in the original framework.
#[allow(clippy::approx_constant)]
const PSI_BINNING: Binning = (628, 0., 2. * 3.14);

/// MC-Glauber events loaded into memory.
#[derive(Debug, Clone)]
pub struct GlauberEvents {
    pub npart: Vec<f32>,
    pub ncoll: Vec<f32>,
    /// Maximum `Npart` over the whole tree (not only the loaded events).
    pub npart_max: f32,
    /// Maximum `Ncoll` over the whole tree (not only the loaded events).
    pub ncoll_max: f32,
    /// Observables for the `*_VS_Multiplicity` histograms (includes `Npart` and `Ncoll`).
    pub observables: Vec<Observable>,
}

impl GlauberEvents {
    /// Loads the first `n_events` events (all if `None`) of the Glauber tree.
    /// `B`, `Npart` and `Ncoll` are required; missing `Ecc*`/`Psi*` branches
    /// are skipped with a warning.
    pub fn load(path: &Path, tree_name: &str, n_events: Option<usize>) -> Result<Self> {
        let file = FileReader::open(path)?;
        let tree = TreeReader::open(&file, tree_name)?;
        let n_total = tree.num_entries() as usize;
        let n = n_events.map_or(n_total, |n| n.min(n_total));
        if n == 0 {
            return Err(Error::Input(format!(
                "Glauber tree '{tree_name}' in {} has no events",
                path.display()
            )));
        }

        /* the maxima are taken over the whole tree, as TTree::GetMaximum does */
        let mut npart = read_f32(&tree, &file, "Npart", n_total)?;
        let mut ncoll = read_f32(&tree, &file, "Ncoll", n_total)?;
        let npart_max = npart.iter().copied().fold(f32::MIN, f32::max);
        let ncoll_max = ncoll.iter().copied().fold(f32::MIN, f32::max);
        npart.truncate(n);
        ncoll.truncate(n);

        let mut observables = Vec::with_capacity(OBSERVABLES.len());
        for (name, label, binning, required) in OBSERVABLES {
            let values = match name {
                "Npart" => npart.clone(),
                "Ncoll" => ncoll.clone(),
                _ => match read_f32(&tree, &file, name, n) {
                    Ok(values) => values,
                    Err(e) if !required => {
                        eprintln!("Warning: skipping Glauber branch '{name}': {e}");
                        continue;
                    }
                    Err(e) => return Err(e),
                },
            };
            observables.push(Observable {
                name,
                label,
                binning,
                values,
            });
        }

        Ok(Self {
            npart,
            ncoll,
            npart_max,
            ncoll_max,
            observables,
        })
    }

    pub fn len(&self) -> usize {
        self.npart.len()
    }

    pub fn is_empty(&self) -> bool {
        self.npart.is_empty()
    }
}

/// Reads the first `n` entries of a numeric scalar branch as `f32`.
fn read_f32(tree: &TreeReader, file: &FileReader, name: &str, n: usize) -> Result<Vec<f32>> {
    let values = tree.read_branch_range(file, name, 0, n as u64)?;
    let out = match values {
        BranchValues::F32(v) => v,
        BranchValues::F64(v) => v.into_iter().map(|x| x as f32).collect(),
        BranchValues::I32(v) => v.into_iter().map(|x| x as f32).collect(),
        BranchValues::U32(v) => v.into_iter().map(|x| x as f32).collect(),
        BranchValues::I16(v) => v.into_iter().map(f32::from).collect(),
        BranchValues::U16(v) => v.into_iter().map(f32::from).collect(),
        BranchValues::I64(v) => v.into_iter().map(|x| x as f32).collect(),
        BranchValues::U64(v) => v.into_iter().map(|x| x as f32).collect(),
        _ => {
            return Err(Error::Input(format!(
                "Glauber branch '{name}' is not a numeric scalar branch"
            )));
        }
    };
    if out.len() != n {
        return Err(Error::Input(format!(
            "Glauber branch '{name}' has {} entries, expected {n}",
            out.len()
        )));
    }
    Ok(out)
}
