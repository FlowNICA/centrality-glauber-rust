use thiserror::Error;

/// Errors of the Glauber fitter.
#[derive(Debug, Error)]
pub enum Error {
    /// Reading or writing a ROOT file failed.
    #[error("ROOT I/O error: {0}")]
    Root(#[from] oxiroot::Error),
    /// A filesystem operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The configuration is inconsistent.
    #[error("invalid configuration: {0}")]
    Config(String),
    /// The input data (Glauber tree or data histogram) cannot be used.
    #[error("invalid input: {0}")]
    Input(String),
    /// The worker thread pool could not be created.
    #[error("thread pool error: {0}")]
    ThreadPool(#[from] rayon::ThreadPoolBuildError),
}

pub type Result<T> = std::result::Result<T, Error>;
