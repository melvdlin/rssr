#[macro_export]
macro_rules! proot {
    ($($paths:literal $(,)?)*) => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/", $($paths,)*)
    };
}
