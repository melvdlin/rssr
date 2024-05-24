use crate::tree::Node;

pub mod gpu;
pub mod singlethreaded;

pub trait Evaluator<T, F> {
    type R;
    type E;
    fn evaluate(
        &mut self,
        tree: &Node<T, F>,
        variables: &[T],
        constants: &[T],
    ) -> Result<Self::R, Self::E>;
}
