use crate::tree::Node;

pub mod singlethreaded;

pub trait Evaluator<T, U, B> {
    type R;
    type E;
    fn evaluate(
        &mut self,
        tree: &Node<T, U, B>,
        variables: &[T],
    ) -> Result<Self::R, Self::E>;
}
