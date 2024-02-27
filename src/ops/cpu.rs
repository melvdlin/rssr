use crate::tree::NAryFunction;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Function<T> {
    arity: usize,
    operation: fn(&[T]) -> T,
}

impl<T> Function<T> {
    pub fn new(arity: usize, operation: fn(&[T]) -> T) -> Self {
        Self { arity, operation }
    }
    pub fn apply(&self, arg: &[T]) -> T {
        assert_eq!(self.arity, arg.len());
        (self.operation)(arg)
    }
}

impl<T> NAryFunction for Function<T> {
    fn arity(&self) -> usize {
        self.arity
    }
}
