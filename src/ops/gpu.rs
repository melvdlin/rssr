use crate::tree::NAryFunction;
use std::hash::{Hash, Hasher};

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct Function {
    pub id: usize,
    pub arity: usize,
}

impl Function {
    pub fn new(id: usize, arity: usize) -> Self {
        Self { id, arity }
    }

    pub fn id(self) -> usize {
        self.id
    }
}

impl Hash for Function {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.id)
    }
}

impl NAryFunction for Function {
    fn arity(&self) -> usize {
        self.arity
    }
}
