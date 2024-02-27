use std::hash::{Hash, Hasher};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Function {
    Builtin(Builtin),
    Custom(Custom),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Builtin {
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
}

#[derive(Debug, Copy, Clone)]
pub struct Custom {
    id: usize,
}

impl PartialEq for Custom {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Custom {}

impl Hash for Custom {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.id)
    }
}
