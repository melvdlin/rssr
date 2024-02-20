use std::hash::{Hash, Hasher};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum UnaryOp {
    Builtin(UnaryBuiltin),
    Custom(UnaryCustom),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]

pub enum BinaryOp {
    Builtin(BinaryBuiltin),
    Custom(BinaryCustom),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum UnaryBuiltin {
    Neg,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum BinaryBuiltin {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
}

#[derive(Debug, Copy, Clone)]
pub struct UnaryCustom {
    id: usize,
}

#[derive(Debug, Copy, Clone)]
pub struct BinaryCustom {
    id: usize,
}

impl PartialEq for UnaryCustom {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for UnaryCustom {}

impl Hash for UnaryCustom {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.id)
    }
}

impl PartialEq for BinaryCustom {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for BinaryCustom {}

impl Hash for BinaryCustom {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.id)
    }
}
