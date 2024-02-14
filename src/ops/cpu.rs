#[derive(Debug, Copy, Clone, Eq, PartialEq)]

pub struct UnaryOp<T> {
    operation: fn(T) -> T,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]

pub struct BinaryOp<T> {
    operation: fn(T, T) -> T,
}

impl<T> UnaryOp<T> {
    pub fn new(operation: fn(T) -> T) -> Self {
        Self { operation }
    }
    pub fn apply(&self, arg: T) -> T {
        (self.operation)(arg)
    }
}

impl<T> BinaryOp<T> {
    pub fn new(operation: fn(T, T) -> T) -> Self {
        Self { operation }
    }
    pub fn apply(&self, arg1: T, arg2: T) -> T {
        (self.operation)(arg1, arg2)
    }
}
