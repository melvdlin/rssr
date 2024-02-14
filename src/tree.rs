pub mod eval;

use std::marker::PhantomData;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Node<T, U, B> {
    Constant(Constant<T>),
    Variable(Variable<T>),
    UnaryOp(UnaryOp<T, U, B>),
    BinaryOp(BinaryOp<T, U, B>),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub struct Constant<T> {
    value: T,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub struct Variable<T> {
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone, Eq, PartialEq)]

pub struct UnaryOp<T, U, B> {
    operator: U,
    operand: Box<Node<T, U, B>>,
}

#[derive(Debug, Clone, Eq, PartialEq)]

pub struct BinaryOp<T, U, B> {
    operator: B,
    lhs: Box<Node<T, U, B>>,
    rhs: Box<Node<T, U, B>>,
}

trait Tree {
    type T: Clone;
    fn size(&self) -> usize;
    fn variable_count(&self) -> usize;
}

impl<T: Clone, U, B> Tree for Node<T, U, B> {
    type T = T;

    fn size(&self) -> usize {
        match self {
            | Node::Constant(constant) => constant.size(),
            | Node::Variable(variable) => variable.size(),
            | Node::UnaryOp(unary_op) => unary_op.size(),
            | Node::BinaryOp(binary_op) => binary_op.size(),
        }
    }

    fn variable_count(&self) -> usize {
        match self {
            | Node::Constant(constant) => constant.variable_count(),
            | Node::Variable(variable) => variable.variable_count(),
            | Node::UnaryOp(unary_op) => unary_op.variable_count(),
            | Node::BinaryOp(binary_op) => binary_op.variable_count(),
        }
    }
}

impl<T> Constant<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Clone> Tree for Constant<T> {
    type T = T;
    fn size(&self) -> usize {
        1
    }

    fn variable_count(&self) -> usize {
        0
    }
}

impl<T> Variable<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Clone> Tree for Variable<T> {
    type T = T;

    fn size(&self) -> usize {
        1
    }

    fn variable_count(&self) -> usize {
        1
    }
}

impl<T, U, B> UnaryOp<T, U, B> {
    pub fn new(operator: U, operand: Node<T, U, B>) -> Self {
        Self {
            operator,
            operand: Box::new(operand),
        }
    }
}

impl<T: Clone, U, B> Tree for UnaryOp<T, U, B> {
    type T = T;

    fn size(&self) -> usize {
        self.operand.size() + 1
    }

    fn variable_count(&self) -> usize {
        self.operand.variable_count()
    }
}

impl<T, U, B> BinaryOp<T, U, B> {
    pub fn new(operator: B, lhs: Node<T, U, B>, rhs: Node<T, U, B>) -> Self {
        Self {
            operator,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

impl<T: Clone, U, B> Tree for BinaryOp<T, U, B> {
    type T = T;

    fn size(&self) -> usize {
        self.lhs.size() + 1 + self.rhs.size()
    }

    fn variable_count(&self) -> usize {
        self.lhs.variable_count() + self.rhs.variable_count()
    }
}
