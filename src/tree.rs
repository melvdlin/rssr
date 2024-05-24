pub mod eval;

use std::marker::PhantomData;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Node<T, F> {
    Constant(Constant<T>),
    Variable(Variable<T>),
    Function(Function<T, F>),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub struct Constant<T> {
    id: usize,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub struct Variable<T> {
    id: usize,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone, Eq, PartialEq)]

pub struct Function<T, F> {
    function: F,
    operands: Box<[Node<T, F>]>,
}

pub trait Tree {
    type T: Clone;
    fn size(&self) -> usize;
}

pub trait NAryFunction {
    fn arity(&self) -> usize;
}

impl<T: Clone, F: NAryFunction> Tree for Node<T, F> {
    type T = T;

    fn size(&self) -> usize {
        match self {
            | Node::Constant(constant) => constant.size(),
            | Node::Variable(variable) => variable.size(),
            | Node::Function(function) => function.size(),
        }
    }
}

impl<T> Constant<T> {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            _phantom: PhantomData,
        }
    }
}

impl<T: Clone> Tree for Constant<T> {
    type T = T;
    fn size(&self) -> usize {
        1
    }
}

impl<T> Variable<T> {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            _phantom: PhantomData,
        }
    }
}

impl<T: Clone> Tree for Variable<T> {
    type T = T;

    fn size(&self) -> usize {
        1
    }
}

impl<T: Clone, F: NAryFunction + Clone> Function<T, F> {
    pub fn new(function: F, operands: &[Node<T, F>]) -> Self {
        assert_eq!(function.arity(), operands.len());
        Self {
            function,
            operands: operands.into(),
        }
    }

    pub fn arity(&self) -> usize {
        self.function.arity()
    }
}

impl<T: Clone, F: NAryFunction> Tree for Function<T, F> {
    type T = T;

    fn size(&self) -> usize {
        self.function.arity() + 1
    }
}
