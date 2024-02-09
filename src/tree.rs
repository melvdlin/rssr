use std::marker::PhantomData;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Node<T> {
    Constant(Constant<T>),
    Variable(Variable<T>),
    UnaryOp(UnaryOp<T>),
    BinaryOp(BinaryOp<T>),
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

pub struct UnaryOp<T> {
    operator: super::ops::UnaryOp<T>,
    operand: Box<Node<T>>,
}

#[derive(Debug, Clone, Eq, PartialEq)]

pub struct BinaryOp<T> {
    operator: super::ops::BinaryOp<T>,
    lhs: Box<Node<T>>,
    rhs: Box<Node<T>>,
}

#[enum_delegate::register]
trait Tree {
    type T: Clone;
    fn size(&self) -> usize;
    fn variable_count(&self) -> usize;
    fn evaluate(&self, variables: &[Self::T]) -> Self::T;
}

impl<T: Clone> Tree for Node<T> {
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

    fn evaluate(&self, variables: &[Self::T]) -> Self::T {
        match self {
            | Node::Constant(constant) => constant.evaluate(variables),
            | Node::Variable(variable) => variable.evaluate(variables),
            | Node::UnaryOp(unary_op) => unary_op.evaluate(variables),
            | Node::BinaryOp(binary_op) => binary_op.evaluate(variables),
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

    fn evaluate(&self, variables: &[Self::T]) -> Self::T {
        debug_assert!(variables.is_empty());
        self.value.clone()
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

    fn evaluate(&self, variables: &[Self::T]) -> Self::T {
        debug_assert!(variables.len() == 1);
        variables[0].clone()
    }
}

impl<T> UnaryOp<T> {
    pub fn new(operator: super::ops::UnaryOp<T>, operand: Node<T>) -> Self {
        Self {
            operator,
            operand: Box::new(operand),
        }
    }
}

impl<T: Clone> Tree for UnaryOp<T> {
    type T = T;

    fn size(&self) -> usize {
        self.operand.size() + 1
    }

    fn variable_count(&self) -> usize {
        self.operand.variable_count()
    }

    fn evaluate(&self, variables: &[Self::T]) -> Self::T {
        self.operator.apply(self.operand.evaluate(variables))
    }
}

impl<T> BinaryOp<T> {
    pub fn new(operator: super::ops::BinaryOp<T>, lhs: Node<T>, rhs: Node<T>) -> Self {
        Self {
            operator,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

impl<T: Clone> Tree for BinaryOp<T> {
    type T = T;

    fn size(&self) -> usize {
        self.lhs.size() + 1 + self.rhs.size()
    }

    fn variable_count(&self) -> usize {
        self.lhs.variable_count() + self.rhs.variable_count()
    }

    fn evaluate(&self, variables: &[Self::T]) -> Self::T {
        let (lvars, rvars) = variables.split_at(self.lhs.variable_count());
        self.operator
            .apply(self.lhs.evaluate(lvars), self.rhs.evaluate(rvars))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn simple_eval() {
        let tree = Node::BinaryOp(BinaryOp::new(
            super::super::ops::BinaryOp::new(std::ops::Add::add),
            Node::Constant(Constant::new(1f64)),
            Node::Constant(Constant::new(2f64)),
        ));
        assert_abs_diff_eq!(3.0f64, tree.evaluate(&[],), epsilon = f64::EPSILON);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn invalid_variable_count() {
        let tree = Node::BinaryOp(BinaryOp::new(
            super::super::ops::BinaryOp::new(std::ops::Add::add),
            Node::Constant(Constant::new(1f64)),
            Node::Constant(Constant::new(2f64)),
        ));
        tree.evaluate(&[1.0]);
    }
}
