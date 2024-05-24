use crate::ops::cpu;
use crate::tree::eval::Evaluator;
use crate::tree::Node;

use std::convert::Infallible;

pub struct SingleThreadedEvaluator;

impl<T: Clone> Evaluator<T, cpu::Function<T>> for SingleThreadedEvaluator {
    type R = T;
    type E = Infallible;

    fn evaluate(
        &mut self,
        tree: &Node<T, cpu::Function<T>>,
        variables: &[T],
        constants: &[T],
    ) -> Result<Self::R, Self::E> {
        match tree {
            | Node::Constant(crate::tree::Constant { id, .. }) => {
                debug_assert!(variables.is_empty());
                Ok(constants[*id].clone())
            }
            | Node::Variable(crate::tree::Variable { id, .. }) => {
                Ok(variables[*id].clone())
            }
            | Node::Function(crate::tree::Function { function, operands }) => {
                Ok(function.apply(
                    operands
                        .iter()
                        .map(|node| self.evaluate(node, variables, constants))
                        .collect::<Result<Vec<_>, _>>()?
                        .as_slice(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{Constant, Function, Variable};
    use approx::assert_abs_diff_eq;

    fn simple_tree() -> Node<f64, cpu::Function<f64>> {
        Node::Function(Function::new(
            cpu::Function::new(2, |operands| operands.iter().sum()),
            [
                Node::Constant(Constant::new(0)),
                Node::Constant(Constant::new(1)),
            ]
            .as_slice(),
        ))
    }

    fn simple_tree_with_variables() -> Node<f64, cpu::Function<f64>> {
        Node::Function(Function::new(
            cpu::Function::new(2, |operands| operands.iter().sum()),
            [
                Node::Variable(Variable::new(0)),
                Node::Variable(Variable::new(1)),
            ]
            .as_slice(),
        ))
    }

    #[test]
    fn simple_eval() {
        let tree = simple_tree();
        assert_abs_diff_eq!(
            3.0,
            SingleThreadedEvaluator
                .evaluate(&tree, &[], &[1.0, 2.0])
                .unwrap(),
            epsilon = f64::EPSILON
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    fn variable_eval() {
        let tree = simple_tree_with_variables();
        assert_abs_diff_eq!(
            3.0,
            SingleThreadedEvaluator
                .evaluate(&tree, &[1.0, 2.0], &[])
                .unwrap(),
            epsilon = f64::EPSILON
        );
    }
}
