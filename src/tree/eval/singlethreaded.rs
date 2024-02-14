use crate::ops::cpu;
use crate::tree::eval::Evaluator;
use crate::tree::{Node, Tree};
use std::convert::Infallible;

pub struct SingleThreadedEvaluator;

impl<T: Clone> Evaluator<T, cpu::UnaryOp<T>, cpu::BinaryOp<T>>
    for SingleThreadedEvaluator
{
    type R = T;
    type E = Infallible;

    fn evaluate(
        &mut self,
        tree: &Node<T, cpu::UnaryOp<T>, cpu::BinaryOp<T>>,
        variables: &[T],
    ) -> Result<Self::R, Self::E> {
        match tree {
            | Node::Constant(crate::tree::Constant { value }) => {
                assert!(variables.is_empty());
                Ok(value.clone())
            }
            | Node::Variable(crate::tree::Variable { .. }) => Ok(variables[0].clone()),
            | Node::UnaryOp(crate::tree::UnaryOp { operator, operand }) => {
                Ok(operator.apply(self.evaluate(operand, variables)?))
            }
            | Node::BinaryOp(crate::tree::BinaryOp { operator, lhs, rhs }) => {
                let lhs_variable_count = lhs.variable_count();
                Ok(operator.apply(
                    self.evaluate(lhs, &variables[..lhs_variable_count])?,
                    self.evaluate(rhs, &variables[lhs_variable_count..])?,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{BinaryOp, Constant};
    use approx::assert_abs_diff_eq;

    fn simple_tree() -> Node<f64, cpu::UnaryOp<f64>, cpu::BinaryOp<f64>> {
        Node::BinaryOp(BinaryOp::new(
            cpu::BinaryOp::new(std::ops::Add::add),
            Node::Constant(Constant::new(1f64)),
            Node::Constant(Constant::new(2f64)),
        ))
    }

    #[test]
    fn simple_eval() {
        let tree = simple_tree();
        assert_abs_diff_eq!(
            3.0f64,
            SingleThreadedEvaluator.evaluate(&tree, &[],).unwrap(),
            epsilon = f64::EPSILON
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn invalid_variable_count() {
        let tree = simple_tree();
        SingleThreadedEvaluator.evaluate(&tree, &[1.0]).unwrap();
    }
}
