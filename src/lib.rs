use lalrpop_util::lalrpop_mod;

pub mod ops;
pub mod tree;
mod utility;

lalrpop_mod!(parse);
mod ast;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast;
    use smol_str::ToSmolStr;
    use std::str::FromStr;

    #[test]
    fn test_constant() {
        let parser = parse::ExpressionParser::new();
        assert_eq!(
            Ok(ast::Node::Constant(f32::from_str("-1234.0e+56").unwrap())),
            parser.parse("-1234.0e+56")
        );
    }

    #[test]
    fn test_function() {
        let parser = parse::ExpressionParser::new();
        assert_eq!(
            Ok(ast::Node::Function(ast::Function {
                ident: "foo4_bar5".to_smolstr(),
                args: vec![
                    ast::Node::Function(ast::Function {
                        ident: "baz31".to_smolstr(),
                        args: vec![ast::Node::Constant(1.2), ast::Node::Constant(2.3)],
                    }),
                    ast::Node::Constant(4.5),
                    ast::Node::Constant(5.6),
                    ast::Node::Function(ast::Function {
                        ident: "sample".to_smolstr(),
                        args: vec![ast::Node::Constant(3.0)],
                    })
                ]
            })),
            parser.parse("(foo4_bar5 (baz31 1.2 2.3) 4.5 5.6 (sample 3.0))")
        )
    }
}
