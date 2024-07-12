mod glsl;

pub use glsl::generate_shader;
use std::fmt::Formatter;

#[derive(Clone, Debug)]
pub enum FunctionSanitizeError {
    InvalidReturnType {
        expected: ::glsl::syntax::FullySpecifiedType,
        found: ::glsl::syntax::FullySpecifiedType,
    },
    InvalidParameter {
        parameter: ::glsl::syntax::FunctionParameterDeclaration,
        position: usize,
        expected_type: ::glsl::syntax::TypeSpecifier,
    },
}

impl std::fmt::Display for FunctionSanitizeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use ::glsl::transpiler::glsl::*;
        match self {
            | FunctionSanitizeError::InvalidReturnType { expected, found } => {
                let mut expected_fmt = String::new();
                let mut found_fmt = String::new();
                show_fully_specified_type(&mut expected_fmt, expected);
                show_fully_specified_type(&mut found_fmt, found);
                write!(
                    f,
                    "invalid return type (expected {expected_fmt}, found {found_fmt})"
                )
            }
            | FunctionSanitizeError::InvalidParameter {
                parameter,
                position,
                expected_type,
            } => {
                let (id_fmt, type_qualifier, type_specifier) = match &parameter {
                    | ::glsl::syntax::FunctionParameterDeclaration::Named(
                        qualifier,
                        declarator,
                    ) => {
                        let mut id = String::new();
                        show_arrayed_identifier(&mut id, &declarator.ident);
                        (id, qualifier, declarator.ty.clone())
                    }
                    | ::glsl::syntax::FunctionParameterDeclaration::Unnamed(
                        qualifier,
                        ty,
                    ) => ("<anonymous>".into(), qualifier, ty.clone()),
                };
                let mut expected_fmt = String::new();
                let mut qualifier_fmt = String::new();
                let mut found_fmt = String::new();
                show_type_specifier(&mut expected_fmt, expected_type);
                if let Some(qualifier) = type_qualifier {
                    show_type_qualifier(&mut qualifier_fmt, qualifier);
                    qualifier_fmt.push(' ')
                };
                show_type_specifier(&mut found_fmt, &type_specifier);

                write!(
                    f,
                    "invalid parameter `{id_fmt}` in position {position} (\
                    expected scalar `{expected_fmt}`,\
                     found `{qualifier_fmt}{expected_fmt})`"
                )
            }
        }
    }
}

impl std::error::Error for FunctionSanitizeError {}
