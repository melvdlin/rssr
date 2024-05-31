use anyhow::Context;
use glsl_lang::ast::{NodeContent, SmolStr};
use itertools::multiunzip;
use shaderc::{
    CompileOptions, Compiler, OptimizationLevel, ShaderKind, SourceLanguage, TargetEnv,
};
use std::fmt::Formatter;
use std::io::Read;
use wgpu::naga::{FastHashMap, Statement};

pub struct StaticEvaluator {
    functions: FastHashMap<usize, crate::ops::gpu::Function>,
}

#[derive(Clone, Debug)]
pub enum FunctionSanitizeError {
    InvalidReturnType {
        expected: glsl_lang::ast::FullySpecifiedType,
        found: glsl_lang::ast::FullySpecifiedType,
    },
    InvalidParameter {
        parameter: glsl_lang::ast::FunctionParameterDeclaration,
        position: usize,
        expected_type: glsl_lang::ast::TypeSpecifier,
    },
}

mod macro_identifiers {
    mod opkind {
        const PREFIX: &str = "OPKIND";
        const VARIABLE: &str = "OPKIND_VARIABLE";
        const CONSTANT: &str = "OPKIND_CONSTANT";
        const BUILTIN_OPERATOR: &str = "OPKIND_BUILTIN_OPERATOR";
        const BUILTIN_FUNCTION: &str = "OPKIND_CUSTOM_FUNCTION";
    }
    const BATCH_SIZE: &str = "BATCH_SIZE";
    const PERMUTATIONS: &str = "PERMUTATIONS";
    const STACK_SIZE: &str = "STACK_SIZE";
}

impl StaticEvaluator {
    const SHADER_SOURCE: &'static str = "";
    const DISPATCH_MACRO_IDENTIFIER: &'static str = "DISPATCH";
    const OPERATORS_MACRO_IDENTIFIER: &'static str = "OPERATORS";

    pub fn new(operators: &[crate::ops::gpu::Function]) -> anyhow::Result<Self> {
        let compiler =
            Compiler::new().with_context(|| "Failed to initialise SPIRV compiler")?;
        let mut options = CompileOptions::new()
            .with_context(|| "Failed to initialise SPIRV compile options")?;

        options.set_target_env(TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_3 as u32);
        options.set_generate_debug_info();
        options.set_optimization_level(OptimizationLevel::Performance);
        options.set_source_language(SourceLanguage::GLSL);

        options.add_macro_definition(
            Self::DISPATCH_MACRO_IDENTIFIER,
            Some(&Self::generate_dispatcher()),
        );
        options.add_macro_definition(
            Self::OPERATORS_MACRO_IDENTIFIER,
            Some(&Self::generate_operators()),
        );

        let artifact = compiler.compile_into_spirv(
            Self::SHADER_SOURCE,
            ShaderKind::Compute,
            "rssr::tree::eval::gpu::StaticEvaluator::SHADER_SOURCE",
            "main",
            Some(&options),
        )?;

        todo!()
    }

    fn sanitize_function_definitions(
        functions: impl IntoIterator<Item = glsl_lang::ast::FunctionDefinition>,
    ) -> Result<
        Vec<(
            glsl_lang::ast::FunctionDefinition,
            crate::ops::gpu::Function,
            SmolStr,
        )>,
        FunctionSanitizeError,
    > {
        use glsl_lang::ast::*;

        fn validate_return_type(
            function: &FunctionDefinition,
        ) -> Result<(), FunctionSanitizeError> {
            if function.prototype.ty.qualifier.is_some()
                || function.prototype.ty.ty.array_specifier.is_some()
                || !matches!(
                    *function.prototype.ty.ty.ty,
                    TypeSpecifierNonArrayData::Float
                )
            {
                return Err(FunctionSanitizeError::InvalidReturnType {
                    expected: FullySpecifiedTypeData {
                        qualifier: None,
                        ty: TypeSpecifierData {
                            ty: TypeSpecifierNonArray::new(
                                TypeSpecifierNonArrayData::Float,
                                None,
                            ),
                            array_specifier: None,
                        }
                        .into(),
                    }
                    .into(),
                    found: function.prototype.ty.clone(),
                });
            }
            Ok(())
        }

        fn validate_parameter_type(
            position: usize,
            parameter: &FunctionParameterDeclaration,
        ) -> Result<(), FunctionSanitizeError> {
            let (qualifier, ty) = match &**parameter {
                | FunctionParameterDeclarationData::Named(qualifier, declarator) => {
                    (qualifier, &declarator.ty)
                }
                | FunctionParameterDeclarationData::Unnamed(qualifier, ty) => {
                    (qualifier, ty)
                }
            };
            if qualifier.is_some()
                || !matches!(
                    ty,
                    TypeSpecifier {
                        content: TypeSpecifierData {
                            ty: TypeSpecifierNonArray {
                                content: TypeSpecifierNonArrayData::Float,
                                ..
                            },
                            array_specifier: None
                        },
                        ..
                    }
                )
            {
                return Err(FunctionSanitizeError::InvalidParameter {
                    parameter: parameter.clone(),
                    position,
                    expected_type: TypeSpecifier {
                        content: TypeSpecifierData {
                            ty: TypeSpecifierNonArray {
                                content: TypeSpecifierNonArrayData::Float,
                                span: None,
                            },
                            array_specifier: None,
                        },
                        span: None,
                    },
                });
            }
            Ok(())
        }

        functions
            .into_iter()
            .enumerate()
            .map(|(id, mut function)| {
                validate_return_type(&function)?;

                for (position, parameter) in
                    function.prototype.parameters.iter().enumerate()
                {
                    validate_parameter_type(position, parameter)?;
                }
                let arity = function.prototype.parameters.len();
                let op = crate::ops::gpu::Function::new(id, arity);
                let mut ident = Self::generate_function_name(id);
                std::mem::swap(&mut function.prototype.name.0, &mut ident);

                Ok((function, op, ident))
            })
            .collect::<Result<Vec<_>, _>>()
    }

    fn generate_function_evaluation(
        functions: impl IntoIterator<Item = crate::ops::gpu::Function>,
        stack_size: usize,
    ) -> glsl_lang::ast::FunctionDefinition {
        use glsl_lang::ast::*;

        fn parameter_declaration(
            ident: Identifier,
            ty: TypeSpecifierNonArrayData,
            array: Option<ArraySpecifierData>,
        ) -> FunctionParameterDeclaration {
            FunctionParameterDeclarationData::Named(
                None,
                FunctionParameterDeclaratorData {
                    ty: TypeSpecifierData {
                        ty: ty.into_node(),
                        array_specifier: None,
                    }
                    .into_node(),
                    ident: ArrayedIdentifierData {
                        ident,
                        array_spec: array.map(Node::from),
                    }
                    .into_node(),
                }
                .into_node(),
            )
            .into_node()
        }

        // prototype:
        // `float _function(uint id, uint sp, float stack[STACK_SIZE])`
        // functions to call:
        // `float _functionXYZ(float p1, float p2, ..., float pn)`

        let function_ident = IdentifierData::from("_function").into_node();
        let id_param_ident: Identifier = IdentifierData::from("id").into_node();
        let sp_param_ident: Identifier = IdentifierData::from("sp").into_node();
        let stack_param_ident: Identifier = IdentifierData::from("stack").into_node();
        let id_param = parameter_declaration(
            id_param_ident.clone(),
            TypeSpecifierNonArrayData::UInt,
            None,
        );
        let sp_param = parameter_declaration(
            sp_param_ident.clone(),
            TypeSpecifierNonArrayData::UInt,
            None,
        );

        let stack_param = parameter_declaration(
            stack_param_ident.clone(),
            TypeSpecifierNonArrayData::Float,
            Some(ArraySpecifierData {
                dimensions: vec![ArraySpecifierDimensionData::ExplicitlySized(Box::new(
                    ExprData::UIntConst(stack_size as u32).into_node(),
                ))
                .into_node()],
            }),
        );

        let prototype = FunctionPrototypeData {
            ty: FullySpecifiedTypeData {
                qualifier: None,
                ty: TypeSpecifierData {
                    ty: TypeSpecifierNonArrayData::Float.into_node(),
                    array_specifier: None,
                }
                .into(),
            }
            .into(),
            name: function_ident,
            parameters: vec![id_param, sp_param, stack_param],
        };

        let switch: Statement = StatementData::Switch(
            SwitchStatementData {
                head: Box::new(ExprData::variable(id_param_ident).into_node()),
                body: functions
                    .into_iter()
                    .map(|function| {
                        let id = function.id;
                        let arity = function.arity as u32;
                        let ident =
                            IdentifierData(Self::generate_function_name(id)).into_node();

                        let label = StatementData::CaseLabel(
                            CaseLabelData::Case(Box::new(
                                ExprData::UIntConst(id as u32).into_node(),
                            ))
                            .into_node(),
                        )
                        .into_node();

                        let ret = StatementData::Jump(
                            JumpStatementData::Return(Some(Box::new(
                                ExprData::FunCall(
                                    FunIdentifierData::Expr(Box::new(
                                        ExprData::variable(ident).into_node(),
                                    ))
                                    .into_node(),
                                    (1..=arity)
                                        .map(|offset| {
                                            ExprData::Bracket(
                                                Box::new(
                                                    ExprData::variable(
                                                        stack_param_ident.clone(),
                                                    )
                                                    .into_node(),
                                                ),
                                                Box::new(
                                                    ExprData::Binary(
                                                        BinaryOpData::Sub.into_node(),
                                                        Box::new(
                                                            ExprData::variable(
                                                                sp_param_ident.clone(),
                                                            )
                                                            .into_node(),
                                                        ),
                                                        Box::new(
                                                            ExprData::UIntConst(offset)
                                                                .into_node(),
                                                        ),
                                                    )
                                                    .into_node(),
                                                ),
                                            )
                                            .into_node()
                                        })
                                        .collect(),
                                )
                                .into_node(),
                            )))
                            .into_node(),
                        )
                        .into_node();

                        StatementData::Compound(
                            CompoundStatementData {
                                statement_list: vec![label, ret],
                            }
                            .into_node(),
                        )
                        .into_node()
                    })
                    .chain(std::iter::once({
                        let label =
                            StatementData::CaseLabel(CaseLabelData::Def.into_node())
                                .into_node();

                        let ret = StatementData::Jump(
                            JumpStatementData::Return(Some(Box::new(
                                ExprData::FloatConst(0.0).into_node(),
                            )))
                            .into_node(),
                        )
                        .into_node();

                        StatementData::Compound(
                            CompoundStatementData {
                                statement_list: vec![label, ret],
                            }
                            .into_node(),
                        )
                        .into_node()
                    }))
                    .collect(),
            }
            .into_node(),
        )
        .into_node();

        FunctionDefinitionData {
            prototype: prototype.into_node(),
            statement: CompoundStatementData {
                statement_list: vec![switch],
            }
            .into_node(),
        }
        .into_node()
    }

    fn generate_function_name(id: usize) -> SmolStr {
        format!("_function{id}").into()
    }
}

impl std::fmt::Display for FunctionSanitizeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use glsl_lang::transpiler::glsl::*;
        match self {
            | FunctionSanitizeError::InvalidReturnType { expected, found } => {
                let mut expected_fmt = String::new();
                let mut found_fmt = String::new();
                show_fully_specified_type(
                    &mut expected_fmt,
                    expected,
                    &mut Default::default(),
                )?;
                show_fully_specified_type(
                    &mut found_fmt,
                    found,
                    &mut Default::default(),
                )?;
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
                let (id_fmt, type_qualifier, type_specifier) = match &parameter.content {
                    | glsl_lang::ast::FunctionParameterDeclarationData::Named(
                        qualifier,
                        declarator,
                    ) => {
                        let mut id = String::new();
                        show_arrayed_identifier(
                            &mut id,
                            &declarator.content.ident,
                            &mut Default::default(),
                        )?;
                        (id, qualifier, declarator.content.ty.clone())
                    }
                    | glsl_lang::ast::FunctionParameterDeclarationData::Unnamed(
                        qualifier,
                        ty,
                    ) => ("<anonymous>".into(), qualifier, ty.clone()),
                };
                let mut expected_fmt = String::new();
                let mut qualifier_fmt = String::new();
                let mut found_fmt = String::new();
                show_type_specifier(
                    &mut expected_fmt,
                    expected_type,
                    &mut Default::default(),
                )?;
                if let Some(qualifier) = type_qualifier {
                    show_type_qualifier(
                        &mut qualifier_fmt,
                        &qualifier,
                        &mut Default::default(),
                    )?;
                    qualifier_fmt.push(' ')
                };
                show_type_specifier(
                    &mut found_fmt,
                    &type_specifier,
                    &mut Default::default(),
                )?;

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
