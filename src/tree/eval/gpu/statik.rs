use anyhow::Context;
use shaderc::{
    CompileOptions, Compiler, OptimizationLevel, ShaderKind, SourceLanguage, TargetEnv,
};
use std::fmt::Formatter;
use wgpu::naga::FastHashMap;

pub struct StaticEvaluator {
    functions: FastHashMap<usize, crate::ops::gpu::Function>,
}

#[derive(Clone, Debug)]
pub enum FuncionSanitizeError {
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
    ) -> Vec<(
        glsl_lang::ast::FunctionDefinition,
        crate::ops::gpu::Function,
        String,
    )> {
        functions
            .into_iter()
            .map(|function| {
                let arity = function.prototype.parameters.len();

                todo!()
            })
            .collect()
    }

    fn generate_function_evaluation(
        functions: impl IntoIterator<Item = crate::ops::gpu::Function>,
    ) -> glsl_lang::ast::FunctionDefinition {
        todo!()
    }

    fn generate_dispatcher() -> String {
        todo!()
    }

    fn generate_operators() -> String {
        todo!()
    }
}

impl std::fmt::Display for FuncionSanitizeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use glsl_lang::transpiler::glsl::*;
        match self {
            | FuncionSanitizeError::InvalidReturnType { expected, found } => {
                let mut expected_fmt = String::new();
                let mut found_fmt = String::new();
                show_fully_specified_type(
                    &mut expected_fmt,
                    expected,
                    &mut FormattingState::default(),
                )?;
                show_fully_specified_type(
                    &mut found_fmt,
                    found,
                    &mut FormattingState::default(),
                )?;
                write!(
                    f,
                    "invalid return type (expected {expected_fmt}, found {found_fmt})"
                )
            }
            | FuncionSanitizeError::InvalidParameter {
                parameter,
                position,
                expected_type,
            } => {
                let (id_fmt, type_qualifier, type_specifier) = match &parameter.content {
                    | glsl_lang::ast::FunctionParameterDeclarationData::Named(
                        qualifier,
                        declarator,
                    ) => (
                        format!("`{}` ", declarator.content.ident.ident.0),
                        qualifier,
                        declarator.content.ty.clone(),
                    ),
                    | glsl_lang::ast::FunctionParameterDeclarationData::Unnamed(
                        qualifier,
                        ty,
                    ) => (String::new(), qualifier, ty.clone()),
                };
                let mut expected_fmt = String::new();
                let mut qualifier_fmt = String::new();
                let mut found_fmt = String::new();
                show_type_specifier(
                    &mut expected_fmt,
                    expected_type,
                    &mut FormattingState::default(),
                )?;
                if let Some(qualifier) = type_qualifier {
                    show_type_qualifier(
                        &mut qualifier_fmt,
                        &qualifier,
                        &mut FormattingState::default(),
                    )?;
                    qualifier_fmt.push(' ')
                };
                show_type_specifier(
                    &mut found_fmt,
                    &type_specifier,
                    &mut FormattingState::default(),
                )?;

                write!(
                    f,
                    "invalid parameter {id_fmt}in position {position} (\
                    expected `{expected_fmt}`,\
                     found `{qualifier_fmt}{expected_fmt})`"
                )
            }
        }
    }
}
