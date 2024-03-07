use crate::tree::{NAryFunction, Node};
use anyhow::Context;
use glsl_lang::ast::{FunctionDefinition, FunctionPrototype};
use shaderc::{CompileOptions, Compiler, OptimizationLevel, SourceLanguage, TargetEnv};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::ops::Index;

pub struct DynamicEvaluator<'a> {
    compiler: Compiler,
    options: CompileOptions<'a>,
}

#[derive(Debug, Clone, PartialEq)]
enum GlslFunction {
    Builtin(BuiltinGlslFunction),
    Custom(CustomGlslFunction),
}

#[derive(Debug, Clone)]
pub enum GlslFunctionDefinition {
    Builtin(BuiltinGlslFunction),
    Custom(CustomGlslFunctionDefinition),
}

#[derive(Debug, Clone, PartialEq)]
enum BuiltinGlslFunction {
    NonOperatorFunction(GlslNonOperatorFunction),
    Operator(GlslOperator),
}

#[derive(Debug, Clone, PartialEq)]
struct GlslNonOperatorFunction {
    prototype: FunctionPrototype,
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
enum GlslOperator {
    Unary(GlslUnaryOperator),
    Binary(GlslBinaryOperator),
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
enum GlslUnaryOperator {
    Neg,
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
enum GlslBinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone)]
struct CustomGlslFunction {
    id: usize,
    arity: usize,
    ast: FunctionDefinition,
}

#[derive(Debug, Clone)]
struct CustomGlslFunctionDefinition {
    ast: FunctionDefinition,
}

impl NAryFunction for BuiltinGlslFunction {
    fn arity(&self) -> usize {
        match self {
            | BuiltinGlslFunction::NonOperatorFunction(non_operator) => {
                non_operator.arity()
            }
            | BuiltinGlslFunction::Operator(operator) => operator.arity(),
        }
    }
}

impl NAryFunction for GlslNonOperatorFunction {
    fn arity(&self) -> usize {
        self.prototype.parameters.len()
    }
}

impl NAryFunction for GlslOperator {
    fn arity(&self) -> usize {
        match self {
            | GlslOperator::Unary(unary) => unary.arity(),
            | GlslOperator::Binary(binary) => binary.arity(),
        }
    }
}

impl GlslUnaryOperator {
    fn postfix(self) -> bool {
        match self {
            | GlslUnaryOperator::Neg => false,
        }
    }
}

impl NAryFunction for GlslUnaryOperator {
    fn arity(&self) -> usize {
        1
    }
}

impl NAryFunction for GlslBinaryOperator {
    fn arity(&self) -> usize {
        2
    }
}

impl From<GlslUnaryOperator> for glsl_lang::ast::UnaryOp {
    fn from(value: GlslUnaryOperator) -> Self {
        use glsl_lang::ast::*;
        match value {
            | GlslUnaryOperator::Neg => UnaryOp::new(UnaryOpData::Minus, None),
        }
    }
}

impl TryFrom<glsl_lang::ast::UnaryOpData> for GlslUnaryOperator {
    type Error = ();

    fn try_from(value: glsl_lang::ast::UnaryOpData) -> Result<Self, Self::Error> {
        use glsl_lang::ast::*;
        Ok(match value {
            | UnaryOpData::Minus => Self::Neg,
            | _ => Err(())?,
        })
    }
}

impl From<GlslBinaryOperator> for glsl_lang::ast::BinaryOpData {
    fn from(value: GlslBinaryOperator) -> Self {
        use glsl_lang::ast::*;
        match value {
            | GlslBinaryOperator::Add => BinaryOpData::Add,
            | GlslBinaryOperator::Sub => BinaryOpData::Sub,
            | GlslBinaryOperator::Mul => BinaryOpData::Mult,
            | GlslBinaryOperator::Div => BinaryOpData::Div,
        }
    }
}

impl TryFrom<glsl_lang::ast::BinaryOpData> for GlslBinaryOperator {
    type Error = ();

    fn try_from(value: glsl_lang::ast::BinaryOpData) -> Result<Self, Self::Error> {
        use glsl_lang::ast::*;
        Ok(match value {
            | BinaryOpData::Add => Self::Add,
            | BinaryOpData::Sub => Self::Sub,
            | BinaryOpData::Mult => Self::Mul,
            | BinaryOpData::Div => Self::Div,
            | _ => Err(())?,
        })
    }
}

impl CustomGlslFunction {
    fn from_definition(definition: &CustomGlslFunctionDefinition, id: usize) -> Self {
        Self {
            id,
            arity: definition.ast.prototype.parameters.len(),
            ast: definition.ast.clone(),
        }
    }
}

impl PartialOrd for CustomGlslFunction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CustomGlslFunction {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialEq for CustomGlslFunction {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id)
    }
}

impl Eq for CustomGlslFunction {}

impl NAryFunction for CustomGlslFunction {
    fn arity(&self) -> usize {
        debug_assert_eq!(self.arity, self.ast.prototype.parameters.len());
        self.arity
    }
}

impl NAryFunction for CustomGlslFunctionDefinition {
    fn arity(&self) -> usize {
        self.ast.prototype.parameters.len()
    }
}

impl<'a> DynamicEvaluator<'a> {
    const SHADER_SOURCE: &'static str = "";
    const OPERATORS_MACRO_IDENTIFIER: &'static str = "OPERATORS";

    pub fn new(functions: &[GlslFunctionDefinition]) -> anyhow::Result<Self> {
        let compiler =
            Compiler::new().with_context(|| "Failed to initialise SPIRV compiler")?;
        let mut options = CompileOptions::new()
            .with_context(|| "Failed to initialise SPIRV compile options")?;

        options.set_target_env(TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_3 as u32);
        options.set_generate_debug_info();
        options.set_optimization_level(OptimizationLevel::Performance);
        options.set_source_language(SourceLanguage::GLSL);

        options.add_macro_definition(Self::OPERATORS_MACRO_IDENTIFIER, Some(todo!()));

        Ok(Self { compiler, options })
    }

    fn generate_functions(
        definitions: &[GlslFunctionDefinition],
    ) -> BTreeMap<crate::ops::gpu::Function, GlslFunction> {
        use crate::ops::gpu;
        definitions
            .iter()
            .enumerate()
            .map(|(idx, definition)| match definition {
                | GlslFunctionDefinition::Builtin(builtin) => (
                    gpu::Function::new(idx, builtin.arity()),
                    GlslFunction::Builtin(builtin.clone()),
                ),
                | GlslFunctionDefinition::Custom(custom) => (
                    gpu::Function::new(idx, custom.arity()),
                    GlslFunction::Custom(CustomGlslFunction::from_definition(
                        custom, idx,
                    )),
                ),
            })
            .collect::<BTreeMap<gpu::Function, GlslFunction>>()
    }

    fn generate_eval_expr_ast(
        tree: &Node<f32, crate::ops::gpu::Function>,
        operators: &impl Index<crate::ops::gpu::Function, Output = GlslFunction>,
    ) -> glsl_lang::ast::Expr {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glsl_lang::ast;
    use glsl_lang::ast::NodeDisplay;

    #[test]
    fn render() -> anyhow::Result<()> {
        use glsl_lang::ast::*;
        use glsl_lang::parse::DefaultParse;
        use glsl_lang::transpiler::glsl::*;

        let source = r"
            float mul(float a, float b) {
                float result = a * b;
                return result;
            }";

        let ast = ast::TranslationUnit::parse(source)?;

        let mut rendered = String::new();
        show_translation_unit(&mut rendered, &ast, FormattingState::default())?;

        println!("rendered GLSL AST:\n{}", rendered);

        Ok(())
    }
}
