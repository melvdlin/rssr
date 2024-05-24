use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::ops::Index;

use anyhow::Context;
use glsl_lang::ast;
use shaderc::{CompileOptions, Compiler, OptimizationLevel, SourceLanguage, TargetEnv};

use crate::proot;
use crate::tree;
use crate::tree::NAryFunction;

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
    prototype: ast::FunctionPrototype,
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
    ast: ast::FunctionDefinition,
}

#[derive(Debug, Clone)]
struct CustomGlslFunctionDefinition {
    ast: ast::FunctionDefinition,
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

impl From<GlslUnaryOperator> for ast::UnaryOp {
    fn from(value: GlslUnaryOperator) -> Self {
        use glsl_lang::ast::*;
        match value {
            | GlslUnaryOperator::Neg => UnaryOp::new(UnaryOpData::Minus, None),
        }
    }
}

impl TryFrom<ast::UnaryOpData> for GlslUnaryOperator {
    type Error = ();

    fn try_from(value: ast::UnaryOpData) -> Result<Self, Self::Error> {
        use glsl_lang::ast::*;
        Ok(match value {
            | UnaryOpData::Minus => Self::Neg,
            | _ => Err(())?,
        })
    }
}

impl From<GlslBinaryOperator> for ast::BinaryOpData {
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

impl tree::NAryFunction for CustomGlslFunction {
    fn arity(&self) -> usize {
        debug_assert_eq!(self.arity, self.ast.prototype.parameters.len());
        self.arity
    }
}

impl tree::NAryFunction for CustomGlslFunctionDefinition {
    fn arity(&self) -> usize {
        self.ast.prototype.parameters.len()
    }
}

impl<'a> DynamicEvaluator<'a> {
    const SKELETON_SHADER_SOURCE: &'static str =
        include_str!(proot!("shaders/src/", "skeleton.comp"));

    const USER_FUNCTION_DECLARATIONS_PATH: &'static str = "user-function-declarations";
    const USER_FUNCTION_DEFINITIONS_PATH: &'static str = "user-function-definitions";
    const EVAL_FUNCTION_DEFINITION_PATH: &'static str = "eval-function-definition";

    const PREFACTOR_FUNCTION_IDENTIFIER: &'static str = "_prefactor";
    const SAMPLE_FUNCTION_IDENTIFIER: &'static str = "_sample";
    const EVAL_FUNCTION_IDENTIFIER: &'static str = "_eval";
    const BATCH_IDX_IDENTIFIER: &'static str = "batchIdx";
    const PERMUTATION_IDX_IDENTIFIER: &'static str = "permutationIdx";
    const DIMENSION_IDENTIFIER: &'static str = "dimension";
    const LOCAL_SIZE_X_MACRO_IDENTIFIER: &'static str = "LOCAL_SIZE_X";
    const LOCAL_SIZE_Y_MACRO_IDENTIFIER: &'static str = "LOCAL_SIZE_Y";

    pub fn new(functions: &[GlslFunctionDefinition]) -> anyhow::Result<Self> {
        let compiler =
            Compiler::new().with_context(|| "Failed to initialise SPIRV compiler")?;
        let mut options = CompileOptions::new()
            .with_context(|| "Failed to initialise SPIRV compile options")?;

        options.set_target_env(TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_3 as u32);
        options.set_generate_debug_info();
        options.set_optimization_level(OptimizationLevel::Performance);
        options.set_source_language(SourceLanguage::GLSL);

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

    fn generate_eval_expr_ast<
        I: for<'b> Index<&'b crate::ops::gpu::Function, Output = GlslFunction>,
    >(
        tree: &tree::Node<f32, crate::ops::gpu::Function>,
        operators: &I,
    ) -> ast::Expr {
        use ast::*;
        fn generate_constant_ast(
            tree::Constant { id, .. }: &tree::Constant<f32>,
        ) -> ExprData {
            ExprData::FloatConst(todo!())
        }

        fn generate_variable_ast(
            tree::Variable { id, .. }: &tree::Variable<f32>,
        ) -> ExprData {
            ExprData::FunCall(
                FunIdentifier::new(
                    FunIdentifierData::Expr(
                        Expr::new(
                            ExprData::Variable(Identifier::new(
                                IdentifierData::from(
                                    DynamicEvaluator::SAMPLE_FUNCTION_IDENTIFIER,
                                ),
                                None,
                            )),
                            None,
                        )
                        .into(),
                    ),
                    None,
                ),
                vec![
                    Expr::new(
                        ExprData::Variable(Identifier::new(
                            IdentifierData::from(DynamicEvaluator::BATCH_IDX_IDENTIFIER),
                            None,
                        )),
                        None,
                    ),
                    Expr::new(
                        ExprData::Variable(Identifier::new(
                            IdentifierData::from(id.to_string().as_str()),
                            None,
                        )),
                        None,
                    ),
                ],
            )
        }

        fn generate_function_ast<
            I: for<'b> Index<&'b crate::ops::gpu::Function, Output = GlslFunction>,
        >(
            tree::Function { function, operands }: &tree::Function<
                f32,
                crate::ops::gpu::Function,
            >,
            operators: &I,
        ) -> ExprData {
            let glsl_function = &operators[function];

            ExprData::FunCall(todo!(), todo!());
            todo!()
        }

        let subtree = Expr::new(
            match tree {
                | tree::Node::Constant(constant) => generate_constant_ast(constant),
                | tree::Node::Variable(variable) => generate_variable_ast(variable),
                | tree::Node::Function(function) => {
                    generate_function_ast(function, operators)
                }
            },
            None,
        );

        let prefactor = Expr::new(
            ExprData::FunCall(
                FunIdentifier::new(
                    FunIdentifierData::Expr(
                        Expr::new(
                            ExprData::Variable(Identifier::new(
                                IdentifierData::from(Self::PREFACTOR_FUNCTION_IDENTIFIER),
                                None,
                            )),
                            None,
                        )
                        .into(),
                    ),
                    None,
                ),
                vec![],
            ),
            None,
        );

        Expr::new(
            ExprData::Binary(
                BinaryOp::new(BinaryOpData::Mult, None),
                prefactor.into(),
                subtree.into(),
            ),
            None,
        )
    }
}
#[cfg(test)]
mod tests {
    use glsl_lang::ast;

    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn new() -> anyhow::Result<()> {
        let _ = DynamicEvaluator::new(&[])?;
        Ok(())
    }

    #[test]
    fn render() -> anyhow::Result<()> {
        use glsl_lang::parse::DefaultParse;
        use glsl_lang::transpiler::glsl::*;

        let source = r"
            void foo(void) {}
            float mul(float a, float b) {
                foo();
                float result = (a * b);
                result = (1 + 2) * 3;
                return result;
            }";

        let ast = ast::TranslationUnit::parse(source)?;
        dbg!(&ast);
        let mut rendered = String::new();
        show_translation_unit(&mut rendered, &ast, FormattingState::default())?;

        println!("rendered GLSL AST:\n{}", rendered);

        Ok(())
    }
}
