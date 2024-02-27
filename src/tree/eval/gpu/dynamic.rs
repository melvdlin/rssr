use anyhow::Context;
use shaderc::{CompileOptions, Compiler, OptimizationLevel, SourceLanguage, TargetEnv};

pub struct DynamicEvaluator<'a> {
    compiler: Compiler,
    options: CompileOptions<'a>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct GlslFunction {
    identifier: String,
    definition: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct GlslUnaryOperator(GlslFunction);

#[derive(Debug, Clone, Eq, PartialEq)]
struct GlslBinaryOperator(GlslFunction);

impl<'a> DynamicEvaluator<'a> {
    const SHADER_SOURCE: &'static str = todo!();
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

        options.add_macro_definition(Self::OPERATORS_MACRO_IDENTIFIER, Some(todo!()));

        Ok(Self { compiler, options })
    }

    fn generate_operators() -> ! {
        todo!()
    }
}
