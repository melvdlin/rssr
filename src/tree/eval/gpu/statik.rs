use anyhow::Context;
use shaderc::{
    CompileOptions, Compiler, OptimizationLevel, ShaderKind, SourceLanguage, TargetEnv,
};

pub struct StaticEvaluator {}

impl StaticEvaluator {
    const SHADER_SOURCE: &'static str = todo!();
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

    fn generate_dispatcher() -> String {
        todo!()
    }

    fn generate_operators() -> String {
        todo!()
    }
}
