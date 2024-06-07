use std::borrow::Cow;
use std::fmt::Formatter;

use glsl_lang::ast::{FunctionDefinition, SmolStr};
use itertools::Itertools;
use nalgebra::DMatrix;
use wgpu::naga::FastHashMap;

pub struct StaticEvaluator {
    functions: FastHashMap<usize, crate::ops::gpu::Function>,
    function_names: FastHashMap<crate::ops::gpu::Function, SmolStr>,
    wgpu_state: WgpuState,
    preimage: DMatrix<f32>,
    image: Box<[f32]>,
    constant_pool_size: usize,
}

struct WgpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    write_stage: wgpu::Buffer,
    read_stage: wgpu::Buffer,
    storage: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
}

#[derive(thiserror::Error, Debug)]
pub enum StaticEvaluatorError {
    #[error(
        r"preimage and image must have the same number of rows
         (preimage has {preimage_rows}, image has {image_rows})"
    )]
    Dataset {
        preimage_rows: usize,
        image_rows: usize,
    },
    #[error(
        r"batch size must not be larger than preimage
         (batch size: {batch_size}, preimage: {preimage_rows} rows)"
    )]
    BatchSize {
        batch_size: usize,
        preimage_rows: usize,
    },
    #[error("shaderc error")]
    ShaderC(#[from] Option<shaderc::Error>),
    #[error("naga error: {0}")]
    Naga(#[from] naga::front::glsl::ParseError),
    #[error("failed to sanitize functions: {0}")]
    FunctionSanitize(#[from] FunctionSanitizeError),
    #[error("WGPU error: {0}")]
    Wgpu(SmolStr),
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

mod rpn {
    use crevice::std140::AsStd140;

    use crate::ops::gpu::Function;
    use crate::tree::{NAryFunction, Node};

    pub const OPKIND_VARIABLE: usize = 0;
    pub const OPKIND_CONSTANT: usize = 1;
    pub const OPKIND_FUNCTION: usize = 2;

    #[derive(AsStd140)]
    pub struct TreeNode {
        opkind: u32,
        immediate: u32,
        function_id: u32,
    }

    impl TreeNode {
        pub fn traverse(root: &Node<f32, Function>, stack: &mut Vec<Self>) {
            match root {
                | Node::Constant(constant) => stack.push(Self {
                    opkind: OPKIND_CONSTANT as u32,
                    immediate: constant.id as u32,
                    function_id: 0,
                }),
                | Node::Variable(variable) => stack.push(Self {
                    opkind: OPKIND_VARIABLE as u32,
                    immediate: variable.id as u32,
                    function_id: 0,
                }),
                | Node::Function(function) => {
                    stack.push(Self {
                        opkind: OPKIND_FUNCTION as u32,
                        immediate: function.function.arity() as u32,
                        function_id: function.function.id() as u32,
                    });
                    for operand in &*function.operands {
                        Self::traverse(operand, stack);
                    }
                }
            }
        }
    }
}

mod macro_identifiers {
    pub mod opkind {
        pub const VARIABLE: &str = "OPKIND_VARIABLE";
        pub const CONSTANT: &str = "OPKIND_CONSTANT";
        pub const FUNCTION: &str = "OPKIND_FUNCTION";
    }
    pub const BATCH_SIZE: &str = "BATCH_SIZE";
    pub const PERMUTATIONS: &str = "PERMUTATIONS";
    pub const STACK_SIZE: &str = "STACK_SIZE";
    pub const FUNCTION_EVALUATION: &str = "FUNCTION_EVALUATION";
    pub const FUNCTION_DEFINITIONS: &str = "FUNCTION_DEFINITIONS";
}

impl StaticEvaluator {
    const SHADER_SOURCE: &'static str =
        include_str!(crate::proot!("shaders/src/skeleton.comp"));

    pub fn new(
        functions: impl IntoIterator<Item = FunctionDefinition>,
        batch_size: usize,
        permutations: usize,
        stack_size: usize,
        constant_pool_size: usize,
        preimage: DMatrix<f32>,
        image: Box<[f32]>,
    ) -> Result<Self, StaticEvaluatorError> {
        if preimage.nrows() != image.len() {
            return Err(StaticEvaluatorError::Dataset {
                preimage_rows: preimage.nrows(),
                image_rows: image.len(),
            });
        }

        if batch_size > preimage.nrows() {
            return Err(StaticEvaluatorError::BatchSize {
                batch_size,
                preimage_rows: preimage.nrows(),
            });
        }

        let (function_names, shader, wg_size) =
            Self::generate_shader(functions, batch_size, permutations, stack_size)?;
        assert_eq!(wg_size, [batch_size as u32, permutations as u32, 0]);
        let functions = function_names.keys().map(|fun| (fun.id(), *fun)).collect();

        let wgpu_state = WgpuState::new(
            &preimage,
            &image,
            batch_size,
            permutations,
            stack_size,
            constant_pool_size,
            shader,
        )?;

        Ok(Self {
            functions,
            function_names,
            wgpu_state,
            preimage,
            image,
            constant_pool_size,
        })
    }

    fn generate_shader(
        functions: impl IntoIterator<Item = FunctionDefinition>,
        batch_size: usize,
        permutations: usize,
        stack_size: usize,
    ) -> Result<
        (
            FastHashMap<crate::ops::gpu::Function, SmolStr>,
            naga::Module,
            [u32; 3],
        ),
        StaticEvaluatorError,
    > {
        let defines = [
            (macro_identifiers::opkind::VARIABLE, rpn::OPKIND_VARIABLE),
            (macro_identifiers::opkind::CONSTANT, rpn::OPKIND_CONSTANT),
            (macro_identifiers::opkind::FUNCTION, batch_size),
            (macro_identifiers::BATCH_SIZE, batch_size),
            (macro_identifiers::PERMUTATIONS, permutations),
            (macro_identifiers::STACK_SIZE, stack_size),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();

        let sanitized = Self::sanitize_function_definitions(functions)?;

        let evaluation = Self::generate_function_evaluation(
            sanitized.iter().map(|(_def, fun, _name)| fun).cloned(),
            stack_size,
        );

        let function_names = sanitized
            .iter()
            .map(|(_def, fun, name)| (*fun, name.clone()))
            .collect();

        let definitions_text = sanitized
            .iter()
            .map(|(def, _fun, _name)| def)
            .map(|fun| {
                let mut text = String::new();
                glsl_lang::transpiler::glsl::show_function_definition(
                    &mut text,
                    fun,
                    &mut Default::default(),
                )?;
                Ok::<_, std::fmt::Error>(text)
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to render functions")
            .join("\n\n");

        let evaluation_text = {
            let mut text = String::new();
            glsl_lang::transpiler::glsl::show_function_definition(
                &mut text,
                &evaluation,
                &mut Default::default(),
            )
            .expect("Failed to render function evaluation");
            text
        };

        let shader_source = format!(
            "{}\n{}\n{}",
            Self::SHADER_SOURCE,
            definitions_text,
            evaluation_text
        );

        let mut naga_front = naga::front::glsl::Frontend::default();
        let options = naga::front::glsl::Options {
            stage: naga::ShaderStage::Compute,
            defines,
        };

        let module = naga_front.parse(&options, &shader_source)?;
        let wg_size = naga_front.metadata().workgroup_size;

        Ok((function_names, module, wg_size))
    }

    fn sanitize_function_definitions(
        functions: impl IntoIterator<Item = FunctionDefinition>,
    ) -> Result<
        Vec<(FunctionDefinition, crate::ops::gpu::Function, SmolStr)>,
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
    ) -> FunctionDefinition {
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

        let switch = StatementData::Switch(
            SwitchStatementData {
                head: Box::new(ExprData::variable(id_param_ident).into_node()),
                body: functions
                    .into_iter()
                    .flat_map(|function| {
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
                        [label, ret]
                    })
                    .collect::<Vec<_>>(),
            }
            .into_node(),
        )
        .into_node();

        let ret = StatementData::Jump(
            JumpStatementData::Return(Some(Box::new(
                ExprData::FloatConst(0.0).into_node(),
            )))
            .into_node(),
        )
        .into_node();

        FunctionDefinitionData {
            prototype: prototype.into_node(),
            statement: CompoundStatementData {
                statement_list: vec![switch, ret],
            }
            .into_node(),
        }
        .into_node()
    }

    fn generate_function_name(id: usize) -> SmolStr {
        format!("_function{id}").into()
    }
}

impl WgpuState {
    fn new(
        preimage: &DMatrix<f32>,
        image: &[f32],
        batch_size: usize,
        permutations: usize,
        stack_size: usize,
        constant_pool_size: usize,
        shader: naga::Module,
    ) -> Result<Self, StaticEvaluatorError> {
        assert_eq!(preimage.nrows(), image.len());
        let preimage_row_major = preimage
            .row_iter()
            .flat_map(|row| row.iter().copied().collect_vec())
            .collect_vec();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN
                | wgpu::Backends::DX12
                | wgpu::Backends::METAL,
            ..Default::default()
        });

        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .ok_or(StaticEvaluatorError::Wgpu(
                "no suitable adapter found".into(),
            ))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::PUSH_CONSTANTS
                    | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
                required_limits: wgpu::Limits {
                    max_push_constant_size: 64,
                    ..Default::default()
                },
            },
            None,
        ))
        .map_err(|err| {
            StaticEvaluatorError::Wgpu(format!("device request failed: {err}").into())
        })?;

        let workgroup_size = [batch_size as u32, permutations as u32, 0];
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Naga(Cow::Owned(shader)),
        });

        todo!()
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

impl std::error::Error for FunctionSanitizeError {}

#[cfg(test)]
mod tests {
    use crate::tree::eval::gpu::statik::{StaticEvaluator, StaticEvaluatorError};
    use glsl_lang::ast::FunctionDefinition;
    use glsl_lang::parse::Parsable;
    use nalgebra::DMatrix;

    fn function_defs() -> Result<Vec<FunctionDefinition>, Box<dyn std::error::Error>> {
        const FUNCTION_DEFS_TEXT: &[&str] = &[
            r"
            float add(float lhs, float rhs) {
                return lhs + rhs;
            }
            ",
            r"
            float sub(float lhs, float rhs) {
                return lhs - rhs;
            }
            ",
            r"
            float mul(float lhs, float rhs) {
                return lhs * rhs;
            }
            ",
            r"
            float div(float lhs, float rhs) {
                return lhs / rhs;
            }
            ",
            r"
            float neg(float rhs) {
                return rhs * -1;
            }
            ",
            r"
            float sin(float x) {
                return sin(x);
            }
            ",
            r"
            float clamp(float x, float lo, float hi) {
                if (x < lo) {
                    return lo;
                }
                if (x > hi) {
                    return hi;
                }
                return x;
            }
            ",
        ];

        Ok(FUNCTION_DEFS_TEXT
            .iter()
            .cloned()
            .map(glsl_lang::ast::FunctionDefinition::parse)
            .collect::<Result<Vec<_>, _>>()?)
    }

    fn test_data() -> (DMatrix<f32>, Box<[f32]>) {
        let dimensions = 4;
        let rows = 1024;
        let preimage =
            DMatrix::from_iterator(rows, dimensions, (0..).take(dimensions * rows))
                .cast::<f32>();
        let image = (0..).take(rows).map(|n| n as f32).collect::<Box<[f32]>>();
        (preimage, image)
    }

    #[test]
    fn parse_function_defs() {
        function_defs().expect("Failed to parse function definitions");
    }

    #[test]
    fn test_shader_gen() -> Result<(), StaticEvaluatorError> {
        let defs = function_defs().expect("Failed to parse function definitions");

        let (preimage, image) = test_data();
        let batch_size = 64;
        let permutations = 32;
        let stack_size = 96;
        let constant_pool_size = 16;

        StaticEvaluator::new(
            defs,
            batch_size,
            permutations,
            stack_size,
            constant_pool_size,
            preimage,
            image,
        )
        .map(|_| ())
    }
}
