use crevice::std430::AsStd430;
use std::borrow::Cow;
use std::fmt::Formatter;
use std::mem;
use std::mem::size_of;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::mpsc;

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
    constant_pool_size: NonZeroUsize,
}

struct WgpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    dataset_stage: wgpu::Buffer,
    sampling_stage: wgpu::Buffer,
    tree_stage: wgpu::Buffer,
    result_stage: wgpu::Buffer,
    dataset_storage: wgpu::Buffer,
    sampling_storage: wgpu::Buffer,
    tree_storage: wgpu::Buffer,
    result_storage: wgpu::Buffer,
    dataset_bind_group: wgpu::BindGroup,
    sampling_bind_group: wgpu::BindGroup,
    eval_bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    workgroup_size: [u32; 3],
}

#[derive(crevice::std430::AsStd430)]
struct PushConstants {
    datapoints: u32,
    preimage_dimensions: u32,
    expression_size: u32,
}

#[derive(thiserror::Error, Debug)]
pub enum StaticEvaluatorError {
    #[error(
        r"preimage and image must be nonempty and have the same number of rows
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
        batch_size: NonZeroUsize,
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

    use crate::ops::gpu::Function;
    use crate::tree::{NAryFunction, Node};

    pub const OPKIND_VARIABLE: usize = 0;
    pub const OPKIND_CONSTANT: usize = 1;
    pub const OPKIND_FUNCTION: usize = 2;

    #[derive(crevice::std140::AsStd140)]
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
        batch_size: NonZeroUsize,
        permutations: NonZeroUsize,
        max_tree_size: NonZeroUsize,
        constant_pool_size: NonZeroUsize,
        preimage: DMatrix<f32>,
        image: Box<[f32]>,
    ) -> Result<Self, StaticEvaluatorError> {
        if preimage.is_empty() || image.is_empty() || preimage.nrows() != image.len() {
            return Err(StaticEvaluatorError::Dataset {
                preimage_rows: preimage.nrows(),
                image_rows: image.len(),
            });
        }

        if batch_size.get() > preimage.nrows() {
            return Err(StaticEvaluatorError::BatchSize {
                batch_size,
                preimage_rows: preimage.nrows(),
            });
        }

        let (function_names, shader, wg_size) = Self::generate_shader(
            functions,
            batch_size.get(),
            permutations.get(),
            max_tree_size.get(),
        )?;
        assert_eq!(
            wg_size,
            [batch_size.get() as u32, permutations.get() as u32, 1]
        );
        let functions = function_names.keys().map(|fun| (fun.id(), *fun)).collect();

        let wgpu_state = WgpuState::new(
            &preimage,
            &image,
            batch_size,
            permutations,
            max_tree_size,
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

    #[allow(clippy::type_complexity)]
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

        println!("{shader_source}");

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
        batch_size: NonZeroUsize,
        permutations: NonZeroUsize,
        max_tree_size: NonZeroUsize,
        constant_pool_size: NonZeroUsize,
        shader: naga::Module,
    ) -> Result<Self, StaticEvaluatorError> {
        assert!(!preimage.is_empty());
        assert!(!image.is_empty());
        assert_eq!(preimage.nrows(), image.len());

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

        let workgroup_size = [batch_size.get() as u32, permutations.get() as u32, 0];
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Naga(Cow::Owned(shader)),
        });

        let preimage_buffer_size =
            (preimage.len() * size_of::<f32>()) as wgpu::BufferAddress;
        let image_buffer_size = std::mem::size_of_val(image) as wgpu::BufferAddress;
        let constant_pool_buffer_size =
            (constant_pool_size.get() * size_of::<f32>()) as wgpu::BufferAddress;
        let batch_buffer_size =
            (batch_size.get() * size_of::<u32>()) as wgpu::BufferAddress;
        let permutation_buffer_size =
            (permutations.get() * max_tree_size.get() * size_of::<f32>())
                as wgpu::BufferAddress;
        let tree_buffer_size = (max_tree_size.get() * size_of::<rpn::Std140TreeNode>())
            as wgpu::BufferAddress;
        let result_buffer_size = (batch_size.get()
            * permutations.get()
            * size_of::<f32>()) as wgpu::BufferAddress;

        let dataset_buffer_size = preimage_buffer_size + image_buffer_size;
        let sampling_buffer_size =
            constant_pool_buffer_size + batch_buffer_size + permutation_buffer_size;
        let dataset_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: dataset_buffer_size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        let sampling_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: sampling_buffer_size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });

        let tree_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: result_buffer_size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });

        let result_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: result_buffer_size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let dataset_storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: dataset_stage.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sampling_storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: sampling_stage.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let tree_storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: tree_stage.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let result_storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: result_stage.size(),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        fn bind_group_layout_entry(
            binding: u32,
            min_size: wgpu::BufferAddress,
            read_only: bool,
        ) -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        (min_size as wgpu::BufferAddress).try_into().unwrap(),
                    ),
                },
                count: None,
            }
        }

        let dataset_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    bind_group_layout_entry(0, preimage_buffer_size, true),
                    bind_group_layout_entry(1, preimage_buffer_size, true),
                ],
            });
        let sampling_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    bind_group_layout_entry(0, constant_pool_buffer_size, true),
                    bind_group_layout_entry(1, batch_buffer_size, true),
                    bind_group_layout_entry(2, permutation_buffer_size, true),
                ],
            });
        let eval_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    bind_group_layout_entry(0, tree_buffer_size, true),
                    bind_group_layout_entry(1, result_buffer_size, false),
                ],
            });

        let dataset_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &dataset_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &dataset_storage,
                        offset: 0,
                        size: Some(preimage_buffer_size.try_into().unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &dataset_storage,
                        offset: preimage_buffer_size,
                        size: Some(image_buffer_size.try_into().unwrap()),
                    }),
                },
            ],
        });

        let sampling_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &sampling_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &sampling_storage,
                        offset: 0,
                        size: Some(constant_pool_buffer_size.try_into().unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &sampling_storage,
                        offset: constant_pool_buffer_size,
                        size: Some(batch_buffer_size.try_into().unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &sampling_storage,
                        offset: batch_buffer_size,
                        size: Some(batch_buffer_size.try_into().unwrap()),
                    }),
                },
            ],
        });

        let eval_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &eval_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tree_storage.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: result_storage.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &dataset_bind_group_layout,
                    &sampling_bind_group_layout,
                    &eval_bind_group_layout,
                ],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: PushConstants::range(),
                }],
            });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "",
            compilation_options: Default::default(),
        });

        let wgpu_state = Self {
            device,
            queue,
            dataset_stage,
            sampling_stage,
            tree_stage,
            result_stage,
            dataset_storage,
            sampling_storage,
            tree_storage,
            result_storage,
            dataset_bind_group,
            sampling_bind_group,
            eval_bind_group,
            pipeline,
            workgroup_size,
        };

        wgpu_state.upload_dataset(preimage, image);

        Ok(wgpu_state)
    }

    fn upload_dataset(&self, preimage: &DMatrix<f32>, image: &[f32]) {
        let row_major_preimage = preimage.transpose().iter().copied().collect_vec();

        let preimage_size = mem::size_of_val(preimage) as wgpu::BufferAddress;

        self.write_buffer(&self.dataset_stage, 0, &row_major_preimage);
        self.write_buffer(&self.dataset_stage, preimage_size, &image);
        self.copy_buffer(
            &self.dataset_stage,
            &self.dataset_storage,
            0,
            0,
            self.dataset_stage.size(),
        );
    }

    fn write_buffer<T: bytemuck::NoUninit>(
        &self,
        stage: &wgpu::Buffer,
        offset: wgpu::BufferAddress,
        data: &[T],
    ) {
        let data_size = mem::size_of_val(data) as wgpu::BufferAddress;
        assert!(stage.size() >= data_size + offset);

        let data_bytes = data.iter().flat_map(bytemuck::bytes_of).copied();
        let stage_slice = stage.slice(offset..offset + data_size);
        let mut mapping = Self::map_buffer_mut(stage_slice);
        for (src, dst) in data_bytes.zip_eq(mapping.iter_mut()) {
            *dst = src;
        }

        drop(mapping);
    }

    fn read_buffer<T: bytemuck::AnyBitPattern>(
        &self,
        stage: &wgpu::Buffer,
        offset: wgpu::BufferAddress,
        count: usize,
        destination: &mut Vec<T>,
    ) {
        let data_size = (size_of::<T>() * count) as wgpu::BufferAddress;
        assert!(stage.size() >= offset + data_size);

        let stage_slice = stage.slice(offset..offset + data_size);
        let mapping = Self::map_buffer(stage_slice);

        destination.extend(
            mapping
                .chunks_exact(size_of::<T>())
                .map(bytemuck::from_bytes::<T>)
                .cloned(),
        );
    }

    fn copy_buffer(
        &self,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        src_offset: wgpu::BufferAddress,
        dst_offset: wgpu::BufferAddress,
        len: wgpu::BufferAddress,
    ) {
        assert!(self
            .device
            .poll(wgpu::Maintain::wait_for(self.queue.submit([{
                let mut encoder = self.device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(src, src_offset, dst, dst_offset, len);
                encoder.finish()
            }])))
            .is_queue_empty())
    }

    fn map_buffer(slice: wgpu::BufferSlice) -> wgpu::BufferView {
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        rx.recv()
            .expect("could not map buffer slice")
            .expect("could not map buffer slice");
        slice.get_mapped_range()
    }

    fn map_buffer_mut(slice: wgpu::BufferSlice) -> wgpu::BufferViewMut {
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Write, move |r| tx.send(r).unwrap());
        rx.recv()
            .expect("could not map buffer slice")
            .expect("could not map buffer slice");
        slice.get_mapped_range_mut()
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
        let batch_size = 64.try_into().unwrap();
        let permutations = 32.try_into().unwrap();
        let max_tree_size = 96.try_into().unwrap();
        let constant_pool_size = 16.try_into().unwrap();

        StaticEvaluator::new(
            defs,
            batch_size,
            permutations,
            max_tree_size,
            constant_pool_size,
            preimage,
            image,
        )
        .map(|_| ())
    }
}

impl PushConstants {
    fn range() -> Range<u32> {
        0..Self::std430_size_static() as u32
    }
}
