use crevice::std430::AsStd430;
use itertools::Itertools;
use nalgebra::DMatrix;
use std::borrow::Cow;
use std::mem::size_of;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::mpsc;
use wgpu::naga::FastHashMap;

use crate::tree::Tree;
use shader::generate_shader;
use shader::FunctionSanitizeError;

mod shader;

pub struct StaticEvaluator {
    functions: FastHashMap<usize, crate::ops::gpu::Function>,
    function_names: FastHashMap<crate::ops::gpu::Function, String>,
    wgpu_state: WgpuState,
    batch_size: NonZeroUsize,
    permutations: NonZeroUsize,
    max_tree_size: NonZeroUsize,
    max_constant_pool_size: NonZeroUsize,
    preimage: DMatrix<f32>,
    image: Box<[f32]>,
    evaluation: Evaluation,
}

#[derive(Debug, Clone, Default)]
pub struct Evaluation {
    tree: Option<crate::tree::Node<f32, crate::ops::gpu::Function>>,
    batch: Option<Vec<usize>>,
    constants: Option<Vec<f32>>,
    permutations: Option<DMatrix<bool>>,
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
    preimage_range: Range<wgpu::BufferAddress>,
    image_range: Range<wgpu::BufferAddress>,
    constant_pool_range: Range<wgpu::BufferAddress>,
    batch_range: Range<wgpu::BufferAddress>,
    permutation_range: Range<wgpu::BufferAddress>,
    dataset_bind_group: wgpu::BindGroup,
    sampling_bind_group: wgpu::BindGroup,
    eval_bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    workgroup_size: [u32; 3],
    tree: Vec<rpn::Std140TreeNode>,
    batch: Vec<u32>,
    constant_pool: Vec<f32>,
    prefactors: Vec<f32>,
    msd: Vec<f32>,
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
        "preimage and image must be nonempty and have the same number of rows \
        (preimage has {preimage_rows}, image has {image_rows})"
    )]
    Dataset {
        preimage_rows: usize,
        image_rows: usize,
    },
    #[error("batch size too large (maximum: {max}, actual: {actual})")]
    BatchSize { max: usize, actual: usize },
    #[error("shader contains cycle")]
    Cycle(String),
    #[error("shaderc error")]
    ShaderC(#[from] Option<shaderc::Error>),
    #[error("naga error: {0}")]
    Naga(#[from] naga::front::glsl::ParseError),
    #[error("failed to sanitize functions: {0}")]
    FunctionSanitize(#[from] FunctionSanitizeError),
    #[error("WGPU error: {0}")]
    Wgpu(String),
    #[error("tree, batch, constants and permutations must be initialised")]
    NotInitialised,
    #[error("tree is too large (maxim size: {max_size}, actual size: {actual_size})")]
    TreeTooLarge { max_size: usize, actual_size: usize },
    #[error(
        "constant pool is too large (max size: {max_size}, actual size: {actual_size})"
    )]
    ConstantPoolTooLarge { max_size: usize, actual_size: usize },
    #[error("batch index {bad_index} at position {position} out of dataset range ({dataset_size} rows)")]
    BadBatch {
        bad_index: usize,
        position: usize,
        dataset_size: usize,
    },
    #[error(
        "permutation matrix has illegal dimensions (\
        expected rows: {expected_rows}, \
        actual rows: {actual_rows}; \
        minimum columns: {min_columns}, \
        maximum columns: {max_columns}, \
        actual columns: {actual_columns})"
    )]
    BadPermutationDimensions {
        actual_rows: usize,
        actual_columns: usize,
        expected_rows: usize,
        min_columns: usize,
        max_columns: usize,
    },
    #[error("function {id} has wrong arity (expected: {expected}, actual: {actual})")]
    BadArity {
        id: usize,
        expected: usize,
        actual: usize,
    },
    #[error("unknown function (id: {0})")]
    UnknownFunction(usize),
    #[error("unknown constant (idx: {idx}, pool size: {pool_size})")]
    UnknownConstant { idx: usize, pool_size: usize },
    #[error("unknown variable (idx: {idx}, sample dimensions: {dimensions})")]
    UnknownVariable { idx: usize, dimensions: usize },
}

mod rpn {
    use crate::ops::gpu::Function;
    use crate::tree::{NAryFunction, Node};

    pub const OPKIND_VARIABLE: usize = 0;
    pub const OPKIND_CONSTANT: usize = 1;
    pub const OPKIND_FUNCTION: usize = 2;

    #[derive(crevice::std140::AsStd140)]
    pub struct TreeNode {
        pub(super) opkind: u32,
        pub(super) immediate: u32,
        pub(super) function_id: u32,
    }

    impl TreeNode {
        pub fn traverse(root: &Node<f32, Function>, mut stack: Vec<Self>) -> Vec<Self> {
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
                        stack = Self::traverse(operand, stack);
                    }
                }
            }
            stack
        }
    }
}

#[allow(unused)]
impl StaticEvaluator {
    pub fn new(
        functions: impl IntoIterator<Item = glsl::syntax::FunctionDefinition>,
        batch_size: NonZeroUsize,
        permutations: NonZeroUsize,
        max_tree_size: NonZeroUsize,
        max_constant_pool_size: NonZeroUsize,
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
                actual: batch_size.get(),
                max: preimage.nrows(),
            });
        }

        let max_workgroup_size = NonZeroUsize::try_from(
            wgpu::Limits::default().max_compute_invocations_per_workgroup as usize,
        )
        .expect("max work group size must be greater than zero");
        let workgroup_size_x = batch_size.min(max_workgroup_size).get() as u32;
        let workgroup_size_y = (permutations.get() as u32)
            .min(max_workgroup_size.get() as u32 / workgroup_size_x);
        let workgroup_size = [workgroup_size_x, workgroup_size_y, 1];

        let (function_names, naga_module) = generate_shader(
            functions,
            batch_size.get(),
            permutations.get(),
            max_tree_size.get(),
            workgroup_size,
        )?;

        let functions = function_names.keys().map(|fun| (fun.id(), *fun)).collect();

        let wgpu_state = WgpuState::new(
            &preimage,
            &image,
            batch_size,
            permutations,
            max_tree_size,
            max_constant_pool_size,
            naga_module,
            workgroup_size,
        )?;

        Ok(Self {
            functions,
            function_names,
            wgpu_state,
            batch_size,
            permutations,
            max_tree_size,
            max_constant_pool_size,
            preimage,
            image,
            evaluation: Default::default(),
        })
    }

    pub fn evaluate(
        &mut self,
        evaluation: Evaluation,
    ) -> Result<&[f32], StaticEvaluatorError> {
        if cfg!(debug_assertions)
            && (self.evaluation.tree.is_none() && evaluation.tree.is_none()
                || self.evaluation.batch.is_none() && evaluation.batch.is_none()
                || self.evaluation.constants.is_none() && evaluation.constants.is_none()
                || self.evaluation.permutations.is_none()
                    && evaluation.permutations.is_none())
        {
            return Err(StaticEvaluatorError::NotInitialised);
        }

        let rpn = if let Some(tree) = evaluation.tree.as_ref() {
            Some(if cfg!(debug_assertions) {
                self.validate_tree(
                    tree,
                    evaluation
                        .constants
                        .as_ref()
                        .or(self.evaluation.constants.as_ref())
                        .expect("constants must be initialised")
                        .len(),
                )?
            } else {
                rpn::TreeNode::traverse(tree, Vec::with_capacity(tree.size()))
            })
        } else {
            None
        };

        if let Some(constants) = evaluation.constants.as_ref() {
            if cfg!(debug_assertions) {
                self.validate_constants(constants);
            }
        }

        if let Some(batch) = evaluation.batch.as_ref() {
            if cfg!(debug_assertions) {
                self.validate_batch(batch);
            }
        }

        if let Some(permutations) = evaluation.permutations.as_ref() {
            if cfg!(debug_assertions) {
                self.validate_permutations(permutations);
            }
        }

        if let Some(tree) = evaluation.tree {
            let rpn = rpn.unwrap();
            self.wgpu_state.upload_tree(&rpn);
            self.evaluation.tree.replace(tree);
        }

        if let Some(batch) = evaluation.batch {
            self.wgpu_state.upload_batch(&batch);
            self.evaluation.batch.replace(batch);
        }

        if let Some(constants) = evaluation.constants {
            self.wgpu_state.upload_constants(&constants);
            self.evaluation.constants.replace(constants);
        }

        if let Some(permutations) = evaluation.permutations {
            self.wgpu_state.upload_permutations(&permutations);
            self.evaluation.permutations.replace(permutations);
        }

        self.wgpu_state.run_evaluation();

        let msd = self.wgpu_state.download_msd();

        Ok(msd)
    }

    fn validate_tree(
        &self,
        root: &crate::tree::Node<f32, crate::ops::gpu::Function>,
        constant_pool_size: usize,
    ) -> Result<Vec<rpn::TreeNode>, StaticEvaluatorError> {
        let tree_size = root.size();
        if tree_size > self.max_tree_size.get() {
            return Err(StaticEvaluatorError::TreeTooLarge {
                max_size: self.max_tree_size.get(),
                actual_size: tree_size,
            });
        }

        let rpn = rpn::TreeNode::traverse(root, Vec::with_capacity(tree_size));

        for &rpn::TreeNode {
            opkind,
            immediate,
            function_id,
        } in &rpn
        {
            match opkind as usize {
                | rpn::OPKIND_VARIABLE => {
                    let id = immediate as usize;
                    if !(0..self.preimage.ncols()).contains(&id) {
                        return Err(StaticEvaluatorError::UnknownVariable {
                            idx: id,
                            dimensions: self.preimage.ncols(),
                        });
                    }
                }
                | rpn::OPKIND_CONSTANT => {
                    let id = immediate as usize;
                    if !(0..constant_pool_size).contains(&id) {
                        return Err(StaticEvaluatorError::UnknownConstant {
                            idx: id,
                            pool_size: constant_pool_size,
                        });
                    }
                }
                | rpn::OPKIND_FUNCTION => {
                    let function_id = function_id as usize;
                    let Some(function) = self.functions.get(&function_id) else {
                        return Err(StaticEvaluatorError::UnknownFunction(function_id));
                    };
                    let arity = immediate as usize;
                    let expected_arity = function.arity;
                    if expected_arity != immediate as usize {
                        return Err(StaticEvaluatorError::BadArity {
                            id: function_id,
                            expected: expected_arity,
                            actual: arity,
                        });
                    }
                }

                | opkind => panic!("illegal opkind {opkind}"),
            }
        }

        Ok(rpn)
    }

    fn validate_batch(&self, batch: &[usize]) -> Result<(), StaticEvaluatorError> {
        if batch.len() > self.batch_size.get() {
            return Err(StaticEvaluatorError::BatchSize {
                actual: batch.len(),
                max: self.batch_size.get(),
            });
        }

        if let Some((pos, idx)) = batch
            .iter()
            .enumerate()
            .find(|(pos, idx)| **idx > self.preimage.nrows())
        {
            return Err(StaticEvaluatorError::BadBatch {
                bad_index: *idx,
                position: pos,
                dataset_size: self.preimage.nrows(),
            });
        }

        Ok(())
    }

    fn validate_constants(&self, constants: &[f32]) -> Result<(), StaticEvaluatorError> {
        if constants.len() > self.max_constant_pool_size.get() {
            return Err(StaticEvaluatorError::ConstantPoolTooLarge {
                max_size: self.max_constant_pool_size.get(),
                actual_size: constants.len(),
            });
        }
        Ok(())
    }

    fn validate_permutations(
        &self,
        permutations: &DMatrix<bool>,
    ) -> Result<(), StaticEvaluatorError> {
        let tree_size = self.wgpu_state.tree.len();
        if permutations.nrows() != self.permutations.get()
            || !(tree_size..=self.max_tree_size.get()).contains(&permutations.ncols())
        {
            return Err(StaticEvaluatorError::BadPermutationDimensions {
                actual_rows: permutations.nrows(),
                actual_columns: permutations.ncols(),
                expected_rows: self.permutations.get(),
                min_columns: tree_size,
                max_columns: self.max_tree_size.get(),
            });
        }
        Ok(())
    }
}

impl Evaluation {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn tree(self, tree: crate::tree::Node<f32, crate::ops::gpu::Function>) -> Self {
        Self {
            tree: Some(tree),
            ..self
        }
    }

    pub fn batch(self, batch: Vec<usize>) -> Self {
        Self {
            batch: Some(batch),
            ..self
        }
    }

    pub fn constants(self, constants: Vec<f32>) -> Self {
        Self {
            constants: Some(constants),
            ..self
        }
    }

    pub fn permutations(self, permutations: DMatrix<bool>) -> Self {
        Self {
            permutations: Some(permutations),
            ..self
        }
    }

    pub fn or(self, other: Self) -> Self {
        Self {
            tree: self.tree.or(other.tree),
            batch: self.batch.or(other.batch),
            constants: self.constants.or(other.constants),
            permutations: self.permutations.or(other.permutations),
        }
    }
}

impl WgpuState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        preimage: &DMatrix<f32>,
        image: &[f32],
        batch_size: NonZeroUsize,
        permutations: NonZeroUsize,
        max_tree_size: NonZeroUsize,
        max_constant_pool_size: NonZeroUsize,
        naga_module: naga::Module,
        workgroup_size: [u32; 3],
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

        let limits = wgpu::Limits {
            max_push_constant_size: 64,
            max_compute_invocations_per_workgroup: workgroup_size.iter().product(),
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::PUSH_CONSTANTS
                    | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
                required_limits: limits.clone(),
            },
            None,
        ))
        .map_err(|err| {
            StaticEvaluatorError::Wgpu(format!("device request failed: {err}"))
        })?;

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Naga(Cow::Owned(naga_module)),
        });

        let preimage_buffer_size =
            (preimage.len() * size_of::<f32>()) as wgpu::BufferAddress;
        let image_buffer_size = std::mem::size_of_val(image) as wgpu::BufferAddress;
        let constant_pool_buffer_size =
            (max_constant_pool_size.get() * size_of::<f32>()) as wgpu::BufferAddress;
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

        let alignment = limits.min_storage_buffer_offset_alignment as wgpu::BufferAddress;

        let preimage_offset = align(0, alignment);
        let image_offset = align(preimage_offset + preimage_buffer_size, alignment);

        let constant_pool_offset = align(0, alignment);
        let batch_offset =
            align(constant_pool_offset + constant_pool_buffer_size, alignment);
        let permutation_offset = align(batch_offset + batch_buffer_size, alignment);

        let preimage_range = preimage_offset..preimage_buffer_size;
        let image_range = image_offset..image_buffer_size;

        let constant_pool_range = constant_pool_offset..constant_pool_buffer_size;
        let batch_range = batch_offset..batch_buffer_size;
        let permutation_range = permutation_offset..permutation_buffer_size;

        let dataset_buffer_size =
            (image_range.start..preimage_range.end).try_len().unwrap();
        let sampling_buffer_size = (constant_pool_range.start..permutation_range.end)
            .try_len()
            .unwrap();

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
                    bind_group_layout_entry(1, image_buffer_size, true),
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
                        offset: preimage_offset,
                        size: Some(preimage_buffer_size.try_into().unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &dataset_storage,
                        offset: image_offset,
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
                        offset: constant_pool_offset,
                        size: Some(constant_pool_buffer_size.try_into().unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &sampling_storage,
                        offset: batch_offset,
                        size: Some(batch_buffer_size.try_into().unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &sampling_storage,
                        offset: permutation_offset,
                        size: Some(permutation_buffer_size.try_into().unwrap()),
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
            entry_point: "main",
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
            preimage_range,
            image_range,
            constant_pool_range,
            batch_range,
            permutation_range,
            dataset_bind_group,
            sampling_bind_group,
            eval_bind_group,
            pipeline,
            workgroup_size,
            tree: Vec::with_capacity(max_tree_size.get()),
            batch: Vec::with_capacity(batch_size.get()),
            constant_pool: Vec::with_capacity(max_constant_pool_size.get()),
            prefactors: Vec::with_capacity(permutations.get() * max_tree_size.get()),
            msd: Vec::with_capacity(permutations.get()),
        };

        wgpu_state.upload_dataset(preimage, image);

        Ok(wgpu_state)
    }

    pub fn upload_tree(&mut self, tree: &[rpn::TreeNode]) {
        todo!()
    }

    pub fn upload_batch(&mut self, batch: &[usize]) {
        todo!()
    }

    pub fn upload_constants(&mut self, constants: &[f32]) {
        todo!()
    }

    pub fn upload_permutations(&mut self, permutations: &DMatrix<bool>) {
        todo!()
    }

    pub fn run_evaluation(&self) {
        todo!()
    }

    pub fn download_msd(&mut self) -> &[f32] {
        todo!()
    }

    fn upload_dataset(&self, preimage: &DMatrix<f32>, image: &[f32]) {
        let row_major_preimage = preimage.transpose().iter().copied().collect_vec();

        let preimage_size = std::mem::size_of_val(preimage) as wgpu::BufferAddress;

        self.write_buffer(&self.dataset_stage, 0, &row_major_preimage, false);
        self.write_buffer(&self.dataset_stage, preimage_size, image, false);
        self.dataset_stage.unmap();
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
        map: bool,
    ) {
        let data_size = std::mem::size_of_val(data) as wgpu::BufferAddress;
        assert!(stage.size() >= data_size + offset);

        let data_bytes = data.iter().flat_map(bytemuck::bytes_of).copied();
        let stage_slice = stage.slice(offset..offset + data_size);
        let mut mapping = if map {
            Self::map_buffer_mut(&self.device, stage_slice)
        } else {
            stage_slice.get_mapped_range_mut()
        };
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
        map: bool,
    ) {
        let data_size = (size_of::<T>() * count) as wgpu::BufferAddress;
        assert!(stage.size() >= offset + data_size);

        let stage_slice = stage.slice(offset..offset + data_size);
        let mapping = if map {
            Self::map_buffer(&self.device, stage_slice)
        } else {
            stage_slice.get_mapped_range()
        };

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

    fn map_buffer<'a>(
        device: &wgpu::Device,
        slice: wgpu::BufferSlice<'a>,
    ) -> wgpu::BufferView<'a> {
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("could not map buffer slice")
            .expect("could not map buffer slice");
        slice.get_mapped_range()
    }

    fn map_buffer_mut<'a>(
        device: &wgpu::Device,
        slice: wgpu::BufferSlice<'a>,
    ) -> wgpu::BufferViewMut<'a> {
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Write, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("could not map buffer slice")
            .expect("could not map buffer slice");
        slice.get_mapped_range_mut()
    }
}

impl PushConstants {
    fn range() -> Range<u32> {
        0..Self::std430_size_static() as u32
    }
}

const fn align(
    offset: wgpu::BufferAddress,
    alignment: wgpu::BufferAddress,
) -> wgpu::BufferAddress {
    offset + (alignment - (offset % alignment)) % alignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use glsl::parser::Parse;
    use glsl::syntax::FunctionDefinition;
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
            .map(str::trim)
            .map(FunctionDefinition::parse)
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
        let max_constant_pool_size = 16.try_into().unwrap();

        StaticEvaluator::new(
            defs,
            batch_size,
            permutations,
            max_tree_size,
            max_constant_pool_size,
            preimage,
            image,
        )
        .map(|_| ())
    }

    #[test]
    fn test_align() {
        assert_eq!(12, align(9, 4));
        assert_eq!(16, align(13, 4));
        assert_eq!(12, align(12, 4));
        assert_eq!(13, align(13, 1));
        assert_eq!(8, align(8, 8));
        assert_eq!(32, align(16, 32));
        assert_eq!(512, align(16, 512));
    }
}
