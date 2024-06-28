use glsl_lang::{ast::*, parse::Parsable};
use itertools::Itertools;
use naga::{FastHashMap, FastHashSet};
use petgraph::graph::DiGraph;
use shaderc::CompilationArtifact;

use crate::tree::eval::gpu::statik::{
    rpn, shader::FunctionSanitizeError, StaticEvaluatorError,
};

mod macro_identifiers {
    pub mod opkind {
        pub const VARIABLE: &str = "OPKIND_VARIABLE";
        pub const CONSTANT: &str = "OPKIND_CONSTANT";
        pub const FUNCTION: &str = "OPKIND_FUNCTION";
    }
    pub const BATCH_SIZE: &str = "BATCH_SIZE";
    pub const PERMUTATIONS: &str = "PERMUTATIONS";
    pub const STACK_SIZE: &str = "STACK_SIZE";
}

const SHADER_SOURCE: &str = include_str!(crate::proot!("shaders/src/skeleton.comp"));

#[allow(clippy::type_complexity)]
pub fn generate_shader(
    functions: impl IntoIterator<Item = FunctionDefinition>,
    batch_size: usize,
    permutations: usize,
    stack_size: usize,
) -> Result<
    (
        FastHashMap<crate::ops::gpu::Function, SmolStr>,
        CompilationArtifact,
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
    .collect_vec();

    let sanitized = sanitize_function_definitions(functions)?;

    let evaluation = generate_function_evaluation(
        sanitized.iter().map(|(_def, fun, _name)| fun).cloned(),
        stack_size,
    );

    let function_names = sanitized
        .iter()
        .map(|(_def, fun, name)| (*fun, name.clone()))
        .collect();
    println!("{SHADER_SOURCE}");
    let mut shader_ast = glsl_lang::ast::TranslationUnit::parse(SHADER_SOURCE)
        .unwrap_or_else(|err| panic!("failed to parse shader skeleton: {err}"));
    shader_ast
        .0
        .extend(sanitized.iter().map(|(def, _fun, _name)| {
            ExternalDeclarationData::FunctionDefinition(def.clone()).into_node()
        }));
    shader_ast.0.push(
        ExternalDeclarationData::FunctionDefinition(evaluation.clone()).into_node(),
    );

    sort_shader(&mut shader_ast).map_err(|err| StaticEvaluatorError::Cycle(err))?;
    let shader_source = {
        let mut shader_source = String::new();
        glsl_lang::transpiler::glsl::show_translation_unit(
            &mut shader_source,
            &shader_ast,
            Default::default(),
        )
        .unwrap_or_else(|err| panic!("failed to render shader: {err}"));
        shader_source
    };

    let compiler = shaderc::Compiler::new().ok_or(StaticEvaluatorError::ShaderC(None))?;
    let mut options =
        shaderc::CompileOptions::new().ok_or(StaticEvaluatorError::ShaderC(None))?;
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    options.set_source_language(shaderc::SourceLanguage::GLSL);
    // options.set_generate_debug_info();
    options.set_target_spirv(shaderc::SpirvVersion::V1_6);
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_3 as u32,
    );

    for (symbol, definition) in defines {
        options.add_macro_definition(&symbol, Some(&definition))
    }

    let artifact = compiler.compile_into_spirv(
        &shader_source,
        shaderc::ShaderKind::Compute,
        "processed_skeleton.comp",
        "main",
        Some(&options),
    )?;

    let text = compiler.compile_into_spirv_assembly(
        &shader_source,
        shaderc::ShaderKind::Compute,
        "processed_skeleton.comp",
        "main",
        Some(&options),
    )?;

    println!("{shader_source}");
    // println!("{}", text.as_text());

    // let mut binary = artifact.as_binary().iter().copied();
    // let magic_number = binary.next().unwrap();
    // let version = binary.next().unwrap();
    // let generator_magic_number = binary.next().unwrap();
    // let bound = binary.next().unwrap();
    // binary.next();
    // let mut instructions = Vec::new();
    // while let Some(inst) = binary.next() {
    //     let op = inst as u16;
    //     let wc = (inst >> 16) as u16;
    //     let operands = binary.by_ref().take(wc as usize - 1).collect_vec();
    //     instructions.push((op, wc, operands));
    // }
    // println!("---------------------");
    // println!("magic number:           {magic_number}");
    // println!("version:                {version}");
    // println!("generator magic number: {generator_magic_number}");
    // println!("bound:                  {bound}");
    // println!(
    //     "{}",
    //     instructions
    //         .into_iter()
    //         .map(|(op, wc, operands)| format!(
    //             "OP: {:04x}, WC: {:04x}; Operands: {}",
    //             op,
    //             wc,
    //             operands.iter().map(|op| format!("{:08x}", op)).join(", ")
    //         ))
    //         .join("\n")
    // );

    Ok((function_names, artifact))
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
            | FunctionParameterDeclarationData::Unnamed(qualifier, ty) => (qualifier, ty),
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

            for (position, parameter) in function.prototype.parameters.iter().enumerate()
            {
                validate_parameter_type(position, parameter)?;
            }
            let arity = function.prototype.parameters.len();
            let op = crate::ops::gpu::Function::new(id, arity);
            let mut ident = generate_function_name(id);
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

    let switch = StatementData::Switch(
        SwitchStatementData {
            head: Box::new(ExprData::variable(id_param_ident).into_node()),
            body: functions
                .into_iter()
                .flat_map(|function| {
                    let id = function.id;
                    let arity = function.arity as u32;
                    let ident = IdentifierData(generate_function_name(id)).into_node();

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
        JumpStatementData::Return(Some(Box::new(ExprData::FloatConst(0.0).into_node())))
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

/// Topoligically sort declared functions.
/// Overloading is not supported.
/// Functions will be placed after all other declarations.
fn sort_shader(shader: &mut TranslationUnit) -> Result<(), SmolStr> {
    // we only sort function definitions, no other declarations
    let mut non_fn_decls = Vec::with_capacity(shader.0.len());
    let mut fn_decls =
        FastHashMap::with_capacity_and_hasher(shader.0.len(), Default::default());
    for decl in std::mem::take(&mut shader.0) {
        if let ExternalDeclarationData::FunctionDefinition(def) = decl.content {
            fn_decls.insert(def.prototype.name.0.clone(), def);
        } else {
            non_fn_decls.push(decl);
        }
    }

    // all the functions declared in the shader
    let fns = fn_decls.keys().unique().cloned().collect_vec();

    // we assign an index to each declared function
    let fn_indices = fns
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, name)| (name, idx))
        .collect::<FastHashMap<_, _>>();

    let mut dependencies = fn_decls
        .iter()
        .map(|(name, def)| {
            let name = name.clone();
            let deps = find_function_dependencies(def);
            (name, deps)
        })
        .collect::<FastHashMap<_, _>>();

    // we can only sort function definitions within our shader,
    // therefore we discard external dependencies
    for deps in dependencies.values_mut() {
        deps.retain(|dep| fns.contains(dep));
    }

    let edges = dependencies
        .into_iter()
        .flat_map(|(name, deps)| {
            deps.into_iter()
                // `name` and `dep` are taken from `dependencies`,
                // which is constructed from fn_decls.
                // `fn_indices` is losslessly derived from `fns`,
                // which is losslessly derived from `fn_decls`.
                // Therefore, `fn_indices` should contain all `name`s and `dep`s.
                .map(|dep| (fn_indices[&name], fn_indices[&dep]))
                .collect_vec()
        })
        .collect_vec();

    let dep_graph = DiGraph::<usize, (usize, usize), _>::from_edges(edges);
    let ordering = petgraph::algo::toposort(&dep_graph, None)
        .map_err(|cycle| fns[cycle.node_id().index()].clone())?;

    let ordered_fn_decls = ordering.iter().map(|id| {
        fn_decls
            .remove(&fns[id.index()])
            .expect("function ordering should contain no duplicates")
    });

    let mut decls = non_fn_decls;
    decls.extend(
        ordered_fn_decls.map(|fn_decl| {
            Node::from(ExternalDeclarationData::FunctionDefinition(fn_decl))
        }),
    );

    shader.0 = decls;

    Ok(())
}

fn find_stmt_deps(stmt: &Statement, deps: &mut FastHashSet<SmolStr>) {
    match &stmt.content {
        | StatementData::Declaration(_) => {}
        | StatementData::Expression(expr) => {
            if let Some(expr) = &expr.content.0.as_ref() {
                find_expr_deps(expr, deps);
            }
        }
        | StatementData::Selection(selection) => {
            find_expr_deps(&selection.cond, deps);
            match &selection.rest.content {
                | SelectionRestStatementData::Statement(expr) => {
                    find_stmt_deps(expr, deps)
                }
                | SelectionRestStatementData::Else(iph, els) => {
                    find_stmt_deps(iph, deps);
                    find_stmt_deps(els, deps);
                }
            }
        }
        | StatementData::Switch(switch) => {
            find_expr_deps(&switch.head, deps);
            for stmt in &switch.body {
                find_stmt_deps(stmt, deps);
            }
        }
        | StatementData::CaseLabel(case_label) => {
            if let CaseLabelData::Case(expr) = &case_label.content {
                find_expr_deps(expr, deps);
            }
        }
        | StatementData::Iteration(iteration) => match &iteration.content {
            | IterationStatementData::While(condition, stmt) => {
                match &condition.content {
                    | ConditionData::Expr(expr) => find_expr_deps(expr, deps),
                    | ConditionData::Assignment(_, _, initializer) => {
                        find_initializer_deps(initializer, deps);
                    }
                }
                find_stmt_deps(stmt, deps);
            }
            | IterationStatementData::DoWhile(stmt, expr) => {
                find_stmt_deps(stmt, deps);
                find_expr_deps(expr, deps);
            }
            | IterationStatementData::For(init, rest, stmt) => {
                match &init.content {
                    | ForInitStatementData::Expression(Some(expr)) => {
                        find_expr_deps(expr, deps)
                    }
                    | ForInitStatementData::Expression(None) => {}
                    | ForInitStatementData::Declaration(decl) => {
                        find_declaration_deps(decl, deps)
                    }
                }
                if let Some(condition) = rest.condition.as_ref() {
                    match &condition.content {
                        | ConditionData::Expr(expr) => find_expr_deps(expr, deps),
                        | ConditionData::Assignment(_, _, initializer) => {
                            find_initializer_deps(initializer, deps)
                        }
                    }
                }
                if let Some(expr) = rest.post_expr.as_ref() {
                    find_expr_deps(expr, deps);
                }
                find_stmt_deps(stmt, deps);
            }
        },
        | StatementData::Jump(jump) => match &jump.content {
            | JumpStatementData::Continue => {}
            | JumpStatementData::Break => {}
            | JumpStatementData::Return(Some(expr)) => find_expr_deps(expr, deps),
            | JumpStatementData::Return(None) => {}
            | JumpStatementData::Discard => {}
        },
        | StatementData::Compound(compound) => {
            for stmt in &compound.statement_list {
                find_stmt_deps(stmt, deps);
            }
        }
    }
}

fn find_expr_deps(expr: &Expr, deps: &mut FastHashSet<SmolStr>) {
    match &expr.content {
        | ExprData::Variable(_ident) => {}
        | ExprData::IntConst(_value) => {}
        | ExprData::UIntConst(_value) => {}
        | ExprData::BoolConst(_value) => {}
        | ExprData::FloatConst(_value) => {}
        | ExprData::DoubleConst(_value) => {}
        | ExprData::Unary(_op, expr) => find_expr_deps(expr, deps),
        | ExprData::Binary(_op, lhs, rhs) => {
            find_expr_deps(lhs, deps);
            find_expr_deps(rhs, deps);
        }
        | ExprData::Ternary(cond, iph, els) => {
            find_expr_deps(cond, deps);
            find_expr_deps(iph, deps);
            find_expr_deps(els, deps);
        }
        | ExprData::Assignment(lhs, _op, rhs) => {
            find_expr_deps(lhs, deps);
            find_expr_deps(rhs, deps);
        }
        | ExprData::Bracket(lhs, rhs) => {
            find_expr_deps(lhs, deps);
            find_expr_deps(rhs, deps);
        }
        | ExprData::FunCall(ident, exprs) => {
            if let FunIdentifierData::Expr(expr) = &ident.content {
                if let ExprData::Variable(ident) = &expr.content {
                    deps.insert(ident.0.clone());
                }
            }
            for expr in exprs {
                find_expr_deps(expr, deps);
            }
        }
        | ExprData::Dot(expr, _field_ident) => find_expr_deps(expr, deps),
        | ExprData::PostInc(expr) => find_expr_deps(expr, deps),
        | ExprData::PostDec(expr) => find_expr_deps(expr, deps),
        | ExprData::Comma(lhs, rhs) => {
            find_expr_deps(lhs, deps);
            find_expr_deps(rhs, deps);
        }
    }
}

fn find_initializer_deps(initializer: &Initializer, deps: &mut FastHashSet<SmolStr>) {
    match &initializer.content {
        | InitializerData::Simple(expr) => find_expr_deps(expr, deps),
        | InitializerData::List(initializers) => {
            for initializer in initializers {
                find_initializer_deps(initializer, deps);
            }
        }
    }
}

fn find_declaration_deps(declaration: &Declaration, deps: &mut FastHashSet<SmolStr>) {
    match &declaration.content {
        | DeclarationData::FunctionPrototype(_prototype) => {}
        | DeclarationData::InitDeclaratorList(declarators) => {
            if let Some(initializer) = declarators.head.initializer.as_ref() {
                find_initializer_deps(initializer, deps);
            }
            for declaration in &declarators.tail {
                if let Some(initializer) = declaration.initializer.as_ref() {
                    find_initializer_deps(initializer, deps);
                }
            }
        }
        | DeclarationData::Precision(_precision_qualifier, _type_specifier) => {}
        | DeclarationData::Block(_block) => {}
        | DeclarationData::Invariant(_ident) => {}
    }
}

fn find_function_dependencies(def: &FunctionDefinition) -> FastHashSet<SmolStr> {
    let mut deps = FastHashSet::default();
    for stmt in &def.statement.statement_list {
        find_stmt_deps(stmt, &mut deps)
    }
    deps
}
