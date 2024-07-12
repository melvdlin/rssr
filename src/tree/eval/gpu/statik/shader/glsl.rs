use glsl::{parser::Parse, syntax::*};
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
        FastHashMap<crate::ops::gpu::Function, String>,
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
    let mut shader_ast = glsl::syntax::TranslationUnit::parse(SHADER_SOURCE)
        .unwrap_or_else(|err| panic!("failed to parse shader skeleton: {err}"));
    shader_ast.0.extend(
        sanitized.iter().map(|(def, _fun, _name)| {
            ExternalDeclaration::FunctionDefinition(def.clone())
        }),
    );
    shader_ast
        .0
        .push(ExternalDeclaration::FunctionDefinition(evaluation.clone()));

    sort_shader(&mut shader_ast).map_err(StaticEvaluatorError::Cycle)?;
    let shader_source = {
        let mut shader_source = String::new();
        glsl::transpiler::glsl::show_translation_unit(&mut shader_source, &shader_ast);
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

    println!("{shader_source}");
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
    functions: impl IntoIterator<Item = FunctionDefinition>,
) -> Result<
    Vec<(FunctionDefinition, crate::ops::gpu::Function, String)>,
    FunctionSanitizeError,
> {
    fn validate_return_type(
        function: &FunctionDefinition,
    ) -> Result<(), FunctionSanitizeError> {
        if function.prototype.ty.qualifier.is_some()
            || function.prototype.ty.ty.array_specifier.is_some()
            || !matches!(function.prototype.ty.ty.ty, TypeSpecifierNonArray::Float)
        {
            return Err(FunctionSanitizeError::InvalidReturnType {
                expected: FullySpecifiedType {
                    qualifier: None,
                    ty: TypeSpecifier {
                        ty: TypeSpecifierNonArray::Float,
                        array_specifier: None,
                    },
                },
                found: function.prototype.ty.clone(),
            });
        }
        Ok(())
    }

    fn validate_parameter_type(
        position: usize,
        parameter: &FunctionParameterDeclaration,
    ) -> Result<(), FunctionSanitizeError> {
        let (qualifier, ty) = match parameter {
            | FunctionParameterDeclaration::Named(qualifier, declarator) => {
                (qualifier, &declarator.ty)
            }
            | FunctionParameterDeclaration::Unnamed(qualifier, ty) => (qualifier, ty),
        };
        if qualifier.is_some()
            || !matches!(
                ty,
                TypeSpecifier {
                    ty: TypeSpecifierNonArray::Float,
                    array_specifier: None
                }
            )
        {
            return Err(FunctionSanitizeError::InvalidParameter {
                parameter: parameter.clone(),
                position,
                expected_type: TypeSpecifier {
                    ty: TypeSpecifierNonArray::Float,
                    array_specifier: None,
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
) -> FunctionDefinition {
    use glsl::syntax::*;

    fn parameter_declaration(
        ident: Identifier,
        ty: TypeSpecifierNonArray,
        array: Option<ArraySpecifier>,
    ) -> FunctionParameterDeclaration {
        FunctionParameterDeclaration::Named(
            None,
            FunctionParameterDeclarator {
                ty: TypeSpecifier {
                    ty,
                    array_specifier: None,
                },
                ident: ArrayedIdentifier {
                    ident,
                    array_spec: array,
                },
            },
        )
    }

    // prototype:
    // `float _function(uint id, uint sp, float stack[STACK_SIZE])`
    // functions to call:
    // `float _functionXYZ(float p1, float p2, ..., float pn)`

    let function_ident = Identifier::from("_function");
    let id_param_ident: Identifier = Identifier::from("id");
    let sp_param_ident: Identifier = Identifier::from("sp");
    let stack_param_ident: Identifier = Identifier::from("stack");
    let id_param =
        parameter_declaration(id_param_ident.clone(), TypeSpecifierNonArray::UInt, None);
    let sp_param =
        parameter_declaration(sp_param_ident.clone(), TypeSpecifierNonArray::UInt, None);

    let stack_param = parameter_declaration(
        stack_param_ident.clone(),
        TypeSpecifierNonArray::Float,
        Some(ArraySpecifier {
            dimensions: NonEmpty(vec![ArraySpecifierDimension::ExplicitlySized(
                Box::new(Expr::UIntConst(stack_size as u32)),
            )]),
        }),
    );

    let prototype = FunctionPrototype {
        ty: FullySpecifiedType {
            qualifier: None,
            ty: TypeSpecifier {
                ty: TypeSpecifierNonArray::Float,
                array_specifier: None,
            },
        },
        name: function_ident,
        parameters: vec![id_param, sp_param, stack_param],
    };

    let switch = Statement::Simple(Box::new(SimpleStatement::Switch(SwitchStatement {
        head: Box::new(Expr::Variable(id_param_ident)),
        body: functions
            .into_iter()
            .flat_map(|function| {
                let id = function.id;
                let arity = function.arity as u32;
                let ident = Identifier(generate_function_name(id));

                let label = Statement::Simple(Box::new(SimpleStatement::CaseLabel(
                    CaseLabel::Case(Box::new(Expr::UIntConst(id as u32))),
                )));

                let ret = Statement::Simple(Box::new(SimpleStatement::Jump(
                    JumpStatement::Return(Some(Box::new(Expr::FunCall(
                        FunIdentifier::Expr(Box::new(Expr::Variable(ident))),
                        (1..=arity)
                            .map(|offset| {
                                Expr::Bracket(
                                    Box::new(Expr::Variable(stack_param_ident.clone())),
                                    ArraySpecifier {
                                        dimensions: NonEmpty(vec![
                                            ArraySpecifierDimension::ExplicitlySized(
                                                Box::new(Expr::Binary(
                                                    BinaryOp::Sub,
                                                    Box::new(Expr::Variable(
                                                        sp_param_ident.clone(),
                                                    )),
                                                    Box::new(Expr::UIntConst(offset)),
                                                )),
                                            ),
                                        ]),
                                    },
                                )
                            })
                            .collect(),
                    )))),
                )));
                [label, ret]
            })
            .collect::<Vec<_>>(),
    })));

    let ret = Statement::Simple(Box::new(SimpleStatement::Jump(JumpStatement::Return(
        Some(Box::new(Expr::FloatConst(0.0))),
    ))));

    FunctionDefinition {
        prototype,
        statement: CompoundStatement {
            statement_list: vec![switch, ret],
        },
    }
}

fn generate_function_name(id: usize) -> String {
    format!("_function{id}")
}

/// Topoligically sort declared functions.
/// Overloading is not supported.
/// Functions will be placed after all other declarations.
fn sort_shader(shader: &mut TranslationUnit) -> Result<(), String> {
    // we only sort function definitions, no other declarations
    let mut non_fn_decls = Vec::with_capacity(shader.0 .0.len());
    let mut fn_decls =
        FastHashMap::with_capacity_and_hasher(shader.0 .0.len(), Default::default());
    for decl in std::mem::take(&mut shader.0 .0) {
        if let ExternalDeclaration::FunctionDefinition(def) = decl {
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
            let deps = find_function_deps(def);
            (name, deps)
        })
        .collect::<FastHashMap<_, _>>();

    // we can only sort function definitions within our shader,
    // therefore, we discard external dependencies
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
    let reverse_ordering = petgraph::algo::toposort(&dep_graph, None)
        .map_err(|cycle| fns[cycle.node_id().index()].clone())?;

    let ordered_fn_decls = reverse_ordering.iter().rev().map(|id| {
        fn_decls
            .remove(&fns[id.index()])
            .expect("function ordering should contain no duplicates")
    });

    let mut decls = non_fn_decls;
    decls.extend(ordered_fn_decls.map(ExternalDeclaration::FunctionDefinition));

    shader.0 .0 = decls;

    Ok(())
}

fn find_stmt_deps(stmt: &Statement, deps: &mut FastHashSet<String>) {
    match stmt {
        | Statement::Simple(simple) => match &**simple {
            | SimpleStatement::Declaration(decl) => {
                find_declaration_deps(decl, deps);
            }
            | SimpleStatement::Expression(expr) => {
                if let Some(expr) = &expr.as_ref() {
                    find_expr_deps(expr, deps);
                }
            }
            | SimpleStatement::Selection(selection) => {
                find_expr_deps(&selection.cond, deps);
                match &selection.rest {
                    | SelectionRestStatement::Statement(expr) => {
                        find_stmt_deps(expr, deps)
                    }
                    | SelectionRestStatement::Else(iph, els) => {
                        find_stmt_deps(iph, deps);
                        find_stmt_deps(els, deps);
                    }
                }
            }
            | SimpleStatement::Switch(switch) => {
                find_expr_deps(&switch.head, deps);
                for stmt in &switch.body {
                    find_stmt_deps(stmt, deps);
                }
            }
            | SimpleStatement::CaseLabel(case_label) => match case_label {
                | CaseLabel::Case(expr) => find_expr_deps(expr, deps),
                | CaseLabel::Def => {}
            },
            | SimpleStatement::Iteration(iteration) => match &iteration {
                | IterationStatement::While(condition, stmt) => {
                    match &condition {
                        | Condition::Expr(expr) => find_expr_deps(expr, deps),
                        | Condition::Assignment(_, _, initializer) => {
                            find_initializer_deps(initializer, deps);
                        }
                    }
                    find_stmt_deps(stmt, deps);
                }
                | IterationStatement::DoWhile(stmt, expr) => {
                    find_stmt_deps(stmt, deps);
                    find_expr_deps(expr, deps);
                }
                | IterationStatement::For(init, rest, stmt) => {
                    match &init {
                        | ForInitStatement::Expression(Some(expr)) => {
                            find_expr_deps(expr, deps)
                        }
                        | ForInitStatement::Expression(None) => {}
                        | ForInitStatement::Declaration(decl) => {
                            find_declaration_deps(decl, deps)
                        }
                    }
                    if let Some(condition) = rest.condition.as_ref() {
                        match &condition {
                            | Condition::Expr(expr) => find_expr_deps(expr, deps),
                            | Condition::Assignment(_, _, initializer) => {
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
            | SimpleStatement::Jump(jump) => match &jump {
                | JumpStatement::Continue => {}
                | JumpStatement::Break => {}
                | JumpStatement::Return(Some(expr)) => find_expr_deps(expr, deps),
                | JumpStatement::Return(None) => {}
                | JumpStatement::Discard => {}
            },
        },
        | Statement::Compound(compound) => {
            for stmt in &compound.statement_list {
                find_stmt_deps(stmt, deps);
            }
        }
    }
}

fn find_expr_deps(expr: &Expr, deps: &mut FastHashSet<String>) {
    match expr {
        | Expr::Variable(_ident) => {}
        | Expr::IntConst(_value) => {}
        | Expr::UIntConst(_value) => {}
        | Expr::BoolConst(_value) => {}
        | Expr::FloatConst(_value) => {}
        | Expr::DoubleConst(_value) => {}
        | Expr::Unary(_op, expr) => find_expr_deps(expr, deps),
        | Expr::Binary(_op, lhs, rhs) => {
            find_expr_deps(lhs, deps);
            find_expr_deps(rhs, deps);
        }
        | Expr::Ternary(cond, iph, els) => {
            find_expr_deps(cond, deps);
            find_expr_deps(iph, deps);
            find_expr_deps(els, deps);
        }
        | Expr::Assignment(lhs, _op, rhs) => {
            find_expr_deps(lhs, deps);
            find_expr_deps(rhs, deps);
        }
        | Expr::Bracket(lhs, rhs) => {
            find_expr_deps(lhs, deps);
            for dimension in &rhs.dimensions.0 {
                match dimension {
                    | ArraySpecifierDimension::ExplicitlySized(dimension) => {
                        find_expr_deps(dimension, deps);
                    }
                    | ArraySpecifierDimension::Unsized => {}
                }
            }
        }
        | Expr::FunCall(ident, exprs) => {
            match ident {
                | FunIdentifier::Expr(expr) => {
                    if let Expr::Variable(ident) = &**expr {
                        deps.insert(ident.0.clone());
                    } else {
                        find_expr_deps(expr, deps);
                    }
                }
                | FunIdentifier::Identifier(ident) => {
                    deps.insert(ident.0.clone());
                }
            }
            for expr in exprs {
                find_expr_deps(expr, deps);
            }
        }
        | Expr::Dot(expr, _field_ident) => find_expr_deps(expr, deps),
        | Expr::PostInc(expr) => find_expr_deps(expr, deps),
        | Expr::PostDec(expr) => find_expr_deps(expr, deps),
        | Expr::Comma(lhs, rhs) => {
            find_expr_deps(lhs, deps);
            find_expr_deps(rhs, deps);
        }
    }
}

fn find_initializer_deps(initializer: &Initializer, deps: &mut FastHashSet<String>) {
    match &initializer {
        | Initializer::Simple(expr) => find_expr_deps(expr, deps),
        | Initializer::List(initializers) => {
            for initializer in initializers {
                find_initializer_deps(initializer, deps);
            }
        }
    }
}

fn find_declaration_deps(declaration: &Declaration, deps: &mut FastHashSet<String>) {
    match &declaration {
        | Declaration::FunctionPrototype(_prototype) => {}
        | Declaration::InitDeclaratorList(declarators) => {
            if let Some(initializer) = declarators.head.initializer.as_ref() {
                find_initializer_deps(initializer, deps);
            }
            for declaration in &declarators.tail {
                if let Some(initializer) = declaration.initializer.as_ref() {
                    find_initializer_deps(initializer, deps);
                }
            }
        }
        | Declaration::Precision(_precision_qualifier, _type_specifier) => {}
        | Declaration::Block(_block) => {}
        | Declaration::Global(_qualifier, _idents) => {}
    }
}

fn find_function_deps(def: &FunctionDefinition) -> FastHashSet<String> {
    let mut deps = FastHashSet::default();
    for stmt in &def.statement.statement_list {
        find_stmt_deps(stmt, &mut deps)
    }
    deps
}
