use smol_str::SmolStr;

#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Function(Function),
    Constant(f32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub ident: SmolStr,
    pub args: Vec<Node>,
}

impl From<Function> for Node {
    fn from(value: Function) -> Self {
        Self::Function(value)
    }
}

impl From<f32> for Node {
    fn from(value: f32) -> Self {
        Self::Constant(value)
    }
}
