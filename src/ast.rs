use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{
    helpers::{
        create_arr, create_bang_bool, create_cast_node, create_comp_node, create_func_call_node,
        create_keyword_node, create_make_var_node, create_set_var_node, create_struct_node,
        get_bool_exp_node, get_exp_node, get_struct_access_tokens, get_type_expression,
        is_bool_exp, is_exp, is_sequence, tokens_to_delimiter,
    },
    interpreter::{CustomFunc, StructInfo, StructProp, VarType},
    tokenizer::{BoolComp, Keyword, Token},
};

#[derive(Debug, Clone)]
pub struct Generic {
    pub name: String,
}

impl Generic {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

#[derive(Debug, Clone)]
pub struct StructShape {
    pub props: HashMap<String, VarType>,
    pub generic_template: Vec<String>,
    pub generics: HashMap<String, VarType>,
}

impl StructShape {
    pub fn new() -> Self {
        Self {
            props: HashMap::new(),
            generic_template: vec![],
            generics: HashMap::new(),
        }
    }
    pub fn new_with_generics(generics: Vec<String>) -> Self {
        Self {
            props: HashMap::new(),
            generic_template: generics,
            generics: HashMap::new(),
        }
    }
    pub fn add(&mut self, name: String, prop_type: VarType) {
        self.props.insert(name.to_owned(), prop_type);
    }
}

pub fn get_value_arr_str(values: &Vec<Value>) -> String {
    let mut res = String::from("[");

    let size_buffer = 50;

    if values.len() < size_buffer * 2 {
        for (i, item) in values.iter().enumerate() {
            if i < values.len() - 1 {
                res.push_str(format!("{}, ", item.get_str()).as_str());
            } else {
                res.push_str(item.get_str().as_str());
            }
        }
    } else {
        for i in 0..size_buffer {
            res.push_str(values[i].get_str().as_str());
            if i < size_buffer - 1 {
                res.push_str(", ");
            } else {
                res.push_str("  ...  ");
            }
        }

        for i in values.len() - size_buffer..values.len() {
            res.push_str(values[i].get_str().as_str());
            if i < values.len() - 1 {
                res.push_str(", ");
            }
        }
    }

    res.push(']');

    res
}

#[derive(Clone, Debug)]
pub enum Value {
    Usize(usize),
    String(String),
    Int(i32),
    Float(f32),
    Double(f64),
    Long(i64),
    Bool(bool),
    Array(Vec<Value>, VarType),
    Null,
    /// struct type name, shape, props
    Struct(String, Rc<RefCell<StructShape>>, Vec<StructProp>),
    Ref(Rc<RefCell<Value>>),
    Fn(Rc<RefCell<CustomFunc>>),
}
impl Value {
    pub fn get_str(&self) -> String {
        match self {
            Value::Ref(v) => v.as_ref().borrow().get_str(),
            Value::Usize(v) => v.to_string(),
            Value::String(v) => v.clone(),
            Value::Float(v) => v.to_string(),
            Value::Double(v) => v.to_string(),
            Value::Long(v) => v.to_string(),
            Value::Int(v) => v.to_string(),
            Value::Bool(v) => {
                let res = if *v { "true" } else { "false" };
                res.to_string()
            }
            Value::Array(arr, _) => get_value_arr_str(arr),
            Value::Null => String::from("null"),
            Value::Struct(_, _, props) => {
                let mut res = String::from("{\n");

                for (i, prop) in props.iter().enumerate() {
                    let temp_str = prop.name.clone().to_owned() + ": " + prop.get_str().as_str();
                    let new_lines: Vec<String> = temp_str
                        .split("\n")
                        .map(|line| "\t".to_owned() + line)
                        .collect();
                    let mut new_str = new_lines.join("\n");

                    if i < props.len() - 1 {
                        new_str = new_str.to_owned() + "\n";
                    }

                    res.push_str(new_str.as_str());
                }

                res.push_str("\n}");

                res
            }
            Value::Fn(func) => {
                "func: (".to_owned()
                    + format!("{:?}", func.borrow().params).as_str()
                    + ") { "
                    + format!("{:?}", func.borrow().return_type).as_str()
                    + " }"
            }
        }
    }
    pub fn get_enum_str(&self) -> String {
        match self {
            Value::Ref(_) => String::from("ref"),
            Value::Int(_) => String::from("int"),
            Value::Usize(_) => String::from("usize"),
            Value::Float(_) => String::from("float"),
            Value::Double(_) => String::from("double"),
            Value::Long(_) => String::from("long"),
            Value::String(_) => String::from("string"),
            Value::Bool(_) => String::from("bool"),
            Value::Struct(_, _, _) => String::from("struct"),
            Value::Null => String::from("null"),
            Value::Fn(_) => String::from("fn"),
            Value::Array(arr, arr_type) => {
                format!("{:?}[{:?}]", arr_type, arr)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct FuncParam {
    pub param_type: VarType,
    pub name: String,
}
impl FuncParam {
    pub fn new(name: String, param_type: VarType) -> Self {
        Self { param_type, name }
    }
}

#[derive(Clone, Debug)]
pub enum AstNode {
    StatementSeq(Vec<Rc<RefCell<AstNode>>>),
    /// type, name, value node
    MakeVar(VarType, Token, Option<Box<AstNode>>),
    MakeFunc(Rc<RefCell<CustomFunc>>),
    SetVar(Token, Box<AstNode>),
    Token(Token),
    CallFunc(String, Vec<AstNode>),
    Bang(Box<AstNode>),
    Exp(Vec<Box<AstNode>>),
    If(Box<AstNode>, Box<AstNode>, Option<Box<AstNode>>),
    Else(Box<AstNode>),
    Array(Vec<AstNode>, VarType),
    AccessStructProp(Token, Vec<AstNode>),
    /// to, node
    Cast(VarType, Box<AstNode>),
    /// operator, left, right
    Comparison(Token, Box<AstNode>, Box<AstNode>),
    /// ident, from, to, inc, node
    ForFromTo(
        Token,
        Box<AstNode>,
        Box<AstNode>,
        Option<Box<AstNode>>,
        Box<AstNode>,
    ),
    /// exp node
    While(Vec<Box<AstNode>>, Box<AstNode>),
    IndexArr(Token, Box<AstNode>),
    /// arr ident, index, value
    SetArrIndex(Token, Box<AstNode>, Box<AstNode>),
    /// struct type name, shape, props
    CreateStruct(String, Rc<RefCell<StructShape>>, Vec<(String, AstNode)>),
    Return(Box<AstNode>),
    Break,
    Continue,
    /// comp type, left, right
    BoolExp(BoolComp, Box<AstNode>, Box<AstNode>),
}

impl AstNode {
    pub fn new_ptr(node: AstNode) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(node))
    }
}

#[derive(Debug)]
pub struct Ast {
    pub node: Rc<RefCell<AstNode>>,
}
impl Ast {
    fn new() -> Self {
        let node = AstNode::new_ptr(AstNode::StatementSeq(vec![]));
        Self { node }
    }
}

pub fn get_ast_node(structs: &mut StructInfo, tokens: &mut Vec<Option<Token>>) -> Option<AstNode> {
    if tokens.len() == 0 {
        None
    } else if tokens.len() == 1 {
        Some(AstNode::Token(tokens[0].take().unwrap()))
    } else {
        if is_sequence(tokens) {
            let sequence_node = generate_sequence_node(structs, tokens);
            return Some(sequence_node);
        }

        while tokens.len() > 0
            && match tokens[0].as_ref().unwrap() {
                Token::NewLine => true,
                _ => false,
            }
        {
            tokens.remove(0);
        }

        while tokens.len() > 1
            && match tokens[0].as_ref().unwrap() {
                Token::LParen => true,
                _ => false,
            }
            && match tokens.last().unwrap().as_ref().unwrap() {
                Token::RParen => true,
                _ => false,
            }
        {
            tokens.remove(tokens.len() - 1);
            tokens.remove(0);
        }

        if is_bool_exp(tokens) {
            return get_bool_exp_node(structs, tokens);
        }

        if let Some(comp_node) = create_comp_node(structs, tokens) {
            return Some(comp_node);
        }

        if is_exp(tokens) {
            let exp_nodes = get_exp_node(structs, tokens);
            return Some(AstNode::Exp(exp_nodes));
        }

        let node_type = match tokens[0].as_ref().unwrap() {
            Token::LAngle => {
                let first_token = tokens[1].as_ref().unwrap().clone();
                let mut type_tokens = tokens_to_delimiter(tokens, 1, ">");
                let type_val = get_type_expression(&mut type_tokens, structs);

                tokens.drain(0..type_tokens.len() + 2);

                match tokens[0].as_ref().unwrap() {
                    Token::LBracket => Some(create_arr(structs, tokens, type_val)),
                    Token::LBrace => {
                        if let Token::Identifier(ident) = first_token {
                            match structs.available_structs.get(&ident) {
                                Some(shape) => {
                                    let mut local_shape = {
                                        let temp_borrow = shape.borrow();
                                        temp_borrow.clone()
                                    };
                                    if let VarType::Struct(_, gen_types) = type_val {
                                        for (i, generic_name) in
                                            local_shape.generic_template.iter().enumerate()
                                        {
                                            local_shape
                                                .generics
                                                .insert(generic_name.clone(), gen_types[i].clone());
                                        }
                                    }

                                    Some(create_struct_node(
                                        tokens,
                                        structs,
                                        Rc::new(RefCell::new(local_shape)),
                                        &ident.clone(),
                                    ))
                                }
                                None => panic!("Struct {} does not exist", ident),
                            }
                        } else {
                            panic!("Expected struct type name for struct type definition");
                        }
                    }
                    _ => panic!("Unexpected type token"),
                }
            }
            Token::Type(_) => match tokens[1].as_ref().unwrap() {
                Token::LParen => Some(create_cast_node(structs, tokens)),
                _ => Some(create_make_var_node(structs, tokens, false)),
            },
            Token::Identifier(ident) => match tokens[1].as_ref().unwrap() {
                Token::LParen => Some(create_func_call_node(structs, tokens)),
                Token::EqSet => Some(create_set_var_node(structs, tokens)),
                Token::LAngle => {
                    if structs.available_structs.contains_key(ident) {
                        Some(create_make_var_node(structs, tokens, true))
                    } else {
                        Some(create_make_var_node(structs, tokens, false))
                    }
                }
                Token::Period => {
                    let struct_token = tokens[0].take().unwrap();
                    tokens.remove(0);
                    tokens.remove(0);

                    let mut access_token_nodes: Vec<AstNode> = Vec::new();
                    let mut tokens_clone = get_struct_access_tokens(tokens);
                    for token_clone in tokens_clone.iter_mut() {
                        let res = get_ast_node(structs, token_clone).unwrap();
                        access_token_nodes.push(res);
                    }

                    Some(AstNode::AccessStructProp(struct_token, access_token_nodes))
                }
                Token::LBracket => {
                    if structs.available_structs.contains_key(ident) {
                        Some(create_make_var_node(structs, tokens, true))
                    } else {
                        let mut index_tokens = tokens_to_delimiter(tokens, 2, "]");
                        let arr_index_end = 3 + index_tokens.len();

                        Some(
                            if let Some(index_node) = get_ast_node(structs, &mut index_tokens) {
                                tokens.drain(1..arr_index_end);

                                if let Token::Semicolon =
                                    tokens.last().as_ref().unwrap().as_ref().unwrap()
                                {
                                    tokens.remove(tokens.len() - 1);
                                }

                                if tokens.len() > 1 {
                                    match tokens[1].as_ref().unwrap() {
                                        Token::EqSet => {
                                            let arr_var_name = tokens.remove(0);
                                            tokens.remove(0);

                                            if let Some(value_node) = get_ast_node(structs, tokens)
                                            {
                                                AstNode::SetArrIndex(
                                                    arr_var_name.unwrap(),
                                                    Box::new(index_node),
                                                    Box::new(value_node),
                                                )
                                            } else {
                                                panic!("expected value to set at index");
                                            }
                                        }
                                        _ => panic!("Unexpected token \"{:?}\"", tokens[1]),
                                    }
                                } else {
                                    AstNode::IndexArr(
                                        tokens[0].take().unwrap(),
                                        Box::new(index_node),
                                    )
                                }
                            } else {
                                panic!("Expected value to index array with");
                            },
                        )
                    }
                }
                _ => {
                    let includes_struct = structs.available_structs.contains_key(ident);
                    if includes_struct {
                        Some(create_make_var_node(structs, tokens, true))
                    } else {
                        panic!("{} is not a valid type", ident);
                    }
                }
            },
            Token::NewLine => None,
            Token::Bang => Some(create_bang_bool(structs, tokens)),
            Token::String(_) => Some(AstNode::Token(tokens[0].take().unwrap())),
            Token::Keyword(keyword) => create_keyword_node(tokens, structs, keyword.clone()),
            _ => {
                panic!("Token not implemented: {:?}", tokens)
            }
        };

        if let Some(nt) = node_type {
            Some(nt)
        } else {
            None
        }
    }
}

fn get_sequence_slice<'a>(tokens: &mut Vec<Option<Token>>, start: usize) -> Vec<Option<Token>> {
    let mut slice = Vec::new();
    let mut open_parens = 0;

    for i in start..tokens.len() {
        slice.push(Some(tokens[i].take().unwrap()));
        match slice[slice.len() - 1].as_ref().unwrap() {
            Token::Semicolon => {
                if open_parens == 0 {
                    break;
                }
            }
            Token::LParen => {
                open_parens += 1;
            }
            Token::RParen => {
                open_parens -= 1;
            }
            Token::LBracket => {
                open_parens += 1;
            }
            Token::RBracket => {
                open_parens -= 1;
            }
            Token::LBrace => {
                open_parens += 1;
            }
            Token::RBrace => {
                open_parens -= 1;
                if open_parens == 0 {
                    break;
                }
            }
            _ => {}
        }
    }

    if open_parens > 0 {
        panic!("Expected token: `}}` or `;`");
    }

    slice
}

fn generate_sequence_node(structs: &mut StructInfo, tokens: &mut Vec<Option<Token>>) -> AstNode {
    let mut seq: Vec<Rc<RefCell<AstNode>>> = Vec::new();

    let mut i = 0;
    while i < tokens.len() {
        if let Token::NewLine = tokens[i].as_ref().unwrap() {
            i += 1;
            continue;
        }

        if i > 0 {
            if match tokens[i - 1].as_ref().unwrap() {
                Token::Keyword(Keyword::If) => true,
                Token::Keyword(Keyword::Else) => true,
                Token::Keyword(Keyword::Return) => true,
                Token::Identifier(_) => true,
                _ => false,
            } {
                i -= 1;
            }
        }
        let mut token_slice = get_sequence_slice(tokens, i);
        let token_num = token_slice.len();

        let node_option = get_ast_node(structs, &mut token_slice);
        if let Some(node) = node_option {
            if let AstNode::Else(block) = node {
                if seq.len() > 0 {
                    if let AstNode::If(_, _, ref mut else_block) =
                        &mut *seq[seq.len() - 1].borrow_mut()
                    {
                        *else_block = Some(block);
                    }
                }
            } else {
                let ptr = Rc::new(RefCell::new(node));
                seq.push(ptr);
            }
        }
        i += token_num + 1;
    }

    AstNode::StatementSeq(seq)
}

pub fn generate_tree(structs: &mut StructInfo, tokens: &mut Vec<Option<Token>>) -> Ast {
    let mut tree = Ast::new();

    let node_option = get_ast_node(structs, tokens);
    if let Some(node) = node_option {
        tree.node = Rc::new(RefCell::new(node));
    }

    tree
}
