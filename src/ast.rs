use std::{cell::RefCell, rc::Rc};

use crate::{
    helpers::{
        create_arr, create_bang_bool, create_cast_node, create_comp_node, create_func_call_node,
        create_keyword_node, create_make_var_node, create_set_var_node, get_exp_node,
        get_struct_access_tokens, is_exp, is_sequence,
    },
    interpreter::VarType,
    tokenizer::Token,
};

pub fn get_value_arr_str(values: &Vec<Value>) -> String {
    let mut res = String::from("[");

    let size_buffer = 100;

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

        for i in values.len() - 1 - size_buffer..values.len() {
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
    Array(Vec<Value>),
}
impl Value {
    pub fn get_str(&self) -> String {
        match self {
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
            Value::Array(arr) => get_value_arr_str(arr),
        }
    }
}

#[derive(Clone, Debug)]
pub enum AstNodeType<'a> {
    StatementSeq(Vec<Rc<RefCell<AstNode<'a>>>>),
    /// type, name, value node
    MakeVar(VarType, Token<'a>, Box<AstNode<'a>>),
    SetVar(Token<'a>, Box<AstNode<'a>>),
    Token(Token<'a>),
    CallFunc(String, Vec<AstNode<'a>>),
    Bang(Box<AstNode<'a>>),
    Exp(Vec<Box<AstNode<'a>>>),
    If(Box<AstNode<'a>>, Box<AstNode<'a>>),
    Array(Vec<AstNode<'a>>),
    AccessStructProp(Token<'a>, Vec<AstNode<'a>>),
    /// to, node
    Cast(VarType, Box<AstNode<'a>>),
    /// operator, left, right
    Comparison(Token<'a>, Box<AstNode<'a>>, Box<AstNode<'a>>),
    /// ident, from, to, node
    ForFromTo(
        Token<'a>,
        Box<AstNode<'a>>,
        Box<AstNode<'a>>,
        Box<AstNode<'a>>,
    ),
}

#[derive(Clone, Debug)]
pub struct AstNode<'a> {
    pub node_type: AstNodeType<'a>,
}
impl<'a> AstNode<'a> {
    pub fn new(node_type: AstNodeType<'a>) -> Self {
        Self { node_type }
    }
    pub fn new_ptr(node_type: AstNodeType<'a>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(AstNode::new(node_type)))
    }
    pub fn get_str(&self) -> String {
        match &self.node_type {
            AstNodeType::ForFromTo(ident, from, to, node) => format!(
                "Looping {} from {} to {}",
                node.get_str(),
                from.get_str(),
                to.get_str(),
            ),
            AstNodeType::Comparison(operator, left, right) => {
                format!(
                    "Comparing: {} to {} with {:?}",
                    left.get_str(),
                    right.get_str(),
                    operator
                )
            }
            AstNodeType::Cast(var_type, node) => {
                format!("Casting {:?} to {:?}", node, var_type)
            }
            AstNodeType::AccessStructProp(struct_token, prop) => {
                format!("Accessing {:?} on {:?}", prop, struct_token)
            }
            AstNodeType::Array(arr) => format!("{:?}", arr),
            AstNodeType::If(condition, node) => {
                format!("If: {} ::then:: {}", condition.get_str(), node.get_str())
            }
            AstNodeType::StatementSeq(seq) => {
                let mut res = String::from("Statement: ");
                for item in seq.iter() {
                    res.push_str(item.borrow().get_str().as_str());
                }
                res
            }
            AstNodeType::MakeVar(var_type, name, value) => {
                format!(
                    "Setting {:?} to {} as {}",
                    name,
                    value.get_str().as_str(),
                    var_type.get_str()
                )
            }
            AstNodeType::SetVar(name, value) => {
                format!("Setting {:?} to {}", name, value.get_str().as_str())
            }
            AstNodeType::Token(t) => match t {
                Token::Number(s) => s.clone(),
                Token::String(s) => s.clone(),
                Token::Bool(b) => b.to_string(),
                Token::Identifier(s) => s.clone(),
                _ => panic!("Unable eval token value of token: {:?}", t),
            },
            AstNodeType::CallFunc(name, args) => {
                let mut res = format!("Calling func {} with args: ", name);
                for (i, arg) in args.iter().enumerate() {
                    if i < args.len() - 1 {
                        res.push_str(format!("{}, ", arg.get_str()).as_str());
                    } else {
                        res.push_str(arg.get_str().as_str())
                    }
                }
                name.to_string()
            }
            AstNodeType::Exp(tokens) => {
                let mut res = String::new();
                for (i, token) in tokens.iter().enumerate() {
                    res.push_str(token.get_str().as_str());
                    if i < tokens.len() - 1 {
                        res.push(' ');
                    }
                }
                res
            }
            AstNodeType::Bang(node) => format!("! -> {:?}", node.get_str().as_str()),
        }
    }
}

pub struct Ast<'a> {
    pub node: Rc<RefCell<AstNode<'a>>>,
}
impl<'a> Ast<'a> {
    fn new() -> Self {
        let node = AstNode::new_ptr(AstNodeType::StatementSeq(vec![]));
        Self { node }
    }
}

pub fn get_ast_node<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> Option<AstNode<'a>> {
    if tokens.len() == 0 {
        None
    } else if tokens.len() == 1 {
        Some(AstNode::new(AstNodeType::Token(tokens[0].take().unwrap())))
    } else {
        if is_sequence(tokens) {
            let sequence_node = generate_sequence_node(tokens);
            return Some(AstNode::new(sequence_node));
        }

        while tokens.len() > 0
            && match tokens[0].as_ref().unwrap() {
                Token::NewLine => true,
                _ => false,
            }
        {
            tokens.remove(0);
        }

        if let Some(comp_node) = create_comp_node(tokens) {
            return Some(AstNode::new(comp_node));
        }

        if is_exp(tokens) {
            let exp_nodes = get_exp_node(tokens);
            return Some(AstNode::new(AstNodeType::Exp(exp_nodes)));
        }

        let node_type = match tokens[0].as_ref().unwrap() {
            Token::Type(_) => match tokens[1].as_ref().unwrap() {
                Token::LParen => Some(create_cast_node(tokens)),
                _ => Some(create_make_var_node(tokens)),
            },
            Token::Identifier(_) => match tokens[1].as_ref().unwrap() {
                Token::LParen => Some(create_func_call_node(tokens)),
                Token::EqSet => Some(create_set_var_node(tokens)),
                Token::Period => {
                    let struct_token = tokens[0].take().unwrap();
                    tokens.remove(0);
                    tokens.remove(0);

                    let mut access_token_nodes: Vec<AstNode> = Vec::new();
                    let mut tokens_clone = get_struct_access_tokens(tokens);
                    for token_clone in tokens_clone.iter_mut() {
                        let res = get_ast_node(token_clone).unwrap();
                        access_token_nodes.push(res);
                    }
                    Some(AstNodeType::AccessStructProp(
                        struct_token,
                        access_token_nodes,
                    ))
                }
                _ => unimplemented!(),
            },
            Token::NewLine => None,
            Token::Bang => Some(create_bang_bool(tokens)),
            Token::String(_) => Some(AstNodeType::Token(tokens[0].take().unwrap())),
            Token::Keyword(keyword) => Some(create_keyword_node(tokens, keyword)),
            Token::LBracket => Some(create_arr(tokens)),
            _ => {
                panic!("Token not implemented: {:?}", tokens)
            }
        };

        if let Some(nt) = node_type {
            Some(AstNode::new(nt))
        } else {
            None
        }
    }
}

fn get_sequence_slice<'a>(
    tokens: &mut Vec<Option<Token<'a>>>,
    start: usize,
) -> Vec<Option<Token<'a>>> {
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

fn generate_sequence_node<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> AstNodeType<'a> {
    let mut seq: Vec<Rc<RefCell<AstNode>>> = Vec::new();

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i].as_ref().unwrap() {
            Token::NewLine => {
                i += 1;
                continue;
            }
            _ => {}
        }

        let mut token_slice = get_sequence_slice(tokens, i);
        let token_num = token_slice.len();

        let node_option = get_ast_node(&mut token_slice);
        if let Some(node) = node_option {
            let ptr = Rc::new(RefCell::new(node));
            seq.push(ptr);
        }
        i += token_num + 1;
    }

    AstNodeType::StatementSeq(seq)
}

pub fn generate_tree<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> Ast<'a> {
    let mut tree = Ast::new();

    let node_option = get_ast_node(tokens);
    if let Some(node) = node_option {
        tree.node = Rc::new(RefCell::new(node));
    }

    tree
}
