use std::{cell::RefCell, rc::Rc};

use crate::{
    helpers::{
        create_arr, create_bang_bool, create_func_call_node, create_keyword_node,
        create_make_var_node, create_set_var_node, get_exp_node, is_exp, tokens_to_delimiter,
    },
    interpreter::VarType,
    tokenizer::{Token, TokenType},
};

#[derive(Clone, Debug)]
pub enum Value {
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
            Value::String(v) => v.to_string(),
            Value::Float(v) => v.to_string(),
            Value::Double(v) => v.to_string(),
            Value::Long(v) => v.to_string(),
            Value::Int(v) => v.to_string(),
            Value::Bool(v) => {
                let res = if *v { "true" } else { "false" };
                res.to_string()
            }
            Value::Array(arr) => {
                let mut res = String::from("[");

                for (i, item) in arr.iter().enumerate() {
                    if i < arr.len() - 1 {
                        res.push_str(format!("{}, ", item.get_str()).as_str());
                    } else {
                        res.push_str(item.get_str().as_str());
                    }
                }

                res.push(']');

                res
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum AstNodeType<'a> {
    StatementSeq(Vec<Rc<RefCell<AstNode<'a>>>>),
    /// type, name, value node
    MakeVar(VarType, Token, Box<AstNode<'a>>),
    SetVar(Token, Box<AstNode<'a>>),
    Token(Token),
    CallFunc(String, Vec<AstNode<'a>>),
    Bang(Box<AstNode<'a>>),
    Exp(Vec<Box<AstNode<'a>>>),
    If(Box<AstNode<'a>>, Box<AstNode<'a>>),
    CreateArr(Value), // possibly change
    Array(Vec<AstNode<'a>>),
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
            AstNodeType::Array(arr) => format!("{:?}", arr),
            AstNodeType::CreateArr(val) => format!("[{}]", val.get_str()),
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
                    "Setting {} to {} as {}",
                    name.value,
                    value.get_str().as_str(),
                    var_type.get_str()
                )
            }
            AstNodeType::SetVar(name, value) => {
                format!("Setting {} to {}", name.get_str(), value.get_str().as_str())
            }
            AstNodeType::Token(t) => t.value.to_string(),
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
    fn add(&mut self, node: Rc<RefCell<AstNode<'a>>>) {
        match &mut self.node.borrow_mut().node_type {
            AstNodeType::StatementSeq(seq) => {
                seq.push(node);
            }
            _ => {}
        }
    }
    pub fn print(&self) {
        match &self.node.borrow().node_type {
            AstNodeType::StatementSeq(seq) => {
                for item in seq.iter() {
                    println!("Ast->node:: {}", item.borrow().get_str());
                }
            }
            _ => {}
        }
    }
}

pub fn get_ast_node<'a>(tokens: Vec<Token>) -> Option<AstNode<'a>> {
    let mut tokens = tokens.clone();
    if tokens.len() == 0 {
        None
    } else if tokens.len() == 1 {
        Some(AstNode::new(AstNodeType::Token(tokens[0].clone())))
    } else {
        while tokens.len() > 0
            && match tokens[0].token_type {
                TokenType::NewLine => true,
                _ => false,
            }
        {
            tokens.remove(0);
        }

        if is_exp(tokens.clone()) {
            let exp_nodes = get_exp_node(tokens.clone());
            return Some(AstNode::new(AstNodeType::Exp(exp_nodes)));
        }

        let node_type = match tokens[0].token_type {
            TokenType::Type(_) => Some(create_make_var_node(tokens)),
            TokenType::Identifier => match tokens[1].token_type {
                TokenType::LParen => Some(create_func_call_node(tokens)),
                TokenType::EqSet => Some(create_set_var_node(tokens)),
                _ => unimplemented!(),
            },
            TokenType::NewLine => None,
            TokenType::Bang => Some(create_bang_bool(tokens)),
            TokenType::String => Some(AstNodeType::Token(tokens[0].clone())),
            TokenType::Keyword => Some(create_keyword_node(tokens)),
            TokenType::LBracket => Some(create_arr(tokens)),
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

pub fn generate_tree<'a>(tokens: Vec<Token>) -> Ast<'a> {
    let mut tree = Ast::new();

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i].token_type {
            TokenType::NewLine => {
                i += 1;
                continue;
            }
            _ => {}
        }

        let token_slice = tokens_to_delimiter(tokens.clone(), i, ";");
        let token_num = token_slice.len();

        let node_option = get_ast_node(token_slice);
        if let Some(node) = node_option {
            let ptr = Rc::new(RefCell::new(node));
            tree.add(ptr);
        }
        i += token_num + 1;
    }

    tree
}
