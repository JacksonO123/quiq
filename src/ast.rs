use std::{cell::RefCell, rc::Rc};

use crate::{
    helpers::{create_func_call_node, create_make_var_node, create_set_var_node},
    tokenizer::{Token, TokenType},
};

#[derive(Clone)]
pub enum Value<'a> {
    String(&'a str),
    Int(i32),
    Float(f32),
    Double(f64),
    Long(i64),
    Bool(bool),
    MathExp(Vec<Token<'a>>),
    ConcatExp(Vec<Token<'a>>),
    AstNode(Rc<RefCell<AstNode<'a>>>),
}
impl<'a> Value<'a> {
    pub fn get_str(&self) -> &str {
        match self {
            Value::Bool(v) => {
                if *v {
                    "true"
                } else {
                    "false"
                }
            }
            _ => "Other val",
        }
    }
}

#[derive(Clone)]
pub enum AstNodeType<'a> {
    StatementSeq(Vec<Rc<RefCell<AstNode<'a>>>>),
    MakeVar(Token<'a>, Token<'a>, Box<AstNode<'a>>),
    SetVar(Token<'a>, Box<AstNode<'a>>),
    Token(Token<'a>),
    CallFunc(&'a str, Vec<AstNode<'a>>),
}

#[derive(Clone)]
pub struct AstNode<'a> {
    pub node_type: AstNodeType<'a>,
}
impl<'a> AstNode<'a> {
    fn new(node_type: AstNodeType<'a>) -> Self {
        Self { node_type }
    }
    fn new_ptr(node_type: AstNodeType<'a>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(AstNode::new(node_type)))
    }
    pub fn get_str(&self) -> String {
        match &self.node_type {
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
                    var_type.value
                )
            }
            AstNodeType::SetVar(name, value) => {
                format!("Setting {} to {}", name.value, value.get_str().as_str())
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

pub fn get_ast_node(tokens: Vec<Token>) -> Option<AstNode> {
    if tokens.len() == 0 {
        None
    } else if tokens.len() == 1 {
        Some(AstNode::new(AstNodeType::Token(tokens[0])))
    } else {
        let node_type = match tokens[0].token_type {
            TokenType::Type(_) => Some(create_make_var_node(tokens)),
            TokenType::Identifier => match tokens[1].token_type {
                TokenType::LParen => Some(create_func_call_node(tokens)),
                TokenType::EqSet => Some(create_set_var_node(tokens)),
                _ => panic!("Unexpected token: {}", tokens[1].value),
            },
            TokenType::NewLine => None,
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

pub fn tokens_until_semicolon(tokens: Vec<Token>, start: usize) -> Vec<Token> {
    let mut res = vec![];

    for i in start..tokens.len() {
        if tokens[i].value != ";" {
            res.push(tokens[i]);
        } else {
            break;
        }
    }

    res
}

pub fn generate_tree(tokens: Vec<Token>) -> Ast {
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

        let token_slice = tokens_until_semicolon(tokens.clone(), i);
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
