use std::{cell::RefCell, rc::Rc};

use crate::{
    helpers::{
        create_arr, create_bang_bool, create_cast_node, create_comp_node, create_func_call_node,
        create_keyword_node, create_make_var_node, create_set_var_node, get_exp_node,
        get_struct_access_tokens, is_exp, is_sequence, tokens_to_delimiter,
    },
    interpreter::VarType,
    tokenizer::Token,
};

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
    pub fn get_enum_str(&self) -> String {
        match self {
            Value::Int(_) => String::from("int"),
            Value::Usize(_) => String::from("usize"),
            Value::Float(_) => String::from("float"),
            Value::Double(_) => String::from("double"),
            Value::Long(_) => String::from("long"),
            Value::String(_) => String::from("string"),
            Value::Bool(_) => String::from("bool"),
            Value::Array(arr) => {
                let mut arr_type = if arr.len() > 0 {
                    arr[0].get_enum_str()
                } else {
                    String::from("unknown")
                };

                arr_type.push_str("[]");

                arr_type
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum AstNode<'a> {
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
    /// ident, from, to, inc, node
    ForFromTo(
        Token<'a>,
        Box<AstNode<'a>>,
        Box<AstNode<'a>>,
        Option<Box<AstNode<'a>>>,
        Box<AstNode<'a>>,
    ),
    IndexArr(Token<'a>, Box<AstNode<'a>>),
    /// arr ident, index, value
    SetArrIndex(Token<'a>, Box<AstNode<'a>>, Box<AstNode<'a>>),
}

impl<'a> AstNode<'a> {
    pub fn new_ptr(node: AstNode<'a>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(node))
    }
    pub fn get_str(&self) -> String {
        match &self {
            AstNode::SetArrIndex(arr, index, value) => {
                format!(
                    "setting {} to {} at {}",
                    arr.get_str(),
                    index.as_ref().get_str(),
                    value.as_ref().get_str()
                )
            }
            AstNode::IndexArr(arr, index) => {
                format!("indexing {} at {}", arr.get_str(), index.get_str())
            }
            AstNode::ForFromTo(ident, from, to, inc, node) => format!(
                "Looping {} from {} to {} by {} using {}",
                node.get_str(),
                from.get_str(),
                to.get_str(),
                if inc.is_some() {
                    inc.as_ref().unwrap().get_str()
                } else {
                    String::from("inc value not provided")
                },
                ident.get_str()
            ),
            AstNode::Comparison(operator, left, right) => {
                format!(
                    "Comparing: {} to {} with {}",
                    left.get_str(),
                    right.get_str(),
                    operator.get_str()
                )
            }
            AstNode::Cast(var_type, node) => {
                format!("Casting {:?} to {:?}", node, var_type)
            }
            AstNode::AccessStructProp(struct_token, prop) => {
                format!("Accessing {:?} on {}", prop, struct_token.get_str())
            }
            AstNode::Array(arr) => format!("{:?}", arr),
            AstNode::If(condition, node) => {
                format!("If: {} ::then:: {}", condition.get_str(), node.get_str())
            }
            AstNode::StatementSeq(seq) => {
                let mut res = String::from("Statement: ");
                for item in seq.iter() {
                    res.push_str(item.borrow().get_str().as_str());
                }
                res
            }
            AstNode::MakeVar(var_type, name, value) => {
                format!(
                    "Setting {} to {} as {}",
                    name.get_str(),
                    value.get_str().as_str(),
                    var_type.get_str()
                )
            }
            AstNode::SetVar(name, value) => {
                format!("Setting {:?} to {}", name, value.get_str().as_str())
            }
            AstNode::Token(t) => match t {
                Token::Number(s) => s.clone(),
                Token::String(s) => s.clone(),
                Token::Bool(b) => b.to_string(),
                Token::Identifier(s) => s.clone(),
                _ => panic!("Unable eval token value of token: {:?}", t),
            },
            AstNode::CallFunc(name, args) => {
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
            AstNode::Exp(tokens) => {
                let mut res = String::new();
                for (i, token) in tokens.iter().enumerate() {
                    res.push_str(token.get_str().as_str());
                    if i < tokens.len() - 1 {
                        res.push(' ');
                    }
                }
                res
            }
            AstNode::Bang(node) => format!("! -> {:?}", node.get_str().as_str()),
        }
    }
}

pub struct Ast<'a> {
    pub node: Rc<RefCell<AstNode<'a>>>,
}
impl<'a> Ast<'a> {
    fn new() -> Self {
        let node = AstNode::new_ptr(AstNode::StatementSeq(vec![]));
        Self { node }
    }
}

pub fn get_ast_node<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> Option<AstNode<'a>> {
    if tokens.len() == 0 {
        None
    } else if tokens.len() == 1 {
        Some(AstNode::Token(tokens[0].take().unwrap()))
    } else {
        if is_sequence(tokens) {
            let sequence_node = generate_sequence_node(tokens);
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

        if let Some(comp_node) = create_comp_node(tokens) {
            return Some(comp_node);
        }

        if is_exp(tokens) {
            let exp_nodes = get_exp_node(tokens);
            return Some(AstNode::Exp(exp_nodes));
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
                    Some(AstNode::AccessStructProp(struct_token, access_token_nodes))
                }
                Token::LBracket => {
                    let mut index_tokens = tokens_to_delimiter(tokens, 2, "]");
                    let arr_index_end = 3 + index_tokens.len();

                    Some(if let Some(index_node) = get_ast_node(&mut index_tokens) {
                        tokens.drain(1..arr_index_end);

                        if let Token::Semicolon = tokens.last().as_ref().unwrap().as_ref().unwrap()
                        {
                            tokens.remove(tokens.len() - 1);
                        }

                        if tokens.len() > 1 {
                            match tokens[1].as_ref().unwrap() {
                                Token::EqSet => {
                                    let arr_var_name = tokens.remove(0);
                                    tokens.remove(0);

                                    if let Some(value_node) = get_ast_node(tokens) {
                                        AstNode::SetArrIndex(
                                            arr_var_name.unwrap(),
                                            Box::new(index_node),
                                            Box::new(value_node),
                                        )
                                    } else {
                                        panic!("expected value to set at index");
                                    }
                                }
                                _ => unimplemented!(),
                            }
                        } else {
                            AstNode::IndexArr(tokens[0].take().unwrap(), Box::new(index_node))
                        }
                    } else {
                        panic!("Expected value to index array with");
                    })

                    // let (res, num) = create_index_arr_node(tokens);
                    // tokens.drain(0..num);

                    // // 1 because of semicolon at end
                    // let num = if let Token::Semicolon =
                    //     tokens.last().as_ref().unwrap().as_ref().unwrap()
                    // {
                    //     1
                    // } else {
                    //     0
                    // };
                    // if tokens.len() > num {
                    //     match tokens[0].as_ref().unwrap() {
                    //         Token::EqSet => {
                    //             println!("setting");
                    //         }
                    //         _ => unreachable!(),
                    //     }
                    //     return None;
                    // }

                    // Some(res)
                }
                _ => unimplemented!(),
            },
            Token::NewLine => None,
            Token::Bang => Some(create_bang_bool(tokens)),
            Token::String(_) => Some(AstNode::Token(tokens[0].take().unwrap())),
            Token::Keyword(keyword) => Some(create_keyword_node(tokens, keyword)),
            Token::LBracket => Some(create_arr(tokens)),
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

fn generate_sequence_node<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> AstNode<'a> {
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

    AstNode::StatementSeq(seq)
}

pub fn generate_tree<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> Ast<'a> {
    let mut tree = Ast::new();

    let node_option = get_ast_node(tokens);
    if let Some(node) = node_option {
        tree.node = Rc::new(RefCell::new(node));
    }

    tree
}
