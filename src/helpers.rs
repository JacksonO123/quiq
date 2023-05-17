use std::{cell::RefCell, rc::Rc};

use crate::{
    ast::{get_ast_node, AstNode, AstNodeType, Value},
    interpreter::{eval_node, value_from_token, EvalValue, Func, VarValue},
    tokenizer::{Token, TokenType},
};

const OPEN_BRACKETS: [&str; 3] = ["(", "{", "["];
const CLOSE_BRACKETS: [&str; 3] = [")", "}", "]"];

pub fn create_make_var_node<'a>(tokens: Vec<Token>) -> AstNodeType<'a> {
    let value_to_set = tokens_to_delimiter(tokens.clone(), 3, ";");
    let node = get_ast_node(value_to_set);
    if node.is_none() {
        panic!("Invalid value, expected value to set variable");
    }
    AstNodeType::MakeVar(
        tokens[0].clone(),
        tokens[1].clone(),
        Box::new(node.unwrap()),
    )
}

pub fn create_set_var_node<'a>(tokens: Vec<Token>) -> AstNodeType<'a> {
    let value_to_set = tokens_to_delimiter(tokens.clone(), 2, ";");
    let node = get_ast_node(value_to_set);
    if node.is_none() {
        panic!("Invalid value, expected value to set variable");
    }
    AstNodeType::SetVar(tokens[0].clone(), Box::new(node.unwrap()))
}

fn get_params<'a>(tokens: Vec<Token>) -> Vec<AstNode<'a>> {
    // TODO: allow for multiple params, reliable
    let mut tokens_between_parens: Vec<Token> = Vec::new();
    let mut i = 2;
    let mut end_found = false;

    let mut num_open_parens = 0;

    while i < tokens.len() {
        match tokens[i].token_type {
            TokenType::RParen => {
                num_open_parens -= 1;
            }
            TokenType::LParen => {
                num_open_parens += 1;
            }
            _ => {}
        }

        if num_open_parens >= 0 {
            tokens_between_parens.push(tokens[i].clone());
            end_found = true;
        } else {
            break;
        }

        i += 1;
    }

    if !end_found {
        panic!("Expected token: )");
    }

    let res_option = get_ast_node(tokens_between_parens);
    if let Some(res) = res_option {
        vec![res]
    } else {
        vec![]
    }
}

pub fn create_func_call_node<'a>(tokens: Vec<Token>) -> AstNodeType<'a> {
    let params = get_params(tokens.clone());
    AstNodeType::CallFunc(tokens[0].clone().value, params)
}

pub fn set_var_value<'a>(vars: &mut Vec<VarValue>, name: String, value: Value) {
    let mut found = false;
    for var in vars.iter_mut() {
        if var.name == name {
            var.value = value;
            found = true;
            break;
        }
    }
    if !found {
        panic!("Cannot set value of undefined variable: {}", name);
    }
}

pub fn tokens_to_delimiter<'a>(tokens: Vec<Token>, start: usize, delimiter: &'a str) -> Vec<Token> {
    let mut res = vec![];

    let mut open_brackets = 0;
    for i in start..tokens.len() {
        if OPEN_BRACKETS
            .iter()
            .position(|&s| tokens[i].value == s)
            .is_some()
        {
            open_brackets += 1;
        } else if CLOSE_BRACKETS
            .iter()
            .position(|&s| tokens[i].value == s)
            .is_some()
        {
            open_brackets -= 1;
        }
        if tokens[i].value != delimiter || open_brackets > 0 {
            res.push(tokens[i].clone());
        } else {
            break;
        }
    }

    res
}

pub fn tokens_to_operator<'a>(tokens: Vec<Token>, start: usize) -> Vec<Token> {
    let mut res = vec![];

    let mut open_brackets = 0;
    for i in start..tokens.len() {
        if OPEN_BRACKETS
            .iter()
            .position(|&s| tokens[i].value == s)
            .is_some()
        {
            open_brackets += 1;
        } else if CLOSE_BRACKETS
            .iter()
            .position(|&s| tokens[i].value == s)
            .is_some()
        {
            open_brackets -= 1;
        }
        if match tokens[i].token_type {
            TokenType::Operator(_) => false,
            _ => true,
        } || open_brackets > 0
        {
            res.push(tokens[i].clone());
        } else {
            break;
        }
    }

    res
}

pub fn get_exp_node<'a>(tokens: Vec<Token>) -> Vec<Box<AstNode<'a>>> {
    let mut res: Vec<Box<AstNode>> = Vec::new();

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i].token_type {
            TokenType::Operator(_) => {
                let node = AstNode::new(AstNodeType::Token(tokens[i].clone()));
                res.push(Box::new(node));
                i += 1;
                continue;
            }
            _ => {}
        }

        let to_op = tokens_to_operator(tokens.clone(), i);

        if to_op.len() > 0 {
            if match to_op[0].token_type {
                TokenType::LParen => true,
                _ => false,
            } {
                let slice = &to_op[1..to_op.len() - 1];
                let slice: Vec<Token> = slice.iter().map(|t| t.clone()).collect();
                let exp_nodes = get_exp_node(slice);
                let node = AstNode::new(AstNodeType::Exp(exp_nodes));
                res.push(Box::new(node));
            } else {
                let node_option = get_ast_node(to_op.clone());
                if let Some(node) = node_option {
                    res.push(Box::new(node));
                }
            }
        }
        i += to_op.len();
    }

    res
}

pub fn is_exp(tokens: Vec<Token>) -> bool {
    let mut open_brackets = 0;

    for token in tokens.iter() {
        if OPEN_BRACKETS
            .iter()
            .position(|&s| token.value == s)
            .is_some()
        {
            open_brackets += 1;
        } else if CLOSE_BRACKETS
            .iter()
            .position(|&s| token.value == s)
            .is_some()
        {
            open_brackets -= 1;
        }
        if open_brackets == 0
            && match token.token_type {
                TokenType::Operator(_) => true,
                _ => false,
            }
        {
            return true;
        }
    }
    false
}

#[derive(Clone, Debug)]
pub enum ExpValue {
    Value(Value),
    Operator(TokenType),
}

pub fn flatten_exp<'a>(
    vars: &mut Vec<VarValue>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    scope: usize,
    exp: Vec<Box<AstNode<'a>>>,
) -> Vec<ExpValue> {
    let mut res: Vec<ExpValue> = Vec::new();

    for exp_node in exp.iter() {
        let tok_option = eval_node(
            vars,
            Rc::clone(&functions),
            scope,
            exp_node.as_ref().clone(),
        );
        if let Some(tok) = tok_option {
            let val = match tok {
                EvalValue::Token(tok) => match tok.token_type {
                    TokenType::Operator(_) => ExpValue::Operator(tok.token_type),
                    _ => ExpValue::Value(value_from_token(vars, tok, None)),
                },
                EvalValue::Value(val) => ExpValue::Value(val),
            };
            res.push(val);
        }
    }

    res
}

pub fn create_bang_bool<'a>(tokens: Vec<Token>) -> AstNodeType<'a> {
    let mut new_tokens = tokens.clone();
    new_tokens.remove(0);

    let ast_node = get_ast_node(new_tokens);

    if let Some(node) = ast_node {
        AstNodeType::Bang(Box::new(node))
    } else {
        panic!("Expected value to !");
    }
}
