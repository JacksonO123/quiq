use crate::{
    ast::{get_ast_node, tokens_until_semicolon, AstNode, AstNodeType, Value},
    interpreter::VarValue,
    tokenizer::{Token, TokenType},
};

pub fn create_make_var_node(tokens: Vec<Token>) -> AstNodeType {
    let value_to_set = tokens_until_semicolon(tokens.clone(), 3);
    let node = get_ast_node(value_to_set);
    if node.is_none() {
        panic!("Invalid value, expected value to set variable");
    }
    AstNodeType::MakeVar(tokens[0], tokens[1], Box::new(node.unwrap()))
}

pub fn create_set_var_node(tokens: Vec<Token>) -> AstNodeType {
    let value_to_set = tokens_until_semicolon(tokens.clone(), 2);
    let node = get_ast_node(value_to_set);
    if node.is_none() {
        panic!("Invalid value, expected value to set variable");
    }
    AstNodeType::SetVar(tokens[0], Box::new(node.unwrap()))
}

fn get_params(tokens: Vec<Token>) -> Vec<AstNode> {
    // TODO: allow for multiple params, reliable
    let mut tokens_between_parens: Vec<Token> = Vec::new();
    let mut i = 2;
    let mut end_found = false;
    while i < tokens.len() {
        match tokens[i].token_type {
            TokenType::RParen => {}
            _ => {
                tokens_between_parens.push(tokens[i]);
                end_found = true;
            }
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

pub fn create_func_call_node(tokens: Vec<Token>) -> AstNodeType {
    let params = get_params(tokens.clone());
    AstNodeType::CallFunc(tokens[0].value, params)
}

pub fn set_var_value<'a>(vars: &mut Vec<VarValue<'a>>, name: &'a str, value: Value<'a>) {
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
