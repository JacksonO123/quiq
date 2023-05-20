use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{
    ast::{get_ast_node, get_value_arr_str, AstNode, AstNodeType, Value},
    interpreter::{eval_node, value_from_token, EvalValue, Func, VarType, VarValue},
    tokenizer::{Token, TokenType},
};

pub fn create_make_var_node<'a>(tokens: Vec<Token>) -> AstNodeType<'a> {
    let mut var_type = VarType::from(tokens[0].value.as_str());

    let mut i = 0;
    while match tokens[i + 1].token_type {
        TokenType::LBracket => true,
        _ => false,
    } {
        var_type = VarType::Array(Box::new(var_type));

        i += 2;
    }

    let eq_pos = tokens
        .iter()
        .position(|t| match t.token_type {
            TokenType::EqSet => true,
            _ => false,
        })
        .unwrap();

    let mut value_to_set = tokens_to_delimiter(&tokens, eq_pos + 1, ";");

    let node = get_ast_node(&mut value_to_set);
    if node.is_none() {
        panic!("Invalid value, expected value to set variable");
    }

    AstNodeType::MakeVar(
        var_type,
        tokens[eq_pos - 1].clone(),
        Box::new(node.unwrap()),
    )
}

pub fn create_set_var_node<'a>(tokens: Vec<Token>) -> AstNodeType<'a> {
    let mut value_to_set = tokens_to_delimiter(&tokens, 2, ";");
    let node = get_ast_node(&mut value_to_set);
    if node.is_none() {
        panic!("Invalid value, expected value to set variable");
    }
    AstNodeType::SetVar(tokens[0].clone(), Box::new(node.unwrap()))
}

pub fn create_keyword_node<'a>(tokens: Vec<Token>) -> AstNodeType<'a> {
    match tokens[0].value.as_str() {
        "if" => {
            let mut condition = tokens_to_delimiter(&tokens, 2, ")");
            let condition_len = condition.len();
            let condition_node = get_ast_node(&mut condition);

            // + 4 because tokens: [if, (, ), {]
            let mut tokens_to_run = tokens_to_delimiter(&tokens, condition_len + 4, "}");
            let to_run_node = get_ast_node(&mut tokens_to_run);

            if let Some(c_node) = condition_node {
                if let Some(tr_node) = to_run_node {
                    AstNodeType::If(Box::new(c_node), Box::new(tr_node))
                } else {
                    panic!("Expected block for `if`");
                }
            } else {
                panic!("Expected condition for `if`");
            }
        }
        "else" => unimplemented!(),
        "for" => unimplemented!(),
        "while" => unimplemented!(),
        _ => panic!("Unexpected keyword: {}", tokens[0].value),
    }
}

fn get_params<'a>(tokens: Vec<Token>) -> Vec<AstNode<'a>> {
    let mut arg_nodes: Vec<AstNode> = Vec::new();
    let mut temp_tokens: Vec<Token> = Vec::new();
    let mut i = 2;

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

        if num_open_parens < 0 {
            break;
        }

        match tokens[i].token_type {
            TokenType::Comma => {
                let arg_node_option = get_ast_node(&mut temp_tokens);
                if let Some(arg_node) = arg_node_option {
                    arg_nodes.push(arg_node);
                } else {
                    panic!("Error in function parameters");
                }
                temp_tokens = Vec::new();
            }
            _ => {
                temp_tokens.push(tokens[i].clone());
            }
        }

        i += 1;
    }

    if num_open_parens >= 0 {
        panic!("Expected token: )");
    }

    if temp_tokens.len() > 0 {
        let arg_node_option = get_ast_node(&mut temp_tokens);
        if let Some(arg_node) = arg_node_option {
            arg_nodes.push(arg_node);
        } else {
            panic!("Error in function parameters");
        }
    }

    arg_nodes
}

pub fn create_func_call_node<'a>(tokens: Vec<Token>) -> AstNodeType<'a> {
    let params = get_params(tokens.clone());
    AstNodeType::CallFunc(tokens[0].value.clone(), params)
}

pub fn set_var_value<'a>(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    name: &'a str,
    value: Value,
) {
    let res_option = vars.get(name);
    if let Some(res) = res_option {
        let mut value_ref = res.borrow_mut();
        let var_value = &value_ref.value;
        let var_value_type = get_var_type_from_value(var_value);
        if ensure_type(&var_value_type, &value) {
            value_ref.value = value;
        } else {
            panic!(
                "expected type: {:?} found {:?}",
                var_value_type,
                get_var_type_from_value(&value)
            );
        }
    } else {
        panic!("Cannot set value of undefined variable: {}", name);
    }
}

pub fn tokens_to_delimiter<'a>(
    tokens: &Vec<Token>,
    start: usize,
    delimiter: &'a str,
) -> Vec<Token> {
    let mut res = vec![];

    let mut open_brackets = 0;
    for i in start..tokens.len() {
        if match tokens[i].token_type {
            TokenType::LParen => true,
            TokenType::LBrace => true,
            TokenType::LBracket => true,
            _ => false,
        } {
            open_brackets += 1;
        } else if match tokens[i].token_type {
            TokenType::RParen => true,
            TokenType::RBrace => true,
            TokenType::RBracket => true,
            _ => false,
        } {
            open_brackets -= 1;
        }
        if tokens[i].value != delimiter || open_brackets > 0 {
            res.push(tokens[i].clone());
        } else {
            return res;
        }
    }

    res
}

pub fn tokens_to_operator<'a>(tokens: Vec<Token>, start: usize) -> Vec<Token> {
    let mut res = vec![];

    let mut open_brackets = 0;
    for i in start..tokens.len() {
        if match tokens[i].token_type {
            TokenType::LParen => true,
            TokenType::LBrace => true,
            TokenType::LBracket => true,
            _ => false,
        } {
            open_brackets += 1;
        } else if match tokens[i].token_type {
            TokenType::RParen => true,
            TokenType::RBrace => true,
            TokenType::RBracket => true,
            _ => false,
        } {
            open_brackets -= 1;
        }
        if match tokens[i].token_type {
            TokenType::Operator(_) => false,
            _ => true,
        } || open_brackets > 0
        {
            res.push(tokens[i].clone());
        } else {
            return res;
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

        let mut to_op = tokens_to_operator(tokens.clone(), i);

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
                let node_option = get_ast_node(&mut to_op);
                if let Some(node) = node_option {
                    res.push(Box::new(node));
                }
            }
        }
        i += to_op.len();
    }

    res
}

pub fn is_exp(tokens: &Vec<Token>) -> bool {
    let mut open_brackets = 0;
    let mut res = true;
    let mut op_found = false;

    for token in tokens.iter() {
        match token.token_type {
            TokenType::Operator(_) => {
                if open_brackets == 0 {
                    op_found = true;
                }
            }
            TokenType::RBrace => {
                open_brackets -= 1;
            }
            TokenType::RBracket => {
                open_brackets -= 1;
            }
            TokenType::RParen => {
                open_brackets -= 1;
            }
            TokenType::LBrace => {
                open_brackets += 1;
            }
            TokenType::LBracket => {
                open_brackets += 1;
            }
            TokenType::LParen => {
                open_brackets += 1;
            }
            TokenType::EqCompare => {
                res = false;
            }
            TokenType::EqNCompare => {
                res = false;
            }
            TokenType::EqSet => {
                res = false;
            }
            TokenType::Bang => {
                res = false;
            }
            TokenType::Identifier => {}
            TokenType::Period => {}
            _ => {}
        }
        if !res {
            break;
        }
    }
    res && open_brackets == 0 && op_found
}

#[derive(Clone, Debug)]
pub enum ExpValue {
    Value(Value),
    Operator(TokenType),
}

pub fn flatten_exp<'a>(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    scope: usize,
    exp: Vec<Box<AstNode<'a>>>,
) -> Vec<ExpValue> {
    let mut res: Vec<ExpValue> = Vec::new();

    for exp_node in exp.iter() {
        let tok_option = eval_node(vars, Rc::clone(&functions), scope, exp_node.as_ref());
        if let Some(tok) = tok_option {
            let val = match tok {
                EvalValue::Token(tok) => match tok.token_type {
                    TokenType::Operator(_) => ExpValue::Operator(tok.token_type),
                    _ => ExpValue::Value(value_from_token(&tok, None)),
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

    let ast_node = get_ast_node(&mut new_tokens);

    if let Some(node) = ast_node {
        AstNodeType::Bang(Box::new(node))
    } else {
        panic!("Expected value to !");
    }
}

fn create_arr_with_tokens<'a>(tokens: Vec<Token>) -> AstNodeType<'a> {
    let mut node_tokens = tokens_to_delimiter(&tokens, 0, ",");
    let mut i = node_tokens.len() + 1;

    let mut arr_values: Vec<AstNode> = Vec::new();

    while node_tokens.len() > 0 {
        let node_option = get_ast_node(&mut node_tokens);

        if let Some(node) = node_option {
            arr_values.push(node);
        }

        node_tokens = tokens_to_delimiter(&tokens, i, ",");
        i += node_tokens.len() + 1;
    }

    let res = AstNodeType::Array(arr_values);

    res
}

pub fn create_arr<'a>(tokens: Vec<Token>) -> AstNodeType<'a> {
    let mut new_tokens: Vec<Token> = Vec::new();
    for i in 1..tokens.len() - 1 {
        new_tokens.push(tokens[i].clone());
    }

    if new_tokens.len() == 0 {
        return AstNodeType::Array(vec![]);
    }

    match new_tokens[0].token_type {
        TokenType::LBracket => create_arr_with_tokens(new_tokens),
        TokenType::Number => create_arr_with_tokens(new_tokens),
        TokenType::String => create_arr_with_tokens(new_tokens),
        TokenType::Bool => create_arr_with_tokens(new_tokens),
        TokenType::Identifier => create_arr_with_tokens(new_tokens),
        _ => panic!("Unexpected token: {}", new_tokens[0].value),
    }
}

pub fn ensure_type<'a>(var_type: &'a VarType, val: &'a Value) -> bool {
    match var_type {
        VarType::Unknown => return true,
        _ => {}
    }

    match val {
        Value::Usize(_) => match var_type {
            VarType::Usize => true,
            _ => false,
        },
        Value::String(_) => match var_type {
            VarType::String => true,
            _ => false,
        },
        Value::Int(_) => match var_type {
            VarType::Int => true,
            _ => false,
        },
        Value::Float(_) => match var_type {
            VarType::Float => true,
            _ => false,
        },
        Value::Double(_) => match var_type {
            VarType::Double => true,
            _ => false,
        },
        Value::Long(_) => match var_type {
            VarType::Long => true,
            _ => false,
        },
        Value::Bool(_) => match var_type {
            VarType::Bool => true,
            _ => false,
        },
        Value::Array(arr) => match var_type {
            VarType::Array(arr_type) => {
                let mut res = true;
                for item in arr.iter() {
                    if !ensure_type(arr_type.as_ref(), item) {
                        res = false;
                        break;
                    }
                }
                res
            }
            _ => false,
        },
    }
}

pub fn get_eval_value(val: EvalValue) -> Value {
    match val {
        EvalValue::Value(v) => v,
        EvalValue::Token(t) => match t.token_type {
            TokenType::Number => value_from_token(&t, None),
            TokenType::String => value_from_token(&t, None),
            TokenType::Bool => value_from_token(&t, None),
            TokenType::Identifier => value_from_token(&t, None),
            _ => panic!("Unexpected token: {}", t.value),
        },
    }
}

pub fn push_to_array<'a>(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    scope: usize,
    arr: &mut Vec<Value>,
    args: &Vec<AstNode<'a>>,
) {
    let arr_item_type = get_array_type(&arr);

    for arg in args.iter() {
        let arg_res_option = eval_node(vars, Rc::clone(&functions), scope, arg);

        if let Some(arg_res) = arg_res_option {
            let val = get_eval_value(arg_res);
            if !ensure_type(&arr_item_type, &val) {
                panic!(
                    "Error pushing to array, expected type: {:?} found {:?}",
                    arr_item_type,
                    get_var_type_from_value(&val),
                );
            }
            arr.push(val);
        }
    }
}

fn get_var_type_from_value(val: &Value) -> VarType {
    match val {
        Value::Usize(_) => VarType::Usize,
        Value::String(_) => VarType::String,
        Value::Int(_) => VarType::Int,
        Value::Float(_) => VarType::Float,
        Value::Double(_) => VarType::Double,
        Value::Long(_) => VarType::Long,
        Value::Bool(_) => VarType::Bool,
        Value::Array(vals) => VarType::Array(Box::new(get_array_type(&vals))),
    }
}

pub fn get_array_type(values: &Vec<Value>) -> VarType {
    if values.len() > 0 {
        get_var_type_from_value(&values[0])
    } else {
        VarType::Unknown
    }
}

pub fn create_cast_node<'a>(tokens: Vec<Token>) -> AstNodeType<'a> {
    let node_tokens = tokens_to_delimiter(&tokens, 2, ")");
    let node_option = get_ast_node(&node_tokens);

    if let Some(node) = node_option {
        AstNodeType::Cast(VarType::from(tokens[0].value.as_str()), Box::new(node))
    } else {
        panic!("")
    }
}

pub fn cast(to_type: VarType, val: Value) -> Value {
    match val {
        Value::Usize(v) => match to_type {
            VarType::Usize => val,
            VarType::Int => Value::Int(v as i32),
            VarType::Float => Value::Float(v as f32),
            VarType::Double => Value::Double(v as f64),
            VarType::Long => Value::Long(v as i64),
            VarType::String => Value::String(v.to_string()),
            VarType::Bool => {
                if v == 0 {
                    Value::Bool(false)
                } else if v == 1 {
                    Value::Bool(true)
                } else {
                    panic!("Cannot convert usize to bool")
                }
            }
            VarType::Array(_) => panic!("Cannot convert usize to array"),
            VarType::Unknown => panic!("Cannot convert unknown"),
        },
        Value::String(v) => match to_type {
            VarType::Usize => {
                Value::Usize(v.parse::<usize>().expect("Error parsing string to usize"))
            }
            VarType::Int => Value::Int(v.parse::<i32>().expect("Error parsing string to int")),
            VarType::Float => {
                Value::Float(v.parse::<f32>().expect("Error parsing string to float"))
            }
            VarType::Double => {
                Value::Double(v.parse::<f64>().expect("Error parsing string to double"))
            }
            VarType::Long => Value::Long(v.parse::<i64>().expect("Error parsing string to long")),
            VarType::String => Value::String(v),
            VarType::Bool => Value::Bool(v.parse::<bool>().expect("Error parsing string to bool")),
            VarType::Array(_) => panic!("Cannot convert usize to array"),
            VarType::Unknown => panic!("Cannot convert unknown"),
        },
        Value::Int(v) => match to_type {
            VarType::Usize => Value::Usize(v as usize),
            VarType::Int => val,
            VarType::Float => Value::Float(v as f32),
            VarType::Double => Value::Double(v as f64),
            VarType::Long => Value::Long(v as i64),
            VarType::String => Value::String(v.to_string()),
            VarType::Bool => {
                if v == 0 {
                    Value::Bool(false)
                } else if v == 1 {
                    Value::Bool(true)
                } else {
                    panic!("Cannot convert int to bool")
                }
            }
            VarType::Array(_) => panic!("Cannot convert int to array"),
            VarType::Unknown => panic!("Cannot convert unknown"),
        },
        Value::Float(v) => match to_type {
            VarType::Usize => Value::Usize(v as usize),
            VarType::Int => Value::Int(v as i32),
            VarType::Float => val,
            VarType::Double => Value::Double(v as f64),
            VarType::Long => Value::Long(v as i64),
            VarType::String => Value::String(v.to_string()),
            VarType::Bool => {
                if v == 0.0 {
                    Value::Bool(false)
                } else if v == 1.0 {
                    Value::Bool(true)
                } else {
                    panic!("Cannot convert float to bool")
                }
            }
            VarType::Array(_) => panic!("Cannot convert float to array"),
            VarType::Unknown => panic!("Cannot convert unknown"),
        },
        Value::Double(v) => match to_type {
            VarType::Usize => Value::Usize(v as usize),
            VarType::Int => Value::Int(v as i32),
            VarType::Float => Value::Float(v as f32),
            VarType::Double => val,
            VarType::Long => Value::Long(v as i64),
            VarType::String => Value::String(v.to_string()),
            VarType::Bool => {
                if v == 0.0 {
                    Value::Bool(false)
                } else if v == 1.0 {
                    Value::Bool(true)
                } else {
                    panic!("Cannot convert double to bool")
                }
            }
            VarType::Array(_) => panic!("Cannot convert double to array"),
            VarType::Unknown => panic!("Cannot convert unknown"),
        },
        Value::Long(v) => match to_type {
            VarType::Usize => Value::Usize(v as usize),
            VarType::Int => Value::Int(v as i32),
            VarType::Float => Value::Float(v as f32),
            VarType::Double => Value::Double(v as f64),
            VarType::Long => val,
            VarType::String => Value::String(v.to_string()),
            VarType::Bool => {
                if v == 0 {
                    Value::Bool(false)
                } else if v == 1 {
                    Value::Bool(true)
                } else {
                    panic!("Cannot convert long to bool")
                }
            }
            VarType::Array(_) => panic!("Cannot convert long to array"),
            VarType::Unknown => panic!("Cannot convert unknown"),
        },
        Value::Bool(v) => {
            let num_val = if v { 1 } else { 0 };
            match to_type {
                VarType::Usize => Value::Usize(num_val),
                VarType::Int => Value::Int(num_val as i32),
                VarType::Float => Value::Float(num_val as f32),
                VarType::Double => Value::Double(num_val as f64),
                VarType::Long => Value::Long(num_val as i64),
                VarType::String => Value::String(v.to_string()),
                VarType::Bool => val,
                VarType::Array(_) => panic!("Cannot convert bool to array"),
                VarType::Unknown => panic!("Cannot convert unknown"),
            }
        }
        Value::Array(arr) => match to_type {
            VarType::Usize => panic!("Cannot convert array to usize"),
            VarType::Int => panic!("Cannot convert array to int"),
            VarType::Float => panic!("Cannot convert array to float"),
            VarType::Double => panic!("Cannot convert array to double"),
            VarType::Long => panic!("Cannot convert array to long"),
            VarType::String => Value::String(get_value_arr_str(&arr)),
            VarType::Bool => panic!("Cannot convert array to bool"),
            VarType::Array(_) => unimplemented!(),
            VarType::Unknown => panic!("Cannot convert unknown"),
        },
    }
}

pub fn get_struct_access_tokens(tokens: Vec<Token>) -> Vec<Vec<Token>> {
    let mut res: Vec<Vec<Token>> = vec![vec![]];
    let mut i = 0;

    let mut open_brackets = 0;
    while i < tokens.len() {
        if match tokens[i].token_type {
            TokenType::LParen => true,
            TokenType::LBrace => true,
            TokenType::LBracket => true,
            _ => false,
        } {
            open_brackets += 1;
        } else if match tokens[i].token_type {
            TokenType::RParen => true,
            TokenType::RBrace => true,
            TokenType::RBracket => true,
            _ => false,
        } {
            open_brackets -= 1;
        }

        if open_brackets == 0
            && match tokens[i].token_type {
                TokenType::Period => {
                    res.push(vec![]);
                    false
                }
                TokenType::Identifier => false,
                TokenType::LParen => false,
                _ => true,
            }
        {
            let index = res.len() - 1;
            res[index].push(tokens[i].clone());
            if i + 1 < tokens.len()
                && match tokens[i + 1].token_type {
                    TokenType::Semicolon => true,
                    _ => false,
                }
            {
                res[index].push(tokens[i + 1].clone());
            }
            break;
        } else {
            let index = res.len() - 1;
            res[index].push(tokens[i].clone());
        }

        i += 1;
    }

    res
}

pub fn is_sequence(tokens: &Vec<Token>) -> bool {
    let mut open_brackets = 0;
    let mut semicolons = 0;

    for i in 0..tokens.len() {
        match tokens[i].token_type {
            TokenType::LParen => {
                open_brackets += 1;
            }
            TokenType::RParen => {
                open_brackets -= 1;
            }
            TokenType::LBrace => {
                open_brackets += 1;
            }
            TokenType::RBrace => {
                open_brackets -= 1;
            }
            TokenType::LBracket => {
                open_brackets += 1;
            }
            TokenType::RBracket => {
                open_brackets -= 1;
            }
            TokenType::Semicolon => {
                if i < tokens.len() - 1 && open_brackets == 0 {
                    semicolons += 1;
                }
            }
            _ => {}
        }
    }

    // <= 1 because sequences with 1 semicolon can be
    // seen as single lines
    open_brackets == 0 && semicolons > 0
}
