use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{
    ast::{get_ast_node, get_value_arr_str, AstNode, AstNodeType, Value},
    interpreter::{eval_node, get_var_ptr, value_from_token, EvalValue, Func, VarType, VarValue},
    tokenizer::Token,
};

pub fn create_make_var_node<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> AstNodeType<'a> {
    let mut var_type = if let Token::Type(t) = tokens[0].take().unwrap() {
        t
    } else {
        panic!("Expected identifier for variable name");
    };

    let mut i = 0;
    while match tokens[i + 1].as_ref().unwrap() {
        Token::LBracket => true,
        _ => false,
    } {
        var_type = VarType::Array(Box::new(var_type));

        i += 2;
    }

    let eq_pos = tokens
        .iter()
        .position(|t| {
            if t.is_some() {
                match t.as_ref().unwrap() {
                    Token::EqSet => true,
                    _ => false,
                }
            } else {
                false
            }
        })
        .unwrap();

    let mut value_to_set = tokens_to_delimiter(tokens, eq_pos + 1, ";");

    let node = get_ast_node(&mut value_to_set);
    if node.is_none() {
        panic!("Invalid value, expected value to set variable");
    }

    AstNodeType::MakeVar(
        var_type,
        tokens[eq_pos - 1].take().unwrap(),
        Box::new(node.unwrap()),
    )
}

pub fn create_set_var_node<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> AstNodeType<'a> {
    let mut value_to_set = tokens_to_delimiter(tokens, 2, ";");
    let node = get_ast_node(&mut value_to_set);
    if node.is_none() {
        panic!("Invalid value, expected value to set variable");
    }
    AstNodeType::SetVar(tokens[0].take().unwrap(), Box::new(node.unwrap()))
}

pub fn create_keyword_node<'a>(
    tokens: &mut Vec<Option<Token<'a>>>,
    keyword: &str,
) -> AstNodeType<'a> {
    match keyword {
        "if" => {
            let mut condition = tokens_to_delimiter(tokens, 2, ")");
            let condition_len = condition.len();
            let condition_node = get_ast_node(&mut condition);

            // + 4 because tokens: [if, (, ), {]
            let mut tokens_to_run = tokens_to_delimiter(tokens, condition_len + 4, "}");
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
        _ => panic!("Unexpected keyword: {:?}", tokens[0].as_ref().unwrap()),
    }
}

fn get_params<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> Vec<AstNode<'a>> {
    let mut arg_nodes: Vec<AstNode> = Vec::new();
    let mut temp_tokens: Vec<Option<Token>> = Vec::new();
    let mut i = 2;

    let mut num_open_parens = 0;

    while i < tokens.len() {
        match tokens[i].as_ref().unwrap() {
            Token::RParen => {
                num_open_parens -= 1;
            }
            Token::LParen => {
                num_open_parens += 1;
            }
            _ => {}
        }

        if num_open_parens < 0 {
            break;
        }

        match tokens[i].as_ref().unwrap() {
            Token::Comma => {
                let arg_node_option = get_ast_node(&mut temp_tokens);
                if let Some(arg_node) = arg_node_option {
                    arg_nodes.push(arg_node);
                } else {
                    panic!("Error in function parameters");
                }
                temp_tokens = Vec::new();
            }
            _ => {
                temp_tokens.push(Some(tokens[i].take().unwrap()));
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

pub fn create_func_call_node<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> AstNodeType<'a> {
    let params = get_params(tokens);

    let func_name = if let Token::Identifier(ident) = tokens[0].take().unwrap() {
        ident
    } else {
        panic!("Expected identifier for func name");
    };

    AstNodeType::CallFunc(func_name, params)
}

pub fn set_var_value<'a>(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    name: String,
    value: Value,
) {
    let res_option = vars.get(&name);
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
    tokens: &mut Vec<Option<Token<'a>>>,
    start: usize,
    delimiter: &'a str,
) -> Vec<Option<Token<'a>>> {
    let mut res = vec![];

    let mut open_brackets = 0;
    for i in start..tokens.len() {
        if match tokens[i].as_ref().unwrap() {
            Token::LParen => true,
            Token::LBrace => true,
            Token::LBracket => true,
            _ => false,
        } {
            open_brackets += 1;
        } else if match tokens[i].as_ref().unwrap() {
            Token::RParen => true,
            Token::RBrace => true,
            Token::RBracket => true,
            _ => false,
        } {
            open_brackets -= 1;
        }

        if tokens[i].as_ref().unwrap().get_str() != delimiter || open_brackets > 0 {
            res.push(Some(tokens[i].take().unwrap()));
        } else {
            return res;
        }
    }

    res
}

pub fn tokens_to_operator<'a>(
    tokens: &mut Vec<Option<Token<'a>>>,
    start: usize,
) -> Vec<Option<Token<'a>>> {
    let mut res = vec![];

    let mut open_brackets = 0;
    for i in start..tokens.len() {
        if match tokens[i].as_ref().unwrap() {
            Token::LParen => true,
            Token::LBrace => true,
            Token::LBracket => true,
            _ => false,
        } {
            open_brackets += 1;
        } else if match tokens[i].as_ref().unwrap() {
            Token::RParen => true,
            Token::RBrace => true,
            Token::RBracket => true,
            _ => false,
        } {
            open_brackets -= 1;
        }
        if match tokens[i].as_ref().unwrap() {
            Token::Operator(_) => false,
            _ => true,
        } || open_brackets > 0
        {
            res.push(Some(tokens[i].take().unwrap()));
        } else {
            return res;
        }
    }

    res
}

pub fn get_exp_node<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> Vec<Box<AstNode<'a>>> {
    let mut res: Vec<Box<AstNode>> = Vec::new();

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i].as_ref().unwrap() {
            Token::Operator(_) => {
                let node = AstNode::new(AstNodeType::Token(tokens[i].take().unwrap()));
                res.push(Box::new(node));
                i += 1;
                continue;
            }
            _ => {}
        }

        let mut to_op = tokens_to_operator(tokens, i);
        println!("to op: {:?}", to_op);

        if to_op.len() > 0 {
            if match to_op[0].as_ref().unwrap() {
                Token::LParen => true,
                _ => false,
            } {
                let slice = &to_op[1..to_op.len() - 1];
                let mut slice: Vec<Option<Token>> = slice.iter().map(|t| t.to_owned()).collect();
                let exp_nodes = get_exp_node(&mut slice);
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

pub fn is_exp(tokens: &mut Vec<Option<Token>>) -> bool {
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i].is_none() {
            i += 1;
            continue;
        }

        match tokens[i].as_ref().unwrap() {
            Token::LParen => {
                while i < tokens.len()
                    && match tokens[i].as_ref().unwrap() {
                        Token::RParen => false,
                        _ => true,
                    }
                {
                    i += 1;
                }
            }
            Token::LBrace => {
                while i < tokens.len()
                    && match tokens[i].as_ref().unwrap() {
                        Token::RBrace => false,
                        _ => true,
                    }
                {
                    i += 1;
                }
            }
            Token::LBracket => {
                while i < tokens.len()
                    && match tokens[i].as_ref().unwrap() {
                        Token::RBracket => false,
                        _ => true,
                    }
                {
                    i += 1;
                }
            }
            Token::Operator(_) => return true,
            _ => {}
        }

        i += 1;
    }

    false
}

#[derive(Clone, Debug)]
pub enum ExpValue<'a> {
    Value(Value),
    Operator(Token<'a>),
}

pub fn flatten_exp<'a>(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    scope: usize,
    exp: &Vec<Box<AstNode<'a>>>,
) -> Vec<Option<ExpValue<'a>>> {
    let mut res = vec![];

    for exp_node in exp.iter() {
        let tok_option = eval_node(vars, Rc::clone(&functions), scope, exp_node.as_ref());
        if let Some(tok) = tok_option {
            let val = match tok {
                EvalValue::Token(tok) => match tok {
                    Token::Operator(_) => ExpValue::Operator(tok),
                    Token::Identifier(ident) => {
                        let var_ptr = get_var_ptr(vars, &ident);
                        let var_ref = var_ptr.borrow();
                        let var_value = var_ref.value.clone();

                        ExpValue::Value(var_value)
                    }
                    _ => ExpValue::Value(value_from_token(&tok, None)),
                },
                EvalValue::Value(val) => ExpValue::Value(val),
            };
            res.push(Some(val));
        }
    }

    res
}

pub fn create_bang_bool<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> AstNodeType<'a> {
    tokens.remove(0);

    let ast_node = get_ast_node(tokens);

    if let Some(node) = ast_node {
        AstNodeType::Bang(Box::new(node))
    } else {
        panic!("Expected value to !");
    }
}

fn create_arr_with_tokens<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> AstNodeType<'a> {
    let mut node_tokens = tokens_to_delimiter(tokens, 0, ",");
    let mut i = node_tokens.len() + 1;

    let mut arr_values: Vec<AstNode> = Vec::new();

    while node_tokens.len() > 0 {
        let node_option = get_ast_node(&mut node_tokens);

        if let Some(node) = node_option {
            arr_values.push(node);
        }

        node_tokens = tokens_to_delimiter(tokens, i, ",");
        i += node_tokens.len() + 1;
    }

    let res = AstNodeType::Array(arr_values);

    res
}

pub fn create_arr<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> AstNodeType<'a> {
    let mut new_tokens: Vec<Option<Token>> = Vec::new();
    for i in 1..tokens.len() - 1 {
        new_tokens.push(Some(tokens[i].take().unwrap()));
    }

    if new_tokens.len() == 0 {
        return AstNodeType::Array(vec![]);
    }

    match new_tokens[0].as_ref().unwrap() {
        Token::LBracket => create_arr_with_tokens(&mut new_tokens),
        Token::Number(_) => create_arr_with_tokens(&mut new_tokens),
        Token::String(_) => create_arr_with_tokens(&mut new_tokens),
        Token::Bool(_) => create_arr_with_tokens(&mut new_tokens),
        Token::Identifier(_) => create_arr_with_tokens(&mut new_tokens),
        _ => panic!("Unexpected token: {:?}", new_tokens[0]),
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
                if arr.len() == 0 {
                    return true;
                }

                ensure_type(arr_type.as_ref(), &arr[0])
            }
            _ => false,
        },
    }
}

pub fn get_eval_value(vars: &mut HashMap<String, Rc<RefCell<VarValue>>>, val: EvalValue) -> Value {
    match val {
        EvalValue::Value(v) => v,
        EvalValue::Token(t) => match t {
            Token::Number(_) => value_from_token(&t, None),
            Token::String(_) => value_from_token(&t, None),
            Token::Bool(_) => value_from_token(&t, None),
            Token::Identifier(ident) => {
                let var_ptr = get_var_ptr(vars, &ident);
                let var_ref = var_ptr.borrow();
                var_ref.value.clone()
            }
            _ => panic!("Unexpected token: {:?}", t),
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
            let val = get_eval_value(vars, arg_res);
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

pub fn create_cast_node<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> AstNodeType<'a> {
    let mut node_tokens = tokens_to_delimiter(tokens, 2, ")");
    let node_option = get_ast_node(&mut node_tokens);

    let to_type = match tokens[0].as_ref().unwrap() {
        Token::Type(t) => t.get_str(),
        _ => panic!("Cannot cast to: {:?}", tokens[0].as_ref().unwrap()),
    };

    if let Some(node) = node_option {
        AstNodeType::Cast(VarType::from(to_type), Box::new(node))
    } else {
        panic!("")
    }
}

pub fn cast(to_type: &VarType, val: Value) -> Value {
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

pub fn get_struct_access_tokens<'a>(
    tokens: &mut Vec<Option<Token<'a>>>,
) -> Vec<Vec<Option<Token<'a>>>> {
    let mut res = vec![vec![]];
    let mut i = 0;

    let mut open_brackets = 0;
    while i < tokens.len() {
        if match tokens[i].as_ref().unwrap() {
            Token::LParen => true,
            Token::LBrace => true,
            Token::LBracket => true,
            _ => false,
        } {
            open_brackets += 1;
        } else if match tokens[i].as_ref().unwrap() {
            Token::RParen => true,
            Token::RBrace => true,
            Token::RBracket => true,
            _ => false,
        } {
            open_brackets -= 1;
        }

        if open_brackets == 0
            && match tokens[i].as_ref().unwrap() {
                Token::Period => {
                    res.push(vec![]);
                    false
                }
                Token::Identifier(_) => false,
                Token::LParen => false,
                _ => true,
            }
        {
            let index = res.len() - 1;
            res[index].push(Some(tokens[i].take().unwrap()));
            if i + 1 < tokens.len()
                && match tokens[i + 1].as_ref().unwrap() {
                    Token::Semicolon => true,
                    _ => false,
                }
            {
                res[index].push(Some(tokens[i + 1].take().unwrap()));
            }
            break;
        } else {
            let index = res.len() - 1;
            res[index].push(Some(tokens[i].take().unwrap()));
        }

        i += 1;
    }

    res
}

pub fn is_sequence(tokens: &mut Vec<Option<Token>>) -> bool {
    let mut open_brackets = 0;
    let mut semicolons = 0;

    for i in 0..tokens.len() {
        match tokens[i].as_ref().unwrap() {
            Token::LParen => {
                open_brackets += 1;
            }
            Token::RParen => {
                open_brackets -= 1;
            }
            Token::LBrace => {
                open_brackets += 1;
            }
            Token::RBrace => {
                open_brackets -= 1;
            }
            Token::LBracket => {
                open_brackets += 1;
            }
            Token::RBracket => {
                open_brackets -= 1;
            }
            Token::Semicolon => {
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

pub fn create_comp_node<'a>(tokens: &mut Vec<Option<Token<'a>>>) -> Option<AstNodeType<'a>> {
    // TODO
    // check for cases like this: if ((i < 10)) { ... }
    // should be treated like: if (i < 10) { ... }
    let mut temp_tokens = vec![];
    let mut open_parens = 0;

    for i in 0..tokens.len() {
        match tokens[i].as_ref().unwrap() {
            Token::LParen => open_parens += 1,
            Token::RParen => open_parens -= 1,
            Token::LBracket => open_parens += 1,
            Token::RBracket => open_parens -= 1,
            Token::LBrace => open_parens += 1,
            Token::RBrace => open_parens -= 1,
            _ => {}
        }

        if match tokens[i].as_ref().unwrap() {
            Token::EqCompare => true,
            Token::EqNCompare => true,
            Token::LAngle => true,
            Token::RAngle => true,
            Token::LAngleEq => true,
            Token::RAngleEq => true,
            _ => false,
        } && open_parens == 0
        {
            for j in 0..i {
                temp_tokens.push(Some(tokens[j].take().unwrap()));
            }

            if let Some(left_node) = get_ast_node(&mut temp_tokens) {
                temp_tokens.drain(0..temp_tokens.len());

                for j in i + 1..tokens.len() {
                    temp_tokens.push(Some(tokens[j].take().unwrap()));
                }

                if let Some(right_node) = get_ast_node(&mut temp_tokens) {
                    return Some(AstNodeType::Comparison(
                        tokens[i].take().unwrap(),
                        Box::new(left_node),
                        Box::new(right_node),
                    ));
                } else {
                    panic!("Expected expression to right of comparison operator");
                }
            } else {
                panic!("Expected expression to left of comparison operator");
            }
        }
    }

    None
}

macro_rules! comp {
    ($l:expr, $r:expr, $c:tt) => {
        match $l {
            Value::Usize(l) => match $r {
                Value::Usize(r) => l $c r,
                Value::String(_) => panic!("Cannot compare usize to string"),
                Value::Int(_) => panic!("Cannot compare usize to int"),
                Value::Float(_) => panic!("Cannot compare usize to float"),
                Value::Double(_) => panic!("Cannot compare usize to double"),
                Value::Long(_) => panic!("Cannot compare usize to long"),
                Value::Bool(_) => panic!("Cannot compare usize to bool"),
                Value::Array(_) => panic!("Cannot compare usize to array"),
            },
            Value::String(l) => match $r {
                Value::Usize(_) => panic!("Cannot compare string to usize"),
                Value::String(r) => l $c r,
                Value::Int(_) => panic!("Cannot compare string to int"),
                Value::Float(_) => panic!("Cannot compare string to float"),
                Value::Double(_) => panic!("Cannot compare string to double"),
                Value::Long(_) => panic!("Cannot compare string to long"),
                Value::Bool(_) => panic!("Cannot compare string to bool"),
                Value::Array(_) => panic!("Cannot compare string to array"),
            },
            Value::Int(l) => match $r {
                Value::Usize(_) => panic!("Cannot compare int to usize"),
                Value::String(_) => panic!("Cannot compare int to string"),
                Value::Int(r) => l $c r,
                Value::Float(_) => panic!("Cannot compare int to float"),
                Value::Double(_) => panic!("Cannot compare int to double"),
                Value::Long(_) => panic!("Cannot compare int to long"),
                Value::Bool(_) => panic!("Cannot compare int to bool"),
                Value::Array(_) => panic!("Cannot compare int to array"),
            },
            Value::Float(l) => match $r {
                Value::Usize(_) => panic!("Cannot compare float to usize"),
                Value::String(_) => panic!("Cannot compare float to string"),
                Value::Int(_) => panic!("Cannot compare float to int"),
                Value::Float(r) => l $c r,
                Value::Double(_) => panic!("Cannot compare float to double"),
                Value::Long(_) => panic!("Cannot compare float to long"),
                Value::Bool(_) => panic!("Cannot compare float to bool"),
                Value::Array(_) => panic!("Cannot compare float to array"),
            },
            Value::Double(l) => match $r {
                Value::Usize(_) => panic!("Cannot compare double to usize"),
                Value::String(_) => panic!("Cannot compare double to string"),
                Value::Int(_) => panic!("Cannot compare double to int"),
                Value::Float(_) => panic!("Cannot compare double to float"),
                Value::Double(r) => l $c r,
                Value::Long(_) => panic!("Cannot compare double to long"),
                Value::Bool(_) => panic!("Cannot compare double to bool"),
                Value::Array(_) => panic!("Cannot compare double to array"),
            },
            Value::Long(l) => match $r {
                Value::Usize(_) => panic!("Cannot compare long to usize"),
                Value::String(_) => panic!("Cannot compare long to string"),
                Value::Int(_) => panic!("Cannot compare long to int"),
                Value::Float(_) => panic!("Cannot compare long to float"),
                Value::Double(_) => panic!("Cannot compare long to double"),
                Value::Long(r) => l $c r,
                Value::Bool(_) => panic!("Cannot compare long to bool"),
                Value::Array(_) => panic!("Cannot compare long to array"),
            },
            Value::Bool(l) => match $r {
                Value::Usize(_) => panic!("Cannot compare bool to usize"),
                Value::String(_) => panic!("Cannot compare bool to string"),
                Value::Int(_) => panic!("Cannot compare bool to int"),
                Value::Float(_) => panic!("Cannot compare bool to float"),
                Value::Double(_) => panic!("Cannot compare bool to double"),
                Value::Long(_) => panic!("Cannot compare bool to long"),
                Value::Bool(r) => l $c r,
                Value::Array(_) => panic!("Cannot compare bool to array"),
            },
            Value::Array(_) => match $r {
                Value::Usize(_) => panic!("Cannot compare array to usize"),
                Value::String(_) => panic!("Cannot compare array to string"),
                Value::Int(_) => panic!("Cannot compare array to int"),
                Value::Float(_) => panic!("Cannot compare array to float"),
                Value::Double(_) => panic!("Cannot compare array to double"),
                Value::Long(_) => panic!("Cannot compare array to long"),
                Value::Bool(_) => panic!("Cannot compare array to bool"),
                Value::Array(_) => unimplemented!()
            },
        }
    };
}

macro_rules! comp_match {
    ($tok:ident, $left:expr, $right:expr) => {
        match $tok {
            Token::EqCompare => comp!($left, $right, ==),
            Token::EqNCompare => comp!($left, $right, !=),
            Token::LAngle => comp!($left, $right, <),
            Token::RAngle => comp!($left, $right, >),
            Token::LAngleEq => comp!($left, $right, <=),
            Token::RAngleEq => comp!($left, $right, >=),
            _ => panic!("Expected comparison operator"),
        }
    }
}

pub fn compare<'a>(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    left: EvalValue,
    right: EvalValue,
    comp_token: &Token,
) -> EvalValue<'a> {
    let res = match left {
        EvalValue::Value(left_val) => match right {
            EvalValue::Value(right_val) => {
                comp_match!(comp_token, left_val, right_val)
            }
            EvalValue::Token(t) => match t {
                Token::Identifier(ident) => {
                    let var_ptr = get_var_ptr(vars, &ident);
                    let var_ref = var_ptr.borrow();
                    let right_val = &var_ref.value;
                    comp_match!(comp_token, &left_val, right_val)
                }
                _ => {
                    let right_val = value_from_token(&t, None);
                    comp_match!(comp_token, left_val, right_val)
                }
            },
        },
        EvalValue::Token(t) => match t {
            Token::Identifier(ident) => {
                let var_ptr = get_var_ptr(vars, &ident);
                let var_ref = var_ptr.borrow();
                let left_val = &var_ref.value;
                match right {
                    EvalValue::Value(right_val) => {
                        comp_match!(comp_token, left_val, &right_val)
                    }
                    EvalValue::Token(t) => match t {
                        Token::Identifier(ident) => {
                            let var_ptr = get_var_ptr(vars, &ident);
                            let var_ref = var_ptr.borrow();
                            let right_val = &var_ref.value;
                            comp_match!(comp_token, &left_val, right_val)
                        }
                        _ => {
                            let right_val = value_from_token(&t, None);
                            comp_match!(comp_token, left_val, &right_val)
                        }
                    },
                }
            }
            _ => {
                let left_val = value_from_token(&t, None);

                match right {
                    EvalValue::Value(right_val) => {
                        comp_match!(comp_token, left_val, right_val)
                    }
                    EvalValue::Token(t) => match t {
                        Token::Identifier(ident) => {
                            let var_ptr = get_var_ptr(vars, &ident);
                            let var_ref = var_ptr.borrow();
                            let right_val = &var_ref.value;
                            comp_match!(comp_token, &left_val, right_val)
                        }
                        _ => {
                            let right_val = value_from_token(&t, None);
                            comp_match!(comp_token, left_val, right_val)
                        }
                    },
                }
            }
        },
    };

    EvalValue::Value(Value::Bool(res))
}
