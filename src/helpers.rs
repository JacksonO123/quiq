use std::{cell::RefCell, io::Stdout, process::exit, rc::Rc};

use crate::{
    ast::{get_ast_node, get_value_arr_str, AstNode, FuncParam, StructShape, Value},
    interpreter::{
        eval_node, get_var_ptr, value_from_token, CustomFunc, EvalValue, Func, StructInfo,
        StructProp, VarType, VarValue,
    },
    tokenizer::{Keyword, Token},
    variables::Variables,
};

pub fn make_var<'a>(
    // vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    vars: &mut Variables,
    functions: &mut Vec<Func<'a>>,
    structs: &mut StructInfo,
    scope: usize,
    var_type: &VarType,
    name: &Token,
    value: &Option<Box<AstNode>>,
    stdout: &mut Stdout,
) {
    let value = if let Some(val) = value {
        eval_node(vars, functions, structs, scope, val.as_ref(), stdout)
    } else {
        (Some(EvalValue::Value(Value::Null)), None)
    };

    if let (Some(tok), _) = value {
        let val = match tok {
            EvalValue::Token(t) => {
                if let Token::Identifier(name) = t {
                    let var = vars.get(&name, scope);
                    if let Some(var) = var {
                        let var_ref = &*var.borrow();
                        let value = &var_ref.value;
                        let val_ref = &*value.borrow();
                        if let Value::Ref(r) = val_ref {
                            Value::Ref(Rc::clone(&r))
                        } else {
                            val_ref.clone()
                        }
                    } else {
                        panic!("Unknown identifier: {}", name);
                    }
                } else {
                    value_from_token(&t, Some(var_type))
                }
            }
            EvalValue::Value(v) => match v {
                Value::Null => v,
                _ => {
                    let var_type = type_from_value(&v);
                    if compare_types(structs, &var_type, &var_type) {
                        v
                    } else {
                        panic!(
                            "Unexpected variable type definition, expected {:?} found {:?}",
                            match v {
                                Value::Array(_, arr_type) =>
                                    VarType::Array(Box::new(arr_type.clone())),
                                _ => panic!(),
                            },
                            var_type
                        )
                    }
                }
            },
        };

        let var_name = if let Token::Identifier(ident) = name {
            ident
        } else {
            panic!("Expected identifier for variable name");
        };

        let var = VarValue::new(var_name.clone(), val, var_type.clone(), scope);
        vars.insert(var_name.clone(), var);
    } else {
        panic!("Expected {} found void", var_type.get_str());
    }
}

pub fn create_make_var_node<'a>(
    structs: &mut StructInfo,
    tokens: &mut Vec<Option<Token>>,
    is_struct: bool,
) -> AstNode {
    let first_token = tokens[0].take().unwrap();
    let mut var_type = if let Token::Type(t) = first_token {
        t
    } else {
        if is_struct {
            if let Token::Identifier(ident) = first_token {
                VarType::Struct(ident.clone())
            } else {
                panic!("Expected identifier for variable name");
            }
        } else {
            let name = match first_token {
                Token::Identifier(ident) => Some(ident),
                _ => None,
            };
            if let Some(n) = name {
                if let Some(t) = get_builtin_generic(&n, structs, tokens) {
                    t
                } else {
                    panic!("Type {} is not generic", n);
                }
            } else {
                panic!("Expected identifier for variable name");
            }
        }
    };

    let mut i = 0;
    if let Token::LAngle = tokens[1].as_ref().unwrap() {
        i += 2;
        while match tokens[i].as_ref() {
            Some(v) => match v {
                Token::RAngle => false,
                _ => true,
            },
            None => true,
        } {
            i += 1;
        }
    }

    while match tokens[i + 1].as_ref().unwrap() {
        Token::LBracket => true,
        _ => false,
    } {
        var_type = VarType::Array(Box::new(var_type));

        i += 2;
    }

    let eq_pos = tokens.iter().position(|t| {
        if t.is_some() {
            match t.as_ref().unwrap() {
                Token::EqSet => true,
                _ => false,
            }
        } else {
            false
        }
    });

    if let Some(pos) = eq_pos {
        let mut value_to_set = tokens_to_delimiter(tokens, pos + 1, ";");

        let node = get_ast_node(structs, &mut value_to_set);
        if node.is_none() {
            panic!("Invalid value, expected value to set variable");
        }

        return AstNode::MakeVar(
            var_type,
            tokens[pos - 1].take().unwrap(),
            Some(Box::new(node.unwrap())),
        );
    }

    let len = tokens.len();
    return AstNode::MakeVar(var_type, tokens[len - 2].take().unwrap(), None);
}

pub fn create_set_var_node<'a>(
    structs: &mut StructInfo,
    tokens: &mut Vec<Option<Token>>,
) -> AstNode {
    let eq_pos = tokens
        .iter()
        .position(|s| {
            if let Some(tok) = s {
                match tok {
                    Token::EqSet => true,
                    _ => false,
                }
            } else {
                false
            }
        })
        .unwrap();

    let mut value_to_set = tokens_to_delimiter(tokens, eq_pos + 1, ";");
    let node = get_ast_node(structs, &mut value_to_set);
    if node.is_none() {
        panic!("Invalid value, expected value to set variable");
    }
    AstNode::SetVar(tokens[0].take().unwrap(), Box::new(node.unwrap()))
}

pub fn create_keyword_node<'a>(
    tokens: &mut Vec<Option<Token>>,
    structs: &mut StructInfo,
    keyword: Keyword,
) -> Option<AstNode> {
    match keyword {
        Keyword::If => {
            // 2 because tokens: [if, (]
            let mut condition = tokens_to_delimiter(tokens, 2, ")");
            let condition_len = condition.len();
            let condition_node = get_ast_node(structs, &mut condition);

            // + 4 because tokens: [if, (, ), {]
            let mut tokens_to_run = tokens_to_delimiter(tokens, condition_len + 4, "}");
            let to_run_node = get_ast_node(structs, &mut tokens_to_run);

            if let Some(c_node) = condition_node {
                if let Some(tr_node) = to_run_node {
                    Some(AstNode::If(Box::new(c_node), Box::new(tr_node), None))
                } else {
                    panic!("Expected block for `if`");
                }
            } else {
                panic!("Expected condition for `if`");
            }
        }
        Keyword::Else => {
            let mut block_tokens = tokens_to_delimiter(tokens, 2, "}");
            let block_ast = get_ast_node(structs, &mut block_tokens).unwrap();

            Some(AstNode::Else(Box::new(block_ast)))
        }
        Keyword::For => {
            // 2 because tokens: [for, (]
            let mut range_tokens = tokens_to_delimiter(tokens, 2, ")");

            let ident = range_tokens[0].take().unwrap();
            let mut start_tokens = tokens_to_delimiter(&mut range_tokens, 2, ";");
            let mut end_tokens =
                tokens_to_delimiter(&mut range_tokens, 3 + start_tokens.len(), ";");

            let mut inc_tokens = tokens_to_delimiter(
                &mut range_tokens,
                4 + start_tokens.len() + end_tokens.len(),
                ")",
            );

            let start_node = match get_ast_node(structs, &mut start_tokens) {
                Some(v) => v,
                None => panic!("Expected start value in for loop"),
            };
            let end_node = match get_ast_node(structs, &mut end_tokens) {
                Some(v) => v,
                None => panic!("Expected end value in for loop"),
            };
            let inc_node = if inc_tokens.len() > 0 {
                Some(Box::new(
                    get_ast_node(structs, &mut inc_tokens).expect("Expected inc value in for loop"),
                ))
            } else {
                None
            };

            let mut node_tokens = tokens_to_delimiter(tokens, range_tokens.len() + 4, "}");
            if let Some(node) = get_ast_node(structs, &mut node_tokens) {
                Some(AstNode::ForFromTo(
                    ident,
                    Box::new(start_node),
                    Box::new(end_node),
                    inc_node,
                    Box::new(node),
                ))
            } else {
                panic!("Expected block to loop");
            }
        }
        Keyword::While => {
            // [while, (, ... exp ..., ), {]
            let mut exp_tokens = tokens_to_delimiter(tokens, 2, ")");
            let exp_node = get_exp_node(structs, &mut exp_tokens);
            let mut block_tokens = tokens_to_delimiter(tokens, 5 + exp_tokens.len(), "}");
            let block_node = get_ast_node(structs, &mut block_tokens).unwrap();

            Some(AstNode::While(exp_node, Box::new(block_node)))
        }
        Keyword::Struct => {
            // [struct, name, {]
            let name_option = tokens[1].take().unwrap().clone();

            let mut shape_tokens = tokens_to_delimiter(tokens, 2, "}");

            if let Token::Identifier(ident) = name_option {
                let shape = create_struct_shape(&mut shape_tokens, structs);
                let temp_shape = structs.available_structs.get_mut(&ident).unwrap();
                *temp_shape.borrow_mut() = shape;
                None
            } else {
                panic!("Expected identifier for struct decleration name");
            }
        }
        Keyword::Func => {
            // [func, name, (, ...params..., ), return_type, {]
            let name_option = tokens[1].take().unwrap();
            let offset = if let Token::LParen = name_option {
                1
            } else {
                0
            };
            let mut param_tokens = tokens_to_delimiter(tokens, 3 - offset, ")");
            let mut return_type_tokens =
                tokens_to_delimiter(tokens, 4 - offset + param_tokens.len(), "{");

            if return_type_tokens.len() == 0 {
                panic!("Expected return type for function");
            }

            let return_type = get_type_expression(&mut return_type_tokens, structs);
            let num = 6 - offset * 2 + param_tokens.len() + return_type_tokens.len();
            let mut block_tokens = tokens_to_delimiter(tokens, num, "}");
            let block_node = get_ast_node(structs, &mut block_tokens)
                .unwrap_or_else(|| AstNode::StatementSeq(vec![]));

            let params = get_param_definitions(structs, &mut param_tokens);

            let func_name = match name_option {
                Token::Identifier(name) => name,
                Token::LParen => String::from(""),
                _ => panic!("Expected identifier for func name"),
            };

            let func = CustomFunc::new(func_name, params, return_type, block_node);
            Some(AstNode::MakeFunc(Rc::new(RefCell::new(func))))
        }
        Keyword::Return => {
            let mut val_tokens = tokens_to_delimiter(tokens, 1, ";");
            let node = get_ast_node(structs, &mut val_tokens).unwrap();
            Some(AstNode::Return(Box::new(node)))
        }
    }
}

fn create_struct_shape<'a>(
    shape: &mut Vec<Option<Token>>,
    structs: &mut StructInfo,
) -> StructShape {
    let mut struct_shape = StructShape::new();
    let mut offset = 1;
    while offset < shape.len() {
        let mut prop = tokens_to_delimiter(shape, offset, ";");

        if prop.len() == 0 {
            break;
        }

        while match prop[0].as_ref().unwrap() {
            Token::NewLine => true,
            _ => false,
        } {
            prop.remove(0).unwrap();
        }

        match prop[0].as_ref().unwrap() {
            Token::RBrace => break,
            _ => {
                let mut temp_offset = 0;

                let mut prop_type = match prop[2].clone().as_ref().unwrap() {
                    Token::Type(t) => t.clone(),
                    Token::Identifier(val) => match structs.available_structs.get(val) {
                        Some(_) => VarType::Struct(val.clone()),
                        None => {
                            let mut clone_tokens: Vec<Option<Token>> =
                                prop.clone().drain(2..).collect();
                            if let Some(t) = get_builtin_generic(val, structs, &mut clone_tokens) {
                                temp_offset = clone_tokens.len() - 1;
                                t
                            } else {
                                panic!("Unexpected struct type name {}", val)
                            }
                        }
                    },
                    _ => panic!("Expected type or identifier for struct property type"),
                };

                let mut array_offset = 3 + temp_offset;
                while array_offset < prop.len() - 1
                    && match prop[array_offset].as_ref().unwrap() {
                        Token::LBracket => true,
                        Token::Semicolon => false,
                        _ => panic!(
                            "Unexpected token: {:?}",
                            prop[array_offset].as_ref().unwrap()
                        ),
                    }
                {
                    prop_type = VarType::Array(Box::new(prop_type));
                    array_offset += 2;
                }

                if let Token::Identifier(ident) = prop[0].take().unwrap() {
                    struct_shape.add(ident, prop_type);
                } else {
                    panic!("Expected struct property to be an identifier");
                }
            }
        }

        offset += 2 + prop.len();
    }

    struct_shape
}

pub fn get_param_definitions(
    structs: &mut StructInfo,
    tokens: &mut Vec<Option<Token>>,
) -> Vec<FuncParam> {
    let mut res = vec![];

    let mut i = 0;
    while i < tokens.len() {
        let mut param_definition = tokens_to_delimiter(tokens, i, ",");
        i += param_definition.len() + 1;
        let name_option = param_definition.remove(param_definition.len() - 1);
        let param_type = get_type_expression(&mut param_definition, structs);
        if let Token::Identifier(name) = name_option.unwrap() {
            let param = FuncParam::new(name, param_type);
            res.push(param);
        } else {
            panic!("Expected identifier for parameter name");
        }
    }

    res
}

fn get_params<'a>(structs: &mut StructInfo, tokens: &mut Vec<Option<Token>>) -> Vec<AstNode> {
    let mut arg_nodes: Vec<AstNode> = Vec::new();
    let mut temp_tokens: Vec<Option<Token>> = Vec::new();
    let mut i = 2;

    let mut num_open_parens = 0;

    while i < tokens.len() {
        match tokens[i].as_ref().unwrap() {
            Token::RBracket => num_open_parens -= 1,
            Token::LBracket => num_open_parens += 1,
            Token::RBrace => num_open_parens -= 1,
            Token::LBrace => num_open_parens += 1,
            Token::RParen => num_open_parens -= 1,
            Token::LParen => num_open_parens += 1,
            _ => {}
        }

        if num_open_parens < 0 {
            break;
        }

        if num_open_parens > 0 {
            temp_tokens.push(Some(tokens[i].take().unwrap()));
            i += 1;
            continue;
        }

        match tokens[i].as_ref().unwrap() {
            Token::Comma => {
                let arg_node_option = get_ast_node(structs, &mut temp_tokens);
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
        let arg_node_option = get_ast_node(structs, &mut temp_tokens);
        if let Some(arg_node) = arg_node_option {
            arg_nodes.push(arg_node);
        } else {
            panic!("Error in function parameters");
        }
    }

    arg_nodes
}

pub fn create_func_call_node<'a>(
    structs: &mut StructInfo,
    tokens: &mut Vec<Option<Token>>,
) -> AstNode {
    let params = get_params(structs, tokens);

    let func_name = if let Token::Identifier(ident) = tokens[0].take().unwrap() {
        ident
    } else {
        panic!("Expected identifier for func name");
    };

    AstNode::CallFunc(func_name, params)
}

pub fn set_var_value<'a>(
    vars: &mut Variables,
    structs: &mut StructInfo,
    name: String,
    value: Value,
    scope: usize,
) {
    let res_option = vars.get(&name, scope);
    if let Some(var_info) = res_option {
        let var_ref = &*var_info.borrow();
        let var_value = &var_ref.value;
        let new_value_type = type_from_value(&value);
        match &var_ref.var_type {
            VarType::Ref(inner_type) => {
                if compare_types(structs, inner_type.as_ref(), &new_value_type) {
                    if let Value::Ref(ref mut inner_val) = &mut *var_value.borrow_mut() {
                        *inner_val.borrow_mut() = value;
                    }
                } else {
                    panic!(
                        "expected type: {:?} found {:?}",
                        new_value_type,
                        type_from_value(&value)
                    );
                }
            }
            _ => {
                if compare_types(structs, &var_ref.var_type, &new_value_type) {
                    *var_value.borrow_mut() = value;
                } else {
                    panic!(
                        "expected type: {:?} found {:?}",
                        new_value_type,
                        type_from_value(&value)
                    );
                }
            }
        }
    } else {
        panic!("Cannot set value of undefined variable: {}", name);
    }
}

pub fn tokens_to_delimiter<'a>(
    tokens: &mut Vec<Option<Token>>,
    start: usize,
    delimiter: &'a str,
) -> Vec<Option<Token>> {
    let mut res = vec![];
    let mut has_unclosed_langle = false;

    let mut open_brackets = 0;
    let mut i = start;
    while i < tokens.len() {
        if tokens[i].as_ref().unwrap().get_str() == delimiter && open_brackets == 0 {
            return res;
        } else {
            if match tokens[i].as_ref().unwrap() {
                Token::LParen => true,
                Token::LBrace => true,
                Token::LBracket => true,
                Token::LAngle => {
                    has_unclosed_langle = true;
                    true
                }
                _ => false,
            } {
                open_brackets += 1;
            } else if match tokens[i].as_ref().unwrap() {
                Token::RParen => true,
                Token::RBrace => true,
                Token::RBracket => true,
                Token::RAngle => {
                    has_unclosed_langle = false;
                    true
                }
                _ => false,
            } {
                open_brackets -= 1;
                if has_unclosed_langle {
                    has_unclosed_langle = false;
                    continue;
                }
            }

            res.push(Some(tokens[i].take().unwrap()));
        }

        i += 1;
    }

    res
}

pub fn tokens_to_operator<'a>(tokens: &mut Vec<Option<Token>>, start: usize) -> Vec<Option<Token>> {
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

pub fn get_exp_node<'a>(
    structs: &mut StructInfo,
    tokens: &mut Vec<Option<Token>>,
) -> Vec<Box<AstNode>> {
    let mut res: Vec<Box<AstNode>> = Vec::new();

    let mut i = 0;
    while i < tokens.len() {
        if tokens[i].as_ref().is_none() {
            i += 1;
            continue;
        }

        match tokens[i].as_ref().unwrap() {
            Token::Operator(_) => {
                let node = AstNode::Token(tokens[i].take().unwrap());
                res.push(Box::new(node));
                i += 1;
                continue;
            }
            _ => {}
        }

        let mut to_op = tokens_to_operator(tokens, i);

        if to_op.len() > 0 {
            if match to_op[0].as_ref().unwrap() {
                Token::LParen => true,
                _ => false,
            } {
                let slice = &to_op[1..to_op.len() - 1];
                let mut slice: Vec<Option<Token>> = slice.iter().map(|t| t.to_owned()).collect();
                let exp_nodes = get_exp_node(structs, &mut slice);
                let node = AstNode::Exp(exp_nodes);
                res.push(Box::new(node));
            } else {
                let node_option = get_ast_node(structs, &mut to_op);
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
    let mut open_paren = 0;
    let mut has_operator = false;
    let mut has_eq_set = false;

    let mut i = 0;
    while i < tokens.len() {
        if tokens[i].is_none() {
            i += 1;
            continue;
        }

        match tokens[i].as_ref().unwrap() {
            Token::LParen => open_paren += 1,
            Token::RParen => open_paren -= 1,
            Token::LBrace => open_paren += 1,
            Token::RBrace => open_paren -= 1,
            Token::LBracket => open_paren += 1,
            Token::RBracket => open_paren -= 1,
            Token::Operator(_) => {
                if open_paren == 0 && i > 0 && i < tokens.len() - 1 {
                    has_operator = true;
                }
            }
            Token::EqSet => {
                if open_paren == 0 {
                    has_eq_set = true;
                }
            }
            _ => {}
        }

        i += 1;
    }

    has_operator && !has_eq_set
}

#[derive(Clone, Debug)]
pub enum ExpValue {
    Value(Value),
    Operator(Token),
}

pub fn flatten_exp<'a>(
    // vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    vars: &mut Variables,
    functions: &mut Vec<Func<'a>>,
    structs: &mut StructInfo,
    scope: usize,
    exp: &Vec<Box<AstNode>>,
    stdout: &mut Stdout,
) -> Vec<Option<ExpValue>> {
    let mut res = vec![];

    for exp_node in exp.iter() {
        let tok_option = eval_node(vars, functions, structs, scope, exp_node.as_ref(), stdout);
        if let (Some(tok), _) = tok_option {
            let val = match tok {
                EvalValue::Token(tok) => match tok {
                    Token::Operator(_) => ExpValue::Operator(tok),
                    Token::Identifier(ident) => {
                        let var_ptr = get_var_ptr(vars, &ident, scope);
                        let var_ref = var_ptr.borrow();
                        let var_value_borrow = var_ref.value.borrow();

                        if let Value::Ref(r) = &*var_value_borrow {
                            ExpValue::Value(Value::Ref(Rc::clone(r)))
                        } else {
                            ExpValue::Value(var_value_borrow.clone())
                        }
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

pub fn create_bang_bool<'a>(structs: &mut StructInfo, tokens: &mut Vec<Option<Token>>) -> AstNode {
    tokens.remove(0);

    let ast_node = get_ast_node(structs, tokens);

    if let Some(node) = ast_node {
        AstNode::Bang(Box::new(node))
    } else {
        panic!("Expected value to !");
    }
}

pub fn get_type_expression(tokens: &mut Vec<Option<Token>>, structs: &mut StructInfo) -> VarType {
    let first_token = tokens[0].take().unwrap();
    let first_type = if let Token::Type(t) = first_token {
        t
    } else {
        match first_token {
            Token::Identifier(ident) => match structs.available_structs.get(&ident) {
                Some(_) => VarType::Struct(ident.clone()),
                None => {
                    let generic_type = get_builtin_generic(&ident, structs, tokens);
                    generic_type.expect("Expected type or struct name for type expression")
                }
            },
            Token::Null => VarType::Null,
            _ => panic!("Expected type or struct name for type expression"),
        }
    };
    if tokens.len() == 1 {
        first_type
    } else {
        let mut index = 1;

        let mut arr_type = first_type;
        while index < tokens.len()
            && match tokens[index].as_ref().unwrap() {
                Token::LBracket => true,
                _ => false,
            }
        {
            arr_type = VarType::Array(Box::new(arr_type));
            index += 1;
        }

        arr_type
    }
}

pub fn create_arr<'a>(
    structs: &mut StructInfo,
    tokens: &mut Vec<Option<Token>>,
    arr_type: VarType,
) -> AstNode {
    // 3 because [[, ]]
    if tokens.len() == 2 {
        return AstNode::Array(vec![], arr_type);
    }

    tokens.remove(tokens.len() - 1);

    let mut node_arr: Vec<AstNode> = Vec::new();
    let mut start = 1;
    while start < tokens.len() {
        while start < tokens.len() && tokens[start].as_ref().is_none() {
            start += 1;
        }

        let mut item_tokens = tokens_to_delimiter(tokens, start, ",");

        if let Some(node) = get_ast_node(structs, &mut item_tokens) {
            node_arr.push(node);
        };

        start += item_tokens.len() + 1;
    }

    AstNode::Array(node_arr, arr_type)
}

pub fn compare_types(structs: &mut StructInfo, type1: &VarType, type2: &VarType) -> bool {
    match type1 {
        VarType::Int => match type2 {
            VarType::Int => true,
            _ => false,
        },
        VarType::Long => match type2 {
            VarType::Long => true,
            _ => false,
        },
        VarType::Double => match type2 {
            VarType::Double => true,
            _ => false,
        },
        VarType::Float => match type2 {
            VarType::Float => true,
            _ => false,
        },
        VarType::Bool => match type2 {
            VarType::Bool => true,
            _ => false,
        },
        VarType::String => match type2 {
            VarType::String => true,
            _ => false,
        },
        VarType::Null => match type2 {
            VarType::Null => true,
            _ => false,
        },
        VarType::Void => match type2 {
            VarType::Void => true,
            _ => false,
        },
        VarType::Usize => match type2 {
            VarType::Usize => true,
            _ => false,
        },
        VarType::Ref(r1) => match type2 {
            VarType::Ref(r2) => compare_types(structs, r1.as_ref(), r2.as_ref()),
            _ => false,
        },
        VarType::Struct(name1) => match type2 {
            VarType::Struct(name2) => name1 == name2,
            VarType::StructShape(shape1) => {
                let shape2 = structs.available_structs.get(name1).unwrap();
                let shape2 = shape2.clone();
                let shape2 = &*shape2.borrow();

                compare_struct_shapes(structs, shape1, shape2)
            }
            _ => false,
        },
        VarType::StructShape(shape1) => match type2 {
            VarType::StructShape(shape2) => compare_struct_shapes(structs, shape1, shape2),
            _ => false,
        },
        VarType::Array(a1) => match type2 {
            VarType::Array(a2) => compare_types(structs, a1.as_ref(), a2.as_ref()),
            _ => false,
        },
        VarType::Fn(params1, return1) => match type2 {
            VarType::Fn(params2, return2) => {
                compare_types(structs, params1.as_ref(), params2.as_ref())
                    && compare_types(structs, return1, return2)
            }
            _ => false,
        },
        VarType::Union(types) => in_union(structs, types, type2),
    }
}

fn compare_struct_shapes(
    structs: &mut StructInfo,
    shape1: &StructShape,
    shape2: &StructShape,
) -> bool {
    for (name1, type1) in shape1.props.iter() {
        let mut found = false;
        for (name2, type2) in shape2.props.iter() {
            if name1 == name2 {
                if compare_types(structs, type1, type2) {
                    found = true;
                    break;
                }
            }
        }
        if !found {
            return false;
        }
    }
    true
}

fn in_union(structs: &mut StructInfo, union: &Vec<VarType>, t: &VarType) -> bool {
    if let VarType::Union(types) = t {
        'outer: for union_item1 in union.iter() {
            let mut found = false;

            for union_item2 in types.iter() {
                if compare_types(structs, union_item1, union_item2) {
                    found = true;
                    break 'outer;
                }
            }

            if !found {
                return false;
            }
        }

        return true;
    }

    for union_item in union.iter() {
        if compare_types(structs, union_item, t) {
            return true;
        }
    }

    false
}

pub fn get_eval_value<'a>(
    // vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    vars: &mut Variables,
    val: EvalValue,
    scope: usize,
    return_ptr: bool,
) -> Value {
    match val {
        EvalValue::Value(v) => v,
        EvalValue::Token(t) => match t {
            Token::Number(_) => value_from_token(&t, None),
            Token::String(_) => value_from_token(&t, None),
            Token::Bool(_) => value_from_token(&t, None),
            Token::Null => value_from_token(&t, None),
            Token::Identifier(ident) => {
                let var_ptr = get_var_ptr(vars, &ident, scope);
                let var_ref = var_ptr.borrow();

                if return_ptr {
                    let ref_value = &var_ref.value;
                    if let Value::Ref(r) = &*ref_value.borrow() {
                        Value::Ref(Rc::clone(&r))
                    } else {
                        let val = Rc::clone(ref_value);
                        Value::Ref(val)
                    }
                } else {
                    let ref_value = &var_ref.value;
                    if let Value::Ref(r) = &*ref_value.borrow() {
                        Value::Ref(Rc::clone(&r))
                    } else {
                        var_ref.value.borrow().clone()
                    }
                }
            }
            _ => panic!("Unexpected token: {:?}", t),
        },
    }
}

pub fn get_prop_ptr(props: &mut Vec<StructProp>, name: &String) -> Option<Rc<RefCell<Value>>> {
    let mut ptr = None;

    for prop in props.iter_mut() {
        if prop.name == *name {
            ptr = Some(Rc::clone(&prop.value));
            break;
        }
    }

    ptr
}

pub fn push_to_array<'a>(
    vars: &mut Variables,
    functions: &mut Vec<Func<'a>>,
    structs: &mut StructInfo,
    scope: usize,
    arr: &mut Vec<Value>,
    arr_type: &VarType,
    args: &Vec<AstNode>,
    stdout: &mut Stdout,
) {
    for arg in args.iter() {
        let arg_res_option = eval_node(vars, functions, structs, scope, arg, stdout);

        if let (Some(arg_res), _) = arg_res_option {
            let val = get_eval_value(vars, arg_res, scope, false);
            let val_type = type_from_value(&val);
            if !compare_types(structs, &arr_type, &val_type) {
                panic!(
                    "Error pushing to array, expected type: {:?} found {:?}",
                    arr_type,
                    type_from_value(&val),
                );
            }
            arr.push(val);
        }
    }
}

pub fn unshift_array<'a>(
    vars: &mut Variables,
    functions: &mut Vec<Func<'a>>,
    structs: &mut StructInfo,
    scope: usize,
    arr: &mut Vec<Value>,
    arr_type: &VarType,
    args: &Vec<AstNode>,
    stdout: &mut Stdout,
) {
    for (i, arg) in args.iter().enumerate() {
        let arg_res_option = eval_node(vars, functions, structs, scope, arg, stdout);

        if let (Some(arg_res), _) = arg_res_option {
            let val = get_eval_value(vars, arg_res, scope, false);
            let val_type = type_from_value(&val);
            if !compare_types(structs, &arr_type, &val_type) {
                panic!(
                    "Error pushing to array, expected type: {:?} found {:?}",
                    arr_type,
                    type_from_value(&val),
                );
            }
            arr.insert(i, val);
        }
    }
}

pub fn type_from_value(val: &Value) -> VarType {
    match val {
        Value::Ref(ref_ptf) => {
            let ref_type = type_from_value(&ref_ptf.as_ref().borrow());
            VarType::Ref(Box::new(ref_type))
        }
        Value::Null => VarType::Null,
        Value::Usize(_) => VarType::Usize,
        Value::String(_) => VarType::String,
        Value::Int(_) => VarType::Int,
        Value::Float(_) => VarType::Float,
        Value::Double(_) => VarType::Double,
        Value::Long(_) => VarType::Long,
        Value::Bool(_) => VarType::Bool,
        Value::Struct(name, _, _) => VarType::Struct(name.clone()),
        Value::Array(_, arr_type) => VarType::Array(Box::new(arr_type.clone())),
        Value::Fn(func) => {
            let return_type = func.borrow().return_type.clone();
            let mut param_shape = StructShape::new();
            for param in func.borrow().params.iter() {
                param_shape.add(param.name.clone(), param.param_type.clone())
            }
            let param_struct = VarType::StructShape(param_shape);
            VarType::Fn(Box::new(param_struct), Box::new(return_type))
        }
    }
}

pub fn create_cast_node<'a>(structs: &mut StructInfo, tokens: &mut Vec<Option<Token>>) -> AstNode {
    let mut node_tokens = tokens_to_delimiter(tokens, 2, ")");
    let node_option = get_ast_node(structs, &mut node_tokens);

    let to_type = match tokens[0].as_ref().unwrap() {
        Token::Type(t) => t.get_str(),
        _ => panic!("Cannot cast to: {:?}", tokens[0].as_ref().unwrap()),
    };

    if let Some(node) = node_option {
        AstNode::Cast(VarType::from(to_type), Box::new(node))
    } else {
        panic!("")
    }
}

macro_rules! cast_panic {
    ($from:expr, $to:expr) => {
        panic!("Cannot convert {} to {}", $from, $to)
    };
}

pub fn cast(to_type: &VarType, val: Value) -> Value {
    match val {
        Value::Ref(_) => panic!("Cannot cast ref"),
        Value::Null => panic!("Cannot cast null"),
        Value::Fn(_) => panic!("Cannot cast fn"),
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
                    cast_panic!("usize", "bool")
                }
            }
            VarType::Array(_) => cast_panic!("usize", "array"),
            VarType::Struct(_) => cast_panic!("usize", "struct"),
            VarType::Null => cast_panic!("usize", "null"),
            VarType::Ref(_) => cast_panic!("usize", "ref"),
            VarType::Void => cast_panic!("usize", "void"),
            VarType::Fn(_, _) => cast_panic!("usize", "fn"),
            VarType::Union(_) => cast_panic!("usize", "union"),
            VarType::StructShape(_) => unreachable!(),
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
            VarType::Array(_) => cast_panic!("string", "array"),
            VarType::Struct(_) => cast_panic!("string", "struct"),
            VarType::Null => cast_panic!("string", "null"),
            VarType::Ref(_) => cast_panic!("string", "ref"),
            VarType::Void => cast_panic!("string", "void"),
            VarType::Fn(_, _) => cast_panic!("string", "fn"),
            VarType::Union(_) => cast_panic!("string", "union"),
            VarType::StructShape(_) => unreachable!(),
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
            VarType::Array(_) => cast_panic!("int", "array"),
            VarType::Struct(_) => cast_panic!("int", "struct"),
            VarType::Null => cast_panic!("int", "null"),
            VarType::Ref(_) => cast_panic!("int", "ref"),
            VarType::Void => cast_panic!("int", "void"),
            VarType::Fn(_, _) => cast_panic!("int", "fn"),
            VarType::Union(_) => cast_panic!("int", "union"),
            VarType::StructShape(_) => unreachable!(),
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
            VarType::Array(_) => cast_panic!("float", "array"),
            VarType::Struct(_) => cast_panic!("float", "struct"),
            VarType::Null => cast_panic!("float", "null"),
            VarType::Ref(_) => cast_panic!("float", "ref"),
            VarType::Void => cast_panic!("float", "void"),
            VarType::Fn(_, _) => cast_panic!("float", "fn"),
            VarType::Union(_) => cast_panic!("float", "union"),
            VarType::StructShape(_) => unreachable!(),
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
                    cast_panic!("double", "bool")
                }
            }
            VarType::Array(_) => cast_panic!("double", "array"),
            VarType::Struct(_) => cast_panic!("double", "struct"),
            VarType::Null => cast_panic!("double", "null"),
            VarType::Ref(_) => cast_panic!("double", "ref"),
            VarType::Void => cast_panic!("double", "void"),
            VarType::Fn(_, _) => cast_panic!("double", "fn"),
            VarType::Union(_) => cast_panic!("double", "union"),
            VarType::StructShape(_) => unreachable!(),
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
            VarType::Array(_) => cast_panic!("long", "array"),
            VarType::Struct(_) => cast_panic!("long", "struct"),
            VarType::Null => cast_panic!("long", "null"),
            VarType::Ref(_) => cast_panic!("long", "ref"),
            VarType::Void => cast_panic!("long", "void"),
            VarType::Fn(_, _) => cast_panic!("long", "fn"),
            VarType::Union(_) => cast_panic!("long", "union"),
            VarType::StructShape(_) => unreachable!(),
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
                VarType::Array(_) => cast_panic!("bool", "array"),
                VarType::Struct(_) => cast_panic!("bool", "struct"),
                VarType::Null => cast_panic!("bool", "null"),
                VarType::Ref(_) => cast_panic!("bool", "ref"),
                VarType::Void => cast_panic!("bool", "void"),
                VarType::Fn(_, _) => cast_panic!("bool", "fn"),
                VarType::Union(_) => cast_panic!("bool", "union"),
                VarType::StructShape(_) => unreachable!(),
            }
        }
        Value::Array(arr, _) => match to_type {
            VarType::Usize => cast_panic!("array", "usize"),
            VarType::Int => cast_panic!("array", "int"),
            VarType::Float => cast_panic!("array", "float"),
            VarType::Double => cast_panic!("array", "double"),
            VarType::Long => cast_panic!("array", "long"),
            VarType::String => Value::String(get_value_arr_str(&arr)),
            VarType::Bool => cast_panic!("array", "bool"),
            VarType::Array(_) => unimplemented!(),
            VarType::Struct(_) => cast_panic!("array", "struct"),
            VarType::Null => cast_panic!("array", "null"),
            VarType::Ref(_) => cast_panic!("array", "ref"),
            VarType::Void => cast_panic!("array", "void"),
            VarType::Fn(_, _) => cast_panic!("array", "fn"),
            VarType::Union(_) => cast_panic!("array", "union"),
            VarType::StructShape(_) => unreachable!(),
        },
        Value::Struct(_, _, _) => panic!("Cannot cast structs"),
    }
}

pub fn get_struct_access_tokens<'a>(tokens: &mut Vec<Option<Token>>) -> Vec<Vec<Option<Token>>> {
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
                    i += 1;
                    continue;
                }
                Token::Identifier(_) => false,
                Token::LParen => false,
                Token::EqSet => false,
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
                break;
            }
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
    let mut closed_blocks = 0;

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
                if open_brackets == 0 {
                    closed_blocks += 1;
                }
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
    open_brackets == 0 && (semicolons > 0 || closed_blocks > 1)
}

pub fn create_comp_node<'a>(
    structs: &mut StructInfo,
    tokens: &mut Vec<Option<Token>>,
) -> Option<AstNode> {
    if let Token::Identifier(name) = tokens[0].as_ref().unwrap().clone() {
        if is_builtin_generic(&name) {
            return None;
        }
    }

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
            Token::LAngle => {
                if i == 0 || i == tokens.len() - 1 {
                    return None;
                } else {
                    match tokens[i - 1].as_ref().unwrap() {
                        Token::EqSet => return None,
                        Token::Keyword(_) => return None,
                        _ => true,
                    }
                }
            }
            Token::RAngle => {
                if i == 0 || i == tokens.len() - 1 {
                    return None;
                } else {
                    match tokens[i - 1].as_ref().unwrap() {
                        Token::EqSet => return None,
                        Token::Keyword(_) => return None,
                        _ => true,
                    }
                }
            }
            Token::LAngleEq => true,
            Token::RAngleEq => true,
            _ => false,
        } && open_parens == 0
        {
            for j in 0..i {
                temp_tokens.push(Some(tokens[j].take().unwrap()));
            }

            if let Some(left_node) = get_ast_node(structs, &mut temp_tokens) {
                temp_tokens.drain(0..temp_tokens.len());

                for j in i + 1..tokens.len() {
                    temp_tokens.push(Some(tokens[j].take().unwrap()));
                }

                if let Some(right_node) = get_ast_node(structs, &mut temp_tokens) {
                    return Some(AstNode::Comparison(
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
    ($structs:expr, $left:expr, $right:expr, ==, $($variants:ident),*) => {
        {
            match $left {
                $(
                    Value::$variants(l) => match $right {
                        Value::$variants(r) => Ok(l == r),
                        Value::Array(_, _) => Err(String::from("Cannot compare arrays")),
                        _ => Ok(false)
                    }
                )*
                Value::Array(left_arr, left_type) => match $right {
                    Value::Array(right_arr, right_type) => Ok(compare_array($structs, left_arr, right_arr, left_type, right_type)),
                    _ => Ok(false)
                }
                Value::Struct(left_name, _, props_left) => match $right {
                    Value::Struct(right_name, _, props_right) => {
                        Ok(if left_name == right_name {
                            false
                        } else {
                            compare_struct_values($structs, props_left, props_right)
                        })
                    },
                    _ => Ok(false)
                },
                Value::Null => match $right {
                    Value::Null => Ok(true),
                    _ => Ok(false),
                }
                Value::Ref(r) => {
                    let new_left = &*r.borrow();
                    comp_eq_abstraction($structs, new_left, $right)
                },
                Value::Fn(_) => Err(String::from("Cannot compare functions"))
            }
        }
    };
    ($structs:expr, $left:expr, $right:expr, $c:tt, $($variants:ident),*) => {
        {
            match $left {
                $(
                    Value::$variants(l) => match $right {
                        Value::$variants(r) => Ok(l $c r),
                        Value::Array(_, _) => Err(String::from("Cannot compare arrays")),
                        _ => Err(format!("Error, different types. Found {} and {}", $left.get_enum_str(), $right.get_enum_str()))
                    }
                )*
                Value::Array(_, _) => Err(String::from("Arrays can only be compared with `==` operator")),
                Value::Struct(_, _, _) => Err(String::from("Structs can only be compared with `==` operator")),
                Value::Null => Err(String::from("Null can only be compared with `==` operator")),
                Value::Ref(_) => Err(String::from("Cannot compare refs")),
                Value::Fn(_) => Err(String::from("Cannot compare functions"))
            }
        }
    };
}

macro_rules! comp_bind {
    ($structs:expr, $left:expr, $right:expr, $tok:tt) => {
        comp!($structs, $left, $right, $tok, Usize, String, Int, Float, Double, Long, Bool)
    };
}

macro_rules! comp_match {
    ($structs:expr, $tok:ident, $left:expr, $right:expr) => {
        match $tok {
            Token::EqCompare => comp_bind!($structs, $left, $right, ==),
            Token::EqNCompare => comp_bind!($structs, $left, $right, !=),
            Token::LAngle => comp_bind!($structs, $left, $right, <),
            Token::RAngle => comp_bind!($structs, $left, $right, >),
            Token::LAngleEq => comp_bind!($structs, $left, $right, <=),
            Token::RAngleEq => comp_bind!($structs, $left, $right, >=),
            _ => panic!("Expected comparison operator"),
        }
    }
}

fn comp_eq_abstraction(
    structs: &mut StructInfo,
    left: &Value,
    right: &Value,
) -> Result<bool, String> {
    comp_bind!(structs, left, right, ==)
}

fn compare_array(
    structs: &mut StructInfo,
    left: &Vec<Value>,
    right: &Vec<Value>,
    left_type: &VarType,
    right_type: &VarType,
) -> bool {
    if !compare_types(structs, left_type, right_type) {
        if left.len() != right.len() {
            return false;
        }

        for i in 0..left.len() {
            if match comp!(structs, &left[i], &right[i], ==, Usize, String, Int, Float, Double, Long, Bool)
            {
                Ok(val) => !val,
                Err(msg) => {
                    println!("{}", msg);
                    exit(1);
                }
            } {
                return false;
            }
        }

        true
    } else {
        false
    }
}

fn compare_struct_values(
    structs: &mut StructInfo,
    left: &Vec<StructProp>,
    right: &Vec<StructProp>,
) -> bool {
    for left_val in left.iter() {
        let mut found = false;

        for right_val in right.iter() {
            if left_val.name == right_val.name {
                let left = left_val.value.borrow().clone();
                let right = left_val.value.borrow().clone();
                if match comp_bind!(structs, &left, &right, ==) {
                    Ok(val) => val,
                    Err(_) => false,
                } {
                    found = true;
                }
            }
        }

        if !found {
            return false;
        }
    }

    true
}

pub fn compare<'a>(
    vars: &mut Variables,
    structs: &mut StructInfo,
    left: EvalValue,
    right: EvalValue,
    comp_token: &Token,
    scope: usize,
) -> EvalValue {
    let res = match left {
        EvalValue::Value(left_val) => match right {
            EvalValue::Value(right_val) => {
                comp_match!(structs, comp_token, &left_val, &right_val)
            }
            EvalValue::Token(t) => match t {
                Token::Identifier(ident) => {
                    let var_ptr = get_var_ptr(vars, &ident, scope);
                    let var_ref = var_ptr.borrow();
                    let right_val = &*var_ref.value.borrow();
                    comp_match!(structs, comp_token, &left_val, right_val)
                }
                _ => {
                    let right_val = value_from_token(&t, None);
                    comp_match!(structs, comp_token, &left_val, &right_val)
                }
            },
        },
        EvalValue::Token(t) => match t {
            Token::Identifier(ident) => {
                let var_ptr = get_var_ptr(vars, &ident, scope);
                let var_ref = var_ptr.borrow();
                let left_val = &*var_ref.value.borrow();
                match right {
                    EvalValue::Value(right_val) => {
                        comp_match!(structs, comp_token, left_val, &right_val)
                    }
                    EvalValue::Token(t) => match t {
                        Token::Identifier(ident) => {
                            let var_ptr = get_var_ptr(vars, &ident, scope);
                            let var_ref = var_ptr.borrow();
                            let right_val = &*var_ref.value.borrow();
                            comp_match!(structs, comp_token, &left_val, right_val)
                        }
                        _ => {
                            let right_val = value_from_token(&t, None);
                            comp_match!(structs, comp_token, left_val, &right_val)
                        }
                    },
                }
            }
            _ => {
                let left_val = value_from_token(&t, None);

                match right {
                    EvalValue::Value(right_val) => {
                        comp_match!(structs, comp_token, &left_val, &right_val)
                    }
                    EvalValue::Token(t) => match t {
                        Token::Identifier(ident) => {
                            let var_ptr = get_var_ptr(vars, &ident, scope);
                            let var_ref = var_ptr.borrow();
                            let right_val = &*var_ref.value.borrow();
                            comp_match!(structs, comp_token, &left_val, right_val)
                        }
                        _ => {
                            let right_val = value_from_token(&t, None);
                            comp_match!(structs, comp_token, &left_val, &right_val)
                        }
                    },
                }
            }
        },
    };

    match res {
        Ok(val) => EvalValue::Value(Value::Bool(val)),
        Err(msg) => {
            println!("{}", msg);
            exit(1);
        }
    }
}

pub fn index_arr_var_value(
    vars: &mut Variables,
    arr: Rc<RefCell<VarValue>>,
    index: EvalValue,
    scope: usize,
) -> Value {
    match &*arr.borrow().value.borrow() {
        Value::Array(arr, _) => {
            let index = get_eval_value(vars, index, scope, false);

            if let Value::Usize(val) = index {
                arr[val].clone()
            } else {
                panic!("Array can only be indexed by usize")
            }
        }
        _ => panic!("Cannot index a non-array type"),
    }
}

pub fn set_index_arr<'a>(
    vars: &mut Variables,
    functions: &mut Vec<Func<'a>>,
    structs: &mut StructInfo,
    scope: usize,
    stdout: &mut Stdout,
    arr: Rc<RefCell<VarValue>>,
    index: EvalValue,
    value: AstNode,
) {
    if let (Some(value), _) = eval_node(vars, functions, structs, scope, &value, stdout) {
        let index_val = get_eval_value(vars, index, scope, true);
        match index_val {
            Value::Usize(val) => match *arr.borrow_mut().value.borrow_mut() {
                Value::Array(ref mut arr_val, _) => {
                    arr_val[val] = get_eval_value(vars, value, scope, true);
                }
                _ => panic!("Only arrays can be indexed"),
            },
            _ => panic!("Array can only be indexed with usize"),
        }
    } else {
        panic!("Expected value to set at array index");
    }
}

pub fn create_struct_node<'a>(
    tokens: &mut Vec<Option<Token>>,
    structs: &mut StructInfo,
    shape: Rc<RefCell<StructShape>>,
    name: &String,
) -> AstNode {
    let mut start = 1;
    let mut props: Vec<(String, AstNode)> = Vec::new();

    let mut i = 0;
    while i < tokens.len() {
        if let Some(tok) = tokens[i].as_ref() {
            if match tok {
                Token::NewLine => true,
                _ => false,
            } {
                tokens.remove(i);
            } else {
                i += 1;
            }
        }
    }

    while start < tokens.len() {
        let mut property = tokens_to_delimiter(tokens, start, ",");

        if let Token::RBrace = property[property.len() - 1].as_ref().unwrap() {
            property.remove(property.len() - 1);
        }

        if property.len() == 0 {
            break;
        }

        let name = if let Token::Identifier(ident) = property[0].take().unwrap() {
            ident
        } else {
            panic!("Expected identifier for struct property name");
        };

        let len = property.len();

        property.remove(0);
        property.remove(0);

        let prop_node = get_ast_node(structs, &mut property);
        if let Some(node) = prop_node {
            props.push((name, node));
        } else {
            panic!("Expected value for struct property");
        }

        start += len + 1;
    }

    AstNode::CreateStruct(name.clone(), shape, props)
}

const BUILTIN_GENERIC: [&'static str; 3] = ["ref", "fn", "union"];

fn is_builtin_generic(name: &String) -> bool {
    BUILTIN_GENERIC.iter().position(|&s| s == name).is_some()
}

fn get_builtin_generic(
    name: &String,
    structs: &mut StructInfo,
    tokens: &mut Vec<Option<Token>>,
) -> Option<VarType> {
    match name.as_str() {
        "ref" => {
            let type_tokens = tokens_to_delimiter(tokens, 2, ">");
            let mut generic_type = match type_tokens[0].as_ref().unwrap() {
                Token::Type(t) => t.clone(),
                Token::Identifier(ident) => {
                    let struct_info = structs.available_structs.get(ident.as_str());
                    if let Some(_) = struct_info {
                        VarType::Struct(ident.clone())
                    } else {
                        panic!("Unexpected type {}", ident);
                    }
                }
                _ => panic!("Expected valid type for generic"),
            };

            let mut i = 0;
            while i < type_tokens.len() - 1
                && match type_tokens[i + 1].as_ref().unwrap() {
                    Token::LBracket => true,
                    _ => false,
                }
            {
                generic_type = VarType::Array(Box::new(generic_type));

                i += 2;
            }

            Some(VarType::Ref(Box::new(generic_type)))
        }
        "fn" => {
            let param_tokens = tokens_to_delimiter(tokens, 2, ",");
            let param_type = if let Token::Identifier(ident) = param_tokens[0].as_ref().unwrap() {
                let struct_info = structs.available_structs.get(ident.as_str());
                if let Some(_) = struct_info {
                    VarType::Struct(ident.clone())
                } else {
                    panic!("Unexpected type {}", ident);
                }
            } else {
                panic!("Expected struct ident for param type");
            };

            let mut return_tokens = tokens_to_delimiter(tokens, 3 + param_tokens.len(), ">");
            let return_type = get_type_expression(&mut return_tokens, structs);

            Some(VarType::Fn(Box::new(param_type), Box::new(return_type)))
        }
        "union" => {
            let mut union: Vec<VarType> = vec![];

            let mut type_tokens = tokens_to_delimiter(tokens, 2, ">");
            let mut first_tokens = tokens_to_delimiter(&mut type_tokens, 0, ",");
            let first_type = get_type_expression(&mut first_tokens, structs);
            union.push(first_type);

            let mut offset = first_tokens.len() + 1;

            loop {
                let mut other_type = tokens_to_delimiter(&mut type_tokens, offset, ",");
                offset += other_type.len() + 1;

                if other_type.len() == 0 {
                    break;
                }

                let type_node = get_type_expression(&mut other_type, structs);
                union.push(type_node);
            }

            Some(VarType::Union(union))
        }
        _ => None,
    }
}

pub fn get_ref_value(val: &Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
    if let Value::Ref(ref r) = *val.borrow() {
        get_ref_value(&Rc::clone(r))
    } else {
        Rc::clone(val)
    }
}
