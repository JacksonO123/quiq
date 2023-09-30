use std::{
    cell::RefCell,
    io::{Stdout, Write},
    rc::Rc,
};

use crate::{
    ast::Value,
    helpers::{get_eval_value, get_file, get_ref_value, input, write_file},
    interpreter::{get_var_ptr, value_from_token, BuiltinFunc, EvalValue, Func, VarType},
    tokenizer::Token,
    variables::Variables,
};

fn expect_params(name: &str, expected: usize, found: usize) {
    if expected != found {
        panic!(
            "Expected {} params in function `{}`, found {}",
            expected, name, found
        );
    }
}

macro_rules! get_value_variant {
    ($val:expr, $variant:ident, $reason:expr) => {
        if let Value::$variant(v) = &$val {
            v
        } else {
            panic!("Expected {:?} in {}", stringify!($variant), $reason);
        }
    };
}

fn print_abstracted(
    vars: &mut Variables,
    mut params: Vec<Option<EvalValue>>,
    scope: usize,
    stdout: &mut Stdout,
) {
    for i in 0..params.len() {
        let param = params[i].take().unwrap();
        let to_print = match param {
            EvalValue::Value(val) => val.get_str(),
            EvalValue::Token(tok) => match tok {
                Token::Identifier(ident) => {
                    let var_ptr = get_var_ptr(vars, &ident, scope);
                    let var_borrow = var_ptr.borrow();
                    let var_value = &*var_borrow.value.borrow();
                    var_value.get_str()
                }
                _ => value_from_token(&tok, None).get_str(),
            },
        };

        let to_print = to_print.as_str();
        let bytes = to_print.as_bytes();

        stdout.write_all(bytes).unwrap();
        if i < &params.len() - 1 {
            stdout.write_all(b" ").unwrap();
        }
    }
}

pub fn init_builtins(functions: &mut Vec<Func>) {
    functions.push(Func::Builtin(BuiltinFunc::new(
        "print",
        |vars, params, scope, stdout| {
            print_abstracted(vars, params, scope, stdout);

            None
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "println",
        |vars, params, scope, stdout| {
            print_abstracted(vars, params, scope, stdout);

            stdout.write_all(b"\n").unwrap();
            None
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "type",
        |vars, mut params, scope, _| {
            expect_params("type", 1, params.len());

            let res = match params[0].take().unwrap() {
                EvalValue::Token(t) => match t {
                    Token::Identifier(ident) => {
                        let ptr = get_var_ptr(vars, &ident, scope);
                        let ptr_borrow = ptr.borrow();
                        let ptr_value = ptr_borrow.value.borrow();
                        ptr_value.get_enum_str().clone()
                    }
                    _ => String::from(t.get_token_name()),
                },
                EvalValue::Value(v) => v.get_enum_str(),
            };

            Some(Value::String(res))
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "ref",
        |vars, mut params, scope, _| {
            expect_params("ref", 1, params.len());

            match &params[0].take().unwrap() {
                // cloning here because the value is defined within the ref function and cannot be
                // referenced anywhere else, therefore it is not important to keep the reference
                // with the original value
                EvalValue::Value(val) => Some(Value::Ref(Rc::new(RefCell::new(val.clone())))),
                EvalValue::Token(t) => match t {
                    Token::Identifier(ident) => {
                        let val = get_var_ptr(vars, &ident, scope);
                        let val_ref = val.borrow_mut();
                        Some(Value::Ref(Rc::clone(&val_ref.value)))
                    }
                    _ => panic!("Expected identifier or value to create ref"),
                },
            }
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "clone",
        |vars, mut params, scope, _| {
            let eval_value = params[0].take().unwrap();
            let to_clone = get_eval_value(vars, eval_value, scope, true);
            Some(match to_clone {
                Value::Ref(r) => get_ref_value(&r).borrow().clone(),
                _ => to_clone,
            })
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "free",
        |vars, mut params, scope, _| {
            for i in 0..params.len() {
                let param = params[i].take().unwrap();
                match param {
                    EvalValue::Value(_) => panic!("Expected variable identifier to free"),
                    EvalValue::Token(t) => {
                        if let Token::Identifier(ident) = t {
                            vars.free(&ident, scope);
                        } else {
                            panic!("Expected variable identifier to free");
                        }
                    }
                }
            }

            None
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "scope",
        |_, _, scope, _| {
            println!("{}", scope);
            None
        },
    )));

    // debug
    functions.push(Func::Builtin(BuiltinFunc::new(
        "showVars",
        |vars, _, _, _| {
            vars.print();
            None
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "input",
        |vars, mut params, scope, _| {
            expect_params("input", 1, params.len());

            let param = params[0].take().unwrap();
            let val = get_eval_value(vars, param, scope, false);

            let s = val.get_str();
            Some(Value::String(input(&s)))
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "readFile",
        |vars, mut params, scope, _| {
            expect_params("readFile", 1, params.len());

            let param = params[0].take().unwrap();
            let val = get_eval_value(vars, param, scope, false);

            let s = get_value_variant!(val, String, "readFile");
            let file = get_file(s);
            Some(Value::String(file))
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "writeFile",
        |vars, mut params, scope, _| {
            expect_params("writeFile", 2, params.len());

            let destination = params[0].take().unwrap();
            let info = params[1].take().unwrap();

            let dest = get_eval_value(vars, destination, scope, false);
            let info = get_eval_value(vars, info, scope, false);

            let dest = get_value_variant!(dest, String, "writeFile");
            let info = get_value_variant!(info, String, "writeFile");
            write_file(&dest, &info).expect("Error writing file");

            None
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "parse",
        |vars, mut params, scope, _| {
            expect_params("parse", 1, params.len());

            let param = params[0].take().unwrap();
            let val = get_eval_value(vars, param, scope, false);

            let s = get_value_variant!(val, String, "parse");
            let token = Token::Number(s.clone());
            Some(value_from_token(&token, None))
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "split",
        |vars, mut params, scope, _| {
            expect_params("split", 2, params.len());

            let string = params[0].take().unwrap();
            let delimiter = params[1].take().unwrap();

            let val = get_eval_value(vars, string, scope, false);
            let delimiter_val = get_eval_value(vars, delimiter, scope, false);

            let string = get_value_variant!(val, String, "split");
            let delimiter = get_value_variant!(delimiter_val, String, "split");

            let res: Vec<&str> = string.split(delimiter).collect();
            let res: Vec<Value> = res.iter().map(|&s| Value::String(s.to_string())).collect();

            let arr_type = VarType::Array(Box::new(VarType::String));

            Some(Value::Array(res, arr_type))
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "substr",
        |vars, mut params, scope, _| {
            expect_params("substr", 3, params.len());

            let str = params[0].take().unwrap();
            let str_val = get_eval_value(vars, str, scope, false);
            let str = get_value_variant!(str_val, String, "substr");

            let from = params[1].take().unwrap();
            let from_val = get_eval_value(vars, from, scope, false);
            let from = get_value_variant!(from_val, Usize, "substr").clone();

            let to = params[2].take().unwrap();
            let to_val = get_eval_value(vars, to, scope, false);
            let to = get_value_variant!(to_val, Usize, "substr").clone();

            let res = &str[from..to];

            Some(Value::String(res.to_string()))
        },
    )));
}
