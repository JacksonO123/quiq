use std::{
    cell::RefCell,
    io::{Stdout, Write},
    ops::RangeInclusive,
    process::exit,
    rc::Rc,
};

use crate::{
    ast::Value,
    data::Data,
    helpers::{get_eval_value, get_file, get_ref_value, input, write_file},
    interpreter::{get_var_ptr, value_from_token, BuiltinFunc, EvalValue, Func, VarType},
    tokenizer::Token,
};

fn expect_params(name: &str, expected: usize, found: usize) {
    if expected != found {
        panic!(
            "Expected {} params in function `{}`, found {}",
            expected, name, found
        );
    }
}

fn expect_params_range(name: &str, expected: RangeInclusive<usize>, found: usize) {
    if !expected.contains(&found) {
        panic!(
            "Expected {} to {} params in function `{}`, found {}",
            expected.start(),
            expected.end(),
            name,
            found
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
    data: &mut Data,
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
                    let var_ptr = get_var_ptr(data, &ident, scope);
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

pub fn init_builtins(data: &mut Data) {
    let functions = data.functions();

    functions.push(Func::Builtin(BuiltinFunc::new(
        "print",
        |data, params, scope, stdout| {
            print_abstracted(data, params, scope, stdout);

            None
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "println",
        |data, params, scope, stdout| {
            print_abstracted(data, params, scope, stdout);

            stdout.write_all(b"\n").unwrap();
            None
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "type",
        |data, mut params, scope, _| {
            expect_params("type", 1, params.len());

            let res = match params[0].take().unwrap() {
                EvalValue::Token(t) => match t {
                    Token::Identifier(ident) => {
                        let ptr = get_var_ptr(data, &ident, scope);
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
        |data, mut params, scope, _| {
            expect_params("ref", 1, params.len());

            match &params[0].take().unwrap() {
                // cloning here because the value is defined within the ref function and cannot be
                // referenced anywhere else, therefore it is not important to keep the reference
                // with the original value
                EvalValue::Value(val) => Some(Value::Ref(Rc::new(RefCell::new(val.clone())))),
                EvalValue::Token(t) => match t {
                    Token::Identifier(ident) => {
                        let val = get_var_ptr(data, ident, scope);
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
        |data, mut params, scope, _| {
            let eval_value = params[0].take().unwrap();
            let to_clone = get_eval_value(data, eval_value, scope, true);
            Some(match to_clone {
                Value::Ref(r) => get_ref_value(&r).borrow().clone(),
                _ => to_clone,
            })
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "free",
        |data, mut params, scope, _| {
            for item in &mut params {
                let param = item.take().unwrap();
                match param {
                    EvalValue::Value(_) => panic!("Expected variable identifier to free"),
                    EvalValue::Token(t) => {
                        if let Token::Identifier(ident) = t {
                            data.vars().free(&ident, scope);
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
        |data, _, _, _| {
            data.vars().print();
            None
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "input",
        |data, mut params, scope, _| {
            expect_params("input", 1, params.len());

            let param = params[0].take().unwrap();
            let val = get_eval_value(data, param, scope, false);

            let s = val.get_str();
            Some(Value::String(input(&s)))
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "readFile",
        |data, mut params, scope, _| {
            expect_params("readFile", 1, params.len());

            let param = params[0].take().unwrap();
            let val = get_eval_value(data, param, scope, false);

            let s = get_value_variant!(val, String, "readFile");
            let file = get_file(s);
            Some(Value::String(file))
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "writeFile",
        |data, mut params, scope, _| {
            expect_params("writeFile", 2, params.len());

            let destination = params[0].take().unwrap();
            let info = params[1].take().unwrap();

            let dest = get_eval_value(data, destination, scope, false);
            let info = get_eval_value(data, info, scope, false);

            let dest = get_value_variant!(dest, String, "writeFile");
            let info = get_value_variant!(info, String, "writeFile");
            write_file(dest, info).expect("Error writing file");

            None
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "parse",
        |data, mut params, scope, _| {
            expect_params("parse", 1, params.len());

            let param = params[0].take().unwrap();
            let val = get_eval_value(data, param, scope, false);

            let s = get_value_variant!(val, String, "parse");
            let token = Token::Number(s.clone());
            Some(value_from_token(&token, None))
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "split",
        |data, mut params, scope, _| {
            expect_params("split", 2, params.len());

            let string = params[0].take().unwrap();
            let delimiter = params[1].take().unwrap();

            let val = get_eval_value(data, string, scope, false);
            let delimiter_val = get_eval_value(data, delimiter, scope, false);

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
        |data, mut params, scope, _| {
            expect_params("substr", 3, params.len());

            let str = params[0].take().unwrap();
            let str_val = get_eval_value(data, str, scope, false);
            let str = get_value_variant!(str_val, String, "substr");

            let from = params[1].take().unwrap();
            let from_val = get_eval_value(data, from, scope, false);
            let from = *get_value_variant!(from_val, Usize, "substr");

            let to = params[2].take().unwrap();
            let to_val = get_eval_value(data, to, scope, false);
            let to = *get_value_variant!(to_val, Usize, "substr");

            let res = &str[from..to];

            Some(Value::String(res.to_string()))
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "exit",
        |data, mut params, scope, _| {
            expect_params_range("exit", 0..=1, params.len());

            let val = if !params.is_empty() {
                let val = params[0].take().unwrap();
                let val = get_eval_value(data, val, scope, false);
                *get_value_variant!(val, Int, "exit")
            } else {
                0
            };

            exit(val);
        },
    )))
}
