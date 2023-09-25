use std::{cell::RefCell, collections::HashMap, io::Write, rc::Rc};

use crate::{
    ast::Value,
    helpers::{get_eval_value, get_ref_value},
    interpreter::{get_var_ptr, value_from_token, BuiltinFunc, EvalValue, Func, VarValue},
    tokenizer::Token,
};

macro_rules! expect_params {
    ($name:expr, $expected:expr, $found:expr) => {
        if $expected != $found {
            panic!(
                "Expected {} params in function `{}`, found {}",
                $expected, $name, $found
            );
        }
    };
}

pub fn init_builtins(
    _vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    functions: &mut Vec<Func>,
) {
    functions.push(Func::Builtin(BuiltinFunc::new(
        "print",
        |vars, params, stdout| {
            for (i, param) in params.iter().enumerate() {
                let to_print = match param {
                    EvalValue::Value(val) => val.get_str(),
                    EvalValue::Token(tok) => match tok {
                        Token::Identifier(ident) => {
                            let var_ptr = get_var_ptr(vars, &ident);
                            let var_borrow = var_ptr.borrow();
                            let var_value = &*var_borrow.value.borrow();
                            var_value.get_str()
                        }
                        _ => value_from_token(tok, None).get_str(),
                    },
                };

                let to_print = to_print.as_str();
                let bytes = to_print.as_bytes();

                stdout.write_all(bytes).unwrap();
                if i < params.len() - 1 {
                    stdout.write_all(b", ").unwrap();
                }
            }
            stdout.write_all(b"\n").unwrap();
            None
        },
    )));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "type",
        |vars, params, _| {
            expect_params!("type", 1, params.len());

            let res = match params[0].to_owned() {
                EvalValue::Token(t) => match t {
                    Token::Identifier(ident) => {
                        let ptr = get_var_ptr(vars, &ident);
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

    functions.push(Func::Builtin(BuiltinFunc::new("ref", |vars, params, _| {
        expect_params!("ref", 1, params.len());

        match &params[0] {
            // cloning here because the value is defined within the ref function and cannot be
            // referenced anywhere else, therefore it is not important to keep the reference
            // with the original value
            EvalValue::Value(val) => Some(Value::Ref(Rc::new(RefCell::new(val.clone())))),
            EvalValue::Token(t) => match t {
                Token::Identifier(ident) => {
                    let val = get_var_ptr(vars, &ident);
                    let val_ref = val.borrow_mut();
                    Some(Value::Ref(Rc::clone(&val_ref.value)))
                }
                _ => panic!("Expected identifier or value to create ref"),
            },
        }
    })));

    functions.push(Func::Builtin(BuiltinFunc::new(
        "clone",
        |vars, params, _| {
            let eval_value = params[0].clone();
            let to_clone = get_eval_value(vars, eval_value, true);
            Some(match to_clone {
                Value::Ref(r) => get_ref_value(&r).borrow().clone(),
                _ => to_clone,
            })
        },
    )))
}
