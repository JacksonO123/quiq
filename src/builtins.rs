use std::{cell::RefCell, collections::HashMap, io::Write, rc::Rc};

use crate::{
    ast::Value,
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
    functions: Rc<RefCell<Vec<Func>>>,
) {
    functions.borrow_mut().push(Func::Builtin(BuiltinFunc::new(
        "print",
        |vars, params, stdout| {
            for (i, param) in params.iter().enumerate() {
                let to_print = match param {
                    EvalValue::Value(val) => val.get_str(),
                    EvalValue::Token(tok) => match tok {
                        Token::Identifier(ident) => {
                            let var_ptr = get_var_ptr(vars, &ident);
                            let var_value = &var_ptr.borrow().value;
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

    functions
        .borrow_mut()
        .push(Func::Builtin(BuiltinFunc::new("type", |_, params, _| {
            expect_params!("type", 1, params.len());

            let res = match params[0].to_owned() {
                EvalValue::Token(t) => String::from(t.get_token_name()),
                EvalValue::Value(v) => v.get_enum_str(),
            };

            Some(Value::String(res))
        })))
}
