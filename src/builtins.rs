use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{
    interpreter::{get_var_ptr, value_from_token, BuiltinFunc, EvalValue, Func, VarValue},
    tokenizer::Token,
};

pub fn init_builtins(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    functions: Rc<RefCell<Vec<Func>>>,
) {
    functions
        .borrow_mut()
        .push(Func::Builtin(BuiltinFunc::new("print", |vars, params| {
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
                if i < params.len() - 1 {
                    print!("{}, ", to_print);
                } else {
                    print!("{}", to_print);
                }
            }
            println!();
            None
        })));
}
