use std::{cell::RefCell, rc::Rc};

use crate::interpreter::{BuiltinFunc, Func, VarValue};

pub fn init_builtins(vars: &mut Vec<VarValue>, functions: Rc<RefCell<Vec<Func>>>) {
    functions
        .borrow_mut()
        .push(Func::Builtin(BuiltinFunc::new("print", |params| {
            for (i, param) in params.iter().enumerate() {
                if i < params.len() - 1 {
                    print!("{}, ", param.get_str());
                } else {
                    print!("{}", param.get_str());
                }
            }
            println!();
            None
        })));
}
