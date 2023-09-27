use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::interpreter::VarValue;

pub struct Variables {
    vars: HashMap<String, Vec<Rc<RefCell<VarValue>>>>,
}

impl Variables {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }
    pub fn insert(&mut self, name: String, value: VarValue) {
        self.insert_ptr(name, Rc::new(RefCell::new(value)));
    }
    pub fn insert_ptr(&mut self, name: String, value: Rc<RefCell<VarValue>>) {
        if let Some(var_arr) = self.vars.get_mut(&name) {
            var_arr.push(value);
        } else {
            self.vars.insert(name, vec![value]);
        }
    }
    pub fn get(&self, name: &String, scope: usize) -> Option<Rc<RefCell<VarValue>>> {
        let var_arr = self.vars.get(name);
        let mut max_scope = 0;
        let mut max_scope_index = 0;
        if let Some(var_arr) = var_arr {
            for (i, var) in var_arr.iter().enumerate() {
                let var_ref = &*var.borrow();
                if var_ref.scope == scope {
                    return Some(Rc::clone(&var));
                } else if var_ref.scope < scope && var_ref.scope > max_scope {
                    max_scope = var_ref.scope;
                    max_scope_index = i;
                }
            }

            Some(Rc::clone(&var_arr[max_scope_index]))
        } else {
            None
        }
    }
}
