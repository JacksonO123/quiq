use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::interpreter::VarValue;

#[derive(Debug)]
struct VarFrame {
    vars: HashMap<String, Vec<Rc<RefCell<VarValue>>>>,
}
impl VarFrame {
    fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }
}

pub struct Variables {
    frames: Vec<VarFrame>,
}

impl Variables {
    pub fn new() -> Self {
        Self {
            frames: vec![VarFrame::new()],
        }
    }
    pub fn create_frame(&mut self) {
        self.frames.push(VarFrame::new());
    }
    pub fn insert(&mut self, name: String, value: VarValue) {
        self.insert_ptr(name, Rc::new(RefCell::new(value)));
    }
    pub fn insert_ptr(&mut self, name: String, value: Rc<RefCell<VarValue>>) {
        let var_frame = self.frames.last_mut().unwrap();
        if let Some(var_arr) = var_frame.vars.get_mut(&name) {
            var_arr.push(value);
        } else {
            var_frame.vars.insert(name, vec![value]);
        }
    }
    pub fn get(&self, name: &String, scope: usize) -> Option<Rc<RefCell<VarValue>>> {
        let var_frame = self.frames.last().unwrap();
        let var_arr = var_frame.vars.get(name);
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

            if max_scope_index < var_arr.len() {
                Some(Rc::clone(&var_arr[max_scope_index]))
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn free(&mut self, name: &String, scope: usize) {
        let var_frame = self.frames.last_mut().unwrap();
        let var_arr = var_frame.vars.get_mut(name);

        let mut max_scope = 0;
        let mut max_scope_index = 0;
        let mut found = false;

        if let Some(var_arr) = var_arr {
            let mut i = 0;

            while i < var_arr.len() {
                if var_arr[i].borrow().scope == scope {
                    var_arr.remove(i);
                    if i > 0 {
                        i -= 1;
                    }
                    found = true;
                } else if var_arr[i].borrow().scope > max_scope && var_arr[i].borrow().scope < scope
                {
                    max_scope = var_arr[i].borrow().scope;
                    max_scope_index = i;
                }

                i += 1;
            }

            if !found {
                var_arr.remove(max_scope_index);
            }
        }
    }
    pub fn print(&self) {
        println!("{:#?}", self.frames);
    }
}
