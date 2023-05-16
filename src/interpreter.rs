use std::{cell::RefCell, rc::Rc};

use crate::{
    ast::{Ast, AstNode, AstNodeType, Value},
    helpers::set_var_value,
    tokenizer::{Token, TokenType, VarType},
};

pub struct CustomFunc<'a> {
    name: &'a str,
    scope: usize,
    tree: AstNode<'a>,
}
impl<'a> CustomFunc<'a> {
    fn new(name: &'a str, scope: usize, tree: AstNode<'a>) -> Self {
        Self { name, scope, tree }
    }
}

pub struct BuiltinFunc<'a> {
    name: &'a str,
    func: fn(Vec<Value<'a>>) -> Option<Vec<Value<'a>>>,
}
impl<'a> BuiltinFunc<'a> {
    pub fn new(name: &'a str, func: fn(Vec<Value<'a>>) -> Option<Vec<Value<'a>>>) -> Self {
        Self { name, func }
    }
}

pub enum Func<'a> {
    Custom(CustomFunc<'a>),
    Builtin(BuiltinFunc<'a>),
}

pub struct VarValue<'a> {
    pub value: Value<'a>,
    pub scope: usize,
    pub name: &'a str,
}
impl<'a> VarValue<'a> {
    fn new(name: &'a str, value: Value<'a>, scope: usize) -> Self {
        Self { name, value, scope }
    }
    pub fn get_str(&self) -> String {
        format!("Var: {} -> ", self.name).to_owned() + self.value.get_str()
    }
}

fn get_var_value<'a>(vars: &mut Vec<VarValue<'a>>, name: &'a str) -> Value<'a> {
    for var in vars.iter() {
        if var.name == name {
            return var.value.clone();
        }
    }
    panic!("Undefined variable: {}", name);
}

fn value_from_token<'a>(
    vars: &mut Vec<VarValue<'a>>,
    t: Token<'a>,
    value_type: Option<Token<'a>>,
) -> Value<'a> {
    match t.token_type {
        TokenType::Number => {
            if let Some(vt) = value_type {
                match vt.token_type {
                    TokenType::Type(tok_type) => match tok_type {
                        VarType::Int => {
                            let num = t.value.parse::<i32>().unwrap();
                            Value::Int(num)
                        }
                        VarType::Float => {
                            let num = t.value.parse::<f32>().unwrap();
                            Value::Float(num)
                        }
                        VarType::Double => {
                            let num = t.value.parse::<f64>().unwrap();
                            Value::Double(num)
                        }
                        VarType::Long => {
                            let num = t.value.parse::<i64>().unwrap();
                            Value::Long(num)
                        }
                        _ => panic!("Unexpected number type"),
                    },
                    _ => panic!("Unexpected type token"),
                }
            } else {
                let num = t.value.parse::<f64>().unwrap();
                Value::Double(num)
            }
        }
        TokenType::String => Value::String(t.value),
        TokenType::Bool => {
            let val = t.value.parse::<bool>().unwrap();
            Value::Bool(val)
        }
        TokenType::Identifier => get_var_value(vars, t.value),
        _ => panic!("Cannot get value from token: {}", t.get_str()),
    }
}

fn call_func<'a>(
    vars: &mut Vec<VarValue<'a>>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    scope: usize,
    name: &'a str,
    args: Vec<AstNode<'a>>,
) -> Option<Token<'a>> {
    for func in functions.borrow().iter() {
        match func {
            Func::Custom(custom) => {
                if custom.name == name {
                    unimplemented!()
                }
            }
            Func::Builtin(builtin) => {
                if builtin.name == name {
                    let f = builtin.func;
                    let args = args.iter().map(|a| {
                        let res = eval_node(vars, Rc::clone(&functions), scope, a.clone());
                        if let Some(tok) = res {
                            value_from_token(vars, tok, None)
                        } else {
                            panic!("Cannot pass void type as parameter to function")
                        }
                    });
                    let args: Vec<Value> = args.collect();
                    f(args);
                }
            }
        }
    }
    None
}

/// evaluate ast node
/// return type is single value
/// Ex: AstNode::Token(bool) -> Token(bool)
fn eval_node<'a>(
    vars: &mut Vec<VarValue<'a>>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    scope: usize,
    node: AstNode<'a>,
) -> Option<Token<'a>> {
    match node.node_type.clone() {
        AstNodeType::StatementSeq(seq) => {
            for node in seq.iter() {
                eval_node(vars, Rc::clone(&functions), scope, node.borrow().clone());
            }
            None
        }
        AstNodeType::MakeVar(var_type, name, value) => {
            let value = eval_node(vars, functions, scope, *value);
            if let Some(tok) = value {
                let val = value_from_token(vars, tok, Some(var_type));
                let var = VarValue::new(name.value, val, scope);
                vars.push(var);
            } else {
                panic!("Expected {} found void", var_type.get_str());
            }
            None
        }
        AstNodeType::SetVar(name, value) => {
            let value = eval_node(vars, functions, scope, *value);
            if let Some(tok) = value {
                let val = value_from_token(vars, tok, None);
                set_var_value(vars, name.value, val);
            }
            None
        }
        AstNodeType::Token(token) => Some(token),
        AstNodeType::CallFunc(name, args) => call_func(vars, functions, scope, name, args),
    }
}

pub fn eval<'a>(
    vars: &mut Vec<VarValue<'a>>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    tree: Ast<'a>,
) {
    let root_node = tree.node.borrow().clone();
    eval_node(vars, functions, 0, root_node);
}
