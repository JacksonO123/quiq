use std::{cell::RefCell, rc::Rc};

use crate::{
    ast::{Ast, AstNode, AstNodeType, Value},
    helpers::{ensure_type, flatten_exp, set_var_value, ExpValue},
    tokenizer::{OperatorType, Token, TokenType},
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
    func: fn(Vec<Value>) -> Option<Vec<Value>>,
}
impl<'a> BuiltinFunc<'a> {
    pub fn new(name: &'a str, func: fn(Vec<Value>) -> Option<Vec<Value>>) -> Self {
        Self { name, func }
    }
}

pub enum Func<'a> {
    Custom(CustomFunc<'a>),
    Builtin(BuiltinFunc<'a>),
}

#[derive(Clone, Debug)]
pub enum VarType {
    Int,
    Float,
    Double,
    Long,
    String,
    Bool,
    Array(Box<VarType>),
}
impl VarType {
    pub fn from(string: &str) -> Self {
        match string {
            "int" => VarType::Int,
            "float" => VarType::Float,
            "double" => VarType::Double,
            "long" => VarType::Long,
            "string" => VarType::String,
            "bool" => VarType::Bool,
            _ => panic!("Inable to infer var type from: {}", string),
        }
    }
    pub fn get_str(&self) -> &str {
        match self {
            VarType::Int => "int",
            VarType::Float => "float",
            VarType::Double => "double",
            VarType::Long => "long",
            VarType::String => "string",
            VarType::Bool => "bool",
            VarType::Array(_) => "arr",
        }
    }
}

pub struct VarValue {
    pub value: Value,
    pub scope: usize,
    pub name: String,
}
impl VarValue {
    fn new(name: String, value: Value, scope: usize) -> Self {
        Self { name, value, scope }
    }
    pub fn get_str(&self) -> String {
        format!("Var: {} -> ", self.name).to_owned() + self.value.get_str().as_str()
    }
}

fn get_var_value<'a>(vars: &mut Vec<VarValue>, name: String) -> Value {
    for var in vars.iter() {
        if var.name == name {
            return var.value.clone();
        }
    }
    panic!("Undefined variable: {}", name);
}

pub fn value_from_token<'a>(
    vars: &mut Vec<VarValue>,
    t: Token,
    value_type: Option<VarType>,
) -> Value {
    match t.token_type {
        TokenType::Number => {
            if let Some(vt) = value_type {
                match vt {
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
                    _ => panic!("Unexpected type token"),
                }
            } else {
                if t.value.contains('.') {
                    let num = t.value.parse::<f64>().unwrap();
                    Value::Double(num)
                } else {
                    let num = t.value.parse::<i32>().unwrap();
                    Value::Int(num)
                }
            }
        }
        TokenType::String => Value::String(t.value.clone()),
        TokenType::Bool => {
            let val = t.value.parse::<bool>().unwrap();
            Value::Bool(val)
        }
        TokenType::Identifier => get_var_value(vars, t.value),
        _ => panic!("Cannot get value from token: {}", t.get_str()),
    }
}

fn call_func<'a>(
    vars: &mut Vec<VarValue>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    scope: usize,
    name: String,
    args: Vec<AstNode<'a>>,
) -> Option<Token> {
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
                        if let Some(val) = res {
                            match val {
                                EvalValue::Token(tok) => value_from_token(vars, tok, None),
                                EvalValue::Value(v) => v,
                            }
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

#[derive(Debug)]
pub enum EvalValue {
    Value(Value),
    Token(Token),
}

pub fn eval_exp<'a>(
    vars: &mut Vec<VarValue>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    scope: usize,
    exp: Vec<Box<AstNode<'a>>>,
) -> EvalValue {
    let mut flattened = flatten_exp(vars, functions, scope, exp);

    // TODO: possibly add exponents

    let pemdas_operations: [OperatorType; 4] = [
        OperatorType::Mult,
        OperatorType::Div,
        OperatorType::Add,
        OperatorType::Sub,
    ];

    for current_op in pemdas_operations.iter() {
        let mut i = 0;
        while i < flattened.len() {
            match flattened[i].clone() {
                ExpValue::Operator(tok) => {
                    match tok {
                        TokenType::Operator(op_type) => {
                            let left = flattened[i - 1].clone();
                            let right = flattened[i + 1].clone();

                            let left = match left {
                                ExpValue::Value(val) => val,
                                ExpValue::Operator(_) => panic!("Unexpected operator"),
                            };
                            let right = match right {
                                ExpValue::Value(val) => val,
                                ExpValue::Operator(_) => panic!("Unexpected operator"),
                            };

                            let new_value = match current_op {
                                OperatorType::Mult => {
                                    match op_type {
                                        OperatorType::Mult => match left {
                                            Value::Int(l) => match right {
                                                Value::Int(r) => Some(Value::Int(l * r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            Value::Float(l) => match right {
                                                Value::Float(r) => Some(Value::Float(l * r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            Value::Double(l) => match right {
                                                Value::Double(r) => Some(Value::Double(l * r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            Value::Long(l) => match right {
                                                Value::Long(r) => Some(Value::Long(l * r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                        }
                                        _ => None
                                    }
                                }
                                OperatorType::Div => {
                                    match op_type {
                                        OperatorType::Div => match left {
                                            Value::Int(l) => match right {
                                                Value::Int(r) => Some(Value::Int(l / r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            Value::Float(l) => match right {
                                                Value::Float(r) => Some(Value::Float(l / r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            Value::Double(l) => match right {
                                                Value::Double(r) => Some(Value::Double(l / r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            Value::Long(l) => match right {
                                                Value::Long(r) => Some(Value::Long(l / r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                        }
                                        _ => None
                                    }
                                }
                                OperatorType::Add => {
                                    match op_type {
                                        OperatorType::Add => match left {
                                            Value::Int(l) => match right {
                                                Value::Int(r) => Some(Value::Int(l + r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            Value::Float(l) => match right {
                                                Value::Float(r) => Some(Value::Float(l + r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            Value::Double(l) => match right {
                                                Value::Double(r) => Some(Value::Double(l + r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            Value::Long(l) => match right {
                                                Value::Long(r) => Some(Value::Long(l + r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                        }
                                        _ => None
                                    }
                                }
                                OperatorType::Sub => {
                                    match op_type {
                                        OperatorType::Sub => match left {
                                            Value::Int(l) => match right {
                                                Value::Int(r) => Some(Value::Int(l - r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            Value::Float(l) => match right {
                                                Value::Float(r) => Some(Value::Float(l - r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            Value::Double(l) => match right {
                                                Value::Double(r) => Some(Value::Double(l - r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            Value::Long(l) => match right {
                                                Value::Long(r) => Some(Value::Long(l - r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            },
                                            _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                        }
                                        _ => None
                                    }
                                }
                            };

                            if let Some(val) = new_value {
                                flattened[i - 1] = ExpValue::Value(val);
                                flattened.remove(i);
                                flattened.remove(i);
                                i -= 1;
                            }
                        }
                        _ => {}
                    };
                }
                _ => {}
            }

            i += 1;
        }
    }

    match flattened[0].clone() {
        ExpValue::Value(val) => EvalValue::Value(val),
        _ => panic!("Invalid token resulting from expression"),
    }
}

/// evaluate ast node
/// return type is single value
/// Ex: AstNode::Token(bool) -> Token(bool)
pub fn eval_node<'a>(
    vars: &mut Vec<VarValue>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    scope: usize,
    node: AstNode<'a>,
) -> Option<EvalValue> {
    match node.node_type.clone() {
        AstNodeType::Array(arr_nodes) => {
            let mut res_arr: Vec<Value> = Vec::new();

            for node in arr_nodes.iter() {
                let res_option = eval_node(vars, Rc::clone(&functions), scope, node.clone());
                if let Some(res) = res_option {
                    let val = match res {
                        EvalValue::Value(v) => v,
                        EvalValue::Token(t) => match t.token_type {
                            TokenType::Number => value_from_token(vars, t, None),
                            TokenType::String => value_from_token(vars, t, None),
                            TokenType::Bool => value_from_token(vars, t, None),
                            TokenType::Identifier => value_from_token(vars, t, None),
                            _ => panic!("Unexpected token: {}", t.value),
                        },
                    };
                    res_arr.push(val);
                }
            }
            Some(EvalValue::Value(Value::Array(res_arr)))
        }
        AstNodeType::CreateArr(val) => Some(EvalValue::Value(val)),
        AstNodeType::If(condition, node) => {
            let condition_res = eval_node(
                vars,
                Rc::clone(&functions),
                scope,
                condition.as_ref().clone(),
            );
            if let Some(res) = condition_res {
                let condition_val = match res {
                    EvalValue::Value(val) => match val {
                        Value::Bool(b) => b,
                        _ => panic!("Error in `if`, expected boolean"),
                    },
                    EvalValue::Token(tok) => {
                        let tok_val = value_from_token(vars, tok, None);
                        match tok_val {
                            Value::Bool(b) => b,
                            _ => panic!("Error in `if`, expected boolean"),
                        }
                    }
                };

                if condition_val {
                    eval_node(
                        vars,
                        Rc::clone(&functions),
                        scope + 1,
                        node.as_ref().clone(),
                    );
                }
            } else {
                panic!("Expected if condition to be type boolean");
            }
            None
        }
        AstNodeType::StatementSeq(seq) => {
            for node in seq.iter() {
                eval_node(vars, Rc::clone(&functions), scope, node.borrow().clone());
            }
            None
        }
        AstNodeType::MakeVar(var_type, name, value) => {
            let value = eval_node(vars, functions, scope, *value);
            if let Some(tok) = value {
                let val = match tok {
                    EvalValue::Token(t) => value_from_token(vars, t, Some(var_type)),
                    EvalValue::Value(v) => ensure_type(var_type, v).unwrap(),
                };
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
                let val = match tok {
                    EvalValue::Token(t) => value_from_token(vars, t, None),
                    EvalValue::Value(v) => v,
                };
                set_var_value(vars, name.value, val);
            }
            None
        }
        AstNodeType::Token(token) => Some(EvalValue::Token(token)),
        AstNodeType::CallFunc(name, args) => {
            let func_res = call_func(vars, functions, scope, name, args);
            if let Some(tok) = func_res {
                Some(EvalValue::Token(tok))
            } else {
                None
            }
        }
        AstNodeType::Exp(nodes) => Some(eval_exp(vars, Rc::clone(&functions), scope, nodes)),
        AstNodeType::Bang(node) => {
            let val = eval_node(vars, functions, scope, node.as_ref().clone());
            let b = if let Some(value) = val {
                match value {
                    EvalValue::Value(v) => match v {
                        Value::Bool(b) => b,
                        _ => panic!("Cannot ! non-boolean type"),
                    },
                    EvalValue::Token(t) => {
                        let token_value = value_from_token(vars, t, None);
                        match token_value {
                            Value::Bool(b) => b,
                            _ => panic!("Cannot ! non-boolean type"),
                        }
                    }
                }
            } else {
                panic!("Expected value to !");
            };
            Some(EvalValue::Value(Value::Bool(!b)))
        }
    }
}

pub fn eval<'a>(vars: &mut Vec<VarValue>, functions: Rc<RefCell<Vec<Func<'a>>>>, tree: Ast<'a>) {
    let root_node = tree.node.borrow().clone();
    eval_node(vars, functions, 0, root_node);
}
