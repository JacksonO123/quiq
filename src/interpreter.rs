use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{
    ast::{Ast, AstNode, AstNodeType, Value},
    helpers::{
        cast, ensure_type, flatten_exp, get_array_type, get_eval_value, push_to_array,
        set_var_value, ExpValue,
    },
    tokenizer::{OperatorType, Token},
};

pub struct CustomFunc<'a> {
    name: &'a str,
    // scope: usize,
    // tree: AstNode<'a>,
}
// impl<'a> CustomFunc<'a> {
//     fn new(name: &'a str, scope: usize, tree: AstNode<'a>) -> Self {
//         Self { name, scope, tree }
//     }
// }

pub struct BuiltinFunc<'a> {
    name: &'a str,
    func: fn(&mut HashMap<String, Rc<RefCell<VarValue>>>, Vec<EvalValue>) -> Option<Vec<Value>>,
}
impl<'a> BuiltinFunc<'a> {
    pub fn new(
        name: &'a str,
        func: fn(&mut HashMap<String, Rc<RefCell<VarValue>>>, Vec<EvalValue>) -> Option<Vec<Value>>,
    ) -> Self {
        Self { name, func }
    }
}

pub enum Func<'a> {
    Custom(CustomFunc<'a>),
    Builtin(BuiltinFunc<'a>),
}

#[derive(Clone, Debug)]
pub enum VarType {
    Usize,
    Int,
    Float,
    Double,
    Long,
    String,
    Bool,
    Array(Box<VarType>),
    Unknown,
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
            "usize" => VarType::Usize,
            _ => panic!("Unable to infer var type from: {}", string),
        }
    }
    pub fn get_str(&self) -> &str {
        match self {
            VarType::Int => "int",
            VarType::Usize => "usize",
            VarType::Float => "float",
            VarType::Double => "double",
            VarType::Long => "long",
            VarType::String => "string",
            VarType::Bool => "bool",
            VarType::Array(_) => "arr",
            VarType::Unknown => "unknown",
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

pub fn get_var_ptr<'a>(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    name: &String,
) -> Rc<RefCell<VarValue>> {
    let res_option = vars.get(name);
    if let Some(res) = res_option {
        return Rc::clone(&res);
    }
    panic!("Undefined variable: {}", name);
}

pub fn value_from_token<'a>(t: &Token, value_type: Option<&VarType>) -> Value {
    match t {
        Token::Number(n) => {
            if let Some(vt) = value_type {
                match vt {
                    VarType::Int => {
                        let num = n.parse::<i32>().unwrap();
                        Value::Int(num)
                    }
                    VarType::Float => {
                        let num = n.parse::<f32>().unwrap();
                        Value::Float(num)
                    }
                    VarType::Double => {
                        let num = n.parse::<f64>().unwrap();
                        Value::Double(num)
                    }
                    VarType::Usize => {
                        let num = n.parse::<usize>().unwrap();
                        Value::Usize(num)
                    }
                    VarType::Long => {
                        let num = n.parse::<i64>().unwrap();
                        Value::Long(num)
                    }
                    _ => panic!("Unexpected type token"),
                }
            } else {
                if n.contains('.') {
                    let num = n.parse::<f64>().unwrap();
                    Value::Double(num)
                } else {
                    let num = n.parse::<i32>().unwrap();
                    Value::Int(num)
                }
            }
        }
        Token::String(s) => Value::String(s.to_string()),
        Token::Bool(b) => Value::Bool(*b),
        Token::Identifier(_) => {
            panic!("Cannot get token value of identifier");
        }
        _ => panic!("Cannot get value from token: {:?}", t),
    }
}

fn call_func<'a>(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    scope: usize,
    name: &String,
    args: &Vec<AstNode<'a>>,
) -> Option<Token<'a>> {
    let mut found = false;
    for func in functions.borrow().iter() {
        match func {
            Func::Custom(custom) => {
                if custom.name == name {
                    unimplemented!()
                }
            }
            Func::Builtin(builtin) => {
                if builtin.name == name {
                    found = true;
                    let f = builtin.func;
                    let args = args.iter().map(|a| {
                        let res = eval_node(vars, Rc::clone(&functions), scope, a);
                        if let Some(val) = res {
                            val
                        } else {
                            panic!("Cannot pass void type as parameter to function")
                        }
                    });
                    let args: Vec<EvalValue> = args.collect();
                    f(vars, args);
                }
            }
        }
    }

    if !found {
        panic!("Unknown function: {}", name);
    }

    None
}

#[derive(Debug, Clone)]
pub enum EvalValue<'a> {
    Value(Value),
    Token(Token<'a>),
}

pub fn eval_exp<'a>(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    scope: usize,
    exp: &Vec<Box<AstNode<'a>>>,
) -> EvalValue<'a> {
    let mut flattened = flatten_exp(vars, functions, scope, exp);

    let pemdas_operations: [OperatorType; 4] = [
        OperatorType::Mult,
        OperatorType::Div,
        OperatorType::Add,
        OperatorType::Sub,
    ];

    for current_op in pemdas_operations.iter() {
        let mut i = 0;
        while i < flattened.len() {
            match &flattened[i].as_ref().unwrap() {
                ExpValue::Operator(tok) => {
                    match tok {
                        Token::Operator(op_type) => {
                            let left = &flattened[i - 1];
                            let right = &flattened[i + 1];

                            let left = match left.as_ref().unwrap() {
                                ExpValue::Value(val) => val,
                                ExpValue::Operator(_) => panic!("Unexpected operator"),
                            };
                            let right = match right.as_ref().unwrap() {
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
                                            Value::Usize(l) => match right {
                                                Value::Usize(r) => Some(Value::Usize(l * r)),
                                                _ => panic!("Cannot multiply non-number values, or numbers of different types"),
                                            }
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
                                                _ => panic!("Cannot divide non-number values, or numbers of different types"),
                                            },
                                            Value::Float(l) => match right {
                                                Value::Float(r) => Some(Value::Float(l / r)),
                                                _ => panic!("Cannot divide non-number values, or numbers of different types"),
                                            },
                                            Value::Double(l) => match right {
                                                Value::Double(r) => Some(Value::Double(l / r)),
                                                _ => panic!("Cannot divide non-number values, or numbers of different types"),
                                            },
                                            Value::Long(l) => match right {
                                                Value::Long(r) => Some(Value::Long(l / r)),
                                                _ => panic!("Cannot divide non-number values, or numbers of different types"),
                                            },
                                            Value::Usize(l) => match right {
                                                Value::Usize(r) => Some(Value::Usize(l / r)),
                                                _ => panic!("Cannot divide non-number values, or numbers of different types"),
                                            }
                                            _ => panic!("Cannot divide non-number values, or numbers of different types"),
                                        }
                                        _ => None
                                    }
                                }
                                OperatorType::Add => {
                                    match op_type {
                                        OperatorType::Add => match left {
                                            Value::Int(l) => match right {
                                                Value::Int(r) => Some(Value::Int(l + r)),
                                                _ => panic!("Cannot add non-number values, or numbers of different types"),
                                            },
                                            Value::Float(l) => match right {
                                                Value::Float(r) => Some(Value::Float(l + r)),
                                                _ => panic!("Cannot add non-number values, or numbers of different types"),
                                            },
                                            Value::Double(l) => match right {
                                                Value::Double(r) => Some(Value::Double(l + r)),
                                                _ => panic!("Cannot add non-number values, or numbers of different types"),
                                            },
                                            Value::Long(l) => match right {
                                                Value::Long(r) => Some(Value::Long(l + r)),
                                                _ => panic!("Cannot add non-number values, or numbers of different types"),
                                            },
                                            Value::Usize(l) => match right {
                                                Value::Usize(r) => Some(Value::Usize(l + r)),
                                                _ => panic!("Cannot add non-number values, or numbers of different types"),
                                            }
                                            _ => panic!("Cannot add non-number values, or numbers of different types"),
                                        }
                                        _ => None
                                    }
                                }
                                OperatorType::Sub => {
                                    match op_type {
                                        OperatorType::Sub => match left {
                                            Value::Int(l) => match right {
                                                Value::Int(r) => Some(Value::Int(l - r)),
                                                _ => panic!("Cannot subtract non-number values, or numbers of different types"),
                                            },
                                            Value::Float(l) => match right {
                                                Value::Float(r) => Some(Value::Float(l - r)),
                                                _ => panic!("Cannot subtract non-number values, or numbers of different types"),
                                            },
                                            Value::Double(l) => match right {
                                                Value::Double(r) => Some(Value::Double(l - r)),
                                                _ => panic!("Cannot subtract non-number values, or numbers of different types"),
                                            },
                                            Value::Long(l) => match right {
                                                Value::Long(r) => Some(Value::Long(l - r)),
                                                _ => panic!("Cannot subtract non-number values, or numbers of different types"),
                                            },
                                            Value::Usize(l) => match right {
                                                Value::Usize(r) => Some(Value::Usize(l - r)),
                                                _ => panic!("Cannot subtract non-number values, or numbers of different types"),
                                            }
                                            _ => panic!("Cannot subtract non-number values, or numbers of different types"),
                                        }
                                        _ => None
                                    }
                                }
                            };

                            if let Some(val) = new_value {
                                flattened[i - 1] = Some(ExpValue::Value(val));
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

    match flattened[0].take().unwrap() {
        ExpValue::Value(val) => EvalValue::Value(val),
        _ => panic!("Invalid token resulting from expression"),
    }
}

/// evaluate ast node
/// return type is single value
/// Ex: AstNode::Token(bool) -> Token(bool)
pub fn eval_node<'a>(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    scope: usize,
    node: &AstNode<'a>,
) -> Option<EvalValue<'a>> {
    match &node.node_type {
        AstNodeType::Cast(var_type, node) => {
            let res_option = eval_node(vars, Rc::clone(&functions), scope, node.as_ref());

            if let Some(eval_value) = res_option {
                let val = get_eval_value(eval_value);
                let casted_value = cast(var_type, val);
                Some(EvalValue::Value(casted_value))
            } else {
                panic!("Error casting value")
            }
        }
        AstNodeType::AccessStructProp(struct_token, prop) => {
            let var_ptr = if let Token::Identifier(ident) = struct_token {
                get_var_ptr(vars, &ident)
            } else {
                panic!("Error in struct property access, expected identifier");
            };

            let mut var_ref = var_ptr.borrow_mut();
            let var_value = &mut var_ref.value;

            let res = match var_value {
                Value::Array(ref mut vals) => match &prop
                    .get(0)
                    .expect("Expected property to access on Array")
                    .node_type
                {
                    AstNodeType::CallFunc(name, args) => match name.as_str() {
                        "push" => {
                            push_to_array(vars, Rc::clone(&functions), scope, vals, args);
                            None
                        }
                        _ => panic!("Unknown array method: {}", name),
                    },
                    AstNodeType::Token(t) => match t {
                        Token::Identifier(ident) => match ident.as_str() {
                            "length" => Some(EvalValue::Value(Value::Usize(vals.len()))),
                            _ => unimplemented!(),
                        },
                        _ => panic!("Unexpected method: {:?}", t),
                    },
                    _ => panic!("Unexpected operation: {:?}", prop),
                },
                _ => unimplemented!(),
            };

            res
        }
        AstNodeType::Array(arr_nodes) => {
            let mut res_arr: Vec<Value> = Vec::new();

            for node in arr_nodes.iter() {
                let res_option = eval_node(vars, Rc::clone(&functions), scope, node);

                if let Some(res) = res_option {
                    let val = get_eval_value(res);
                    res_arr.push(val);
                }
            }
            Some(EvalValue::Value(Value::Array(res_arr)))
        }
        AstNodeType::If(condition, node) => {
            let condition_res = eval_node(vars, Rc::clone(&functions), scope, condition.as_ref());
            if let Some(res) = condition_res {
                let condition_val = match res {
                    EvalValue::Value(val) => match val {
                        Value::Bool(b) => b,
                        _ => panic!("Error in `if`, expected boolean"),
                    },
                    EvalValue::Token(tok) => match tok {
                        Token::Identifier(ident) => {
                            let var_ptr = get_var_ptr(vars, &ident);
                            let val_ref = var_ptr.borrow();
                            let val = &val_ref.value;
                            match val {
                                Value::Bool(b) => *b,
                                _ => panic!("Error in `if`, expected boolean"),
                            }
                        }
                        _ => {
                            let token_value = value_from_token(&tok, None);
                            match token_value {
                                Value::Bool(b) => b,
                                _ => panic!("Error in `if`, expected boolean"),
                            }
                        }
                    },
                };

                if condition_val {
                    eval_node(vars, Rc::clone(&functions), scope + 1, node.as_ref());
                }
            } else {
                panic!("Expected if condition to be type boolean");
            }
            None
        }
        AstNodeType::StatementSeq(seq) => {
            for node in seq.iter() {
                eval_node(
                    vars,
                    Rc::clone(&functions),
                    scope,
                    &node.borrow().to_owned(),
                );
            }
            None
        }
        AstNodeType::MakeVar(var_type, name, value) => {
            let value = eval_node(vars, functions, scope, value.as_ref());
            if let Some(tok) = value {
                let val = match tok {
                    EvalValue::Token(t) => value_from_token(&t, Some(var_type)),
                    EvalValue::Value(v) => {
                        if ensure_type(&var_type, &v) {
                            v
                        } else {
                            panic!(
                                "Unexpected variable type definition, expected {:?} found {:?}",
                                match v {
                                    Value::Array(arr) =>
                                        VarType::Array(Box::new(get_array_type(&arr))),
                                    _ => panic!(),
                                },
                                var_type
                            )
                        }
                    }
                };

                let var_name = if let Token::Identifier(ident) = name {
                    ident
                } else {
                    panic!("Expected identifier for variable name");
                };

                let var = Rc::new(RefCell::new(VarValue::new(var_name.clone(), val, scope)));
                vars.insert(var_name.clone(), var);
            } else {
                panic!("Expected {} found void", var_type.get_str());
            }
            None
        }
        AstNodeType::SetVar(name, value) => {
            let value = eval_node(vars, functions, scope, value.as_ref());
            if let Some(tok) = value {
                let val = match tok {
                    EvalValue::Token(t) => value_from_token(&t, None),
                    EvalValue::Value(v) => v,
                };

                let var_name = if let Token::Identifier(ident) = name {
                    ident.clone()
                } else {
                    panic!("Expected identifier for setting variable");
                };

                set_var_value(vars, var_name, val);
            }
            None
        }
        AstNodeType::Token(token) => Some(EvalValue::Token(token.to_owned())),
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
            let val = eval_node(vars, functions, scope, node.as_ref());
            let b = if let Some(value) = val {
                match value {
                    EvalValue::Value(v) => match v {
                        Value::Bool(b) => b,
                        _ => panic!("Cannot ! non-boolean type"),
                    },
                    EvalValue::Token(t) => {
                        let token_value = value_from_token(&t, None);
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

pub fn eval<'a>(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    tree: Ast<'a>,
) {
    let root_node = tree.node.borrow();
    eval_node(vars, functions, 0, &root_node.to_owned());
}
