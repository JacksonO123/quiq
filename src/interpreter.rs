use std::{cell::RefCell, collections::HashMap, io::Stdout, rc::Rc};

use crate::{
    ast::{Ast, AstNode, FuncParam, StructShape, Value},
    helpers::{
        cast, compare, ensure_type, flatten_exp, get_eval_value, get_prop_ptr, get_ref_value,
        index_arr_var_value, make_var, push_to_array, set_index_arr, set_var_value, ExpValue,
    },
    tokenizer::{OperatorType, Token},
};

#[derive(Debug)]
pub struct StructInfo {
    pub available_structs: HashMap<String, StructShape>,
    pub structs: HashMap<String, VarValue>,
}
impl StructInfo {
    pub fn new() -> Self {
        Self {
            available_structs: HashMap::new(),
            structs: HashMap::new(),
        }
    }
    pub fn add_available_struct(&mut self, name: String, shape: StructShape) {
        self.available_structs.insert(name, shape);
    }
}

#[derive(Clone, Debug)]
pub struct StructProp {
    pub name: String,
    pub value: Rc<RefCell<Value>>,
}
impl StructProp {
    pub fn new(name: String, value: Value) -> Self {
        let ptr = Rc::new(RefCell::new(value));
        StructProp { name, value: ptr }
    }
    pub fn get_str(&self) -> String {
        self.value.as_ref().borrow().get_str()
    }
}

#[derive(Clone, Debug)]
pub struct CustomFunc {
    name: String,
    scope: usize,
    block: AstNode,
    return_type: VarType,
    params: Vec<FuncParam>,
}
impl CustomFunc {
    pub fn new(name: String, params: Vec<FuncParam>, return_type: VarType, block: AstNode) -> Self {
        Self {
            name,
            scope: 0,
            params,
            return_type,
            block,
        }
    }
    pub fn set_scope(&mut self, scope: usize) {
        self.scope = scope;
    }
    pub fn call<'a>(
        &self,
        vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
        functions: Rc<RefCell<Vec<Func<'a>>>>,
        structs: &mut StructInfo,
        scope: usize,
        stdout: &mut Stdout,
        args: &Vec<AstNode>,
    ) -> Option<Value> {
        if args.len() != self.params.len() {
            panic!("Wrong number of arguments to func \"{}\"", self.name);
        }

        for (i, param) in self.params.iter().enumerate() {
            make_var(
                vars,
                Rc::clone(&functions),
                structs,
                scope,
                &param.param_type,
                &Token::Identifier(param.name.clone()),
                &Some(Box::new(args[i].clone())),
                stdout,
            )
        }

        let (_, quit) = eval_node(
            vars,
            Rc::clone(&functions),
            structs,
            scope + 1,
            &self.block,
            stdout,
        );

        if let VarType::Void = self.return_type {
            if quit.is_some() {
                panic!("Expected void return value")
            }
        }

        match quit {
            Some(q) => match q {
                QuitType::Return(eval_val) => {
                    let val = get_eval_value(vars, eval_val, true);

                    if ensure_type(&self.return_type, &val) {
                        return Some(val);
                    } else {
                        panic!(
                            "Expected return type {:?} found {:?}",
                            self.return_type, val
                        );
                    }
                }
                _ => {}
            },
            None => {}
        }

        None
    }
}

pub struct BuiltinFunc<'a> {
    name: &'a str,
    func: fn(
        &mut HashMap<String, Rc<RefCell<VarValue>>>,
        Vec<EvalValue>,
        &mut Stdout,
    ) -> Option<Value>,
}
impl<'a> BuiltinFunc<'a> {
    pub fn new(
        name: &'a str,
        func: fn(
            &mut HashMap<String, Rc<RefCell<VarValue>>>,
            Vec<EvalValue>,
            &mut Stdout,
        ) -> Option<Value>,
    ) -> Self {
        Self { name, func }
    }
}

pub enum Func<'a> {
    Custom(Rc<RefCell<CustomFunc>>),
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
    /// struct type name, shape
    Struct(String, StructShape),
    Null,
    Ref(Box<VarType>),
    Void,
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
            VarType::Ref(_) => "ref",
            VarType::Int => "int",
            VarType::Usize => "usize",
            VarType::Float => "float",
            VarType::Double => "double",
            VarType::Long => "long",
            VarType::String => "string",
            VarType::Bool => "bool",
            VarType::Array(_) => "arr",
            VarType::Struct(_, _) => "struct",
            VarType::Null => "null",
            VarType::Void => "void",
        }
    }
}

#[derive(Debug)]
pub struct VarValue {
    pub value: Rc<RefCell<Value>>,
    pub scope: usize,
    pub name: String,
}
impl VarValue {
    pub fn new(name: String, value: Value, scope: usize) -> Self {
        let ptr = Rc::new(RefCell::new(value));
        Self {
            name,
            value: ptr,
            scope,
        }
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
    structs: &mut StructInfo,
    scope: usize,
    name: &String,
    args: &Vec<AstNode>,
    stdout: &mut Stdout,
) -> Option<Value> {
    for func in functions.borrow().iter() {
        match func {
            Func::Custom(custom) => {
                if &custom.borrow().name == name {
                    return custom.borrow().call(
                        vars,
                        Rc::clone(&functions),
                        structs,
                        scope + 1,
                        stdout,
                        args,
                    );
                }
            }
            Func::Builtin(builtin) => {
                if builtin.name == name {
                    let f = builtin.func;

                    let args = args.iter().map(|arg| {
                        let res =
                            eval_node(vars, Rc::clone(&functions), structs, scope, arg, stdout);
                        if let (Some(val), _) = res {
                            val
                        } else {
                            panic!("Cannot pass void type as parameter to function")
                        }
                    });
                    let args: Vec<EvalValue> = args.collect();

                    return f(vars, args, stdout);
                }
            }
        }
    }

    panic!("Unknown function: {}", name);
}

#[derive(Debug, Clone)]
pub enum EvalValue {
    Value(Value),
    Token(Token),
}

macro_rules! expr {
    ($left:expr, $right:expr, $ch:tt, $($variants:ident),+) => {
        match $left {
            $(
                Value::$variants(l) => match $right {
                    Value::$variants(r) => Some(Value::$variants(l $ch r)),
                    _ => panic!("Cannot {} values of types {:?} and {:?}", stringify!($ch), $left, $right)
                }
            )*
            _ => panic!("Cannot {} values of types {:?} and {:?}", stringify!($ch), $left, $right)
        }
    };
}

macro_rules! expr_abstracted {
    ($left:expr, $right:expr, $ch:tt) => {
        expr!($left, $right, $ch, Int, Float, Double, Long, Usize)
    };
}

pub fn eval_exp<'a>(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    structs: &mut StructInfo,
    scope: usize,
    exp: &Vec<Box<AstNode>>,
    stdout: &mut Stdout,
) -> EvalValue {
    let mut flattened = flatten_exp(vars, functions, structs, scope, exp, stdout);

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
                                OperatorType::Mult => match op_type {
                                    OperatorType::Mult => expr_abstracted!(left, right, *),
                                    _ => None,
                                },
                                OperatorType::Div => match op_type {
                                    OperatorType::Div => expr_abstracted!(left, right, /),
                                    _ => None,
                                },
                                OperatorType::Add => match op_type {
                                    OperatorType::Add => expr_abstracted!(left, right, +),
                                    _ => None,
                                },
                                OperatorType::Sub => match op_type {
                                    OperatorType::Sub => expr_abstracted!(left, right, -),
                                    _ => None,
                                },
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

macro_rules! for_loop {
    ($stdout:ident, $vars:ident, $functions:ident, $structs:expr, $scope:expr, $variant:ident, $type:ty, $start:expr, $to:expr, $inc:expr, $node:ident, $name:expr) => {{
        let mut res: Option<QuitType> = None;

        let to_num = match $to {
            Value::$variant(num) => num,
            _ => panic!("Expected from to and inc to be of same type"),
        };

        let mut current = $start;

        let inc = if let Some(inc_value) = $inc {
            match inc_value {
                Value::$variant(num) => num,
                _ => panic!("Expected from to and inc to be of same type"),
            }
        } else {
            let num = if current <= to_num { 1 } else { -1 };
            num as $type
        };

        let var_value = VarValue::new($name, Value::$variant(current), $scope);
        $vars.insert($name.clone(), Rc::new(RefCell::new(var_value)));

        if inc >= 0 {
            while current < to_num {
                let (_, quit) = eval_node(
                    $vars,
                    Rc::clone(&$functions),
                    $structs,
                    $scope + 1,
                    $node.as_ref(),
                    $stdout,
                );

                if quit.is_some() {
                    res = quit;
                    break;
                }

                current += inc;

                let var_ptr = $vars.get(&$name).unwrap();
                *var_ptr.borrow_mut().value.borrow_mut() = Value::$variant(current);
            }
        } else {
            while to_num < current {
                let (_, quit) = eval_node(
                    $vars,
                    Rc::clone(&$functions),
                    $structs,
                    $scope + 1,
                    $node.as_ref(),
                    $stdout,
                );

                if quit.is_some() {
                    res = quit;
                    break;
                }

                current += inc;

                let var_ptr = $vars.get(&$name).unwrap();
                *var_ptr.borrow_mut().value.borrow_mut() = Value::$variant(current);
            }
        }

        res
    }};
}

#[derive(Debug)]
pub enum QuitType {
    Return(EvalValue),
    Break,
    Continue,
}

/// evaluate ast node
/// return type is single value
/// Ex: AstNode::Token(bool) -> Token(bool)
pub fn eval_node<'a>(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    structs: &mut StructInfo,
    scope: usize,
    node: &AstNode,
    stdout: &mut Stdout,
) -> (Option<EvalValue>, Option<QuitType>) {
    match &node {
        AstNode::CreateStruct(name, shape, props) => {
            let mut struct_props: Vec<StructProp> = Vec::new();
            let mut prop_keys: Vec<&String> = shape.props.keys().collect();
            for prop in props.iter() {
                let (prop_name, node) = prop;

                if !shape.props.contains_key(prop_name.as_str()) {
                    panic!("Unexpected property \"{}\" on struct {}", prop_name, name);
                }

                let (val, _) = eval_node(vars, Rc::clone(&functions), structs, scope, node, stdout);

                if let Some(value) = val {
                    let value = get_eval_value(vars, value, true);

                    let val_type_option = shape.props.get(prop_name);
                    if let Some(val_type) = val_type_option {
                        if !ensure_type(val_type, &value) {
                            panic!(
                                "Type mismatch in struct creation. Expected {:?} found {:?}",
                                val_type, value
                            );
                        }
                    }

                    let mut i = 0;
                    while i < prop_keys.len() {
                        if prop_keys[i] == prop_name {
                            prop_keys.remove(i);
                        }
                        i += 1;
                    }

                    struct_props.push(StructProp::new(prop_name.clone(), value));
                } else {
                    panic!("Expected value for struct property");
                }
            }

            if prop_keys.len() > 0 {
                let prop_keys: Vec<String> = prop_keys.iter().map(|&s| s.clone()).collect();
                panic!(
                    "Expected props {:?} on struct {}",
                    prop_keys.join(", "),
                    name
                )
            }

            let res = Value::Struct(name.clone(), shape.clone(), struct_props);

            (Some(EvalValue::Value(res)), None)
        }
        AstNode::SetArrIndex(arr_tok, index_node, value) => {
            if let (Some(index), _) = eval_node(
                vars,
                Rc::clone(&functions),
                structs,
                scope,
                index_node,
                stdout,
            ) {
                if let Token::Identifier(ident) = arr_tok {
                    let var_ptr = get_var_ptr(vars, &ident);
                    set_index_arr(
                        vars,
                        Rc::clone(&functions),
                        structs,
                        scope,
                        stdout,
                        var_ptr,
                        index,
                        value.as_ref().clone(),
                    );

                    (None, None)
                } else {
                    panic!("Expected identifier to index");
                }
            } else {
                panic!("Expected value to index array with");
            }
        }
        AstNode::IndexArr(arr_tok, index_node) => {
            if let (Some(index), _) = eval_node(
                vars,
                Rc::clone(&functions),
                structs,
                scope,
                index_node,
                stdout,
            ) {
                if let Token::Identifier(ident) = arr_tok {
                    let var_ptr = get_var_ptr(vars, &ident);
                    let res = index_arr_var_value(vars, var_ptr, index);

                    (Some(EvalValue::Value(res)), None)
                } else {
                    panic!("Expected identifier to index");
                }
            } else {
                panic!("Expected value to index array with");
            }
        }
        AstNode::While(condition, block) => {
            loop {
                let res = eval_exp(
                    vars,
                    Rc::clone(&functions),
                    structs,
                    scope,
                    condition,
                    stdout,
                );

                let val = get_eval_value(vars, res, false);

                if let Value::Bool(v) = val {
                    if !v {
                        break;
                    }
                } else {
                    panic!("Expected bool in while loop");
                }

                let (_, quit) =
                    eval_node(vars, Rc::clone(&functions), structs, scope, block, stdout);

                if quit.is_some() {
                    return (None, quit);
                }
            }

            (None, None)
        }
        AstNode::ForFromTo(ident, from, to, inc, node) => {
            if let Token::Identifier(var_name) = ident {
                let from_val = match eval_node(
                    vars,
                    Rc::clone(&functions),
                    structs,
                    scope,
                    from.as_ref(),
                    stdout,
                ) {
                    (Some(ev), _) => get_eval_value(vars, ev, false),
                    (None, _) => panic!("Expected from value in for loop"),
                };
                let to_val = match eval_node(
                    vars,
                    Rc::clone(&functions),
                    structs,
                    scope,
                    to.as_ref(),
                    stdout,
                ) {
                    (Some(ev), _) => get_eval_value(vars, ev, false),
                    (None, _) => panic!("Expected to value in for loop"),
                };
                let inc_val = if let Some(inc_value_node) = inc {
                    match eval_node(
                        vars,
                        Rc::clone(&functions),
                        structs,
                        scope,
                        inc_value_node.as_ref(),
                        stdout,
                    ) {
                        (Some(ev), _) => Some(get_eval_value(vars, ev, false)),
                        (None, _) => panic!("Expected to value in for loop"),
                    }
                } else {
                    None
                };

                let res = match from_val {
                    Value::Int(start) => {
                        for_loop!(
                            stdout,
                            vars,
                            functions,
                            structs,
                            scope,
                            Int,
                            i32,
                            start,
                            to_val,
                            inc_val,
                            node,
                            var_name.to_owned()
                        )
                    }
                    Value::Usize(start) => {
                        for_loop!(
                            stdout,
                            vars,
                            functions,
                            structs,
                            scope,
                            Usize,
                            usize,
                            start,
                            to_val,
                            inc_val,
                            node,
                            var_name.to_owned()
                        )
                    }
                    Value::Long(start) => {
                        for_loop!(
                            stdout,
                            vars,
                            functions,
                            structs,
                            scope,
                            Long,
                            i64,
                            start,
                            to_val,
                            inc_val,
                            node,
                            var_name.to_owned()
                        )
                    }
                    _ => panic!("Unexpected start type in for loop, expected int or long"),
                };

                if res.is_some() {
                    return (None, res);
                }
            } else {
                panic!("Expected identifier for loop iterator");
            }
            (None, None)
        }
        AstNode::Comparison(comp_token, left, right) => {
            if let (Some(left_res), _) =
                eval_node(vars, Rc::clone(&functions), structs, scope, left, stdout)
            {
                if let (Some(right_res), _) =
                    eval_node(vars, functions, structs, scope, right, stdout)
                {
                    return (Some(compare(vars, left_res, right_res, comp_token)), None);
                } else {
                    panic!("Expected result value from left of condition");
                }
            } else {
                panic!("Expected result value from left of condition");
            }
        }
        AstNode::Cast(var_type, node) => {
            let res_option = eval_node(
                vars,
                Rc::clone(&functions),
                structs,
                scope,
                node.as_ref(),
                stdout,
            );

            if let (Some(eval_value), _) = res_option {
                let val = get_eval_value(vars, eval_value, true);
                let casted_value = cast(var_type, val);
                (Some(EvalValue::Value(casted_value)), None)
            } else {
                panic!("Error casting value")
            }
        }
        AstNode::AccessStructProp(struct_token, access_path) => {
            let var_ptr = if let Token::Identifier(ident) = struct_token {
                get_var_ptr(vars, &ident)
            } else {
                panic!("Error in struct property access, expected identifier");
            };

            let var_value = &*var_ptr.borrow_mut();
            let temp_val = &var_value.value;
            let mut value = get_ref_value(temp_val);

            let mut i = 0;
            while i < access_path.len() {
                let mut temp_value: Option<Rc<RefCell<Value>>> = None;

                match &mut *value.borrow_mut() {
                    Value::Array(ref mut vals, arr_type) => {
                        match &access_path
                            .get(i)
                            .expect("Expected property to access on Array")
                        {
                            AstNode::CallFunc(name, args) => match name.as_str() {
                                "push" => {
                                    push_to_array(
                                        vars,
                                        Rc::clone(&functions),
                                        structs,
                                        scope,
                                        vals,
                                        arr_type,
                                        args,
                                        stdout,
                                    );
                                    None
                                }
                                _ => panic!("Unknown array method: {}", name),
                            },
                            AstNode::Token(t) => match t {
                                Token::Identifier(ident) => match ident.as_str() {
                                    "length" => Some(EvalValue::Value(Value::Usize(vals.len()))),
                                    _ => unimplemented!(),
                                },
                                _ => panic!("Unexpected method: {:?}", t),
                            },
                            _ => panic!("Unexpected operation: {:?}", access_path),
                        }
                    }
                    Value::Struct(_, _, ref mut props) => {
                        if i < access_path.len() {
                            let path_item = access_path.get(i).unwrap();
                            match path_item {
                                AstNode::Token(tok) => match tok {
                                    Token::Identifier(ident) => {
                                        temp_value = Some(get_prop_ptr(props, ident).unwrap());
                                    }
                                    _ => panic!("Unexpected operation on struct: {:?}", tok),
                                },
                                AstNode::SetVar(tok, node) => {
                                    let (res_val, _) = eval_node(
                                        vars,
                                        Rc::clone(&functions),
                                        structs,
                                        scope,
                                        node,
                                        stdout,
                                    );

                                    let eval_val = get_eval_value(
                                        vars,
                                        res_val.expect("Expected value to set to struct property"),
                                        true,
                                    );

                                    let ptr = if let Token::Identifier(ident) = tok {
                                        get_prop_ptr(props, ident)
                                    } else {
                                        panic!("Expected identifier to set struct property")
                                    }
                                    .unwrap();

                                    *ptr.borrow_mut() = eval_val;
                                }
                                AstNode::IndexArr(tok, node) => {
                                    let ptr_option = if let Token::Identifier(ident) = tok {
                                        get_prop_ptr(props, ident)
                                    } else {
                                        panic!(
                                            "Expected identifier to index array on struct property"
                                        )
                                    };

                                    let (eval_value, _) = eval_node(
                                        vars,
                                        Rc::clone(&functions),
                                        structs,
                                        scope,
                                        node,
                                        stdout,
                                    );

                                    let eval_value = get_eval_value(
                                        vars,
                                        eval_value
                                            .expect("Expected value to index struct property by"),
                                        true,
                                    );

                                    let index_value = if let Value::Ref(r) = eval_value {
                                        get_ref_value(&r)
                                    } else {
                                        Rc::new(RefCell::new(eval_value))
                                    };

                                    if let Value::Usize(index) = *index_value.borrow() {
                                        if let Some(ptr) = ptr_option {
                                            let ptr_val = &*ptr.borrow();
                                            if let Value::Array(items, _) = ptr_val {
                                                if let Value::Ref(r) = &items[index] {
                                                    temp_value = Some(Rc::new(RefCell::new(
                                                        Value::Ref(Rc::clone(&r)),
                                                    )));
                                                } else {
                                                    if i == access_path.len() - 1 {
                                                        return (
                                                            Some(EvalValue::Value(
                                                                items[index].clone(),
                                                            )),
                                                            None,
                                                        );
                                                    } else {
                                                        temp_value = Some(Rc::new(RefCell::new(
                                                            items[index].clone(),
                                                        )));
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        panic!("Array can only be indexed by usize")
                                    };
                                }
                                _ => panic!("Unexpected operation: {:?}", path_item),
                            }
                        }

                        None
                    }
                    _ => unimplemented!(),
                };

                if let Some(val) = temp_value {
                    value = val;
                }

                i += 1;
            }

            (Some(EvalValue::Value(Value::Ref(value))), None)
        }
        AstNode::Array(arr_nodes, arr_type) => {
            let mut res_arr: Vec<Value> = Vec::new();

            for node in arr_nodes.iter() {
                let res_option =
                    eval_node(vars, Rc::clone(&functions), structs, scope, node, stdout);

                if let (Some(res), _) = res_option {
                    let val = get_eval_value(vars, res, true);

                    if !ensure_type(arr_type, &val) {
                        panic!("Wrong type in array, expected {:?}", arr_type);
                    }

                    res_arr.push(val);
                }
            }
            (
                Some(EvalValue::Value(Value::Array(res_arr, arr_type.clone()))),
                None,
            )
        }
        AstNode::Else(_) => unreachable!(),
        AstNode::If(condition, node, else_branch) => {
            let (condition_res, _) = eval_node(
                vars,
                Rc::clone(&functions),
                structs,
                scope,
                condition.as_ref(),
                stdout,
            );

            if let Some(res) = condition_res {
                let condition_val = match res {
                    EvalValue::Value(val) => match val {
                        Value::Bool(b) => b,
                        _ => panic!("Error in `if`, expected boolean"),
                    },
                    EvalValue::Token(tok) => match tok {
                        Token::Identifier(ident) => {
                            let var_ptr = get_var_ptr(vars, &ident);
                            let val = var_ptr.borrow();

                            let res = match *val.value.borrow() {
                                Value::Bool(b) => b,
                                _ => panic!("Error in `if`, expected boolean"),
                            };

                            res
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
                    let (_, quit) = eval_node(
                        vars,
                        Rc::clone(&functions),
                        structs,
                        scope + 1,
                        node.as_ref(),
                        stdout,
                    );

                    if quit.is_some() {
                        return (None, quit);
                    }
                } else {
                    if let Some(branch) = else_branch {
                        let (_, quit) = eval_node(
                            vars,
                            Rc::clone(&functions),
                            structs,
                            scope + 1,
                            branch.as_ref(),
                            stdout,
                        );

                        if quit.is_some() {
                            return (None, quit);
                        }
                    }
                }
            } else {
                panic!("Expected if condition to be type boolean");
            }
            (None, None)
        }
        AstNode::StatementSeq(seq) => {
            for node in seq.iter() {
                let (_, quit) = eval_node(
                    vars,
                    Rc::clone(&functions),
                    structs,
                    scope,
                    &node.borrow(),
                    stdout,
                );
                match quit {
                    Some(q) => match q {
                        QuitType::Return(val) => {
                            return (None, Some(QuitType::Return(val)));
                        }
                        _ => {}
                    },
                    _ => {}
                };
            }
            (None, None)
        }
        AstNode::MakeVar(var_type, name, value) => {
            if let Token::Identifier(ident) = name {
                if vars.get(ident.as_str()).is_none() {
                    make_var(
                        vars, functions, structs, scope, var_type, name, value, stdout,
                    );
                    (None, None)
                } else {
                    panic!(
                        "Variable \"{}\" already exists, cannot be re-defined",
                        ident
                    );
                }
            } else {
                panic!("Expected variable name to be identifier");
            }
        }
        AstNode::MakeFunc(func) => {
            func.borrow_mut().set_scope(scope);
            functions.borrow_mut().push(Func::Custom(Rc::clone(func)));
            (None, None)
        }
        AstNode::SetVar(name, value) => {
            let value = eval_node(vars, functions, structs, scope, value.as_ref(), stdout);
            if let (Some(tok), _) = value {
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
            (None, None)
        }
        AstNode::Token(token) => (Some(EvalValue::Token(token.to_owned())), None),
        AstNode::CallFunc(name, args) => {
            let func_res = call_func(vars, functions, structs, scope, name, args, stdout);
            if let Some(tok) = func_res {
                (Some(EvalValue::Value(tok)), None)
            } else {
                (None, None)
            }
        }
        AstNode::Exp(nodes) => (
            Some(eval_exp(
                vars,
                Rc::clone(&functions),
                structs,
                scope,
                nodes,
                stdout,
            )),
            None,
        ),
        AstNode::Bang(node) => {
            let val = eval_node(vars, functions, structs, scope, node.as_ref(), stdout);
            let b = if let (Some(value), _) = val {
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
            (Some(EvalValue::Value(Value::Bool(!b))), None)
        }
        AstNode::Return(return_node) => {
            let (eval_value, _) = eval_node(
                vars,
                Rc::clone(&functions),
                structs,
                scope,
                return_node,
                stdout,
            );

            let eval_value = eval_value.unwrap();

            if let EvalValue::Value(val) = &eval_value {
                if let Value::Ref(r) = val {
                    (
                        None,
                        Some(QuitType::Return(EvalValue::Value(Value::Ref(Rc::clone(
                            &r,
                        ))))),
                    )
                } else {
                    (None, Some(QuitType::Return(eval_value)))
                }
            } else {
                (None, Some(QuitType::Return(eval_value)))
            }
        }
    }
}

pub fn eval<'a>(
    vars: &mut HashMap<String, Rc<RefCell<VarValue>>>,
    functions: Rc<RefCell<Vec<Func<'a>>>>,
    structs: &mut StructInfo,
    tree: Ast,
    stdout: &mut Stdout,
) {
    let root_node = tree.node.borrow();
    eval_node(vars, functions, structs, 0, &root_node.to_owned(), stdout);
}
