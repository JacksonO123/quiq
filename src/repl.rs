use std::{io, io::Write};

use crate::{
    ast::{generate_tree, Value},
    builtins::init_builtins,
    helpers::get_eval_value,
    interpreter::{eval, StructInfo},
    tokenizer::tokenize,
    variables::Variables,
};

pub fn repl() {
    println!("Quiq");
    println!("--------------\n");
    let mut struct_info = StructInfo::new();
    let mut vars = Variables::new();
    let mut functions = vec![];
    let mut stdout = io::stdout();

    init_builtins(&mut functions);
    loop {
        print!("> ");
        stdout.flush().unwrap();
        let mut buf = String::new();
        std::io::stdin()
            .read_line(&mut buf)
            .expect("Failed to read line");
        let buf = buf.trim();

        let mut tokens = tokenize(buf, &mut struct_info);
        let tree = generate_tree(&mut struct_info, &mut tokens);
        let res = eval(
            &mut vars,
            &mut functions,
            &mut struct_info,
            tree,
            &mut stdout,
        );

        if let Some(val) = res {
            let val = get_eval_value(&mut vars, val, 0, false);
            if let Value::String(s) = val {
                println!("{:?}", s);
            } else {
                println!("{}", val.get_str());
            }
        }
    }
}
