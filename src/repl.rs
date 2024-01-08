use std::{io, io::Write};

use crate::{
    ast::{generate_tree, Value},
    builtins::init_builtins,
    data::Data,
    helpers::get_eval_value,
    interpreter::eval,
    tokenizer::tokenize,
};

pub fn repl() {
    println!("Quiq");
    println!("--------------\n");
    let mut data = Data::new();
    let mut stdout = io::stdout();

    init_builtins(&mut data);
    loop {
        print!("> ");
        stdout.flush().unwrap();
        let mut buf = String::new();
        std::io::stdin()
            .read_line(&mut buf)
            .expect("Failed to read line");
        let buf = buf.trim();

        let mut tokens = tokenize(buf, &mut data);
        let tree = generate_tree(&mut data, &mut tokens);
        let res = eval(&mut data, tree, &mut stdout);

        if let Some(val) = res {
            let val = get_eval_value(&mut data, val, 0, false);
            if let Value::String(s) = val {
                println!("{:?}", s);
            } else {
                println!("{}", val.get_str());
            }
        }
    }
}
