mod ast;
mod builtins;
mod helpers;
mod interpreter;
mod tokenizer;

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::{env::args, fs};

use crate::ast::generate_tree;
use crate::builtins::init_builtins;
use crate::interpreter::{eval, VarValue};
use crate::tokenizer::tokenize;

fn main() {
    let args: Vec<String> = args().collect();
    let filename = if args.len() == 1 {
        // repl here
        panic!("Repl not implemented yet");
    } else {
        "main.quiq"
    };

    let file = get_file(filename);

    let tokens = tokenize(file.as_str());

    let tree = generate_tree(tokens);
    tree.print();

    let mut vars: Vec<VarValue> = Vec::new();
    let functions = Rc::new(RefCell::new(Vec::new()));

    init_builtins(&mut vars, Rc::clone(&functions));
    eval(&mut vars, functions, tree);

    print_vars(&vars);
}

fn print_vars<'a>(vars: &'a Vec<VarValue<'a>>) {
    for var in vars.iter() {
        println!("{}", var.get_str());
    }
}

fn get_file(name: &str) -> String {
    let path_str = "src/input/".to_owned() + name;
    let path = Path::new(&path_str);
    fs::read_to_string(path).unwrap()
}
