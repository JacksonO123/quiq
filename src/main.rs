mod ast;
mod builtins;
mod helpers;
mod interpreter;
mod tokenizer;
mod variables;

use std::cell::RefCell;
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;
use std::{env::args, fs};

use interpreter::StructInfo;
use variables::Variables;

use crate::ast::generate_tree;
use crate::builtins::init_builtins;
use crate::interpreter::{eval, VarValue};
use crate::tokenizer::tokenize;

fn main() {
    let start = Instant::now();

    let args: Vec<String> = args().collect();
    let filename = if args.len() == 1 {
        // repl here
        panic!("Repl not implemented yet");
    } else {
        args[1].as_str()
    };

    let file_start = Instant::now();
    let file = get_file(filename);
    let file_end = file_start.elapsed();

    let mut struct_info = StructInfo::new();

    let tokens_start = Instant::now();
    let mut tokens = tokenize(file.as_str(), &mut struct_info);
    let tokens_end = tokens_start.elapsed();

    let tree_start = Instant::now();
    let tree = generate_tree(&mut struct_info, &mut tokens);
    let tree_end = tree_start.elapsed();

    let mut vars = Variables::new();
    let mut functions = vec![];

    init_builtins(&mut functions);

    let mut stdout = io::stdout();

    let eval_start = Instant::now();
    eval(
        &mut vars,
        &mut functions,
        &mut struct_info,
        tree,
        &mut stdout,
    );
    let eval_end = eval_start.elapsed();

    let end = start.elapsed();

    if args.iter().position(|a| a == "-b").is_some() {
        println!();
        println!(
            "[{}] read in {}s | {}ms | {}µs | {}ns",
            filename,
            file_end.as_secs(),
            file_end.as_millis(),
            file_end.as_micros(),
            file_end.as_nanos()
        );
        println!(
            "Tokenized in {}s | {}ms | {}µs | {}ns",
            tokens_end.as_secs(),
            tokens_end.as_millis(),
            tokens_end.as_micros(),
            tokens_end.as_nanos()
        );
        println!(
            "Ast generated in {}s | {}ms | {}µs | {}ns",
            tree_end.as_secs(),
            tree_end.as_millis(),
            tree_end.as_micros(),
            tree_end.as_nanos()
        );
        println!(
            "Evaluated in {}s | {}ms | {}µs | {}ns",
            eval_end.as_secs(),
            eval_end.as_millis(),
            eval_end.as_micros(),
            eval_end.as_nanos()
        );
        println!(
            "Total: {}s | {}ms | {}µs | {}ns",
            end.as_secs(),
            end.as_millis(),
            end.as_micros(),
            end.as_nanos()
        );
    }
}

fn get_file(name: &str) -> String {
    let path_str = "src/input/".to_owned() + name;
    let path = Path::new(&path_str);
    fs::read_to_string(path).unwrap()
}
