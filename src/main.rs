mod ast;
mod builtins;
mod data;
mod helpers;
mod interpreter;
mod repl;
mod tokenizer;

use std::env::args;
use std::io;
use std::time::Instant;

use data::Data;
use helpers::get_file;
use repl::repl;

use crate::ast::generate_tree;
use crate::builtins::init_builtins;
use crate::interpreter::eval;
use crate::tokenizer::tokenize;

fn main() {
    let start = Instant::now();

    let args: Vec<String> = args().collect();
    if args.len() == 1 {
        repl();
    }

    let filename = args[1].as_str();

    let file_start = Instant::now();
    let file = get_file(format!("src/input/{}", filename).as_str());
    let file_end = file_start.elapsed();

    let mut data = Data::new();

    let tokens_start = Instant::now();
    let mut tokens = tokenize(file.as_str(), &mut data);
    let tokens_end = tokens_start.elapsed();

    let tree_start = Instant::now();
    let tree = generate_tree(&mut data, &mut tokens);
    let tree_end = tree_start.elapsed();

    init_builtins(&mut data);

    let mut stdout = io::stdout();

    let eval_start = Instant::now();
    eval(&mut data, tree, &mut stdout);
    let eval_end = eval_start.elapsed();

    let end = start.elapsed();

    if args.iter().any(|a| a == "-b") {
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
