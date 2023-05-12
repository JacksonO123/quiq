mod tokenizer;

use std::path::Path;
use std::{env::args, fs};

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
    println!("{}", file);

    let tokens = tokenize(file.as_str());
    println!("{:#?}", tokens);
}

fn get_file(name: &str) -> String {
    let path_str = "src/input/".to_owned() + name;
    let path = Path::new(&path_str);
    fs::read_to_string(path).unwrap()
}
