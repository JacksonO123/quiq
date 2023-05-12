use regex::Regex;

#[derive(Debug)]
enum TokenType {
    NumberType,
    StringType,
    BoolType,
    Number,
    String,
    Bool,
    Keyword,
    Operator,
    RParen,
    LParen,
    RBrace,
    LBrace,
    RBracket,
    LBracket,
    EqCompare,
    EqNCompare,
    EqSet,
    Identifier,
    Semicolon,
    NewLine,
}

#[derive(Debug)]
pub struct Token<'a> {
    token_type: TokenType,
    value: &'a str,
}
impl<'a> Token<'a> {
    fn new(value: &'a str, token_type: TokenType) -> Self {
        Self { token_type, value }
    }
}

pub fn tokenize(code: &str) -> Vec<Token> {
    let reg_str = "\"[\\w\\s]*\"".to_owned() + "|" + r"\w+|\(|\)|\d+|,|;|==|=|!=|\{|\}|\n";
    let re = Regex::new(reg_str.as_str()).unwrap();
    let matches: Vec<&str> = re.find_iter(code).map(|m| m.as_str()).collect();
    println!("matches: {:?}", matches);

    let mut tokens = Vec::new();
    matches.iter().for_each(|&m| {
        let token = generate_token(m);
        tokens.push(token);
    });
    tokens
}

fn generate_token(value: &str) -> Token {
    let token_type = match value {
        // types
        "int" => TokenType::NumberType,
        "float" => TokenType::NumberType,
        "double" => TokenType::NumberType,
        "bool" => TokenType::BoolType,
        "string" => TokenType::StringType,
        // keywords (add more)
        "if" => TokenType::Keyword,
        "else" => TokenType::Keyword,
        "for" => TokenType::Keyword,
        "while" => TokenType::Keyword,
        // operators
        "*" => TokenType::Operator,
        "/" => TokenType::Operator,
        "+" => TokenType::Operator,
        "-" => TokenType::Operator,
        // booleans
        "true" => TokenType::Bool,
        "false" => TokenType::Bool,
        // comparison
        "==" => TokenType::EqCompare,
        "!=" => TokenType::EqNCompare,
        "=" => TokenType::EqSet,
        // semicolon yay
        ";" => TokenType::Semicolon,
        // brackets
        ")" => TokenType::RParen,
        "(" => TokenType::LParen,
        "}" => TokenType::RBrace,
        "{" => TokenType::LBrace,
        "]" => TokenType::RBracket,
        "[" => TokenType::LBracket,
        // newline
        "\n" => TokenType::NewLine,
        _ => {
            let chars: Vec<char> = value.chars().collect();
            let first_char = chars[0];
            // check string, check number
            if first_char == '"' {
                TokenType::String
            } else if is_number(value) {
                TokenType::Number
            } else if first_char.is_alphabetic() {
                TokenType::Identifier
            } else {
                panic!("Unknown token: {}", value)
            }
        }
    };

    Token::new(value, token_type)
}

fn is_number(string: &str) -> bool {
    string.chars().all(|c| c.is_digit(10))
}
