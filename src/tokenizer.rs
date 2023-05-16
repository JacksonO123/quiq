use regex::Regex;

#[derive(Copy, Clone, Debug)]
pub enum VarType {
    Int,
    Float,
    Double,
    Long,
    String,
    Bool,
}

#[derive(Debug, Copy, Clone)]
pub enum TokenType {
    Type(VarType),
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
    Comment,
}

#[derive(Debug, Copy, Clone)]
pub struct Token<'a> {
    pub token_type: TokenType,
    pub value: &'a str,
}
impl<'a> Token<'a> {
    fn new(value: &'a str, token_type: TokenType) -> Self {
        Self { token_type, value }
    }
    pub fn get_str(&self) -> &'a str {
        self.value
    }
}

pub fn tokenize(code: &str) -> Vec<Token> {
    let reg_str = "\\/\\/.*|\"[\\w\\s]*\"|\\w+|\\(|\\)|\\d+|,|;|==|=|!=|\\{|\\}|\\n";
    let re = Regex::new(reg_str).unwrap();
    let matches: Vec<&str> = re.find_iter(code).map(|m| m.as_str()).collect();

    let mut tokens = Vec::new();
    matches.iter().for_each(|&m| {
        let token = generate_token(m);
        match token.token_type {
            TokenType::Comment => {}
            _ => {
                tokens.push(token);
            }
        }
    });
    tokens
}

fn generate_token(value: &str) -> Token {
    let token_type = match value {
        // types
        "int" => TokenType::Type(VarType::Int),
        "float" => TokenType::Type(VarType::Float),
        "double" => TokenType::Type(VarType::Double),
        "long" => TokenType::Type(VarType::Long),
        "bool" => TokenType::Type(VarType::Bool),
        "string" => TokenType::Type(VarType::String),
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
            // check string, check number
            if chars[0] == '"' {
                TokenType::String
            } else if is_number(value) {
                TokenType::Number
            } else if chars[0].is_alphabetic() {
                TokenType::Identifier
            } else if chars[0] == '/' && chars[1] == '/' {
                TokenType::Comment
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
