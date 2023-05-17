use crate::interpreter::VarType;

#[derive(Clone, Copy, Debug)]
pub enum OperatorType {
    Add,
    Sub,
    Mult,
    Div,
}

#[derive(Debug, Copy, Clone)]
pub enum TokenType {
    Type(VarType),
    Number,
    String,
    Bool,
    Keyword,
    Operator(OperatorType),
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
    Bang,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub value: String,
}
impl Token {
    pub fn new(value: String, token_type: TokenType) -> Self {
        Self { token_type, value }
    }
    pub fn get_str(&self) -> String {
        self.value.clone()
    }
}

fn get_full_token(chars: Vec<char>, start: usize) -> String {
    let mut res = String::new();

    let mut dot_found = false;
    let is_number = chars[0].is_numeric();
    let mut i = start;
    while (chars[i].is_alphabetic() && !is_number)
        || chars[i].is_numeric()
        || (chars[i] == '.' && !dot_found)
    {
        if chars[i] == '.' {
            dot_found = true;
        }

        res.push(chars[i]);
        i += 1;
    }

    res
}

fn get_string_token(chars: Vec<char>, start: usize) -> String {
    let mut res = String::new();
    let mut i = start + 1;
    while i < chars.len() {
        if chars[i] == '"' && i > 0 && chars[i - 1] != '\\' {
            break;
        } else {
            res.push(chars[i]);
        }

        i += 1;
    }

    res
}

fn get_full_line(chars: Vec<char>, start: usize) -> String {
    let mut res = String::new();
    let mut i = start;
    while i < chars.len() {
        if chars[i] != '\n' {
            res.push(chars[i]);
        } else {
            break;
        }

        i += 1;
    }

    res
}

pub fn tokenize(code: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = code.chars().collect();
    while i < chars.len() {
        let token: Token;
        if chars[i].is_alphabetic() || chars[i].is_numeric() {
            let value = get_full_token(chars.clone(), i);
            i += value.len() - 1;

            let token_type = match value.as_str() {
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
                // booleans
                "true" => TokenType::Bool,
                "false" => TokenType::Bool,
                _ => {
                    let chars: Vec<char> = value.chars().collect();
                    // check string, check number
                    if is_number(value.clone()) {
                        TokenType::Number
                    } else if chars[0].is_alphabetic() {
                        TokenType::Identifier
                    } else {
                        panic!("Unknown token: {}", value)
                    }
                }
            };
            token = Token::new(value, token_type);
        } else if chars[i] == '"' {
            let string = get_string_token(chars.clone(), i);
            i += string.len();

            token = Token::new(string, TokenType::String);
        } else if i < chars.len() - 2 && chars[i] == '/' && chars[i + 1] == '/' {
            let line = get_full_line(chars.clone(), i);
            i += line.len();

            token = Token::new(line, TokenType::Comment);
        } else if chars[i] == ' ' {
            i += 1;
            continue;
        } else if chars[i] == '\n' {
            token = Token::new(String::from("\n"), TokenType::NewLine);
        } else if chars[i] == '*' {
            token = Token::new(String::from("*"), TokenType::Operator(OperatorType::Mult));
        } else if chars[i] == '/' {
            token = Token::new(String::from("/"), TokenType::Operator(OperatorType::Div));
        } else if chars[i] == '+' {
            token = Token::new(String::from("+"), TokenType::Operator(OperatorType::Add));
        } else if chars[i] == '-' {
            token = Token::new(String::from("-"), TokenType::Operator(OperatorType::Sub));
        } else if chars[i] == '(' {
            token = Token::new(String::from("("), TokenType::LParen);
        } else if chars[i] == ')' {
            token = Token::new(String::from(")"), TokenType::RParen);
        } else if chars[i] == '{' {
            token = Token::new(String::from("{"), TokenType::LBrace);
        } else if chars[i] == '}' {
            token = Token::new(String::from("}"), TokenType::RBrace);
        } else if chars[i] == '[' {
            token = Token::new(String::from("["), TokenType::LBracket);
        } else if chars[i] == ']' {
            token = Token::new(String::from("]"), TokenType::RBracket);
        } else if chars[i] == ';' {
            token = Token::new(String::from(";"), TokenType::Semicolon);
        } else if chars[i] == '=' {
            if i < chars.len() - 1 && chars[i + 1] == '=' {
                token = Token::new(String::from("=="), TokenType::EqCompare);
            } else {
                token = Token::new(String::from("="), TokenType::EqSet);
            }
        } else if chars[i] == '!' {
            if i < chars.len() - 1 && chars[i + 1] == '=' {
                token = Token::new(String::from("!="), TokenType::EqNCompare);
            } else {
                token = Token::new(String::from("!"), TokenType::Bang);
            }
        } else {
            panic!("Unexpected token: {}", chars[i]);
        }

        match token.token_type {
            TokenType::Comment => {}
            _ => {
                tokens.push(token);
            }
        }

        i += 1;
    }
    tokens
}

fn is_number(string: String) -> bool {
    let mut found_decimal = false;
    string.chars().all(|c| {
        if c == '.' {
            if found_decimal {
                return false;
            }
            found_decimal = true;
            true
        } else {
            c.is_digit(10)
        }
    })
}
