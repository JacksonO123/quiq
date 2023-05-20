use crate::interpreter::VarType;

#[derive(Clone, Copy, Debug)]
pub enum OperatorType {
    Add,
    Sub,
    Mult,
    Div,
}

#[derive(Debug, Clone)]
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
    Comma,
    Period,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub value: String,
}
impl Token {
    pub fn new_from_string(value: String, token_type: TokenType) -> Self {
        Self { value, token_type }
    }
    pub fn new(value: &str, token_type: TokenType) -> Self {
        Self {
            value: String::from(value),
            token_type,
        }
    }
}

fn get_full_token<'a>(chars: &Vec<char>, start: usize) -> String {
    let mut res = String::new();

    let mut dot_found = false;
    let is_number = chars[start].is_numeric() || chars[start] == '-';
    let mut i = start;
    while (chars[i].is_alphabetic() && !is_number)
        || (i == start && chars[i] == '-')
        || chars[i].is_numeric()
        || (chars[i] == '.' && !dot_found)
    {
        if chars[i] == '.' {
            if !is_number {
                return res;
            }
            dot_found = true;
        }

        res.push(chars[i]);
        i += 1;
    }

    res
}

fn get_string_token<'a>(chars: &Vec<char>, start: usize) -> String {
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

fn get_full_line<'a>(chars: &Vec<char>, start: usize) -> String {
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

pub fn tokenize(code: &str) -> Vec<Option<Token>> {
    let mut tokens = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = code.chars().collect();
    while i < chars.len() {
        let token: Token;
        if chars[i].is_alphabetic()
            || chars[i].is_numeric()
            || (i < chars.len() - 1 && chars[i] == '-' && chars[i + 1].is_numeric())
        {
            let value = get_full_token(&chars, i);

            i += value.len() - 1;

            let token_type = match value.as_str() {
                // types
                "int" => TokenType::Type(VarType::Int),
                "float" => TokenType::Type(VarType::Float),
                "double" => TokenType::Type(VarType::Double),
                "long" => TokenType::Type(VarType::Long),
                "bool" => TokenType::Type(VarType::Bool),
                "string" => TokenType::Type(VarType::String),
                "usize" => TokenType::Type(VarType::Usize),
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
                    if is_number(&value) {
                        TokenType::Number
                    } else if chars[0].is_alphabetic() {
                        TokenType::Identifier
                    } else {
                        panic!("Unknown token: {}", value)
                    }
                }
            };
            token = Token::new_from_string(value, token_type);
        } else if chars[i] == '"' {
            let string = get_string_token(&chars, i);
            i += string.len() + 1;

            token = Token::new_from_string(string, TokenType::String);
        } else if i < chars.len() - 2 && chars[i] == '/' && chars[i + 1] == '/' {
            let line = get_full_line(&chars, i);
            i += line.len();

            token = Token::new_from_string(line, TokenType::Comment);
        } else if chars[i] == ' ' {
            i += 1;
            continue;
        } else if chars[i] == '\n' {
            token = Token::new("\n", TokenType::NewLine);
        } else if chars[i] == '*' {
            token = Token::new("*", TokenType::Operator(OperatorType::Mult));
        } else if chars[i] == '/' {
            token = Token::new("/", TokenType::Operator(OperatorType::Div));
        } else if chars[i] == '+' {
            token = Token::new("+", TokenType::Operator(OperatorType::Add));
        } else if chars[i] == '-' {
            token = Token::new("-", TokenType::Operator(OperatorType::Sub));
        } else if chars[i] == '(' {
            token = Token::new("(", TokenType::LParen);
        } else if chars[i] == ')' {
            token = Token::new(")", TokenType::RParen);
        } else if chars[i] == '{' {
            token = Token::new("{", TokenType::LBrace);
        } else if chars[i] == '}' {
            token = Token::new("}", TokenType::RBrace);
        } else if chars[i] == '[' {
            token = Token::new("[", TokenType::LBracket);
        } else if chars[i] == ']' {
            token = Token::new("]", TokenType::RBracket);
        } else if chars[i] == ',' {
            token = Token::new(",", TokenType::Comma);
        } else if chars[i] == '.' {
            token = Token::new(".", TokenType::Period);
        } else if chars[i] == ';' {
            token = Token::new(";", TokenType::Semicolon);
        } else if chars[i] == '=' {
            if i < chars.len() - 1 && chars[i + 1] == '=' {
                token = Token::new("==", TokenType::EqCompare);
            } else {
                token = Token::new("=", TokenType::EqSet);
            }
        } else if chars[i] == '!' {
            if i < chars.len() - 1 && chars[i + 1] == '=' {
                token = Token::new("!=", TokenType::EqNCompare);
            } else {
                token = Token::new("!", TokenType::Bang);
            }
        } else {
            panic!("Unexpected token: {}", chars[i]);
        }

        match token.token_type {
            TokenType::Comment => {}
            _ => {
                tokens.push(Some(token));
            }
        }

        i += 1;
    }

    tokens
}

fn is_number(string: &String) -> bool {
    let mut found_decimal = false;
    let mut first = true;
    string.chars().all(|c| {
        if first && c == '-' {
            first = false;
            return true;
        }
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
