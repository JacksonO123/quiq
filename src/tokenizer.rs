use crate::interpreter::VarType;

#[derive(Clone, Copy, Debug)]
pub enum OperatorType {
    Add,
    Sub,
    Mult,
    Div,
}
impl OperatorType {
    pub fn get_str(&self) -> &str {
        match self {
            OperatorType::Add => "+",
            OperatorType::Sub => "-",
            OperatorType::Mult => "*",
            OperatorType::Div => "/",
        }
    }
}

#[derive(Debug, Clone)]
pub enum Token<'a> {
    Type(VarType),
    Number(String),
    String(String),
    Bool(bool),
    Keyword(&'a str),
    Identifier(String),
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
    Semicolon,
    NewLine,
    Comment,
    Bang,
    Comma,
    Period,
    LAngle,
    RAngle,
    LAngleEq,
    RAngleEq,
}
impl<'a> Token<'a> {
    pub fn get_str(&self) -> &str {
        match self {
            Token::Type(var_type) => &var_type.get_str(),
            Token::Number(n) => n,
            Token::String(s) => s,
            Token::Bool(b) => {
                if *b {
                    "true"
                } else {
                    "false"
                }
            }
            Token::Keyword(k) => k,
            Token::Identifier(ident) => ident,
            Token::Operator(op_type) => op_type.get_str(),
            Token::RParen => ")",
            Token::LParen => "(",
            Token::RBrace => "}",
            Token::LBrace => "{",
            Token::RBracket => "]",
            Token::LBracket => "[",
            Token::EqCompare => "==",
            Token::EqNCompare => "!=",
            Token::EqSet => "=",
            Token::Semicolon => ";",
            Token::NewLine => "\n",
            Token::Comment => panic!("Cannot get str of comment"),
            Token::Bang => "!",
            Token::Comma => ",",
            Token::Period => ".",
            Token::LAngle => "<",
            Token::RAngle => ">",
            Token::LAngleEq => "<=",
            Token::RAngleEq => ">=",
        }
    }
    pub fn get_token_name(&self) -> &str {
        match self {
            Token::Number(_) => "Number",
            Token::String(_) => "String",
            Token::Bool(_) => "Bool",
            Token::Keyword(_) => "Keyword",
            Token::Identifier(_) => "Identifier",
            _ => unreachable!(),
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

            token = match value.as_str() {
                // types
                "int" => Token::Type(VarType::Int),
                "float" => Token::Type(VarType::Float),
                "double" => Token::Type(VarType::Double),
                "long" => Token::Type(VarType::Long),
                "bool" => Token::Type(VarType::Bool),
                "string" => Token::Type(VarType::String),
                "usize" => Token::Type(VarType::Usize),
                // keywords (add more)
                "if" => Token::Keyword("if"),
                "else" => Token::Keyword("else"),
                "for" => Token::Keyword("for"),
                "while" => Token::Keyword("while"),
                // booleans
                "true" => Token::Bool(true),
                "false" => Token::Bool(false),
                _ => {
                    let chars: Vec<char> = value.chars().collect();
                    // check string, check number
                    if is_number(&value) {
                        Token::Number(value)
                    } else if chars[0].is_alphabetic() {
                        Token::Identifier(value)
                    } else {
                        panic!("Unknown token: {}", value)
                    }
                }
            };
        } else if chars[i] == '"' {
            let string = get_string_token(&chars, i);
            i += string.len() + 1;

            token = Token::String(string);
        } else if i < chars.len() - 2 && chars[i] == '/' && chars[i + 1] == '/' {
            let line = get_full_line(&chars, i);
            i += line.len();

            token = Token::Comment;
        } else if chars[i] == ' ' {
            i += 1;
            continue;
        } else if chars[i] == '\n' {
            token = Token::NewLine;
        } else if chars[i] == '*' {
            token = Token::Operator(OperatorType::Mult);
        } else if chars[i] == '/' {
            token = Token::Operator(OperatorType::Div);
        } else if chars[i] == '+' {
            token = Token::Operator(OperatorType::Add);
        } else if chars[i] == '-' {
            token = Token::Operator(OperatorType::Sub);
        } else if chars[i] == '(' {
            token = Token::LParen;
        } else if chars[i] == ')' {
            token = Token::RParen;
        } else if chars[i] == '{' {
            token = Token::LBrace;
        } else if chars[i] == '}' {
            token = Token::RBrace;
        } else if chars[i] == '[' {
            token = Token::LBracket;
        } else if chars[i] == ']' {
            token = Token::RBracket;
        } else if chars[i] == '<' {
            token = if i < chars.len() - 1 && chars[i + 1] == '=' {
                Token::LAngleEq
            } else {
                Token::LAngle
            }
        } else if chars[i] == '>' {
            token = if i < chars.len() - 1 && chars[i + 1] == '=' {
                Token::RAngleEq
            } else {
                Token::RAngle
            }
        } else if chars[i] == ',' {
            token = Token::Comma;
        } else if chars[i] == '.' {
            token = Token::Period;
        } else if chars[i] == ';' {
            token = Token::Semicolon;
        } else if chars[i] == '=' {
            token = if i < chars.len() - 1 && chars[i + 1] == '=' {
                i += 1;
                Token::EqCompare
            } else {
                Token::EqSet
            }
        } else if chars[i] == '!' {
            token = if i < chars.len() - 1 && chars[i + 1] == '=' {
                i += 1;
                Token::EqNCompare
            } else {
                Token::Bang
            }
        } else {
            panic!("Unexpected token: {}", chars[i]);
        }

        match token {
            Token::Comment => {}
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
