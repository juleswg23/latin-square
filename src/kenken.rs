#![allow(dead_code)]
use crate::latin::LatinSolver;

// The possible operations on a clue for a KenKen
#[derive(Clone, Debug, strum_macros::Display)]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Given,
    Unknown,
    NoOp,
}

// A data structure representing each clue of a KenKen puzzle
#[derive(Debug)]
struct Clue {
    op: Operation,
    target: u32,
}

// A KenKen struct has the dimension of the KenKen and a list of the regions
struct KenKen {
    order: usize,
    regions: Vec<(Clue, Vec<u32>)>,
}

impl KenKen {
    // Constructor takes the order as a param and initializes regions with no data
    fn new(order: usize) -> KenKen {
        KenKen {
            order,
            regions: vec![], // TODO make this more complete
        }
    }

    // TODO maybe try making a to_string()
}

// Extends LatinSolver in the sense that it solves specific KenKens by leaning on LatinSolver methods
struct KenKenSolver {
    ken_ken: KenKen,
    latin_solver: LatinSolver,
}

impl KenKenSolver {
    fn new(order: usize) -> KenKenSolver {
        KenKenSolver {
            ken_ken: KenKen::new(order),
            latin_solver: LatinSolver::new(order),
        }
    }
}

// Takes a string input of a KenKen, and converts it to a KenKen object
// Example format looks like 3: 3+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:

// Takes a string input of a KenKen, and converts it to a KenKen object
// Example format looks like "3: 3+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:"

// TODO remove panics
fn read_ken_ken(input: String) -> KenKen {
    let order: usize = input[0..=0].to_string().parse().expect("Not a number");

    let mut ken_ken = KenKen::new(order);

    let mut chars = input[3..].chars();

    while let Some(c) = chars.next() {
        let target = c.to_digit(10).expect("Not a digit");
        let op: Operation = match chars.next() {
            Some('+') => Operation::Add,
            Some('-') => Operation::Subtract,
            Some('*') => Operation::Multiply,
            Some('/') => Operation::Divide,
            Some(' ') => Operation::Given,
            None => Operation::Unknown,
            _ => panic!("Not a valid operation character"),
        };
        if !matches!(op, Operation::Unknown | Operation::Given) {
            chars.next();
        }
        let mut cells:Vec<u32> = vec![];
        loop {
            // or use unwrap?
            let row: u32 = chars.next().unwrap().to_digit(10).expect("Not a row digit");
            let col:u32 = chars.next().expect("No more characters").to_digit(10).expect("Not a row digit");
            cells.push(row*order as u32 + col);
            match chars.next() {
                Some(':') => break,
                Some(' ') => (),
                _ => panic!("Not a valid operation character"),
            };
        }
        ken_ken.regions.push((Clue{op, target}, cells));
        chars.next();
    }
    ken_ken
}

pub fn main() {
    let k = read_ken_ken("3: 3+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
    println!("regions: {:?}, ", k.regions);
    // println!("no debug: {}", k.regions);
}

