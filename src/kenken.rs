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
    target: usize,
}

pub mod math {

    pub fn xy_pair(index: usize, order:usize) -> (usize, usize) {
        (index / order, index % order)
    }
    pub fn x_value(index: usize, order: usize) -> usize {
        index / order
    }

    pub fn y_value(index: usize, order: usize) -> usize {
        index % order
    }
}

#[derive(Debug)]
struct Region {
    clue: Clue,
    cells: Vec<usize>,
}

impl Region {
    pub fn new(clue: Clue, cells: Vec<usize>) -> Self {
        Self { clue, cells, }
    }

    pub fn set_clue(&mut self, clue: Clue) {
        self.clue = clue;
    }

    pub fn add_cell(&mut self, cell: usize) {
        self.cells.push(cell);
    }

    pub fn remove_cell(&mut self, cell: usize) {
        self.cells.retain(|&c| c != cell);
    }

    pub fn clue(&self) -> &Clue {
        &self.clue
    }

    pub fn cells(&self) -> &Vec<usize> {
        &self.cells
    }
}

// A KenKen struct has the dimension of the KenKen and a list of the regions
struct KenKen {
    order: usize,
    regions: Vec<Region>,
}

impl KenKen {
    // Constructor takes the order as a param and initializes regions with no data
    fn new(order: usize) -> Self {
        Self {
            order,
            regions: vec![], // TODO make this more complete
        }
    }

    // TODO maybe try making a to_string()

    pub fn order(&self) -> usize {
        self.order
    }

    pub fn regions(&self) -> &Vec<Region> {
        &self.regions
    }
}

// Extends LatinSolver in the sense that it solves specific KenKens by leaning on LatinSolver methods
struct KenKenSolver {
    ken_ken: KenKen,
    latin_solver: LatinSolver,
}

impl KenKenSolver {
    fn new(ken_ken: KenKen) -> KenKenSolver {
        let order = ken_ken.order();
        KenKenSolver {
            ken_ken,
            latin_solver: LatinSolver::new(order),
        }
    }

    // Returns true if the ken_ken_solver made any mutations.
    fn apply_constraint(&mut self, region_index: usize) -> bool {
        // find region associated with index
        // TODO unwrap value
        let region = &self.ken_ken.regions()[region_index];
        let operation = &region.clue().op;
        match operation {
            Operation::Given => {
                let (x, y) = math::xy_pair(region.cells()[0], self.ken_ken.order());
                self.latin_solver.place_digit(x, y, region.clue().target);
                true
            },
            Operation::Add => {

                true
            },
            Operation::Subtract => {
                let mut available_masks: Vec<usize> = vec![0; region.cells().len()];
                for i in 0..region.cells().len() {
                    let (x, y) = math::xy_pair(region.cells()[i], self.ken_ken.order());
                    let subarray = self.latin_solver.get_cube_loc_subarray(x, y);
                    let mut val = 0b0;

                    for (index, available) in subarray.iter().enumerate() {
                        val |= (*available as usize) << index; // maybe use .copy() or .clone() on available instead of deref
                    }
                    available_masks[i] = val;

                }

                let mask_a = available_masks[0];
                let mask_b = available_masks[1];
                let mask_a_greater = mask_a & mask_b << region.clue().target;
                let mask_b_greater = mask_a & mask_b >> region.clue().target;

                available_masks[0] = mask_a_greater | mask_b_greater;
                available_masks[1] = (mask_a_greater >> region.clue().target) | (mask_b_greater << region.clue().target);


                for i in 0..region.cells().len() {
                    let (x, y) = math::xy_pair(region.cells()[i], self.ken_ken.order());
                    let mut subarray: Vec<bool> = vec![];
                    // This whole loop is to convert back to a vector rather than the nice binary int
                    for index in 0..self.ken_ken.order() {
                        subarray.push((0b1 << index) & available_masks[i] != 0);
                    }
                    self.latin_solver.set_cube_loc_subarray(x, y, subarray);
                }
                // TODO choose when true and when false
                true
            },
            Operation::Multiply => {
                true
            },
            Operation::Divide => {
                true
            },
            _ => {false},
        }

    }
}

fn read_ken_ken(input: String) -> KenKen {
    // Takes a string input of a KenKen, and converts it to a KenKen object
    // Example format looks like 3: 3+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:

    // Takes a string input of a KenKen, and converts it to a KenKen object
    // Example format looks like "3: 3+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:"

    // TODO remove panics
    let order: usize = input[0..=0].to_string().parse().expect("Not a number");

    let mut ken_ken = KenKen::new(order);

    let mut chars = input[3..].chars();

    while let Some(c) = chars.next() {
        let target:usize = c.to_digit(10).expect("Not a digit") as usize;
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
        let mut cells: Vec<usize> = vec![];
        loop {
            // or use unwrap?
            let row: usize = chars.
                next().
                expect("No more characters for row")
                .to_digit(10).
                expect("Not a row digit") as usize;
            let col: usize = chars
                .next()
                .expect("No more characters")
                .to_digit(10)
                .expect("Not a row digit") as usize;
            cells.push(row * order + col);
            match chars.next() {
                Some(':') => break,
                Some(' ') => (),
                _ => panic!("Not a valid operation character"),
            };
        }

        ken_ken.regions.push(Region::new(Clue { op, target }, cells));
        chars.next();
    }
    ken_ken
}

pub fn main() {
    let k = read_ken_ken("3: 3+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
    println!("regions: {:?}, ", k.regions);
    let mut k_solver: KenKenSolver = KenKenSolver::new(k);
    k_solver.apply_constraint(1);
    println!("no debug:\n {}", k_solver.latin_solver.cube_to_string());
}
