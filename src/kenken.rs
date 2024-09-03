#![allow(dead_code)]
use crate::latin::LatinSolver;
use core::cmp::min;

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

// Some useful mathemtatic operations reused in a couple different contexts
pub mod math {
    // Given an index into a flat array, return the (x,y) index pair.
    pub fn xy_pair(index: usize, order: usize) -> (usize, usize) {
        assert!(index < order.pow(2), "The index provided to xy_pair is out of range.");
        (index / order, index % order)
    }

    // Returns the location of the rightmost set bit
    // Note that this is in digit notation, so 0b110 returns 2
    pub fn min_bit(mask: i32) -> usize {
        assert_ne!(mask, 0b0, "Trying to find min_bit but no bits are set.");
        ((mask & -mask) as f32).log2() as usize + 1
    }

    // Returns the location of the leftmose set bit
    // Note that this is in digit notation, so 0b110 returns 3
    pub fn max_bit(mask: i32) -> usize {
        assert_ne!(mask, 0b0, "Trying to find max_bit but no bits are set.");
        (mask as f32 + 1.0).log2().ceil() as usize
    }
}

// A data structure for each region of the ken ken puzzle.
// It will not know about the contents of the cells, just the clue and locations.
#[derive(Debug)]
struct Region {
    clue: Clue,
    cells: Vec<usize>,
}

impl Region {
    // Constructor takes a Clue type and a vector of cell indices in a flattended 2-d vector
    pub fn new(clue: Clue, cells: Vec<usize>) -> Self {
        Self { clue, cells }
    }

    // Clue setter
    pub fn set_clue(&mut self, clue: Clue) {
        self.clue = clue;
    }

    // Add cell index to cell vector
    pub fn add_cell(&mut self, cell: usize) {
        self.cells.push(cell);
    }

    // Removes cell iff it is present.
    pub fn remove_cell(&mut self, cell: usize) {
        self.cells.retain(|&c| c != cell);
    }

    // CLue getter
    pub fn clue(&self) -> &Clue {
        &self.clue
    }

    // Cell getter
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
            regions: vec![], // Could potentially make this build the kenken
        }
    }

    // TODO maybe try making a to_string()

    // Order getter
    pub fn order(&self) -> usize {
        self.order
    }

    // Regions getter
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
    // Takes a kenken, and builds a generic latin solver to overlay on top
    fn new(ken_ken: KenKen) -> KenKenSolver {
        let order = ken_ken.order();
        KenKenSolver {
            ken_ken,
            latin_solver: LatinSolver::new(order),
        }
    }

    // Returns true if the ken_ken_solver made any mutations.
    fn apply_constraint(&mut self, region_index: usize) -> bool {
        assert!(region_index < self.ken_ken.regions().len(),
                "Region selected in apply_contstraint is out of range.");

        // find region associated with index
        let region = &self.ken_ken.regions()[region_index];
        let operation = &region.clue().op;

        // Start with the vector of available digits in each cell
        let mut available_masks = self.available_masks(region);

        // Winnow down possibilities in the latin solver based on the operation being chosen.
        match operation {
            // If the cluie is a given,
            Operation::Given => {
                assert!(region.clue().target >= 1 && region.clue().target <= self.ken_ken().order(),
                        "target given value is either 0 or greater than the order.");

                // update the latin solver with the given value
                let new_mask = 0b1 << region.clue().target - 1;
                Self::update_latin_solver(&mut self.latin_solver, region, vec![new_mask]);
                true // TODO decide how to handle true/false
            }
            Operation::Add => {
                // TODO only winnows based on min/max, doesn't rule out all like the multiply branch

                // Then find the min and max sums of the region given the available cells
                let mut min_sum = 0;
                let mut max_sum = 0;
                for mask in &available_masks {
                    min_sum += math::min_bit(*mask); // or mask.clone() instead??
                    max_sum += math::max_bit(*mask);
                }

                // If target is over max or under min, set all possibilities to false.
                if region.clue().target < min_sum || region.clue().target > max_sum {
                    Self::update_latin_solver(
                        &mut self.latin_solver,
                        region,
                        vec![0b0; region.cells().len()],
                    );
                    return true;
                }

                // Find our degrees of freedom from the max and min for the region
                let room_from_min = region.clue().target - min_sum;
                let room_from_max = max_sum - region.clue().target;

                // update the cube data structure with available masks
                for i in 0..region.cells().len() {
                    let (x, y) = math::xy_pair(region.cells()[i], self.ken_ken().order());
                    let mut mask: i32 = available_masks[i];
                    let max_bit = math::max_bit(mask);

                    // update with mask of 1s that are available from min
                    mask &= (0b1
                        << (min(math::min_bit(mask) + room_from_min, self.ken_ken().order())))
                        - 1;

                    // update with mask of 1s that are available from max
                    // This avoids subtractions problems being less than 0.
                    // If the condition is false, the max will exceed any possibilities
                    if max_bit > room_from_max {
                        mask &= -(0b1 << max_bit - room_from_max - 1); // This could be broken
                    }
                    self.latin_solver.set_cube_available(x, y, mask);
                }
                // TODO choose when true and when false
                true
            }
            Operation::Subtract => {
                // Get the two masks - subtract regions must have length 2.
                let mask_a = available_masks[0];
                let mask_b = available_masks[1];

                // Shift each masks over and and it with the other. This will give us
                // a mask where the 1's are set if the values subtract to the target
                let mask_a_greater = mask_a & mask_b << region.clue().target;
                let mask_b_greater = mask_a & mask_b >> region.clue().target;

                // Update the avalable masks accordingly
                available_masks[0] = mask_a_greater | mask_b_greater;
                available_masks[1] = (mask_a_greater >> region.clue().target)
                    | (mask_b_greater << region.clue().target);

                Self::update_latin_solver(&mut self.latin_solver, region, available_masks);
                // TODO choose when true and when false
                true
            }
            Operation::Multiply => {
                // Takes a given digit from a cell, then compares product possibilities
                // with the other masks. Returns a vector of masks with on bits for hits
                fn multiply_helper(
                    new_masks: &mut Vec<i32>,
                    order: usize,
                    target: usize,
                    given: usize,
                    other_masks: &Vec<i32>,
                ) -> bool {
                    if other_masks.len() == 0 {
                        panic!("should never have 0 len");
                    }

                    // base case: there is only one other mask to check
                    if other_masks.len() == 1 {
                        if target % given == 0 {
                            // If we can multiply the given by a value in the current cell to
                            // reach the target, update the last elem of new_masks and return true.
                            if (0b1 << ((target / given) - 1)) & other_masks[0] != 0b0 {
                                let len = new_masks.len();
                                new_masks[len - 1] |= 0b1 << ((target / given) - 1);
                                return true;
                            }
                            return false;
                        } else {
                            return false;
                        }
                    }

                    // new_mask is the next on the list.
                    let len = new_masks.len(); // So we don't immutably borrow
                    let mut new_mask: i32 = new_masks[len - other_masks.len()];

                    // for all the possible digits
                    for i in 1..=order {
                        // First check if the digit is in the current cell
                        if (0b1 << (i - 1)) & other_masks[0] != 0 {

                            // recursive call including i from the current cell
                            if multiply_helper(
                                new_masks,
                                order,
                                target,
                                given * i,
                                &other_masks[1..].to_vec(),
                            ) {
                                new_mask |= 0b1 << (i - 1);
                            }
                        }
                    }

                    new_masks[len - other_masks.len()] = new_mask;
                    return new_mask != 0b0;
                }

                let mut updated_masks: Vec<i32> = vec![0b0; region.cells().len()];
                multiply_helper(
                    &mut updated_masks,
                    self.ken_ken().order(),
                    region.clue().target,
                    1,
                    &available_masks,
                );

                Self::update_latin_solver(&mut self.latin_solver, region, updated_masks);
                true //TODO
            }
            Operation::Divide => {
                // Similar to subtract,
                let mask_a = available_masks[0];
                let mask_b = available_masks[1];
                let mut mask_a_updated = 0b0;
                let mut mask_b_updated = 0b0;

                // Updates the masks in place, if i * x = target and i is available in mask a,
                // and x is available in mask b.
                fn divide_helper(
                    mask_a: i32,
                    mask_b: i32,
                    mask_a_updated: &mut i32,
                    mask_b_updated: &mut i32,
                    target: usize,
                    i: usize,
                ) {
                    assert!(target > 1, "Division clue target cannot be 1 or 0");
                    if ((0b1 << (i - 1)) & mask_a != 0)
                        && ((0b1 << ((i * target) - 1)) & mask_b != 0)
                    {
                        *mask_a_updated |= 0b1 << (i - 1);
                        *mask_b_updated |= 0b1 << ((i * target) - 1);
                    }
                }

                // Dividing by 2 because any larger value won't have a higher pair
                for i in 1..=self.ken_ken().order() / 2 {
                    divide_helper(mask_a, mask_b, &mut mask_a_updated, &mut mask_b_updated, region.clue().target, i);
                    divide_helper(mask_b, mask_a, &mut mask_b_updated, &mut mask_a_updated, region.clue().target, i);
                }

                available_masks[0] = mask_a_updated;
                available_masks[1] = mask_b_updated;

                Self::update_latin_solver(&mut self.latin_solver, region, available_masks);
                // TODO choose when true and when false
                true
            }
            _ => false,
        }
    }

    // Helper function to call the LatinSolver method of setting available cell values in the cube.
    fn update_latin_solver(latin: &mut LatinSolver, region: &Region, mut new_masks: Vec<i32>) {
        assert_eq!(region.cells().len(), new_masks.len());

        for i in 0..region.cells().len() {
            let (x, y) = math::xy_pair(region.cells()[i], latin.order());
            latin.set_cube_available(x, y, new_masks[i]);
        }
    }

    // Helper function to grab the
    fn available_masks(&self, region: &Region) -> Vec<i32> {
        let mut available_masks: Vec<i32> = vec![0b0; region.cells().len()];
        for i in 0..region.cells().len() {
            let (x, y) = math::xy_pair(region.cells()[i], self.ken_ken.order());
            available_masks[i] = self.latin_solver.get_cube_available(x, y);
        }
        available_masks
    }

    // kenken getter
    pub fn ken_ken(&self) -> &KenKen {
        &self.ken_ken
    }

    // latin solver getter
    pub fn latin_solver(&self) -> &LatinSolver {
        &self.latin_solver
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

    // Skip to fourth character.
    let mut chars = input[3..].chars();

    while let Some(character) = chars.next() {
        // Target may be mutliple digits long
        let mut target_builder: String = String::new();
        let mut c = character;
        while c.is_digit(10) {
            target_builder.push(c);
            c = chars.next().unwrap();
        }
        let target: usize = target_builder.parse::<i32>().expect("Not a digit") as usize;
        let op: Operation = match c {
            '+' => Operation::Add,
            '-' => Operation::Subtract,
            '*' => Operation::Multiply,
            '/' => Operation::Divide,
            ' ' => Operation::Given,
            '?' => Operation::Unknown,
            _ => panic!("Not a valid operation character"),
        };
        if !matches!(op, Operation::Unknown | Operation::Given) {
            chars.next();
        }
        let mut cells: Vec<usize> = vec![];
        loop {
            // or use unwrap?
            let row: usize = chars
                .next()
                .expect("No more characters for row")
                .to_digit(10)
                .expect("Not a row digit") as usize;
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

        ken_ken
            .regions
            .push(Region::new(Clue { op, target }, cells));
        chars.next();
    }
    ken_ken
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO refactor test code so it's not so repetitive.
    

    #[test]
    fn test_subtraction() {
        let k = read_ken_ken("3: 3+ 00 01: 1- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(1);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 2), 0b111);
        assert_eq!(k_solver.latin_solver.get_cube_available(1, 2), 0b111);

        let k = read_ken_ken("3: 3+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(1);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 2), 0b101);
        assert_eq!(k_solver.latin_solver.get_cube_available(1, 2), 0b101);

        let k = read_ken_ken("3: 3+ 00 01: 3- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(1);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 2), 0b000);
        assert_eq!(k_solver.latin_solver.get_cube_available(1, 2), 0b000);

        let k = read_ken_ken(
            "4: 12* 00 01 11: 6+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                .to_string(),
        );
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(3);
        assert_eq!(k_solver.latin_solver.get_cube_available(1, 3), 0b1111);
        assert_eq!(k_solver.latin_solver.get_cube_available(2, 3), 0b1111);

        let k = read_ken_ken(
            "4: 12* 00 01 11: 6+ 02 03 12: 2/ 10 20: 3- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                .to_string(),
        );
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(3);
        assert_eq!(k_solver.latin_solver.get_cube_available(1, 3), 0b1001);
        assert_eq!(k_solver.latin_solver.get_cube_available(2, 3), 0b1001);
    }

    #[test]
    fn test_addition() {
        let k = read_ken_ken("3: 1+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(0);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 0), 0b000);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 1), 0b000);

        let k = read_ken_ken("3: 3+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(0);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 0), 0b011);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 1), 0b011);

        let k = read_ken_ken("3: 5+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(0);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 0), 0b110);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 1), 0b110);

        let k = read_ken_ken("3: 7+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(0);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 0), 0b000);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 1), 0b000);

        let k = read_ken_ken(
            "4: 12* 00 01 11: 6+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                .to_string(),
        );
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(1);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 2), 0b1111);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 3), 0b1111);
        assert_eq!(k_solver.latin_solver.get_cube_available(1, 2), 0b1111);

        let k = read_ken_ken(
            "4: 12* 00 01 11: 4+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                .to_string(),
        );
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(1);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 2), 0b0011);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 3), 0b0011);
        assert_eq!(k_solver.latin_solver.get_cube_available(1, 2), 0b0011);

        let k = read_ken_ken(
            "4: 12* 00 01 11: 11+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                .to_string(),
        );
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(1);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 2), 0b1100);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 3), 0b1100);
        assert_eq!(k_solver.latin_solver.get_cube_available(1, 2), 0b1100);
    }

    #[test]
    fn test_division() {
        let k = read_ken_ken("3: 2/ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(0);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 0), 0b011);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 1), 0b011);

        let k = read_ken_ken("3: 3/ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(0);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 0), 0b101);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 1), 0b101);

        let k = read_ken_ken("3: 4/ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(0);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 0), 0b000);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 1), 0b000);

        let k = read_ken_ken(
            "4: 12* 00 01 11: 6+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                .to_string(),
        );
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(2);
        assert_eq!(k_solver.latin_solver.get_cube_available(1, 0), 0b1011);
        assert_eq!(k_solver.latin_solver.get_cube_available(2, 0), 0b1011);
    }

    #[test]
    fn test_multiplication() {
        let k = read_ken_ken("3: 2* 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(0);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 0), 0b011);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 1), 0b011);

        let k = read_ken_ken("3: 3* 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(0);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 0), 0b101);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 1), 0b101);

        let k = read_ken_ken("3: 5* 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(0);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 1), 0b000);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 0), 0b000);

        let k = read_ken_ken(
            "4: 12* 00 01 11: 6+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                .to_string(),
        );
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(0);
        //println!("{}", k_solver.latin_solver.cube_to_string());
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 0), 0b1111);
        assert_eq!(k_solver.latin_solver.get_cube_available(0, 1), 0b1111);
        assert_eq!(k_solver.latin_solver.get_cube_available(1, 1), 0b1111);
    }

    #[test]
    fn test_given() {
        let k = read_ken_ken("3: 2/ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(2);
        assert_eq!(k_solver.latin_solver.get_cube_available(2, 2), 0b010);

        let k = read_ken_ken("3: 2/ 00 01: 2- 02 12: 3 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(2);
        assert_eq!(k_solver.latin_solver.get_cube_available(2, 2), 0b100);

        let k = read_ken_ken(
            "4: 12* 00 01 11: 6+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                .to_string(),
        );
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        k_solver.apply_constraint(4);
        assert_eq!(k_solver.latin_solver.get_cube_available(3, 0), 0b0001);
    }

    #[test]
    fn wholistic_test() {
        let k = read_ken_ken("3: 2/ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        for i in 0..k_solver.ken_ken().regions().len() {
            k_solver.apply_constraint(i);
        }
        assert_eq!(k_solver.latin_solver.get_cube_available(2, 2), 0b010);

        let k = read_ken_ken(
            "4: 12* 00 01 11: 6+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                .to_string(),
        );
        let mut k_solver: KenKenSolver = KenKenSolver::new(k);
        for i in 0..k_solver.ken_ken().regions().len() {
            k_solver.apply_constraint(i);
        }
        assert_eq!(k_solver.latin_solver.get_cube_available(3, 0), 0b0001);
    }

    #[test]
    fn bit_tests() {
        let number = 0b1;
        assert_eq!(math::max_bit(number), 1);
        assert_eq!(math::min_bit(number), 1);

        let number = 0b10;
        assert_eq!(math::max_bit(number), 2);
        assert_eq!(math::min_bit(number), 2);

        let number = 0b11;
        assert_eq!(math::max_bit(number), 2);
        assert_eq!(math::min_bit(number), 1);

        let number = 0b1001;
        assert_eq!(math::max_bit(number), 4);
        assert_eq!(math::min_bit(number), 1);

        let number = 0b00100100;
        assert_eq!(math::max_bit(number), 6);
        assert_eq!(math::min_bit(number), 3);

        let number = 0b00101110100;
        assert_eq!(math::max_bit(number), 9);
        assert_eq!(math::min_bit(number), 3);
    }
}
