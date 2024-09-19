// #![allow(dead_code)]

use itertools::Itertools;

// The possible operations on a clue for a KenKen
#[derive(Clone, Debug, strum_macros::Display)]
pub enum Operation {
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
pub struct Clue {
    op: Operation,
    target: usize,
}

impl Clue {
    pub fn new(op: Operation, target: usize) -> Self {
        Self { op, target }
    }

    pub fn to_string(&self) -> String {
        let op_string = match self.op {
            Operation::Add => "+",
            Operation::Subtract => "-",
            Operation::Multiply => "*",
            Operation::Divide => "/",
            Operation::Given => "G",
            Operation::Unknown => "Unknown",
            Operation::NoOp => "No-op"
        };

        self.target.to_string() + op_string
    }
}

// A data structure for each region of the ken ken puzzle.
// It will not know about the contents of the cells, just the clue and locations.
#[derive(Debug)]
pub struct Region {
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
        self.cells.sort(); // maintain sorted so the first cell is leftmost of top row
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
pub struct KenKen {
    order: usize,
    pub(crate) regions: Vec<Region>,
}

impl KenKen {
    // Constructor takes the order as a param and initializes regions with no data
    pub fn new(order: usize) -> Self {
        Self {
            order,
            regions: vec![], // Could potentially make this build the kenken
        }
    }

    pub fn read_ken_ken(input: String) -> KenKen {
        // Takes a string input of a KenKen, and converts it to a KenKen object
        // Example format looks like 3: 3+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:

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

    // Sort of workable to_string
    // Not as great when there are clues of more than 1 digit
    pub fn to_string(&self) -> String {
        assert!(self.regions().len() <= 26, "print won't work on too many regions");

        let mut arr: Vec<String> = vec!["".to_string(); self.order()*self.order()];

        for (index, region) in self.regions().iter().enumerate() {
            for cell in region.cells() {
                arr[*cell] = ((b'a' + index as u8) as char).to_string() + &region.clue.to_string();
            }
        }


        let mut result = String::from("");

        for i in 0..self.order() {
            let mut builder: Vec<String> = Vec::new();
            for j in 0..self.order() {
                builder.push(arr[i * self.order() + j].clone());
            }

            let row: String = " ".to_string() + &builder.iter().join(" | ") + &" ";
            let gap: String = if i < self.order() - 1 {
                "\n".to_string() + &"_".repeat((self.order()) * 6 - 1) + "\n"
            } else {
                "\n".to_string()
            };
            result = result + &row + &gap;
        }

        result
    }

    // Order getter
    pub fn order(&self) -> usize {
        self.order
    }

    // Regions getter
    pub fn regions(&self) -> &Vec<Region> {
        &self.regions
    }

    // Regions getter
    pub fn region_n(&self, n: usize) -> &Region {
        assert!(n < self.regions.len(), "Region index out of range.");
        &self.regions[n]
    }
    
    pub fn add_region(&mut self, r: Region) {
        self.regions.push(r);
    }
}

// Extends LatinSolver in the sense that it solves specific KenKens by leaning on LatinSolver methods
pub mod kenken_solve {
    use crate::grid::Grid;
    use crate::kenken::Operation;
    use crate::kenken::{KenKen, Region};
    use crate::latin::{latin_solve, SolvedStatus};
    use crate::math::math;
    use core::cmp::min;

    // Returns true if the ken_ken_solver made any mutations.
    fn apply_constraint(grid: &mut Grid, region: &Region) -> bool {
        // find region associated with index
        let operation = &region.clue().op;

        // Start with the vector of available digits in each cell
        let mut available_masks = available_masks(&grid, region);

        // Winnow down possibilities in the latin solver based on the operation being chosen.
        match operation {
            // If the cluie is a given,
            Operation::Given => {
                assert!(
                    region.clue().target >= 1 && region.clue().target <= grid.order(),
                    "target given value is either 0 or greater than the order."
                );

                // update the latin solver with the given value
                let new_mask = 0b1 << region.clue().target - 1;
                update_grid(grid, region, vec![new_mask]);
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
                    update_grid(grid, region, vec![0b0; region.cells().len()]);
                    return true;
                }

                // Find our degrees of freedom from the max and min for the region
                let room_from_min = region.clue().target - min_sum;
                let room_from_max = max_sum - region.clue().target;

                // update the candidates data structure with available masks
                for i in 0..region.cells().len() {
                    let (x, y) = math::xy_pair(region.cells()[i], grid.order());
                    let mut mask: i32 = available_masks[i];
                    let max_bit = math::max_bit(mask);

                    // update with mask of 1s that are available from min
                    mask &= (0b1 << (min(math::min_bit(mask) + room_from_min, grid.order()))) - 1;

                    // update with mask of 1s that are available from max
                    // This avoids subtractions problems being less than 0.
                    // If the condition is false, the max will exceed any possibilities
                    if max_bit > room_from_max {
                        mask &= -(0b1 << max_bit - room_from_max - 1); // This could be broken
                    }
                    grid.set_candidates_available(x, y, mask);
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

                update_grid(grid, region, available_masks);
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
                        if target % given == 0 && target / given <= order {
                            // If we can multiply the given by a value in the current cell to
                            // reach the target, update the last elem of new_masks and return true.
                            if target / given > 60 {
                                println!("am here");
                            }
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
                    grid.order(),
                    region.clue().target,
                    1,
                    &available_masks,
                );

                update_grid(grid, region, updated_masks);
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
                for i in 1..=grid.order() / 2 {
                    divide_helper(
                        mask_a,
                        mask_b,
                        &mut mask_a_updated,
                        &mut mask_b_updated,
                        region.clue().target,
                        i,
                    );
                    divide_helper(
                        mask_b,
                        mask_a,
                        &mut mask_b_updated,
                        &mut mask_a_updated,
                        region.clue().target,
                        i,
                    );
                }

                available_masks[0] = mask_a_updated;
                available_masks[1] = mask_b_updated;

                update_grid(grid, region, available_masks);
                // TODO choose when true and when false
                true
            }
            _ => false,
        }
    }

    // Helper function to call the LatinSolver method of setting available cell values in the candidates.
    fn update_grid(grid: &mut Grid, region: &Region, new_masks: Vec<i32>) {
        assert_eq!(region.cells().len(), new_masks.len());

        for i in 0..region.cells().len() {
            let (x, y) = math::xy_pair(region.cells()[i], grid.order());
            grid.set_candidates_available(x, y, new_masks[i]);
        }
    }

    // Helper function to grab the
    fn available_masks(grid: &Grid, region: &Region) -> Vec<i32> {
        let mut available_masks: Vec<i32> = vec![0b0; region.cells().len()];
        for i in 0..region.cells().len() {
            let (x, y) = math::xy_pair(region.cells()[i], grid.order());
            available_masks[i] = *grid.get_candidates_available(x, y);
        }
        available_masks
    }

    pub fn ken_ken_logical_solver(ken_ken: &KenKen) -> (SolvedStatus, Grid) {
        let mut g = Grid::new(ken_ken.order());
        let solved = ken_ken_logical_solver_with_grid(&mut g, ken_ken);
        (solved, g)
        // TODO do something with grid, allow me to see it when debugging
    }
    
    pub fn ken_ken_logical_solver_with_grid(grid: &mut Grid, ken_ken: &KenKen) -> SolvedStatus {
        let mut old_grid_candidates: Vec<i32> = vec![];
        
        // As long as we're making progress
        while old_grid_candidates != grid.candidates().clone() {
            old_grid_candidates = grid.candidates().clone();
            
            // Apply constraint to every region
            for region in ken_ken.regions() {
                apply_constraint(grid, region);
            }
            
            // Use our latin solver pruning
            match latin_solve::stepped_logical_solver(grid) {
                SolvedStatus::Complete => return SolvedStatus::Complete,
                SolvedStatus::Incomplete => (),
                SolvedStatus::Broken => return SolvedStatus::Broken,
            }
        }
        SolvedStatus::Incomplete
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::kenken::{kenken_solve, KenKen};

        // TODO refactor test code so it's not so repetitive.

        #[test]
        fn test_subtraction() {
            let k =
                KenKen::read_ken_ken("3: 3+ 00 01: 1- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(1));
            assert_eq!(*grid.get_candidates_available(0, 2), 0b111);
            assert_eq!(*grid.get_candidates_available(1, 2), 0b111);

            let k =
                KenKen::read_ken_ken("3: 3+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(1));
            assert_eq!(*grid.get_candidates_available(0, 2), 0b101);
            assert_eq!(*grid.get_candidates_available(1, 2), 0b101);

            let k =
                KenKen::read_ken_ken("3: 3+ 00 01: 3- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(1));
            assert_eq!(*grid.get_candidates_available(0, 2), 0b000);

            let k = KenKen::read_ken_ken(
                "4: 12* 00 01 11: 6+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                    .to_string(),
            );
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(3));
            assert_eq!(*grid.get_candidates_available(1, 3), 0b1111);
            assert_eq!(*grid.get_candidates_available(2, 3), 0b1111);

            let k = KenKen::read_ken_ken(
                "4: 12* 00 01 11: 6+ 02 03 12: 2/ 10 20: 3- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                    .to_string(),
            );
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(3));
            assert_eq!(*grid.get_candidates_available(1, 3), 0b1001);
            assert_eq!(*grid.get_candidates_available(2, 3), 0b1001);
        }

        #[test]
        fn test_addition() {
            let k =
                KenKen::read_ken_ken("3: 1+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(0));
            assert_eq!(*grid.get_candidates_available(0, 0), 0b000);
            assert_eq!(*grid.get_candidates_available(0, 1), 0b000);

            let k =
                KenKen::read_ken_ken("3: 3+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(0));
            assert_eq!(*grid.get_candidates_available(0, 0), 0b011);
            assert_eq!(*grid.get_candidates_available(0, 1), 0b011);

            let k =
                KenKen::read_ken_ken("3: 5+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(0));
            assert_eq!(*grid.get_candidates_available(0, 0), 0b110);
            assert_eq!(*grid.get_candidates_available(0, 1), 0b110);

            let k =
                KenKen::read_ken_ken("3: 7+ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(0));
            assert_eq!(*grid.get_candidates_available(0, 0), 0b000);
            assert_eq!(*grid.get_candidates_available(0, 1), 0b000);

            let k = KenKen::read_ken_ken(
                "4: 12* 00 01 11: 6+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                    .to_string(),
            );
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(1));
            assert_eq!(*grid.get_candidates_available(0, 2), 0b1111);
            assert_eq!(*grid.get_candidates_available(0, 3), 0b1111);
            assert_eq!(*grid.get_candidates_available(1, 2), 0b1111);

            let k = KenKen::read_ken_ken(
                "4: 12* 00 01 11: 4+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                    .to_string(),
            );
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(1));
            assert_eq!(*grid.get_candidates_available(0, 2), 0b0011);
            assert_eq!(*grid.get_candidates_available(0, 3), 0b0011);
            assert_eq!(*grid.get_candidates_available(1, 2), 0b0011);

            let k = KenKen::read_ken_ken(
                "4: 12* 00 01 11: 11+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                    .to_string(),
            );
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(1));
            assert_eq!(*grid.get_candidates_available(0, 2), 0b1100);
            assert_eq!(*grid.get_candidates_available(0, 3), 0b1100);
            assert_eq!(*grid.get_candidates_available(1, 2), 0b1100);
        }

        #[test]
        fn test_division() {
            let k =
                KenKen::read_ken_ken("3: 2/ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(0));
            assert_eq!(*grid.get_candidates_available(0, 0), 0b011);
            assert_eq!(*grid.get_candidates_available(0, 1), 0b011);

            let k =
                KenKen::read_ken_ken("3: 3/ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(0));
            assert_eq!(*grid.get_candidates_available(0, 0), 0b101);
            assert_eq!(*grid.get_candidates_available(0, 1), 0b101);

            let k =
                KenKen::read_ken_ken("3: 4/ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(0));
            assert_eq!(*grid.get_candidates_available(0, 0), 0b000);
            assert_eq!(*grid.get_candidates_available(0, 1), 0b000);

            let k = KenKen::read_ken_ken(
                "4: 12* 00 01 11: 6+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                    .to_string(),
            );
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(2));
            assert_eq!(*grid.get_candidates_available(1, 0), 0b1011);
            assert_eq!(*grid.get_candidates_available(2, 0), 0b1011);
        }

        #[test]
        fn test_multiplication() {
            let k =
                KenKen::read_ken_ken("3: 2* 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(0));
            assert_eq!(*grid.get_candidates_available(0, 0), 0b011);
            assert_eq!(*grid.get_candidates_available(0, 1), 0b011);

            let k =
                KenKen::read_ken_ken("3: 3* 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(0));
            assert_eq!(*grid.get_candidates_available(0, 0), 0b101);
            assert_eq!(*grid.get_candidates_available(0, 1), 0b101);

            let k =
                KenKen::read_ken_ken("3: 5* 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(0));
            assert_eq!(*grid.get_candidates_available(0, 1), 0b000);
            assert_eq!(*grid.get_candidates_available(0, 0), 0b000);

            let k = KenKen::read_ken_ken(
                "4: 12* 00 01 11: 6+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                    .to_string(),
            );
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(0));
            //println!("{}", k_solver.latin_solver.candidates_to_string());
            assert_eq!(*grid.get_candidates_available(0, 0), 0b1111);
            assert_eq!(*grid.get_candidates_available(0, 1), 0b1111);
            assert_eq!(*grid.get_candidates_available(1, 1), 0b1111);
        }

        #[test]
        fn test_given() {
            let k =
                KenKen::read_ken_ken("3: 2/ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(2));
            assert_eq!(*grid.get_candidates_available(2, 2), 0b010);

            let k =
                KenKen::read_ken_ken("3: 2/ 00 01: 2- 02 12: 3 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(2));
            assert_eq!(*grid.get_candidates_available(2, 2), 0b100);

            let k = KenKen::read_ken_ken(
                "4: 12* 00 01 11: 6+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                    .to_string(),
            );
            let mut grid: Grid = Grid::new(k.order());
            kenken_solve::apply_constraint(&mut grid, k.region_n(4));
            assert_eq!(*grid.get_candidates_available(3, 0), 0b0001);
        }

        #[test]
        fn holistic_test() {
            let k =
                KenKen::read_ken_ken("3: 2/ 00 01: 2- 02 12: 2 22: 9+ 10 11 20 21:".to_string());
            let mut grid: Grid = Grid::new(k.order());
            for region in k.regions() {
                apply_constraint(&mut grid, region);
            }
            assert_eq!(*grid.get_candidates_available(2, 2), 0b010);

            let (solved, grid) = ken_ken_logical_solver(&k);
            
            let k = KenKen::read_ken_ken(
                "4: 12* 00 01 11: 6+ 02 03 12: 2/ 10 20: 1- 13 23: 1 30: 1- 21 31: 8* 22 32 33:"
                    .to_string(),
            );
            let mut grid: Grid = Grid::new(k.order());
            for region in k.regions() {
                kenken_solve::apply_constraint(&mut grid, region);
            }
            assert_eq!(*grid.get_candidates_available(3, 0), 0b0001);
        }

        #[test]
        fn broken_puzzle() {
            let k = KenKen::read_ken_ken("3: 81* 00 01 02 11: 81* 10 20 22 21: 3 12:".to_string());
            let (solved, grid) = ken_ken_logical_solver(&k);
            assert_eq!(SolvedStatus::Broken, solved);

            let k = KenKen::read_ken_ken("3: 12+ 00 01 02 10 11 10 20 22: 2 21:".to_string());
            let (solved, grid) = ken_ken_logical_solver(&k);
            
            println!("here\n {:?} \n {}", solved, grid.candidates_to_string());
            println!("{}", k.to_string());

            // assert_eq!(SolvedStatus::Broken, solved); // TODO eventually make sure this assert passes
        }
        #[test]
        fn broken_grid() {
            let k = KenKen::read_ken_ken("3: 81* 00 01 02 10:".to_string());
            let (solved, grid) = ken_ken_logical_solver(&k);
            assert_eq!(SolvedStatus::Broken, solved);
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
}

#[cfg(test)]
mod tests {
    use super::KenKen;

    #[test]
    fn ken_ken_to_string() {
        let k = KenKen::read_ken_ken("3: 2/ 00 01: 2- 02 12: 3 22: 9+ 10 11 20 21:".to_string());
        //println!("{}", k.to_string());
    }

}
