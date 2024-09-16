#![allow(dead_code)]

use itertools::Itertools;
use std::collections::HashMap;
use crate::math::math;

/// Enum for the solver to track the status of completion
#[derive(Debug, PartialEq)]
pub enum SolvedStatus {
    Complete,
    Incomplete,
    Broken,
}

/// An object that contains solving data for Latin Square puzzles
pub struct SolvingGrid {
    order: usize,     // the dimension of the square KenKen grid
    candidates: Vec<i32>,   // order^2 - a 2d array of binary ints. 0b111111

    // might be useful to have grid appear elsewhere as its own type
    digits: Vec<usize>, // order^2
}

pub struct LatinSolver {
    order: usize,     // the dimension of the square KenKen grid
    cube: Vec<i32>,   // order^2 - a 2d array of binary ints. 0b111111

    // might be useful to have grid appear elsewhere as its own type
    grid: Vec<usize>, // order^2
}

impl LatinSolver {
    /**************************** Initializers ****************************/

    pub fn new(order: usize) -> LatinSolver {
        LatinSolver {
            order,
            cube: vec![(0b1 << order) - 1; order.pow(2)], // 0 at bit k when k is not possible in that cell
            grid: vec![0; order.pow(2)], // The completed grid of values 1 through order, 0s before solved
        }
    }

    /**************************** Getter functions ****************************/

    pub fn order(&self) -> usize {
        self.order
    }

    pub fn cube(&self) -> &Vec<i32> {
        &self.cube
    }

    pub fn grid(&self) -> &Vec<usize> {
        &self.grid
    }

    // Get the location at coordinates (x,y)
    fn get_loc(&self, x: usize, y: usize) -> usize {
        x * self.order + y
    }


    /**************************** Cube functions ****************************/

    // TODO eventually delete these first two functions.
    // True means the value (n) is still possible at that coordinate (x,y)
    fn get_cube_value(&self, x: usize, y: usize, n: usize) -> bool {
        let location = self.get_loc(x, y);
        (0b1 << (n-1)) & self.cube[location] != 0
    }

    // Update the cube data structure to be true or false at (x,y) to bool b
    pub fn set_cube_value(&mut self, x: usize, y: usize, n: usize, b: bool) -> () {
        let location = self.get_loc(x, y);
        match b {
            true => { // turn on the bit
                self.cube[location] |= 0b1 << (n-1);
            },
            false => { // turn off the bit
                self.cube[location] &= !(0b1 << (n-1));
            },
        }
    }

    // Get the cube data structure representing available digits at (x, y)
    pub fn get_cube_available(&self, x: usize, y: usize) -> &i32 {
        &self.cube[self.get_loc(x, y)]
    }

    // Update a subarray at a particular position with pruned (or expanded) choices of available digits
    pub fn set_cube_available(&mut self, x: usize, y: usize, available: i32) -> () {
        assert!(available < 0b1 << self.order && available >= 0);
        let location = self.get_loc(x, y);
        self.cube[location] = available;
    }

    // Returns a vector of a row or column from the cube (a copy)
    fn get_cube_vector(&self, index: usize, is_row: bool) -> Vec<i32> {
        match is_row {
            // Take a row slice
            true => self.cube()[index*self.order()..(index+1)*self.order()].to_vec(),
            // Take every nth element, since we are in a flattened 2d vector
            false => self.cube().iter()
                .skip(index)
                .step_by(self.order())
                .copied().collect(),
        }
    }

    // To_String method for the cube data structure
    pub fn cube_to_string(&self) -> String {
        let mut result = String::from("");

        for i in 0..self.order {
            // Make an array of all the contents of each cell in the row
            let mut row_arr: Vec<String> = Vec::new();
            for j in 0..self.order {
                // Make an array for the contents of each cell
                // It will have a digit if the digit is still available, otherwise a '*'
                let mut cell_arr: Vec<char> = Vec::new();
                for n in 1..=self.order {
                    if self.get_cube_value(i, j, n) {
                        cell_arr.push(char::from_digit(n as u32, 10).unwrap());
                    } else {
                        cell_arr.push('*');
                    }
                }
                row_arr.push(cell_arr.iter().join(""));
            }

            // At the end of the row, add the row, then add a gap below it
            let row = " ".to_string() + &row_arr.iter().join(" | ") + &" ";
            let gap = if i < self.order - 1 {
                "\n".to_string() + &"_".repeat(self.order * (self.order + 3) - 1) + "\n"
            } else {
                "\n".to_string()
            };
            result = result + &row + &gap;
        }
        result
    }

    /**************************** Grid functions ****************************/

    // Get the value at the grid
    fn get_grid_value(&self, x: usize, y: usize) -> usize {
        self.grid[self.get_loc(x, y)]
    }

    // Set the final value in the grid of where the digit belongs
    fn set_grid_value(&mut self, x: usize, y: usize, n: usize) -> () {
        // First assert attempt
        assert!(
            x < self.order && y < self.order && n <= self.order,
            "All quantities must be within the grid dimensions"
        );
        let location = self.get_loc(x, y);
        self.grid[location] = n;
    }

    // Set grid value to 0 at location (x, y)
    fn reset_grid_value(&mut self, x: usize, y: usize) -> () {
        self.set_grid_value(x, y, 0);
    }

    // To string method for the cube data structure
    pub fn grid_to_string(&self) -> String {
        let mut result = String::from("");

        for i in 0..self.order {
            let mut arr: Vec<usize> = Vec::new();
            for j in 0..self.order {
                arr.push(self.get_grid_value(i, j));
            }

            let row: String = " ".to_string() + &arr.iter().join(" | ") + &" ";
            let gap: String = if i < self.order - 1 {
                "\n".to_string() + &"_".repeat((self.order) * 4 - 1) + "\n"
            } else {
                "\n".to_string()
            };
            result = result + &row + &gap;
        }
        result
    }

    /************************** Solving functions **************************/

    // Place a digit in the final grid,
    // and update the data structures storing the availability of digits
    pub fn place_digit_xy(&mut self, x: usize, y: usize, n: usize) -> () {
        // place it in the grid structure
        self.set_grid_value(x, y, n);

        // update the cube structure along the row
        for i in 0..self.order {
            if i != x {
                self.set_cube_value(i, y, n, false)
            }
        }

        // update the cube structure along the column
        for i in 0..self.order {
            if i != y {
                self.set_cube_value(x, i, n, false)
            }
        }

        // update all other n's at that location
        for i in 1..=self.order {
            if i != n {
                self.set_cube_value(x, y, i, false);
            }
        }
        //old_data
    }
    
    pub fn place_digit_xy_deepcopy(&mut self, x: usize, y: usize, n: usize) -> (Vec<usize>, Vec<i32>) {
        let old_data: (Vec<usize>, Vec<i32>) = (self.grid.clone(), self.cube.clone());
        self.place_digit_xy(x, y, n);
        old_data
    }

    pub fn place_digit_flat(&mut self, flat_index: usize, n: usize) -> () {
        let (x, y) = math::xy_pair(flat_index, self.order());
        self.place_digit_xy(x, y, n)
    }

    // Given a row, remove the candidate from all cells in cube except the locations provided in do_not_remove
    // TODO MAYBE try not to call if row has no changes, cuz this looks sorta expensive
    // Returns true if candidates were removed
    fn cube_remove_candidate(&mut self, axis_a: usize, digit_mask: i32, is_row: bool, do_not_remove: &Vec<usize>) -> bool {
        assert_ne!(&Vec::<usize>::new(), do_not_remove, "cannot remove candidate from every position");
        assert_ne!(0b0, digit_mask, "Cannot remove no digit");
        assert_ne!(0b0, !digit_mask, "Cannot remove all digits");
        let mut result = false;

        // digit mask is 1 at position n-1 for the digit n's we want to remove.

        for axis_b in 0..self.order() {
            if !do_not_remove.contains(&axis_b) { // skip unless not in do not remove
                let (row, col) = match is_row {
                    true => (axis_a, axis_b),
                    false => (axis_b, axis_a),
                };
                let old_mask = *self.get_cube_available(row, col);
                let new_mask = old_mask & !digit_mask;
                self.set_cube_available(row, col, new_mask);

                // updates result to true if old_mask is different from new_mask
                result |= old_mask != new_mask;
            }
        }
        result
    }

    /******************************** Static functions ********************************/

    // Returns a vector where the mask at vec[i] are the locations that the digit i-1 is available
    fn flip_cube_structure(set: &[i32]) -> Vec<i32> {
        let mut result: Vec<i32> = vec![0b0; set.len()];
        for (index, elem) in set.iter().enumerate() {
            for bit in 0..set.len() {
                if elem & (0b1 << bit) != 0 {
                    result[bit] |= 0b1 << index;
                }
            }
        }
        result
    }

    // Returns the first instance of a naked single, specifically an optional of the index and row
    fn naked_single_finder(set: &[i32]) -> Option<(usize, &i32)> {
        for (index, cell) in set.iter().enumerate() {
            // if there is a single digit
            if cell != &0b0 && (cell & (cell - 1)) == 0 {
                return Some((index, cell));
            }
        }
        None
    }
    
    // First variable that gets returned are the indices of the pair.
    // Second variable is a mask digits that make up the pair, i.e. the candidates in the cube.
    fn naked_pair_finder(set: &[i32])-> Option<(Vec<usize>, &i32)> {
        let mut found: HashMap<i32, usize> = HashMap::new(); // map from availables to index
        for (index, cell) in set.iter().enumerate() {
            if cell.count_ones() == 2 {
                match found.get(&cell) {
                    Some(old_index) => {
                        return Some((vec![index, *old_index], cell));
                    },
                    None => {
                        found.insert(*cell, index);
                    },
                }
            }
        }
        None
    }

    // TODO
    fn naked_set(set: &[i32], set_size: u32) -> Option<(usize, &i32)> {
        for (index, cell) in set.iter().enumerate() {
            // if there is a single digit
            if cell.count_ones() == set_size {
                return Some((index, cell));
            }
            
            
        }
        None
    }

    /************************** Brute force solving functions **************************/
    
    // Returns the location and vector of available digits at that location from the cube structure
    // Or, if no location is found with multiple possibilities, returns none
    // CAREFUL - it updates the cube and grid structures if they are out of sync.
    // TODO later change this to read-only on the self param
    // TODO delete this function... why do I need it?
    fn find_unsolved_location(&mut self) -> Option<(usize, usize, Vec<usize>)> {
        for i in 0..self.order {
            for j in 0..self.order {
                // Digits from 1 through order that are available at that location
                let mut available_digits: Vec<usize> = Vec::new();

                for n in 1..=self.order {
                    if self.get_cube_value(i, j, n)  {
                        available_digits.push(n);
                    }
                }

                // Return the digits available at i, j
                if available_digits.len() > 1 {
                    return Some((i, j, available_digits));

                // otherwise make sure all data structures updated
                } else if available_digits.len() == 1 {
                    if self.get_grid_value(i, j) == 0 {
                        // This might cause problems of version/clone
                        self.place_digit_xy(i, j, available_digits[0]);
                    }
                } else {
                    // Should only happen when some location has no possibilities
                    return None;
                }
            }
        }
        // Should only happen when whole puzzle is solved.
        None
    }

    // Returns true iff puzzle is solved.
    fn check_solved(&self) -> bool {
        for &elem in &self.grid {
            if elem == 0 {
                return false;
            }
        }
        true
    }

    // Solve by recursively guessing, then backtracking up the tree
    fn recursive_solve(&mut self, count: &mut u64, deep: bool) -> u64 {
        if let Some((x, y, available_digits)) = self.find_unsolved_location() {
            for n in available_digits {
                // Don't loop if we already finished and don't want to search exhaustively
                if !deep && *count > 0 {
                    break;
                }

                // Solve and then restore data structures from before said solve
                let (prev_grid, prev_cube) = self.place_digit_xy_deepcopy(x, y, n);
                self.recursive_solve(count, deep);
                self.cube = prev_cube;
                self.grid = prev_grid;
            }
            *count

        // else no available digits were found
        // check if this means we're solved or not
        } else {
            // todo maybe revisit and fix
            if self.check_solved() {
                *count += 1;

                // Print the first solved puzzle
                if *count == 1 {
                    println!("{}", self.cube_to_string());
                    println!("{}", self.grid_to_string());
                }
            }
            *count
        }
    }

    pub fn solve(&mut self, deep: bool) -> u64 {
        let mut count: u64 = 0;

        self.recursive_solve(&mut count, deep)
    }

    /********************** sudokuwiki Solving functions **********************/

    // Check for solved cells in cube structure (cells with only one candidate remaining) and update grid
    // Equivalent to a naked single.
    // Returns problem status if the whole puzzle is solved, incomplete, or broken. // TODO maybe some other true?
    fn update_solved_grid_cells(&mut self) -> SolvedStatus {
        let mut outcome = SolvedStatus::Complete;

        for index in 0..self.cube().len() {
            let cell = self.cube[index];
            if cell == 0b0 {
                //panic!("Shouldn't ever return broken"); //TODO decide what to do with a broken grid
                return SolvedStatus::Broken;
            }

            // This will be true when there is exactly one set bit in the variable
            if (cell & (cell - 1)) == 0 {
                let (x, y ) = math::xy_pair(index, self.order());
                let n = (cell as f32).log2() as usize + 1;
                self.set_grid_value(x, y, n);
            } else {
                outcome = SolvedStatus::Incomplete;
            }

        }
        outcome
    }

    // For each unknown cell we eliminate all candidates where the digit is known in the row or
    // column. This may reveal a single candidate, in which case we have a solution for that cell.
    // NOTE will not update the grid, only the cube (candidates) data structure
    //
    // Returns true if candidates were successfully removed
    fn update_candidates(&mut self) -> bool {
        let mut result = false;
        for i in 0..self.order() {
            for j in 0..self.order() {
                let n = self.get_grid_value(i, j);
                if n != 0 {
                    let digit_mask = 0b1 << n-1;
                    result |= self.cube_remove_candidate(i, digit_mask, true, &vec![j]) |
                        self.cube_remove_candidate(j, digit_mask, false, &vec![i]);
                }
            }
        }
        result
    }

    // If a candidate occurs once only in a row or column we can make it the solution to the cell.
    // Returns true if at least one hidden single is found.
    // This function can find multiple hidden singles at once if they are in different rows/columns.
    fn hidden_single(&mut self) -> bool {
        let mut result = false;
        for is_row in vec![true, false] {
            for axis_a in 0..self.order() {

                // A vector of just the row or column (axis) candidates
                let axis_vec = self.get_cube_vector(axis_a, is_row);

                // Pre-process before flipping to ensure that we ignore naked singles
                let mut pre_processed: Vec<i32> = vec![];
                for entry in axis_vec {
                    if entry & (entry - 1) == 0 {
                        pre_processed.push(0b0);
                    } else {
                        pre_processed.push(entry);
                    }
                }

                // Flip the vector because hidden single and naked single are inverses
                let flipped_vec = LatinSolver::flip_cube_structure(&pre_processed);

                result |= match LatinSolver::naked_single_finder(&flipped_vec) {
                    Some((digit, cell)) => {

                        let axis_b= (flipped_vec[digit] as f32).log2() as usize;
                        match is_row {
                            true => self.place_digit_xy(axis_a, axis_b, digit + 1),
                            false => self.place_digit_xy(axis_b, axis_a, digit + 1),
                        };
                        
                        true
                    },
                    None => false,
                }
            }
        }
        result
    }

    // We check for 'naked' pairs. For example, if we have two pairs, e.g. 3-4 and 3-4 in the same
    // row or column, then both 3 and 4 must occupy those squares (in what ever order). 3 and 4
    // can then be eliminated from the rest of the row and column.
    fn naked_pair(&mut self) -> bool {
        assert!(self.order() > 2); // TODO this fails for a 2x2 grid, requires order > 2

        let mut result = false;
        for is_row in vec![true, false] {
            for axis_a in 0..self.order() {

                // A vector of just the row or column (axis) candidates
                let axis_vec = self.get_cube_vector(axis_a, is_row);

                result |= match LatinSolver::naked_pair_finder(&axis_vec) {
                    Some((indices, available)) => {
                        for i in 0..self.order() {
                            if (0b1 << i) & available != 0 {
                                result |= self.cube_remove_candidate(axis_a,0b1 << i, is_row, &indices);
                            }
                        }
                        true
                    },
                    None => false,
                }
            }
        }
        result
    }

    // If two candidates occur only twice in a row or column we can make then a naked pair, and call
    // that function to eliminate candidates from the row/col.
    fn hidden_pair(&mut self) -> bool {
        let mut result = false;
        for is_row in vec![true, false] {
            for axis_a in 0..self.order() {
    
                // A vector of just the row or column (axis) candidates
                let axis_vec = self.get_cube_vector(axis_a, is_row);
    
                // Pre-process before flipping to ensure that we ignore naked singles
                let mut pre_processed: Vec<i32> = vec![];
                for entry in axis_vec {
                    if entry & (entry - 1) == 0 {
                        pre_processed.push(0b0);
                    } else {
                        pre_processed.push(entry);
                    }
                }
    
                // Flip the vector because hidden single and naked single are inverses
                let flipped_vec = LatinSolver::flip_cube_structure(&pre_processed);
    
                result |= match LatinSolver::naked_pair_finder(&flipped_vec) {
                    Some((digits, cell)) => {
                        let mut axis_b: Vec<usize> = vec![];
                        for d in digits {
                            axis_b.push((flipped_vec[d] as f32).log2() as usize);
                        }
    
                        //TODO
                        // for i in 0..self.order() {
                        //     if (0b1 << i) & available != 0 {
                        //         result |= self.cube_remove_candidate(axis_a, i + 1, is_row, &indices);
                        //     }
                        // }
                        true
                    },
                    None => false,
                }
            }
        }
        result
    }

    // We check for 'naked' triples and eliminate candidates seen by them
    fn naked_triple(&mut self) -> () {

    }

    // If three candidates occur only thrice in a row or column we can make then a naked triple,
    // and call that function to eliminate candidates from the row/col.
    fn hidden_triple(&mut self) -> () {

    }

    pub fn stepped_logical_solver(&mut self) -> SolvedStatus {
        loop {
            'simple_updates: loop {

                match self.update_solved_grid_cells() {
                    SolvedStatus::Complete => { return SolvedStatus::Complete; },
                    SolvedStatus::Incomplete => (),
                    SolvedStatus::Broken => { return SolvedStatus::Broken; }
                }

                match self.update_candidates() {
                    true => (),
                    false => break 'simple_updates,
                }
            }

            match self.hidden_single() {
                true => continue,
                false => (),
            }
            match self.naked_pair() {
                true => continue,
                false => {return SolvedStatus::Incomplete},
            }

            // missing hidden pair/set
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_place_digit() {
        let mut ls = LatinSolver::new(6);
        ls.place_digit_xy(0, 1, 2);
        assert_eq!(2, ls.get_grid_value(0, 1));
        assert_eq!(0, ls.get_cube_available(0,0) & 0b000010);
        // println!("{}", ls.grid_to_string());
        // println!("{}", ls.cube_to_string());
    }

    #[test]
    fn test_update_solved_grid_cells() {
        let mut ls = LatinSolver::new(6);
        ls.set_cube_available(4, 4, 0b001000);
        ls.set_cube_available(4, 5, 0b100000);
        ls.set_cube_available(4, 2, 0b111000);
        assert_eq!(SolvedStatus::Incomplete, ls.update_solved_grid_cells());
        assert_eq!(4, ls.get_grid_value(4, 4));
        assert_eq!(6, ls.get_grid_value(4, 5));
        assert_eq!(0, ls.get_grid_value(4, 2));

        // TODO remove print lines
        // println!("{}", ls.cube_to_string());
        // println!("{}", ls.grid_to_string());
    }

    #[test]
    fn test_update_candidates() {
        let mut ls = LatinSolver::new(6);
        ls.set_cube_available(0, 3, 0b001000);
        ls.set_cube_available(1, 3, 0b100000);
        ls.set_cube_available(2, 3, 0b111000);

        assert_eq!(SolvedStatus::Incomplete, ls.update_solved_grid_cells());
        assert_eq!(true, ls.update_candidates());
        assert_eq!(0b010000, *ls.get_cube_available(2, 3));
        assert_eq!(0b010111, *ls.get_cube_available(4, 3));
        ls.update_solved_grid_cells();
        assert_eq!(true, ls.update_candidates());
        assert_eq!(0b000111, *ls.get_cube_available(4, 3));
        ls.update_solved_grid_cells();
        assert_eq!(false, ls.update_candidates());
    }

    #[test]
    fn test_hidden_singles() {
        let mut ls = LatinSolver::new(6);
        ls.set_cube_available(1, 0, 0b111111);
        ls.set_cube_available(1, 1, 0b100100);
        ls.set_cube_available(1, 2, 0b100100);
        ls.set_cube_available(1, 3, 0b110111);
        ls.set_cube_available(1, 4, 0b110001);
        ls.set_cube_available(1, 5, 0b100001);
        assert!(ls.hidden_single());
        assert_eq!(4, ls.get_grid_value(1, 0));
        assert!(!ls.get_cube_value(3, 0, 4));
        assert!(ls.get_cube_value(1, 0, 4));
        assert!(ls.hidden_single());
        assert_eq!(2, ls.get_grid_value(1, 3));
        assert!(!ls.get_cube_value(3, 3, 2));
        assert!(ls.get_cube_value(1, 3, 2));
        assert!(ls.hidden_single());
        assert_eq!(5, ls.get_grid_value(1, 4));
        assert!(ls.hidden_single());
        assert_eq!(1, ls.get_grid_value(1, 5));
        assert!(!ls.hidden_single());

        // column case
        let mut ls = LatinSolver::new(6);
        ls.set_cube_available(0, 1, 0b111111);
        ls.set_cube_available(1, 1, 0b100100);
        ls.set_cube_available(2, 1, 0b100100);
        ls.set_cube_available(3, 1, 0b110111);
        ls.set_cube_available(4, 1, 0b110001);
        ls.set_cube_available(5, 1, 0b100001);
        assert!(ls.hidden_single());
        assert_eq!(4, ls.get_grid_value(0, 1));
        assert!(!ls.get_cube_value(0, 3, 4));
        assert!(ls.get_cube_value(0, 1, 4));
        assert!(ls.hidden_single());
        assert_eq!(2, ls.get_grid_value(3, 1));
        assert!(!ls.get_cube_value(3, 3, 2));
        assert!(ls.get_cube_value(3, 1, 2));
        assert!(ls.hidden_single());
        assert_eq!(5, ls.get_grid_value(4, 1));
        assert!(ls.hidden_single());
        assert_eq!(1, ls.get_grid_value(5, 1));
        assert!(!ls.hidden_single());
    }

    #[test]
    fn test_cube_remove_candidate() {
        let mut ls = LatinSolver::new(6);
        assert!(ls.cube_remove_candidate(3, 0b1 << 0, false, &vec![4, 5]));
        assert_eq!(0b111110, *ls.get_cube_available(0, 3));
        assert_eq!(0b111111, *ls.get_cube_available(5, 3));

        assert!(ls.cube_remove_candidate(3, 2, true, &vec![4, 5]));
        assert_eq!(0b111100, *ls.get_cube_available(3, 3));
        assert_eq!(0b111111, *ls.get_cube_available(3, 4));

        assert!(!ls.cube_remove_candidate(5, 6, true, &vec![0, 1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_naked_pair() {
        let mut ls = LatinSolver::new(6);
        ls.set_cube_available(1, 0, 0b111111);
        ls.set_cube_available(1, 1, 0b011000);
        ls.set_cube_available(1, 2, 0b010001);
        ls.set_cube_available(1, 3, 0b001100); //pair
        ls.set_cube_available(1, 4, 0b111111);
        ls.set_cube_available(1, 5, 0b001100); //pair
        assert!(ls.naked_pair());
        assert_eq!(0b110011, *ls.get_cube_available(1, 0));
        assert_eq!(0b010000, *ls.get_cube_available(1, 1));


    }

    #[test]
    fn test_naked_pair_incomplete() {
        let mut ls = LatinSolver::new(3);
        assert!(!ls.naked_pair());
        // assert_eq!(0b110011, *ls.get_cube_available(1, 0));
        // assert_eq!(0b010000, *ls.get_cube_available(1, 1));


    }

    #[test]
    fn test_full_solve() {
        let mut ls = LatinSolver::new(3);
        ls.set_cube_available(0, 0, 0b110);
        ls.set_cube_available(0, 1, 0b110);
        ls.set_cube_available(0, 2, 0b111);
        ls.set_cube_available(1, 0, 0b101);
        ls.set_cube_available(1, 1, 0b111);
        ls.set_cube_available(1, 2, 0b101);
        ls.set_cube_available(2, 0, 0b111);
        ls.set_cube_available(2, 1, 0b111);
        ls.set_cube_available(2, 2, 0b111);

        assert_eq!(SolvedStatus::Complete, ls.stepped_logical_solver());
        assert_eq!(3, ls.get_grid_value(2, 0));

        assert!(ls.check_solved());
        // println!("{}", ls.cube_to_string());
        // println!("{}", ls.grid_to_string());

    }


}

    // #[test]
    // fn test_playground() {
    //     let mut ls = LatinSolver::new(4);
    //     ls.set_cube_available(0, 0, 0b1101);
    //     ls.set_cube_available(0, 1, 0b1110);
    //     ls.set_cube_available(0, 2, 0b1101);
    //     ls.set_cube_available(0, 3, 0b0101);
    //
    //     LatinSolver::flip_cube_structure(&ls.cube()[0..ls.order()]);
    // }

