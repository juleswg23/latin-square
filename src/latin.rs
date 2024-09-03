#![allow(unused_imports)]
#![allow(dead_code)]

use itertools::Itertools;
use std::cmp::Ordering;
use crate::math::math;

/// Enum for the solver to track the status of completion
#[derive(Debug, PartialEq)]
enum SolvedStatus {
    Complete,
    Incomplete,
    Broken,
}

/// An object that contains solving data for Latin Square puzzles
pub struct LatinSolver {
    order: usize,     // the dimension of the square KenKen grid
    cube: Vec<i32>,   // order^3 // TODO maybe refactor cube to be a 2d array of binary ints.

    // might be useful to have grid appear elsewhere as its own type
    grid: Vec<usize>, // order^2

    // row[x] is represented by the indices order*x..order*(x+1)
    // so 3 is in row x if row[x*order + 3] is true
    row: Vec<bool>,   // order^2
    col: Vec<bool>,   // order^2
}

impl LatinSolver {
    /**************************** Initializers ****************************/

    pub fn new(order: usize) -> LatinSolver {
        LatinSolver {
            order,
            cube: vec![(0b1 << order) - 1; order.pow(2)], // 0 at bit k when k is not possible in that cell
            grid: vec![0; order.pow(2)], // The completed grid of values 1 through order, 0s before solved

            row: vec![false; order.pow(2)], // false when the val is not yet present in row x
            col: vec![false; order.pow(2)], // false when the val is not yet present in col y
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

    pub fn row(&self) -> &Vec<bool> {
        &self.row
    }

    pub fn col(&self) -> &Vec<bool> {
        &self.col
    }

    // Get the location at coordinates (x,y)
    fn get_loc(&self, x: usize, y: usize) -> usize {
        x * self.order + y
    }


    /**************************** Cube functions ****************************/

    // True means the value (n) is still possible at that coordinate (x,y)
    fn get_cube_value(&self, x: usize, y: usize, n: usize) -> bool {
        let location = self.get_loc(x, y);
        (0b1 << (n-1)) & self.cube[location] != 0
    }

    // Update the cube data structure to be true or false at (x,y) to bool b
    fn set_cube_value(&mut self, x: usize, y: usize, n: usize, b: bool) -> () {
        let location = self.get_loc(x, y);
        match b {
            true => {
                self.cube[location] |= 0b1 << (n-1);
            },
            false => {
                let all_ones_mask = (0b1 << self.order) - 1;
                self.cube[location] &= (0b1 << (n-1)) ^ all_ones_mask;
            },
        }
    }

    pub fn get_cube_available(&self, x: usize, y: usize) -> i32 {
        self.cube[self.get_loc(x, y)]
    }

    // Update a subarray at a particular position with pruned (or expanded) choices of available digits
    pub fn set_cube_available(&mut self, x: usize, y: usize, available: i32) -> () {
        assert!(available < 0b1 << self.order && available >= 0);
        let location = self.get_loc(x, y);
        self.cube[location] = available;
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
    pub fn place_digit(&mut self, x: usize, y: usize, n: usize) -> (Vec<usize>, Vec<i32>) {
        let old_data: (Vec<usize>, Vec<i32>) = (self.grid.clone(), self.cube.clone());

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

        // TODO update rows and cols data structures

        old_data
    }

    // Returns the location and vector of available digits at that location from the cube structure
    // Or, if no location is found with multiple possibilities, returns none
    // CAREFUL - it updates the cube and grid structures if they are out of sync.
    // TODO later change this to read-only on the self param
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
                        self.place_digit(i, j, available_digits[0]);
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
                let (prev_grid, prev_cube) = self.place_digit(x, y, n);
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

    // Check for solved grid cells (cells with only one candidate remaining) and update grid
    // Returns problem status if the whole puzzle is solved, incomplete, or broken. // TODO maybe some other true?
    fn update_solved_grid_cells(&mut self) -> SolvedStatus {
        let mut outcome = SolvedStatus::Complete;

        for index in 0..self.cube().len() {
            let cell = self.cube[index];
            if cell == 0b0 {
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
    fn impossible_candidates(&mut self) -> () {
        for row in 0..self.order() {
            
            for col in 0..self.order() {
                
            }
        }
        for index in 0..self.grid().len() {
            if self.grid[index] != 0 {
                
            }
        }
    }

    // If a candidate occurs once only in a row or column we can make it the solution to the cell.
    fn hidden_single(&mut self) -> () {

    }

    // We check for 'naked' pairs. For example, if we have two pairs, e.g. 3-4 and 3-4 in the same
    // row or column, then both 3 and 4 must occupy those squares (in what ever order). 3 and 4
    // can then be eliminated from the rest of the row and column.
    fn naked_pair(&mut self) -> () {

    }

    // If two candidates occur only twice in a row or column we can make then a naked pair, and call
    // that function to eliminate candidates from the row/col.
    fn hidden_pair(&mut self) -> () {

    }

    // We check for 'naked' triples and eliminate candidates seen by them
    fn naked_triple(&mut self) -> () {

    }

    // If three candidates occur only thrice in a row or column we can make then a naked triple,
    // and call that function to eliminate candidates from the row/col.
    fn hidden_triple(&mut self) -> () {

    }

    pub fn stepped_logical_solver(&mut self) {
        let mut old_cube = 0;
        let mut current_cube = 1; // TODO choose how to represent cubes
        while old_cube != current_cube {
            // TODO call different logical functions to update
        }
    }
    
}


#[cfg(test)]
mod tests {
    use super::*;

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

        // TODO remove printlns
        println!("{}", ls.cube_to_string());
        println!("{}", ls.grid_to_string());
    }

    fn test_impossible_candidates() {
        let mut ls = LatinSolver::new(6);
        ls.set_cube_available(4, 4, 0b001000);
        ls.set_cube_available(4, 2, 0b111000);
        assert_eq!(SolvedStatus::Incomplete, ls.update_solved_grid_cells());
    }
}