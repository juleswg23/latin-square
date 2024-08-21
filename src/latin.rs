#![allow(unused_imports)]
#![allow(dead_code)]

use itertools::Itertools;
use std::cmp::Ordering;
/// An object that contains solving data for Latin Square puzzles
pub struct LatinSolver {
    order: usize,       // the dimension of the square KenKen grid
    cube: Vec<bool>,    // order^3

    // might be useful to have grid appear elsewhere as its own type
    grid: Vec<usize>,   // order^2

    // Currently unused
    //row: Vec<bool>,     // order^2
    //col: Vec<bool>,     // order^2
}

impl LatinSolver {

    /**************************** Initializers ****************************/

    pub fn new(order: usize) -> LatinSolver {
        LatinSolver {
            order,
            cube: vec![true; order.pow(3)], // false when value is not a possibility in that square
            grid: vec![0; order.pow(2)],    // The completed grid of values 1 through order, 0s before solved
            
            // Currently unused
            //row: vec![false; order.pow(2)], // false when the val is not yet present in row x
            //col: vec![false; order.pow(2)], // false when the val is not yet present in col y
        }
    }

    /**************************** Cube functions ****************************/

    // Get the index in the cube vector of the boolean for value n at (x, y)
    fn get_cube_loc(&self, x: usize, y: usize, n: usize) -> usize {
        let location = (x * self.order + y) * self.order; // try removing &
        location + n - 1
    }

    // True means the value (n) is still possible at that coordinate (x,y)
    fn get_cube_value(&self, x: usize, y: usize, n: usize) -> bool {
        let location = self.get_cube_loc(x, y, n);
        self.cube[location]
    }

    // Update the cube data structure to be true or false at (x,y) to bool b
    fn set_cube_value(&mut self, x: usize, y: usize, n: usize, b: bool) -> () {
        let location = self.get_cube_loc(x, y, n);
        self.cube[location] = b;
    }

    fn get_cube_loc_subarray(&self, x: usize, y:usize) -> Vec<bool> {
        let location = (x * self.order + y) * self.order;
        let result = &self.cube[location..location+self.order];
        result.to_vec() // try without the to_vec()
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

    // Get the location in the grid data structure at coordinates (x,y)
    fn get_grid_loc(&self, x: usize, y: usize) -> usize {
        x * self.order + y
    }

    // Get the value at the grid
    fn get_grid_value(&self, x: usize, y: usize) -> usize {
        self.grid[self.get_grid_loc(x, y)]
    }

    // Set the final value in the grid of where the digit belongs
    fn set_grid_value(&mut self, x: usize, y: usize, n: usize) -> () {
        // First assert attempt
        assert!(x < self.order && y < self.order && n <= self.order, "All quantities must be within the grid dimensions");

        let location = self.get_grid_loc(x, y);
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
    fn place_digit(&mut self, x: usize, y:usize, n: usize) -> (Vec<usize>, Vec<bool>) {
        let old_data:(Vec<usize>, Vec<bool>) = (self.grid.clone(), self.cube.clone());

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

        //todo update rows and cols data structures

        old_data

    }

    // Returns the location and vector of available digits at that location from the cube structure
    // Or, if no location is found with multiple possibilities, returns none
    // CAREFUL - it updates the cube and grid structures if they are out of sync.
    // TODO later change this to read-only on the self param
    fn find_unsolved_location(&mut self) -> Option<(usize, usize, Vec<usize>)> {
        for i in 0..self.order {
            for j in 0..self.order {
                // Our subarray of the cube array at location i,j
                let cube_subarray: Vec<bool> = self.get_cube_loc_subarray(i, j);

                // Digits from 1 through order that are available at that location
                let mut available_digits: Vec<usize> = Vec::new();

                for n in 0..self.order {
                    if cube_subarray[n] {
                        available_digits.push(n+1);
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
}
