#![allow(unused_imports)]
use itertools::Itertools;
use std::env;

#[allow(dead_code)]
struct LatinSolver {
    order: usize,
    cube: Vec<bool>, // order^3

    // might be useful to have grid appear elsewhere as it's own type
    // digit is a number from 1 to order (not just at 0 and 1)
    grid: Vec<usize>, // order^2
    row: Vec<bool>, // order^2
    col: Vec<bool>, // order^2
}

#[allow(dead_code)]
impl LatinSolver {

    /**************************** Initializers ****************************/

    fn new(order: usize) -> LatinSolver {
        LatinSolver {
            order: order,
            cube: vec![true; order.pow(3)], // false when value is not a possibility in that square
            grid: vec![0; order.pow(2)],
            row: vec![false; order.pow(2)], // set to true when the val is present in row x
            col: vec![false; order.pow(2)], // set to true when the val is present in col y
        }
    }

    /**************************** Cube functions ****************************/

    // Get the index in the cube vector of the boolean for value n at (x, y)
    fn get_cube_pos(&self, x: usize, y: usize, n: usize) -> usize {
        let location = (x * self.order + y) * self.order; // try removing &
        location + n - 1
    }

    // True means the value (n) is still possible at that coordinate (x,y)
    fn get_cube_value(&self, x: usize, y: usize, n: usize) -> bool {
        let position = self.get_cube_pos(x, y, n);
        self.cube[position]
    }

    // Update the cube data structure to be true or false at (x,y) to bool b
    fn set_cube_value(&mut self, x: usize, y: usize, n: usize, b: bool) -> () {
        let position = self.get_cube_pos(x, y, n);
        self.cube[position] = b;
    }
    
    // To string method for the cube data structure
    fn cube_to_string(&self) -> String {
        let mut result = String::from("");

        for i in 0..self.order {
            let mut row_arr: Vec<String> = Vec::new();
            for j in 0..self.order {
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

    // Get the position in the grid data structure at coordinates (x,y)
    fn get_grid_pos(&self, x: usize, y: usize) -> usize {
        x * self.order + y
    }

    // Get the value at the grid
    fn get_grid_value(&self, x: usize, y: usize) -> usize {
        self.grid[self.get_grid_pos(x, y)]
    }

    // Set the final value in the grid of where the digit belongs
    fn set_grid_value(&mut self, x: usize, y: usize, n: usize) -> () {
        // First assert attempt
        assert!(x < self.order && y < self.order && n <= self.order, "All quantities must be within the grid dimensions");

        let location = self.get_grid_pos(x, y);
        self.grid[location] = n;
    }

    // To string method for the cube data structure
    fn grid_to_string(&self) -> String {
        let mut result = String::from("");

        for i in 0..self.order {
            let mut arr: Vec<usize> = Vec::new();
            for j in 0..self.order {
                arr.push(self.get_grid_value(i, j));
            }

            let row = " ".to_string() + &arr.iter().join(" | ") + &" ";
            let gap = if i < self.order - 1 {
                "\n".to_string() + &"_".repeat((self.order) * 4 - 1) + "\n"
            } else {
                 "\n".to_string()
            };
            result = result + &row + &gap;
        }
        result
    }




    // Place a digit in the final grid,
    // and update the data strutures storing the availibility of digits
    fn place_digit(&mut self, x: usize, y:usize, n: usize) -> () {
        
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
        
        // update all other n's at that position
        for i in 1..=self.order {
            if i != n { 
                self.set_cube_value(x, y, i, false);
            }
        }

        //todo update rows and cols data structures

    }

}

fn main() {
    //env::set_var("RUST_BACKTRACE", "1");
    let mut ls = LatinSolver::new(6);
    println!("{}", ls.cube.len());
    println!("{}", ls.get_cube_pos(3, 3, 2));
    println!("{}", ls.get_cube_value(3, 3, 2));
    println!("{}", ls.get_grid_value(3, 3));

    ls.place_digit(3, 3, 3);
    ls.place_digit(1, 3, 2);

    println!("{}", ls.get_cube_value(3, 3, 2));
    println!("{}", ls.get_cube_value(3, 3, 3));

    println!("{}", ls.cube_to_string());
    println!("{}", ls.grid_to_string())
}
