// #![allow(dead_code)]
use crate::math::math;
use itertools::Itertools;

/// An object that contains solving data for Latin Square puzzles

pub struct Grid {
    order: usize,         // the dimension of the square KenKen grid
    candidates: Vec<i32>, // order^2 - a 2d array of binary ints. 0b111111

    // might be useful to have grid appear elsewhere as its own type
    digits: Vec<usize>, // order^2
}

impl Grid {
    pub fn new(order: usize) -> Grid {
        Grid {
            order,
            candidates: vec![(0b1 << order) - 1; order.pow(2)], // 0 at bit k when k is not possible in that cell
            digits: vec![0; order.pow(2)], // The completed grid of values 1 through order, 0s before solved
        }
    }

    pub fn order(&self) -> usize {
        self.order
    }

    pub fn candidates(&self) -> &Vec<i32> {
        &self.candidates
    }

    pub fn digits(&self) -> &Vec<usize> {
        &self.digits
    }

    // Get the location at coordinates (x,y)
    fn get_loc(&self, x: usize, y: usize) -> usize {
        x * self.order + y
    }

    /**************************** Cube functions ****************************/

    // TODO eventually delete these first two functions.
    // True means the value (n) is still possible at that coordinate (x,y)
    pub fn get_candidates_value(&self, x: usize, y: usize, n: usize) -> bool {
        let location = self.get_loc(x, y);
        (0b1 << (n - 1)) & self.candidates[location] != 0
    }

    // Update the cube data structure to be true or false at (x,y) to bool b
    fn set_candidates_value(&mut self, x: usize, y: usize, n: usize, b: bool) -> () {
        let location = self.get_loc(x, y);
        match b {
            true => {
                // turn on the bit
                self.candidates[location] |= 0b1 << (n - 1);
            }
            false => {
                // turn off the bit
                self.candidates[location] &= !(0b1 << (n - 1));
            }
        }
    }

    // Get the cube data structure representing available digits at (x, y)
    pub fn get_candidates_available(&self, x: usize, y: usize) -> &i32 {
        &self.candidates[self.get_loc(x, y)]
    }

    // Update a subarray at a particular position with pruned (or expanded) choices of available digits
    pub fn set_candidates_available(&mut self, x: usize, y: usize, available: i32) -> () {
        assert!(available < 0b1 << self.order && available >= 0);
        let location = self.get_loc(x, y);
        self.candidates[location] = available;
    }

    // Returns a vector of a row or column from the cube (a copy)
    pub fn get_candidates_vector(&self, index: usize, is_row: bool) -> Vec<i32> {
        match is_row {
            // Take a row slice
            true => self.candidates()[index * self.order()..(index + 1) * self.order()].to_vec(),
            // Take every nth element, since we are in a flattened 2d vector
            false => self
                .candidates()
                .iter()
                .skip(index)
                .step_by(self.order())
                .copied()
                .collect(),
        }
    }

    // To_String method for the cube data structure
    pub fn candidates_to_string(&self) -> String {
        let mut result = String::from("");

        for i in 0..self.order {
            // Make an array of all the contents of each cell in the row
            let mut row_arr: Vec<String> = Vec::new();
            for j in 0..self.order {
                // Make an array for the contents of each cell
                // It will have a digit if the digit is still available, otherwise a '*'
                let mut cell_arr: Vec<char> = Vec::new();
                for n in 1..=self.order {
                    if self.get_candidates_value(i, j, n) {
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

    /**************************** Digits functions ****************************/

    // Get the digit
    pub fn get_digits_value(&self, x: usize, y: usize) -> usize {
        self.digits[self.get_loc(x, y)]
    }

    // Set the final digit in the grid of where the digit belongs
    pub fn set_digits_value(&mut self, x: usize, y: usize, n: usize) -> () {
        // First assert attempt
        assert!(
            x < self.order && y < self.order && n <= self.order,
            "All quantities must be within the grid dimensions"
        );
        let location = self.get_loc(x, y);
        self.digits[location] = n;
    }

    // Set digits value to 0 at location (x, y)
    fn reset_digits_value(&mut self, x: usize, y: usize) -> () {
        self.set_digits_value(x, y, 0);
    }

    // To string method for the cube data structure
    pub fn digits_to_string(&self) -> String {
        let mut result = String::from("");

        for i in 0..self.order {
            let mut arr: Vec<usize> = Vec::new();
            for j in 0..self.order {
                arr.push(self.get_digits_value(i, j));
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

    // Place a digit in the final digits grid,
    // and update the data structures storing the availability of digits
    pub fn place_digit_xy(&mut self, x: usize, y: usize, n: usize) -> () {
        // place it in the digits grid structure
        self.set_digits_value(x, y, n);

        // update the cube structure along the row
        for i in 0..self.order {
            if i != x {
                self.set_candidates_value(i, y, n, false)
            }
        }

        // update the cube structure along the column
        for i in 0..self.order {
            if i != y {
                self.set_candidates_value(x, i, n, false)
            }
        }

        // update all other n's at that location
        for i in 1..=self.order {
            if i != n {
                self.set_candidates_value(x, y, i, false);
            }
        }
        //old_data
    }

    // TODO maybe remove
    fn place_digit_xy_deepcopy(
        &mut self,
        x: usize,
        y: usize,
        n: usize,
    ) -> (Vec<usize>, Vec<i32>) {
        let old_data: (Vec<usize>, Vec<i32>) = (self.digits.clone(), self.candidates.clone());
        self.place_digit_xy(x, y, n);
        old_data
    }

    // TODO maybe remove
    fn place_digit_flat(&mut self, flat_index: usize, n: usize) -> () {
        let (x, y) = math::xy_pair(flat_index, self.order());
        self.place_digit_xy(x, y, n)
    }

    // Given a row, remove the candidate from all cells in cube except the locations provided in do_not_remove
    // TODO MAYBE try not to call if row has no changes, cuz this looks sorta expensive
    // Returns true if candidates were removed
    pub fn remove_candidate_axis(
        &mut self,
        axis_a: usize,
        digit_mask: i32,
        is_row: bool,
        do_not_remove: &Vec<usize>,
    ) -> bool {
        assert_ne!(
            &Vec::<usize>::new(),
            do_not_remove,
            "cannot remove candidate from every position"
        );
        assert_ne!(0b0, digit_mask, "Cannot remove no digit");
        assert_ne!(0b0, !digit_mask, "Cannot remove all digits");
        let mut result = false;

        // digit mask is 1 at position n-1 for the digit n's we want to remove.

        for axis_b in 0..self.order() {
            if !do_not_remove.contains(&axis_b) {
                // skip unless not in do not remove
                let (row, col) = match is_row {
                    true => (axis_a, axis_b),
                    false => (axis_b, axis_a),
                };
                let old_mask = *self.get_candidates_available(row, col);
                let new_mask = old_mask & !digit_mask;
                self.set_candidates_available(row, col, new_mask);

                // updates result to true if old_mask is different from new_mask
                result |= old_mask != new_mask;
            }
        }
        result
    }
}

mod tests {
    use super::*;

    #[test]
    fn place_digit() {
        let mut ls = Grid::new(6);
        ls.place_digit_xy(0, 1, 2);
        assert_eq!(2, ls.get_digits_value(0, 1));
        assert_eq!(0, ls.get_candidates_available(0, 0) & 0b000010);
    }

    #[test]
    fn remove_candidate_axis() {
        let mut ls = Grid::new(6);
        assert!(ls.remove_candidate_axis(3, 0b1 << 0, false, &vec![4, 5]));
        assert_eq!(0b111110, *ls.get_candidates_available(0, 3));
        assert_eq!(0b111111, *ls.get_candidates_available(5, 3));

        assert!(ls.remove_candidate_axis(3, 2, true, &vec![4, 5]));
        assert_eq!(0b111100, *ls.get_candidates_available(3, 3));
        assert_eq!(0b111111, *ls.get_candidates_available(3, 4));

        assert!(!ls.remove_candidate_axis(5, 6, true, &vec![0, 1, 2, 3, 4, 5]));
    }

    #[test]
    fn get_candidates_vector() {
        assert!(true);
        //todo!();
    }
}
