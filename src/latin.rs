#![allow(dead_code)]

use itertools::Itertools;

/// Enum for the solver to track the status of completion
#[derive(Debug, PartialEq)]
pub enum SolvedStatus {
    Complete,
    Incomplete,
    Broken,
}

pub(crate) mod latin_solve {
    use std::collections::HashMap;
    use crate::grid::Grid;
    use crate::latin::SolvedStatus;
    use crate::math::math;
    /******************************** Static functions ********************************/

    // Returns a vector where the mask at vec[i] are the locations that the digit i-1 is available
    fn flip_candidates_structure(set: &[i32]) -> Vec<i32> {
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
    
    /********************** sudokuwiki Solving functions **********************/
    
    // Check for solved cells in cube structure (cells with only one candidate remaining) and update grid
    // Equivalent to a naked single.
    // Returns problem status if the whole puzzle is solved, incomplete, or broken. // TODO maybe some other true?
    fn update_solved_grid_cells(grid: &mut Grid) -> SolvedStatus {
        let mut outcome = SolvedStatus::Complete;
    
        for index in 0..grid.candidates().len() {
            let cell = grid.candidates()[index];
            if cell == 0b0 {
                //panic!("Shouldn't ever return broken"); //TODO decide what to do with a broken grid
                return SolvedStatus::Broken;
            }
    
            // This will be true when there is exactly one set bit in the variable
            if (cell & (cell - 1)) == 0 {
                let (x, y) = math::xy_pair(index, grid.order());
                let n = (cell as f32).log2() as usize + 1;
                grid.set_digits_value(x, y, n);
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
    fn update_candidates(grid: &mut Grid) -> bool {
        let mut result = false;
        for i in 0..grid.order() {
            for j in 0..grid.order() {
                let n = grid.get_digits_value(i, j);
                if n != 0 {
                    let digit_mask = 0b1 << n-1;
                    result |= grid.remove_candidate_axis(i, digit_mask, true, &vec![j]) |
                        grid.remove_candidate_axis(j, digit_mask, false, &vec![i]);
                }
            }
        }
        result
    }
    
    // If a candidate occurs once only in a row or column we can make it the solution to the cell.
    // Returns true if at least one hidden single is found.
    // This function can find multiple hidden singles at once if they are in different rows/columns.
    fn hidden_single(grid: &mut Grid) -> bool {
        let mut result = false;
        for is_row in vec![true, false] {
            for axis_a in 0..grid.order() {
    
                // A vector of just the row or column (axis) candidates
                let axis_vec = grid.get_candidates_vector(axis_a, is_row);
    
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
                let flipped_vec = flip_candidates_structure(&pre_processed);
    
                result |= match naked_single_finder(&flipped_vec) {
                    Some((digit, cell)) => {
    
                        let axis_b= (flipped_vec[digit] as f32).log2() as usize;
                        match is_row {
                            true => grid.place_digit_xy(axis_a, axis_b, digit + 1),
                            false => grid.place_digit_xy(axis_b, axis_a, digit + 1),
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
    fn naked_pair(grid: &mut Grid) -> bool {
        assert!(grid.order() > 2); // TODO this fails for a 2x2 grid, requires order > 2
    
        let mut result = false;
        for is_row in vec![true, false] {
            for axis_a in 0..grid.order() {
    
                // A vector of just the row or column (axis) candidates
                let axis_vec = grid.get_candidates_vector(axis_a, is_row);
    
                result |= match naked_pair_finder(&axis_vec) {
                    Some((indices, available)) => {
                        for i in 0..grid.order() {
                            if (0b1 << i) & available != 0 {
                                result |= grid.remove_candidate_axis(axis_a,0b1 << i, is_row, &indices);
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
    fn hidden_pair(grid: &mut Grid) -> bool {
        let mut result = false;
        for is_row in vec![true, false] {
            for axis_a in 0..grid.order() {
    
                // A vector of just the row or column (axis) candidates
                let axis_vec = grid.get_candidates_vector(axis_a, is_row);
    
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
                let flipped_vec = flip_candidates_structure(&pre_processed);
    
                result |= match naked_pair_finder(&flipped_vec) {
                    Some((digits, cell)) => {
                        let mut axis_b: Vec<usize> = vec![];
                        for d in digits {
                            axis_b.push((flipped_vec[d] as f32).log2() as usize);
                        }
    
                        //TODO
                        // for i in 0..grid.order() {
                        //     if (0b1 << i) & available != 0 {
                        //         result |= grid.cube_remove_candidate(axis_a, i + 1, is_row, &indices);
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
    fn naked_triple(grid: &mut Grid) -> bool {
        todo!()
    }
    
    // If three candidates occur only thrice in a row or column we can make then a naked triple,
    // and call that function to eliminate candidates from the row/col.
    fn hidden_triple(grid: &mut Grid) -> () {
    
    }
    
    pub fn stepped_logical_solver(grid: &mut Grid) -> SolvedStatus {
        loop {
            'simple_updates: loop {
    
                match update_solved_grid_cells(grid) {
                    SolvedStatus::Complete => { return SolvedStatus::Complete; },
                    SolvedStatus::Incomplete => (),
                    SolvedStatus::Broken => { return SolvedStatus::Broken; }
                }
    
                match update_candidates(grid) {
                    true => (),
                    false => break 'simple_updates,
                }
            }
    
            match hidden_single(grid) {
                true => continue,
                false => (),
            }
            match naked_pair(grid) {
                true => continue,
                false => {return SolvedStatus::Incomplete},
            }
    
            // missing hidden pair/set
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::grid::Grid;
        use super::*;

        #[test]
        fn test_update_solved_grid_cells() {
            let mut g = Grid::new(6);
            g.set_candidates_available(4, 4, 0b001000);
            g.set_candidates_available(4, 5, 0b100000);
            g.set_candidates_available(4, 2, 0b111000);
            assert_eq!(SolvedStatus::Incomplete, update_solved_grid_cells(&mut g));
            assert_eq!(4, g.get_digits_value(4, 4));
            assert_eq!(6, g.get_digits_value(4, 5));
            assert_eq!(0, g.get_digits_value(4, 2));

            // TODO remove print lines
            // println!("{}", g.cube_to_string());
            // println!("{}", g.grid_to_string());
        }

        #[test]
        fn test_update_candidates() {
            let mut g = Grid::new(6);
            g.set_candidates_available(0, 3, 0b001000);
            g.set_candidates_available(1, 3, 0b100000);
            g.set_candidates_available(2, 3, 0b111000);

            assert_eq!(SolvedStatus::Incomplete, update_solved_grid_cells(&mut g));
            assert_eq!(true, update_candidates(&mut g));
            assert_eq!(0b010000, *g.get_candidates_available(2, 3));
            assert_eq!(0b010111, *g.get_candidates_available(4, 3));
            update_solved_grid_cells(&mut g);
            assert_eq!(true, update_candidates(&mut g));
            assert_eq!(0b000111, *g.get_candidates_available(4, 3));
            update_solved_grid_cells(&mut g);
            assert_eq!(false, update_candidates(&mut g));
        }

        #[test]
        fn test_hidden_singles() {
            let mut g = Grid::new(6);
            g.set_candidates_available(1, 0, 0b111111);
            g.set_candidates_available(1, 1, 0b100100);
            g.set_candidates_available(1, 2, 0b100100);
            g.set_candidates_available(1, 3, 0b110111);
            g.set_candidates_available(1, 4, 0b110001);
            g.set_candidates_available(1, 5, 0b100001);
            assert!(hidden_single(&mut g));
            assert_eq!(4, g.get_digits_value(1, 0));
            assert!(!g.get_candidates_value(3, 0, 4));
            assert!(g.get_candidates_value(1, 0, 4));
            assert!(hidden_single(&mut g));
            assert_eq!(2, g.get_digits_value(1, 3));
            assert!(!g.get_candidates_value(3, 3, 2));
            assert!(g.get_candidates_value(1, 3, 2));
            assert!(hidden_single(&mut g));
            assert_eq!(5, g.get_digits_value(1, 4));
            assert!(hidden_single(&mut g));
            assert_eq!(1, g.get_digits_value(1, 5));
            assert!(!hidden_single(&mut g));

            // column case
            let mut g = Grid::new(6);
            g.set_candidates_available(0, 1, 0b111111);
            g.set_candidates_available(1, 1, 0b100100);
            g.set_candidates_available(2, 1, 0b100100);
            g.set_candidates_available(3, 1, 0b110111);
            g.set_candidates_available(4, 1, 0b110001);
            g.set_candidates_available(5, 1, 0b100001);
            assert!(hidden_single(&mut g));
            assert_eq!(4, g.get_digits_value(0, 1));
            assert!(!g.get_candidates_value(0, 3, 4));
            assert!(g.get_candidates_value(0, 1, 4));
            assert!(hidden_single(&mut g));
            assert_eq!(2, g.get_digits_value(3, 1));
            assert!(!g.get_candidates_value(3, 3, 2));
            assert!(g.get_candidates_value(3, 1, 2));
            assert!(hidden_single(&mut g));
            assert_eq!(5, g.get_digits_value(4, 1));
            assert!(hidden_single(&mut g));
            assert_eq!(1, g.get_digits_value(5, 1));
            assert!(!hidden_single(&mut g));
        }

        #[test]
        fn test_naked_pair() {
            let mut g = Grid::new(6);
            g.set_candidates_available(1, 0, 0b111111);
            g.set_candidates_available(1, 1, 0b011000);
            g.set_candidates_available(1, 2, 0b010001);
            g.set_candidates_available(1, 3, 0b001100); //pair
            g.set_candidates_available(1, 4, 0b111111);
            g.set_candidates_available(1, 5, 0b001100); //pair
            assert!(naked_pair(&mut g));
            assert_eq!(0b110011, *g.get_candidates_available(1, 0));
            assert_eq!(0b010000, *g.get_candidates_available(1, 1));

            let mut g = Grid::new(6);
            g.set_candidates_available(0, 3, 0b111111);
            g.set_candidates_available(1, 3, 0b011000);
            g.set_candidates_available(2, 3, 0b010001);
            g.set_candidates_available(3, 3, 0b001100); //pair
            g.set_candidates_available(4, 3, 0b111111);
            g.set_candidates_available(5, 3, 0b001100); //pair
            naked_pair(&mut g);
            assert_eq!(0b110011, *g.get_candidates_available(0, 3));
            assert_eq!(0b010000, *g.get_candidates_available(1, 3));
        }

        #[test]
        fn test_naked_pair_incomplete() {
            let mut g = Grid::new(3);
            assert!(!naked_pair(&mut g));
            // assert_eq!(0b110011, *g.get_candidates_available(1, 0));
            // assert_eq!(0b010000, *g.get_candidates_available(1, 1));

        }

        #[test]
        fn test_full_solve() {
            let mut g = Grid::new(3);
            g.set_candidates_available(0, 0, 0b110);
            g.set_candidates_available(0, 1, 0b110);
            g.set_candidates_available(0, 2, 0b111);
            g.set_candidates_available(1, 0, 0b101);
            g.set_candidates_available(1, 1, 0b111);
            g.set_candidates_available(1, 2, 0b101);
            g.set_candidates_available(2, 0, 0b111);
            g.set_candidates_available(2, 1, 0b111);
            g.set_candidates_available(2, 2, 0b111);

            assert_eq!(SolvedStatus::Complete, stepped_logical_solver(&mut g));
            assert_eq!(3, g.get_digits_value(2, 0));

            //assert!(check_solved(&mut g)); TODO
            // println!("{}", g.cube_to_string());
            // println!("{}", g.grid_to_string());

        }
    }
}
