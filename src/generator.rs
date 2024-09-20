use crate::grid::Grid;
use crate::kenken::{Clue, KenKen, Operation, Region, kenken_solve};
use crate::latin::{SolvedStatus, latin_solve};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::collections::{HashMap, VecDeque};
use std::ops::Deref;

fn insert_random_region(ken_ken: &mut KenKen) -> Option<()> {
    let mut empty_cells = get_empty_cells(ken_ken.deref());
    empty_cells.shuffle(&mut thread_rng()); // TODO Problem is they are not contiguous
    let region_size = weighted_random(empty_cells.len());

    // randomly choose operation
    let ops_vec = match region_size {
        1 => vec![Operation::Given],
        2 => vec![Operation::Add, Operation::Subtract, Operation::Divide, Operation::Multiply],
        _ => vec![Operation::Add, Operation::Multiply],
    };

    let new_op = ops_vec.choose(&mut thread_rng())?.clone();
    let new_target = match new_op { // different ranges depending on the op
        Operation::Add => thread_rng().gen_range(1..region_size *ken_ken.order()),
        Operation::Subtract => thread_rng().gen_range(1..ken_ken.order()),
        Operation::Multiply => thread_rng().gen_range(1..=ken_ken.order().pow(region_size as u32)),
        //Operation::Multiply => ken_ken.order().pow(region_size as u32), // TODO remove
        Operation::Divide => thread_rng().gen_range(2..=ken_ken.order()),
        Operation::Given => thread_rng().gen_range(1..=ken_ken.order()),
        _ => panic!("Should never pick these operations")
    };

    let new_region = Region::new(Clue::new(new_op, new_target), empty_cells[0..region_size].to_vec());
    ken_ken.add_region(new_region);

    Some(())
}

// Check if two cells are adjacent
fn is_adjacent(order: usize, first: &usize, second: &usize) -> bool {
    assert!(order > 1, "A grid must be at least 2x2");

    let first_x = first / order;
    let first_y = first % order;
    let second_x = second / order;
    let second_y = second % order;

    assert!(first_x < order, "first_x value out of bounds");
    assert!(second_x < order, "second_x value out of bounds");

    (first_x.abs_diff(second_x) + first_y.abs_diff(second_y)) == 1
}

/// Return a vector of all the cells not attributed to a region in a kenken
fn get_empty_cells(ken_ken: &KenKen) -> Vec<usize> {
    let vec: Vec<usize> = (0..ken_ken.order() * ken_ken.order()).collect();
    let mut result: HashMap<usize, bool> = vec.into_iter().map(|x| (x, true)).collect::<HashMap<_, _>>();
    for elem in ken_ken.regions() {
        for cell in elem.cells() {
            result.remove(cell);
        }
    }

    // Have the empty cells
    let empty_cells: Vec<usize> = result.keys().copied().collect();
    empty_cells
}

// Right now this isn't actually weighted
fn weighted_random(n: usize) -> usize {
    let result: usize = thread_rng().gen_range(1..=(n) as u32) as usize;

    // match random_number {
    //     1..n/2 => {}
    // } 

    assert!(result <= n && result > 0);
    result
}

fn create_puzzle_clues(order: usize) -> KenKen {
    // Options

    // 1- Grid first
    // Populate the grid with digits
    // Then add clues that are accurate to the digits that are present
    // Check if unique solution

    // 2- Clue first
    // Add a clue
    // Run solve
    //      If Complete, then done
    //      If Broken, delete last clue, go back to top
    //      If Incomplete, go to top

    // 3- a hybrid
    // Some hybrid, where we have some restrictions already given in grid?

    // grid.place_digit_xy(0, 0, 1);
    // grid.place_digit_xy(0, 1, 2);
    // grid.place_digit_xy(0, 2, 3);
    //
    // grid.place_digit_xy(1, 0, 3);
    // grid.place_digit_xy(1, 1, 1);
    // grid.place_digit_xy(1, 2, 2);
    //
    // grid.place_digit_xy(2, 0, 2);
    // grid.place_digit_xy(2, 1, 3);
    // grid.place_digit_xy(2, 2, 1);

    // implementing 2 here

    let mut ken_ken = KenKen::new(order);

    while get_empty_cells(&ken_ken).len() > 0 {
        insert_random_region(&mut ken_ken);
        println!("{}", ken_ken.to_string());
    }

    ken_ken
}

fn create_puzzle_from_grid(grid: &mut Grid) -> KenKen {
    let mut ken_ken = KenKen::new(grid.order());
    
    while get_empty_cells(&ken_ken).len() != 0 {
        let region_size = weighted_random(get_empty_cells(&ken_ken).len());
        let new_cells: Vec<usize> = find_contiguous_region(&ken_ken, region_size);

        // randomly choose operation
        let ops_vec = match new_cells.len() {
            1 => vec![Operation::Given],
            2 => vec![Operation::Add, Operation::Subtract, Operation::Divide, Operation::Multiply],
            _ => vec![Operation::Add, Operation::Multiply],
        };
        let new_op = ops_vec.choose(&mut thread_rng()).unwrap().clone();

        let new_target = match new_op { // different ranges depending on the op
            Operation::Add => todo!(), // sum the values in the cells,
            Operation::Subtract => todo!(), // get diff abs value
            Operation::Multiply => todo!(), // product
            Operation::Divide => thread_rng().gen_range(2..=ken_ken.order()),
            Operation::Given => thread_rng().gen_range(1..=ken_ken.order()),
            _ => panic!("Should never pick these operations")
        };

        let new_region = Region::new(Clue::new(new_op, new_target), new_cells);
        ken_ken.add_region(new_region);
    }

    // TODO wrap above in loop
    ken_ken
    
}

// O(order^2 * region_size) implementation of getting a contiguous region from the kenken grid
/// Returns a vector of contiguous squares 
fn find_contiguous_region(ken_ken: &KenKen, region_size: usize) -> Vec<usize> {
    let mut empty_cells = get_empty_cells(ken_ken);
    // let original_len = empty_cells.len();
    assert!(empty_cells.len() > 0, "Should never have no empty cells and call this function");
    assert!(region_size > 0, "Cannot have a region of size 0");

    empty_cells.shuffle(&mut thread_rng()); // TODO Problem is they are not contiguous
    let mut region_size = region_size; // TODO maybe different way of finding region size

    let mut contiguous_region: Vec<usize> = Vec::new();
    let mut empty_cells: VecDeque<usize> = VecDeque::from(empty_cells);
    contiguous_region.push(empty_cells.pop_front().unwrap()); // can't panic because of assert
    // empty_cells.push_back(ken_ken.order().pow(3)); // This is our end of queue value

    region_size -= 1;
    let mut cycles_without_reset = 0;

    while region_size > 0 && cycles_without_reset <= empty_cells.len() + 1 && empty_cells.len() > 0 {
        let empty_cell = empty_cells.pop_front().unwrap();
        let mut to_insert = None;

        // check all the cells in the contiguous region to see if this would be contiguous
        for used_cell in &contiguous_region {
            if is_adjacent(ken_ken.order(), &empty_cell, used_cell) {
                to_insert = Some(empty_cell);
                break;
            }
        }

        // check if we found a cell to insert
        match to_insert {
            Some(cell) => {
                cycles_without_reset = 0;
                region_size -= 1;
                contiguous_region.push(cell);
            },
            None => {
                cycles_without_reset += 1;
                empty_cells.push_back(empty_cell);
            },
        }
    }

    assert_eq!(0, cycles_without_reset, "A full region was not found, based on cycles_without_reset");
    assert_eq!(0, region_size, "A full region was not found, based on region_size");

    contiguous_region
}

// Takes a grid, and reports if it's solved or now
fn check_solved(grid: &mut Grid) -> SolvedStatus {
    latin_solve::stepped_logical_solver(grid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kenken::kenken_solve::ken_ken_logical_solver;

    #[test]
    fn add_region() {
        let mut ken_ken = KenKen::new(3);
        let result = insert_random_region(&mut ken_ken);
        assert_ne!(None, result);
        assert_eq!(1, ken_ken.regions().len());
        //println!("{}", ken_ken.to_string());
    }

    #[test]
    fn not_solved() { // testing check_solved()
        let ken_ken = KenKen::new(3);
        let mut grid = Grid::new(ken_ken.order());
        kenken_solve::ken_ken_logical_solver_with_grid(&mut grid, &ken_ken);
        let solved= check_solved(&mut grid);
        assert_eq!(SolvedStatus::Incomplete, solved);
    }

    #[test]
    fn test_is_adjacent() {
        assert!(is_adjacent(3, &4, &3));
        assert!(!is_adjacent(3, &2, &3));
        assert!(is_adjacent(3, &2, &5));
        assert!(!is_adjacent(3, &2, &2));
        assert!(!is_adjacent(2, &1, &2));
        assert!(!is_adjacent(2, &0, &3));
    }


    #[test]
    fn test_create_puzzle() {
        let order = 3;
        let ken_ken = create_puzzle_clues(order);
        assert_ne!(0, ken_ken.regions().len());
        let (status, grid) = ken_ken_logical_solver(&ken_ken);
        // TODO no assert yet
        
        // println!("{}", grid.candidates_to_string());
        // println!("{}", grid.digits_to_string());
    }

    #[test]
    fn find_region_one_cell() {
        let mut k = KenKen::new(1);
        let region = find_contiguous_region(&k, 1);
        assert_eq!(vec![0], region);
    }
    #[test]
    fn find_region_many_cells() {
        let mut k = KenKen::new(2);
        let mut region = find_contiguous_region(&k, 4);
        region.sort();
        assert_eq!(vec![0, 1, 2, 3], region);

        let mut k = KenKen::new(3);
        let mut region = find_contiguous_region(&k, 9);
        region.sort();
        assert_eq!(vec![0, 1, 2, 3, 4, 5, 6, 7, 8], region);
    }

    #[test]
    fn find_region_finds_contiguous() {
        let mut k = KenKen::new(6);
        let mut region = find_contiguous_region(&k, 6);
        for i in 0..region.len() {
            let mut is_adjacent_to_something = false;
            for j in 0..region.len() {
                if i != j {
                    is_adjacent_to_something |= is_adjacent(k.order(), &region[i], &region[j]);
                }
            }
            assert!(is_adjacent_to_something, "{:?}", region);
        }

        let mut k = KenKen::new(10);
        let mut region = find_contiguous_region(&k, 9);
        for i in 0..region.len() {
            let mut is_adjacent_to_something = false;
            for j in 0..region.len() {
                if i != j {
                    is_adjacent_to_something |= is_adjacent(k.order(), &region[i], &region[j]);
                }
            }
            assert!(is_adjacent_to_something, "{:?}", region);
        }
    }


}
