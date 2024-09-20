#![allow(dead_code)]

use crate::grid::Grid;
use crate::kenken::{Clue, KenKen, Operation, Region, kenken_solve};
use crate::latin::{SolvedStatus, latin_solve};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::collections::{HashMap, VecDeque};
use std::io::Error;
use std::ops::Deref;
use crate::math::math;

fn insert_random_region(ken_ken: &mut KenKen) -> Option<()> {
    let mut empty_cells = get_empty_cells(ken_ken.deref());
    empty_cells.shuffle(&mut thread_rng()); // TODO Problem is they are not contiguous
    let region_size = weighted_random(empty_cells.len());

    // randomly choose operation
    let new_op = generate_operation(region_size);

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
    assert!(n > 0, "Cannot get a random region size of size less than 1");
    let result: usize = thread_rng().gen_range(1..=(n) as u32) as usize;

    // match random_number {
    //     1..n/2 => {}
    // } 

    assert!(result <= n && result > 0);
    result
}

///
///
/// # Arguments
///
/// * `order`:
///
/// returns: KenKen
///
/// # Examples
///
/// ```
///
/// ```
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

/* TODO major issue here is that it's not getting a good distribution of cell sizes.
 *  Especially when it splits up a double cell into a two ones, and there is another one available
 *  It should try to prioritize looking for another grouping before jumping down to a smaller region.
 *  Or maybe there's a better way to iterate that builds all the regions at once.
 */
fn create_puzzle_from_grid(grid: &Grid) -> KenKen {
    let mut ken_ken = KenKen::new(grid.order());
    
    while get_empty_cells(&ken_ken).len() != 0 {
        println!("{:?}", get_empty_cells(&ken_ken));
        let max_region_size = get_empty_cells(&ken_ken).len();
        let mut region_size = weighted_random(max_region_size);
        region_size = 3; // TODO remove
        let mut new_cells = vec![]; // TODO Maybe give better default
        while region_size > 0 {
            if let Some(region_cells) = find_contiguous_region(&ken_ken, region_size) {
                new_cells = region_cells;
                break;
            } else {
                region_size -= 1;
            }
        }

        // randomly choose operation
        let new_op = generate_operation(new_cells.len());
        let new_target = generate_target(grid, &new_cells, &new_op);

        let new_region = Region::new(Clue::new(new_op, new_target), new_cells);
        ken_ken.add_region(new_region);
    }

    // TODO wrap above in loop
    ken_ken
    
}

fn generate_operation(new_cells_len: usize) -> Operation {
    let ops_vec = match new_cells_len {
        1 => vec![Operation::Given],
        2 => vec![Operation::Add, Operation::Subtract, Operation::Divide, Operation::Multiply],
        _ => vec![Operation::Add, Operation::Multiply],
    };
    ops_vec.choose(&mut thread_rng()).unwrap().clone()
}

///
///
/// # Arguments
///
/// * `grid`: to be used to get the digits we've assigned to the grid
/// * `new_cells`: the cells to be added to a new region
/// * `new_op`: the `Operation` for the region `Clue`
///
/// returns: usize of the target for the region `Clue`
///
fn generate_target(grid: &Grid, new_cells: &Vec<usize>, new_op: &Operation) -> usize {
    let new_target: usize = match new_op { // different ranges depending on the op
        Operation::Add => new_cells.iter().fold(0, |acc, &cell| acc + grid.get_digits_flat(cell)),
        Operation::Subtract => {
            let first_cell = grid.get_digits_flat(new_cells[0]);
            let second_cell = grid.get_digits_flat(new_cells[1]);
            first_cell.abs_diff(second_cell)
        },
        Operation::Multiply => {
            new_cells.iter().fold(1, |acc, &cell| acc * grid.get_digits_flat(cell))
            // let mut product = 1;
            // for cell in &new_cells{
            //     product *= grid.get_digits_flat(*cell);
            // }
            // product
        }, // product
        Operation::Divide => {
            let first_cell = grid.get_digits_flat(new_cells[0]);
            let second_cell = grid.get_digits_flat(new_cells[1]);
            assert_ne!(first_cell, second_cell, "Cannot have divide region with the same number twice");

            // TODO fix error happening here... maybe handle by switching to subtract
            assert!(first_cell%second_cell == 0 && second_cell%first_cell == 0, "Ended up with two digits not divisible by eachother");

            match first_cell > second_cell {
                true => first_cell / second_cell,
                false => second_cell / first_cell,
            }
        },
        Operation::Given => grid.get_digits_flat(new_cells[0]),
        _ => panic!("Should never pick these operations")
    };
    new_target
}

// O(order^2 * region_size) implementation of getting a contiguous region from the kenken grid
/// Returns a vector of contiguous squares 
fn find_contiguous_region(ken_ken: &KenKen, region_size: usize) -> Option<Vec<usize>> { // TODO switch `Option` to `Result`
    let mut empty_cells = get_empty_cells(ken_ken);
    // let original_len = empty_cells.len();
    assert!(empty_cells.len() > 0, "Should never have no empty cells and call this function");
    assert!(region_size > 0, "Cannot have a region of size 0");

    empty_cells.shuffle(&mut thread_rng()); // TODO Problem is they are not contiguous
    let mut region_size = region_size; // TODO maybe different way of finding region size

    let mut contiguous_region: Vec<usize> = Vec::new();
    let mut empty_cells: VecDeque<usize> = VecDeque::from(empty_cells);
    contiguous_region.push(empty_cells.pop_front().unwrap()); // can't panic because of assert

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

    if region_size > 0 || cycles_without_reset > 0 {
        // assert_eq!(0, cycles_without_reset, "A full region was not found, based on cycles_without_reset");
        // assert_eq!(0, region_size, "A full region was not found, based on region_size");
        None
    } else {
        Some(contiguous_region)
    }


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
    fn test_is_adjacent() {
        assert!(is_adjacent(3, &4, &3));
        assert!(!is_adjacent(3, &2, &3));
        assert!(is_adjacent(3, &2, &5));
        assert!(!is_adjacent(3, &2, &2));
        assert!(!is_adjacent(2, &1, &2));
        assert!(!is_adjacent(2, &0, &3));
    }

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
    fn test_create_puzzle_from_grid() {
        let order = 3;
        let mut grid = Grid::new(order);
        grid.place_digit_xy(0, 0, 1);
        grid.place_digit_xy(0, 1, 2);
        grid.place_digit_xy(0, 2, 3);

        grid.place_digit_xy(1, 0, 3);
        grid.place_digit_xy(1, 1, 1);
        grid.place_digit_xy(1, 2, 2);

        grid.place_digit_xy(2, 0, 2);
        grid.place_digit_xy(2, 1, 3);
        grid.place_digit_xy(2, 2, 1);
        let ken_ken = create_puzzle_from_grid(&grid);

        assert_eq!(Vec::<usize>::new(), get_empty_cells(&ken_ken));
        println!("{}", ken_ken.to_string());

        let (status, mut solved_grid) = ken_ken_logical_solver(&ken_ken);

        assert_ne!(SolvedStatus::Broken, check_solved(&mut solved_grid));
        // Eventually replace with assert_eq!(SolvedStatus::Complete, check_solved(&mut solved_grid));
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
        let k = KenKen::new(1);
        let region = find_contiguous_region(&k, 1).unwrap();
        assert_eq!(vec![0], region);
    }
    #[test]
    fn find_impossible_region() {
        let k = KenKen::new(2);
        let region = find_contiguous_region(&k, 6);
        assert_eq!(None, region);

        let mut k = KenKen::new(3);
        k.add_region(Region::new(Clue::new(Operation::Add, 6), vec![3, 4, 5]), );
        let region = find_contiguous_region(&k, 4);
        assert_eq!(None, region);
    }

    #[test]
    fn find_region_many_cells() {
        let k = KenKen::new(2);
        let mut region = find_contiguous_region(&k, 4).unwrap();
        region.sort();
        assert_eq!(vec![0, 1, 2, 3], region);

        let k = KenKen::new(3);
        let mut region = find_contiguous_region(&k, 9).unwrap();
        region.sort();
        assert_eq!(vec![0, 1, 2, 3, 4, 5, 6, 7, 8], region);
    }

    #[test]
    fn find_region_finds_contiguous() {
        let k = KenKen::new(6);
        let region = find_contiguous_region(&k, 6).unwrap();
        for i in 0..region.len() {
            let mut is_adjacent_to_something = false;
            for j in 0..region.len() {
                if i != j {
                    is_adjacent_to_something |= is_adjacent(k.order(), &region[i], &region[j]);
                }
            }
            assert!(is_adjacent_to_something, "{:?}", region);
        }

        let k = KenKen::new(10);
        let region = find_contiguous_region(&k, 9).unwrap();
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
