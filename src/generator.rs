use crate::grid::Grid;
use crate::kenken::{Clue, KenKen, Operation, Region};
use crate::latin::SolvedStatus;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::collections::HashMap;
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
        Operation::Multiply => thread_rng().gen_range(1..ken_ken.order().pow(region_size as u32)),
        Operation::Divide | Operation::Given => thread_rng().gen_range(1..=ken_ken.order()),
        _ => panic!("Should never pick these operations")
    };

    let new_region = Region::new(Clue::new(new_op, new_target), empty_cells[0..region_size].to_vec());
    ken_ken.add_region(new_region);

    Some(())
}

// Check if two cells are adjacent
fn is_adjacent(order: usize, first: usize, second: usize) -> bool {
    assert!(order > 1, "A grid must be at least 2x2");

    let first_x = first / order;
    let first_y = first % order;
    let second_x = second / order;
    let second_y = second % order;

    assert!(first_x < order, "first_x value out of bounds");
    assert!(second_x < order, "second_x value out of bounds");

    (first_x.abs_diff(second_x) + first_y.abs_diff(second_y)) == 1
}

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

fn create_puzzle(grid: &Grid) -> KenKen {
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

    let mut ken_ken = KenKen::new(grid.order());

    while get_empty_cells(&ken_ken).len() > 0 {
        insert_random_region(&mut ken_ken);
        println!("{}", ken_ken.to_string());
    }

    ken_ken
}

fn check_solved(grid: &Grid) -> SolvedStatus {
    //let mut ken_ken = KenKen::new(grid.order());

    // for (index, digit) in grid.digits().iter().enumerate() {
    //     let region = Region::new(Clue::new(Operation::Given, *digit), vec![index]);
    //     ken_ken.regions.push(region);
    // }

    SolvedStatus::Complete
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
    fn test_is_adjacent() {
        assert!(is_adjacent(3, 4, 3));
        assert!(!is_adjacent(3, 2, 3));
        assert!(is_adjacent(3, 2, 5));
        assert!(!is_adjacent(3, 2, 2));
        assert!(!is_adjacent(2, 1, 2));
        assert!(!is_adjacent(2, 0, 3));
    }


    #[test]
    fn test_create_puzzle() {
        let order = 3;
        let grid = Grid::new(order);

        // println!("{}", grid.candidates_to_string());
        // println!("{}", grid.digits_to_string());

        let ken_ken = create_puzzle(&grid);

        assert_ne!(0, ken_ken.regions().len());

        let status = ken_ken_logical_solver(ken_ken);

        assert_eq!(
            SolvedStatus::Broken,
            status
        );

        // println!("{}", grid.candidates_to_string());
        // println!("{}", grid.digits_to_string());
    }


}
