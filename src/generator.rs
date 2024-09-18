use crate::grid::Grid;
use crate::kenken::{kenken_solve, Clue, KenKen, Operation, Region};
use crate::latin::{SolvedStatus};

fn create_puzzle(grid: &mut Grid) -> KenKen {
    grid.place_digit_xy(0, 0, 1);
    grid.place_digit_xy(0, 1, 2);
    grid.place_digit_xy(0, 2, 3);

    grid.place_digit_xy(1, 0, 3);
    grid.place_digit_xy(1, 1, 1);
    grid.place_digit_xy(1, 2, 2);

    grid.place_digit_xy(2, 0, 2);
    grid.place_digit_xy(2, 1, 3);
    grid.place_digit_xy(2, 2, 1);

    let mut ken_ken = KenKen::new(grid.order());

    for (index, digit) in grid.digits().iter().enumerate() {
        let region = Region::new(Clue::new(Operation::Given, *digit), vec![index]);
        ken_ken.regions.push(region);
    }

    ken_ken
}

#[cfg(test)]
mod tests {
    use crate::kenken::kenken_solve::ken_ken_logical_solver;
    use super::*;

    #[test]
    fn test_create_puzzle() {
        let order = 3;
        let mut grid = Grid::new(order);
        
        // println!("{}", grid.candidates_to_string());
        // println!("{}", grid.digits_to_string());

        let ken_ken = create_puzzle(&mut grid);

        grid = Grid::new(order);
        let status = ken_ken_logical_solver(&mut grid, ken_ken);

        assert_eq!(
            SolvedStatus::Complete,
            status
        );

        // println!("{}", grid.candidates_to_string());
        // println!("{}", grid.digits_to_string());
    }


}
