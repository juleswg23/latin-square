use crate::grid::Grid;
use crate::latin::SolvedStatus;
use crate::latin::latin_solve;
use crate::kenken::KenKenSolver;
use crate::kenken::KenKen;
use crate::kenken::Region;
use crate::kenken::Clue;
use crate::kenken::Operation;


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
        let region =Region::new(Clue::new(Operation::Given, *digit), vec![index]);
        ken_ken.regions.push(region);

    }

    ken_ken
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate() {
        let order = 3;
        let mut puzzle = Grid::new(order);

        let ken_ken = create_puzzle(&mut puzzle);
        let mut solver = KenKenSolver::new(ken_ken);
        //let mut grid = solver.grid;

        assert_eq!(SolvedStatus::Complete, solver.ken_ken_logical_solver())
        
    }

    #[test]
    fn empty_grid_solve() {
        let mut grid = Grid::new(3);
        let output = latin_solve::stepped_logical_solver(&mut grid);
        assert_eq!(output, SolvedStatus::Incomplete);


    }

    #[test]
    fn conflicting_grid_solve() {
        let mut grid = Grid::new(2);
        grid.set_candidates_available(0, 0, 0b1);
        grid.set_candidates_available(1, 1, 0b10);

        let output = latin_solve::stepped_logical_solver(&mut grid);
        assert_eq!(output, SolvedStatus::Broken);
    }



}