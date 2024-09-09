use crate::latin::SolvedStatus;
use crate::latin::LatinSolver;
use crate::kenken::KenKenSolver;
use crate::kenken::KenKen;
use crate::kenken::Region;
use crate::kenken::Clue;
use crate::kenken::Operation;


fn create_puzzle(latin_solver: &mut LatinSolver) -> KenKen {

    latin_solver.place_digit_xy(0, 0, 1);
    latin_solver.place_digit_xy(0, 1, 2);
    latin_solver.place_digit_xy(0, 2, 3);

    latin_solver.place_digit_xy(1, 0, 3);
    latin_solver.place_digit_xy(1, 1, 1);
    latin_solver.place_digit_xy(1, 2, 2);

    latin_solver.place_digit_xy(2, 0, 2);
    latin_solver.place_digit_xy(2, 1, 3);
    latin_solver.place_digit_xy(2, 2, 1);

    let mut ken_ken = KenKen::new(latin_solver.order());

    for (index, digit) in latin_solver.grid().iter().enumerate() {
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
        let mut puzzle = LatinSolver::new(order);

        let ken_ken = create_puzzle(&mut puzzle);
        let mut solver = KenKenSolver::new(ken_ken);
        //let mut latin_solver = solver.latin_solver;

        assert_eq!(SolvedStatus::Complete, solver.ken_ken_logical_solver())
        
    }

    #[test]
    fn empty_grid_solve() {
        let mut blank_solver = LatinSolver::new(3);
        let output = blank_solver.stepped_logical_solver();
        assert_eq!(output, SolvedStatus::Incomplete);


    }

    #[test]
    fn conflicting_grid_solve() {
        let mut blank_solver = LatinSolver::new(2);
        blank_solver.set_cube_available(0, 0, 0b1);
        blank_solver.set_cube_available(1, 1, 0b10);

        let output = blank_solver.stepped_logical_solver();
        assert_eq!(output, SolvedStatus::Broken);
    }



}