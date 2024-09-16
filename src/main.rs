#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

use std::env;
mod latin;
use crate::grid::Grid;

mod kenken;
mod math;
mod generator;
mod grid;

fn main() {
    use std::time::Instant;
    env::set_var("RUST_BACKTRACE", "1");
    let mut g = Grid::new(6);
    // println!("{}", g.cube.len());
    // println!("{}", g.get_cube_loc(3, 3, 4));
    // println!("{}", g.get_cube_value(3, 3, 2));
    // println!("{}", g.get_grid_value(3, 3));

    g.place_digit_xy(3, 3, 3);
    g.place_digit_xy(1, 3, 2);
    g.place_digit_xy(0, 0, 2);

    // println!("{}", g.get_cube_value(3, 3, 2));
    // println!("{}", g.get_cube_value(3, 3, 3));

    println!("{}", g.candidates_to_string());
    println!("{}", g.digits_to_string());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    use std::env;
    // use latin::LatinSolver;

    //#[test]
    #[allow(dead_code)]
    fn test2() {
        env::set_var("RUST_BACKTRACE", "1");
        let mut g = Grid::new(3);
        // println!("{}", g.cube.len());
        // println!("{}", g.get_cube_loc(3, 3, 4));
        // println!("{}", g.get_cube_value(3, 3, 2));
        // println!("{}", g.get_grid_value(3, 3));

        g.place_digit_xy(0, 1, 3);
        g.place_digit_xy(0, 0, 1);

        // println!("{}", g.get_cube_value(3, 3, 2));
        // println!("{}", g.get_cube_value(3, 3, 3));

        println!("{}", g.candidates_to_string());
        println!("{}", g.digits_to_string());

        // let now = Instant::now();
        // 
        // //let mut count: u64 = 0;
        // println!("solve success? {}", g.solve(false));
        // //println!("count {}", count);
        // 
        // let elapsed = now.elapsed();
        // println!("Elapsed: {:.2?}", elapsed);
        // assert_eq!(3, 3);
    }

    // #[test]
    // fn simple_solve_test() {
    //     let mut g = LatinSolver::new(3);
    // }


}
