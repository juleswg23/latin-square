#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

use std::env;
mod latin;
use latin::LatinSolver;
mod kenken;
mod math;

fn main() {
    use std::time::Instant;
    env::set_var("RUST_BACKTRACE", "1");
    let mut ls = LatinSolver::new(6);
    // println!("{}", ls.cube.len());
    // println!("{}", ls.get_cube_loc(3, 3, 4));
    // println!("{}", ls.get_cube_value(3, 3, 2));
    // println!("{}", ls.get_grid_value(3, 3));

    ls.place_digit(3, 3, 3);
    ls.place_digit(1, 3, 2);
    ls.place_digit(0, 0, 2);

    // println!("{}", ls.get_cube_value(3, 3, 2));
    // println!("{}", ls.get_cube_value(3, 3, 3));

    println!("{}", ls.cube_to_string());
    println!("{}", ls.grid_to_string());

    let now = Instant::now();

    //let mut count: u64 = 0;
    println!("solve success? {}", ls.solve(false));
    //println!("count {}", count);

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
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
        let mut ls = LatinSolver::new(3);
        // println!("{}", ls.cube.len());
        // println!("{}", ls.get_cube_loc(3, 3, 4));
        // println!("{}", ls.get_cube_value(3, 3, 2));
        // println!("{}", ls.get_grid_value(3, 3));

        ls.place_digit(0, 1, 3);
        ls.place_digit(0, 0, 1);

        // println!("{}", ls.get_cube_value(3, 3, 2));
        // println!("{}", ls.get_cube_value(3, 3, 3));

        println!("{}", ls.cube_to_string());
        println!("{}", ls.grid_to_string());

        let now = Instant::now();

        //let mut count: u64 = 0;
        println!("solve success? {}", ls.solve(false));
        //println!("count {}", count);

        let elapsed = now.elapsed();
        println!("Elapsed: {:.2?}", elapsed);
        assert_eq!(3, 3);
    }

    #[test]
    fn simple_solve_test() {
        let mut ls = LatinSolver::new(3);
    }


}
