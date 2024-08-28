#![allow(unreachable_code)]
#![allow(unused_mut)]

use std::env;
mod latin;
use latin::LatinSolver;
mod kenken;

fn main() {
    kenken::main();

    return;
    use std::time::Instant;
    env::set_var("RUST_BACKTRACE", "1");
    let mut ls = LatinSolver::new(6);
    // println!("{}", ls.cube.len());
    // println!("{}", ls.get_cube_loc(3, 3, 4));
    // println!("{}", ls.get_cube_value(3, 3, 2));
    // println!("{}", ls.get_grid_value(3, 3));

    //ls.place_digit(3, 3, 3);
    //ls.place_digit(1, 3, 2);
    //ls.place_digit(0, 0, 2);

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
