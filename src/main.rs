// #![allow(unreachable_code)]
// #![allow(unused_mut)]
// #![allow(unused_variables)]

use std::env;
mod latin;
use crate::grid::Grid;

mod kenken;
mod math;
mod generator;
mod grid;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let mut g = Grid::new(6);

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
}
