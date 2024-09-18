// #![allow(unreachable_code)]
// #![allow(unused_mut)]
// #![allow(unused_variables)]
#![allow(dead_code)]

use std::env;
mod generator;
mod grid;
mod kenken;
mod latin;
mod math;

use crate::grid::Grid;
use crate::kenken::{Clue, KenKen, Operation, Region};
use crate::latin::latin_solve;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let mut g = Grid::new(6);

    g.place_digit_xy(3, 3, 3);
    g.place_digit_xy(1, 3, 2);
    g.place_digit_xy(0, 0, 2);

    latin_solve::stepped_logical_solver(&mut g);

    // println!("{}", g.get_cube_value(3, 3, 2));
    // println!("{}", g.get_cube_value(3, 3, 3));

    println!("{}", g.candidates_to_string());
    println!("{}", g.digits_to_string());

    let mut k = KenKen::new(g.order());
    k.regions
        .push(Region::new(Clue::new(Operation::Add, 3), vec![3, 4]));
    let r = k.region_n(2);
    println!("{:?}", r.clue());
}

#[cfg(test)]
mod tests {}
