#[allow(dead_code)]
struct LatinSolver {
    order: usize,
    cube: Vec<bool>, // order^3

    // might be useful to have grid appear elsewhere as it's own type
    // digit is a number from 1 to order (not just at 0 and 1)
    grid: Vec<i32>, // order^2
    row: Vec<bool>, // order^2
    col: Vec<bool>, // order^2
}

#[allow(dead_code)]
impl LatinSolver {
    fn new(order: usize) -> LatinSolver {
        LatinSolver {
            order: order,
            cube: vec![true; order.pow(3)],
            grid: vec![0; order.pow(2)],
            col: vec![true; order.pow(2)],
            row: vec![true; order.pow(2)],
        }
    }

    fn get_cube_pos(&self, x: usize, y: usize, n: usize) -> usize {
        let location = (x * &self.order + y) * &self.order; // try removing &
        location + n - 1
    }

    fn get_cube_value(&self, x: usize, y: usize, n: usize) -> bool {
        let position = self.get_cube_pos(x, y, n);
        self.cube[position]
    }

    fn get_grid_pos(&self, x: usize, y: usize) -> usize {
        x * &self.order + y
    }

    fn get_grid_value(&self, x: usize, y: usize) -> i32 {
        self.grid[self.get_grid_pos(x, y)]
    }

    


}




fn main() {
    let ls = LatinSolver::new(6);
    println!("{}", ls.cube.len());
    println!("{}", ls.get_cube_pos(3, 3, 2));
    println!("{}", ls.get_cube_value(3, 3, 2));
    println!("{}", ls.get_grid_value(3, 3));

}
