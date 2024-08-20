mod latin

struct KenKen {
    order: usize,       // the dimension of the square KenKen grid
    cube: Vec<bool>,    // order^3

    // might be useful to have grid appear elsewhere as it's own type
    grid: Vec<usize>,   // order^2

    // Currently unused
    //row: Vec<bool>,     // order^2
    //col: Vec<bool>,     // order^2
}

impl KenKen {
    
}


struct KenKenSolver {
    ken_ken: KenKen,
    latin_solver: LatinSolver,
}