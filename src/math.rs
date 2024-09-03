// Some useful math operations reused in a couple different contexts

pub mod math {
    // Given an index into a flat array, return the (x,y) index pair.
    pub fn xy_pair(index: usize, order: usize) -> (usize, usize) {
        assert!(index < order.pow(2), "The index provided to xy_pair is out of range.");
        (index / order, index % order)
    }

    // Returns the location of the rightmost set bit
    // Note that this is in digit notation, so 0b110 returns 2
    pub fn min_bit(mask: i32) -> usize {
        assert_ne!(mask, 0b0, "Trying to find min_bit but no bits are set.");
        ((mask & -mask) as f32).log2() as usize + 1
    }

    // Returns the location of the leftmost set bit
    // Note that this is in digit notation, so 0b110 returns 3
    pub fn max_bit(mask: i32) -> usize {
        assert_ne!(mask, 0b0, "Trying to find max_bit but no bits are set.");
        (mask as f32 + 1.0).log2().ceil() as usize
    }
}