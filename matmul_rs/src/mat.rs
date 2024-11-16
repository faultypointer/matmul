use rand::Fill;

pub struct Mat {
    pub row: usize,
    pub col: usize,
    pub data: Vec<f32>,
}

impl Mat {
    pub fn new(row: usize, col: usize) -> Self {
        Self {
            row,
            col,
            data: vec![0.0_f32; row * col],
        }
    }

    pub fn constant(row: usize, col: usize, val: f32) -> Self {
        Self {
            row,
            col,
            data: vec![val; row * col],
        }
    }

    pub fn random(row: usize, col: usize) -> Self {
        let mut temp = Self::new(row, col);
        let mut rng = rand::thread_rng();
        temp.data.try_fill(&mut rng).unwrap();
        temp
    }

    #[inline]
    pub fn at(&self, r: usize, c: usize) -> f32 {
        self.data[self.row * c + r]
    }

    pub fn print(&self) {
        for i in 0..self.row {
            for j in 0..self.col {
                print!("{:.3} ", self.data[j * self.row + i]);
            }
            println!("");
        }
    }
}
#[macro_export]
macro_rules! matrix {
    [$([$($x:expr),* $(,)*]),* $(,)*] => {{
        let nested = vec![$(vec![$($x as f32),*]),*];
        let rows = nested.len();
        let cols = if rows > 0 { nested[0].len() } else { 0 };

        assert!(nested.iter().all(|row| row.len() == cols),
            "All rows must have the same length");

        // Convert to column-major
        let mut data = Vec::with_capacity(rows * cols);
        for c in 0..cols {
            for r in 0..rows {
                data.push(nested[r][c]);
            }
        }

        Mat {
            row: rows,
            col: cols,
            data,
        }
    }};
}
