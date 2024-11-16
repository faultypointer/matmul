use crate::mat::Mat;

pub fn matmul(a: &Mat, b: &Mat, c: &mut Mat) {
    let m = a.row;
    let n = a.col;
    let k = b.col;

    assert!(n == b.row);
    assert!(m == c.row);
    assert!(k == c.col);

    for i in 0..m {
        for j in 0..k {
            for l in 0..n {
                c.data[j * m + i] += a.data[l * m + i] * b.data[j * k + l];
            }
        }
    }
}
