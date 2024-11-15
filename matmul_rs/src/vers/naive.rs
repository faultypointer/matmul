type Mat = Vec<Vec<f32>>;
pub fn matmul(a: &Mat, b: &Mat, c: &mut Mat) {
    assert!(a.len() > 0);
    assert!(b.len() > 0);
    assert!(c.len() > 0);

    let m = a.len();
    let n = a[0].len();
    let k = b[0].len();

    assert!(n == b.len());
    assert!(m == c.len());
    assert!(k == c[0].len());

    for i in 0..m {
        for j in 0..k {
            c[i][j] = 0.0;
            for l in 0..n {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
}
