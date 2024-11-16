use std::arch::x86_64::{self, __m256};

const MR: usize = 16;
const NR: usize = 6;

use crate::mat::Mat;

pub fn matmul(a: &Mat, b: &Mat, c: &mut Mat) {
    let m = a.row;
    let n = a.col;
    let k = b.col;

    assert!(n == b.row);
    assert!(m == c.row);
    assert!(k == c.col);

    assert!(m % MR == 0);
    assert!(n % NR == 0);

    for i in (0..m).step_by(MR) {
        for j in (0..n).step_by(NR) {
            unsafe {
                let mut c_buff: [[__m256; 2]; 6] = [[x86_64::_mm256_setzero_ps(); 2]; 6];
                let mut b_pack_float_8: __m256;
                let mut a0_pack_float_8: __m256;
                let mut a1_pack_float_8: __m256;

                for l in 0..6 {
                    c_buff[l][0] =
                        x86_64::_mm256_loadu_ps(c.data.as_ptr().offset((j * m) as isize));
                    c_buff[l][1] =
                        x86_64::_mm256_loadu_ps(c.data.as_ptr().offset((j * m + 8) as isize));
                }

                for p in 0..k {
                    let a_idx = (i + p * m) as isize;
                    a0_pack_float_8 = x86_64::_mm256_loadu_ps(a.data.as_ptr().offset(a_idx));
                    a1_pack_float_8 = x86_64::_mm256_loadu_ps(a.data.as_ptr().offset(a_idx + 8));

                    for n in 0..6 {
                        b_pack_float_8 = x86_64::_mm256_broadcast_ss(&b.data[(j + n) * k + p]);
                        c_buff[n][0] =
                            x86_64::_mm256_fmadd_ps(a0_pack_float_8, b_pack_float_8, c_buff[n][0]);
                        c_buff[n][1] =
                            x86_64::_mm256_fmadd_ps(a1_pack_float_8, b_pack_float_8, c_buff[n][1]);
                    }

                    for l in 0..6 {
                        x86_64::_mm256_storeu_ps(
                            c.data.as_mut_ptr().offset((j * m) as isize),
                            c_buff[l][0],
                        );

                        x86_64::_mm256_storeu_ps(
                            c.data.as_mut_ptr().offset((j * m + 8) as isize),
                            c_buff[l][1],
                        );
                    }
                }
            }
        }
    }
}
