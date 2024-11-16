mod mat;
mod vers;

use mat::Mat;
use std::env;
use std::io::Write;
use vers::kernel;
use vers::naive;

fn main() {
    let args: Vec<String> = env::args().collect();
    let (min_size, max_size, n_pts, warmup) = if args.len() < 5 {
        // Default values if not enough arguments provided
        (200, 5000, 50, 15)
    } else {
        // Parse command line arguments
        (
            args[1].parse::<usize>().expect("Error parsing MINSIZE"),
            args[2].parse::<usize>().expect("Error parsing MAXSIZE"),
            args[3].parse::<usize>().expect("Error parsing NPTS"),
            args[4].parse::<usize>().expect("Error parsing WARMUP"),
        )
    };
    println!("======================");
    println!("MINSIZE = {min_size}");
    println!("MAXSIZE = {max_size}");
    println!("NPTS = {n_pts}");
    println!("WARMUP = {warmup}");
    println!("======================");
    let mut avg_gflops = vec![0.0; n_pts];
    let mut min_gflops = vec![0.0; n_pts];
    let mut max_gflops = vec![0.0; n_pts];
    let mut global_max_gflops: f64 = 0.0;
    let mut mat_sizes = vec![0; n_pts];

    let delta_size = (max_size - min_size) / (n_pts - 1);
    for i in 0..(n_pts - 1) {
        mat_sizes[i] = min_size + i * delta_size;
    }
    mat_sizes[n_pts - 1] = max_size;

    // warmup
    {
        println!("Warm-up");
        let a = Mat::random(max_size, max_size);
        let b = Mat::random(max_size, max_size);
        let mut c = Mat::constant(max_size, max_size, 0.0);
        for i in 0..warmup {
            print!("\r{} / {warmup}", i + 1);
            std::io::stdout().flush().unwrap();
            naive::matmul(&a, &b, &mut c);
        }
    }

    println!("\nBenchmark: ");
    for i in 0..n_pts {
        print!("\r {} / {n_pts}", i + 1);
        std::io::stdout().flush().unwrap();
        let mat_size = mat_sizes[i];
        let a = Mat::random(mat_size, mat_size);
        let b = Mat::random(mat_size, mat_size);
        let mut c = Mat::constant(mat_size, mat_size, 0.0);

        let flop: f64 = 2.0 * mat_size.pow(3) as f64;
        let mut avg_exec_time: f64 = 0.0;
        let mut max_exec_time: f64 = 0.0;
        let mut min_exec_time: f64 = 1e69;
        let n_iter = 200000 / mat_size as u128;
        for _ in 0..n_iter {
            let start = std::time::Instant::now();
            kernel::matmul(&a, &b, &mut c);
            let elapsed = start.elapsed().as_secs_f64();
            max_exec_time = max_exec_time.max(elapsed);
            min_exec_time = min_exec_time.min(elapsed);
            avg_exec_time += elapsed;
        }
        avg_exec_time /= n_iter as f64;
        avg_gflops[i] = (flop / avg_exec_time) / 1e9;
        max_gflops[i] = flop / min_exec_time / 1e9;
        min_gflops[i] = flop / max_exec_time / 1e9;
        global_max_gflops = global_max_gflops.max(max_gflops[i]);
    }

    println!("\n=========================");
    let mut bench_string = String::with_capacity(n_pts * 100);
    for i in 0..n_pts {
        bench_string.push_str(&format!(
            "{} {} {} {}\n",
            mat_sizes[i], min_gflops[i], max_gflops[i], avg_gflops[i]
        ));
    }
    std::fs::write("benchmark_rust.txt", bench_string).unwrap();
    println!("PEAK GFLOPS = {global_max_gflops}");
    println!("Benchmark result were saved in benchmark_rust.txt");
}

#[test]
fn test_naive() {
    for _ in 0..100 {
        let a = Mat::random(1200, 1200);
        let b = Mat::random(1200, 1200);
        let mut c = Mat::constant(1200, 1200, 0.0);

        let flops = 2.0 * (1200.0_f64).powf(3.0);
        let start = std::time::Instant::now();
        kernel::matmul(&a, &b, &mut c);
        let elapsed = start.elapsed().as_secs_f64();
        println!("Exec. time: {}ms", elapsed * 1000.0);
        println!("GFLOPS = {}\n\n", flops / elapsed / 1e9);
    }
}
