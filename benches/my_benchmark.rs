use criterion::async_executor::FuturesExecutor;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use wgpu_test::warshall_floyd::{stage_adjacency_matrix, WarshallFloyd, WarshallFloydKernel};

fn warshall_floyd_cpu(n: usize, edges: &[(usize, usize)], distance: &mut [f32]) {
    distance.fill(std::f32::INFINITY);
    for i in 0..n {
        distance[i * n + i] = 0.;
    }
    for &(i, j) in edges.iter() {
        distance[i * n + j] = 1.;
        distance[j * n + i] = 1.;
    }
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let d = distance[i * n + k] + distance[k * n + j];
                if d < distance[i * n + j] {
                    distance[i * n + j] = d;
                }
            }
        }
    }
}

fn create_graph(n: usize) -> Vec<(usize, usize)> {
    (1..n).map(|i| (i - 1, i)).collect::<Vec<_>>()
}

fn criterion_benchmark(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("ASAP");
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        for n in (100..=1000).step_by(100) {
            let graph = create_graph(n);

            group.bench_with_input(BenchmarkId::new("CPU", n), &graph, |bench, graph| {
                let mut distance = vec![0.; n * n];
                bench.iter(|| warshall_floyd_cpu(n, &graph, &mut distance));
            });

            group.bench_with_input(BenchmarkId::new("GPU", n), &graph, |bench, graph| {
                bench.to_async(FuturesExecutor).iter(|| async {
                    let adapter = instance
                        .request_adapter(&wgpu::RequestAdapterOptions::default())
                        .await
                        .unwrap();
                    let (device, queue) = adapter
                        .request_device(
                            &wgpu::DeviceDescriptor {
                                label: None,
                                features: wgpu::Features::empty(),
                                limits: wgpu::Limits::downlevel_defaults(),
                            },
                            None,
                        )
                        .await
                        .unwrap();
                    let wf = WarshallFloyd::new(&device);
                    let size = WarshallFloydKernel::buffer_size(n);
                    let in_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: size as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: size as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });

                    stage_adjacency_matrix(&device, &queue, &graph, n, &in_buffer);
                    wf.run(&device, &queue, &in_buffer, &out_buffer, n);
                    device.poll(wgpu::Maintain::Wait);
                });
            });
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::new(5, 0));
    targets = criterion_benchmark
}
criterion_main!(benches);
