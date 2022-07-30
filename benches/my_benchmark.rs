use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use wgpu_test::tropical_matmul::{BlockedTropicalMatmulKernel, NaiveTropicalMatmulKernel};

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
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
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
        let wf = wgpu_test::warshall_floyd::WarshallFloyd::new(&device);
        let tm_naive = wgpu_test::tropical_matmul::TropicalMatmul::new(
            NaiveTropicalMatmulKernel::new(&device),
        );
        let tm_block = wgpu_test::tropical_matmul::TropicalMatmul::new(
            BlockedTropicalMatmulKernel::new(&device),
        );
        {
            let mut group = c.benchmark_group("APSP");
            for n in (128..=1024).step_by(128) {
                let graph = create_graph(n);

                group.bench_with_input(
                    BenchmarkId::new("CPU warshall-floyd", n),
                    &graph,
                    |bench, graph| {
                        let mut distance = vec![0.; n * n];
                        bench.iter(|| warshall_floyd_cpu(n, &graph, &mut distance));
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("GPU warshall-floyd", n),
                    &graph,
                    |bench, graph| {
                        let size = wgpu_test::warshall_floyd::WarshallFloydKernel::buffer_size(n);
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
                        bench.iter(|| {
                            wgpu_test::warshall_floyd::stage_adjacency_matrix(
                                &device, &queue, &graph, n, &in_buffer,
                            );
                            wf.run(&device, &queue, &in_buffer, &out_buffer, n);
                            device.poll(wgpu::Maintain::Wait);
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("GPU tropical-matmul", n),
                    &graph,
                    |bench, graph| {
                        let (in_buffer, out_buffer) = tm_naive.create_buffer(&device, n);
                        bench.iter(|| {
                            tm_naive.stage_adjacency_matrix(&device, &queue, &graph, n, &in_buffer);
                            tm_naive.run(&device, &queue, &in_buffer, &out_buffer, n);
                            device.poll(wgpu::Maintain::Wait);
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("GPU tropical-matmul-block", n),
                    &graph,
                    |bench, graph| {
                        let (in_buffer, out_buffer) = tm_block.create_buffer(&device, n);
                        bench.iter(|| {
                            tm_block.stage_adjacency_matrix(&device, &queue, &graph, n, &in_buffer);
                            tm_block.run(&device, &queue, &in_buffer, &out_buffer, n);
                            device.poll(wgpu::Maintain::Wait);
                        });
                    },
                );
            }
        }
    })
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::new(5, 0));
    targets = criterion_benchmark
}
criterion_main!(benches);
