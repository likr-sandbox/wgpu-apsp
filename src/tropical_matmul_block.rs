pub struct TropicalMatmulKernel {
    pipeline: wgpu::ComputePipeline,
}

impl TropicalMatmulKernel {
    const WORKGROUP_SIZE_X: usize = 16;
    const WORKGROUP_SIZE_Y: usize = 16;

    pub fn new(device: &wgpu::Device) -> TropicalMatmulKernel {
        let module = device.create_shader_module(wgpu::include_wgsl!("tropical_matmul_block.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            module: &module,
            layout: None,
            entry_point: "tropical_matmul",
        });
        TropicalMatmulKernel { pipeline }
    }

    pub fn bind(
        &self,
        device: &wgpu::Device,
        in_buffer: &wgpu::Buffer,
        out_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        })
    }

    pub fn run(&self, encoder: &mut wgpu::CommandEncoder, bind_group: &wgpu::BindGroup, n: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(
            Self::num_workgroups_x(n) as u32,
            Self::num_workgroups_y(n) as u32,
            1,
        );
    }

    pub fn num_workgroups_x(n: usize) -> usize {
        (n + Self::WORKGROUP_SIZE_X - 1) / Self::WORKGROUP_SIZE_X
    }

    pub fn num_workgroups_y(n: usize) -> usize {
        (n + Self::WORKGROUP_SIZE_Y - 1) / Self::WORKGROUP_SIZE_Y
    }

    pub fn stride_x(n: usize) -> usize {
        Self::num_workgroups_x(n) * Self::WORKGROUP_SIZE_X
    }

    pub fn stride_y(n: usize) -> usize {
        Self::num_workgroups_y(n) * Self::WORKGROUP_SIZE_Y
    }

    pub fn buffer_size(n: usize) -> usize {
        Self::stride_x(n) * n * 4
    }
}

pub struct TropicalMatmul {
    kernel: TropicalMatmulKernel,
}

impl TropicalMatmul {
    pub fn new(device: &wgpu::Device) -> TropicalMatmul {
        let kernel = TropicalMatmulKernel::new(device);
        TropicalMatmul { kernel }
    }

    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        in_buffer: &wgpu::Buffer,
        out_buffer: &wgpu::Buffer,
        n: usize,
    ) {
        let size = TropicalMatmulKernel::buffer_size(n);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self
            .kernel
            .bind(&device, &in_buffer, &out_buffer, &params_buffer);

        let mut params_buffer_staging = wgpu::util::StagingBelt::new(8);
        let mut k = 1;
        while k < n {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let params = vec![n as u32, TropicalMatmulKernel::stride_x(n) as u32];
            params_buffer_staging
                .write_buffer(
                    &mut encoder,
                    &params_buffer,
                    0,
                    std::num::NonZeroU64::new(8).unwrap(),
                    &device,
                )
                .copy_from_slice(bytemuck::cast_slice(&params));
            params_buffer_staging.finish();
            if k != 1 {
                encoder.copy_buffer_to_buffer(&out_buffer, 0, &in_buffer, 0, size as u64);
            }
            self.kernel.run(&mut encoder, &bind_group, n);
            queue.submit(Some(encoder.finish()));
            params_buffer_staging.recall();
            k *= 2;
        }
    }
}

pub fn stage_adjacency_matrix(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    edges: &[(usize, usize)],
    n: usize,
    dst: &wgpu::Buffer,
) {
    let size = TropicalMatmulKernel::buffer_size(n);
    let stride = TropicalMatmulKernel::stride_x(n);
    let mut distance = vec![std::f32::INFINITY; stride * n];
    for i in 0..n {
        distance[i * stride + i] = 0.;
    }
    for &(i, j) in edges.iter() {
        distance[i * stride + j] = 1.;
        distance[j * stride + i] = 1.;
    }

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let mut staging = wgpu::util::StagingBelt::new(size as u64);
    staging
        .write_buffer(
            &mut encoder,
            &dst,
            0,
            std::num::NonZeroU64::new(size as u64).unwrap(),
            &device,
        )
        .copy_from_slice(bytemuck::cast_slice(&distance));
    staging.finish();
    queue.submit(Some(encoder.finish()));
}

pub async fn download_distance_matrix(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
) -> Option<Vec<f32>> {
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    wgpu::util::DownloadBuffer::read_buffer(&device, &queue, &src.slice(..), move |result| {
        let buffer = result.unwrap();
        let result = bytemuck::cast_slice::<u8, f32>(&buffer).to_vec();
        sender.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    receiver.receive().await
}

#[tokio::test]
async fn test_tropical_matmul_block() {
    let instance = wgpu::Instance::new(wgpu::Backends::VULKAN);
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

    let n = 512usize;
    let size = TropicalMatmulKernel::buffer_size(n);
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

    let edges = (1..n).map(|i| (i - 1, i)).collect::<Vec<_>>();
    stage_adjacency_matrix(&device, &queue, &edges, n, &in_buffer);
    TropicalMatmul::new(&device).run(&device, &queue, &in_buffer, &out_buffer, n);
    let result = download_distance_matrix(&device, &queue, &out_buffer)
        .await
        .unwrap();

    for i in 0..n {
        for j in 0..n {
            let d = (i as f32 - j as f32).abs();
            assert_eq!(result[i * TropicalMatmulKernel::stride_x(n) + j], d);
        }
    }
}
