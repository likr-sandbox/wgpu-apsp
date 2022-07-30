use crate::{buffer::stage_adjacency_matrix, kernel::Kernel};

pub trait TropicalMatmulKernel: Kernel {
    fn pipeline(&self) -> &wgpu::ComputePipeline;

    fn bind(
        &self,
        device: &wgpu::Device,
        in_buffer: &wgpu::Buffer,
        out_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bind_group_layout = self.pipeline().get_bind_group_layout(0);
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

    fn run(&self, encoder: &mut wgpu::CommandEncoder, bind_group: &wgpu::BindGroup, n: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        pass.set_pipeline(&self.pipeline());
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(
            Self::num_workgroups_x(n) as u32,
            Self::num_workgroups_y(n) as u32,
            1,
        );
    }

    fn buffer_size(n: usize) -> usize {
        Self::num_threads_x(n) * Self::num_threads_y(n) * 4
    }
}

pub struct NaiveTropicalMatmulKernel {
    pipeline: wgpu::ComputePipeline,
}

impl NaiveTropicalMatmulKernel {
    pub fn new(device: &wgpu::Device) -> NaiveTropicalMatmulKernel {
        let module = device.create_shader_module(wgpu::include_wgsl!("tropical_matmul.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            module: &module,
            layout: None,
            entry_point: "tropical_matmul",
        });
        NaiveTropicalMatmulKernel { pipeline }
    }
}

impl Kernel for NaiveTropicalMatmulKernel {
    const WORKGROUP_SIZE_X: usize = 1;
    const WORKGROUP_SIZE_Y: usize = 1;
    const WORKGROUP_SIZE_Z: usize = 1;
}

impl TropicalMatmulKernel for NaiveTropicalMatmulKernel {
    fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

pub struct BlockedTropicalMatmulKernel {
    pipeline: wgpu::ComputePipeline,
}

impl BlockedTropicalMatmulKernel {
    pub fn new(device: &wgpu::Device) -> BlockedTropicalMatmulKernel {
        let module = device.create_shader_module(wgpu::include_wgsl!("tropical_matmul_block.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            module: &module,
            layout: None,
            entry_point: "tropical_matmul",
        });
        BlockedTropicalMatmulKernel { pipeline }
    }
}

impl Kernel for BlockedTropicalMatmulKernel {
    const WORKGROUP_SIZE_X: usize = 16;
    const WORKGROUP_SIZE_Y: usize = 16;
    const WORKGROUP_SIZE_Z: usize = 1;
}

impl TropicalMatmulKernel for BlockedTropicalMatmulKernel {
    fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

pub struct TropicalMatmul<K: TropicalMatmulKernel = BlockedTropicalMatmulKernel> {
    kernel: K,
}

impl<K: TropicalMatmulKernel> TropicalMatmul<K> {
    pub fn new(kernel: K) -> TropicalMatmul<K> {
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
        let size = K::buffer_size(n);
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
            let params = vec![n as u32, K::num_threads_x(n) as u32];
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

    pub fn stage_adjacency_matrix(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        edges: &[(usize, usize)],
        n: usize,
        dst: &wgpu::Buffer,
    ) {
        stage_adjacency_matrix(
            device,
            queue,
            edges,
            K::num_threads_x(n),
            K::buffer_size(n),
            dst,
        );
    }

    pub fn create_buffer(&self, device: &wgpu::Device, n: usize) -> (wgpu::Buffer, wgpu::Buffer) {
        let size = K::buffer_size(n);
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
        (in_buffer, out_buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn init() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap()
    }

    async fn test_tropical_matmul<K: TropicalMatmulKernel>(
        kernel: K,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let n = 100usize;
        let tm = TropicalMatmul::new(kernel);
        let (in_buffer, out_buffer) = tm.create_buffer(&device, n);

        let edges = (1..n).map(|i| (i - 1, i)).collect::<Vec<_>>();
        tm.stage_adjacency_matrix(&device, &queue, &edges, n, &in_buffer);
        tm.run(&device, &queue, &in_buffer, &out_buffer, n);
        let result = crate::buffer::download_distance_matrix(&device, &queue, &out_buffer)
            .await
            .unwrap();

        for i in 0..n {
            for j in 0..n {
                let d = (i as f32 - j as f32).abs();
                assert_eq!(result[i * K::num_threads_x(n) + j], d);
            }
        }
    }

    #[tokio::test]
    async fn test_tropical_matmul_naive() {
        let (device, queue) = init().await;
        test_tropical_matmul(NaiveTropicalMatmulKernel::new(&device), &device, &queue).await;
    }

    #[tokio::test]
    async fn test_tropical_matmul_block() {
        let (device, queue) = init().await;
        test_tropical_matmul(BlockedTropicalMatmulKernel::new(&device), &device, &queue).await;
    }
}
