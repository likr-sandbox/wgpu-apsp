pub fn stage_adjacency_matrix(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    edges: &[(usize, usize)],
    n: usize,
    size: usize,
    dst: &wgpu::Buffer,
) {
    let mut distance = vec![std::f32::INFINITY; n * n];
    for i in 0..n {
        distance[i * n + i] = 0.;
    }
    for &(i, j) in edges.iter() {
        distance[i * n + j] = 1.;
        distance[j * n + i] = 1.;
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
