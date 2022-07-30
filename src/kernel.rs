pub trait Kernel {
    const WORKGROUP_SIZE_X: usize;
    const WORKGROUP_SIZE_Y: usize;
    const WORKGROUP_SIZE_Z: usize;

    fn num_workgroups_x(n: usize) -> usize {
        (n + Self::WORKGROUP_SIZE_X - 1) / Self::WORKGROUP_SIZE_X
    }

    fn num_workgroups_y(n: usize) -> usize {
        (n + Self::WORKGROUP_SIZE_Y - 1) / Self::WORKGROUP_SIZE_Y
    }

    fn num_workgroups_z(n: usize) -> usize {
        (n + Self::WORKGROUP_SIZE_Z - 1) / Self::WORKGROUP_SIZE_Z
    }

    fn num_threads_x(n: usize) -> usize {
        Self::num_workgroups_x(n) * Self::WORKGROUP_SIZE_X
    }

    fn num_threads_y(n: usize) -> usize {
        Self::num_workgroups_y(n) * Self::WORKGROUP_SIZE_Y
    }

    fn num_threads_z(n: usize) -> usize {
        Self::num_workgroups_z(n) * Self::WORKGROUP_SIZE_Z
    }
}
