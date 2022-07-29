struct WfParams {
  size : u32,
  k : u32,
};

@group(0)
@binding(0)
var<storage, read> wf_buffer_in: array<f32>;
@group(0)
@binding(1)
var<storage, write> wf_buffer_out: array<f32>;
@group(0)
@binding(2)
var<uniform> wf_params: WfParams;

@compute
@workgroup_size(16, 16)
fn warshall_floyd(@builtin(global_invocation_id) global_invocation_id : vec3<u32>) {
  var x : u32 = global_invocation_id.x;
  var y : u32 = global_invocation_id.y;
  var k : u32 = wf_params.k;
  var n : u32 = wf_params.size;
  if (x < n && y < n) {
    wf_buffer_out[y * n + x] = min(
      wf_buffer_in[y * n + x],
      wf_buffer_in[y * n + k] + wf_buffer_in[k * n + x]
    );
  }
}
