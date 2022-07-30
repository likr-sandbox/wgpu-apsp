struct Params {
  n : u32,
  stride : u32,
};

@group(0)
@binding(0)
var<storage, read> buffer_in: array<f32>;
@group(0)
@binding(1)
var<storage, write> buffer_out: array<f32>;
@group(0)
@binding(2)
var<uniform> params: Params;

@compute
@workgroup_size(1, 1)
fn tropical_matmul(
  @builtin(global_invocation_id) global_invocation_id : vec3<u32>,
) {
  var x : u32 = global_invocation_id.x;
  var y : u32 = global_invocation_id.y;
  var s : f32 = 1000.;
  var k : u32 = 0u;
  var n : u32 = params.n;
  var stride : u32 = params.stride;
  loop {
    if (k >= n) {
      break;
    }
    s = min(s, buffer_in[y * n + k] + buffer_in[k * n + x]);
    k = k + 1u;
  }
  buffer_out[y * n + x] = s;
}
