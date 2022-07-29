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

var<workgroup> a_local : array<f32, 256>;
var<workgroup> b_local : array<f32, 256>;
var<workgroup> c_local : array<f32, 256>;

@compute
@workgroup_size(16, 16)
fn tropical_matmul(
  @builtin(global_invocation_id) global_invocation_id : vec3<u32>,
  @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
  @builtin(workgroup_id) workgroup_id : vec3<u32>,
) {
  var x : u32 = global_invocation_id.x;
  var y : u32 = global_invocation_id.y;
  var x_local : u32 = local_invocation_id.x;
  var y_local : u32 = local_invocation_id.y;
  var n : u32 = params.n;
  var stride : u32 = params.stride;

  c_local[y_local * 16u + x_local] = 1000000.;
  var k : u32 = u32(0);

  loop {
    if (16u * k >= n) {
      break;
    }
    workgroupBarrier();
    a_local[y_local * 16u + x_local] = buffer_in[(16u * k + y_local) * stride + (16u * workgroup_id.x + x_local)];
    b_local[y_local * 16u + x_local] = buffer_in[(16u * workgroup_id.y + y_local) * stride + (16u * k + x_local)];
    workgroupBarrier();
    var z : u32 = u32(0);
    loop {
      if (z >= 16u) {
        break;
      }
      c_local[y_local * 16u + x_local] = min(
        c_local[y_local * 16u + x_local],
        a_local[y_local * 16u + z] + b_local[z * 16u + x_local]
      );
      z = z + 1u;
    }
    k = k + 1u;
  }
  buffer_out[y * stride + x] = c_local[y_local * 16u + x_local];
}
