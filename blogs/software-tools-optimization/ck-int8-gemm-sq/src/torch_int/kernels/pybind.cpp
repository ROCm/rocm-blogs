
#include "include/linear.h"
#include "include/bmm.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("linear_ab_i8_de_f32", &linear_ab_i8_de_f32);
  m.def("linear_relu_abde_i8", &linear_relu_abde_i8);
  m.def("linear_abde_i8", &linear_abde_i8);
  m.def("bmm_abe_i8", &bmm_abe_i8);
  m.def("bmm_ab_i8_e_f32", &bmm_ab_i8_e_f32);
}
