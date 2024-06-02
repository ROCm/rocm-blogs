#ifndef BMM_H
#define BMM_H
#include <torch/types.h>

torch::Tensor bmm_abe_i8(torch::Tensor A,
			 torch::Tensor B,
			 float alpha    
			 );


torch::Tensor bmm_ab_i8_e_f32(torch::Tensor A,
			      torch::Tensor B,
			      float alpha
			      );

#endif
