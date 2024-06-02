#ifndef LINEAR_H
#define LINEAR_H
#include <torch/types.h>


torch::Tensor linear_abde_i8(torch::Tensor A,
			     torch::Tensor B,
			     torch::Tensor D,
			     float alpha,
			     float beta
			     );

torch::Tensor linear_ab_i8_de_f32(torch::Tensor A,  
				  torch::Tensor B, 
				  torch::Tensor D, 
				  float alpha,          
				  float beta            
				  );

torch::Tensor linear_relu_abde_i8(torch::Tensor A,  
				  torch::Tensor B, 
				  torch::Tensor D, 
				  float alpha,          
				  float beta            
				  );

#endif
