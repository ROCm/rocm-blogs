#include <torch/extension.h>
#include <vector>
#include <iostream>

torch::Tensor mlp_forward(  
    torch::Tensor input,  
    torch::Tensor hidden_weights,  
    torch::Tensor hidden_bias,  
    torch::Tensor output_weights,  
    torch::Tensor output_bias) {  
  // Compute the input/hidden layer  
  auto hidden = torch::addmm(hidden_bias, input, hidden_weights.t());  
  hidden = torch::relu(hidden);  
  
  // Compute the output layer  
  auto output = torch::addmm(output_bias, hidden, output_weights.t());   
  
  // Return the output  
  return output;  
    
}  
  
std::vector<torch::Tensor> mlp_backward(  
    torch::Tensor input,  
    torch::Tensor hidden_weights,  
    torch::Tensor hidden_bias,  
    torch::Tensor output_weights,  
    torch::Tensor output_bias,
    torch::Tensor grad_output) {  
  
  // Compute the input/hidden layer
  auto hidden = torch::addmm(hidden_bias, input, hidden_weights.t());
  hidden = torch::relu(hidden);  
  // Compute the output layer  
  auto output = torch::addmm(output_bias, hidden, output_weights.t());  
  // Compute the gradients for output layer
  auto grad_output_weights = torch::mm(grad_output.t(), hidden);
  auto grad_output_bias = torch::sum(grad_output, /*dim=*/0).unsqueeze(0); 
  // Compute the gradients for input/hidden layer using chain rule
  auto grad_hidden = torch::mm(grad_output, output_weights);
  // grad_hidden = grad_hidden
  auto grad_hidden_weights = torch::mm(grad_hidden.t(), input);
  auto grad_hidden_bias = torch::sum(grad_hidden, /*dim=*/0).unsqueeze(0);
  // Compute the gradients for input
  auto grad_input = torch::mm(grad_hidden , hidden_weights);
    
  // Return the gradients  
  return {grad_input, grad_hidden_weights, grad_hidden_bias, grad_output_weights, grad_output_bias};  
}  
  
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  
  m.def("forward", &mlp_forward, "MLP forward");  
  m.def("backward", &mlp_backward, "MLP backward");  
}  
