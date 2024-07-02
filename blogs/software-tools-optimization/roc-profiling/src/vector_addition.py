import torch

def vector_addition():
    A = torch.tensor([1,2,3], device='cuda:0')
    B = torch.tensor([1,2,3], device='cuda:0')

    C = torch.add(A,B)

    return C

if __name__=="__main__":
   print(vector_addition())