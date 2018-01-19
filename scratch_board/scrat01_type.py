import torch as tt

print(tt.FloatTensor().type())
tt.set_default_tensor_type('torch.cuda.FloatTensor')
print(tt.Tensor().type())
