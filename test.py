import torch

tensor_0 = torch.arange(0, 24).view(2, 3, 4)
print(tensor_0)

index = [[[1,1,1,1],
          [1,1,1,1],
          [0,0,0,0]],
         [[0,0,0,0],
         [0,0,0,0],
         [0,0,0,0]]]
# index = [[[1,1,1,1],
#           [1,1,1,1],
#           [0,0,0,0]]]
index = torch.tensor(index)
tensor_1 = torch.gather(tensor_0,0,index)
print(tensor_1)

# index = [[[1,1,1,1]]]
# index = torch.tensor(index)
tensor_2 = torch.gather(tensor_1,0,index)
print(tensor_2)