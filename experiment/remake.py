import torch
import os 

dir = "./"


#진짜
path = "./rdn_x2.pth"

#껍데기
path2 = "./model_best.pt"

weight = torch.load(path)
weight2 = torch.load(path2)

new_dict = {}

# for k, v in weight.items():
#     print(k)
#     new_name = str(k)[6:]
#     new_dict[new_name] =  v
a = 1
for k, v in weight2.items():
    print(k)
    a+=1
    if a > 4 :
        new_dict[k] =  v



# for k, v in weight.items():
#     print(k)
#     a +=1
#     print(a)
# torch.save(new_dict, os.path.join(dir,'gaze_model_best_epoch_1.pt'))

# for k, v in new_dict.items():
#     print(k)