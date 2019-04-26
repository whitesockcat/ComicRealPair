import pandas as pd
import os
import random
# ann_path = 'af2019-ksyun-training-20190416/annotations.csv'
# df = pd.read_csv(ann_path, sep=',', header=0)
# # TO dict
# real_dict = {}
# comi_dict = {}
# num_real = 0
# num_all = 23814
# for i in range(num_all):
#     line = df.loc[i]
#     name = line[0]
#     attr = int(line[1])
#     cate = line[2]
#     if attr == 1:
#         num_real += 1
    
#         if cate in real_dict.keys():
#             real_dict[cate].append(name)
#         else:
#             real_dict[cate] = [name]
#     else:
#         if cate in comi_dict.keys():
#             comi_dict[cate].append(name)
#         else:
#             comi_dict[cate] = [name]

# pair_list = []
# for i, cate in enumerate(real_dict.keys()):
#     for real_name in real_dict[cate]:
#         if random.random() > 0.5:
#             same = 1
#             comi_name = random.sample(comi_dict[cate],1)[0]
#         else:
#             same = 0
#             random_cate = cate
#             while random_cate == cate:
#                 random_cate = random.sample(real_dict.keys(), 1)[0]
#             comi_name = random.sample(comi_dict[random_cate],1)[0]  
#         pair_list.append([real_name, comi_name, same])
    
# # print(num_real) 13544

# save = pd.DataFrame(pair_list)
# save.to_csv('pair_list.csv',index=False,sep=',')


# ann_path = 'af2019-ksyun-testA-20190416/list.csv'
# df = pd.read_csv(ann_path, sep=',', header=0)
# print(df.iat[0, 0])
# df = df.ix[:, 1:]
# print(df.iat[0,0])
# pic_dir = 'af2019-ksyun-testA-20190416/images/'
# test = [[0, 0.9], [1, 1], [2, 3]]
# save = pd.DataFrame(test)
# save.to_csv('test.csv',index=False,sep=',', header=['group_id', 'confidence'])