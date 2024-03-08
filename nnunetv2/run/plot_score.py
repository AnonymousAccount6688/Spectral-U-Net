import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# wandb.login(key="66b58ac7004a123a43487d7a6cf34ebb4571a7ea")
# run = wandb.init(
#             project="isic2017_case_select",
#             dir=".",
#             name="isic2017_case_select_dif",
#             resume="allow",  # must resume, otherwise crash allow
#             # id=id,
#             # config=config
#         )

df = pd.read_csv("result.csv", header=0)
# print(df.head())

my_score = df['my_socre'].tolist()
unet_score = df['unet_score'].tolist()
unetplus_score = df['unetplus_score'].tolist()
ege_score = df['ege_score'].tolist()

cases = df['filename'].tolist()

print(f"mean, my: {np.mean(my_score)}, unet: {np.mean(unet_score)},"
      f"unetplus: {np.mean(unetplus_score)}, ege: {np.mean(ege_score)}")

diff_df = {"case": [], "diff_1": [], "diff_2": [], "diff_3": []}
for i in range(len(my_score)):

      dif_1 = my_score[i] - unet_score[i]
      dif_2 = my_score[i] - unetplus_score[i]
      dif_3 = my_score[i] - ege_score[i]

      if dif_1>0 and dif_2 > 0 and dif_3 >0:
          diff_df['case'].append(cases[i])
          diff_df['diff_1'].append(dif_1)
          diff_df['diff_2'].append(dif_2)
          diff_df['diff_3'].append(dif_3)

pd.DataFrame(diff_df).to_csv("diff.csv", index=False)


      # wandb.log({"dsc/dif_1": dif_1}, step=i)
      # wandb.log({"dsc/dif_2": dif_2}, step=i)
      # wandb.log({"dsc/dif_3": dif_3}, step=i)
      # wandb.log({"dsc/ege_score": ege_score[i]}, step=i)

      # wandb.log({"dsc/my_score": my_score[i]}, step=i)
      # wandb.log({"dsc/unet_score": unet_score[i]}, step=i)
      # wandb.log({"dsc/unetplus_score": unetplus_score[i]}, step=i)
      # wandb.log({"dsc/ege_score": ege_score[i]}, step=i)

# plt.plot(my_score, label="my")
# plt.plot(unet_score, label="unet")
# plt.plot(unetplus_score, label="unetplus")
# plt.plot(ege_score, label="ege")
# plt.legend()
# plt.show()
