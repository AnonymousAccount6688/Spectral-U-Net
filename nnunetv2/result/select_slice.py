import pandas as pd
import numpy as np

df = pd.read_csv("slice_result.csv", header=0)

names = []
diff = []
for i in range(len(df)):
    method1_mean = df.loc[i, 'method1_mean']
    method2_mean = df.loc[i, 'method2_mean']
    method3_mean = df.loc[i, 'method3_mean']
    method4_mean = df.loc[i, 'method4_mean']

    # and method2_mean > method1_mean \
    if method4_mean > method3_mean and method4_mean > method2_mean \
            and method1_mean > 0.7 and \
            2 > (np.array([df.loc[i, 'method1_irf'],
                      df.loc[i, 'method1_srf'],
                      df.loc[i, 'method1_ped']])==1).sum():
        names.append(df.loc[i, 'name'])
        diff.append(method4_mean-method3_mean)
        print(df.loc[i, 'name'], method4_mean-method3_mean)


# pd.DataFrame({"name": names, "diff": diff}).to_csv("slices.csv", index=False)