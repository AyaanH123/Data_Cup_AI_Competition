import Json_Read as jr
import pandas as pd

df1 = jr.create_dataframes('../True North AI')
df2 = jr.create_dataframes2('../True North AI')

print(df1)
print(df2)