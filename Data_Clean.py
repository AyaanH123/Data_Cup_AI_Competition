import Json_Read as jr

df1 = jr.create_dataframes('../True North AI')
df2 = jr.create_dataframes2('../True North AI')
df3 = jr.create_dataframes_claim_label('../True North AI')

print(df1)
print(df2)
print(df3)
