import pandas as pd
df = pd.read_json(r'C:/Users/dhruve/Desktop/TrueNorthAi/train.json')
df.set_index('id', inplace=True)
print(df)
