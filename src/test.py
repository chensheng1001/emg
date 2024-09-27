import pandas as pd

store = pd.HDFStore('/workspace/processed_data/data_list.hf', mode='r')
data = store['data']  # 读取 'data' 表
store.close()
print(data)
