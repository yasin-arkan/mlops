import pickle
import pandas as pd

year = input("Year: ")
month = input("Month: ")

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print("Std:", y_pred.std()) 
print("Mean:", y_pred.mean())

# df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

# result = {
  #  'pred': y_pred,
   # 'ride_id': df['ride_id']
#}

#df_result = pd.DataFrame.from_dict(result)

#df_result.to_parquet(
 #   "output",
  #  engine='pyarrow',
   # compression=None,
    #index=False
#)

