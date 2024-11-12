import os
from datetime import datetime,timedelta
import pandas as pd
start_date=datetime.strptime('2015-01-01','%Y-%m-%d')
end_date=datetime.strptime('2024-03-10','%Y-%m-%d')
current_date=start_date
concatenated_df=pd.DataFrame()
while current_date<=end_date:
    file_name=current_date.strftime("%Y-%m-%d.csv")
    print(file_name)
    if os.path.exists(file_name):
        df=pd.read_csv(file_name)
        column_name=file_name.split('.')[0]
        df["Date"]=column_name
        concatenated_df=pd.concat([df,concatenated_df],join='outer',sort=True)
        # Move to the next date
    current_date += timedelta(days=1)
concatenated_df.to_csv('SortedData.csv',index=False)