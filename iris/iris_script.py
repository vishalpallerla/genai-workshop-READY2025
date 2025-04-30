import glob
import iris
import os
import pandas as pd
from sqlalchemy import create_engine
from iris import ipm
import csv

# switch namespace to the %SYS namespace
iris.system.Process.SetNamespace("%SYS")

# set credentials to not expire
iris.cls('Security.Users').UnExpireUserPasswords("*")

# switch namespace to IRISAPP built by merge.cpf
iris.system.Process.SetNamespace("IRISAPP")

# load ipm package listed in module.xml
#iris.cls('%ZPM.PackageManager').Shell("load /home/irisowner/dev -v")
#assert ipm('load /home/irisowner/dev -v')


# load demo data
engine = create_engine('iris+emb:///')
#list all csv files in the demo data folder
for file in glob.glob('/home/irisowner/dev/data/*.csv'):
    # get the file name without the extension
    print(f"file {file}")
    table_name = os.path.splitext(os.path.basename(file))[0]
    print(f"tablename {table_name}")
    #table_name = file_name.replace(".csv","")
    # load the csv file into a pandas dataframe
    df = pd.read_csv(file)
    # write the dataframe to IRIS
    df.to_sql(table_name, engine, if_exists='replace', index=False, schema="GenAI")

# connection = engine.connect()
# for file in glob.glob('/home/irisowner/dev/data/*.csv'):
#     with open (file, 'r') as f:
#         file_name = os.path.basename(file)
#         reader = csv.reader(f)
#         columns = next(reader) 
#         table_name = file_name.replace(".csv","")
#         query = 'insert into {0}({1}) values ({2})'
        
#         query = query.format(table_name, ','.join(columns), ','.join('?' * len(columns)))
#         print(f"Query - {query}")
#         #cursor = connection.cursor()
#         for data in reader:
#             print(f"data {data}")
#             connection.execute(query, data)
#         connection.commit()
