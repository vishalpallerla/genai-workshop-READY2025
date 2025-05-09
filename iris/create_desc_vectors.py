#import sentence_transformers
import connect
import pandas as pd
import json 
from sqlalchemy import create_engine, text
from time import sleep
import os 
import iris


table_name = 'GenAI.encounters'
model_dir = "/home/irisowner/dev"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
fields_to_vectorize = ['DESCRIPTION_OBSERVATIONS', 'DESCRIPTION_PROCEDURES', 'DESCRIPTION_MEDICATIONS', 'DESCRIPTION_CONDITIONS', 'CLINICAL_NOTES']
app_namespace = 'IRISAPP'

# TODO - use fields_to_vectorize
new_cols = [
    f"ALTER TABLE {table_name} ADD DESCRIPTION_OBSERVATIONS_Vector VECTOR(FLOAT, 384)",
    f"ALTER TABLE {table_name} ADD DESCRIPTION_PROCEDURES_Vector VECTOR(FLOAT, 384)",
    f"ALTER TABLE {table_name} ADD DESCRIPTION_MEDICATIONS_Vector VECTOR(FLOAT, 384)",
    f"ALTER TABLE {table_name} ADD DESCRIPTION_CONDITIONS_Vector VECTOR(FLOAT, 384)",
    f"ALTER TABLE {table_name} ADD CLINICAL_NOTES_Vector VECTOR(FLOAT, 384)",
]

del_cols = [
    f"ALTER TABLE {table_name} DROP DESCRIPTION_OBSERVATIONS_Vector",
    f"ALTER TABLE {table_name} DROP DESCRIPTION_PROCEDURES_Vector",
    f"ALTER TABLE {table_name} DROP DESCRIPTION_MEDICATIONS_Vector",
    f"ALTER TABLE {table_name} DROP DESCRIPTION_CONDITIONS_Vector",
    f"ALTER TABLE {table_name} DROP CLINICAL_NOTES_Vector"
]


def execute_sql(conn, stmt):
    response = None
    try:
        with conn.cursor() as cursor:
            response = cursor.execute(stmt) 
    except Exception as ex:
        print(ex)

    return response

def update_columns(add_columns: list = [], del_columns: list = []):

    conn = connect.get_connection()

    if del_columns:
        for stmt in del_columns:
            execute_sql(conn, stmt)

    if add_columns:
        for stmt in add_columns:
            print(f"Executing {stmt}")
            execute_sql(conn, stmt)

def add_embedding_config(delete: bool = True):
  
  embed_config_table = '%Embedding.Config' 

  try:
    if delete:
        conn = connect.get_connection()
        del_sql = F"DELETE FROM {embed_config_table} where name = '{model_name}'"
        execute_sql(conn,del_sql)
        conn.close()
  except Exception as ex:
      print(ex)

  model_config = {"modelName":model_name,
                "hfCachePath":model_dir,
                "checkTokenCount": False}
  
  stmt = f"""INSERT INTO {embed_config_table} (Name, Configuration, EmbeddingClass, Description)
  VALUES ('{model_name}',
          '{json.dumps(model_config)}',
          '%Embedding.SentenceTransformers',
          'a small SentenceTransformers embedding model')"""
  
  print(stmt)
  conn = connect.get_connection()
  execute_sql(conn, stmt)

def add_vectors():

    fields = fields_to_vectorize

    conn = connect.get_connection()
    table_data = pd.read_sql(f"Select * from {table_name}", conn)

    flds_with_values = []
    with conn.cursor() as cursor:
        for index, row in table_data.iterrows():
            flds_with_values = []
            for fld_name in fields:
                if row[fld_name] :
                    flds_with_values.append(f"{fld_name}_Vector = Embedding('{row[fld_name][:255]}', '{model_name}')")

            if flds_with_values:
                updates = ','.join(flds_with_values)

                sql = text(f"""
                    UPDATE {table_name} 
                    SET {updates}
                    WHERE ENCOUNTER_ID = {row['ENCOUNTER_ID']}
                """)
                print(sql)
                cursor.execute(sql)

def load_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(f"This is the model that was loaded {model}")

def load_data():
    
    iris.system.Process.SetNamespace("IRISAPP")

    # load demo data
    engine = create_engine('iris+emb:///')
    file = '/home/irisowner/dev/data/encounters.csv'
    table_name = os.path.splitext(os.path.basename(file))[0]
    print(f"tablename {table_name}, filename {file}")

    # load the csv file into a pandas dataframe
    data = pd.read_csv(file)
    print(f"Loaded {len(data)} records")

    # write the dataframe to IRIS
    results = data.to_sql(table_name, engine, if_exists='replace', index=False, schema="GenAI")
    print(f"Results: {results}")

    update_columns(add_columns=new_cols)

    return data

def vectorize_data(data, table_name):
    
    iris.system.Process.SetNamespace(app_namespace)

    # load demo data
    engine = create_engine('iris+emb:///')
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    # Vectorize the data in the cols with string data, the columns named without the _Vector, in batches of 1000 and add to data
    # Vectorize the data in the columns without the _Vector suffix
    batch_size = 200
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        batch = data.iloc[start:end]
        
        # Vectorize each column
        for col in fields_to_vectorize:
            # Ensure all values are strings and non-null
            text_data = batch[col].dropna().apply(lambda x: str(x) if not isinstance(x, str) else x).tolist()
            col_vectors = model.encode(text_data, normalize_embeddings=True)

            vector_col_name = f'{col}_Vector'
            #batch[vector_col_name] = col_vectors.tolist()
            
            # Create a series with the same index as the batch and fill with vectors
            vector_series = pd.Series([None] * len(batch), index=batch.index)
            vector_series.loc[batch[col].notna()] = col_vectors.tolist()
            
            # Assign the vector series back to the DataFrame
            batch[vector_col_name] = vector_series
            #print(batch[vector_col_name].head())
            
            # Update the database with the vector data
            with engine.connect() as conn:
                with conn.begin():
                    for index, row in batch.iterrows():
                        if row[vector_col_name] is not None:
                            sql = text(f"""
                                UPDATE {table_name} 
                                SET {vector_col_name} = TO_VECTOR(:vector)
                                WHERE ENCOUNTER_ID = :encounter_id
                            """)
                            conn.execute(sql, {
                                'vector': str(row[vector_col_name]),
                                'encounter_id': row['ENCOUNTER_ID']
                            })
            
            print(f"Processed batch: {start} to {end} of {len(data)} for {col}")

        
  
if __name__ == '__main__':

    data = load_data()
    # # add_embedding_config(delete=False)
    vectorize_data(data, table_name)
