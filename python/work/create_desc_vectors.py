#import sentence_transformers
import connect
import pandas as pd
import json 
from sqlalchemy import create_engine, text

'''
INSERT INTO %Embedding.Config (Name, Configuration, EmbeddingClass, VectorLength, Description)
  VALUES ('my-openai-config', 
          '{"apiKey":"<api key>", 
            "sslConfig": "llm_ssl", 
            "modelName": "text-embedding-3-small"}',
          '%Embedding.OpenAI', 
          1536,  
          'a small embedding model provided by OpenAI')

EMBEDDING(model,source)

# create the model and form the embeddings
model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(text)
'''

table_name = 'GenAI.encounters'
model_dir = "/home/irisowner/dev"
model_name = "sentence-transformers/all-MiniLM-L6-v2"

new_cols = [
    f"ALTER TABLE {table_name} ADD DESCRIPTION_OBSERVATIONS_Vector VECTOR(FLOAT, 384)",
    f"ALTER TABLE {table_name} ADD DESCRIPTION_PROCEDURES_Vector VECTOR(FLOAT, 384)",
    f"ALTER TABLE {table_name} ADD DESCRIPTION_MEDICATIONS_Vector VECTOR(FLOAT, 384)",
    f"ALTER TABLE {table_name} ADD DESCRIPTION_CONDITIONS_Vector VECTOR(FLOAT, 384)",
]

del_cols = [
    f"ALTER TABLE {table_name} DROP DESCRIPTION_OBSERVATIONS_Vector",
    f"ALTER TABLE {table_name} DROP DESCRIPTION_PROCEDURES_Vector",
    f"ALTER TABLE {table_name} DROP DESCRIPTION_MEDICATIONS_Vector",
    f"ALTER TABLE {table_name} DROP DESCRIPTION_CONDITIONS_Vector"
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
        for stmt in add_columns:
            execute_sql(conn, stmt)

    if add_columns:
        for stmt in add_columns:
            execute_sql(conn, stmt)

def add_embedding_config(delete: bool = True):
  
  embed_config_table = '%Embedding.Config' 
  if delete:
      conn = connect.get_connection()
      del_sql = F"DELETE FROM {embed_config_table} where model_name = '{model_name}'"
      execute_sql(conn,del_sql)
      conn.close()

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

    fields = ['DESCRIPTION_OBSERVATIONS', 'DESCRIPTION_PROCEDURES',
              'DESCRIPTION_MEDICATIONS', 'DESCRIPTION_CONDITIONS']
    
    fields_str = ','.join(fields)

    vector_fields_str = ','.join([f"{fld}_Vector" for fld in fields])
    #vector_flds_embed =','.join(["Embedding({row['{fld}']}" for fld in fields])

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
    
if __name__ == '__main__':
    update_columns(add_columns=new_cols, del_columns=del_cols)
    add_embedding_config()
    add_vectors()
    





