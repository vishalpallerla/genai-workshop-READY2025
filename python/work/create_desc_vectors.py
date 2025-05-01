#import sentence_transformers
import connect
import pandas as pd
import json 

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

def add_embedding_config():
  
  model_name = "sentence-transformers/all-MiniLM-L6-v2"
  model_config = {"modelName":model_name,
                "hfCachePath":model_dir,
                "checkTokenCount": True}
  
  stmt = f"""INSERT INTO %Embedding.Config (Name, Configuration, EmbeddingClass, Description)
  VALUES ('{model_name}',
          '{json.dumps(model_config)}',
          '%Embedding.SentenceTransformers',
          'a small SentenceTransformers embedding model')"""
  
  print(stmt)
  conn = connect.get_connection()
  execute_sql(conn, stmt)


def add_vectors(source_col: str, vector_col: str):

    fields = ['DESCRIPTION_OBSERVATIONS', 'DESCRIPTION_PROCEDURES',
              'DESCRIPTION_MEDICATIONS', 'DESCRIPTION_CONDITIONS' ]
    
    conn = connect.get_connection()
    data = pd.read_sql(f"Select {','.join(fields)} from {table_name}")

    # model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
    # embeddings = model.encode(text)
    # with engine.connect() as conn:
    # with conn.begin():
    #     for index, row in df.iterrows():
    #         sql = text("""
    #             INSERT INTO scotch_reviews 
    #             (name, category, review_point, price, description, description_vector) 
    #             VALUES (:name, :category, :review_point, :price, :description, TO_VECTOR(:description_vector))
    #         """)
    #         conn.execute(sql, {
    #             'name': row['name'], 
    #             'category': row['category'], 
    #             'review_point': row['review.point'], 
    #             'price': row['price'], 
    #             'description': row['description'], 
    #             'description_vector': str(row['description_vector'])
    #         })
    
if __name__ == '__main__':
    #update_columns(add_columns=new_cols, del_columns=del_cols)
    add_embedding_config()
    





