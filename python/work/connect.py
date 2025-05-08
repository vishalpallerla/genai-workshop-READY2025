# import iris

# con_str = "IRIS://iris:1972/IRISAPP"
# user = "superuser"
# pwd = "SYS"
# connection = iris.connect(con_str, user, pwd)

import intersystems_iris.dbapi._DBAPI as iris

config = {
    "hostname": "iris",
    "port": 1972,
    "namespace": "IRISAPP",
    "username": "superuser",
    "password": "SYS",
}

def get_conn_url():
    url = f"iris://{config['username']}@{config['password']}:{config['hostname']}:{config['port']}/{config['namespace']}" 
    return url 

def get_connection():
    connection = iris.connect(**config)
    return connection

def test():
    with iris.connect(**config) as conn:
        with conn.cursor() as cursor:
            cursor.execute("select top 10 encounter_id from GenAI.encounters") 
            for row in cursor:
                print(row)

def load_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(model)

if __name__ == '__main__':
    try:
        test()
    except Exception as e:
        print(e)