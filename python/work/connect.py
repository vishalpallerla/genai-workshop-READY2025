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

def get_connection():
    connection = iris.connect(**config)
    return connection

def test():
    with iris.connect(**config) as conn:
        with conn.cursor() as cursor:
            cursor.execute("select top 10 encounter_id from GenAI.encounters") 
            for row in cursor:
                print(row)