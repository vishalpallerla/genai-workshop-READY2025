import iris
from create_desc_vectors import load_data, vectorize_data, load_model

# switch namespace to the %SYS namespace
iris.system.Process.SetNamespace("%SYS")

# set credentials to not expire
iris.cls('Security.Users').UnExpireUserPasswords("*")

# switch namespace to IRISAPP built by merge.cpf
iris.system.Process.SetNamespace("IRISAPP")

# load ipm package listed in module.xml
#iris.cls('%ZPM.PackageManager').Shell("load /home/irisowner/dev -v")
assert iris.ipm('load /home/irisowner/dev -v')

if __name__ == '__main__':

    table_name = "GenAI.encounters"
    data = load_data()
    load_model()
    #add_embedding_config(delete=False)
    vectorize_data(data, table_name)