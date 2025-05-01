## Issues encountered with Environment
1. Codespaces seems very finicky and unpredictable in certain features
   1. Port forwarding 
      1. Sometimes works and sometimes not with same config
      2. With different config but same port forwarding will not work
      3. Even when port is forwarded sometimes service seems to not be available
   2. Sometime what works in VsCode as DevContainer does not work in Codespaces - which means you have to change and deploy to troubleshoot and that takes at least 5 minutes. Since there is really no docs on why this should happen this process is partially trial and error making it a very slow process.
2. DevContainers, Docker-compose and docker 
   1. All have implicit configuration dependencies that are very hard to know - lots of magic strings that need to work together across the config files. For example, the volume mount in docker compose must match the workspaceMount.target and usually workspaceFolder in .devcontainer.json
   2. Rebuilding with --no-cache with any change except for in docker-compose.yml is usually necessary
3. IRIS python
   1. Official db-api drivers might not work with many of the other community libs like iris-alchemy, iris-llama, iris-langchain
   2. Official driver throws ssl error

## Notes
1. Using the db-api driver from the community
   1. https://github.com/intersystems-community/intersystems-irispython/releases/download/3.9.2/intersystems_iris-3.9.2-py3-none-any.whl
2. Existing data load uses pandas, which creates the GenAI.encounters table from the CSV
   