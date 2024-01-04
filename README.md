Setupvirtual environment for the Python project:

1. Install virtualenv (if not already installed):
```
pip install virtualenv
```
2. Create a virtual environment:
```
virtualenv my_project_env
```
This command creates a new directory named my_project_env that contains a self-contained Python environment.

3. Activate the virtual environment:
```
source my_project_env/bin/activate
```
Your command prompt or terminal should now indicate that you are in the virtual environment.

4. Install packages within the virtual environment:
Now, you can use pip to install packages, and they will be installed only within the virtual environment.
```
pip install package_name
```
5. Save project dependencies:
To save the project dependencies, you can use a requirements.txt file. Inside your project directory, activate the virtual environment and then run:
```
pip freeze > requirements.txt
```
This command writes the names and versions of installed packages to the requirements.txt file. You can later use this file to recreate the virtual environment on another machine or to deploy your project.

6. Deactivate the virtual environment:
When you're done working on your project, deactivate the virtual environment:
```
deactivate
```
7. Using the requirements.txt file to recreate the environment:
To recreate the virtual environment on another machine or for deployment, you can use the following commands:
```
virtualenv my_project_env
```

# Activate the virtual environment
```
source my_project_env/bin/activate
```

# Install dependencies from requirements.txt
```
pip install -r requirements.txt
```
This process ensures that your project dependencies are isolated from the global environment and can be easily reproduced on different systems.

# Run linter
```
flake8
```

#Run fix automatically lint issues
```
autopep8 --in-place --aggressive --aggressive src/*.py
```

# Run test local env

1. run the local server with file 'src/40_api.py'
2. Find in the termimal "Running on http://{some port}" (usually it's http://127.0.0.1:5000)
3. Open Postman create POST request with url which you found in step 2 
4. Add body with format 'raw' and put there some data, based on which you want to predict some info and hit the run button
For exaple 
```
[{
    "TV": 230.1,
    "radio": 37.8,
    "newspaper": 69.2
}]
```
5. Stop the server with Ctr+C once you finished work with it