import pandas as pd
from litellm import completion
import dotenv
import traceback

from diskcache import Cache
cache = Cache()

dotenv.load_dotenv()

path = "/Users/psulin/repos/genai-workshop-READY2025/data/1000_patients_encounters/patient_encounters1.csv"
output_path = "/Users/psulin/repos/genai-workshop-READY2025/data/encounters"
model = "gpt-4.1-mini"


encounter_prompt = """
Generate a clinical encounter note for each patient using the existing data for each encounter record. Vary the notes, don't always include the age, for example.
Some comments can be made about the general health of the patient and previous encounters, but don't always include them.
Add the note to the clinical notes column. The data must be returned in csv format and a bio section.
Only include the PATIENT_ID,ENCOUNTER_ID and CLINICAL_NOTES fields in the CSV response. Clinical notes must be in double quotes.

encounters: 
```csv
{csv_data}
```

Example response:
```csv
PATIENT_ID,ENCOUNTER_ID,CLINICAL_NOTES
1,1,"Something here"
1,2,"A clinical note for encounter 2"
```
"""



encounter_prompt_bak = """
Generate a clinical encounter note for each patient using the existing data for each encounter record.
Add the note to the clinical notes column. The data must be returned in csv format and a bio section.
The CLINICAL_NOTES field must be on it's respective row and in double quotes. There can be no spaces between the csv comma delimiters
and the data. Never use triple double quotes.

Also include a short patient bio based on the encounters and return it only within the ```bio block

encounters: 
```csv
{csv_data}
```

```bio

```

"""

df = pd.read_csv(path)
fields = ['PATIENT_ID', 'FIRST', 'ENCOUNTER_ID', 'DESCRIPTION', 'PATIENT_AGE', 'DESCRIPTION_PROCEDURES', 
          'DESCRIPTION_MEDICATIONS', 'DESCRIPTION_CONDITIONS', 'CLINICAL_NOTES']

df['CLINICAL_NOTES'] = ''
encounters = df[fields]

# Remove the term (procedure) from all DESCRIPTION
encounters['DESCRIPTION'] = encounters['DESCRIPTION'].str.replace(' (procedure)', '')

patients = encounters["PATIENT_ID"].unique()

@cache.memoize()
def get_response(csv_data):
    response = completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": encounter_prompt.format(csv_data=csv_data),
                "temperature": .7,
                "max_tokens": 1500
            }
        ],
    )
    return response

def compute_cost(response):
    
    print(response._hidden_params["response_cost"])
    return response._hidden_params["response_cost"]
     

def extract_section(content, section_name):
    start = content.find(f'```{section_name}')
    end = content.find('```', start + len(f'```{section_name}'))
    section = content[start + len(f'```{section_name}'):end]
    # remove empty lines
    section = section.split('\n')
    section = [line for line in section if line.strip()]
    section = '\n'.join(section)
    return section

# iterate through patients, get all of their encounters, and create a csv
total_cost = 0
encounter_max = 500
for i in range(100, 150):

    try:
        batch = encounters[encounters['PATIENT_ID'] == patients[i]]
        # Include only the first encounter_max encounters
        first = batch.head(encounter_max)
        # If there are more than encounter_max encounters, include the last encounter
        if len(batch) > encounter_max:
            last = batch.tail(1)
            batch = pd.concat([first, last])
            
        batch_str = batch.to_csv(index=False)
        print(batch_str)
        
        response = get_response(batch_str)
        total_cost += compute_cost(response)
        content = response['choices'][0]['message']['content']
        
        file_name = f"{output_path}/{patients[i]}.csv"
        csv_section = extract_section(content, 'csv')
        bio_section = extract_section(content, 'bio')
        
        # Write the individual files
        with open(file_name, 'w') as f:
            f.write(csv_section)
        
        with open(f"{output_path}/{patients[i]}.txt", 'w') as f:
            f.write(bio_section) 
            
        # Write the csv to one file
        with open(f"{output_path}/all_encounters34.csv", 'a') as f:
            
            # Remove header row
            csv_section_list = csv_section.split('\n')[1:]
            csv_section_str = '\n'.join(csv_section_list)
            f.write(csv_section_str + '\n')
        
        # Write the bio to one file
        with open(f"{output_path}/all_bios.txt", 'a') as f:
            f.write(f'{i},"{bio_section}"\n')
        
    except Exception as e:
        print(f"Error processing patient {patients[i]}: {e}")
        traceback.print_exc()
        
    
print(f"Total cost: {total_cost}")
