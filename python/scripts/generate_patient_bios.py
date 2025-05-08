import pandas as pd
from litellm import completion
import dotenv

dotenv.load_dotenv()

path = "/Users/psulin/repos/genai-workshop-READY2025/data/1000_patients_encounters/patient_encounters.csv"
model = "gpt-4.1-mini"

bio_prompt = """
Generate a patient bio for each patient. First create a name for the patient and then write a bio. The bio should include these components:
demographics(urban, rural, suburban), socio-economic status, types of travel they used, family structure and relations, social support, and how these
things impact what the patient eats, and their levels of exercise, stress, and sleep. 
This should be in the form of a story, not a list of facts or generic descriptions, but personal and unique. It should be 1-2 paragraphs in length.
Respond in csv format. Vary the stories for each patient along all components. Don't make them all have wonderful lives - be realistic.

patients: 
{patients}
"""

df = pd.read_csv(path)
patients = df[["PATIENT_ID", "BIRTHDATE"]].groupby("PATIENT_ID").first()

# Calculate age
patients['AGE'] = (pd.Timestamp.now() - pd.to_datetime(patients['BIRTHDATE'])).dt.days / 365.25
patients['AGE'] = patients['AGE'].astype(int)
patients['AGE'] = patients['AGE'].clip(upper=103)

patients.info()
print(patients.head())

# print(patients.head())
def get_patient_bio(patients):
    response = completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": bio_prompt.format(patients=patients),
                "temperature": 1
            }
        ],
    )
    return response

def compute_cost(response):
    cost = response['usage']['total_tokens'] / 1000 * 0.0001
    print(f"Cost: ${cost}")

set_size = 5
# iterate through patients in sets of 10
for i in range(0, len(patients), set_size):
    batch = patients.iloc[i:i+set_size]
    batch['bio'] = ''
    # convert into csv formatted string but not a csv file
    batch_str = batch.to_csv(index=False)
    print(batch_str)
    response = get_patient_bio(batch_str)
    compute_cost(response)
    print(response['choices'][0]['message']['content'])
    break
