import pandas as pd
import os
import pytz
import datetime
from maps import observations_short_map, conditions_short_map, medications_short_map, procedures_short_map, can_be_removed_observes

def convert_time_to_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    # ensure column is datettime type
    df[time_col] = pd.to_datetime(df[time_col])
    df[time_col] = df[time_col].dt.date
    return df

def load_csvs_to_pandas(dirpath: str, filenames: list[str],
                        from_hugging = True) -> dict[str, pd.DataFrame]:
    df_dict = {}
    for filename in filenames:
        if from_hugging:
            filepath = load_file(filename, dirpath)
        else:
            filepath = os.path.join(dirpath, filename)
        print(f"Filepath={filepath}")
        df = pd.read_csv(filepath)
        #remove the .csv from filename for key
        key = filename[:-4]
        df_dict[key] = df
    return df_dict

def create_id_map(df: pd.DataFrame, id_col: str) -> dict[str, int]:
    """ Create a map from id to the index"""
    # Create a column that contains the index data and convert to dict
    df = df.reset_index()
    map = dict(zip(df[id_col], df['index']))
    return map

def replace_complex_ids(df: pd.DataFrame, id_map: dict[str, int], col_name: str) -> pd.DataFrame:
    """
    Replace complex ids with integer indices
    """
    df[col_name] = df[col_name].map(id_map)
    return df

def flatten_rows_to_string(df, group_column, value_column, separator=';'):
    """
    Flattens rows within groups into a single string in a new column.

    Args:
      df: The Pandas DataFrame.
      group_column: The column to group by.
      value_column: The column containing values to concatenate.
      separator: The string separator for concatenating values (default: ',').

    Returns:
      A new DataFrame with an additional column containing the flattened strings.
    """
    # Group by the specified column and aggregate with join
    grouped = df.groupby(group_column)[value_column].agg(lambda x: separator.join(map(str, x))).reset_index()

    # Rename the aggregated column
    #grouped.rename(columns={value_column: f'{value_column}_flat'}, inplace=True)

    return grouped

def count_non_null_pairs(df: pd.DataFrame, col1: str, col2: str) -> int:
    # Find the number of times where col1 is not null and col2 is null as a percent of total rows
    return len(df[(df[col1].notnull()) & (df[col2].isnull())]) / len(df)

def get_unique_vals(df: pd.DataFrame, col: str, col2: str) -> set[str]:
    return set(df[col,col2].unique())

def process_table(df_dict: dict[str, pd.DataFrame], merged_df: pd.DataFrame, table_name: str,
                  group_column: str, value_columns: list[str], prefix: str = None) -> pd.DataFrame:
    """
    Process a specified table by filtering, concatenating specified columns,
    flattening rows, and merging with the main DataFrame.

    Args:
        df_dict: Dictionary of DataFrames containing the tables.
        merged_df: The main DataFrame to merge the table into.
        table_name: The name of the table to process.
        prefix: The prefix to add to the concatenated description.
        group_column: The column to group by.
        value_columns: The columns containing values to concatenate.

    Returns:
        The merged DataFrame with the specified table processed and added.
    """

    # Filter out all Surveys if table is observations
    table = df_dict[table_name]
    if table_name == 'observations':
        df = table[table['CATEGORY'] != 'survey']
    else:
        df = df_dict[table_name]

    # Concatenate specified columns
    if prefix:
      df['DESCRIPTION'] = f"{prefix}: " + df[value_columns].astype(str).agg(' '.join, axis=1)
    else:
      df['DESCRIPTION'] = df[value_columns].astype(str).agg(' '.join, axis=1)

    # Flatten rows within groups into a single string in a new column
    df = flatten_rows_to_string(df, group_column, 'DESCRIPTION', ', ')

    # Drop columns that are no longer needed
    df = df[[group_column, 'DESCRIPTION']]

    # Merge with the main DataFrame and drop the group column used for joining
    merged_df = pd.merge(merged_df, df, left_on='ENCOUNTER_ID', right_on=group_column, how='left', suffixes=('', f'_{table_name.upper()}'))
    merged_df = merged_df.drop(columns=[group_column])
    return merged_df

def create_normalized_encounter(df_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a normalized encounter DataFrame from the given CSVs
    Should include the patient id, all fields from encounter, and the Description field
    from each table concatenated in a single column per encounter

    """
    # First create a normalized patient and encounter DataFrame, only including
    # the patient id and all fields from encounter
    patient_df = df_dict['patients'][['Id','BIRTHDATE']].set_index('Id')
    encounter_df = df_dict['encounters']
    encounter_df = encounter_df.drop(columns=['ORGANIZATION', 'PROVIDER', 'PAYER'])

    # REMOVE ME
    # Use to see small sample file in Colab
    #patient_df = patient_df.head(10)

    merged_df = pd.merge(patient_df, encounter_df, left_index=True, right_on='PATIENT')

    # Change name of ID to Encounter ID and Patient to Patient ID
    merged_df = merged_df.rename(columns={'Id': 'ENCOUNTER_ID', 'PATIENT': 'PATIENT_ID'})

    # Set patient Age
    merged_df['START'] = pd.to_datetime(merged_df['START'], utc=True)
    merged_df['BIRTHDATE'] = pd.to_datetime(merged_df['BIRTHDATE'], utc=True)

    # df['Date_of_Birth'].apply(calculate_age)
    merged_df['PATIENT_AGE'] = ((merged_df['START'] - merged_df['BIRTHDATE']).dt.days / 365.25).astype(int)

    # Synthea birthdates don't make any sense so let's nudge them to be a a little better
    merged_df['PATIENT_AGE'] = merged_df['PATIENT_AGE'].clip(lower=3)

    print(f"total patients: {len(patient_df)}")
    print(f"total patient encounters: {len(merged_df)}")

    # Remove unneccessary observations with can_be_removed_observes list
    observes = df_dict['observations']
    observes = observes[observes['CATEGORY'] != 'survey']
    observes = observes[~observes['DESCRIPTION'].isin(can_be_removed_observes)]
    df_dict['observations'] = observes

    # Merge OBSERVATIONS
    # Map the description to the shorthand using observations_short_map
    df_dict['observations']['DESCRIPTION'] = df_dict['observations']['DESCRIPTION'].map(observations_short_map)
    merged_df = process_table(df_dict, merged_df, 'observations', 'ENCOUNTER', ['DESCRIPTION', 'VALUE', 'UNITS'])

    # Merge CONDITIONS
    # Map the description to the shorthand using conditions_short_map
    df_dict['conditions']['DESCRIPTION'] = df_dict['conditions']['DESCRIPTION'].map(conditions_short_map)
    merged_df = process_table(df_dict, merged_df, 'conditions', 'ENCOUNTER', ['DESCRIPTION'])

    # Merge medications
    # Map the description to the shorthand using medications_short_map
    df_dict['medications']['DESCRIPTION'] = df_dict['medications']['DESCRIPTION'].map(medications_short_map)
    merged_df = process_table(df_dict, merged_df, 'medications', 'ENCOUNTER', ['DESCRIPTION'])

    # Merge procedures
    # Map the description to the shorthand using procedures_short_map
    df_dict['procedures']['DESCRIPTION'] = df_dict['procedures']['DESCRIPTION'].map(procedures_short_map)
    merged_df = process_table(df_dict, merged_df, 'procedures', 'ENCOUNTER', ['DESCRIPTION'])

    return merged_df

def get_all_values(df: pd.DataFrame, col: str) -> set[str]:
    return set(df[col].unique())

if __name__ == '__main__':
    
    path = '/Users/psulin/repos/genai-workshop-READY2025/data/1000_patients_encounters'
    output_path = '/Users/psulin/repos/genai-workshop-READY2025/data/1000_patients_encounters'
    output_file = 'patient_encounters1'
    
    filenames = ['patients.csv', 'encounters.csv', 'observations.csv', 'conditions.csv',
                'medications.csv', 'procedures.csv']

    df_dict = load_csvs_to_pandas(path, filenames, from_hugging=False)
    normalized_encounter_df = create_normalized_encounter(df_dict)

    # Convert time columns to datetime
    normalized_encounter_df = convert_time_to_datetime(normalized_encounter_df, 'START')
    normalized_encounter_df = convert_time_to_datetime(normalized_encounter_df, 'STOP')

    # Map GUIDs to ints
    patient_id_map = create_id_map(df_dict['patients'], 'Id')
    encounter_id_map = create_id_map(df_dict['encounters'], 'Id')

    final_df = replace_complex_ids(normalized_encounter_df, patient_id_map, 'PATIENT_ID')
    final_df = replace_complex_ids(final_df, encounter_id_map, 'ENCOUNTER_ID')

    # Save final df
    final_df.to_csv(f'{output_path}/{output_file}.csv', index=False, date_format='%Y-%m-%d')
    final_df.to_json(f'{output_path}/{output_file}.json', orient='records', lines=True, date_format='%Y-%m-%d')

