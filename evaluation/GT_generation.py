import duckdb
import pandas as pd
import json
import os

agent_questions = [
    "What was the most popular product SKU?",
    "What was the total revenue across all stores?",
    "Which store had the highest sales volume?",
    "Create a bar chart showing total sales by store",
    "What was the average transaction value?",
    "Retrieve the sales made in November 2021 and December 2021 and tell me when more money were earned?",
    "Show me the sales in Nov 2021",
    "Collect sales data for December 2021 and tell which day had the highest sales?",
    "How many products were sold for a promo in May 2023?",
    "Weekly sales in 2021"
]

TRANSACTION_DATA_FILE_PATH = 'data/Store_Sales_Price_Elasticity_Promotions_Data.parquet'
DATASET_FILE_PATH = 'evaluation/query_dataset.json'
table_name = "sales"

df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")

SQL_Generation_Prompt = "" \
"Generate an SQL query based on the prompt. Please just reply with the SQL query and NO MORE, just the query. Really there is no need to create any comment besides the query, that's the only important thing." \
"The prompt is : {prompt}" \
"The available columns are: {columns}. " \
"The table name is: {table_name}. " \
"If you need to use a DATE column with LIKE or pattern matching, first CAST it to VARCHAR like this: CAST(date_column AS VARCHAR) LIKE '%2021-11%' " \
"Return only the SQL query, with no explanations or markdown formatting."

# Load existing dataset if it exists
if os.path.exists(DATASET_FILE_PATH):
    with open(DATASET_FILE_PATH, 'r') as f:
        dataset = json.load(f)
    print(f'\nLoaded existing dataset with {len(dataset)} entries.')
else:
    dataset = []
    print('\nNo existing dataset found. Starting fresh.')


# Loop over all questions and ask user to input SQL for each
for i, prompt in enumerate(agent_questions, 1):
    if any(entry['prompt'] == prompt for entry in dataset):
        print(f'\nSkipping question {i}/{len(agent_questions)} as it already exists in the dataset.')
        continue
    formatted_prompt = SQL_Generation_Prompt.format(prompt=prompt, columns=df.columns.to_list(), table_name=table_name)
    
    print(f'\n{"="*80}')
    print(f'Question {i}/{len(agent_questions)}: {prompt}')
    print(f'{"="*80}')
    print('\nPrompt for LLM:')
    print(formatted_prompt)
    print('\n' + '-'*80)
    
    # Ask user to input the SQL query
    user_sql = input('\nPlease enter the SQL query for this question: ')

    # Data extraction part
    sql_query = user_sql.strip()
    sql_query = sql_query.replace("```sql", "").replace("```", "")
    
    try:
        result = duckdb.sql(sql_query).df()
        result = result.to_string()
        print("\nQuery Result:")
        print(result)
    except Exception as e:
        print(f"\nError executing query: {e}")
   
    # Save to dataset
    dataset.append({
        'prompt': prompt,
        'sql_query': sql_query,
        'ground_truth': result,
    })

# Save dataset as json file
with open('evaluation/query_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)

print(f'\n{"="*80}')
print(f'Dataset completed and saved!')
print(f'Total entries: {len(dataset)}')
print(f'{"="*80}')