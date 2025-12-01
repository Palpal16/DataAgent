import os
import json
from Agent.data_agent import SalesDataAgent
from Agent.utils import best_of_n

if __name__ == "__main__":
    
    PREFIX = 'claude'
    i=10

    DATASET_FILE_PATH = f"evaluation/{PREFIX}_dataset.json"
    if os.path.exists(DATASET_FILE_PATH):
        with open(DATASET_FILE_PATH, 'r') as f:
            dataset = json.load(f)
    data = dataset[i]
    agent = SalesDataAgent(
        enable_tracing=True,
        phoenix_endpoint="http://localhost:6006/v1/traces",
        project_name="evaluating-agent",
        model="llama3.2:3b"
    )
    output = agent.run(
        data['prompt'],
        only_lookup=True
    )
    '''
    output = best_of_n(
        agent, 
        data['prompt'],
        expected_csv=data['gt_csv_path'],
        csv_path=data['gen_csv_path'],
        n=3,
        temperature=[0.05,0.15] #Or also single values: 0.1
    )'''
    # Minimal printout
    print(json.dumps({k: v for k, v in output.items() if k != "data"}, indent=2))