import os
import json
from Agent.utils import *

PREFIX = 'llama3-gpt'

def make_csvs(dataset):
    for i in range(len(dataset)):
        csv_path = f"evaluation/csv_queries/{PREFIX}_{i}_gen.csv"
        result_rows = text_to_csv(dataset[i]['gen_data'])
        save_csv(result_rows, csv_path)
        dataset[i]["gen_csv_path"]=csv_path
        csv_path = f"evaluation/csv_queries/{PREFIX}_{i}_gt.csv"
        result_rows = text_to_csv(dataset[i]['gt_data'])
        save_csv(result_rows, csv_path)
        if 'annotated_accuracy' in dataset[i]:
            dataset[i].pop('annotated_accuracy')
        dataset[i]["gt_csv_path"]=csv_path
    return dataset


if __name__ == "__main__":
    DATASET_FILE_PATH = f"evaluation/results/{PREFIX}_dataset.json"
    if os.path.exists(DATASET_FILE_PATH):
        with open(DATASET_FILE_PATH, 'r') as f:
            dataset = json.load(f)

    dataset = make_csvs(dataset)
    with open(DATASET_FILE_PATH, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    ColIoU = np.zeros(len(dataset))
    RowIoU = np.zeros(len(dataset))
    DataIoU = np.zeros(len(dataset))
    for i in range(len(dataset)):
        data = dataset[i]
        path_gt = data['gt_csv_path']
        path_gen = data['gen_csv_path']
        
        ColIoU[i], RowIoU[i], DataIoU[i] = compare_csv(path_gt, path_gen)

        print(f"Index: {i}\nColumns Names IoU: {ColIoU[i]:.2f} -- Rows IoU: {RowIoU[i]:.2f} -- Data IoU: {DataIoU[i]:.2f}")
        if 'annotated_accuracy' in data:
            print(f"Expected IoU: {data['annotated_accuracy']}")
        print('\n')

    print(f"Overall: Columns Names IoU: {ColIoU.mean():.2f} -- Rows IoU: {RowIoU.mean():.2f} -- Data IoU: {DataIoU.mean():.2f}")

    # Save results to CSV table
    RESULTS_PATH = f"evaluation/results/ious.csv"
    
    # Create new results row
    new_row = {
        'name': PREFIX,
        'columns_iou': ColIoU.mean(),
        'rows_iou': RowIoU.mean(),
        'data_iou': DataIoU.mean()
    }
    
    # Load existing results or create new dataframe
    if os.path.exists(RESULTS_PATH):
        results_df = pd.read_csv(RESULTS_PATH)
        if PREFIX not in results_df['name'].values:
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        results_df = pd.DataFrame([new_row])
    
    # Save to CSV
    results_df.to_csv(RESULTS_PATH, index=False)