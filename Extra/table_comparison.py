import pandas as pd
import math
import duckdb
from typing import List


def compare_pandas_table(pred, gold, condition_cols=[], ignore_order=False):
    
    tolerance = 1e-2

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        try:
            if ignore_order_:
                v1, v2 = (sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                        sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))))
            if len(v1) != len(v2):
                return False
            for a, b in zip(v1, v2):
                if pd.isna(a) and pd.isna(b):
                    continue
                elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if not math.isclose(float(a), float(b), abs_tol=tol):
                        return False
                elif a != b:
                    return False
            return True
        except Exception as e:
            return False
    
    if condition_cols != []:
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    pred_cols = pred
    
    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    score = 1
    for _, gold in enumerate(t_gold_list):
        if not any(vectors_match(gold, pred, ignore_order_=ignore_order) for pred in t_pred_list):
            score = 0
        else:
            for j, pred in enumerate(t_pred_list):
                if vectors_match(gold, pred, ignore_order_=ignore_order):
                    break

    return score
    

def compare_multi_pandas_table(pred, multi_gold, multi_condition_cols=[], multi_ignore_order=False):
    if multi_condition_cols == [] or multi_condition_cols == [[]] or multi_condition_cols == [None] or multi_condition_cols == None:
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    multi_ignore_order = [multi_ignore_order for _ in range(len(multi_gold))]
    
    for i, gold in enumerate(multi_gold):
        if compare_pandas_table(pred, gold, multi_condition_cols[i], multi_ignore_order[i]):
            return 1
    return 0


def table_match(result: str, gold, condition_cols=[], ignore_order=False) -> float:
    """ 
    @args:
        result (str):
        gold (str|List):
        condition_cols (List[int])
        ignore_order (bool)
    """
    df1 = pd.read_csv(result, low_memory=False)
    
    if isinstance(gold, str):
        df2 = pd.read_csv(gold, low_memory=False)
        score = compare_pandas_table(df1, df2, condition_cols=condition_cols, ignore_order=ignore_order)
    elif isinstance(gold, List):
        df_list = [pd.read_csv(g, low_memory=False) for g in gold]
        score = compare_multi_pandas_table(df1, df_list, multi_condition_cols=condition_cols, multi_ignore_order=ignore_order)
    
    return score



def duckdb_match(result: str, gold: str, condition_tabs=None, condition_cols: List[List[int]]=None, ignore_orders: List[bool]=None):
    """
    Parameters:
    - result (str): Path to the DuckDB file containing the result tables.
    - gold (str): Path to the DuckDB file containing the gold standard tables.
    - condition_tabs (List[str], optional): List of table names to be checked. If not provided, all tables in the gold DuckDB file will be considered.
    - condition_cols (List[List[int]], optional): A list of lists, where each inner list contains column indices used for matching conditions for the corresponding table. Defaults to considering all columns.
    - ignore_orders (List[bool], optional): A list of boolean values indicating whether to ignore the row order for each table comparison. Defaults to [False] for each table.
    """
   
   
    def get_duckdb_table_names(db: str) -> List[str]:
        """
        Retrieves the names of all tables in the DuckDB database.

        Parameters:
        - db (str): The path to the DuckDB database file.

        Returns:
        - List[str]: A list of table names in the DuckDB database.
        """
        con = duckdb.connect(database=db, read_only=True)
        result = con.execute("SHOW TABLES").fetchall()
        con.close()
        return [row[0] for row in result]
    
    
    def get_duckdb_pandas_table(db, table_name):
        con = duckdb.connect(database=db, read_only=True)
        df = con.execute(f'SELECT * FROM {table_name}').fetchdf()
        con.close()
        return df
    
    if condition_tabs is None:
        condition_tabs = get_duckdb_table_names(gold)

    gold_tables = [get_duckdb_pandas_table(gold, table_name) for table_name in condition_tabs]
    try:
        pred_tables = [get_duckdb_pandas_table(result, table_name) for table_name in condition_tabs]
    except:
        return 0
    
    assert len(gold_tables) == len(pred_tables)

    if ignore_orders is None:
        ignore_orders = [False] * len(gold_tables)
        
    assert len(ignore_orders) == len(gold_tables)

    if condition_cols is None:
        condition_cols = [[]] * len(gold_tables)


    assert len(condition_cols) == len(gold_tables)
    
    for i, (gold_table, pred_table) in enumerate(zip(gold_tables, pred_tables)):
        if not compare_pandas_table(pred_table, gold_table, condition_cols=condition_cols[i], ignore_order=ignore_orders[i]):
            return 0
        
    return 1


def tables_match(result: List[str], gold: List[str], condition_cols: List[List[int]]=None, ignore_orders: List[bool]=None):
    """
    Parameters:
    - result (Lstr): Path to the result tables.
    - gold (str): Path to the gold standard tables.
    - condition_cols (List[List[int]], optional): A list of lists, where each inner list contains column indices used for matching conditions for the corresponding table. Defaults to considering all columns.
    - ignore_orders (List[bool], optional): A list of boolean values indicating whether to ignore the row order for each table comparison. Defaults to [False] for each table.
    """

    def get_tables_to_dfs(csv_file: str):
        df = pd.read_csv(csv_file)
        return df

    gold_tables = [get_tables_to_dfs(table_name) for table_name in gold]
    try:
        pred_tables = [get_tables_to_dfs(table_name) for table_name in result]
    except:
        return 0
    
    assert len(gold_tables) == len(pred_tables)

    if ignore_orders is None:
        ignore_orders = [False] * len(gold_tables)
        
    assert len(ignore_orders) == len(gold_tables)

    if condition_cols is None:
        condition_cols = [[]] * len(gold_tables)


    assert len(condition_cols) == len(gold_tables)
    
    for i, (gold_table, pred_table) in enumerate(zip(gold_tables, pred_tables)):
        if not compare_pandas_table(pred_table, gold_table, condition_cols=condition_cols[i], ignore_order=ignore_orders[i]):
            return 0
        
    return 1