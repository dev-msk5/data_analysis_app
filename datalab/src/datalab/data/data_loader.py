import pandas as pd
import pathlib as Path

class DataLoader:
    def __init__(self):
        pass

    

    def load_data(self,file_path:str|Path,np:bool=False)->pd.DataFrame:
        path=Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found {file_path}")
        
        if path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        elif path.suffix.lower() == ".xls" or path.suffix.lower() == ".xlsx":
            return pd.read_excel(file_path)
        if path.suffix.lower() == ".json":
            return pd.read_json


