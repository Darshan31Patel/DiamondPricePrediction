from pydantic import BaseModel

class PredictionDataset(BaseModel):
    cut:str
    color:str
    clarity:str
    carat:float
    depth:float
    table:float
    x:float
    y:float
    z:float
