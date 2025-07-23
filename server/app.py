from fastapi import FastAPI
from pydantic import BaseModel
from ta_engine.data import load_price_data
from ta_engine.strategies import run_strategy

app = FastAPI()


class StrategyRequest(BaseModel):
    filepath: str
    config: dict


@app.post("/run_strategy")
def run_strategy_api(req: StrategyRequest):
    df = load_price_data(req.filepath)
    result = run_strategy(df, req.config)
    return result.to_dict(orient="records")

