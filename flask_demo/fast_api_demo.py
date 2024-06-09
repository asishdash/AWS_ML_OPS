from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello":"world"}

@app.get("/items/{item_id}")
def read_item(item_id :int ,q:str=None):
    return{"item_id":item_id,"q":q}

#first run the file then :--
#python -m uvicorn fast_api_demo:app --reload

