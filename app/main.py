from fastapi import FastAPI,UploadFile,File,Request
from fastapi.templating import Jinja2Templates
import predictor

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})

@app.post("/")
async def predict(request: Request, file: UploadFile = File(...)):
    result = predictor.get_result(file)
    return templates.TemplateResponse("index.html",{"request":request,"result":result})