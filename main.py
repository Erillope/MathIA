from fastapi import FastAPI
from controller import Controller
import time

app = FastAPI()
controller = Controller()

print("API is running...")

@app.get("/")
def root():
    return {"message": "Hola FastAPI"}

@app.get("/preguntas")
def get_user(materia: str, nivel: int):
    start_time = time.time()
    result = {"problemas": controller.predict(materia, nivel)}
    end_time = time.time()
    print(f"Request processed in {end_time - start_time} seconds")
    return result