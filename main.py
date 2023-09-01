import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
import httpx

app = FastAPI()

EMOTION_SERVICE_URL = "http://127.0.0.1:8003/predict"  # Replace with the URL of your emotion prediction service
SUMMARIZATION_SERVICE_URL = "http://localhost:8000/api"
CONVERT_URL = "http://127.0.0.1:8005/convert"

@app.post("/predict_emotion")
async def predict_emotion(image: dict):
    try:
        print("Predict emotion request received")

        # Forward the image file to the emotion prediction service
        response = requests.post(EMOTION_SERVICE_URL,json=image)
        if response.status_code == 200:
            return response.json()
        else:
            return HTTPException(status_code=500,detail="Error predicting emotion")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while predicting emotion: {str(e)}")


@app.post("/summarize")
async def summarize_text(textdata: str, token: str):
    try:
        key = "RRshJy4beYdlNbu"

        if token != key:
            raise HTTPException(status_code=401, detail="Unauthorized")

        # Forward the request to the summarization service
        async with httpx.AsyncClient() as client:
            response = await client.get(SUMMARIZATION_SERVICE_URL, params={"tinput": textdata, "token": token})
            response_data = response.json()

            if response.status_code == 200:
                return {"Status": "Done", "Summary": response_data.get("Summery", "")}
            else:
                raise HTTPException(status_code=response.status_code,
                                    detail=response_data.get("Status", "Unknown error"))
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while summarizing text: {str(e)}")


@app.post("/convert")
def callToConvertApi(request:dict):
    request_data = {
        "url": request["url"]
    }

    response = requests.post(CONVERT_URL, json=request_data)

    if response.status_code == 200:
        response_data = response.json()
        converted_text = response_data.get("text")
        print("Converted Text:", converted_text)
        return  {"text" : converted_text}
    else:
        print("Request failed. Status code:", response.status_code)

    return HTTPException(status_code=500,detail="convert failed")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5002, timeout_keep_alive=1000000)
