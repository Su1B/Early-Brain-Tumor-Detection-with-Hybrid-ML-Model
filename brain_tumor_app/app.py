import os
import shutil
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import torchvision.transforms as transforms
from PIL import Image

from model import InferenceModel
from pdf_gen import generate_medical_report

app = FastAPI(title="Brain Tumor Detection App")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PDF_DIR = os.path.join(BASE_DIR, "reports")

# Ensure required directories exist
# Static folders
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/reports", StaticFiles(directory=PDF_DIR), name="reports")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize ML Model (will load best_model.pth if it exists)
ml_model = InferenceModel()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save the file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        img = Image.open(file_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0) # Add batch dimension
        
        # Predict using hybrid model
        result = ml_model.predict(img_tensor, file.filename)
        
        # Determine paths and URLs
        pdf_filename = f"report_{file.filename.split('.')[0]}.pdf"
        pdf_path = os.path.join(PDF_DIR, pdf_filename)
        
        # Generate PDF
        generate_medical_report(
            patient_name="Alex Mercer",
            age=48,
            gender="Male",
            detection_result=result["prediction"],
            confidence=result["confidence"],
            original_img_path=file_path,
            output_pdf_path=pdf_path
        )
        
        return {
            "status": "success",
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "image_url": f"/uploads/{file.filename}",
            "pdf_url": f"/reports/{pdf_filename}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5050, reload=True)
