"""
  -*- encoding: utf-8 -*-
  @Author: Deepwind
  @Time  : 10/13/2024 7:28 PM
  @Email: deepwind32@163.com
"""
import os
from pathlib import Path
import time
from cachetools import TTLCache
from fastapi import FastAPI, File, UploadFile, Request
import hashlib
import uuid
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware

from predict import predict

current_dir = Path(os.path.abspath(__file__)).parent

global_map = {
    "upload_path": current_dir / "upload",
}
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")
app.mount("/static", StaticFiles(directory="./static"), name="static")

# memory db
file_cache = TTLCache(maxsize=50, ttl=600)

def cleanup_files():
    file_uid = set(f.stem for f in global_map['upload_path'].iterdir())
    exist_uid = set(file_cache.values())
    for uid in file_uid - exist_uid:
        (global_map['upload_path'] / f"{uid}.edf").unlink(missing_ok=True)

@app.on_event("startup")
async def startup_event():
    from threading import Timer
    global_map["cleanup_thread"] = Timer(interval=300, function=cleanup_files)
    global_map["cleanup_thread"].start()

@app.on_event("shutdown")
async def shutdown_event():
    global_map["cleanup_thread"].cancel()
    global_map["cleanup_thread"].join()
    

def get_filepath(file_hash):
    if file_hash in file_cache:
        return file_cache[file_hash]
    else:
        return None

# 上传文件
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # hash计算
    file_content = file.file.read()
    file_hash = hashlib.md5(file_content).hexdigest()
    if file_hash in file_cache:
        file_cache[file_hash] = file_cache[file_hash]  # 重置TTL
        return {"code": 200, "data": file_cache[file_hash], "msg": "success"}

    uid = uuid.uuid4().hex
    suffix = Path(file.filename).suffix.lstrip('.')
    if suffix != "edf":
        return {"code": 400, "data": None, "msg": "file type error"}

    file_path = global_map['upload_path'] / f"{uid}.edf"
    with open(file_path, "wb") as f:
        f.write(file_content)
    file_cache[file_hash] = uid
    return {"code": 200, "data": uid, "msg": "success"}

@app.post("/process")
async def process_report(info: dict):
    if info["uid"] == "example":
        file = (current_dir / "static/example/eeg5.edf").as_posix()
    else:
        file = (global_map["upload_path"] / f"{info['uid']}.edf").as_posix()

    return {
        "code": 200,
        "data": predict(
                    file, 
                    info["channels"], 
                    info["sample_rate"], 
                    64, 
                    info["bipolar"],
                ),
        "msg": "success"
    }

@app.get("/index")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/report")
async def report_page(request: Request):
    return templates.TemplateResponse("report.html", {"request": request})