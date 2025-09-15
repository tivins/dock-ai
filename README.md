# Dock-ai
Artificial Intelligence inside Docker.


> [!IMPORTANT]
> 
> This repository is in development mode.
> 
> DO NOT USE IN PRODUCTION.



Models :

* [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
* [hexgrad/Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)


Install

    docker compose up -d --build

Call

    docker exec -it dock_ai python /app/ai/chat.py

Next steps :

* Use FastAPI