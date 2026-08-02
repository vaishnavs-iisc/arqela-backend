import modal

# Define the Modal Image with dependencies and local project code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root")
)

# Create the Modal App
app = modal.App("arqela-backend")

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("arqela-secrets"),
    ],
    timeout=300,
)
@modal.asgi_app()
def fastapi_entrypoint():
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from main import app as fastapi_app
    return fastapi_app
