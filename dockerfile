# Usa Python como imagen base
FROM python:3.9-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de la aplicación al contenedor
COPY . /app

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl && apt-get clean

# Actualiza pip e instala dependencias
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto para la aplicación
EXPOSE 11435

# Comando para iniciar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "11435"]