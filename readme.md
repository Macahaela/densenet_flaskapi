
### Instalasi

```bash
   cd flaskapi_densenetrcnn
   ```

1. **Buat virtual environment (.env)**:

   ```bash
   python -m venv .venv
   ```

3. **Aktifkan virtual environment**:

   * **Windows**:

     ```bash
     .venv\Scripts\activate
     ```

   * **macOS/Linux**:

     ```bash
     source .env/bin/activate
     ```

4. **Install dependensi**:

   ```bash
   pip install -r requirements.txt
   ```

---

### Menjalankan API

```bash
python app.py
```

API akan berjalan di: `http://127.0.0.1:5000`

---

### Contoh Request (Postman)

* **Method**: `POST`
* **Endpoint**: `http://127.0.0.1:5000/predict`
* **Body**: JSON (raw)

```json
{
  "image": "https://link-to-image.jpg"
}
```
