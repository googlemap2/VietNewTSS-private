# yourtts

Dự án TTS cá nhân theo kiến trúc VieNeu, hiện đã tích hợp chạy **VieNeu Turbo GGUF** với preset voice tiếng Việt.

## 1. Tính năng chính

- Engine `standard` (mô phỏng tín hiệu) để test nhanh pipeline.
- Engine `turbo` dùng `vieneu` + model GGUF (`pnnbao-ump/VieNeu-TTS-0.3B-q8-gguf`).
- Lấy danh sách preset voice trực tiếp từ VieNeu (`/voices`, Web UI dropdown).
- Web UI Gradio có chỉnh cảm xúc realtime:
  - `Temperature`
  - `Top-k`
- API Flask:
  - `/health`, `/voices`, `/warmup`
  - `/synthesize`, `/synthesize_batch`, `/synthesize_stream`, `/synthesize_clone`
- Hỗ trợ `.env` (auto-load) cho biến môi trường như `HF_TOKEN`.

## 2. Yêu cầu môi trường

- Python `3.11` (khuyến nghị, đã test với `.venv311`).
- Windows PowerShell (hoặc shell tương đương).
- Mạng internet để tải model lần đầu từ Hugging Face.

## 3. Cài đặt

### Bước 1: Tạo môi trường ảo

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
```

### Bước 2: Cài project + VieNeu

```powershell
python -m pip install -e .[vieneu]
```

## 4. Cấu hình

### 4.1 `config.yaml`

Đảm bảo cấu hình Turbo như sau:

```yaml
engine_mode: turbo
sample_rate: 24000
output_dir: outputs
voice: default
model_name: pnnbao-ump/VieNeu-TTS-0.3B-q8-gguf
device: cpu
cache_size: 128
```

Lưu ý:
- Với VieNeu Turbo, runtime sẽ tự đồng bộ sample rate model (`24000 Hz`).
- `voice: default` chỉ là fallback; danh sách preset thực sẽ lấy từ model.

### 4.2 `.env` (khuyến nghị)

Tạo file `.env` ở root project:

```env
HF_TOKEN=your_huggingface_token_here
```

Ứng dụng sẽ tự nạp `.env` khi chạy `smoke`, `api`, `web_ui`.

## 5. Chạy và kiểm tra nhanh

### 5.1 Smoke test

```powershell
.\.venv311\Scripts\python.exe -m yourtts.smoke
```

Kết quả kỳ vọng: tạo file `outputs/smoke.wav`.

### 5.2 Chạy API

```powershell
.\.venv311\Scripts\python.exe apps/api.py
```

### 5.3 Chạy Web UI

```powershell
.\.venv311\Scripts\python.exe apps/web_ui.py
```

Mặc định Web UI ở `http://127.0.0.1:7860`.

## 6. Hướng dẫn sử dụng Web UI

### Tab Single

1. Nhập text.
2. Chọn preset voice (ví dụ `Bích Ngọc (Nữ - Miền Bắc)`).
3. Mở `Emotion Settings` và chỉnh:
   - `Temperature`: cao hơn => giàu cảm xúc hơn, nhưng dễ kém ổn định.
   - `Top-k`: cao hơn => đa dạng hơn.
4. Bấm `Synthesize`.

### Tab Batch

- Mỗi dòng là một câu.
- Chọn voice + emotion settings tương tự Single.
- Kết quả trả về danh sách file WAV.

### Tab Clone

- Nhập text + file audio mẫu.
- Chọn fallback voice.
- Bấm `Synthesize Clone`.

## 7. API nhanh

### 7.1 Health

```bash
curl http://127.0.0.1:8000/health
```

Kỳ vọng `engine_class` là `VieneuTurboEngine`.

### 7.2 Voices

```bash
curl http://127.0.0.1:8000/voices
```

Trả về preset voices hiện có từ VieNeu.

### 7.3 Synthesize 1 câu

```bash
curl -X POST http://127.0.0.1:8000/synthesize \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Xin chào bạn\",\"voice\":\"Bích Ngọc (Nữ - Miền Bắc)\"}"
```

### 7.4 Batch

```bash
curl -X POST http://127.0.0.1:8000/synthesize_batch \
  -H "Content-Type: application/json" \
  -d "{\"texts\":[\"xin chào\",\"chúc bạn một ngày tốt lành\"],\"voice\":\"Phạm Tuyên (Nam - Miền Bắc)\"}"
```

### 7.5 Clone

```bash
curl -X POST http://127.0.0.1:8000/synthesize_clone \
  -F "text=Xin chào từ giọng clone" \
  -F "voice=Bích Ngọc (Nữ - Miền Bắc)" \
  -F "ref_audio=@examples/ref.wav"
```

## 8. Lỗi thường gặp

### 8.1 Nghe như sai giọng / bị ù

- Đảm bảo đang chạy Turbo thật: gọi `/health`, kiểm tra `engine_class`.
- Đảm bảo sample rate đúng 24k (`config.yaml` dùng `24000`).
- Restart lại API/Web UI sau mỗi lần đổi code.

### 8.2 Cảnh báo unauthenticated Hugging Face

- Kiểm tra `.env` đã có `HF_TOKEN=...`.
- Chạy lại app sau khi tạo/sửa `.env`.

### 8.3 Voice list trống hoặc sai

- Gọi `/voices` để kiểm tra runtime.
- Nếu model chưa tải xong lần đầu, chờ download hoàn tất rồi thử lại.

---

Nếu cần mở rộng tiếp, bước hợp lý là thêm `temperature/top_k` vào API payload để client ngoài Web UI cũng chỉnh cảm xúc được.
