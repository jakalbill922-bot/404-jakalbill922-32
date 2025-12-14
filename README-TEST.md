# 3D Generation Pipeline - IMPROVED VERSION

Улучшенная версия с оптимизациями.

## Изменения от оригинала:

1. **BiRefNet** вместо BEN2 — более качественное удаление фона
2. **Random Seeds** — генерация с 3 разными seeds, выбор медианного результата

## Требования

- **GPU:** минимум 48GB VRAM (A6000, L40, A100)
- Рекомендуется 80GB для стабильной работы

## Запуск

```bash
cd docker
docker compose up --build -d

# Логи
docker logs -f pipeline

# Тест
curl http://localhost:10006/health
```

## Тестовая генерация

```bash
curl -X POST "http://localhost:10006/generate" \
  -F "prompt_image_file=@test.png" \
  -F "seed=42" \
  -o output.ply
```

## Оценка VRAM

| Модель | VRAM |
|--------|------|
| Qwen Edit | ~16-18 GB |
| BiRefNet | ~2 GB |
| Trellis + DINOv2 | ~15-18 GB |
| **Итого** | **~35-40 GB** |
