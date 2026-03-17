# Задача

- Пользователь: исправить поломку сборки после включения CUDA.

# Краткий лог

- Воспроизведена сборка `libgpu_test` с `GPU_CUDA_SUPPORT=ON`.
- Линковка падала на конфликте `cudart.lib` и `cudart_static.lib`.
- Источник конфликта: `libs/utils` использовал legacy `FindCUDA` и `cuda_add_library`, хотя target не содержит `.cu` файлов.
- В `libs/utils/CMakeLists.txt` удалён legacy CUDA build path; target теперь всегда собирается через обычный `add_library`, а при `GPU_CUDA_SUPPORT=ON` получает только define `CUDA_SUPPORT`.
- Для CUDA targets выставлен `CUDA_RUNTIME_LIBRARY Shared`, чтобы CUDA runtime не конфликтовал с уже используемым `CUDA::cudart`.

# Результат

- `libs/utils` больше не подтягивает `cudart_static`.
- Конфликт между `cudart.lib` и `cudart_static.lib` устранён.
