# Маскирование видеокарт через GPGPU_VISIBLE_DEVICES

## Сообщения пользователя

- `начни новый промпт - добавь простое средство для маскирования используемых видеокарт через GPGPU_VISIBLE_DEVICES по аналогии с CUDA_VISIBLE_DEVICES`
- `Use GPU 0 and 2: export CUDA_VISIBLE_DEVICES=0,2`
- `Use only GPU 1: export CUDA_VISIBLE_DEVICES=1`
- `Disable all GPUs: export CUDA_VISIBLE_DEVICES=""`
- `добавь в AGENTS.md что при запуске юнит-тестов сначала запускать на дискретной видеокарте ...`
- `выполни коммиты и запуш (вместе с записанным промптом)`

## Короткий лог действий

- В `gpu::enumDevices()` добавлен общий env-based фильтр `GPGPU_VISIBLE_DEVICES`.
- Добавлены unit-тесты на фильтрацию списка устройств по индексам, пустую маску и валидацию некорректных значений.
- В `AGENTS.md` добавлено правило сначала гонять GPU unit-tests на дискретной видеокарте через `GPGPU_VISIBLE_DEVICES`, а уже потом на всех устройствах.

## Короткий лог результатов

- `GPGPU_VISIBLE_DEVICES=0,2` оставляет только устройства с индексами `0` и `2` из общего списка `gpu::enumDevices()`.
- `GPGPU_VISIBLE_DEVICES=1` оставляет только одно устройство.
- `GPGPU_VISIBLE_DEVICES=""` скрывает все GPU.
- `./build/libs/gpu/libgpu_test --gtest_filter='gpu_device.*'` проходит: `6` тестов `PASSED`.
- Проверка `vulkan.aplusb` в этой среде не подошла для smoke test: он падает и без нового env var с `Vulkan devices enumeration failed: Context::createInstance: ErrorIncompatibleDriver`, так что это не регрессия `GPGPU_VISIBLE_DEVICES`.
