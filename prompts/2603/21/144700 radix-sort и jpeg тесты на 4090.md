# Задача

## Сообщения пользователя

- «прогони юнит тесты radix-sort и jpeg на 4090»

## Лог действий

- Проверил актуальный набор тестов в текущем `origin/main`.
- Обнаружил, что в исходниках есть новые `vulkan.radixSort` и `vulkan.jpeg*` тесты, но текущий бинарник `build/libs/gpu/libgpu_test` собран по старой ревизии и их не содержит.
- Пересобрал тестовый бинарник на текущем `HEAD`.
- Воспроизвёл падения `vulkan.jpegDecodeBenchmark` и `vulkan.jpegDecodeBenchmarkGpu` на RTX 4090.
- Исследовал код benchmark-путей и reduce shaders.
- Исправил вычисление линейного `group_index` в Vulkan reduce shaders для 2D launch geometry.
- Исправил grayscale JPEG GPU benchmark так, чтобы reduction не переполнялась на больших изображениях: теперь он останавливает `u32 -> u32` редукцию до переполнения и суммирует partial sums на CPU в `uint64_t`.
- Повторно пересобрал `libgpu_test`.
- Повторно прогнал JPEG benchmark-тесты и `vulkan.radixSort` на RTX 4090 с `AVK_ENABLE_VALIDATION_LAYERS=true`.

## Результат

- Для запуска только на RTX 4090 будет использоваться `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.
- `vulkan.jpegDecodeBenchmark` проходит на RTX 4090 с validation layers.
- `vulkan.jpegDecodeBenchmarkGpu` проходит на RTX 4090 с validation layers.
- `vulkan.radixSort` проходит на RTX 4090 с validation layers.
