# Исправление старого коммита aplusb unit test

## Сообщения пользователя

- `исправь старый коммит с A+B unit test, падает запуск - ./libgpu_test: error while loading shared libraries: libcuda.so.1: cannot open shared object file: No such file or directory`
- `и сделай это в старом коммите с юнит тестом а+б`

## Короткий лог действий

- Найден целевой коммит: `2b4493b Add CUDA aplusb unit test`.
- Проверена сборка `libgpu_test` с `GPU_CUDA_SUPPORT=ON`: бинарь имел `DT_NEEDED` на `libcudart.so.13` и `libcuda.so.1`.
- В `libs/gpu/CMakeLists.txt` убрана жёсткая линковка на `CUDA::cuda_driver`, CUDA runtime переведён в `Static`.
- В `libs/gpu/libgpu/cuda/cuda_api.*` добавлен lazy-loading для `cuDeviceGetAttribute` и `cuCtxCreate`.
- В `libs/gpu/libgpu/context.cpp` прямые вызовы driver API переведены на dynamic wrappers.

## Короткий лог результатов

- `cmake --build build-cuda-test --target libgpu_test -j$(nproc)` проходит.
- `readelf -d build-cuda-test/libs/gpu/libgpu_test` больше не показывает `libcuda.so.1` и `libcudart.so.13` в `NEEDED`.
- `./build-cuda-test/libs/gpu/libgpu_test --gtest_list_tests` успешно стартует.
