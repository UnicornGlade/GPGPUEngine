# Задача

- Пользователь попросил не начинать до `02:30 GMT+3`, потому что на GPU шли эксперименты.
- После этого нужно было:
  - запустить `cuda.radixSort` и `vulkan.radixSort` на GTX 1650 вне sandbox в `RelWithDebInfo`;
  - понять, почему Vulkan сильно медленнее;
  - попробовать приблизить Vulkan-реализацию к CUDA;
  - при хорошем результате закоммитить и запушить.

# Ход работы

- Дождался `02:30 GMT+3` и только после этого начал работу.
- Снял baseline:
  - `cuda.radixSort` median около `0.00182 s`
  - `vulkan.radixSort` median около `0.01444 s`
- Сравнил CUDA и Vulkan kernels:
  - CUDA использует компактную схему из `4` kernels/pass;
  - Vulkan использовал больше фаз, дерево reduction/accumulation и отдельный `copy`.
- Переделал Vulkan radix sort:
  - включил `glslc --target-env=vulkan1.2`;
  - заменил дерево reduction/accumulation на CUDA-подобный путь;
  - убрал отдельный `copy`, перешёл на ping-pong buffers;
  - перевёл `local_counting` и `scatter` на subgroup ballot-подход.
- На первом быстром варианте поймал correctness bug:
  - sort стал быстрым, но неправильным;
  - локализовал причину в `subgroupBallot`, который вызывался не uniform-но во всех lane-ах subgroup;
  - исправил это.
- После исправления переснял чистые sequential замеры и `nsys` profiles.

# Результат

- `cuda.radixSort`:
  - median `0.00179835 s`
  - effective bandwidth `4.14300 GB/s`
- `vulkan.radixSort` после изменений:
  - median `0.00367953 s`
  - effective bandwidth `2.12324 GB/s`
  - async stats: busy `81.11%`, idle `18.89%`, launches `32`

Итог:

- Vulkan ускорился примерно в `3.92x`
- разрыв с CUDA сократился примерно с `7.9x` до `2.05x`

# Профили и подробности

- Подробный разбор записан в:
  - `docs/vulkan_radixsort_gtx1650_optimization.md`
- Профили:
  - `artifacts/profiles/radixsort_vulkan_vs_cuda_20260317/vulkan_radixsort_nsys.nsys-rep`
  - `artifacts/profiles/radixsort_vulkan_vs_cuda_20260317/cuda_radixsort_nsys.nsys-rep`
