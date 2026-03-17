# Ускорение Vulkan radix sort на NVIDIA GeForce GTX 1650

## Постановка

Нужно было сравнить `cuda.radixSort` и `vulkan.radixSort` на `NVIDIA GeForce GTX 1650` в `RelWithDebInfo`, понять причину сильного отставания Vulkan-версии и по возможности приблизить её к CUDA.

Запуски и profiling делались вне sandbox, c:

```bash
GPGPU_VISIBLE_DEVICES=2
AVK_ENABLE_VALIDATION_LAYERS=false
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

## Baseline до изменений

Фиксировался на той же машине до переписывания Vulkan radix sort:

- `cuda.radixSort`
  - median: `0.00182428 s`
  - effective bandwidth: `4.08411 GB/s`
- `vulkan.radixSort`
  - median: `0.0144401 s`
  - effective bandwidth: `0.541027 GB/s`
  - async stats:
    - total: `0.0143656 s`
    - busy: `72.29%`
    - idle: `27.71%`
    - launches: `215`

Итого baseline: Vulkan был примерно в `7.9x` медленнее CUDA.

## Гипотеза

Главная разница с CUDA была не только в самом API, а в структуре алгоритма:

- CUDA делала по `4` kernel launch-а на radix pass:
  - `local_counting`
  - `reduction`
  - `accumulation`
  - `scatter`
- Vulkan делала намного больше фаз:
  - дерево `reduction level`
  - дерево `accumulation level`
  - дополнительный `copy`
- в CUDA `local_counting` и `scatter` активно использовали warp/subgroup primitives вроде `__ballot_sync`, `__popc`, `__shfl_up_sync`

Отсюда план:

1. приблизить структуру Vulkan radix sort к CUDA;
2. убрать многоуровневое дерево reduction/accumulation;
3. убрать дополнительный `copy` и перейти на ping-pong buffers;
4. использовать subgroup ballot там, где это даёт тот же тип ускорения, что и warp ballot в CUDA.

## Что изменено

### 1. Shader target

Для Vulkan shader compilation включён target `vulkan1.2`, чтобы subgroup operations компилировались стабильно через `glslc`.

Файл:

- `libs/gpu/libgpu/vulkan/CMakeLists.txt`

### 2. Новый host-side radix pass

Vulkan test переписан так, чтобы каждая radix-итерация делала только:

1. `local_counting`
2. `reduction`
3. `accumulation`
4. `scatter`

и использовала ping-pong между `tmp_gpu` и `output_gpu`, без отдельного `copy`.

Файл:

- `libs/gpu/libgpu/vulkan/tests/radix_sort_test.cpp`

### 3. Новый `local_counting`

Вместо простого atomic-based подхода используется subgroup ballot logic, близкая к CUDA `__ballot_sync`.

Критичный correctness bug во время работы тоже был найден именно здесь:

- `subgroupBallot(...)` сначала вызывался только в половине lane-ов (`lane_ind < 16`);
- это было некорректно для subgroup operation;
- из-за этого суммарные counters получались примерно `n / 2`, а sort становился быстрым, но неправильным.

Исправление:

- ballot теперь вызывается uniform-но для всех invocation в subgroup;
- только запись результата в shared-memory ограничивается нужным lane.

Файл:

- `libs/gpu/libgpu/vulkan/tests/kernels/radix_sort_01_local_counting.comp`

### 4. Новый reduction / accumulation / scatter

Reduction и scatter переписаны ближе к CUDA-структуре:

- reduction считает per-bin block offsets сразу за один kernel;
- accumulation считает exclusive prefix по `16` bins;
- scatter использует subgroup ballot/exclusive bit count для ранга внутри warp/subgroup.

Файлы:

- `libs/gpu/libgpu/vulkan/tests/kernels/radix_sort_02_global_prefixes_scan_sum_reduction.comp`
- `libs/gpu/libgpu/vulkan/tests/kernels/radix_sort_03_global_prefixes_scan_accumulation.comp`
- `libs/gpu/libgpu/vulkan/tests/kernels/radix_sort_04_scatter.comp`

## Результат после изменений

### Чистые последовательные unit-test замеры

Логи:

- `artifacts/profiles/radixsort_vulkan_vs_cuda_20260317/cuda_radixsort_seq.stdout`
- `artifacts/profiles/radixsort_vulkan_vs_cuda_20260317/vulkan_radixsort_seq.stdout`

Результат:

- `cuda.radixSort`
  - median: `0.00179835 s`
  - effective bandwidth: `4.14300 GB/s`
- `vulkan.radixSort`
  - median: `0.00367953 s`
  - effective bandwidth: `2.12324 GB/s`
  - async stats:
    - total: `0.00364966 s`
    - busy: `81.11%`
    - idle: `18.89%`
    - launches: `32`

### Итоговое сравнение

- Vulkan ускорился с `0.0144401 s` до `0.00367953 s`
- это примерно `3.92x` ускорение
- отставание от CUDA сократилось с примерно `7.9x` до примерно `2.05x`

## Nsight Systems profiling

Профили:

- Vulkan:
  - `artifacts/profiles/radixsort_vulkan_vs_cuda_20260317/vulkan_radixsort_nsys.nsys-rep`
  - `artifacts/profiles/radixsort_vulkan_vs_cuda_20260317/vulkan_radixsort_nsys.sqlite`
- CUDA:
  - `artifacts/profiles/radixsort_vulkan_vs_cuda_20260317/cuda_radixsort_nsys.nsys-rep`
  - `artifacts/profiles/radixsort_vulkan_vs_cuda_20260317/cuda_radixsort_nsys.sqlite`

Открытие в GUI:

```bash
nsys-ui artifacts/profiles/radixsort_vulkan_vs_cuda_20260317/vulkan_radixsort_nsys.nsys-rep
nsys-ui artifacts/profiles/radixsort_vulkan_vs_cuda_20260317/cuda_radixsort_nsys.nsys-rep
```

### CUDA

По `nsys`:

- kernel span: `18.598 ms`
- total kernel busy time: `14.997 ms`
- busy примерно `80.6%`
- idle примерно `19.4%`

Kernel breakdown:

- `scatter`: `9.414 ms`
- `local_counting`: `4.277 ms`
- `reduction`: `1.173 ms`
- `accumulation`: `0.134 ms`

Runtime API hot spots:

- `cudaStreamSynchronize`: `17.5 ms`
- `cudaLaunchKernel`: `1.22 ms`

### Vulkan

По тестовой встроенной async-статистике на целевом проходе:

- total: `0.0036655 s`
- busy: `80.44%`
- idle: `19.56%`
- launches: `32`

По `nsys` API summary основное runtime overhead:

- `vkWaitForFences`: `30.906 ms` на `212` calls
- `vkQueueSubmit`: `1.314 ms` на `355` calls
- `vkAllocateDescriptorSets`: `1.126 ms` на `352` calls

Важно: в `nsys` Vulkan API summary попадает и setup/teardown, поэтому для процента busy/idle надёжнее использовать встроенную статистику самого теста на окне sort-прохода.

## Вывод

Главный выигрыш дало не одно “магическое” subgroup-API, а совокупность трёх вещей:

1. переход от многоуровневой Vulkan-схемы к CUDA-подобным `4` kernel phases на pass;
2. устранение отдельного `copy` и переход на ping-pong buffers;
3. использование subgroup ballot logic в `local_counting` и `scatter`.

### Что ещё осталось до CUDA

Vulkan всё ещё медленнее CUDA примерно в `2x`. Наиболее вероятные причины:

- reduction в Vulkan всё ещё опирается на shared-memory scan с barrier-ами, а не на полноценный shuffle-style путь как в CUDA;
- Vulkan всё ещё платит заметный host/API overhead на `vkWaitForFences`, `vkQueueSubmit`, descriptor allocation и command buffer churn;
- CUDA kernel path исторически компактнее и лучше совпадает с моделью warp execution на NVIDIA.

### Практический итог

Изменение оправдано:

- correctness сохранён;
- ускорение почти `4x`;
- разрыв с CUDA резко сократился;
- архитектура Vulkan radix sort стала проще и ближе к CUDA-референсу, а значит её легче дальше профилировать и улучшать.

## Follow-up: корректность на Intel UHD 770 и lavapipe

После ускорения обнаружился portability regression:

- на `Intel UHD 770` быстрый путь сначала давал неправильный результат;
- на `lavapipe` он тоже ломался.

### Причина

Быстрый путь оказался чувствителен к subgroup semantics:

- kernels используют subgroup ballot path;
- код был жёстко написан под `subgroup = 32`;
- при этом pipeline requirement на `requiredSubgroupSize = 32` выставлялся по неправильному условию, через `VK_KHR_cooperative_matrix`, а не через `VK_EXT_subgroup_size_control`.

Это исправлено в:

- `libs/gpu/libgpu/vulkan/engine.cpp`

После этого:

- `Intel UHD 770` стал корректно исполнять быстрый subgroup path;
- `lavapipe` всё ещё давал неверный результат, несмотря на поддержку части subgroup machinery.

### Финальное решение

Сделан явный split:

- `NVIDIA` и `Intel` используют `fast-subgroup` path;
- все остальные Vulkan devices используют `portable-fallback`, основанный на старом дереве reduction/accumulation без subgroup-зависимого fast path.

Для этого в тестовые Vulkan kernels добавлен второй, portable набор:

- `radix_sort_01_local_counting_portable.comp`
- `radix_sort_02_global_prefixes_scan_sum_reduction_portable.comp`
- `radix_sort_03_global_prefixes_scan_accumulation_portable.comp`
- `radix_sort_04_scatter_portable.comp`

### Проверка

Последовательные прогоны после исправления:

- `Intel UHD 770`
  - path: `fast-subgroup`
  - median: `0.0102515 s`
  - `PASSED`
- `NVIDIA GeForce GTX 1650`
  - path: `fast-subgroup`
  - median: `0.00364193 s`
  - `PASSED`
- `llvmpipe`
  - path: `portable-fallback`
  - median: `0.161213 s`
  - `PASSED`

Итого:

- NVIDIA speed не пострадала;
- Intel correctness восстановлена и быстрый путь сохранён;
- lavapipe теперь идёт через безопасный portable fallback и тоже корректен.

## Follow-up: ещё попытка сократить разрыв с CUDA на GTX 1650

После предыдущих изменений на GTX 1650 оставался разрыв примерно `2.13x`:

- `cuda.radixSort`: median `0.00172542 s`
- `vulkan.radixSort`: median `0.00367852 s`

Профили:

- `artifacts/profiles/radixsort_gap_closure_20260317/cuda_radixsort_before.nsys-rep`
- `artifacts/profiles/radixsort_gap_closure_20260317/vulkan_radixsort_before.nsys-rep`

### Что ещё удалось улучшить

1. `reduction` для fast Vulkan path переведён с shared-memory scan на `subgroupInclusiveAdd`.
2. Отдельный `accumulation` dispatch убран из fast path:
   - prefix по `16` bins теперь считается прямо в начале `scatter`.
3. `scatter` упрощён:
   - убран лишний цикл по `chunk_size`;
   - fast path теперь исходит из фактической схемы `one workgroup -> one 256-element block`.

### Результат на GTX 1650

После этих двух дополнительных оптимизаций:

- `vulkan.radixSort`: median `0.00345173 s`
- async stats:
  - total `0.00344186 s`
  - busy `83.40%`
  - idle `16.60%`
  - launches `24`

Это значит:

- дополнительное ускорение Vulkan примерно `1.07x` относительно `0.00367852 s`
- суммарное ускорение относительно исходного baseline `0.0144401 s` уже около `4.18x`
- разрыв с CUDA сократился с `2.13x` примерно до `2.00x`

### Profiling before/after

После изменений:

- `artifacts/profiles/radixsort_gap_closure_20260317/vulkan_radixsort_after.nsys-rep`

Что изменилось по `nsys`:

- `VULKAN_WORKLOAD` busy sum:
  - было `44.772 ms`
  - стало `35.447 ms`
- GPU span:
  - было `29.472 ms`
  - стало `24.118 ms`
- workload count:
  - было `442`
  - стало `272`
- `vkQueueSubmit`:
  - было `355` calls / `1.433 ms`
  - стало `267` calls / `0.894 ms`

### Вывод

Удалось ещё немного приблизиться к CUDA, но уже без драматического скачка.

На этом этапе:

- low-risk улучшения fast Vulkan path уже в основном выбраны;
- remaining gap около `2x` выглядит уже как сочетание:
  - оставшегося kernel-level разрыва;
  - неизбежного Vulkan dispatch overhead;
  - более удачной для GTX 1650 native CUDA реализации.

Следующий уровень ускорения уже потребовал бы заметно более инвазивных изменений:

- deeper rewrite `scatter`/`local_counting` под ещё более жёстко NVIDIA-oriented subgroup path;
- дальнейшее слияние фаз;
- либо отдельный NVIDIA-only path с более агрессивными предположениями.
