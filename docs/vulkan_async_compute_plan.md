# План и реализация асинхронного Vulkan compute submit

## Кратко

Задача: убрать обязательное блокирующее ожидание после каждого Vulkan compute kernel launch и заменить его на ограниченный async submit с безопасным lifetime, отложенной проверкой `rassert` и встроенной статистикой GPU idle/busy.

Внешний API для алгоритмов не меняется:
- код по-прежнему вызывает `write/read/resize/exec`;
- асинхронность скрыта внутри Vulkan backend;
- синхронизация происходит в точках host-visible операций, explicit stats report и teardown.

## Что реализовано

### 1. Async launch manager в `VulkanEngine`

- `KernelSource::exec()` для compute больше не делает `waitForFences(...)` сразу после submit.
- В `VulkanEngine` добавлен `InflightComputeLaunch`:
  - fence;
  - command buffer;
  - descriptor sets;
  - ссылки на все используемые buffers/images;
  - `rassert` slot;
  - timestamp query ids;
  - submit id;
  - metadata по доступам к ресурсам.
- Добавлен явный лимит числа inflight launches:
  - `setMaxInflightComputeLaunches(size_t)`;
  - default `32`.

### 2. SPIR-V access metadata

- `ShaderModuleInfo` теперь отражает не только descriptor types, но и `DescriptorAccess`:
  - `ReadOnly`;
  - `WriteOnly`;
  - `ReadWrite`;
  - `Unknown`.
- Эти access-моды извлекаются из SPIR-V reflection и сохраняются в `VulkanKernel`.
- В `v1` это используется для conservative whole-resource tracking и корректного lifetime/host sync.

### 3. Два `rassert`-слота на compute kernel

- У каждого Vulkan compute kernel теперь два отдельных `rassert` buffer-а.
- Каждый launch bind-ит свой slot.
- Это позволяет отложенно проверить `rassert` первого launch-а, пока второй того же kernel-а уже inflight.
- Если slot reuse происходит слишком рано, backend ждёт завершения старого launch-а этого kernel-а.

### 4. Async statistics

- Добавлены:
  - `resetAsyncComputeStats()`;
  - `getAsyncComputeStats(bool wait_for_all=true)`;
  - `logAsyncComputeStats(const std::string& label="", bool wait_for_all=true)`.
- Внутри используются Vulkan timestamp queries.
- Считаются:
  - `total`;
  - `busy`;
  - `idle`;
  - `busy %`;
  - `idle %`;
  - `launch count`;
  - `gap count`;
  - `median gap`;
  - `max gap`;
  - host waits из-за inflight-limit / slot-pressure.

### 5. Безопасный teardown ресурсов

- Уничтожение Vulkan buffers/images теперь дожидается конфликтующих inflight launches.
- Важный фикс: inflight launch больше не разрушается под `inflight_compute_launches_mutex`, иначе происходил deadlock через рекурсивный `waitForBuffer()/waitForImage()`.
- Для async path увеличен descriptor pool:
  - `VK_MAX_DESCRIPTORS_PER_TYPE = 8192`;
  - `max_descriptor_sets = 8192`.

## Базовые наблюдения до async

GTX 1650, `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`, `AVK_ENABLE_VALIDATION_LAYERS=false`.

- `vulkan.radixSort`
  - median `47.18 ms`
  - `22.22 M uint/s`
  - `nsys`: busy `23.4%`, idle `76.8%`, median gap `28 us`, max gap `2.82 ms`
- `vulkan.atomicAdd`
  - `10M -> 33.5M`: median `32.31 ms`
  - `10M -> 1000`: median `9.62 ms`
- `vulkan.binarySearch`
  - median `11.65 ms`

Этого было достаточно, чтобы оправдать задачу: у `radixSort` уже до изменений были большие пузыри между compute submits.

## Новые unit tests

- `vulkan.asyncTinyWriteValueChain`
  - 1024 коротких launches `write_value_at_index.comp`
  - один финальный readback
  - логирует launches/sec и async stats
- `vulkan.asyncAplusbChain`
  - 16 подряд launches `aplusb.comp` без промежуточного readback
  - логирует effective bandwidth и async stats

Оба теста запускаются на всех Vulkan GPU для correctness, но speedup-выводы делаются только по GTX 1650.

## Как профилировать

Использовать:

```bash
AVK_ENABLE_VALIDATION_LAYERS=false \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
nsys profile -t vulkan,osrt -s none --cpuctxsw=none --vulkan-gpu-workload=individual \
  --output /tmp/nsys_radixsort \
  ./build/libs/gpu/libgpu_test --gtest_filter='vulkan.radixSort'
```

Разбор SQLite:

```bash
python3 scripts/profile_vulkan_async_nsys.py /tmp/nsys_radixsort.sqlite
```

Скрипт считает:
- total GPU span;
- busy time;
- idle time;
- busy/idle %;
- gap count;
- median/max gap.

## Что должно проверяться дальше

- Повторить `nsys` для:
  - `vulkan.radixSort`;
  - `vulkan.asyncTinyWriteValueChain`;
  - `vulkan.asyncAplusbChain`.
- Сравнить code-side stats и `nsys` на GTX 1650.
- Прогнать весь `libgpu_test` на GTX 1650.
- Зафиксировать итоговые before/after speedups и пузырьность в этом документе.
