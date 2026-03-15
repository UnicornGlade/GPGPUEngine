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

## Дополнительный эксперимент: pool/reuse для Vulkan fence

После просмотра `nsys` GUI выяснилось, что у первой async-реализации значительная часть CPU-side времени уходила в:

- `vkCreateFence`
- `vkDestroyFence`
- `vkWaitForFences`

Минимальная правка:

- для async compute launch теперь используется отдельный pool `available_compute_fences_`;
- новый launch берёт fence через `acquireInflightComputeFence()`;
- после retire launch-а fence возвращается через `recycleInflightComputeFence()`;
- старые keyed fence-ы (`findFence("readBuffer")`, `findFence("writeImage")` и т.п.) не менялись.

### Сравнение `vulkan.radixSort` на GTX 1650

Текущий async baseline до pool/reuse:

- без профайлера:
  - median `156.63 ms`
  - async stats: total `153.30 ms`, busy `6.68%`, idle `93.32%`
- под `nsys`:
  - median `167.03 ms`
  - `nsys`: total `1.837 s`, busy `11.53%`, idle `88.47%`
  - top CPU Vulkan API:
    - `vkWaitForFences`: `774.81 ms`
    - `vkCreateFence`: `417.49 ms`
    - `vkDestroyFence`: `374.54 ms`
    - `vkQueueSubmit`: `114.76 ms`

Вариант с pool/reuse fence-ов:

- без профайлера:
  - повторяемые запуски дали median `80.22 ms` и `83.23 ms`
  - async stats: total около `76-77 ms`, busy около `10.9%`, idle около `89.1%`
- под `nsys`:
  - повторный прогон дал median `90.70 ms`
  - `nsys`: total `0.975 s`, busy `19.82%`, idle `80.18%`
  - top CPU Vulkan API:
    - `vkWaitForFences`: `719.31 ms`
    - `vkQueueSubmit`: `108.29 ms`
    - `vkCreateFence`: `0.80 ms`
    - `vkDestroyFence`: `0.54 ms`

### Вывод по этому эксперименту

- Pool/reuse fence-ов оказался полезным:
  - `vkCreateFence` и `vkDestroyFence` практически исчезли из hot path.
  - `vulkan.radixSort` ускорился примерно в `1.9x` относительно текущего async baseline без pool/reuse:
    - `156.63 ms -> 80-83 ms` без профайлера;
    - `167.03 ms -> 90.70 ms` под `nsys`.
- После этого главным CPU-side bottleneck остался `vkWaitForFences`, а вторым — `vkQueueSubmit`.
- Значит следующий кандидат на исправление уже не fence lifecycle, а сама модель submit/wait:
  - уменьшение числа submit-ов;
  - batching нескольких dispatch в один command buffer/submit;
  - возможно, переход с per-launch fence waiting на другой completion-tracking primitive.

### Какой вариант оказался проще

Из двух идей:

- `pool` свободных fence-ов;
- более специфическое `reuse` fence-ов по фиксированным ключам/слотам;

проще в реализации оказался именно общий pool/free-list:

- не требуется привязывать fence к конкретному kernel family или submit slot;
- не нужно отдельно синхронизировать ownership по именованным ключам;
- код локализован в одном месте `VulkanEngine` и почти не затрагивает call sites.

То есть на практике здесь самый простой рабочий вариант — это именно `pool + reuse` как единый механизм.
