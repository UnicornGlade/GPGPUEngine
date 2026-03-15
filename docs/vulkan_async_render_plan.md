# План асинхронного Vulkan rendering поверх уже реализованного async compute

## Кратко

Задача: убрать обязательное блокирующее ожидание после каждого `RenderBuilder::exec()` и перевести Vulkan raster path на тот же принцип bounded async submit, который уже используется для compute kernels.

Внешний API алгоритмов не меняется:
- вызывающий код по-прежнему использует `initRender(...).geometry(...).addAttachment(...).exec(...)`;
- асинхронность остаётся внутренней реализацией Vulkan backend;
- синхронизация происходит в точках host-visible операций, explicit stats report, teardown и при reuse ограниченных runtime-ресурсов.

## Что уже важно учесть из async compute

### 1. Нельзя оценивать оптимизацию вслепую

Перед изменением render path нужно:
- снять baseline на GTX 1650;
- смотреть не только GPU timeline, но и суммарное CPU API time;
- отдельно отмечать expensive API churn вроде `vkCreateFence`, `vkDestroyFence`, `vkWaitForFences`, `vkAllocate*`, `vkFree*`.

### 2. Одна Vulkan queue уже даёт kernel-to-kernel / render-to-render ordering

Для submit-ов в одну queue:
- между compute/render launch-ами не нужен host wait для сохранения порядка исполнения;
- нужны только:
  - lifetime guarantees;
  - resource-conflict tracking для host-visible операций;
  - явный bounded inflight limit;
  - безопасный teardown.

### 3. Временные объекты нельзя уничтожать слишком рано

Async compute уже показал, что launch должен держать alive:
- command buffer;
- fence;
- descriptor sets;
- rassert state;
- все buffers/images, на которые ссылается launch.

Render path требует того же, плюс:
- framebuffer;
- render attachments;
- geometry buffers;
- push-constant copy;
- raster pipeline ref.

### 4. Guard/readback overhead может легко испортить результат

Из profiling async compute уже известно:
- основной регресс был не в самой идее async submit, а в побочном host-side churn;
- отдельные runtime-checks и staging readback могут полностью съесть ожидаемый speedup.

Поэтому для render async:
- не добавлять новых обязательных readback в hot path;
- reuse ограниченных ресурсов делать сразу;
- любые дополнительные profiling markers делать compile-time выключаемыми.

## Целевой scope

Первая версия:
- только Vulkan raster/render path;
- одна queue;
- без multi-queue parallelism;
- без изменения публичного API;
- с сохранением текущего compute async path.

Не делать в `v1`:
- отдельный async API наружу;
- complicated subresource alias analysis;
- сложный scheduler между несколькими queue family.

## Предлагаемая архитектура

### 1. Обобщить inflight launch model с compute-only до unified queue-launch

Вместо отдельной структуры только для compute ввести общую сущность вида:
- `InflightQueueLaunch`

Она должна хранить:
- `kind = Compute | Render`;
- `submit_id`;
- `command_buffer`;
- `fence`;
- descriptor sets;
- optional rassert descriptor set / layout;
- keepalive buffers;
- keepalive images;
- keepalive framebuffer;
- keepalive raster pipeline или compute kernel;
- resource accesses;
- optional timestamp query ids;
- optional rassert slot + owner.

Compute path остаётся на этой же модели, только как один из `kind`.

### 2. Единый inflight queue для compute и render

Нужен один контейнер inflight launch-ей:
- и compute, и render должны попадать в одну очередь;
- host-visible операции должны ждать конфликтующие inflight launch-и любого типа;
- compute и render submit-ы, идущие подряд, не должны заставлять CPU ждать между собой.

Это важно, иначе:
- render сможет конфликтовать с compute вне tracking-а;
- host `readBuffer/readImage` увидит только compute inflight state и пропустит незавершённый render.

### 3. Ограничить общее число inflight launch-ей

Нужен отдельный limit для unified queue-launch model:
- либо общий `max_inflight_launches`;
- либо сохранить compute limit и добавить render limit, но practical вариант проще сделать одним общим лимитом.

Предпочтительный вариант:
- общий `max_inflight_launches`, default `32`.

Причина:
- launch-и живут на одной queue;
- pressure на fence/descriptor/framebuffer ownership общий.

### 4. Resource tracking для render

Для render нужно явно собрать `resource_accesses`.

Минимально консервативно:
- geometry vertex/index buffers: `Read`;
- descriptor buffers/images из `args`: по merged SPIR-V access metadata;
- color attachments:
  - если attachment с clear/load и fragment shader пишет в него: `Write`;
  - если attachment используется с blending: считать `ReadWrite`;
- depth attachment:
  - если depth test включён и writes включены: `ReadWrite`;
  - если depth test включён и writes выключены: `Read`;
  - если только clear/load без shader access: всё равно считать минимум `ReadWrite`, чтобы не ошибиться.

В `v1` tracking остаётся whole-resource, без subrange/subrect analysis.

### 5. Два rassert slot-а и для raster pipeline

Сейчас два slot-а есть только у compute kernel.

Для async render нужно симметрично:
- у `VulkanRasterPipeline` тоже сделать два `rassert` buffer slot-а;
- launch render-а выбирает свободный slot;
- если оба slot-а inflight для того же pipeline, ждать старейший;
- проверять `rassert` строго по slot-у конкретного launch-а.

Это нужно и для correctness, и чтобы не делать sync только ради reuse одного rassert buffer.

### 6. Async statistics и для render

Нужна отдельная статистика по render launch-ам:
- `resetAsyncRenderStats()`
- `getAsyncRenderStats(bool wait_for_all=true)`
- `logAsyncRenderStats(const std::string &label="", bool wait_for_all=true)`

Лучший practical вариант:
- сохранить существующую compute статистику отдельно;
- добавить аналогичную render статистику;
- timestamps собирать отдельно в render query pool.

Плюсы:
- проще интерпретация профилей;
- проще проверить, что speedup у render действительно есть;
- не смешиваются compute и render окна.

### 7. Sync points

Host-visible операции должны ждать конфликтующие inflight launch-и обоих типов:
- `readBuffer`, `writeBuffer`, `resize`, `decref`;
- `readImage`, `writeImage`, `resize`, `decref`;
- explicit stats report;
- teardown;
- pipeline cache clearing / staging buffer clearing / fence clearing.

Кроме того:
- destruction framebuffer/image attachment/pipeline refs не должно происходить до завершения соответствующего launch-а.

## Пошаговая реализация

### Этап 1. Документирование и baseline

- сохранить этот план;
- снять baseline на GTX 1650 по render tests;
- выбрать главный performance-oriented render test.

### Этап 2. Unified inflight launch abstraction

- обобщить compute-only inflight структуру;
- перевести wait/retire/resource-conflict logic на общий inflight queue;
- не менять ещё логику `launchRender`, кроме подготовки к async.

### Этап 3. Async render submit

- заменить `launchRender` с sync `submitCommandBuffer(...)` на async submit;
- добавить keepalive нужных render ресурсов;
- добавить render rassert slots;
- добавить retire/check path для render launch-ей.

### Этап 4. Render statistics и profiling markers

- добавить timestamp queries и logging для render;
- при необходимости добавить отключаемые NVTX ranges для logical phases render submit / retire / readback.

### Этап 5. Тесты

Добавить:
- correctness tests на несколько подряд render submit-ов без промежуточного readback;
- test на reuse одного и того же pipeline и тех же attachments;
- test на конфликт render -> host read;
- test на конфликт render -> compute -> host read;
- test на rassert slot reuse для двух подряд render launch-ей одного pipeline.

Отдельно добавить performance-oriented test:
- много последовательных render launch-ей в один и тот же attachment без промежуточного readback;
- ожидание: заметное ускорение после async render по сравнению с sync baseline.

Самый естественный кандидат:
- вариант на основе `renderRectangleWithBlending`, потому что там уже есть серия одинаковых render launch-ей в один accumulator buffer.

### Этап 6. Замеры и вывод

После реализации:
- прогнать unit tests;
- снять before/after на GTX 1650;
- профилировать `nsys`;
- сравнить:
  - end-to-end time;
  - render async stats;
  - GPU idle/busy;
  - top Vulkan API total time.

## Критерии успеха

- correctness не сломан на существующих Vulkan raster tests;
- новый async render correctness test проходит;
- performance-oriented render test действительно ускоряется на GTX 1650;
- в профиле нет нового большого CPU API churn;
- итоговая архитектура остаётся понятной и не ломает already working async compute path.

## Рабочая гипотеза по ожидаемому speedup

Наиболее вероятный выигрыш должен быть в сценариях, где:
- есть длинная цепочка render launch-ей;
- нет промежуточного host readback;
- используется один и тот же framebuffer/image accumulator;
- раньше CPU блокировался после каждого submit, а теперь сможет быстро накидать несколько launch-ей в queue.

Наиболее слабый выигрыш ожидается там, где:
- сразу после каждого render есть host readback;
- bottleneck уже в GPU fillrate, а не в CPU submit/wait;
- число render launch-ей мало.

## Что реализовано

### 1. Async render submit для `RenderBuilder::exec()`

- `KernelSource::launchRender(...)` больше не обязан делать blocking wait после каждого render submit.
- Добавлен runtime toggle:
  - `GPGPU_ENABLE_VULKAN_ASYNC_RENDER=true|false`
  - default: `true`
- При `false` render path остаётся синхронным fallback-ом для before/after comparison.

### 2. Отдельная bounded inflight render queue

Добавлены:
- `InflightRenderLaunch`
- `inflight_render_launches_`
- `setMaxInflightRenderLaunches(size_t)`
- `waitForAllInflightRenderLaunches()`
- `retireCompletedInflightRenderLaunches()`
- `waitForConflictingInflightRenderLaunches(...)`

Каждый render launch удерживает:
- command buffer;
- fence;
- descriptor sets;
- framebuffer keepalive;
- raster pipeline ref;
- keepalive buffers/images;
- resource accesses;
- timestamp query ids;
- `rassert` slot.

### 3. Render `rassert` c двумя slot-ами

У `VulkanRasterPipeline` добавлены:
- два `rassert` buffer-а;
- `resetRassertCode(slot)`;
- `checkRassertCode(slot)`;
- `acquireRasterRassertSlot(...)`.

Это позволяет запускать несколько inflight render launch-ей одного и того же pipeline без преждевременного reuse одного `rassert` buffer-а.

### 4. Render stats

Добавлены:
- `resetAsyncRenderStats()`
- `getAsyncRenderStats(bool wait_for_all=true)`
- `logAsyncRenderStats(const std::string& label="", bool wait_for_all=true)`

Считаются:
- total;
- busy;
- idle;
- busy/idle %;
- launch count;
- gap count;
- median/max gap;
- waits due to inflight limit.

### 5. Host sync points теперь учитывают и inflight render

`waitForBuffer()` и `waitForImage()` теперь ждут не только inflight compute, но и inflight render launch-и.

Это важно для:
- image readback после render chain;
- buffer/image destruction;
- safe teardown.

## Новые unit tests

Добавлены:
- `vulkan.asyncRenderBlendingChainAccumulatesCorrectly`
  - correctness для длинной цепочки render launch-ей без промежуточного readback;
  - проверяет render stats и итоговое накопление blending output.
- `vulkan.asyncRenderBlendingChainRespectsInflightLimit`
  - correctness при `maxInflightRenderLaunches=2`;
  - проверяет, что backend действительно упирается в inflight limit и аккуратно дожидается старых launch-ей.
- `vulkan.asyncRenderBlendingChainShowsSpeedup`
  - benchmark-oriented unit test;
  - внутри одного теста сравнивает sync fallback (`GPGPU_ENABLE_VULKAN_ASYNC_RENDER=false`) и async path;
  - на NVIDIA GTX 1650 требует заметный speedup.

Дополнительно скорректирован существующий `vulkan.aplusb` test:
- ошибка async compute `rassert` теперь проверяется на explicit sync point `waitForAllInflightComputeLaunches()`, а не жёстко на самом `exec()`.

## Результаты на NVIDIA GeForce GTX 1650

### End-to-end unit tests

Полный прогон:
- `AVK_ENABLE_VALIDATION_LAYERS=false`
- `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`
- `./build/libs/gpu/libgpu_test`

Результат:
- `44/44` tests `PASSED`
- total time: `26.694 s`

### Главный speedup test

`vulkan.asyncRenderBlendingChainShowsSpeedup` на GTX 1650:
- sync: `0.0200728 s`
- async: `0.0101298 s`
- speedup: `1.98x`

Повторный полный прогон дал ещё более сильный результат:
- sync: `0.0218769 s`
- async: `0.00968926 s`
- speedup: `2.26x`

Практический вывод:
- async render действительно оправдан в цепочке многих маленьких render launch-ей без промежуточного readback;
- на GTX 1650 ускорение стабильное и заметное.

### Render stats по correctness test

`vulkan.asyncRenderBlendingChainAccumulatesCorrectly`:
- total `0.00258 .. 0.00310 s`
- busy `53.6% .. 55.5%`
- idle `44.5% .. 46.4%`
- launches `64`

`vulkan.asyncRenderBlendingChainRespectsInflightLimit`:
- launches `128`
- waits_due_to_limit `52 .. 63`
- wait_due_to_limit_total `~1.1 .. 1.4 ms`

## Профилирование

Before/after `nsys` профили сохранены здесь:
- sync fallback before:
  - `artifacts/profiles/async_render/before/async_render_chain_sync.nsys-rep`
  - `artifacts/profiles/async_render/before/async_render_chain_sync.sqlite_db`
- async after:
  - `artifacts/profiles/async_render/after/async_render_chain_async.nsys-rep`
  - `artifacts/profiles/async_render/after/async_render_chain_async.sqlite_db`

Профилировался workload:
- `vulkan.asyncRenderBlendingChainAccumulatesCorrectly`
- в sync режиме через `GPGPU_ENABLE_VULKAN_ASYNC_RENDER=false`
- в async режиме через `GPGPU_ENABLE_VULKAN_ASYNC_RENDER=true`

### Что видно в API totals

Sync fallback:
- `vkWaitForFences`: `75` calls, `9.172 ms`
- `vkQueueSubmit`: `75` calls, `0.273 ms`
- `vkCreateFence`: `5` calls, `0.933 ms`
- `vkDestroyFence`: `5` calls, `0.777 ms`

Async render:
- `vkWaitForFences`: `18` calls, `5.742 ms`
- `vkQueueSubmit`: `75` calls, `0.427 ms`
- `vkCreateFence`: `14` calls, `2.365 ms`
- `vkDestroyFence`: `14` calls, `1.845 ms`

Интерпретация:
- главный выигрыш пришёл именно из сокращения blocking waits;
- submit count не уменьшился, потому что launch granularity пока прежний;
- fence churn для render всё ещё не идеален и остаётся следующей понятной целью оптимизации, если понадобится further speedup.

### Что видно в async NVTX ranges

Async profile показывает:
- `vk async render retire completed launches`: `0.346 ms`, `82` instances
- `vk async render wait fence`: `0.319 ms`, `7` instances
- `vk async render scan fence status`: `0.193 ms`, `139` instances
- `vk async render append interval`: `0.056 ms`, `64` instances

То есть:
- housekeeping сам по себе недорогой;
- текущий path уже достаточно лёгкий, чтобы дать почти `2x` speedup на CPU-bound chain;
- следующий ограничитель — не retire bookkeeping, а per-launch submit granularity и fence lifecycle.

## Вывод

Для render path асинхронность оказалась оправданной:
- correctness сохранён;
- bounded inflight model работает;
- явный speedup на GTX 1650 есть и он заметный;
- полный suite проходит.

При этом есть очевидный следующий резерв:
- reuse/pooling fence-ов и для render hot path;
- возможно batching нескольких render launch-ей, если позже потребуется дальнейшее ускорение.
