# План по кэшированию Vulkan pipeline

## Кратко

Нужно добавить in-memory кэширование Vulkan pipeline с понятным lifetime и явной диагностикой:
- generic LRU cache общего назначения;
- compute pipeline cache;
- raster pipeline cache;
- unit tests на корректность, hit/miss и крайние случаи;
- stress test, который показывает заметное ускорение за счёт cache hits.

Главный принцип реализации:
- lifetime pipeline и связанных Vulkan-объектов должен принадлежать `VulkanEngine`, а не отдельному временному `KernelSource`;
- cache должен быть `per kernel family`;
- по умолчанию нужно хранить до `4` variants на одну family;
- для отладки должен быть простой способ узнать hit/miss, общее число cached pipelines и вывести в лог содержимое кэша.

## Шаги реализации

### 1. Generic LRU cache

- Добавить reusable LRU cache без Vulkan-специфики.
- Покрыть его unit tests:
  - вставка и чтение;
  - обновление recency на hit;
  - overwrite существующего ключа;
  - eviction least-recently-used;
  - corner cases по capacity.

### 2. План по Vulkan cache ownership

- Убрать зависимость кэширования compute pipeline от временного `KernelSource::id_`.
- Перенести владение cache entries в `VulkanEngine`.
- Добавить отдельные cache managers для:
  - compute pipeline;
  - raster pipeline.
- Для каждого добавить:
  - `hits`;
  - `misses`;
  - `resetStats()`;
  - `clear()`;
  - число cached pipelines;
  - debug log содержимого кэша.

### 3. Compute pipeline cache

- Использовать stable key, не зависящий от identity конкретного `KernelSource`.
- Кэш должен переиспользовать pipeline между разными объектами `KernelSource` одной shader family внутри одного `gpu::Context`.
- Добавить unit test:
  - сначала functional correctness, который должен проходить на текущей кодовой базе до нового cache API;
  - затем проверка ожидаемых compute cache hits/misses;
  - дополнительные eviction scenarios и corner cases.

### 4. Raster pipeline cache

- Вынести pipeline-affecting render state в стабильный key.
- Кэшировать graphics pipeline и связанные layout/render-pass объекты в `VulkanEngine`.
- Добавить unit tests:
  - reuse при одинаковом pipeline-affecting state;
  - miss при изменении действительно значимых параметров;
  - отсутствие лишнего miss при изменении параметров, не влияющих на pipeline;
  - expected raster cache hits/misses;
  - eviction scenarios.

### 5. Stress test

- Написать benchmark-сценарий, где cache hits должны заметно ускорить host-side preparing time.
- Сравнивать нужно `preparing`/launch overhead, а не только полное wall time.
- Прогон должен логировать:
  - uncached median;
  - cached median;
  - ratio ускорения;
  - итоговые hit/miss.

## Инварианты

- correctness и diagnosability важнее performance tuning;
- если variant не проверен тестом, считать, что он потенциально сломан;
- compute и raster cache stats должны проверяться в конце соответствующих unit tests;
- изменения лучше делать поэтапно:
  - сначала generic cache;
  - потом raster cache;
  - потом compute cache;
  - потом benchmark и финальная проверка.
