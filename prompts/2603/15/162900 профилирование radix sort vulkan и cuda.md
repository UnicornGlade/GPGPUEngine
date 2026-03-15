# Профилирование radix sort Vulkan и CUDA

## Сообщения пользователя

- `новый промпт, отпрофилируй radix sort, скажи в каком файле результаты профилирования, я хочу взглянуть`
- `кроме того скажи свое мнение - насколько это оптимальный результат? можно ли лучше?`
- `сравни со скоростью и результатом профилирования radix sort реализованного на CUDA`
- `посмотри сколько времени и какая утилизация видеокарты, что происходит в пузырях между compute нагрузкой видеокарты?`
- `если надо - воткни дополнительных меток чтобы связать nsys результаты профилирования с логическими операциями ...`
- `... только сделай так чтобы был флажок которым легко выключить все такие метки (чтобы стал нулевой оверхед)`
- `если тут есть хорошие идеи того как вести анализ - внеси эти идеи в AGENTS.md`

## Короткий лог действий

- Сняты Nsight Systems профили для `vulkan.radixSort` и `cuda.radixSort` на GTX 1650.
- Выполнен анализ busy/idle времени, пузырей и CPU-side API overhead.
- Добавлены отключаемые NVTX-метки для связи timeline с логическими операциями radix sort.

## Короткий лог результатов

- Профили лежат в `artifacts/profiles/radixsort_compare_nvtx/`.
- `vulkan.radixSort`: median `0.0885965 s`, внутренний async stats `total=0.0839326 s`, `busy=9.9977%`, `idle=90.0023%`.
- `nsys` по Vulkan: GPU window `974.305 ms`, busy `191.919 ms` (`19.7%`), idle `782.386 ms` (`80.3%`), median gap около `18 us`, max gap около `2.47 ms`.
- `cuda.radixSort`: median `0.00185114 s`, effective bandwidth `4.02487 GB/s`.
- `nsys` по CUDA: GPU window `25.587 ms`, busy `18.814 ms` (`73.5%`), idle `6.773 ms` (`26.5%`), max gap около `1.97 ms`.
- Vulkan radix sort сейчас сильно хуже CUDA на GTX 1650 и выглядит далёким от оптимума; главная проблема — host-side orchestration и многочисленные мелкие submit/wait gaps.
- Добавлены отключаемые NVTX markers через `-DGPGPU_ENABLE_NVTX_MARKERS=ON`, чтобы связать timeline с логическими фазами radix sort и async housekeeping.
- Детализация markers показала, что большая часть `vkWaitForFences` в исходной картине приходила не из самого compute dispatch, а из guard checks при уничтожении временных Vulkan buffer keepalive-объектов.
- Выяснилось, что guards для Vulkan tests включались не в `Context` по умолчанию, а в `libs/gpu/libgpu/vulkan/tests/test_utils.h`.
- После переключения default на `guards off` для Vulkan tests получено сильное ускорение `vulkan.radixSort` на GTX 1650:
  - было примерно `0.0880 s` median;
  - стало `0.014545 s` median;
  - ускорение около `6.0x`.
- Новый профиль с guards-off:
  - `vulkan_radixsort_nvtx_guards_off.nsys-rep`
  - `vkWaitForFences`: `98.8 ms` вместо порядка `703 ms`
  - `vk readBuffer wait fence`: `2.74 ms` вместо порядка `711 ms`
  - async stats: busy `67.8%`, idle `32.2%`
