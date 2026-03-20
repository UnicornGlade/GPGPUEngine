# JPEG decode benchmark на Vulkan

## Цель

Добавить воспроизводимый Vulkan unit-test для сценария:

1. чтение JPEG-файла с диска в RAM как массива байтов;
2. CPU-декодирование JPEG из памяти;
3. upload распакованного изображения в VRAM;
4. вычисление средней яркости на GPU;
5. readback результата на CPU;
6. сравнение с CPU reference;
7. сбор timings по стадиям и сверка их с `nsys`.

## Текущее решение

- CPU JPEG decode: `libjpeg` из памяти, чтобы disk I/O не смешивался с decode.
- GPU reduction: compute shader по буферу `uint8 RGB/gray` с редукцией суммы яркости в несколько проходов.
- GPU JPEG decode: отдельный Vulkan compute path для узкого, но воспроизводимого baseline-case:
  - grayscale JPEG;
  - baseline DCT;
  - `restart_interval = 1`, то есть один `8x8` block на restart interval;
  - CPU делает JPEG parsing/preprocess и строит полные `16-bit` Huffman LUT;
  - GPU делает entropy decode, dequantization, IDCT и запись grayscale pixels.
- Метрики:
  - медиана CPU decode;
  - медиана host->device upload;
  - медиана GPU compute с явным `waitForAllInflightComputeLaunches()`;
  - медиана device->host readback;
  - CPU reference brightness и GPU result.
- Для быстрых итераций запускать на одной дискретной GPU:
  - `GPGPU_VISIBLE_DEVICES=3`
  - `AVK_ENABLE_VALIDATION_LAYERS=false`
  - в текущем merged device ordering:
    - `2` = `Intel UHD 770`
    - `3` = `NVIDIA GeForce GTX 1650`
    - `4` = `llvmpipe`

## Candidate images

Локально скачаны и сконвертированы в JPEG `quality=95` вне git:

- `kodak_kodim03.jpg`
  - источник: `https://www.r0k.us/graphics/kodak/kodak/kodim03.png`
  - набор: Kodak Lossless True Color Image Suite
  - разрешение: `768x512`
  - размер JPEG: `116052 B`
- `kodak_kodim13.jpg`
  - источник: `https://www.r0k.us/graphics/kodak/kodak/kodim13.png`
  - набор: Kodak Lossless True Color Image Suite
  - разрешение: `768x512`
  - размер JPEG: `246246 B`
- `kodak_kodim23.jpg`
  - источник: `https://www.r0k.us/graphics/kodak/kodak/kodim23.png`
  - набор: Kodak Lossless True Color Image Suite
  - разрешение: `768x512`
  - размер JPEG: `115842 B`
- `sipi_4.1.07.jpg`
  - источник: `https://sipi.usc.edu/database/download.php?vol=misc&img=4.1.07`
  - набор: USC-SIPI Misc, Jelly beans
  - разрешение: `256x256`
  - размер JPEG: `14905 B`
- `sipi_4.2.03.jpg`
  - источник: `https://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03`
  - набор: USC-SIPI Misc, Mandrill
  - разрешение: `512x512`
  - размер JPEG: `181925 B`
- `sipi_4.2.07.jpg`
  - источник: `https://sipi.usc.edu/database/download.php?vol=misc&img=4.2.07`
  - набор: USC-SIPI Misc, Peppers
  - разрешение: `512x512`
  - размер JPEG: `119508 B`
- `sipi_5.3.01.jpg`
  - источник: `https://sipi.usc.edu/database/download.php?vol=misc&img=5.3.01`
  - набор: USC-SIPI Misc, Male
  - разрешение: `1024x1024`
  - размер JPEG: `428600 B`
- `sipi_7.2.01.jpg`
  - источник: `https://sipi.usc.edu/database/download.php?vol=misc&img=7.2.01`
  - набор: USC-SIPI Misc, Airplane
  - разрешение: `1024x1024`
  - размер JPEG: `385791 B`

Итого candidates сейчас занимают `1608869 B`, то есть примерно `1.54 MiB`.

## Проверка и profiling

План проверки:

1. `RelWithDebInfo` build.
2. Локальный запуск нового `gtest` только на одной GPU.
3. `nsys profile` на том же тесте без validation layers.
4. Сопоставление timings из stdout теста с timeline/API trace из `nsys`.

## Итоговые результаты

### Команды запуска

Основной smoke/perf run:

```bash
GPGPU_VISIBLE_DEVICES=3 \
AVK_ENABLE_VALIDATION_LAYERS=false \
GPGPU_VULKAN_JPEG_BENCHMARK_DIR=/tmp/gpgpu_jpeg_candidates_260320/jpeg_q95 \
./cmake-build-relwithdebinfo/libs/gpu/libgpu_test --gtest_filter='vulkan.jpegDecodeBenchmark'
```

`nsys` run:

```bash
GPGPU_VISIBLE_DEVICES=3 \
AVK_ENABLE_VALIDATION_LAYERS=false \
GPGPU_VULKAN_JPEG_BENCHMARK_DIR=/tmp/gpgpu_jpeg_candidates_260320/jpeg_q95 \
nsys profile --trace=vulkan,nvtx,osrt,cuda --sample=none --cpuctxsw=none --force-overwrite true --export sqlite \
    -o /tmp/nsys_jpeg_vulkan_gtx1650_nvtx \
    ./cmake-build-relwithdebinfo/libs/gpu/libgpu_test --gtest_filter='vulkan.jpegDecodeBenchmark'
```

### Что получилось

Тест проходит на `NVIDIA GeForce GTX 1650`.

По stdout benchmark:

- `256x256`:
  - CPU decode около `0.22 ms`
  - upload около `0.13 ms`
  - GPU reduction около `0.11 ms`
  - readback около `0.04 ms`
- `512x512`:
  - CPU decode около `1.38-2.09 ms`
  - upload около `0.61-0.62 ms`
  - GPU reduction около `0.26-0.38 ms`
  - readback около `0.07-0.29 ms`
- `768x512`:
  - CPU decode около `1.64-2.84 ms`
  - upload около `0.72-0.78 ms`
  - GPU reduction около `0.25-0.48 ms`
  - readback около `0.05-0.10 ms`
- `1024x1024`:
  - CPU decode около `5.49-5.58 ms`
  - upload около `1.38-1.45 ms`
  - GPU reduction около `0.52-0.60 ms`
  - readback около `0.17-0.24 ms`

### `nsys` summary

`VULKAN_WORKLOAD`:

- workloads: `476`
- total timeline span: `152624915 ns`
- GPU busy: `18655676 ns`
- GPU idle: `133969239 ns`
- busy: `12.22%`
- idle: `87.78%`

`nvtx_sum` по фазам:

- `jpeg cpu decode`: median `1884317.5 ns` (`1.88 ms`)
- `jpeg upload`: median `718495 ns` (`0.72 ms`)
- `jpeg gpu reduce`: median `339695 ns` (`0.34 ms`)
- `jpeg readback`: median `102368 ns` (`0.10 ms`)

`vulkan_api_sum`:

- `vkWaitForFences`: median `198639 ns` (`0.20 ms`), total `76.50 ms`
- `vkQueueSubmit`: median `5197 ns`, total `3.55 ms`
- `vkAllocateDescriptorSets`: median `636 ns`, but long tail up to `848492 ns`

### Интерпретация

- Числа из stdout и из `nsys` согласуются по порядку величин.
- На этих размерах доминирует не GPU compute, а CPU JPEG decode.
- Upload через PCI-E уже заметный, но всё ещё ощутимо дешевле decode.
- Сам GPU reduction дешёвый и для всех картинок остаётся меньше `1 ms`.
- По `VULKAN_WORKLOAD` видно, что GPU большую часть общего wall-clock времени простаивает, потому что pipeline здесь в основном host-driven и последовательно синхронизируется между фазами.

## Второй этап: GPU JPEG decode benchmark и correctness

### Open-source references и практические выводы

- `GPUJPEG`: mature CUDA/OpenGL decoder/encoder, ориентирован на baseline JPEG и активно использует restart markers для распараллеливания entropy decode.
  - repo: `https://github.com/CESNET/GPUJPEG`
- `compeg`: baseline JPEG decoder на `WebGPU`, где CPU делает parsing и подготовку scan-метаданных, а GPU делает Huffman decode и DCT/IDCT.
  - repo: `https://github.com/sludgephd/compeg`
- Общий практический паттерн у перспективных GPU решений оказался один и тот же:
  - не пытаться делать весь JPEG parser на GPU;
  - на CPU разобрать markers/tables/scan structure;
  - обеспечить coarse-grained параллелизм через restart intervals;
  - на GPU выносить entropy decode, dequantization и IDCT;
  - отдельно измерять upload compressed bitstream, decode, postprocess, readback.

### Что именно реализовано в репозитории

- `vulkan.jpegDecodeBenchmark`
  - старый benchmark-path, где JPEG decode происходит на CPU;
  - в stdout теперь явно подписано `cpu jpeg decode`.
- `vulkan.jpegDecodeBenchmarkGpu`
  - на GPU отправляется уже закодированный grayscale JPEG;
  - decode происходит в Vulkan compute shader;
  - после decode на GPU считается средняя яркость;
  - в stdout явно подписано `gpu jpeg decode`.
- `vulkan.jpegDecodeGpuMatchesCpu`
  - проверяет близость CPU decode и GPU decode по pixel-wise `avg_abs_diff` и `max_abs_diff`.

### Ограничения текущего GPU decoder prototype

- Пока поддержан только `grayscale baseline JPEG`.
- Для надёжного распараллеливания тест сам генерирует временный grayscale JPEG c `restart_interval = 1` в `/tmp/gpgpu_vulkan_gpu_jpeg_decode_dataset`.
- Исходные source images остаются обычными JPEG; узкий GPU-friendly JPEG строится автоматически перед benchmark/correctness run.
- На практике это хороший prototype для измерений Vulkan path, но ещё не production-ready universal JPEG decoder.

### Команды запуска

`NVIDIA GeForce GTX 1650`, validation layers enabled:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
AVK_ENABLE_VALIDATION_LAYERS=true \
GPGPU_VULKAN_JPEG_BENCHMARK_DIR=/tmp/gpgpu_jpeg_candidates_260320/jpeg_q95 \
./cmake-build-relwithdebinfo/libs/gpu/libgpu_test \
    --gtest_filter='vulkan.jpegDecodeBenchmarkGpu:vulkan.jpegDecodeGpuMatchesCpu'
```

`Intel UHD 770`, validation layers enabled:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json \
AVK_ENABLE_VALIDATION_LAYERS=true \
GPGPU_VULKAN_JPEG_BENCHMARK_DIR=/tmp/gpgpu_jpeg_candidates_260320/jpeg_q95 \
./cmake-build-relwithdebinfo/libs/gpu/libgpu_test \
    --gtest_filter='vulkan.jpegDecodeBenchmarkGpu:vulkan.jpegDecodeGpuMatchesCpu'
```

### Результаты на NVIDIA GeForce GTX 1650

Оба новых теста проходят с validation layers.

Медианы для `vulkan.jpegDecodeBenchmarkGpu`:

- `kodak_kodim03_gray_rst1_q95.jpg`
  - upload compressed: `0.55 ms`
  - GPU JPEG decode: `0.83 ms`
  - GPU brightness reduce: `0.11 ms`
  - readback: `0.04 ms`
- `kodak_kodim13_gray_rst1_q95.jpg`
  - upload compressed: `0.43 ms`
  - GPU JPEG decode: `0.93 ms`
  - GPU brightness reduce: `0.10 ms`
  - readback: `0.03 ms`
- `kodak_kodim23_gray_rst1_q95.jpg`
  - upload compressed: `0.38 ms`
  - GPU JPEG decode: `0.87 ms`
  - GPU brightness reduce: `0.11 ms`
  - readback: `0.03 ms`
- `sipi_4.1.07_gray_rst1_q95.jpg`
  - upload compressed: `0.34 ms`
  - GPU JPEG decode: `0.24 ms`
  - GPU brightness reduce: `0.07 ms`
  - readback: `0.03 ms`
- `sipi_4.2.03_gray_rst1_q95.jpg`
  - upload compressed: `0.40 ms`
  - GPU JPEG decode: `0.75 ms`
  - GPU brightness reduce: `0.09 ms`
  - readback: `0.03 ms`
- `sipi_4.2.07_gray_rst1_q95.jpg`
  - upload compressed: `0.38 ms`
  - GPU JPEG decode: `0.72 ms`
  - GPU brightness reduce: `0.10 ms`
  - readback: `0.03 ms`
- `sipi_5.3.01_gray_rst1_q95.jpg`
  - upload compressed: `0.53 ms`
  - GPU JPEG decode: `2.57 ms`
  - GPU brightness reduce: `0.13 ms`
  - readback: `0.03 ms`
- `sipi_7.2.01_gray_rst1_q95.jpg`
  - upload compressed: `0.54 ms`
  - GPU JPEG decode: `2.51 ms`
  - GPU brightness reduce: `0.13 ms`
  - readback: `0.03 ms`

### Результаты на Intel UHD 770

Оба новых теста тоже проходят с validation layers.

Медианы для `vulkan.jpegDecodeBenchmarkGpu`:

- `kodak_kodim03_gray_rst1_q95.jpg`
  - upload compressed: `2.07 ms`
  - GPU JPEG decode: `2.25 ms`
  - GPU brightness reduce: `0.70 ms`
  - readback: `0.19 ms`
- `kodak_kodim13_gray_rst1_q95.jpg`
  - upload compressed: `1.42 ms`
  - GPU JPEG decode: `2.31 ms`
  - GPU brightness reduce: `0.60 ms`
  - readback: `0.15 ms`
- `kodak_kodim23_gray_rst1_q95.jpg`
  - upload compressed: `1.33 ms`
  - GPU JPEG decode: `1.85 ms`
  - GPU brightness reduce: `0.53 ms`
  - readback: `0.11 ms`
- `sipi_4.1.07_gray_rst1_q95.jpg`
  - upload compressed: `1.34 ms`
  - GPU JPEG decode: `1.13 ms`
  - GPU brightness reduce: `0.24 ms`
  - readback: `0.11 ms`
- `sipi_4.2.03_gray_rst1_q95.jpg`
  - upload compressed: `1.28 ms`
  - GPU JPEG decode: `1.85 ms`
  - GPU brightness reduce: `0.38 ms`
  - readback: `0.11 ms`
- `sipi_4.2.07_gray_rst1_q95.jpg`
  - upload compressed: `1.42 ms`
  - GPU JPEG decode: `1.72 ms`
  - GPU brightness reduce: `0.39 ms`
  - readback: `0.11 ms`
- `sipi_5.3.01_gray_rst1_q95.jpg`
  - upload compressed: `2.01 ms`
  - GPU JPEG decode: `5.57 ms`
  - GPU brightness reduce: `1.72 ms`
  - readback: `0.70 ms`
- `sipi_7.2.01_gray_rst1_q95.jpg`
  - upload compressed: `4.60 ms`
  - GPU JPEG decode: `6.57 ms`
  - GPU brightness reduce: `2.01 ms`
  - readback: `0.74 ms`

### Корректность CPU decode vs GPU decode

На обеих GPU фактическая картина одинакова:

- `avg_abs_diff` держится около `0.495-0.504`
- `max_abs_diff = 1`

Это выглядит как детерминированная разница в rounding policy у IDCT, а не как логическая ошибка decode.

### Интерпретация второго этапа

- Текущий Vulkan JPEG decode prototype работает воспроизводимо и проходит validation layers и на NVIDIA, и на Intel.
- Для этой узкой baseline/grayscale/restart-friendly формы JPEG путь уже корректен и измерим.
- На `GTX 1650` GPU JPEG decode для `1024x1024` занимает около `2.5-2.6 ms`, то есть он уже сопоставим с предыдущим CPU decode path, но не радикально быстрее end-to-end из-за host uploads и узости текущего формата.
- На `Intel UHD 770` тот же путь заметно медленнее и по decode, и по upload/readback.
- Следующий практический шаг, если нужен универсальный decoder:
  - добавить поддержку color JPEG и restart intervals больше `1`;
  - либо держать portable Vulkan path как fallback, а аппаратно-специфичный `nvJPEG` использовать как optional fast-path.

## Третий этап: color baseline JPEG и большие изображения

### Что добавлено

- Новый Vulkan kernel для color baseline JPEG:
  - `YCbCr`
  - baseline DCT
  - interleaved scan
  - `restart_interval = 1`
  - поддержаны `4:2:0` и `4:4:4`
- Новый benchmark test:
  - `vulkan.jpegDecodeBenchmarkGpuColor`
- Новый correctness test:
  - `vulkan.jpegDecodeGpuColorMatchesCpu`
- Новый reduction kernel для `packed RGB8 -> u32`.
- Для больших изображений поднят глобальный Vulkan fence timeout в test runtime:
  - было `1 s`
  - стало `30 s`
  - иначе long-running compute decode на `9-64 MP` кадрах падал не по ошибке алгоритма, а по timeout harness-а.

### Ограничения текущего color path

- Это всё ещё prototype, а не оптимизированный decoder.
- Для цветного JPEG pipeline сейчас такой:
  - CPU разбирает markers, DQT, DHT, SOS, restart markers;
  - CPU строит full Huffman LUT и стартовые позиции по MCU;
  - GPU декодирует MCU, делает dequantization, IDCT, upsampling chroma и `YCbCr -> RGB`.
- Основной bottleneck теперь уже не upload/readback, а сам compute shader.

### Результаты на малом наборе, color path

`GTX 1650`, validation layers enabled, `4:2:0` benchmark:

- `256x256`: decode около `25.2 ms`
- `512x512`: decode около `51.7 ms`
- `768x512`: decode около `75.2-75.6 ms`
- `1024x1024`: decode около `204.3-204.5 ms`

`UHD 770`, validation layers enabled, `4:2:0` benchmark:

- `256x256`: decode около `480 ms`
- `512x512`: decode около `494-495 ms`
- `768x512`: decode около `503-526 ms`
- `1024x1024`: decode около `700-702 ms`

Color correctness на обеих GPU проходит:

- `4:2:0`
  - `avg_abs_brightness_diff` около `0.84-1.05`
  - `max_abs_brightness_diff` до `10.0`
  - `avg_abs_channel_diff` около `0.86-2.22`
  - `max_abs_channel_diff` до `42`
- `4:4:4`
  - на проверенных images `avg_abs_brightness_diff` около `0.77-0.86`
  - `max_abs_brightness_diff` около `2.33`
  - `max_abs_channel_diff` около `3`

### Большие candidate images

Локально собраны вне git:

- `/tmp/gpgpu_large_image_candidates_260320/nasa_earth_from_orbit_8000x8000.jpg`
  - источник: `NASA SVS Earth from Orbit`
  - разрешение: `8000x8000`
  - размер: `17224423 B`
  - класс: `64 MP`
- `/tmp/gpgpu_large_image_candidates_260320/commons_6000x4000.jpg`
  - источник: `Wikimedia Commons, Tentacles in the sky`
  - разрешение: `6000x4000`
  - размер: `5977973 B`
  - класс: `24 MP`
- `/tmp/gpgpu_large_jpeg_sources_260320/jpegai14_3680x2456_q95.jpg`
  - source PNG: `JPEG-AI MMSP test set, jpegai14`
  - разрешение: `3680x2456`
  - размер JPEG: `3674953 B`
  - класс: `9.04 MP`

### Большие результаты на GTX 1650

`4:2:0` benchmark:

- `jpegai14_3680x2456_q95`
  - upload compressed: `2.42 ms`
  - GPU JPEG decode: `1684.17 ms`
  - GPU brightness reduce: `0.56 ms`
  - readback: `0.026 ms`
- `commons_6000x4000`
  - upload compressed: `5.24 ms`
  - GPU JPEG decode: `4473.94 ms`
  - GPU brightness reduce: `1.31 ms`
  - readback: `0.057 ms`
- `nasa_earth_from_orbit_8000x8000`
  - upload compressed: `12.47 ms`
  - GPU JPEG decode: `11957.93 ms`
  - GPU brightness reduce: `5.39 ms`
  - readback: `0.72 ms`

Correctness для больших кадров:

- `jpegai14_3680x2456_q95`, `4:2:0`
  - `avg_abs_brightness_diff = 0.861290`
  - `max_abs_brightness_diff = 10.000000`
- `commons_6000x4000`, `4:2:0`
  - `avg_abs_brightness_diff = 0.861570`
  - `max_abs_brightness_diff = 3.333333`
- `nasa_earth_from_orbit_8000x8000`, `4:2:0`
  - `avg_abs_brightness_diff = 0.777058`
  - `max_abs_brightness_diff = 7.000000`
- `commons_6000x4000`, `4:4:4`
  - `avg_abs_brightness_diff = 0.859354`
  - `max_abs_brightness_diff = 2.333333`
- `nasa_earth_from_orbit_8000x8000`, `4:4:4`
  - `avg_abs_brightness_diff = 0.768063`
  - `max_abs_brightness_diff = 2.333333`

### Что это означает

- Portable Vulkan color decoder уже корректен, но performance сейчас плохой.
- На `GTX 1650` рост decode-time почти линейный по числу пикселей и быстро уходит в секунды:
  - `~9 MP`: `1.68 s`
  - `24 MP`: `4.47 s`
  - `64 MP`: `11.96 s`
- На `UHD 770` даже `~1 MP` уже стоит около `0.5-0.7 s`, то есть large-image profiling на этой реализации сейчас инженерно малополезен без серьёзной оптимизации kernel-а.
- Главный practical вывод:
  - этот path годится как portable correctness/fallback prototype;
  - для performance-нужд нужен либо более параллельный entropy path, либо hardware/vendor-specific fast-path вроде `nvJPEG`.

### Остаточный issue

- Старый `vulkan.jpegDecodeBenchmark` для very large RGB frames пока не доведён:
  - legacy raw-byte GPU reduction даёт некорректный итог на больших dispatch-ах;
  - для текущей задачи основные большие numbers снимались через новый color decode benchmark;
  - если нужно, следующий отдельный шаг — переписать старый benchmark на packed-RGB reduction path, чтобы и CPU-decode variant масштабировался на `9-64 MP`.

## План оптимизации color path

Цель — не “сделать красиво”, а последовательно убрать самые дорогие места с минимальным риском регрессий.

### Шаг 1. Специализированный fast-path для `4:2:0`

- убрать из shader runtime-динамику по sampling factors;
- убрать лишние private arrays, рассчитанные на общий случай;
- сделать отдельный kernel под самый важный practical case:
  - `YCbCr 4:2:0`
  - `restart_interval = 1`
  - 4 Y blocks + 1 Cb + 1 Cr на MCU.

Ожидание:

- уменьшение register pressure;
- меньше private-memory spill;
- меньше branchy address arithmetic;
- заметный выигрыш на всех size classes без изменения общей архитектуры.

Фактический результат:

- `GTX 1650`, малый набор, `4:2:0`
  - было примерно `25 ms .. 204 ms`
  - стало примерно `1.0 ms .. 4.9 ms`
- `UHD 770`, малый набор, `4:2:0`
  - было примерно `480 ms .. 702 ms`
  - стало примерно `6.0 ms .. 10.5 ms`

То есть первый шаг дал ускорение на порядок и больше.

### Шаг 2. Повторный замер

- GTX 1650
- маленький набор
- `9 MP`, `24 MP`, `64 MP`

Если выигрыш недостаточный, переход к следующему шагу.

### Шаг 3. Разделение decode на стадии

Разбить monolithic color kernel на несколько kernels:

1. entropy decode + IDCT -> planar `Y/Cb/Cr`;
2. `YCbCr420 -> RGB`.

Ожидание:

- лучшее распределение регистров;
- меньше live-state на invocation;
- возможность отдельно оптимизировать hot stage;
- лучшая observability по timings.

Фактический результат для `4:2:0` после этого шага:

- `GTX 1650`, `3680x2456`
  - было `87.39 ms`
  - стало `53.29 ms`
- `GTX 1650`, `6000x4000`
  - было `102.20 ms`
  - стало `70.90 ms`
- `GTX 1650`, `8000x8000`
  - было `280.07 ms`
  - стало `148.44 ms`

Correctness сохранился в допустимом диапазоне:

- большие `4:2:0` кадры теперь дают `avg_abs_brightness_diff` примерно `0.90 .. 1.21`
- `max_abs_brightness_diff` остаётся в пределах `~3.67 .. 10.33`
- channel-wise различия остаются умеренными для lossy JPEG path

Это уже означает, что staged split действительно полезен, а не просто “архитектурно красив”.

### Шаг 4. Дальше только по profile data

Фактический шаг, который уже проверен:

- `4:2:0` path дополнительно разрезан на:
  1. `entropy decode + dequant -> coefficients`
  2. `coefficients -> planar Y/Cb/Cr420`
  3. `YCbCr420 -> RGB`
- ключевой practical effect:
  - из hot kernel ушли одновременно и bitstream parsing, и IDCT, и RGB conversion;
  - сильно упал live-state на одно invocation;
  - появилась возможность независимо измерять coefficient stage и postprocess stages в будущем.

Фактический результат для `4:2:0` на `GTX 1650`:

- малый набор:
  - `256x256 .. 768x512`: `~0.76 .. 0.91 ms`
  - `1024x1024`: `~1.53 .. 1.55 ms`
- большие кадры:
  - `3680x2456`
    - было `53.29 ms`
    - стало `27.71 ms`
  - `6000x4000`
    - было `70.90 ms`
    - стало `33.60 ms`
  - `8000x8000`
    - было `148.44 ms`
    - стало `84.86 ms`

Сравнение CPU vs GPU decode для больших JPEG:

- `3680x2456`
  - CPU decode: `70.58 ms`
  - GPU decode: `27.71 ms`
  - ускорение GPU: примерно `2.55x`
- `6000x4000`
  - CPU decode: `156.39 ms`
  - GPU decode: `33.60 ms`
  - ускорение GPU: примерно `4.65x`
- `8000x8000`
  - CPU decode: `365.50 ms`
  - GPU decode: `84.86 ms`
  - ускорение GPU: примерно `4.31x`

Correctness после coefficient split:

- `GTX 1650`, большие `4:2:0` кадры:
  - `avg_abs_brightness_diff` примерно `0.90 .. 1.21`
  - `max_abs_brightness_diff` примерно `3.67 .. 10.33`
  - `avg_abs_channel_diff` примерно `1.24 .. 1.41`
  - `max_abs_channel_diff` примерно `10 .. 42`
- validation layers снова прогнаны отдельно на:
  - `Intel UHD 770`
  - `NVIDIA GeForce GTX 1650`
- новых validation issues не обнаружено.

Следующие возможные меры:

- integer IDCT вместо float;
- более узкие типы коэффициентов;
- workgroup-cooperative IDCT;
- более компактное представление preprocessed scan metadata;
- optional vendor fast-path (`nvJPEG`) поверх portable fallback.

## Дополнительное исследование hot stages

Чтобы не оптимизировать вслепую дальше, в benchmark-path добавлены отдельные timings для стадий:

- `entropy -> coeffs`
- `coeffs -> ycbcr`
- `ycbcr -> rgb`

На `GTX 1650` именно `coeffs -> ycbcr` стабильно доминирует:

- `gnome_4096`
  - total decode: `18.79 ms`
  - `entropy -> coeffs`: `5.86 ms`
  - `coeffs -> ycbcr`: `11.02 ms`
  - `ycbcr -> rgb`: `1.87 ms`
- `gnome_8192`
  - total decode: `70.00 ms`
  - `entropy -> coeffs`: `17.64 ms`
  - `coeffs -> ycbcr`: `45.70 ms`
  - `ycbcr -> rgb`: `6.67 ms`

Практический вывод:

- следующий реальный bottleneck уже не Huffman path и не final RGB conversion;
- основное время сидит в IDCT + planar write stage;
- значит дальнейшие meaningful optimizations нужно делать именно там.

### Проверенные, но неперспективные гипотезы

1. Упаковка промежуточных `Y/Cb/Cr` planes по 4 байта в `uint32`

- гипотеза была уменьшить inter-kernel bandwidth между `coeffs -> ycbcr` и `ycbcr -> rgb`;
- на практике получилось хуже:
  - `24 MP`: примерно `33.6 ms -> 101.7 ms`
  - `64 MP`: примерно `84.9 ms -> 276.7 ms`
- correctness не ломался, но performance regression слишком большой, поэтому подход сразу откатан.

2. Уменьшение `local_size_x` с `256` до `64` для двух тяжёлых `4:2:0` kernels

- A/B на том же synthetic dataset дал практически нулевой эффект:
  - `gnome_4096`: `18.77 ms` vs `18.79 ms`
  - `gnome_8192`: `70.06 ms` vs `70.00 ms`
- значит workgroup-size tuning здесь не является meaningful lever и в текущем виде не даёт measurable gain.

### Проверенная полезная гипотеза: zero-AC metadata из stage 1

Дальше была проверена более точная версия sparsity fast-path:

- сначала измерена доля блоков, у которых `all AC = 0`;
- затем этот флаг стали вычислять прямо в `entropy -> coeffs`, где coefficients уже и так читаются;
- в `coeffs -> ycbcr` kernel теперь читается готовый `all_ac_zero` flag и для таких блоков выполняется constant-fill по DC, без дополнительного сканирования `64` коэффициентов.

Что важно:

- naive version этой идеи была плохой:
  - если заново сканировать `64` коэффициентов прямо в hot stage, становится только хуже;
- но версия с precomputed metadata оказалась уже правильной.

Измеренная sparsity на synthetic `gnome` dataset:

- `gnome_4096`: `all_ac_zero_blocks=0.7350`
- `gnome_8192`: `all_ac_zero_blocks=0.7650`

Результат на `GTX 1650` после переноса флага в stage 1:

- `gnome_4096`
  - было `18.79 ms`
  - стало `13.89 ms`
  - `coeffs -> ycbcr`: `11.02 ms -> 4.90 ms`
- `gnome_8192`
  - было `70.00 ms`
  - стало `45.83 ms`
  - `coeffs -> ycbcr`: `45.70 ms -> 17.34 ms`

Correctness после этого шага перепроверен:

- `GTX 1650`, validation layers enabled: small-set correctness проходит
- `Intel UHD 770`, validation layers enabled: small-set correctness проходит

Практический вывод:

- sparsity сама по себе здесь действительно полезна;
- но metadata о sparsity нужно вычислять в том kernel-е, который уже touching coefficients, а не повторять expensive scan в следующей стадии;
- это первый confirmed optimization после coefficient split, который даёт значимый выигрыш именно по hot stage.

### Следующий осмысленный шаг

Следующий practical candidate теперь выглядит так:

- переписать `coeffs -> ycbcr` stage на integer IDCT;
- по возможности сократить private state и float pressure в `idctBlock`;
- только после этого снова сравнить timings по тем же stage markers.

## Benchmark set по размерным классам

Для отдельной итерации profiling/benchmark был собран временный набор в `/tmp/gpgpu_jpeg_benchmark_sizes_260321`:

- `small_sipi_4.2.03.jpg`
  - `512x512`
  - `0.26 MP`
  - `190020 B`
  - источник: `https://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03`
- `mp4_first_night_mary_river.jpg`
  - `2304x1728`
  - `3.98 MP`
  - `1862972 B`
  - источник: `https://commons.wikimedia.org/wiki/Special:Redirect/file/First%20night%20at%20Mary%20River%20%289103309112%29.jpg`
- `k4_ts_2014_12_23_2268.jpg`
  - `3840x2160`
  - `8.29 MP`
  - `3462089 B`
  - источник: `https://commons.wikimedia.org/wiki/Special:Redirect/file/TS%202014-12-23-2268%20%2815464308094%29.jpg`
- `mp10_jpegai14_3680x2456.jpg`
  - `3680x2456`
  - `9.04 MP`
  - `2958293 B`
  - source PNG zip: `https://jpegai.github.io/public/test_set/00014_TE_3680x2456.png.zip`
  - локально сконвертирован в JPEG для benchmark
- `mp20_namwon_nongak.jpg`
  - `5594x3729`
  - `20.86 MP`
  - `5010202 B`
  - источник: `https://commons.wikimedia.org/wiki/Special:Redirect/file/%EB%82%A8%EC%9B%90%EB%86%8D%EC%95%85.jpg`
- `mp40_vaz_2109_8000x6000.jpg`
  - `8000x6000`
  - `48.00 MP`
  - `18334341 B`
  - источник: `https://commons.wikimedia.org/wiki/Special:Redirect/file/VAZ%202109%208000x6000.jpg`
- `mp60_earth_from_orbit_8000x8000.jpg`
  - `8000x8000`
  - `64.00 MP`
  - `17224423 B`
  - источник: `https://svs.gsfc.nasa.gov/vis/a010000/a011200/a011268/cover-original.jpg`

## GTX 1650 benchmark на размерном ряду

Benchmark path:

- `GPGPU_VISIBLE_DEVICES=3`
- `AVK_ENABLE_VALIDATION_LAYERS=false`
- `vulkan.jpegDecodeBenchmarkGpuColor`

Для apples-to-apples CPU comparison отдельно замерялся decode из памяти тех же generated `*_rgb_420_rst1_q95.jpg`, которые benchmark отправляет на GPU.

Результаты:

- `0.26 MP`, `512x512`, `all_ac_zero_blocks=0.0000`
  - CPU decode: `1.91 ms`
  - GPU decode: `1.46 ms`
  - upload: `1.70 ms`
  - reduce: `0.12 ms`
  - readback: `0.05 ms`
- `3.98 MP`, `2304x1728`, `all_ac_zero_blocks=0.0383`
  - CPU decode: `19.26 ms`
  - GPU decode: `6.70 ms`
  - upload: `3.28 ms`
  - reduce: `0.61 ms`
  - readback: `0.20 ms`
- `8.29 MP (4K)`, `3840x2160`, `all_ac_zero_blocks=0.0258`
  - CPU decode: `39.39 ms`
  - GPU decode: `12.29 ms`
  - upload: `4.18 ms`
  - reduce: `0.80 ms`
  - readback: `0.27 ms`
- `9.04 MP`, `3680x2456`, `all_ac_zero_blocks=0.0471`
  - CPU decode: `39.65 ms`
  - GPU decode: `13.07 ms`
  - upload: `4.17 ms`
  - reduce: `0.81 ms`
  - readback: `0.28 ms`
- `20.86 MP`, `5594x3729`, `all_ac_zero_blocks=0.0349`
  - CPU decode: `95.94 ms`
  - GPU decode: `27.75 ms`
  - upload: `5.55 ms`
  - reduce: `1.30 ms`
  - readback: `0.35 ms`
- `48.00 MP`, `8000x6000`, `all_ac_zero_blocks=0.0855`
  - CPU decode: `216.05 ms`
  - GPU decode: `56.43 ms`
  - upload: `9.08 ms`
  - reduce: `2.41 ms`
  - readback: `0.36 ms`
- `64.00 MP`, `8000x8000`, `all_ac_zero_blocks=0.4218`
  - CPU decode: `301.28 ms`
  - GPU decode: `54.79 ms`
  - upload: `11.05 ms`
  - reduce: `3.06 ms`
  - readback: `0.28 ms`

Практические наблюдения:

- GPU decode уже быстрее CPU на всём диапазоне, кроме совсем маленького `512x512`, где разница несущественная.
- Самый большой выигрыш получается там, где у image много `all_ac_zero` blocks:
  - `64 MP` Earth image оказался быстрее `48 MP` VAZ image, хотя пикселей больше.
- Это подтверждает, что content/sparsity для JPEG decode сейчас важны почти так же, как и pure pixel count.

## `nsys` profile на `64 MP`

Отдельно был снят `nsys` trace на `mp60_earth_from_orbit_8000x8000.jpg`.

Code-level timings:

- upload: `11.29 ms`
- GPU decode: `55.05 ms`
  - `entropy -> coeffs`: `24.47 ms`
  - `coeffs -> ycbcr`: `24.18 ms`
  - `ycbcr -> rgb`: `6.36 ms`
- reduce: `3.16 ms`
- readback: `0.14 ms`

`nsys` `nvtx_sum` подтверждает ту же картину:

- `jpeg gpu decode color`: median `54.91 ms`
- `jpeg gpu upload compressed color`: median `11.23 ms`
- `jpeg gpu brightness reduce color`: median `3.14 ms`
- `jpeg gpu brightness readback color`: median `0.14 ms`

`nsys` `vulkan_api_sum`:

- `vkWaitForFences`: `742.30 ms`, `83.4%` API времени
- `vkQueueSubmit`: `6.25 ms`
- `vkAllocateMemory`: `12.02 ms`
- `vkFreeMemory`: `8.88 ms`

Интерпретация:

- profiling хорошо согласуется с code-level timers;
- главный runtime cost всё ещё внутри compute stages, а не в upload/readback;
- но на уровне Vulkan API очень хорошо видно, что test/benchmark orchestration намеренно тратит много wall-clock в `vkWaitForFences`;
- для реального throughput path следующий большой engineering reserve — меньше синхронизаций и больше batching/pipelining между dispatch-ами и image iterations.

## Следующие practical ideas

После этого profiling шага наиболее реалистичные идеи такие:

1. Integer IDCT в `coeffs -> ycbcr`

- это всё ещё главный hot stage почти на всех content classes;
- float-heavy IDCT с большими private arrays выглядит как основной compute bottleneck.

2. Richer sparsity metadata из `entropy -> coeffs`

- `all_ac_zero` уже дал хороший выигрыш;
- следующий шаг может быть:
  - `last_nonzero_idx`
  - или несколько coarse sparsity classes
- это позволит делать не только DC-only shortcut, но и более дешёвые reduced IDCT variants для “почти пустых” blocks.

3. Меньше forced waits в benchmark/throughput path

- `vkWaitForFences` доминирует API time;
- для реального pipeline имеет смысл:
  - собирать несколько dispatch stages в один async batch;
  - readback делать реже;
  - мерить throughput отдельно от fully-synchronized per-stage latency.

4. Подумать о fuse между `coeffs -> ycbcr` и `ycbcr -> rgb` только после integer IDCT

- сейчас split полезен для observability и occupancy;
- blind fusion делать рано;
- но после удешевления IDCT стоит перепроверить, не станет ли planar write/read уже заметным bottleneck.

## Исследование трёх запланированных ускорений

На этом шаге были исследованы три отдельные ветки:

1. `integer IDCT`
2. `richer sparsity metadata`
3. `less waits / batching`

### `integer IDCT`

Для проверки был добавлен отдельный experimental kernel `jpeg_decode_color_420_coeffs_to_ycbcr_int.comp`.

На `GTX 1650` он оказался непригодным в текущем виде:

- correctness:
  - `gnome_small`
  - `avg_abs_brightness_diff=25.411930`
  - `max_abs_brightness_diff=230.333333`
  - `avg_abs_channel_diff=27.460686`
  - `max_abs_channel_diff=250`
- benchmark:
  - `gnome_4096`
  - float baseline decode `13.30 ms`
  - integer prototype decode `13.62 ms`
  - float `coeffs -> ycbcr` `4.82 ms`
  - integer prototype `coeffs -> ycbcr` `5.13 ms`

Вывод:

- наивный fixed-point port текущего float butterfly не дал speedup;
- correctness у него сильно хуже допустимого;
- если продолжать эту ветку, то уже нужен не approximate port, а более faithful перенос `libjpeg/jidctint`-style integer IDCT с точной схемой shifts/consts.

### `richer sparsity metadata`

Для проверки был добавлен отдельный exact-path:

- `jpeg_decode_color_420_to_coeffs_colmeta.comp`
- `jpeg_decode_color_420_coeffs_to_ycbcr_colmeta.comp`

Он передаёт из `entropy -> coeffs` две bitmap metadata:

- `nonzero_col_mask`
- `nonzero_rows_gt0_col_mask`

И использует их для двух exact shortcuts:

- полностью нулевые natural columns
- columns, где активен только `row0`

Correctness у этого варианта нормальный:

- `gnome_small`
  - `avg_abs_brightness_diff=1.459403`
  - `max_abs_brightness_diff=3.666667`
  - `avg_abs_channel_diff=1.575076`
  - `max_abs_channel_diff=12`

Но performance получился хуже baseline:

- `gnome_4096`
  - baseline decode `13.30 ms`
  - `colmeta` decode `18.91 ms`
  - baseline `coeffs -> ycbcr` `4.82 ms`
  - `colmeta` `coeffs -> ycbcr` `8.62 ms`
- `gnome_8192`
  - baseline decode `46.96 ms`
  - `colmeta` decode `93.77 ms`
  - baseline `coeffs -> ycbcr` `17.60 ms`
  - `colmeta` `coeffs -> ycbcr` `46.56 ms`

Вывод:

- richer metadata сама по себе не даёт выигрыша;
- в tested exact form overhead от bookkeeping и branchy stage 2 оказался дороже, чем выигрыш от skipped columns.

### `less waits / batching`

Для benchmark path был добавлен второй orchestration mode:

- `GPGPU_VULKAN_JPEG_BENCH_SYNC_MODE=stage`
  - старый режим
  - wait после каждой decode sub-stage
- `GPGPU_VULKAN_JPEG_BENCH_SYNC_MODE=iteration`
  - stage1/stage2/stage3 отправляются подряд
  - один wait на весь decode итерации

На `GTX 1650`:

- `gnome_4096`
  - `stage`: decode `13.30 ms`
  - `iteration`: decode `12.19 ms`
- `gnome_8192`
  - `stage`: decode `46.96 ms`
  - `iteration`: decode `45.67 ms`

То есть выигрыш умеренный, но реальный.

Для `gnome_8192` отдельно снят `nsys`:

- `.local_data/nsys_profiles/jpeg_gnome8192_stage.*`
- `.local_data/nsys_profiles/jpeg_gnome8192_iteration.*`

`vulkan_api_sum`:

- `stage`
  - `vkWaitForFences`: `666.88 ms`, `178` calls
- `iteration`
  - `vkWaitForFences`: `624.57 ms`, `158` calls

`nvtx_sum`:

- `stage`
  - `jpeg gpu decode color`: total `469.15 ms`, median `46.80 ms`
  - `vk async wait fence`: total `527.94 ms`
- `iteration`
  - `jpeg gpu decode color`: total `460.61 ms`, median `46.03 ms`
  - `vk async wait fence`: total `285.66 ms`

Вывод:

- batching/fewer waits действительно уменьшают host-side synchronization overhead;
- но основной bottleneck всё ещё остаётся в compute stages, а не в orchestration.

## Итог по трём веткам

На текущем шаге практический результат такой:

- `integer IDCT`: пока отрицательный результат
- `richer sparsity metadata`: тоже отрицательный результат в tested exact form
- `less waits / batching`: единственный вариант из трёх, который дал чистый и полезный выигрыш

Следующие разумные шаги после этого исследования:

1. Если продолжать `integer IDCT`, то только как faithful `libjpeg-style` port.
2. Если продолжать `sparsity`, то искать metadata, которая открывает более coarse shortcut, а не per-column micro-branching.
3. Если продолжать `batching`, то идти дальше к batching нескольких image iterations с более редким readback.
