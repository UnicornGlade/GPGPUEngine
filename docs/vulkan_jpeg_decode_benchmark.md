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
