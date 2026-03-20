# JPEG decode benchmark на Vulkan

## Сообщения пользователя

- `новый промпт, я хочу изучить jpeg декодирование на видеокарте (на Vulkan), пожалуйста напиши новый юнит-тест который это будет тестировать, скачай из интернета какие-нибудь стандартные изображения которые часто используют для такого теста (разного разрешения, но итого они должны занимать не больше 10 мегабайт в сумме, для начала можешь скачать их и больше гораздо, потом прежде чем их коммитить расскажи какие есть картинки и откуда они взяты и сколько весят - я выберу из них те что нужно будет в дальнейшем использовать в юнит-тесте), итак хочется сделать следующе: юнит тест должен в одном своем варианте считывать с диска в оперативную память изображения (не распаковывая, просто как байты с диска), дальше много раз (10 раз) в цикле (для надежности замера) одну из этих картинок (потом так же 10 раз следующую картинку) декодировать на CPU в распакованное изображение, отправлять его по PCI-E шине, считать на видеокарте простую редукцию - средняя яркость по всему изображению, после этого считывать это число результат на CPU и сверять его с эталоном (заранее посчитанным на CPU), после всего этого вывести в консоль сколько времени CPU распаковывал одно изображение (медиана), сколько времени картинка отправлялась по PCI-E шине, сколько времени картинка обрабатывалась на GPU (считалась средняя яркость), сколько времени считывался результат на CPU, после этого сверь числа с nsys профилировщиком что профилирование в логе близко к правде и расскажи мне все числа и общую картину`
- `дальше найди в интернете стандартные практики и в идеале open source варианты кода который показывает как на GPU декодировать jpeg изображения
хочется чтобы оно работало на любой видеокарте, поэтому nvidia jpeg нам подходит только как оптимизация, но это потом
тестируй пока что на NVIDIA 1650 (вне sandbox)
а для начала найди open source варианты (может на opencl/vulkan/cuda/directx/opengl может еще на чем) которые выглядят перспективно
и ориентируясь на них напиши vulkan кернел который декодирует jpeg на GPU
после этого добавь второй юнит-тест очень похожий на первый но теперь отправляй на видеокарту закодированную jpeg картинку а декодируй уже на GPU
и выводи в логи почти те же тайминги что и первый юнит-тест, но явно подчеркни что теперь jpeg-декодирование на GPU (а в первом юнит-тесте вывод в консоль явно говорит что декодирование на CPU)
добавь третий юнит-тест который проверяет корректность декодирования, что разжатая на CPU и на GPU картинка очень похожи (проверяй и среднюю разницу яркости и максимальную разницу яркости по всей картинке, выводи их в консоль, сообщи их мне в конце работы)
проверь что все работает, что средняя яркость все та же, проверь с валидационными слоями что все ок, проверь так же теперь на Intel UHD 770
после этого найди большие изображения (4K например, или даже больше, условно 10, 20, 40 мегапикселей)`

## Короткий лог действий

- Создан рабочий docs-файл с планом и итоговыми замерами по JPEG decode benchmark.
- Подбираются стандартные source images и собирается локальный набор JPEG-кандидатов вне git до выбора пользователя.
- Добавляется новый Vulkan unit-test для цепочки `disk bytes -> CPU JPEG decode -> PCI-E upload -> GPU brightness reduction -> CPU readback`.
- Добавлены два compute shader-а для простой редукции суммы байтов `u8 -> u32 -> u32`.
- Для benchmark и `nsys` добавлены NVTX phase markers.
- Исправлен `libbase` CMake-path для NVTX headers при `GPGPU_ENABLE_NVTX_MARKERS=ON`.
- Исследованы open-source GPU JPEG decode references:
  - `GPUJPEG` (`CUDA/OpenGL`)
  - `compeg` (`WebGPU`)
- Добавлен Vulkan compute kernel для GPU JPEG decode узкого baseline-case:
  - grayscale
  - baseline DCT
  - `restart_interval = 1`
  - CPU parsing + GPU entropy/dequant/IDCT
- Добавлены новые unit-tests:
  - `vulkan.jpegDecodeBenchmarkGpu`
  - `vulkan.jpegDecodeGpuMatchesCpu`
- Новый GPU decode path прогнан с validation layers на `NVIDIA GeForce GTX 1650` и `Intel UHD Graphics 770`.
- Начат подбор больших публичных benchmark images из JPEG/JPEG-AI/AIC datasets.
- Добавлен отдельный color baseline JPEG path:
  - `4:2:0`
  - `4:4:4`
  - MCU-level Vulkan decode с `restart_interval = 1`
- Добавлены новые unit-tests:
  - `vulkan.jpegDecodeBenchmarkGpuColor`
  - `vulkan.jpegDecodeGpuColorMatchesCpu`
- Для длинных compute benchmark-ов поднят Vulkan fence timeout в runtime с `1 s` до `30 s`.
- Собраны большие public-domain / benchmark image candidates:
  - `~9 MP`
  - `24 MP`
  - `64 MP`
- Большие benchmark/correctness прогоны выполнены на `GTX 1650`.

## Короткий лог результатов

- Новый `vulkan.jpegDecodeBenchmark` проходит на `NVIDIA GeForce GTX 1650` в `RelWithDebInfo`.
- Под `nsys` собран sqlite trace `/tmp/nsys_jpeg_vulkan_gtx1650_nvtx.sqlite`.
- Собрано `8` candidate JPEG images из Kodak и USC-SIPI суммарным размером `1608869 B` вне git.
- Для `1024x1024` картинок порядок величин сейчас такой:
  - CPU decode: около `5.5 ms`
  - upload: около `1.4 ms`
  - GPU reduction: около `0.5-0.6 ms`
  - readback: около `0.17-0.24 ms`
- Новый `vulkan.jpegDecodeBenchmarkGpu` проходит на `GTX 1650` и `UHD 770`.
- Новый `vulkan.jpegDecodeGpuMatchesCpu` показывает стабильную точность:
  - `avg_abs_diff` около `0.495-0.504`
  - `max_abs_diff = 1`
- Для `1024x1024` в GPU decode path:
  - `GTX 1650`: decode около `2.51-2.57 ms`
  - `UHD 770`: decode около `5.57-6.57 ms`
- Validation layers для новых Vulkan tests не показали проблем ни на NVIDIA, ни на Intel.
- Новый color path проходит на `GTX 1650` и `UHD 770` с validation layers.
- Для color `4:2:0` на малом наборе:
  - `GTX 1650`: decode примерно `25 ms` .. `204 ms`
  - `UHD 770`: decode примерно `480 ms` .. `702 ms`
- Для больших кадров на `GTX 1650`:
  - `~9 MP`: GPU color decode около `1.68 s`
  - `24 MP`: GPU color decode около `4.47 s`
  - `64 MP`: GPU color decode около `11.96 s`
- Brightness correctness для больших кадров остаётся хорошей:
  - `avg_abs_brightness_diff` около `0.77-0.86`
  - `max_abs_brightness_diff` до `10.0`
- После отдельного `4:2:0 fast-path`:
  - `GTX 1650`, малый набор: примерно `1.0-4.9 ms`
  - `UHD 770`, малый набор: примерно `6.0-10.5 ms`
- После следующего staged split `JPEG -> planar YCbCr420 -> RGB`:
  - `~9 MP`: decode около `53.29 ms`
  - `24 MP`: decode около `70.90 ms`
  - `64 MP`: decode около `148.44 ms`
- После дополнительного split `JPEG -> coeffs -> planar YCbCr420 -> RGB`:
  - `GTX 1650`, малый набор: decode примерно `0.76-1.55 ms`
  - `~9 MP`: decode около `27.71 ms`
  - `24 MP`: decode около `33.60 ms`
  - `64 MP`: decode около `84.86 ms`
  - относительно CPU decode это уже примерно:
    - `2.55x` быстрее на `~9 MP`
    - `4.65x` быстрее на `24 MP`
    - `4.31x` быстрее на `64 MP`
- Validation layers перепроверены после coefficient split:
  - `Intel UHD 770`: без новых ошибок
  - `NVIDIA GeForce GTX 1650`: без новых ошибок
- Для воспроизводимых одиночных прогонов в текущем merged device ordering:
  - `GPGPU_VISIBLE_DEVICES=2` -> `Intel UHD 770`
  - `GPGPU_VISIBLE_DEVICES=3` -> `NVIDIA GeForce GTX 1650`
- Остался отдельный technical debt:
  - старый `vulkan.jpegDecodeBenchmark` для very large RGB images пока не даёт корректный итог из-за legacy raw-byte reduction path.
- Подробный инженерный журнал: `docs/vulkan_jpeg_decode_benchmark.md`.
