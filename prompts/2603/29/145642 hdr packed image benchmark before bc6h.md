## Задача

Пользователь попросил сначала реализовать не BC6H, а промежуточный Vulkan unit-test:
- растеризовать гнома в float32 изображение;
- незаполненные пиксели оставить равными `FLT_MAX`;
- на GPU упаковать изображение в более простой HDR packed format;
- на GPU оценить ошибку при чтении упакованного изображения;
- вывести скорость упаковки, размеры и сравнение скорости вычисления RMS яркости по оригинальному и packed изображению.

Подробный инженерный план пока находится в рабочем контексте задачи; по итогам нужно будет при необходимости вынести его в `docs/...`.

## Журнал

- 14:56: проверено, что дерево чистое.
- 14:56: зафиксировано, что в репозитории нет готового BC6H path, поэтому сначала делается промежуточный HDR packed pipeline.
- 15:03: добавлены `hdrPackedImageBenchmark`, новые Vulkan kernels для HDR rasterize, GPU pack, brightness stats и error evaluation.
- 15:04: расширен Vulkan engine: явное создание custom-format image с usage flags и Vulkan `buffer -> image` copy для GPU-only packing path.
- 15:05: первый вариант на `RGB9E5` упёрся в отсутствие рабочего GLSL builtin в используемом toolchain; промежуточный формат заменён на `RGBA16F`, упаковка сделана через `packHalf2x16`.
- 15:06: тест успешно прошёл на RTX 4090 без validation layers.
- 15:06: тест успешно прошёл на `llvmpipe`, для него автоматически используется меньший resolution `512x512`.
- 15:07: тест успешно прошёл на RTX 4090 с validation layers.
- 15:12: пользователь попросил добавить отдельный unit-test с настоящим `BC6H`, не убирая уже существующий packed-format test.
- 15:20: реализован минимальный собственный Vulkan BC6H encoder: на каждый `4x4` block считается средний HDR цвет valid-пикселей и кодируется валидный `BC6H UF16` block в single-subset mode с одинаковыми endpoints.
- 15:22: добавлен отдельный `vulkan.bc6hImageBenchmark`, использующий настоящий `VK_FORMAT_BC6H_UFLOAT_BLOCK`.
- 15:23: `BC6H` test успешно прошёл на RTX 4090 без validation layers.
- 15:24: `BC6H` test успешно прошёл на `llvmpipe`.
- 15:24: `BC6H` test успешно прошёл на RTX 4090 с validation layers.
- 15:25: оба теста (`hdrPackedImageBenchmark` и `bc6hImageBenchmark`) успешно прошли вместе на RTX 4090.
- 15:30: дополнительно исследованы внешние реализации и статьи по BC6H:
  - GPURealTimeBC6H (K. Narkowicz): bbox / inset / fix-up / partition search / RMSLE;
  - DirectXTex: наличие GPU BC6H path через DirectCompute;
  - сделана попытка адаптации части эвристик в наш one-mode Vulkan encoder.
- 15:33: практическая проверка показала, что для текущего one-mode kernel дополнительные candidate search / HDR metric ухудшают отношение quality/speed.
- 15:34: код возвращён к лучшему из измеренных вариантов: `min/max endpoints + brute-force per-texel indices` для single-subset BC6H mode.
- 15:41: данные benchmark'а сделаны более шумными: HDR image теперь строится как noisy depth-like поле, а не просто гладкая функция от проекции.
- 15:43: проверено, что для новых noisy depth-like данных попытка выбирать endpoints по brightness extrema не улучшает BC6H sufficiently; лучший результат остаётся у baseline encoder.
- 15:45: шум откалиброван так, чтобы оба теста проходили и на RTX 4090, и на `llvmpipe`, но данные оставались заметно не-гладкими.

## Результаты

- RTX 4090, `2048x2048`:
  - `compression total median`: около `0.658 ms`, `6376 MPix/s`
  - `pack kernel median`: около `0.0466 ms`, `90024 MPix/s`
  - `buffer -> image median`: около `0.611 ms`
  - `RMS brightness error`: около `0.000412`
  - `brightness stddev kernel throughput`: около `36305 MPix/s` по исходному float32 image и `36835 MPix/s` по packed image
- llvmpipe, `512x512`:
  - `compression total median`: около `0.849 ms`, `309 MPix/s`
  - `RMS brightness error`: около `0.000701`
- RTX 4090, `BC6H`, `2048x2048`:
  - лучший подтверждённый вариант:
    - `compression total median`: около `0.190 ms`, `22083 MPix/s`
    - `bc6h encode median`: около `0.0956 ms`, `43856 MPix/s`
    - `RMS brightness error`: около `0.0450` на noisy depth-like HDR данных
  - `buffer -> image median`: около `0.094 ms`
  - `compression ratio`: `12x`
- RTX 4090, `RGBA16F packed`, `2048x2048`, noisy depth-like HDR:
  - `compression total median`: около `0.658 ms`, `6379 MPix/s`
  - `RMS brightness error`: около `0.000423`
- llvmpipe, noisy depth-like HDR:
  - `BC6H RMS brightness error`: около `0.0473`
