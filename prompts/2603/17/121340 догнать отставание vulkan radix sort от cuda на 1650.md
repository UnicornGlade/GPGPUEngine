# Задача

- Пользователь попросил попробовать ещё сильнее сократить отставание `vulkan.radixSort` от `cuda.radixSort` на `GTX 1650`.
- Обязательное требование: profiling делать вне sandbox.

# Ход работы

- Снимается новый baseline `cuda.radixSort` и `vulkan.radixSort` на `GTX 1650`.
- Затем анализируются `nsys` профили и принимается решение, какой участок оптимизировать следующим.
- Честный sequential baseline:
  - `cuda.radixSort` median `0.00172542 s`
  - `vulkan.radixSort` median `0.00367852 s`
  - разрыв около `2.13x`
- По profiling стало видно, что remaining gap состоит и из kernel time, и из dispatch overhead.
- Для fast Vulkan path сделаны ещё две оптимизации:
  - `reduction` переведён на `subgroupInclusiveAdd`;
  - `accumulation` убран как отдельный dispatch и слит в `scatter`;
  - `scatter` упрощён под реальный one-workgroup-per-block layout.
- После этого:
  - `vulkan.radixSort` median стал `0.00345173 s`
  - launches `32 -> 24`
  - разрыв сократился примерно до `2.00x`
- Корректность снова перепроверена:
  - `GTX 1650` passed
  - `Intel UHD 770` passed
  - `llvmpipe` passed

# Результат

- До:
  - `cuda.radixSort` median `0.00172542 s`
  - `vulkan.radixSort` median `0.00367852 s`
- После:
  - `vulkan.radixSort` median `0.00345173 s`
- Итог:
  - дополнительное ускорение Vulkan примерно `1.07x`
  - разрыв с CUDA сократился примерно с `2.13x` до `2.00x`
- Подробности записаны в:
  - `docs/vulkan_radixsort_gtx1650_optimization.md`
