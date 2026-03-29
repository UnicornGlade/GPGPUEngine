#version 450

#define NCHANNELS 3
#define REQUIRE_TEMPLATE_NCHANNELS
#include <libgpu/vulkan/vk/common.vk>

layout(location = 0) in vec3 inProjection;

LAYOUT_NCHANNELS(0, outColor);

float hash11(float p)
{
	p = fract(p * 0.1031f);
	p *= p + 33.33f;
	p *= p + p;
	return fract(p);
}

float hash31(vec3 p3)
{
	p3 = fract(p3 * vec3(0.1031f, 0.1030f, 0.0973f));
	p3 += dot(p3, p3.yzx + 33.33f);
	return fract((p3.x + p3.y) * p3.z);
}

float gaussianNoise(vec3 seed)
{
	// Irwin-Hall approximation of Gaussian noise: sum of uniforms centered around zero.
	float acc = 0.0f;
	for (int i = 0; i < 6; ++i) {
		acc += hash31(seed + vec3(float(i) * 1.371f, float(i) * 2.173f, float(i) * 0.713f));
	}
	return (acc - 3.0f) / sqrt(0.5f);
}

void main()
{
	const float depth = clamp(inProjection.z, 0.0f, 1.0f);
	const float fine_noise = gaussianNoise(vec3(gl_FragCoord.xy * 0.125f, float(gl_PrimitiveID) * 0.03125f));
	const float coarse_noise = gaussianNoise(vec3(floor(gl_FragCoord.xy / 7.0f), float(gl_PrimitiveID) * 0.0078125f + 17.0f));
	const float noisy_depth = clamp(depth + 0.014f * fine_noise + 0.005f * coarse_noise, 0.0f, 1.0f);
	const float channel_jitter = 0.0035f * (hash11(float(gl_PrimitiveID) * 0.173f) - 0.5f);

	float res[NCHANNELS];
	res[0] = 0.25f + 8.5f * noisy_depth + 0.042f * fine_noise;
	res[1] = 0.35f + 6.0f * noisy_depth + channel_jitter + 0.029f * coarse_noise;
	res[2] = 0.50f + 4.5f * noisy_depth - channel_jitter + 0.023f * fine_noise;
	ASSIGN_NCHANNELS(outColor, res);
}
