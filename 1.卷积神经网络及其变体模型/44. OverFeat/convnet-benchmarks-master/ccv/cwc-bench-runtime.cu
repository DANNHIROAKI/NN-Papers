#undef USE_DISPATCH // nvcc doesn't support libdispatch
extern "C" {
#include "ccv.h"
}
#include <ctype.h>
#define CASE_TESTS // so that we don't include public available methods
#include "../lib/cuda/cwc_convnet.cu"
#include "../lib/ccv_convnet.c"

extern "C" void cwc_bench_runtime(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_convnet_train_param_t params)
{
	int batch = params.mini_batch;
	int i;
	const int device_id = 0;
	_cwc_convnet_alloc_reserved_both(convnet, batch, 0, params.layer_params);
	cwc_convnet_context_t* context = GPU(convnet)->contexts;
	for (i = 0; i < convnet->rows * convnet->cols * convnet->channels; i++)
		convnet->mean_activity->data.f32[i] = 128;
	cwc_convnet_batch_formation(0, categorizeds, convnet->mean_activity, 0, 0, 0, 0, 0, ccv_size(128, 128), 128, 128, convnet->rows, convnet->cols, convnet->channels, 1000, 0, batch, 0, batch, context->host[device_id].input, context->host[device_id].c);
	cudaMemcpy(context->device[device_id].input, context->host[device_id].input, sizeof(float) * convnet->rows * convnet->cols * convnet->channels * batch, cudaMemcpyHostToDevice);

	cudaEvent_t overallStart;
	cudaEvent_t overallStop;
	cudaEventCreate(&overallStart);
	cudaEventCreate(&overallStop);
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time;
	EXTRA(GPU(convnet)->device[0].layers + 0)->vary.convolutional.forward.x = 4;
	EXTRA(GPU(convnet)->device[0].layers + 0)->vary.convolutional.forward.y = 8;
	EXTRA(GPU(convnet)->device[0].layers + 0)->vary.convolutional.forward.z = 32;
	EXTRA(GPU(convnet)->device[0].layers + 1)->vary.convolutional.forward.x = 4;
	EXTRA(GPU(convnet)->device[0].layers + 1)->vary.convolutional.forward.y = 8;
	EXTRA(GPU(convnet)->device[0].layers + 1)->vary.convolutional.forward.z = 32;
	EXTRA(GPU(convnet)->device[0].layers + 2)->vary.convolutional.forward.x = 4;
	EXTRA(GPU(convnet)->device[0].layers + 2)->vary.convolutional.forward.y = 8;
	EXTRA(GPU(convnet)->device[0].layers + 2)->vary.convolutional.forward.z = 32;
	EXTRA(GPU(convnet)->device[0].layers + 3)->vary.convolutional.forward.x = 4;
	EXTRA(GPU(convnet)->device[0].layers + 3)->vary.convolutional.forward.y = 8;
	EXTRA(GPU(convnet)->device[0].layers + 3)->vary.convolutional.forward.z = 32;
	EXTRA(GPU(convnet)->device[0].layers + 4)->vary.convolutional.forward.x = 4;
	EXTRA(GPU(convnet)->device[0].layers + 4)->vary.convolutional.forward.y = 8;
	EXTRA(GPU(convnet)->device[0].layers + 4)->vary.convolutional.forward.z = 32;
	cudaEventRecord(overallStart, context->device[0].data_stream);
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->device[0].layers + i;
		cudaEventRecord(start, context->device[0].data_stream);
		_cwc_convnet_layer_forward_propagate(layer, 0, i, layer->input.matrix.rows, layer->input.matrix.cols, batch, 0, i == 0 ? context->device[0].input : GPU(convnet)->device[0].forwards[i - 1], GPU(convnet)->device[0].forwards[i], GPU(convnet)->device[0].denoms[i], GPU(convnet)->device[0].unit, context);
		cudaEventRecord(stop, context->device[0].data_stream);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_time, start, stop);
		if (layer->type == CCV_CONVNET_CONVOLUTIONAL)
			printf("%d %d %d, elapsed time for layer %d fprop: %f milliseconds\n", EXTRA(layer)->vary.convolutional.forward.x, EXTRA(layer)->vary.convolutional.forward.y, EXTRA(layer)->vary.convolutional.forward.z, i + 1, elapsed_time);
		else
			printf("elapsed time for layer %d fprop: %f milliseconds\n", i + 1, elapsed_time);
	}
	cudaEventRecord(overallStop, context->device[0].data_stream);
	cudaEventSynchronize(overallStop);
	cudaEventElapsedTime(&elapsed_time, overallStart, overallStop);
	printf("forward pass %f milliseconds\n", elapsed_time);

	/*
	EXTRA(GPU(convnet)->device[0].layers + 0)->vary.convolutional.backward.coefficient.x = 1;
	EXTRA(GPU(convnet)->device[0].layers + 0)->vary.convolutional.backward.coefficient.y = 3;
	EXTRA(GPU(convnet)->device[0].layers + 0)->vary.convolutional.backward.coefficient.z = 1;
	EXTRA(GPU(convnet)->device[0].layers + 3)->vary.convolutional.backward.coefficient.x = 4;
	EXTRA(GPU(convnet)->device[0].layers + 3)->vary.convolutional.backward.coefficient.y = 4;
	EXTRA(GPU(convnet)->device[0].layers + 3)->vary.convolutional.backward.coefficient.z = 16;
	EXTRA(GPU(convnet)->device[0].layers + 3)->vary.convolutional.backward.gradient.x = 4;
	EXTRA(GPU(convnet)->device[0].layers + 3)->vary.convolutional.backward.gradient.y = 6;
	EXTRA(GPU(convnet)->device[0].layers + 3)->vary.convolutional.backward.gradient.z = 24;
	EXTRA(GPU(convnet)->device[0].layers + 6)->vary.convolutional.backward.coefficient.x = 8;
	EXTRA(GPU(convnet)->device[0].layers + 6)->vary.convolutional.backward.coefficient.y = 3;
	EXTRA(GPU(convnet)->device[0].layers + 6)->vary.convolutional.backward.coefficient.z = 32;
	EXTRA(GPU(convnet)->device[0].layers + 6)->vary.convolutional.backward.gradient.x = 4;
	EXTRA(GPU(convnet)->device[0].layers + 6)->vary.convolutional.backward.gradient.y = 8;
	EXTRA(GPU(convnet)->device[0].layers + 6)->vary.convolutional.backward.gradient.z = 32;
	EXTRA(GPU(convnet)->device[0].layers + 7)->vary.convolutional.backward.coefficient.x = 8;
	EXTRA(GPU(convnet)->device[0].layers + 7)->vary.convolutional.backward.coefficient.y = 3;
	EXTRA(GPU(convnet)->device[0].layers + 7)->vary.convolutional.backward.coefficient.z = 32;
	EXTRA(GPU(convnet)->device[0].layers + 7)->vary.convolutional.backward.gradient.x = 4;
	EXTRA(GPU(convnet)->device[0].layers + 7)->vary.convolutional.backward.gradient.y = 8;
	EXTRA(GPU(convnet)->device[0].layers + 7)->vary.convolutional.backward.gradient.z = 32;
	EXTRA(GPU(convnet)->device[0].layers + 8)->vary.convolutional.backward.coefficient.x = 8;
	EXTRA(GPU(convnet)->device[0].layers + 8)->vary.convolutional.backward.coefficient.y = 4;
	EXTRA(GPU(convnet)->device[0].layers + 8)->vary.convolutional.backward.coefficient.z = 32;
	EXTRA(GPU(convnet)->device[0].layers + 8)->vary.convolutional.backward.gradient.x = 4;
	EXTRA(GPU(convnet)->device[0].layers + 8)->vary.convolutional.backward.gradient.y = 8;
	EXTRA(GPU(convnet)->device[0].layers + 8)->vary.convolutional.backward.gradient.z = 32;
	float* a = 0;
	cudaMalloc(&a, sizeof(float) * 1000 * batch);
	cudaMemcpy(a, GPU(convnet)->device[0].forwards[convnet->count - 1], sizeof(float) * 1000 * batch, cudaMemcpyDeviceToDevice);
	cudaEventRecord(overallStart, context->device[0].data_stream);
	for (i = convnet->count - 1; i >= 0; i--)
	{
		ccv_convnet_layer_t* layer = GPU(convnet)->device[0].layers + i;
		ccv_convnet_layer_t* configuration = GPU(convnet)->device[0].configurations + i;
		cudaEventRecord(start, context->device[0].data_stream);
		switch (layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				if (context->device[0].dor[i])
				{
					int out_rows, out_cols, out_partition;
					ccv_convnet_make_output(layer, layer->input.matrix.rows, layer->input.matrix.cols, &out_rows, &out_cols, &out_partition);
					_cwc_kern_mute_neuron
					<<<out_rows * out_cols * layer->net.convolutional.count, batch, 0, context->device[0].data_stream>>>
					(i == convnet->count - 1 ? a : GPU(convnet)->device[0].backwards[i + 1], context->device[0].dor[i]);
				}
				_cwc_convnet_convolutional_backward_propagate(layer, batch, i == convnet->count - 1 ? a : GPU(convnet)->device[0].backwards[i + 1], GPU(convnet)->device[0].forwards[i], i > 0 ? GPU(convnet)->device[0].forwards[i - 1] : context->device[0].input, GPU(convnet)->device[0].backwards[i], configuration, GPU(convnet)->device[0].scratch, GPU(convnet)->device[0].unit, context->device[0].data_stream, context->device[0].data_cublas);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				if (context->device[0].dor[i])
					_cwc_kern_mute_neuron
					<<<layer->net.full_connect.count, batch, 0, context->device[0].data_stream>>>
					(i == convnet->count - 1 ? a : GPU(convnet)->device[0].backwards[i + 1], context->device[0].dor[i]);
				_cwc_convnet_full_connect_backward_propagate(layer, batch,  i == convnet->count - 1 ? a : GPU(convnet)->device[0].backwards[i + 1], GPU(convnet)->device[0].forwards[i], i > 0 ? GPU(convnet)->device[0].forwards[i - 1] : context->device[0].input, GPU(convnet)->device[0].backwards[i], GPU(convnet)->device[0].unit, configuration, context->device[0].data_stream, context->device[0].data_cublas);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
				_cwc_convnet_rnorm_backward_propagate(layer, batch, i == convnet->count - 1 ? a : GPU(convnet)->device[0].backwards[i + 1], GPU(convnet)->device[0].forwards[i], i > 0 ? GPU(convnet)->device[0].forwards[i - 1] : context->device[0].input, GPU(convnet)->device[0].denoms[i], GPU(convnet)->device[0].backwards[i], context->device[0].data_stream);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_MAX_POOL:
				_cwc_convnet_max_pool_backward_propagate(layer, batch, i == convnet->count - 1 ? a : GPU(convnet)->device[0].backwards[i + 1], GPU(convnet)->device[0].forwards[i], i > 0 ? GPU(convnet)->device[0].forwards[i - 1] : context->device[0].input, GPU(convnet)->device[0].backwards[i], context->device[0].data_stream);
				assert(cudaGetLastError() == cudaSuccess);
				break;
			case CCV_CONVNET_AVERAGE_POOL:
				_cwc_convnet_average_pool_backward_propagate(layer, batch, i == convnet->count - 1 ? a : GPU(convnet)->device[0].backwards[i + 1], GPU(convnet)->device[0].backwards[i], context->device[0].data_stream);
				assert(cudaGetLastError() == cudaSuccess);
				break;
		}
		cudaEventRecord(stop, context->device[0].data_stream);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_time, start, stop);
		if (layer->type == CCV_CONVNET_CONVOLUTIONAL)
			printf("%d %d %d, %d %d %d, elapsed time for layer %d bprop: %f milliseconds\n", EXTRA(layer)->vary.convolutional.backward.coefficient.x, EXTRA(layer)->vary.convolutional.backward.coefficient.y, EXTRA(layer)->vary.convolutional.backward.coefficient.z, EXTRA(layer)->vary.convolutional.backward.gradient.x, EXTRA(layer)->vary.convolutional.backward.gradient.y, EXTRA(layer)->vary.convolutional.backward.gradient.z, i + 1, elapsed_time);
		else
			printf("elapsed time for layer %d bprop: %f milliseconds\n", i + 1, elapsed_time);
	}
	cudaEventRecord(overallStop, context->device[0].data_stream);
	cudaEventSynchronize(overallStop);
	cudaEventElapsedTime(&elapsed_time, overallStart, overallStop);
	printf("backward pass %f milliseconds\n", elapsed_time);
	cudaFree(a);
	*/
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventDestroy(overallStart);
	cudaEventDestroy(overallStop);
}
