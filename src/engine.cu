#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>
#include "engine.h"

// Kernel to initialize CURAND state for each thread.
__global__ void init_particles_kernel(curandState *states, unsigned long seed, ParticleSystem ps, int num_particles, float max_speed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles)
    {
        // curand_init(seed, sequence number, offset, state pointer)
        curand_init(seed, idx, 0, &states[idx]);
        ps.x[idx] = curand_uniform(&states[idx]);
        ps.y[idx] = curand_uniform(&states[idx]);
        ps.vx[idx] = max_speed * curand_uniform(&states[idx]);
        ps.vy[idx] = max_speed * curand_uniform(&states[idx]);
        ps.mass[idx] = 1.0f;
    }
}

__global__ void update_particles_kernel(ParticleSystem ps_new, ParticleSystem ps_old, int num_particles, float dt, float G, float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles)
    {
        float net_accelx = 0.0f;
        float net_accely = 0.0f;
        for (int j = 0; j < num_particles; j++)
        {
            if (idx == j)
            {
                continue;
            }
            float dx = ps_old.x[idx] - ps_old.x[j];
            float dy = ps_old.y[idx] - ps_old.y[j];
            float dist_sq = dx * dx + dy * dy + epsilon;  // avoid division by 0
            float accel = -G * ps_old.mass[j] / dist_sq;
            net_accelx += dx * accel;
            net_accely += dy * accel;
        }
        ps_new.vx[idx] = ps_old.vx[idx] + net_accelx * dt;
        ps_new.vy[idx] = ps_old.vy[idx] + net_accely * dt;
        ps_new.x[idx] = ps_old.x[idx] + ps_new.vx[idx] * dt;
        ps_new.y[idx] = ps_old.y[idx] + ps_new.vy[idx] * dt;

        // Toroidal boundaries: Wrap positions around the domain [0,1]
        if (ps_new.x[idx] < 0.0f)
            ps_new.x[idx] += 1.0f;
        else if (ps_new.x[idx] >= 1.0f)
            ps_new.x[idx] -= 1.0f;
        if (ps_new.y[idx] < 0.0f)
            ps_new.y[idx] += 1.0f;
        else if (ps_new.y[idx] >= 1.0f)
            ps_new.y[idx] -= 1.0f;
    }
}

void allocate_curand_states(curandState **states, int num_particles)
{
    size_t size = num_particles * sizeof(curandState);
    cudaMalloc(states, size);
}

void free_curand_states(curandState *states)
{
    cudaFree(states);
}

extern "C" void allocate_particle_system(ParticleSystem *ps, int num_particles)
{
    size_t size = num_particles * sizeof(float);

    cudaMalloc(&ps->x, size);
    cudaMalloc(&ps->y, size);
    cudaMalloc(&ps->vx, size);
    cudaMalloc(&ps->vy, size);
    cudaMalloc(&ps->mass, size);
}

extern "C" void free_particle_system(ParticleSystem *ps)
{
    cudaFree(ps->x);
    cudaFree(ps->y);
    cudaFree(ps->vx);
    cudaFree(ps->vy);
    cudaFree(ps->mass);
}

extern "C" void initialize_simulation(ParticleSystem *ps_start, int num_particles, float max_speed)
{
    // Allocate curand state
    curandState *d_states;
    allocate_curand_states(&d_states, num_particles);

    dim3 threadsPerBlock(256);
    dim3 numBlocks((num_particles + threadsPerBlock.x - 1) / threadsPerBlock.x);

    //
    init_particles_kernel<<<numBlocks, threadsPerBlock>>>(d_states, time(NULL), *ps_start, num_particles, max_speed);

    cudaDeviceSynchronize();

    free_curand_states(d_states);
}

extern "C" void launch_update_particles(ParticleSystem *ps_new, ParticleSystem *ps_old, int num_particles, float dt, float G, float epsilon)
{
    dim3 threadsPerBlock(256);
    dim3 numBlocks((num_particles + threadsPerBlock.x - 1) / threadsPerBlock.x);
    update_particles_kernel<<<numBlocks, threadsPerBlock>>>(*ps_new, *ps_old, num_particles, dt, G, epsilon);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s", cudaGetErrorString(err));
    }
}

extern "C" void swap_particle_systems(ParticleSystem *ps1, ParticleSystem *ps2)
{
    ParticleSystem temp = *ps1;
    *ps1 = *ps2;
    *ps2 = temp;
}

extern "C" void copy_ps_dev_to_host(ParticleSystem *host, ParticleSystem *dev, int num_particles)
{
    size_t size = num_particles * sizeof(float);

    cudaMemcpy(host->x, dev->x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host->y, dev->y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host->vx, dev->vx, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host->vy, dev->vy, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host->mass, dev->mass, size, cudaMemcpyDeviceToHost);
}

extern "C" void copy_ps_dev_to_dev(ParticleSystem *dev_new, ParticleSystem *dev_old, int num_particles)
{
    size_t size = num_particles * sizeof(float);

    cudaMemcpy(dev_new->x, dev_old->x, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_new->y, dev_old->y, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_new->vx, dev_old->vx, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_new->vy, dev_old->vy, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_new->mass, dev_old->mass, size, cudaMemcpyDeviceToDevice);
}