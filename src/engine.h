#ifndef ENGINE_H
#define ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ParticleSystem
{
    float *x, *y;
    float *vx, *vy;
    float *mass;
} ParticleSystem;

void allocate_particle_system(ParticleSystem *ps, int num_particles);
void free_particle_system(ParticleSystem *ps);

void initialize_simulation(ParticleSystem *ps_start, int num_particles, float max_speed);
void launch_update_particles(ParticleSystem *ps_new, ParticleSystem *ps_old, int num_particles, float dt, float G, float epsilon);
void swap_particle_systems(ParticleSystem *ps1, ParticleSystem *ps2);

void copy_ps_dev_to_host(ParticleSystem *host, ParticleSystem *dev, int num_particles);
void copy_ps_dev_to_dev(ParticleSystem *dev_new, ParticleSystem *dev_old, int num_particles);

#ifdef __cplusplus
}
#endif

#endif // ENGINE_H