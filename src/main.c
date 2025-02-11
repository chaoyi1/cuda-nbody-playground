#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "engine.h"

// Helper: convert HSV to RGB. h in [0,360), s and v in [0,1]
void hsv_to_rgb(float h, float s, float v, unsigned char *r, unsigned char *g, unsigned char *b)
{
    float c = v * s;
    float h_prime = fmod(h / 60.0f, 6);
    float x = c * (1 - fabs(fmod(h_prime, 2) - 1));
    float m = v - c;
    float r_temp, g_temp, b_temp;

    if (h_prime < 1)
    {
        r_temp = c;
        g_temp = x;
        b_temp = 0;
    }
    else if (h_prime < 2)
    {
        r_temp = x;
        g_temp = c;
        b_temp = 0;
    }
    else if (h_prime < 3)
    {
        r_temp = 0;
        g_temp = c;
        b_temp = x;
    }
    else if (h_prime < 4)
    {
        r_temp = 0;
        g_temp = x;
        b_temp = c;
    }
    else if (h_prime < 5)
    {
        r_temp = x;
        g_temp = 0;
        b_temp = c;
    }
    else
    {
        r_temp = c;
        g_temp = 0;
        b_temp = x;
    }

    *r = (unsigned char)((r_temp + m) * 255);
    *g = (unsigned char)((g_temp + m) * 255);
    *b = (unsigned char)((b_temp + m) * 255);
}

void colour_based_on_velocity(ParticleSystem host_ps, int i, float max_speed, unsigned char *r, unsigned char *g, unsigned char *b)
{
    float vx = host_ps.vx[i];
    float vy = host_ps.vy[i];
    float speed = sqrtf(vx * vx + vy * vy);
    float angle = atan2f(vy, vx);  // angle in radians

    // Normalize angle to [0, 360)
    float pi = atan(1) * 4;
    float hue = ((angle + pi) / (2.0f * pi)) * 360.0f;
    // Normalize speed to a brightness value in [0,1]. Adjust MAX_SPEED as needed.
    float brightness = speed / max_speed;
    if (brightness > 1.0f)
        brightness = 1.0f;
    float saturation = 1.0f;  // full saturation

    hsv_to_rgb(hue, saturation, brightness, r, g, b);
}

int main(int argc, char const *argv[])
{
    // Simulation config
    const float MAX_SPEED = 0.001f;
    const float G = 0.00001f;
    const float dt = 0.01f;
    const float epsilon = 1e-6f;
    const int NUM_PARTICLES = 131072;

    // SDL stuff
    const int WINDOW_WIDTH = 3840;
    const int WINDOW_HEIGHT = 2160;
    const int RECT_LENGTH = 2;

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *const window = SDL_CreateWindow("particle sim", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_RESIZABLE);

    SDL_Renderer *const renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    // Simulation initialization
    ParticleSystem dev_ps_new;
    ParticleSystem dev_ps_old;
    ParticleSystem host_ps;

    host_ps.x = malloc(sizeof(float) * NUM_PARTICLES);
    host_ps.y = malloc(sizeof(float) * NUM_PARTICLES);
    host_ps.vx = malloc(sizeof(float) * NUM_PARTICLES);
    host_ps.vy = malloc(sizeof(float) * NUM_PARTICLES);
    host_ps.mass = malloc(sizeof(float) * NUM_PARTICLES);

    allocate_particle_system(&dev_ps_new, NUM_PARTICLES);
    allocate_particle_system(&dev_ps_old, NUM_PARTICLES);
    // Generate random data for old
    initialize_simulation(&dev_ps_old, NUM_PARTICLES, MAX_SPEED);
    // Make sure old and new have same data initially
    copy_ps_dev_to_dev(&dev_ps_new, &dev_ps_old, NUM_PARTICLES);

    // Simulation loop
    int quit = 0;

    while (!quit)
    {
        SDL_Event event = {0};
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
                case SDL_QUIT:
                {
                    quit = 1;
                }
                break;
            }
        }

        // Calculate new positions of particles, store them in dev_ps_new
        launch_update_particles(&dev_ps_new, &dev_ps_old, NUM_PARTICLES, dt, G, epsilon);
        // Copy dev_ps_new to host_ps for rendering
        copy_ps_dev_to_host(&host_ps, &dev_ps_new, NUM_PARTICLES);
        // Swap the buffer
        swap_particle_systems(&dev_ps_old, &dev_ps_new);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        SDL_Rect rect;
        rect.w = RECT_LENGTH;
        rect.h = RECT_LENGTH;
        for (int i = 0; i < NUM_PARTICLES; i++)
        {
            unsigned char r, g, b;
            colour_based_on_velocity(host_ps, i, MAX_SPEED, &r, &g, &b);

            // Map normalized coordinates to screen coordinates.
            rect.x = (int)(host_ps.x[i] * (WINDOW_WIDTH - rect.w));
            rect.y = (int)(host_ps.y[i] * (WINDOW_HEIGHT - rect.h));
            SDL_SetRenderDrawColor(renderer, r, g, b, 255);
            SDL_RenderFillRect(renderer, &rect);
        }

        SDL_RenderPresent(renderer);
    }

    free_particle_system(&dev_ps_new);
    free_particle_system(&dev_ps_old);
    free(host_ps.x);
    free(host_ps.y);
    free(host_ps.vx);
    free(host_ps.vy);
    free(host_ps.mass);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}
