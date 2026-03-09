#pragma once
struct SolverContext {

    int ndim;

    int nx;
    int ny;
    int nz;

    int B;

    float dt;
    unsigned int nt;

    int M;
    int abcn;

    bool free_surface;

    const float* lap_coeff;
    const float* grad_coeff;

    float dx;
    float dy;
    float dz;


    // ===============================
    // Physical domain (computed)
    // ===============================

    __host__ __device__
    inline int phys_x0() const { return abcn + M; }

    __host__ __device__
    inline int phys_x1() const { return nx - abcn - M; }

    __host__ __device__
    inline int phys_y0() const { return abcn + M; }

    __host__ __device__
    inline int phys_y1() const { return ny - abcn - M; }

    __host__ __device__
    inline int phys_z0() const { return free_surface ? M : abcn + M; }

    __host__ __device__
    inline int phys_z1() const { return nz - abcn - M; }

    __host__ __device__
    inline int nx_phys() const { return phys_x1() - phys_x0(); }

    __host__ __device__
    inline int ny_phys() const { return phys_y1() - phys_y0(); }

    __host__ __device__
    inline int nz_phys() const { return phys_z1() - phys_z0(); }

};
